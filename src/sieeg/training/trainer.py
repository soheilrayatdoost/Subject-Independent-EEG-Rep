"""Adversarial training of SI-EEG (Eq. 1), SIDA-EEG (Eq. 3) and the baselines by gradient reversal.

One TF2 training step (``tf.GradientTape`` inside a ``tf.function``) minimises::

    L = CE(classifier(z_S), y_S) + sum_k L_k(head_k(GRL(z, lambda_k)))

with a single optimiser over the encoder, the emotion classifier and every
discriminator head. lambda only scales the reversed gradient inside the GRL
(paper Sec. 5.1), so each head minimises its loss while the encoder maximises
it, i.e. ``min L_emotion - lambda L_subject`` of Eq. 1 / Eq. 3. Batches and
model selection follow decisions D13-D16 of ``docs/DESIGN.md``: every step uses
b labeled source samples S plus b samples (with replacement) from each unlabeled
pool that an adversary needs: the out-group O and/or the target domain T.
With a target pool, the in-group/out-group head sees S, O and T, because the
subject term of Eq. 3 runs over source and target samples: a target sample is
out-group (1) if its subject is an out-group subject and in-group (0) otherwise
(D14). The identity head weighs pool samples so that every subject counts
equally (D15). Emotion labels of the pools are never used.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score

from ..config import ModelConfig, TrainConfig
from ..models.grl import gradient_reversal
from ..models.networks import AdversarialNet
from .methods import MethodSpec
from .optimizers import make_optimizer
from .schedules import lambda_factor

logger = logging.getLogger(__name__)

#: selection_metric -> history column it is read from.
_SELECTION_COLUMNS = {"f1": "val_f1", "accuracy": "val_acc", "loss": "val_loss"}


def _subject_index(subjects: np.ndarray | None, n: int) -> np.ndarray:
    if subjects is None:
        return np.full(n, -1, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.int64)
    if subjects.shape != (n,):
        raise ValueError(f"subjects must have shape ({n},), got {subjects.shape}")
    return subjects


@dataclass
class LabeledData:
    """Labeled samples: images ``x`` (n, H, W, C), emotion labels ``y`` (n,) and the
    global subject index of every sample (n,), -1 when unknown (the default)."""

    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.y.shape != (len(self.x),):
            raise ValueError(f"y must have shape ({len(self.x)},), got {self.y.shape}")
        self.subjects = _subject_index(self.subjects, len(self.x))

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class UnlabeledData:
    """An unlabeled pool (out-group or target domain): images and subject index, no emotion labels."""

    x: np.ndarray
    subjects: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.subjects = _subject_index(self.subjects, len(self.x))

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class FitResult:
    """``history`` has one row per epoch; ``best_epoch`` is 1-based (the last epoch without validation).

    ``identity_alpha`` is the weight of an identity-pool sample relative to a labeled sample in
    the identity cross-entropy (D15), None without an identity pool.
    """

    history: pd.DataFrame
    best_epoch: int
    best_score: float | None
    seconds: float
    identity_alpha: float | None = None


def _accuracy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))


class AdversarialTrainer:
    """Builds an :class:`AdversarialNet` for ``method`` and trains it with one optimiser.

    Discriminator heads: ``group`` (in-group 0 vs out-group 1, binary), ``domain``
    (source 0 vs target 1, binary) and ``identity`` (``n_subjects``-way subject index).
    """

    def __init__(
        self,
        method: MethodSpec,
        input_shape,
        *,
        n_classes: int = 2,
        n_subjects: int = 0,
        model_cfg: ModelConfig | None = None,
        train_cfg: TrainConfig | None = None,
        seed: int = 0,
    ):
        self.method = method
        self.input_shape = tuple(int(d) for d in input_shape)
        self.n_classes = int(n_classes)
        self.n_subjects = int(n_subjects)
        self.model_cfg = model_cfg if model_cfg is not None else ModelConfig()
        self.train_cfg = train_cfg if train_cfg is not None else TrainConfig()
        self.seed = int(seed)

        cfg = self.train_cfg
        if cfg.selection_metric not in _SELECTION_COLUMNS:
            raise ValueError(
                f"Unknown selection_metric {cfg.selection_metric!r}; use one of {list(_SELECTION_COLUMNS)}"
            )
        if cfg.class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {cfg.class_weight!r}")
        lambda_factor(0.0, cfg.lambda_schedule)  # rejects an unknown schedule before training
        self._kinds = tuple(a.kind for a in method.adversaries)
        if len(set(self._kinds)) != len(self._kinds):
            raise ValueError(f"Method {method.name!r} lists an adversary kind twice: {self._kinds}")
        if "identity" in self._kinds and self.n_subjects < 2:
            raise ValueError(f"The identity adversary needs n_subjects >= 2, got {self.n_subjects}")

        keras.utils.set_random_seed(self.seed)
        n_logits = {"group": 1, "domain": 1, "identity": self.n_subjects}
        adversaries = {kind: n_logits[kind] for kind in self._kinds}
        self.net = AdversarialNet(self.input_shape, self.n_classes, adversaries, self.model_cfg)
        self.optimizer = make_optimizer(cfg)
        self.optimizer.build(self.net.trainable_variables)

        image = tf.TensorSpec((None, *self.input_shape), tf.float32)
        self._logits_fn = tf.function(
            lambda x: self.net(x, training=False), input_signature=[image], reduce_retracing=True
        )
        self._embed_fn = tf.function(
            lambda x: self.net.embed(x, training=False), input_signature=[image], reduce_retracing=True
        )
        self._train_steps: dict[tuple[int, tuple[tuple[int, ...], ...]], Callable] = {}

    # ------------------------------------------------------------------ training
    def _train_step(self, n_pools: int, pool_of: tuple[tuple[int, ...], ...]) -> Callable:
        """The compiled step for one batch layout, traced once with a None batch dimension.

        The batch is ``x = concat[S, pool blocks]`` with ``|S|`` rows per block; adversary k
        sees ``concat[z_S, *(block b for b in pool_of[k])]`` (no block: S only, i.e. identity
        without a pool). The group head sees every block, so it reads z itself, and heads that
        read the same rows share one concat tensor (SIDANN's domain and identity heads): no third
        gradient path into z_S whose summation order could vary between traces. Its labels come
        in as ``group_labels``, one per row of the batch (D14); ``identity_alpha`` is the weight
        of the identity pool's rows (D15).
        """
        key = (n_pools, pool_of)
        if key in self._train_steps:
            return self._train_steps[key]
        net, optimizer, kinds = self.net, self.optimizer, self._kinds
        variables = net.trainable_variables

        def step(x, y, w, subjects, lams, group_labels, identity_alpha):
            with tf.GradientTape() as tape:
                z = net.encoder(x, training=True)  # one forward pass over S and the pool samples
                blocks = tf.split(z, n_pools + 1, axis=0)
                z_s = blocks[0]
                logits = net.classifier(z_s, training=True)
                loss_cls = tf.reduce_mean(w * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
                metrics = {"loss_cls": loss_cls, "acc_cls": _accuracy(y, logits)}
                total = loss_cls
                ones = tf.ones_like(w)
                rows = {(): z_s}  # pool blocks -> concat[z_S, blocks], one tensor per row set (heads share it)
                for i, (kind, pool_blocks) in enumerate(zip(kinds, pool_of)):
                    if kind == "group":  # every block: rows S, O, T?
                        z_k = z
                    else:
                        if pool_blocks not in rows:
                            rows[pool_blocks] = tf.concat([z_s, *(blocks[b] for b in pool_blocks)], axis=0)
                        z_k = rows[pool_blocks]
                    logits_k = net.heads[kind](gradient_reversal(z_k, lams[i]), training=True)
                    if kind == "identity":
                        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=subjects, logits=logits_k)
                        if pool_blocks:  # S rows weigh 1, pool rows alpha: every subject counts equally (D15)
                            weights = tf.concat([ones, identity_alpha * ones], axis=0)
                            loss = tf.reduce_sum(weights * ce) / tf.reduce_sum(weights)
                        else:
                            loss = tf.reduce_mean(ce)
                        acc = _accuracy(subjects, logits_k)
                    else:  # group: in-group 0 / out-group 1 (given per row); domain: source 0 / target 1
                        labels = group_labels if kind == "group" else tf.concat([tf.zeros_like(w), ones], axis=0)
                        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits_k[:, 0])
                        loss = tf.reduce_mean(bce)
                        predicted = tf.cast(logits_k[:, 0] > 0, tf.float32)
                        acc = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), tf.float32))
                    metrics[f"loss_{kind}"], metrics[f"acc_{kind}"] = loss, acc
                    total += loss
            grads = tape.gradient(total, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return metrics

        signature = [
            tf.TensorSpec((None, *self.input_shape), tf.float32),  # x: blocks [S, O?, T?]
            tf.TensorSpec((None,), tf.int64),  # y: emotion labels of S
            tf.TensorSpec((None,), tf.float32),  # w: class weight of each S sample
            tf.TensorSpec((None,), tf.int64),  # subjects: S, then the identity pool block
            tf.TensorSpec((len(kinds),), tf.float32),  # lambda of each adversary
            tf.TensorSpec((None,), tf.float32),  # group label of every batch row (S, O, T?); empty without group
            tf.TensorSpec((), tf.float32),  # weight of an identity-pool row relative to an S row
        ]
        compiled = tf.function(
            step, input_signature=signature, reduce_retracing=True, jit_compile=self.train_cfg.jit_compile
        )
        self._train_steps[key] = compiled
        return compiled

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        """Per-class weight of the emotion CE: ones, or ``n / (K * count_k)`` for ``class_weight="balanced"``."""
        if self.train_cfg.class_weight is None:
            return np.ones(self.n_classes, dtype=np.float32)
        counts = np.bincount(y, minlength=self.n_classes)
        return np.where(counts > 0, len(y) / (self.n_classes * np.maximum(counts, 1)), 0.0).astype(np.float32)

    def _check_data(self, name: str, data: LabeledData | UnlabeledData) -> None:
        if len(data) == 0:
            raise ValueError(f"{name} is empty")
        if data.x.shape[1:] != self.input_shape:
            raise ValueError(f"{name}.x has sample shape {data.x.shape[1:]}, expected {self.input_shape}")
        if isinstance(data, LabeledData) and (data.y.min() < 0 or data.y.max() >= self.n_classes):
            raise ValueError(f"{name}.y must hold class indices in [0, {self.n_classes})")

    def _check_subjects(self, name: str, subjects: np.ndarray) -> None:
        if subjects.min() < 0 or subjects.max() >= self.n_subjects:
            raise ValueError(
                f"The identity adversary needs subject indices in [0, {self.n_subjects}) for every {name} "
                f"sample, got values in [{subjects.min()}, {subjects.max()}]"
            )

    def fit(
        self,
        train: LabeledData,
        val: LabeledData | None = None,
        *,
        outgroup: UnlabeledData | None = None,
        target: UnlabeledData | None = None,
    ) -> FitResult:
        """Train for ``train_cfg.epochs`` epochs (one epoch = one pass over ``train``).

        ``outgroup`` feeds the ``group`` adversary and ``target`` the ``domain`` adversary.
        Given a ``target``, the ``group`` adversary also sees T: a target sample is out-group
        (1) if its subject is one of the out-group's subjects, else in-group (0), so the subject
        term covers source and target samples (Eq. 3, D14). The ``identity`` adversary contrasts
        S with T if ``target`` is given, else with O (else S only); a pool sample weighs
        ``n_pool_subjects / n_train_subjects`` in its cross-entropy, so every subject counts
        equally (D15). With ``val``, the weights of the best epoch on
        ``train_cfg.selection_metric`` are restored at the end (D16).
        """
        cfg, method = self.train_cfg, self.method
        for name, pool in (("outgroup", outgroup), ("target", target)):
            if pool is not None and not isinstance(pool, UnlabeledData):
                raise TypeError(f"{name} must be UnlabeledData: emotion labels of the {name} are never used")
        for kind, name, pool in (("group", "outgroup", outgroup), ("domain", "target", target)):
            if method.has(kind) and pool is None:
                raise ValueError(f"Method {method.name!r} has a {kind!r} adversary and needs the {name} pool, got None")

        # Pool blocks follow S in the batch: O (if drawn), then T (if drawn).
        identity, group = method.has("identity"), method.has("group")
        draw_outgroup = group or (identity and target is None and outgroup is not None)
        draw_target = method.has("domain") or (target is not None and (identity or group))
        named_pools = [
            (name, pool)
            for name, pool, drawn in (("outgroup", outgroup, draw_outgroup), ("target", target, draw_target))
            if drawn
        ]
        pools = [pool for _, pool in named_pools]
        outgroup_block = 1 if draw_outgroup else 0
        target_block = 1 + draw_outgroup if draw_target else 0
        identity_block = target_block or outgroup_block
        blocks_of = {
            "group": tuple(range(1, len(pools) + 1)),  # every block: S, O and, with a target, T
            "domain": (target_block,),
            "identity": (identity_block,) if identity_block else (),
        }
        pool_of = tuple(blocks_of[kind] for kind in self._kinds)

        self._check_data("train", train)
        if val is not None:
            self._check_data("val", val)
        for name, pool in named_pools:
            self._check_data(name, pool)
        level = logging.INFO if cfg.verbose else logging.DEBUG
        identity_pool, identity_alpha = None, None
        if identity:
            self._check_subjects("train", train.subjects)
            if identity_block:
                name, identity_pool = named_pools[identity_block - 1]
                self._check_subjects(name, identity_pool.subjects)
                n_pool_subjects = len(np.unique(identity_pool.subjects))
                n_train_subjects = len(np.unique(train.subjects))
                identity_alpha = n_pool_subjects / n_train_subjects
                logger.log(
                    level, "%s: identity pool rows weigh %.4g (%d pool / %d training subjects, D15)",
                    method.name, identity_alpha, n_pool_subjects, n_train_subjects,
                )
        alpha = np.float32(1.0 if identity_alpha is None else identity_alpha)
        target_is_out = None  # 1 for the target rows of an out-group subject, 0 for the others (D14)
        if group and draw_target:
            for name, pool in (("outgroup", outgroup), ("target", target)):
                if pool.subjects.min() < 0:
                    raise ValueError(
                        "With a target pool, the group adversary labels target samples by subject: every "
                        f"{name} sample needs a subject index >= 0"
                    )
            target_is_out = np.isin(target.subjects, np.unique(outgroup.subjects)).astype(np.float32)

        step_fn = self._train_step(len(pools), pool_of)
        sample_weight = self._class_weights(train.y)[train.y]
        rng = np.random.default_rng(self.seed)
        n, batch_size = len(train), cfg.batch_size
        total_steps = cfg.epochs * math.ceil(n / batch_size)
        select_column = _SELECTION_COLUMNS[cfg.selection_metric]
        lower_is_better = cfg.selection_metric == "loss"
        best_score: float | None = None
        best_epoch, best_weights, stale = 0, None, 0
        rows: list[dict[str, float]] = []
        step, fit_start = 0, time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            order = rng.permutation(n)
            sums: dict[str, float] = {}
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                draws = [rng.integers(0, len(pool), size=len(idx)) for pool in pools]
                x = np.concatenate([train.x[idx], *(pool.x[d] for pool, d in zip(pools, draws))])
                subjects = train.subjects[idx]
                if identity_pool is not None:
                    subjects = np.concatenate([subjects, identity_pool.subjects[draws[identity_block - 1]]])
                if not group:
                    group_labels = np.zeros(0, dtype=np.float32)
                else:  # in batch order: in-group S 0, out-group O 1, then T by subject
                    group_labels = np.repeat(np.array([0, 1], dtype=np.float32), len(idx))
                    if target_is_out is not None:
                        group_labels = np.concatenate([group_labels, target_is_out[draws[target_block - 1]]])
                factor = lambda_factor(step / total_steps, cfg.lambda_schedule)
                lams = [a.lam * factor for a in method.adversaries]  # Python floats for the history
                lam_args = np.asarray(lams, dtype=np.float32)
                out = step_fn(x, train.y[idx], sample_weight[idx], subjects, lam_args, group_labels, alpha)
                for key, value in out.items():
                    sums[key] = sums.get(key, 0.0) + float(value) * len(idx)
                step += 1

            row: dict[str, float] = {"epoch": epoch}
            row.update({f"lambda_{kind}": lam for kind, lam in zip(self._kinds, lams)})
            row.update({key: total / n for key, total in sums.items()})
            stop = False
            if val is not None:
                row.update(self._evaluate(val))
                score = row[select_column]
                if math.isfinite(score) and (
                    best_score is None or (score < best_score if lower_is_better else score > best_score)
                ):
                    best_score, best_epoch, best_weights, stale = score, epoch, self.net.get_weights(), 0
                else:
                    stale += 1
                    stop = cfg.patience is not None and stale >= cfg.patience
            row["seconds"] = time.perf_counter() - epoch_start
            rows.append(row)
            logger.log(
                level,
                "%s epoch %d/%d: %s",
                method.name,
                epoch,
                cfg.epochs,
                ", ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "epoch"),
            )
            if stop:
                logger.log(level, "%s: no improvement for %d epochs, stopping", method.name, stale)
                break

        if best_weights is not None:
            self.net.set_weights(best_weights)
            logger.log(level, "%s: restored epoch %d (%s=%.4f)", method.name, best_epoch, select_column, best_score)
        else:
            best_epoch = len(rows)
        history = pd.DataFrame(rows, columns=self._history_columns())
        seconds = time.perf_counter() - fit_start
        return FitResult(history, best_epoch, best_score, seconds, identity_alpha=identity_alpha)

    def _history_columns(self) -> list[str]:
        return [
            "epoch",
            *(f"lambda_{kind}" for kind in self._kinds),
            "loss_cls",
            "acc_cls",
            *(f"{metric}_{kind}" for kind in self._kinds for metric in ("loss", "acc")),
            "val_loss",
            "val_acc",
            "val_f1",
            "seconds",
        ]

    def _evaluate(self, data: LabeledData) -> dict[str, float]:
        """Emotion CE loss, accuracy and weighted F1 (fractions) on labeled data."""
        logits = self._batched(self._logits_fn, data.x, 512).astype(np.float64)
        loss = -np.mean(log_softmax(logits, axis=1)[np.arange(len(data)), data.y])
        pred = logits.argmax(axis=1)
        return {
            "val_loss": float(loss),
            "val_acc": float(np.mean(pred == data.y)),
            "val_f1": float(f1_score(data.y, pred, average="weighted", zero_division=0)),
        }

    # ---------------------------------------------------------------- inference
    @staticmethod
    def _batched(fn: Callable, x: np.ndarray, batch_size: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.concatenate([fn(x[i : i + batch_size]).numpy() for i in range(0, max(len(x), 1), batch_size)])

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Softmax emotion probabilities, shape (n, n_classes)."""
        return softmax(self._batched(self._logits_fn, x, batch_size).astype(np.float64), axis=1)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predicted emotion class per sample, shape (n,)."""
        return self.predict_proba(x, batch_size).argmax(axis=1)

    def embed(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Representation z (encoder output) per sample, shape (n, dense_units[-1])."""
        return self._batched(self._embed_fn, x, batch_size)
