"""Experiment 2 protocols: leave-one-subject-out within a database and transfer across databases.

Paper Sec. 5.3; docs/DESIGN.md 8.4 and decisions D13-D16.

* Within a database (:func:`run_within_database`): each subject is the test subject once.
  ``n_val_subjects`` of the other subjects (a seeded draw) validate for model selection and
  the rest train. The test subject is the out-group of SI-EEG (Eq. 1) and the identity pool
  of SINN (D15): its EEG is used during training, its emotion labels are not.
* Across databases (:func:`run_cross_database`): the source database is labeled and the
  target database is not. Validation subjects come from the source. A method with an
  in-group/out-group adversary trains one model per target subject, whose trials form the
  out-group; the other target subjects are in-group, as Eq. 3's subject term covers source
  and target samples (D14). The other methods train one model for the whole target database.

Emotion labels of test subjects and of the target database only score the predictions.
They never reach the trainer, model selection or class weights: target labels are not even
sent to the worker processes. Splits depend only on (subjects, n_val, seed), so every method
is trained and validated on the same subjects, which the paired tests across subjects need
(Table 1). The model that tests subject ``i`` (natural order) trains with seed
``cfg.seed * 1000 + i``. With ``cfg.protocol.subject_probe``, linear probes measure the
subject information left in each fold's representation (:mod:`sieeg.analysis.probe`; not in
the paper).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..config import ExperimentConfig, ModelConfig, TrainConfig
from ..features.featureset import FeatureSet, feature_config_differences
from ..features.topomap import images_from_config
from ..training.methods import MethodSpec, get_method
from ..utils import natural_sorted
from .metrics import classification_rate, summarize_predictions, weighted_f1
from .report import RunResult

logger = logging.getLogger(__name__)

#: Columns of ``RunResult.predictions`` and ``RunResult.folds`` (docs/DESIGN.md 8.3).
PREDICTION_COLUMNS = ("fold", "dataset", "subject", "trial", "rating", "y_true", "y_pred", "p1")
FOLD_COLUMNS = (
    "fold", "test_subjects", "val_subjects", "n_train", "n_val", "n_test", "n_outgroup", "n_target",
    "best_epoch", "best_val", "cr", "f1", "seconds", "probe_subject_acc", "probe_subject_chance", "probe_group_bacc",
)
#: Scores of the subject probes (``cfg.protocol.subject_probe``): fold columns ``probe_<key>``, ``summary["probe"]``.
PROBE_KEYS = ("subject_acc", "subject_chance", "group_bacc")
#: Values of ``cfg.protocol.cross_per_subject``.
CROSS_MODES = ("auto", "always", "never")


@dataclass
class Split:
    """Subject ids of one leave-one-subject-out fold, each list in natural order."""

    test: list[str]
    val: list[str]
    train: list[str]


def _draw_validation(candidates: list[str], n_val: int, rng: np.random.Generator) -> list[str]:
    """``n_val`` of ``candidates`` drawn without replacement, in natural order (D16).

    Raises:
        ValueError: unless ``0 <= n_val < len(candidates)``, so at least one subject is left for training.
    """
    if not 0 <= n_val < len(candidates):
        raise ValueError(
            f"n_val_subjects={n_val} must be >= 0 and leave at least one of the {len(candidates)} "
            "candidate subjects for training"
        )
    return natural_sorted(rng.choice(candidates, size=n_val, replace=False))


def loso_splits(subjects: Iterable[str], n_val: int, seed: int) -> list[Split]:
    """Leave-one-subject-out splits, one per subject in natural order (paper Sec. 5.3).

    The validation subjects of test subject ``i`` are ``n_val`` of the other subjects, drawn
    with ``numpy.random.default_rng([seed, i])``; all remaining subjects train. A split
    depends only on (subjects, n_val, seed), never on the method or on which folds run.
    """
    ordered = natural_sorted(subjects)
    splits = []
    for i, test in enumerate(ordered):
        others = ordered[:i] + ordered[i + 1 :]
        val = _draw_validation(others, n_val, np.random.default_rng([seed, i]))
        splits.append(Split(test=[test], val=val, train=[s for s in others if s not in val]))
    return splits


# ------------------------------------------------------------------- folds
@dataclass
class _Shared:
    """Images, labels and settings shared by every fold. Sent once to each worker process.

    ``y`` holds the emotion label of every row that may be training or validation data, and
    -1 for target-database rows, whose labels never leave the parent process. ``subjects``
    is the global subject index of every row (identity adversary, D15). ``subject_probe``
    runs the linear subject probes after training (``cfg.protocol.subject_probe``).
    """

    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    method: MethodSpec
    n_subjects: int
    model_cfg: ModelConfig
    train_cfg: TrainConfig
    subject_probe: bool = False


@dataclass
class _Fold:
    """One model to train: its name, seed, subjects, and its rows of :class:`_Shared` per role.

    ``outgroup`` and ``target`` are None when the method does not use that pool.
    """

    name: str
    seed: int
    test_subjects: list[str]
    val_subjects: list[str]
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    outgroup: np.ndarray | None = None
    target: np.ndarray | None = None


@dataclass
class _FoldOutput:
    """What a worker returns for one fold: test-row probabilities and the training record."""

    proba: np.ndarray  # (n_test, 2) emotion probabilities of the test rows
    history: pd.DataFrame
    best_epoch: int
    best_score: float | None
    seconds: float
    probe: dict[str, float]  # PROBE_KEYS -> fraction, NaN when the probe is off or undefined


def _fit_fold(shared: _Shared, fold: _Fold) -> _FoldOutput:
    """Train the model of ``fold`` in a fresh Keras session and predict its test rows.

    Only training and validation rows carry emotion labels. The out-group and target pools
    pass images and subject indices only, and the test rows are only predicted. The fold's
    entries in TensorFlow's gradient registry are removed at the end: each pins the fold's
    traced train step, which would otherwise stay in memory for the rest of the process.
    """
    import keras

    from ..training.trainer import AdversarialTrainer, LabeledData, UnlabeledData

    start = time.perf_counter()

    def labeled(rows: np.ndarray) -> LabeledData:
        return LabeledData(shared.x[rows], shared.y[rows], shared.subjects[rows])

    def unlabeled(rows: np.ndarray | None) -> UnlabeledData | None:
        return None if rows is None else UnlabeledData(shared.x[rows], shared.subjects[rows])

    keras.backend.clear_session()
    registry = _gradient_registry()
    registered = set(registry) if registry is not None else set()
    trainer = None
    try:
        trainer = AdversarialTrainer(
            shared.method,
            shared.x.shape[1:],
            n_classes=2,
            n_subjects=shared.n_subjects,
            model_cfg=shared.model_cfg,
            train_cfg=shared.train_cfg,
            seed=fold.seed,
        )
        fit = trainer.fit(
            labeled(fold.train),
            labeled(fold.val) if len(fold.val) else None,
            outgroup=unlabeled(fold.outgroup),
            target=unlabeled(fold.target),
        )
        proba = trainer.predict_proba(shared.x[fold.test])
        probe = _probe_fold(trainer, shared, fold) if shared.subject_probe else dict.fromkeys(PROBE_KEYS, math.nan)
    finally:
        del trainer  # the fold's network, optimiser and compiled step are not used again
        if registry is not None:  # tf.custom_gradient entries of this fold (GRL, optimiser) pin its step graph
            for key in [k for k in registry if k not in registered and str(k).startswith("CustomGradient-")]:
                registry.pop(key, None)
    return _FoldOutput(proba, fit.history, fit.best_epoch, fit.best_score, time.perf_counter() - start, probe)


def _gradient_registry() -> dict | None:
    """TensorFlow's process-global gradient registry (private API), or None if it is not available."""
    try:
        from tensorflow.python.framework import ops

        registry = ops._gradient_registry._registry
    except (ImportError, AttributeError):
        return None
    return registry if isinstance(registry, dict) else None


def _probe_fold(trainer: Any, shared: _Shared, fold: _Fold) -> dict[str, float]:
    """Linear probes of the subject information left in the fold's representation z (not in the paper).

    ``subject_acc`` / ``subject_chance``: decoding the subject of the validation rows, whose
    subjects the encoder never trained on (NaN with fewer than 2 validation subjects).
    ``group_bacc``: separating the in-group rows (training rows and the target-pool rows that
    are not tested) from the test rows. Uses subject indices only, never emotion labels.
    """
    from ..analysis.probe import group_probe, subject_probe

    scores = dict.fromkeys(PROBE_KEYS, math.nan)
    val_subjects = shared.subjects[fold.val]
    if len(np.unique(val_subjects)) >= 2:
        probe = subject_probe(trainer.embed(shared.x[fold.val]), val_subjects, seed=fold.seed)
        scores["subject_acc"], scores["subject_chance"] = probe["subject_acc"], probe["subject_chance"]
    in_group = fold.train if fold.target is None else np.concatenate([fold.train, np.setdiff1d(fold.target, fold.test)])
    z_in, z_test = trainer.embed(shared.x[in_group]), trainer.embed(shared.x[fold.test])
    scores["group_bacc"] = group_probe(z_in, z_test, seed=fold.seed)["group_bacc"]
    return scores


_worker_shared: _Shared | None = None  # set in each worker process by _init_worker


def _init_worker(shared: _Shared, intra_op_threads: int) -> None:
    """Process-pool initializer: cap TensorFlow's intra-op threads before any op runs; keep ``shared``."""
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    global _worker_shared
    _worker_shared = shared


def _fit_fold_in_worker(fold: _Fold) -> _FoldOutput:
    return _fit_fold(_worker_shared, fold)


def _fit_folds(shared: _Shared, folds: list[_Fold], jobs: int) -> Iterator[_FoldOutput]:
    """The output of every fold, in fold order. With ``jobs > 1``, folds train in spawned processes.

    Each worker uses ``cpu_count // workers`` intra-op threads. Splits, seeds and data are the
    same as with ``jobs=1``; only float32 summation order can differ with the thread count.
    """
    workers = min(jobs, len(folds))
    if workers <= 1:
        for fold in folds:
            yield _fit_fold(shared, fold)
        return
    threads = max(1, (os.cpu_count() or 1) // workers)
    logger.info("Training %d folds in %d processes, %d intra-op threads each", len(folds), workers, threads)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker, initargs=(shared, threads)) as pool:
        yield from pool.map(_fit_fold_in_worker, folds)


def _run(
    protocol: str,
    shared: _Shared,
    folds: list[_Fold],
    trials: pd.DataFrame,
    *,
    label: str,
    tag: str | None,
    train_dataset: str,
    test_dataset: str,
    cfg: ExperimentConfig,
    progress: bool,
) -> RunResult:
    """Train every fold, score its test predictions and assemble the run (docs/DESIGN.md 8.3).

    ``trials`` holds ``dataset, subject, trial, rating, y_true`` for every row of ``shared``.
    It stays in this process: the true labels of the test rows are only read here. A fold's
    ``best_val`` is the trainer's score of the selected epoch (``train.selection_metric``: a
    fraction for f1 and accuracy, the cross-entropy for loss), NaN without validation subjects.
    The ``probe_*`` fold columns are fractions, NaN unless ``shared.subject_probe``;
    ``summary["probe"]`` holds their means over folds (None when the probe is off).
    """
    level = logging.INFO if progress else logging.DEBUG
    predictions, records, histories = [], [], {}
    outputs = _fit_folds(shared, folds, cfg.protocol.jobs)
    for k, (fold, out) in enumerate(zip(folds, outputs, strict=True), start=1):
        pred = trials.iloc[fold.test].assign(fold=fold.name, y_pred=out.proba.argmax(axis=1), p1=out.proba[:, 1])
        pred = pred[list(PREDICTION_COLUMNS)].reset_index(drop=True)
        cr = classification_rate(pred["y_true"], pred["y_pred"])
        f1 = weighted_f1(pred["y_true"], pred["y_pred"])
        records.append(
            {
                "fold": fold.name,
                "test_subjects": fold.test_subjects,
                "val_subjects": fold.val_subjects,
                "n_train": len(fold.train),
                "n_val": len(fold.val),
                "n_test": len(fold.test),
                "n_outgroup": 0 if fold.outgroup is None else len(fold.outgroup),
                "n_target": 0 if fold.target is None else len(fold.target),
                "best_epoch": out.best_epoch,
                "best_val": np.nan if out.best_score is None else out.best_score,
                "cr": cr,
                "f1": f1,
                "seconds": out.seconds,
                **{f"probe_{key}": out.probe[key] for key in PROBE_KEYS},
            }
        )
        predictions.append(pred)
        histories[fold.name] = out.history
        logger.log(
            level,
            "%s %s %s fold %d/%d (%s): CR %.1f, F1 %.1f, best epoch %d, %.1f s",
            protocol, shared.method.name, label, k, len(folds), fold.name, cr, f1, out.best_epoch, out.seconds,
        )

    predictions = pd.concat(predictions, ignore_index=True)
    scores = summarize_predictions(predictions)
    probe = None
    if shared.subject_probe:  # mean over the folds where a probe is defined, NaN if none is
        probe = {key: _finite_mean(record[f"probe_{key}"] for record in records) for key in PROBE_KEYS}
    summary = {
        "protocol": protocol,
        "method": shared.method.name,
        "tag": tag,
        "label": label,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "lambdas": {a.kind: a.lam for a in shared.method.adversaries},
        "seed": cfg.seed,
        "n_folds": len(folds),
        "n_subjects": scores["n_subjects"],
        "n_trials": scores["n_trials"],
        "pooled": scores["pooled"],
        "subject_mean": scores["subject_mean"],
        "probe": probe,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sieeg_version": __version__,
    }
    return RunResult(summary, predictions, pd.DataFrame(records, columns=list(FOLD_COLUMNS)), histories)


def _finite_mean(values: Iterable[float]) -> float:
    """Mean of the finite ``values``; NaN if there is none."""
    finite = [float(v) for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


# ------------------------------------------------------------------ inputs
def _method_spec(method: str | MethodSpec, cfg: ExperimentConfig) -> MethodSpec:
    """A :class:`MethodSpec` as given, or a method name with the lambdas of ``cfg.protocol``
    (None: the paper's values)."""
    if isinstance(method, MethodSpec):
        return method
    return get_method(method, lambda_subject=cfg.protocol.lambda_subject, lambda_domain=cfg.protocol.lambda_domain)


def _rated(fs: FeatureSet, label: str) -> FeatureSet:
    """The trials of ``fs`` with a finite ``label`` rating."""
    keep = fs.has_rating(label)
    if not keep.any():
        raise ValueError(f"No trial of {fs.dataset} has a {label} rating")
    if not keep.all():
        logger.info("%s: leaving out %d trials without a %s rating", fs.dataset, int((~keep).sum()), label)
    return fs.subset(keep)


def _test_subjects(subject_ids: list[str], folds: Iterable[str] | None, dataset: str) -> list[str]:
    """All of ``subject_ids``, or the subjects listed in ``folds``, in natural order."""
    if folds is None:
        return list(subject_ids)
    wanted = {str(f) for f in ([folds] if isinstance(folds, str) else folds)}
    if not wanted:
        raise ValueError("folds lists no test subject")
    unknown = wanted.difference(subject_ids)
    if unknown:
        raise ValueError(
            f"Unknown test subjects {natural_sorted(unknown)}: {dataset} subjects with a rating are {subject_ids}"
        )
    return [s for s in subject_ids if s in wanted]


def _trial_table(fs: FeatureSet, label: str, y: np.ndarray) -> pd.DataFrame:
    """``dataset, subject, trial, rating, y_true`` per row of ``fs``."""
    return pd.DataFrame(
        {"dataset": fs.dataset, "subject": fs.subjects, "trial": fs.trials, "rating": fs.ratings(label), "y_true": y}
    )


def _common_montage(source: FeatureSet, target: FeatureSet) -> tuple[FeatureSet, FeatureSet]:
    """``source`` and ``target`` on the same channels, so their images share one grid.

    Channels match case-insensitively; if the sets differ, both keep their intersection in
    source order (with a warning).

    Raises:
        ValueError: if the bands differ or no channel is shared.
    """
    if source.bands != target.bands:
        raise ValueError(f"Bands differ: {source.dataset} has {source.bands}, {target.dataset} has {target.bands}")
    in_target = {c.lower() for c in target.ch_names}
    common = [c for c in source.ch_names if c.lower() in in_target]
    if len(common) == source.n_channels == target.n_channels:
        return source, target
    if not common:
        raise ValueError(f"{source.dataset} and {target.dataset} have no channel in common")
    logger.warning(
        "Channels differ between %s (%d) and %s (%d): using their %d common channels",
        source.dataset, source.n_channels, target.dataset, target.n_channels, len(common),
    )
    return source.pick_channels(common), target.pick_channels(common)


def _check_disjoint(source: FeatureSet, target: FeatureSet) -> None:
    """Raise if a target trial has exactly the band power of a source trial (channels in source order).

    That trial would be tested after training (or model selection) on its own label, e.g. when
    one feature file is passed twice, or again under another dataset name.
    """
    seen = {row.tobytes() for row in source.band_power}
    aligned = target.pick_channels(source.ch_names).band_power
    n_shared = sum(row.tobytes() in seen for row in aligned)
    if n_shared:
        raise ValueError(
            f"{n_shared} of {len(target)} target trials are identical to source trials: the target would be tested "
            "on labeled training data"
        )


# --------------------------------------------------------------- protocols
def run_within_database(
    fs: FeatureSet,
    method: str | MethodSpec,
    label: str,
    cfg: ExperimentConfig,
    *,
    folds: Iterable[str] | None = None,
    tag: str | None = None,
    progress: bool = True,
) -> RunResult:
    """Leave-one-subject-out evaluation inside one database (paper Sec. 5.3, Experiment 2).

    Args:
        fs: the database. Trials without a finite ``label`` rating are left out.
        method: a method name, with the lambdas of ``cfg.protocol`` (None: the paper's values),
            or a :class:`MethodSpec`. A method with a domain adversary needs a target database.
        label: ``"valence"`` or ``"arousal"``, binarised at ``cfg.protocol.label_threshold``.
        cfg: the experiment configuration. ``cfg.protocol.jobs > 1`` trains folds in parallel.
        folds: the test subjects to run (default: all). Splits and seeds do not depend on it.
        tag: free-form run tag stored in the summary.
        progress: log every finished fold at INFO level instead of DEBUG.

    Returns:
        One fold per test subject, named after it. The out-group (group and identity
        adversaries) is the test subject, plus the validation subjects if
        ``cfg.protocol.outgroup_includes_val``. The identity adversary weighs each out-group
        subject like one training subject (D15).

    Raises:
        ValueError: for a method with a domain adversary, unknown ``folds``, or too few subjects.
    """
    spec = _method_spec(method, cfg)
    if spec.requires_target_domain:
        raise ValueError(
            f"Method {spec.name!r} has a domain adversary and needs a target database: use run_cross_database"
        )
    fs = _rated(fs, label)
    subject_ids = fs.subject_ids
    splits = loso_splits(subject_ids, cfg.protocol.n_val_subjects, cfg.seed)
    chosen = set(_test_subjects(subject_ids, folds, fs.dataset))
    index = {s: i for i, s in enumerate(subject_ids)}
    y = fs.labels(label, cfg.protocol.label_threshold)
    shared = _Shared(
        x=images_from_config(fs, cfg),
        y=y,
        subjects=np.array([index[s] for s in fs.subjects], dtype=np.int64),
        method=spec,
        n_subjects=len(subject_ids),
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        subject_probe=cfg.protocol.subject_probe,
    )

    def rows_of(subjects: list[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(fs.subjects, subjects))

    uses_outgroup = spec.has("group") or spec.has("identity")  # identity contrasts S with the out-group (D15)
    fold_list = []
    for i, split in enumerate(splits):
        if split.test[0] not in chosen:
            continue
        test, val = rows_of(split.test), rows_of(split.val)
        outgroup = np.concatenate([test, val]) if cfg.protocol.outgroup_includes_val else test
        fold_list.append(
            _Fold(
                name=split.test[0],
                seed=cfg.seed * 1000 + i,
                test_subjects=split.test,
                val_subjects=split.val,
                train=rows_of(split.train),
                val=val,
                test=test,
                outgroup=outgroup if uses_outgroup else None,
            )
        )
    logger.log(
        logging.INFO if progress else logging.DEBUG,
        "within %s, %s, %s: %d subjects, %d folds, %d validation subjects per fold",
        fs.dataset, spec.name, label, len(subject_ids), len(fold_list), cfg.protocol.n_val_subjects,
    )
    return _run(
        "within", shared, fold_list, _trial_table(fs, label, y),
        label=label, tag=tag, train_dataset=fs.dataset, test_dataset=fs.dataset, cfg=cfg, progress=progress,
    )


def run_cross_database(
    source: FeatureSet,
    target: FeatureSet,
    method: str | MethodSpec,
    label: str,
    cfg: ExperimentConfig,
    *,
    folds: Iterable[str] | None = None,
    tag: str | None = None,
    progress: bool = True,
) -> RunResult:
    """Train on the labeled ``source`` database and test on the unlabeled ``target`` (paper Sec. 5.3).

    ``n_val_subjects`` source subjects (``default_rng(cfg.seed)``) validate and the other
    source subjects train. The domain, identity and group adversaries use every target trial
    as the target pool. ``cfg.protocol.cross_per_subject`` picks the mode (D14): ``"auto"``
    trains one model per target subject if the method has a group adversary, and one model
    otherwise; ``"always"`` and ``"never"`` force either mode. Per subject, the out-group is
    that subject's trials and the group adversary counts the other target subjects as in-group
    (Eq. 3); the fold is named after the subject and its seed is ``cfg.seed * 1000 + j`` (j:
    its position in natural order). With one model, the out-group is every target trial, so a
    group adversary duplicates a domain adversary (logged as a warning); the fold is ``"all"``
    and its seed ``cfg.seed * 1000``. ``cfg.protocol.outgroup_includes_val`` adds the
    validation subjects to the out-group. Differences between the extraction settings
    recorded in the two feature files are logged as a warning.

    Args:
        source, target: two different databases with the same bands. If their channels differ
            (case-insensitive), both keep the common channels. Trials without a finite ``label``
            rating are left out.
        method, label, cfg, tag, progress: as in :func:`run_within_database`. Every method is valid.
        folds: target subjects to test (default: all). With one model for the whole target, it
            only restricts the predicted subjects; training is unchanged.

    Raises:
        ValueError: if source and target have the same dataset name or share a trial (identical
            band power), for different bands, no common channel, an unknown mode, unknown
            ``folds`` or too few source subjects.
    """
    spec = _method_spec(method, cfg)
    mode = cfg.protocol.cross_per_subject
    if mode not in CROSS_MODES:
        raise ValueError(f"cross_per_subject must be one of {CROSS_MODES}, got {mode!r}")
    per_subject = mode == "always" or (mode == "auto" and spec.has("group"))
    if source.dataset == target.dataset:
        raise ValueError(
            f"Source and target are both named {source.dataset!r}: the cross-database protocol needs two different "
            "databases (one database is evaluated within the database)"
        )
    source, target = _common_montage(source, target)
    differences = feature_config_differences(source, target)
    if differences:
        logger.warning(
            "%s and %s were extracted with different settings, which adds to the gap between the databases: %s",
            source.dataset, target.dataset, "; ".join(differences),
        )
    source, target = _rated(source, label), _rated(target, label)
    _check_disjoint(source, target)
    if spec.has("group") and not per_subject:
        logger.warning(
            "%s with one model for the whole target database: the out-group is every target trial, so the group "
            "adversary's labels equal the domain labels and it duplicates a domain adversary (D14)",
            spec.name,
        )
    source_ids, target_ids = source.subject_ids, target.subject_ids
    chosen = _test_subjects(target_ids, folds, target.dataset)
    val_subjects = _draw_validation(source_ids, cfg.protocol.n_val_subjects, np.random.default_rng(cfg.seed))
    train_subjects = [s for s in source_ids if s not in val_subjects]

    # Global subject index (D15): source subjects, then target subjects, each in natural order.
    index = {s: i for i, s in enumerate(source_ids)}
    index_target = {t: len(source_ids) + j for j, t in enumerate(target_ids)}
    n_source = len(source)
    y_source = source.labels(label, cfg.protocol.label_threshold)
    shared = _Shared(
        x=np.concatenate([images_from_config(source, cfg), images_from_config(target, cfg)]),
        y=np.concatenate([y_source, np.full(len(target), -1, dtype=np.int64)]),
        subjects=np.array(
            [index[s] for s in source.subjects] + [index_target[t] for t in target.subjects], dtype=np.int64
        ),
        method=spec,
        n_subjects=len(source_ids) + len(target_ids),
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        subject_probe=cfg.protocol.subject_probe,
    )
    # The true target labels stay in this process, to score the predictions.
    trials = pd.concat(
        [
            _trial_table(source, label, y_source),
            _trial_table(target, label, target.labels(label, cfg.protocol.label_threshold)),
        ],
        ignore_index=True,
    )

    def target_rows(subjects: list[str]) -> np.ndarray:
        return n_source + np.flatnonzero(np.isin(target.subjects, subjects))

    train = np.flatnonzero(np.isin(source.subjects, train_subjects))
    val = np.flatnonzero(np.isin(source.subjects, val_subjects))
    all_target = n_source + np.arange(len(target))
    # The group adversary needs the target pool too: its subject term covers source and target (Eq. 3, D14).
    pool = all_target if any(spec.has(kind) for kind in ("domain", "identity", "group")) else None

    def make_fold(name: str, seed: int, test_subjects: list[str], outgroup: np.ndarray) -> _Fold:
        if cfg.protocol.outgroup_includes_val:
            outgroup = np.concatenate([outgroup, val])
        return _Fold(
            name=name,
            seed=seed,
            test_subjects=test_subjects,
            val_subjects=val_subjects,
            train=train,
            val=val,
            test=target_rows(test_subjects),
            outgroup=outgroup if spec.has("group") else None,
            target=pool,
        )

    if per_subject:
        fold_list = [
            make_fold(t, cfg.seed * 1000 + j, [t], target_rows([t])) for j, t in enumerate(target_ids) if t in chosen
        ]
    else:
        fold_list = [make_fold("all", cfg.seed * 1000, chosen, all_target)]
    logger.log(
        logging.INFO if progress else logging.DEBUG,
        "cross %s -> %s, %s, %s: %d training and %d validation source subjects, %d target subjects, %s",
        source.dataset, target.dataset, spec.name, label, len(train_subjects), len(val_subjects), len(target_ids),
        "one model per target subject" if per_subject else "one model",
    )
    return _run(
        "cross", shared, fold_list, trials,
        label=label, tag=tag, train_dataset=source.dataset, test_dataset=target.dataset, cfg=cfg, progress=progress,
    )
