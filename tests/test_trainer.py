"""AdversarialTrainer: batches, losses, model selection and inference (DESIGN.md 7.5, D11-D16)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import log_softmax

from sieeg.data.synthetic import make_synthetic_featureset
from sieeg.training.methods import AdversarySpec, MethodSpec, available_methods, get_method
from sieeg.training.schedules import lambda_factor
from sieeg.training.trainer import AdversarialTrainer, FitResult, LabeledData, UnlabeledData

SHAPE = (8, 8, 6)
N_SUBJECTS = 8  # train 0-3, val 4, out-group 5, target 6-7


def make_trainer(cfg, method="si_eeg", *, seed=0, **train_overrides) -> AdversarialTrainer:
    if isinstance(method, str):
        method = get_method(method)
    train_cfg = dataclasses.replace(cfg.train, **train_overrides)
    return AdversarialTrainer(method, SHAPE, n_subjects=N_SUBJECTS, model_cfg=cfg.model, train_cfg=train_cfg, seed=seed)


def spy_steps(trainer: AdversarialTrainer) -> list[list[np.ndarray]]:
    """Record the arguments (x, y, w, subjects, lambdas, group_labels, identity_alpha) of every train-step call."""
    calls: list[list[np.ndarray]] = []
    compile_step = trainer._train_step

    def recording(*layout):
        step = compile_step(*layout)

        def call(*args):
            calls.append([np.asarray(a) for a in args])
            return step(*args)

        return call

    trainer._train_step = recording
    return calls


@pytest.fixture
def toy(rng):
    """Random images; channel 0 marks the pool (train 0, out-group 1, target -1), pixel (0, 0, 1) the row."""

    def images(n: int, marker: float) -> np.ndarray:
        x = rng.standard_normal((n, *SHAPE)).astype(np.float32)
        x[..., 0] = marker
        x[:, 0, 0, 1] = np.arange(n)
        return x

    return SimpleNamespace(
        train=LabeledData(images(40, 0.0), rng.integers(0, 2, 40), np.repeat(np.arange(4), 10)),
        val=LabeledData(images(12, 0.0), rng.integers(0, 2, 12), np.full(12, 4)),
        outgroup=UnlabeledData(images(10, 1.0), np.full(10, 5)),
        target=UnlabeledData(images(20, -1.0), np.repeat([6, 7], 10)),
    )


def outgroup_of_target(toy, subject: int = 6) -> UnlabeledData:
    """The target trials of ``subject`` as the out-group (cross-database per-subject mode, D14).

    Marked 1 like ``toy.outgroup``; pixel (0, 0, 1) keeps the row index in ``toy.target``.
    """
    rows = toy.target.subjects == subject
    x = toy.target.x[rows].copy()
    x[..., 0] = 1.0
    return UnlabeledData(x, toy.target.subjects[rows])


@pytest.fixture
def easy():
    """Synthetic band power with a strong valence effect, projected to 8x8x6 images.

    Subjects S01-S04 train, S05 validates, S06 is the out-group.
    """
    fs = make_synthetic_featureset(n_subjects=6, n_trials=40, emotion_effect=1.0, seed=0)
    projection = np.random.default_rng(0).standard_normal((fs.n_channels, 64)) / np.sqrt(fs.n_channels)
    x = np.einsum("ncb,cp->npb", fs.normalized("subject"), projection).reshape(len(fs), *SHAPE)
    y = fs.labels("valence")
    subjects = np.array([fs.subject_ids.index(s) for s in fs.subjects])
    train, val, test = subjects < 4, subjects == 4, subjects == 5
    return SimpleNamespace(
        train=LabeledData(x[train], y[train], subjects[train]),
        val=LabeledData(x[val], y[val], subjects[val]),
        outgroup=UnlabeledData(x[test], subjects[test]),
    )


# ------------------------------------------------------------------- structure
@pytest.mark.parametrize("name", available_methods())
def test_each_method_builds_its_heads_and_history(tiny_config, toy, name):
    method = get_method(name)
    trainer = make_trainer(tiny_config, method, epochs=1)
    kinds = [a.kind for a in method.adversaries]
    assert sorted(trainer.net.heads) == sorted(kinds)
    for kind, head in trainer.net.heads.items():
        assert head.output.shape[-1] == (N_SUBJECTS if kind == "identity" else 1)
    assert trainer.net.classifier.output.shape[-1] == 2

    result = trainer.fit(toy.train, toy.val, outgroup=toy.outgroup, target=toy.target)
    assert isinstance(result, FitResult)
    assert list(result.history.columns) == [
        "epoch",
        *(f"lambda_{k}" for k in kinds),
        "loss_cls",
        "acc_cls",
        *(f"{m}_{k}" for k in kinds for m in ("loss", "acc")),
        "val_loss",
        "val_acc",
        "val_f1",
        "seconds",
    ]
    assert result.history["epoch"].tolist() == [1]
    assert not result.history.isna().any().any()
    for spec in method.adversaries:
        assert result.history[f"lambda_{spec.kind}"].iloc[0] == pytest.approx(spec.lam)


def test_history_without_validation(tiny_config, toy):
    result = make_trainer(tiny_config, epochs=2).fit(toy.train, outgroup=toy.outgroup)
    history = result.history
    assert history["epoch"].tolist() == [1, 2]
    assert history[["val_loss", "val_acc", "val_f1"]].isna().all().all()
    assert history[["loss_cls", "acc_cls", "loss_group", "acc_group"]].notna().all().all()
    assert result.best_epoch == 2 and result.best_score is None
    assert (history["seconds"] > 0).all() and result.seconds >= history["seconds"].sum()


def test_predict_outputs(tiny_config, toy):
    trainer = make_trainer(tiny_config, "cnn")
    proba = trainer.predict_proba(toy.val.x)
    assert proba.shape == (12, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_array_equal(trainer.predict(toy.val.x), proba.argmax(axis=1))
    np.testing.assert_allclose(trainer.predict_proba(toy.val.x, batch_size=5), proba, atol=1e-6)
    z = trainer.embed(toy.val.x, batch_size=5)
    assert z.shape == (12, tiny_config.model.dense_units[-1])
    np.testing.assert_allclose(z, trainer.net.embed(toy.val.x).numpy(), atol=1e-6)


# ----------------------------------------------------------------------- pools
@pytest.mark.parametrize(
    ("name", "pools", "missing"),
    [
        ("si_eeg", {"target": "target"}, "outgroup"),
        ("dann", {"outgroup": "outgroup"}, "target"),
        ("sida_eeg", {"outgroup": "outgroup"}, "target"),
        ("sida_eeg", {"target": "target"}, "outgroup"),
        ("sidann", {}, "target"),
    ],
)
def test_missing_pool_raises(tiny_config, toy, name, pools, missing):
    trainer = make_trainer(tiny_config, name)
    with pytest.raises(ValueError, match=missing):
        trainer.fit(toy.train, **{key: getattr(toy, value) for key, value in pools.items()})


def test_pools_never_carry_emotion_labels(tiny_config, toy):
    assert not hasattr(toy.outgroup, "y")
    trainer = make_trainer(tiny_config, "si_eeg")
    with pytest.raises(TypeError, match="UnlabeledData"):
        trainer.fit(toy.train, outgroup=toy.val)


def test_batch_composition(tiny_config, toy):
    """S is a shuffled pass over train (last batch smaller); O and T blocks of |S| rows each follow it."""
    trainer = make_trainer(tiny_config, "sida_eeg", epochs=2)
    calls = spy_steps(trainer)
    trainer.fit(toy.train, outgroup=toy.outgroup, target=toy.target)
    assert [len(y) for _, y, *_ in calls] == [16, 16, 8] * 2
    for epoch in range(2):
        rows = np.concatenate([x[: len(y), 0, 0, 1] for x, y, *_ in calls[3 * epoch : 3 * epoch + 3]])
        np.testing.assert_array_equal(np.sort(rows), np.arange(40))  # every training sample once per epoch
    for x, y, w, subjects, lams, group_labels, _ in calls:
        n = len(y)
        assert x.shape == (3 * n, *SHAPE)
        markers = x[:, 0, 0, 0]
        assert np.all(markers[:n] == 0) and np.all(markers[n : 2 * n] == 1) and np.all(markers[2 * n :] == -1)
        np.testing.assert_array_equal(y, toy.train.y[x[:n, 0, 0, 1].astype(int)])
        np.testing.assert_array_equal(w, 1.0)
        np.testing.assert_allclose(lams, [0.1, 0.15])  # (domain, group) lambdas of SIDA-EEG
        # group labels of the batch rows S, O, T: the target subjects 6 and 7 are not out-group subjects (5)
        np.testing.assert_array_equal(group_labels, np.r_[np.zeros(n), np.ones(n), np.zeros(n)])


@pytest.mark.parametrize("name", ["si_eeg", "sida_eeg"])
def test_group_labels_mark_the_out_group_subjects_in_the_target_pool(tiny_config, toy, name):
    """Across databases the group head sees S (0), O (1) and T. A target sample is out-group only if
    its subject is (Eq. 3, D14): the other target subjects are in-group, so the group labels are
    not a function of the domain labels (every T sample is domain 1)."""
    trainer = make_trainer(tiny_config, name, epochs=2)
    calls = spy_steps(trainer)
    trainer.fit(toy.train, outgroup=outgroup_of_target(toy, 6), target=toy.target)
    t_labels = []
    for x, y, *_, group_labels, _ in calls:
        n = len(y)
        assert x.shape == (3 * n, *SHAPE)  # S, O, T: SI-EEG also draws the target pool
        assert np.all(x[n : 2 * n, 0, 0, 0] == 1) and np.all(x[2 * n :, 0, 0, 0] == -1)
        t_subjects = toy.target.subjects[x[2 * n :, 0, 0, 1].astype(int)]
        np.testing.assert_array_equal(group_labels, np.r_[np.zeros(n), np.ones(n), t_subjects == 6])
        t_labels.append(group_labels[2 * n :])
    t_labels = np.concatenate(t_labels)
    assert 0 < t_labels.mean() < 1  # T samples of both subjects: both group labels
    (compiled,) = trainer._train_steps.values()
    assert compiled.experimental_get_tracing_count() == 1  # new labels every step: no retracing


def test_group_labels_within_a_database(tiny_config, toy):
    """Without a target pool the group head contrasts S (in-group 0) with O (out-group 1) only."""
    trainer = make_trainer(tiny_config, "si_eeg", epochs=1)
    calls = spy_steps(trainer)
    trainer.fit(toy.train, outgroup=UnlabeledData(toy.outgroup.x))  # no subject index needed here
    for x, y, *_, group_labels, _ in calls:
        n = len(y)
        assert x.shape == (2 * n, *SHAPE)
        np.testing.assert_array_equal(group_labels, np.r_[np.zeros(n), np.ones(n)])


def test_group_with_a_target_pool_needs_subject_indices(tiny_config, toy):
    for outgroup, target in ((toy.outgroup, UnlabeledData(toy.target.x)), (UnlabeledData(toy.outgroup.x), toy.target)):
        with pytest.raises(ValueError, match="subject index"):
            make_trainer(tiny_config, "si_eeg").fit(toy.train, outgroup=outgroup, target=target)


@pytest.mark.parametrize(
    ("name", "pools", "pool_marker", "alpha"),
    [
        ("sinn", ("outgroup",), 1.0, 1 / 4),  # within database: P = out-group (1 subject vs 4 training subjects)
        ("sinn", ("outgroup", "target"), -1.0, 2 / 4),  # P = target whenever a target is given (subjects 6, 7)
        ("sidann", ("target",), -1.0, 2 / 4),
        ("sinn", (), None, None),  # no pool: S only
    ],
)
def test_identity_labels_combine_source_and_pool_subjects(tiny_config, toy, name, pools, pool_marker, alpha):
    """Pool rows weigh n_pool_subjects / n_train_subjects, so every subject counts equally (D15)."""
    trainer = make_trainer(tiny_config, name, epochs=1)
    calls = spy_steps(trainer)
    result = trainer.fit(toy.train, **{pool: getattr(toy, pool) for pool in pools})
    assert result.identity_alpha == alpha
    for x, y, _, subjects, *_, identity_alpha in calls:
        n = len(y)
        rows = x[:, 0, 0, 1].astype(int)
        np.testing.assert_array_equal(subjects[:n], toy.train.subjects[rows[:n]])
        assert identity_alpha == np.float32(1.0 if alpha is None else alpha)
        if pool_marker is None:
            assert len(subjects) == n and len(x) == n
            continue
        assert len(subjects) == 2 * n and len(x) == 2 * n
        assert np.all(x[n:, 0, 0, 0] == pool_marker)
        pool = toy.outgroup if pool_marker == 1.0 else toy.target
        np.testing.assert_array_equal(subjects[n:], pool.subjects[rows[n:]])


def test_identity_needs_subject_indices(tiny_config, toy):
    with pytest.raises(ValueError, match="n_subjects"):
        AdversarialTrainer(get_method("sinn"), SHAPE, n_subjects=1, model_cfg=tiny_config.model)
    unknown = LabeledData(toy.train.x, toy.train.y)  # subjects default to -1
    with pytest.raises(ValueError, match="train"):
        make_trainer(tiny_config, "sinn").fit(unknown, outgroup=toy.outgroup)
    with pytest.raises(ValueError, match="target"):
        make_trainer(tiny_config, "sidann").fit(toy.train, target=UnlabeledData(toy.target.x))


def test_invalid_settings_and_data(tiny_config, toy):
    for overrides in ({"selection_metric": "auc"}, {"class_weight": "auto"}, {"lambda_schedule": "linear"}):
        with pytest.raises(ValueError):
            make_trainer(tiny_config, "cnn", **overrides)
    trainer = make_trainer(tiny_config, "cnn")
    with pytest.raises(ValueError, match="sample shape"):
        trainer.fit(LabeledData(toy.train.x[:, :4, :4], toy.train.y))
    with pytest.raises(ValueError, match="class indices"):
        trainer.fit(LabeledData(toy.train.x, toy.train.y + 1))


def test_balanced_class_weight(tiny_config, toy):
    y = np.r_[np.zeros(30, int), np.ones(10, int)]
    train = LabeledData(toy.train.x, y, toy.train.subjects)
    trainer = make_trainer(tiny_config, "cnn", epochs=1, class_weight="balanced")
    expected = np.array([40 / (2 * 30), 40 / (2 * 10)])
    np.testing.assert_allclose(trainer._class_weights(train.y), expected, rtol=1e-6)
    calls = spy_steps(trainer)
    trainer.fit(train)
    for _, y_batch, w, *_ in calls:
        np.testing.assert_allclose(w, expected[y_batch], rtol=1e-6)


def test_dann_lambda_schedule(tiny_config, toy):
    trainer = make_trainer(tiny_config, "si_eeg", epochs=3, lambda_schedule="dann")
    calls = spy_steps(trainer)
    history = trainer.fit(toy.train, outgroup=toy.outgroup).history
    total = 9  # 3 epochs x 3 steps
    expected = [0.2 * lambda_factor(step / total, "dann") for step in range(total)]
    np.testing.assert_allclose([lams[0] for _, _, _, _, lams, *_ in calls], expected, rtol=1e-6)
    np.testing.assert_allclose(history["lambda_group"], expected[2::3], rtol=1e-6)  # last step of each epoch
    (compiled,) = trainer._train_steps.values()
    assert compiled.experimental_get_tracing_count() == 1  # new lambdas and a smaller last batch: no retracing


# -------------------------------------------------------------------- learning
def test_encoder_receives_the_reversed_adversary_gradient(tiny_config, toy):
    """One step: the first Nadam update moves every weight by about -lr * sign(gradient), so the
    encoder must follow sign(dL_cls - lam dL_group) and the head sign(dL_group) (Eq. 1 via the GRL)."""
    lam = 1.0
    method = MethodSpec("si_eeg", "", (AdversarySpec("group", lam),))
    trainer = make_trainer(tiny_config, method, epochs=1, batch_size=len(toy.train))  # exactly one step
    net = trainer.net
    parts = {"encoder": net.encoder, "classifier": net.classifier, "head": net.heads["group"]}
    before = {key: model.get_weights() for key, model in parts.items()}
    calls = spy_steps(trainer)
    trainer.fit(toy.train, outgroup=toy.outgroup)
    after = {key: model.get_weights() for key, model in parts.items()}
    ((x, y, *_),) = calls
    for key, model in parts.items():
        model.set_weights(before[key])

    n = len(y)
    with tf.GradientTape(persistent=True) as tape:
        z = net.encoder(x)
        loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, net.classifier(z[:n])))
        labels = np.r_[np.zeros(n), np.ones(n)].astype(np.float32)
        loss_group = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, net.heads["group"](z)[:, 0]))
    encoder_vars = net.encoder.trainable_weights
    g_cls = [g.numpy() for g in tape.gradient(loss_cls, encoder_vars)]
    g_group = [g.numpy() for g in tape.gradient(loss_group, encoder_vars)]
    g_head = [g.numpy() for g in tape.gradient(loss_group, net.heads["group"].trainable_weights)]

    def flat(arrays):
        return np.concatenate([a.ravel() for a in arrays])

    def assert_moved_against(gradient, key):
        step = flat(before[key]) - flat(after[key])
        mask = np.abs(gradient) > 1e-3 * np.abs(gradient).max()
        np.testing.assert_array_equal(np.sign(step[mask]), np.sign(gradient[mask]))
        return step, mask

    reversed_grad = flat(g_cls) - lam * flat(g_group)
    step, mask = assert_moved_against(reversed_grad, "encoder")
    plain = flat(g_cls) + lam * flat(g_group)
    assert np.any(np.sign(step[mask]) != np.sign(plain[mask]))  # the test can tell the two signs apart
    assert_moved_against(flat(g_head), "head")


def test_lambda_zero_leaves_the_encoder_to_the_emotion_loss(tiny_config, toy):
    """lambda only lives inside the GRL: at lambda = 0 the head still learns, the encoder ignores it."""
    cnn = make_trainer(tiny_config, "cnn", epochs=1)
    cnn.fit(toy.train)
    zero = make_trainer(tiny_config, MethodSpec("si_eeg", "", (AdversarySpec("group", 0.0),)), epochs=1)
    head_init = zero.net.heads["group"].get_weights()
    result = zero.fit(toy.train, outgroup=toy.outgroup)
    assert result.history["lambda_group"].iloc[0] == 0.0
    for model in ("encoder", "classifier"):
        for a, b in zip(getattr(cnn.net, model).get_weights(), getattr(zero.net, model).get_weights(), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    changed = [not np.allclose(a, b) for a, b in zip(head_init, zero.net.heads["group"].get_weights())]
    assert all(changed)


@pytest.mark.parametrize(
    ("adversaries", "pools"),
    [
        ((("domain", 0.1), ("group", 0.37)), ("outgroup_of_target", "target")),  # SIDA-EEG across databases
        ((("domain", 0.1), ("identity", 0.6)), ("target",)),  # SIDANN: identity pool = target, alpha 2/4
        ((("group", 0.25), ("identity", 0.45)), ("outgroup",)),  # within a database: identity alpha 1/4
    ],
)
def test_each_head_reverses_the_gradient_with_its_own_lambda(tiny_config, toy, adversaries, pools):
    """One full-batch SGD(lr=1) step moves every weight by exactly -gradient. The encoder must move by
    dL_cls - sum_k lam_k dL_k and head k by dL_k, with L_k recomputed here from the batch: group
    labels by out-group subject (D14), identity pool rows weighted n_pool / n_train subjects (D15).
    Exact float32 algebra: conftest turns TensorFloat-32 off, whose error of about 5e-4 exceeds the
    1e-4 tolerance."""
    data = {"outgroup_of_target": outgroup_of_target(toy, 6), "outgroup": toy.outgroup, "target": toy.target}
    given = {pool.replace("_of_target", ""): data[pool] for pool in pools}
    method = MethodSpec("custom", "", tuple(AdversarySpec(kind, lam) for kind, lam in adversaries))
    trainer = make_trainer(tiny_config, method, epochs=1, batch_size=len(toy.train))  # exactly one step
    trainer.optimizer = keras.optimizers.SGD(learning_rate=1.0)  # the step uses the optimiser of its first fit
    trainer.optimizer.build(trainer.net.trainable_variables)
    net = trainer.net
    before = [v.numpy().copy() for v in net.trainable_variables]
    calls = spy_steps(trainer)
    history = trainer.fit(toy.train, **given).history
    moved = {id(v): b - v.numpy() for v, b in zip(net.trainable_variables, before)}  # the gradient
    for v, b in zip(net.trainable_variables, before):
        v.assign(b)
    ((x, y, *_, lams),) = [call[:5] for call in calls]
    np.testing.assert_allclose(lams, [lam for _, lam in adversaries], rtol=1e-6)

    n, rows, marker = len(y), x[:, 0, 0, 1].astype(int), x[:, 0, 0, 0]
    s_rows, t_rows, o_rows = rows[marker == 0], rows[marker == -1], rows[marker == 1]
    zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
    with tf.GradientTape(persistent=True) as tape:
        z = net.encoder(x, training=True)
        z_s, z_t, z_o = (tf.boolean_mask(z, marker == m) for m in (0, -1, 1))
        logits = net.classifier(z_s, training=True)
        losses = {"cls": tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(toy.train.y[s_rows], logits))}
        for kind, _ in adversaries:
            if kind == "identity":  # P = T if a target is given, else O
                if "target" in given:
                    pool_z, pool_subjects = z_t, toy.target.subjects[t_rows]
                else:
                    pool_z, pool_subjects = z_o, toy.outgroup.subjects[o_rows]
                alpha = len(np.unique(pool_subjects)) / 4  # 4 training subjects
                labels = np.r_[toy.train.subjects[s_rows], pool_subjects]
                identity_logits = net.heads[kind](tf.concat([z_s, pool_z], 0))
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, identity_logits)
                weights = np.r_[ones, alpha * ones]
                losses[kind] = tf.reduce_sum(weights * ce) / weights.sum()
                continue
            if kind == "domain":
                z_k, labels = tf.concat([z_s, z_t], 0), np.r_[zeros, ones]
            elif "target" in given:  # group across databases: S 0, T 1 for subject 6 only, O 1
                z_k, labels = tf.concat([z_s, z_t, z_o], 0), np.r_[zeros, toy.target.subjects[t_rows] == 6, ones]
            else:
                z_k, labels = tf.concat([z_s, z_o], 0), np.r_[zeros, ones]
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels.astype(np.float32), net.heads[kind](z_k)[:, 0])
            losses[kind] = tf.reduce_mean(bce)

    def gradient(loss, variables) -> np.ndarray:
        return np.concatenate([g.numpy().ravel() for g in tape.gradient(loss, variables)])

    def assert_moved_by(variables, expected):
        step = np.concatenate([moved[id(v)].ravel() for v in variables])
        assert np.abs(step - expected).max() <= 1e-4 * np.abs(expected).max()

    encoder = net.encoder.trainable_weights
    adversarial = {kind: gradient(losses[kind], encoder) for kind, _ in adversaries}
    g_cls = gradient(losses["cls"], encoder)
    assert_moved_by(encoder, g_cls - sum(lam * adversarial[kind] for kind, lam in adversaries))
    assert_moved_by(net.classifier.trainable_weights, gradient(losses["cls"], net.classifier.trainable_weights))
    for kind, _ in adversaries:
        assert_moved_by(net.heads[kind].trainable_weights, gradient(losses[kind], net.heads[kind].trainable_weights))
        assert history[f"loss_{kind}"].iloc[0] == pytest.approx(float(losses[kind]), rel=1e-5)
    assert history["loss_cls"].iloc[0] == pytest.approx(float(losses["cls"]), rel=1e-5)
    # The check tells the heads apart: with the lambdas swapped, the encoder would move differently.
    (k1, lam1), (k2, lam2) = adversaries
    swapped = g_cls - lam2 * adversarial[k1] - lam1 * adversarial[k2]
    correct = g_cls - lam1 * adversarial[k1] - lam2 * adversarial[k2]
    assert np.abs(swapped - correct).max() > 1e-2 * np.abs(correct).max()


@pytest.mark.parametrize("dropout_in", ["encoder and classifier", "encoder", "classifier"])
def test_dropout_is_on_in_the_train_step_and_off_at_inference(tiny_config, toy, dropout_in):
    """The step runs the encoder and the classifier with training=True; validation, predict and embed
    with training=False. Leaving dropout in one sub-model only checks that sub-model's flag on its own."""
    hidden = 0 if dropout_in == "encoder" else tiny_config.model.classifier_hidden  # 0: linear read-out, no dropout
    model = dataclasses.replace(tiny_config.model, dropout=0.9, classifier_hidden=hidden)
    trainer = make_trainer(dataclasses.replace(tiny_config, model=model), "cnn", epochs=1, batch_size=len(toy.train))
    if dropout_in == "classifier":
        trainer.net.encoder.get_layer("encoder_dropout").rate = 0.0  # before fit traces the train step
    logits = trainer.net(toy.train.x, training=False).numpy().astype(np.float64)
    deterministic = -np.mean(log_softmax(logits, axis=1)[np.arange(len(toy.train)), toy.train.y])
    result = trainer.fit(toy.train, toy.val)  # one step, at the initial weights
    assert not np.isclose(result.history["loss_cls"].iloc[0], deterministic, rtol=1e-3)  # dropout was active
    np.testing.assert_array_equal(trainer.predict_proba(toy.val.x), trainer.predict_proba(toy.val.x))
    np.testing.assert_array_equal(trainer.embed(toy.val.x), trainer.embed(toy.val.x))
    assert trainer._evaluate(toy.val) == trainer._evaluate(toy.val)


@pytest.mark.parametrize("name", ["cnn", "si_eeg"])
def test_learns_an_easy_problem(tiny_config, easy, name):
    trainer = make_trainer(tiny_config, name, epochs=15)
    result = trainer.fit(easy.train, easy.val, outgroup=easy.outgroup)
    assert result.best_score > 0.7  # validation weighted F1; chance is about 0.5
    assert (trainer.predict(easy.val.x) == easy.val.y).mean() > 0.7


@pytest.mark.parametrize("name", available_methods())
def test_same_seed_gives_identical_predictions(tiny_config, toy, name):
    def run(seed):
        trainer = make_trainer(tiny_config, name, seed=seed, epochs=3)
        trainer.fit(toy.train, toy.val, outgroup=toy.outgroup, target=toy.target)
        return trainer.predict_proba(toy.val.x)

    first = run(5)
    # on a GPU this needs the deterministic kernels of conftest: TF_DETERMINISTIC_OPS=1
    np.testing.assert_array_equal(first, run(5))
    assert not np.allclose(first, run(6))


# ------------------------------------------------------------- model selection
def script_validation(trainer, column: str, scores: list[float]) -> list[list[np.ndarray]]:
    """Replace validation by scripted scores; returns the weights seen at each evaluation."""
    snapshots: list[list[np.ndarray]] = []
    queue = list(scores)

    def evaluate(_data):
        snapshots.append(trainer.net.get_weights())
        values = {"val_loss": 1.0, "val_acc": 0.5, "val_f1": 0.5}
        values[column] = queue.pop(0)
        return values

    trainer._evaluate = evaluate
    return snapshots


def assert_weights_equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_restores_the_earliest_best_epoch(tiny_config, toy):
    trainer = make_trainer(tiny_config, "si_eeg", epochs=4)
    snapshots = script_validation(trainer, "val_f1", [0.5, 0.8, 0.8, 0.6])
    result = trainer.fit(toy.train, toy.val, outgroup=toy.outgroup)
    assert result.best_epoch == 2 and result.best_score == 0.8
    assert result.history["val_f1"].tolist() == [0.5, 0.8, 0.8, 0.6]
    assert_weights_equal(trainer.net.get_weights(), snapshots[1])
    assert not np.array_equal(snapshots[1][0], snapshots[3][0])  # training went on after the best epoch


def test_loss_selection_prefers_lower(tiny_config, toy):
    trainer = make_trainer(tiny_config, "cnn", epochs=3, selection_metric="loss")
    snapshots = script_validation(trainer, "val_loss", [0.9, 0.4, 0.7])
    result = trainer.fit(toy.train, toy.val)
    assert result.best_epoch == 2 and result.best_score == 0.4
    assert_weights_equal(trainer.net.get_weights(), snapshots[1])


def test_patience_stops_early(tiny_config, toy):
    trainer = make_trainer(tiny_config, "cnn", epochs=6, selection_metric="accuracy", patience=2)
    script_validation(trainer, "val_acc", [0.5, 0.6, 0.6, 0.55, 0.9, 0.9])
    result = trainer.fit(toy.train, toy.val)
    assert result.history["epoch"].tolist() == [1, 2, 3, 4]
    assert result.best_epoch == 2 and result.best_score == 0.6


def test_restored_model_reproduces_its_validation_scores(tiny_config, toy):
    trainer = make_trainer(tiny_config, "si_eeg", epochs=4)
    result = trainer.fit(toy.train, toy.val, outgroup=toy.outgroup)
    history = result.history
    best = history.iloc[result.best_epoch - 1]
    assert result.best_epoch == int(history["val_f1"].idxmax()) + 1  # earliest maximum
    assert result.best_score == best["val_f1"]
    proba = trainer.predict_proba(toy.val.x)
    assert (proba.argmax(axis=1) == toy.val.y).mean() == pytest.approx(best["val_acc"])
    loss = -np.mean(np.log(proba[np.arange(len(toy.val)), toy.val.y]))
    assert loss == pytest.approx(best["val_loss"], rel=1e-5)


@pytest.mark.parametrize("policy", ["float64", "mixed_float16"])
def test_trains_in_float32_whatever_the_global_floatx_and_dtype_policy(tiny_config, toy, policy):
    """~/.keras/keras.json may set floatx or a dtype policy; the network still trains and predicts in float32."""
    tiny_config = dataclasses.replace(tiny_config, model=dataclasses.replace(tiny_config.model, dropout=0.2))
    previous = keras.config.floatx(), keras.config.dtype_policy()
    try:
        # The reference trains under float32 whatever keras.json says (floatx float16 rounds the learning rate).
        keras.config.set_floatx("float32")
        keras.config.set_dtype_policy("float32")
        reference = make_trainer(tiny_config, "si_eeg", epochs=1)  # dropout > 0 builds the Dropout layers too
        reference.fit(toy.train, toy.val, outgroup=toy.outgroup)
        keras.config.set_floatx("float64")
        keras.config.set_dtype_policy(policy)  # Keras caches the policy on first use; set_floatx alone does not change it
        trainer = make_trainer(tiny_config, "si_eeg", epochs=1)
        trainer.fit(toy.train, toy.val, outgroup=toy.outgroup)
        z, proba = trainer.embed(toy.val.x), trainer.predict_proba(toy.val.x)
    finally:
        keras.config.set_floatx(previous[0])
        keras.config.set_dtype_policy(previous[1])
    assert z.dtype == np.float32
    assert {w.dtype for w in trainer.net.weights} == {"float32"}
    np.testing.assert_array_equal(proba, reference.predict_proba(toy.val.x))
    assert (keras.config.floatx(), keras.config.dtype_policy().name) == (previous[0], previous[1].name)
