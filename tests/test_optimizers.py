"""LegacyNadam reproduces the Keras 2.2.4 Nadam update (decision D12)."""

from __future__ import annotations

import dataclasses

import keras
import numpy as np
import pytest
import tensorflow as tf

from sieeg.config import TrainConfig
from sieeg.training.optimizers import LegacyNadam, make_optimizer

# Quadratic f(p) = 0.5 * sum(a * (p - c)^2) with gradient a * (p - c).
A = np.array([1.0, 4.0, 0.25, 10.0])
C = np.array([1.0, -2.0, 0.5, 3.0])
STEPS = 20


def quadratic_grad(p: np.ndarray) -> np.ndarray:
    return A * (p - C)


def keras2_nadam(grad_fn, p0, steps, *, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, schedule_decay=0.004):
    """``keras.optimizers.Nadam.get_updates`` of Keras 2.2.4 in NumPy (float64); returns the trajectory."""
    p = np.asarray(p0, dtype=np.float64)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    m_schedule = 1.0
    path = []
    for iterations in range(steps):
        t = iterations + 1
        g = grad_fn(p)
        momentum_cache_t = beta_1 * (1.0 - 0.5 * 0.96 ** (t * schedule_decay))
        momentum_cache_t_1 = beta_1 * (1.0 - 0.5 * 0.96 ** ((t + 1) * schedule_decay))
        m_schedule_new = m_schedule * momentum_cache_t
        m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
        g_prime = g / (1.0 - m_schedule_new)
        m = beta_1 * m + (1.0 - beta_1) * g
        m_t_prime = m / (1.0 - m_schedule_next)
        v = beta_2 * v + (1.0 - beta_2) * np.square(g)
        v_t_prime = v / (1.0 - beta_2**t)
        m_t_bar = (1.0 - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime
        p = p - lr * m_t_bar / (np.sqrt(v_t_prime) + epsilon)
        m_schedule = m_schedule_new
        path.append(p)
    return np.array(path)


def run_keras(optimizer, grad_fn, p0, steps) -> np.ndarray:
    var = keras.Variable(np.asarray(p0, dtype=np.float32))
    path = []
    for _ in range(steps):
        grad = grad_fn(var.numpy().astype(np.float64)).astype(np.float32)
        optimizer.apply_gradients([(tf.constant(grad), var)])
        path.append(var.numpy().copy())
    return np.array(path)


P0 = np.zeros(4)


@pytest.mark.parametrize("lr", [0.002, 0.05])
def test_matches_keras2_nadam_on_a_quadratic(lr):
    expected = keras2_nadam(quadratic_grad, P0, STEPS, lr=lr)
    actual = run_keras(LegacyNadam(learning_rate=lr), quadratic_grad, P0, STEPS)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    assert np.abs(expected[-1] - P0).max() > 10 * lr  # the parameters actually moved


def test_builds_without_add_optimizer_variables(monkeypatch):
    """Keras 3.0-3.9 (allowed by ``keras>=3.0``) lack ``Optimizer.add_optimizer_variables`` (Keras 3.10)."""
    for cls in LegacyNadam.__mro__:  # simulate them on a newer Keras
        if "add_optimizer_variables" in vars(cls):
            monkeypatch.delattr(cls, "add_optimizer_variables")
    optimizer = make_optimizer(TrainConfig())  # the default optimiser of AdversarialTrainer
    assert not hasattr(optimizer, "add_optimizer_variables")
    actual = run_keras(optimizer, quadratic_grad, P0, STEPS)
    np.testing.assert_allclose(actual, keras2_nadam(quadratic_grad, P0, STEPS, lr=0.002), rtol=1e-5, atol=1e-7)


def test_momentum_products_are_kept_per_variable():
    """Two variables updated together follow the same per-variable Keras-2 trajectories."""
    optimizer = LegacyNadam(learning_rate=0.05)
    a = keras.Variable(np.zeros(4, dtype=np.float32))
    b = keras.Variable(np.full(4, 2.0, dtype=np.float32))
    for _ in range(STEPS):
        grads = [quadratic_grad(var.numpy().astype(np.float64)).astype(np.float32) for var in (a, b)]
        optimizer.apply_gradients([(tf.constant(grads[0]), a), (tf.constant(grads[1]), b)])
    np.testing.assert_allclose(a.numpy(), keras2_nadam(quadratic_grad, P0, STEPS, lr=0.05)[-1], rtol=1e-5)
    np.testing.assert_allclose(b.numpy(), keras2_nadam(quadratic_grad, np.full(4, 2.0), STEPS, lr=0.05)[-1], rtol=1e-5)


def test_inside_tf_function_matches_eager():
    eager = run_keras(LegacyNadam(learning_rate=0.05), quadratic_grad, P0, 5)
    optimizer = LegacyNadam(learning_rate=0.05)
    var = keras.Variable(np.zeros(4, dtype=np.float32))
    optimizer.build([var])

    @tf.function
    def step():
        grad = tf.constant(A, tf.float32) * (var.value - tf.constant(C, tf.float32))
        optimizer.apply_gradients([(grad, var)])

    for expected in eager:
        step()
        np.testing.assert_allclose(var.numpy(), expected, rtol=1e-6, atol=1e-8)


def test_schedule_decay_one_is_keras3_nadam():
    """With schedule_decay=1 the Keras-2 formulas reduce to keras.optimizers.Nadam."""
    legacy = run_keras(LegacyNadam(learning_rate=0.05, schedule_decay=1.0), quadratic_grad, P0, STEPS)
    keras3 = run_keras(keras.optimizers.Nadam(learning_rate=0.05), quadratic_grad, P0, STEPS)
    np.testing.assert_allclose(legacy, keras3, rtol=1e-5, atol=1e-7)


def test_default_momentum_schedule_differs_from_keras3_nadam():
    legacy = run_keras(LegacyNadam(learning_rate=0.05), quadratic_grad, P0, STEPS)
    keras3 = run_keras(keras.optimizers.Nadam(learning_rate=0.05), quadratic_grad, P0, STEPS)
    assert np.abs(legacy[-1] - keras3[-1]).max() > 1e-3


def test_config_round_trip():
    optimizer = LegacyNadam(learning_rate=0.01, beta_1=0.8, beta_2=0.99, epsilon=1e-6, schedule_decay=0.002)
    config = optimizer.get_config()
    assert {k: config[k] for k in ("beta_1", "beta_2", "epsilon", "schedule_decay")} == {
        "beta_1": 0.8,
        "beta_2": 0.99,
        "epsilon": 1e-6,
        "schedule_decay": 0.002,
    }
    assert config["learning_rate"] == pytest.approx(0.01)
    clone = keras.optimizers.deserialize(keras.optimizers.serialize(optimizer))
    assert isinstance(clone, LegacyNadam)
    assert clone.schedule_decay == 0.002 and float(clone.learning_rate.numpy()) == pytest.approx(0.01)


def test_make_optimizer():
    cfg = TrainConfig()
    legacy = make_optimizer(cfg)
    assert isinstance(legacy, LegacyNadam)
    assert float(legacy.learning_rate.numpy()) == pytest.approx(0.002)
    assert (legacy.beta_1, legacy.beta_2, legacy.epsilon, legacy.schedule_decay) == (0.9, 0.999, 1e-7, 0.004)
    assert type(make_optimizer(dataclasses.replace(cfg, optimizer="nadam"))) is keras.optimizers.Nadam
    adam = make_optimizer(dataclasses.replace(cfg, optimizer="adam", learning_rate=1))
    assert type(adam) is keras.optimizers.Adam and float(adam.learning_rate.numpy()) == 1.0
    with pytest.raises(ValueError, match="sgd"):
        make_optimizer(dataclasses.replace(cfg, optimizer="sgd"))
