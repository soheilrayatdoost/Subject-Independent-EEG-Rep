"""Gradient reversal: identity forward, -lambda * upstream gradient backward, no gradient for lambda."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from sieeg.models.grl import GradientReversal, gradient_reversal


def _grads(fn, x: np.ndarray, upstream: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Output of fn(x) and d<fn(x), upstream>/dx."""
    x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
        objective = tf.reduce_sum(y * upstream)
    return y.numpy(), tape.gradient(objective, x).numpy()


def test_forward_is_identity(rng):
    x = rng.standard_normal((4, 3)).astype(np.float32)
    np.testing.assert_array_equal(gradient_reversal(tf.constant(x), 0.7).numpy(), x)


@pytest.mark.parametrize("lam", [0.0, 0.15, 1.0, 2.5])
def test_gradient_is_minus_lambda_times_upstream(rng, lam):
    x = rng.standard_normal((5, 4)).astype(np.float32)
    upstream = rng.standard_normal((5, 4)).astype(np.float32)
    y, grad = _grads(lambda t: gradient_reversal(t, tf.constant(lam)), x, upstream)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_allclose(grad, -lam * upstream, rtol=1e-6, atol=1e-7)


def test_no_gradient_flows_into_lambda(rng):
    x = tf.constant(rng.standard_normal((3, 2)).astype(np.float32))
    lam = tf.constant(0.4)
    with tf.GradientTape() as tape:
        tape.watch(lam)
        objective = tf.reduce_sum(gradient_reversal(x, lam) ** 2)
    assert float(tape.gradient(objective, lam)) == 0.0


def test_works_inside_tf_function_with_a_changing_lambda(rng):
    x = tf.constant(rng.standard_normal((2, 3)).astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec((), tf.float32)])
    def grad_of_sum(lam):
        with tf.GradientTape() as tape:
            tape.watch(x)
            objective = tf.reduce_sum(gradient_reversal(x, lam))
        return tape.gradient(objective, x)

    for lam in (0.1, 0.5):
        np.testing.assert_allclose(grad_of_sum(tf.constant(lam)).numpy(), -lam * np.ones((2, 3)), rtol=1e-6)
    assert grad_of_sum.experimental_get_tracing_count() == 1


def test_layer_lam_property_and_setter(rng):
    layer = GradientReversal(lam=0.2)
    assert layer.lam == pytest.approx(0.2)
    assert not layer.trainable_weights and len(layer.non_trainable_weights) == 1
    x = rng.standard_normal((3, 4)).astype(np.float32)
    upstream = rng.standard_normal((3, 4)).astype(np.float32)
    _, grad = _grads(layer, x, upstream)
    np.testing.assert_allclose(grad, -0.2 * upstream, rtol=1e-6, atol=1e-7)

    layer.lam = 0.75
    assert layer.lam == pytest.approx(0.75)
    y, grad = _grads(layer, x, upstream)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_allclose(grad, -0.75 * upstream, rtol=1e-6, atol=1e-7)


def test_layer_in_a_functional_model_and_config_round_trip(rng):
    inputs = keras.Input((4,))
    outputs = keras.layers.Dense(1)(GradientReversal(lam=0.3, name="grl")(inputs))
    model = keras.Model(inputs, outputs)
    x = rng.standard_normal((6, 4)).astype(np.float32)
    assert model(x).shape == (6, 1)

    clone = GradientReversal.from_config(model.get_layer("grl").get_config())
    assert clone.lam == pytest.approx(0.3)
