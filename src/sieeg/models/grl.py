"""Gradient reversal layer (GRL; Ganin & Lempitsky 2015), paper Sec. 5.1.

The GRL is the identity in the forward pass and multiplies the gradient by
``-lam`` in the backward pass. Placed between the encoder and a discriminator,
it lets a single minimisation of ``emotion loss + discriminator loss`` train the
discriminator to detect subjects / databases while pushing the encoder to hide
them (Eq. 1 / Eq. 3). ``lam`` only scales the reversed gradient; it never
weights a loss term.
"""

from __future__ import annotations

import keras
import tensorflow as tf


@tf.custom_gradient
def _reverse_gradient(x: tf.Tensor, lam: tf.Tensor):
    def grad(dy: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return -lam * dy, tf.zeros_like(lam)

    return tf.identity(x), grad


def gradient_reversal(x, lam) -> tf.Tensor:
    """Identity forward; backward ``(-lam * dy, 0)``. ``lam`` is a scalar (float, tensor or variable)."""
    x = tf.convert_to_tensor(x)
    lam = tf.cast(tf.convert_to_tensor(lam), x.dtype)
    return _reverse_gradient(x, lam)


@keras.saving.register_keras_serializable(package="sieeg")
class GradientReversal(keras.layers.Layer):
    """Keras layer around :func:`gradient_reversal`.

    ``lam`` lives in a non-trainable scalar weight, so it can be changed between
    steps (``layer.lam = 0.5``) without retracing a compiled function.
    """

    def __init__(self, lam: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._lam = self.add_weight(
            shape=(), initializer=keras.initializers.Constant(float(lam)), trainable=False, name="lam"
        )

    @property
    def lam(self) -> float:
        return float(keras.ops.convert_to_numpy(self._lam))

    @lam.setter
    def lam(self, value: float) -> None:
        self._lam.assign(float(value))

    def call(self, inputs):
        return gradient_reversal(inputs, self._lam)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        return {**super().get_config(), "lam": self.lam}
