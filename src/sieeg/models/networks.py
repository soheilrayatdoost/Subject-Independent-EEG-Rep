"""Encoder, classifier and discriminator heads (paper Sec. 5.1 / 5.3, decisions D8-D10).

Encoder: 2x conv16, maxpool, 2x conv32, maxpool, FC256, FC128 (relu convolutions,
tanh fully connected layers) on the 32x32x6 spectral topographic maps. The
emotion classifier (FC64 tanh) and every discriminator (FC64 elu) read the
128-d representation. Weights use the Keras defaults (glorot_uniform kernels,
zero biases), as in the original Keras code.
"""

from __future__ import annotations

from collections.abc import Sequence

import keras

from ..config import ModelConfig


def build_encoder(input_shape: Sequence[int], cfg: ModelConfig, name: str = "encoder") -> keras.Model:
    """Convolutional encoder; 574,528 parameters for a (32, 32, 6) input with the paper config."""
    inputs = keras.Input(shape=tuple(input_shape), dtype="float32", name=f"{name}_input")
    x = inputs
    for b, block in enumerate(cfg.conv_blocks, start=1):
        for i, filters in enumerate(block, start=1):
            x = keras.layers.Conv2D(
                filters,
                cfg.kernel_size,
                padding=cfg.padding,
                activation=cfg.conv_activation,
                data_format="channels_last",
                dtype="float32",
                name=f"{name}_conv{b}_{i}",
            )(x)
        x = keras.layers.MaxPooling2D(2, data_format="channels_last", dtype="float32", name=f"{name}_pool{b}")(x)
    x = keras.layers.Flatten(data_format="channels_last", dtype="float32", name=f"{name}_flatten")(x)
    if cfg.dropout > 0:
        x = keras.layers.Dropout(cfg.dropout, dtype="float32", name=f"{name}_dropout")(x)
    for j, units in enumerate(cfg.dense_units, start=1):
        x = keras.layers.Dense(units, activation=cfg.dense_activation, dtype="float32", name=f"{name}_fc{j}")(x)
    return keras.Model(inputs, x, name=name)


def build_head(
    input_dim: int,
    n_outputs: int,
    *,
    hidden: int,
    activation: str,
    dropout: float = 0.0,
    name: str,
) -> keras.Model:
    """``Dense(hidden, activation) [-> Dropout] -> Dense(n_outputs)`` logits; ``hidden == 0`` is a linear read-out."""
    inputs = keras.Input(shape=(int(input_dim),), dtype="float32", name=f"{name}_input")
    x = inputs
    if hidden > 0:
        x = keras.layers.Dense(hidden, activation=activation, dtype="float32", name=f"{name}_hidden")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, dtype="float32", name=f"{name}_dropout")(x)
    outputs = keras.layers.Dense(n_outputs, dtype="float32", name=f"{name}_logits")(x)
    return keras.Model(inputs, outputs, name=name)


class AdversarialNet(keras.Model):
    """Shared encoder + emotion classifier + one discriminator head per adversary kind.

    ``adversaries`` maps kind -> number of logits: ``{"group": 1, "domain": 1,
    "identity": n_subjects}``. The heads output raw logits; the gradient reversal
    is applied by the trainer between :attr:`encoder` and each head (Sec. 5.1).
    Heads are built in sorted kind order: weight files store them by position.
    """

    def __init__(self, input_shape: Sequence[int], n_classes: int, adversaries: dict[str, int], cfg: ModelConfig):
        super().__init__(name="adversarial_net", dtype="float32")  # not the global policy, which would cast the images
        self.encoder = build_encoder(input_shape, cfg)
        z_dim = int(self.encoder.output.shape[-1])
        self.classifier = build_head(
            z_dim,
            n_classes,
            hidden=cfg.classifier_hidden,
            activation=cfg.classifier_activation,
            dropout=cfg.dropout,
            name="classifier",
        )
        self.heads = {
            kind: build_head(
                z_dim,
                n_outputs,
                hidden=cfg.discriminator_hidden,
                activation=cfg.discriminator_activation,
                dropout=cfg.dropout,
                name=f"{kind}_head",
            )
            for kind, n_outputs in sorted(adversaries.items())
        }
        self.built = True  # every sub-model is a functional model, built above

    def call(self, x, training=False):
        """Emotion logits."""
        return self.classifier(self.encoder(x, training=training), training=training)

    def embed(self, x, training=False):
        """Representation z (output of the last encoder layer)."""
        return self.encoder(x, training=training)
