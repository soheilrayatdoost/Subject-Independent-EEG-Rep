"""Encoder / heads / AdversarialNet (paper Sec. 5.1, decisions D8-D10)."""

from __future__ import annotations

import dataclasses

import keras
import numpy as np

from sieeg.config import ModelConfig
from sieeg.models.networks import AdversarialNet, build_encoder, build_head


def test_paper_encoder_has_574528_parameters():
    encoder = build_encoder((32, 32, 6), ModelConfig())
    assert encoder.count_params() == 574_528
    assert encoder.output.shape == (None, 128)
    layers = encoder.layers[1:]
    kinds = [type(layer).__name__ for layer in layers]
    block = ["Conv2D", "Conv2D", "MaxPooling2D"]
    assert kinds == [*block, *block, "Flatten", "Dense", "Dense"]
    activations = {
        (type(layer).__name__, layer.activation.__name__) for layer in layers if hasattr(layer, "activation")
    }
    assert activations == {("Conv2D", "relu"), ("Dense", "tanh")}
    assert [layer.filters for layer in encoder.layers if isinstance(layer, keras.layers.Conv2D)] == [16, 16, 32, 32]
    assert [layer.units for layer in encoder.layers if isinstance(layer, keras.layers.Dense)] == [256, 128]


def test_encoder_is_channels_last_whatever_the_global_image_format():
    previous = keras.config.image_data_format()
    keras.config.set_image_data_format("channels_first")
    try:
        encoder = build_encoder((32, 32, 6), ModelConfig())
        z = encoder(np.zeros((2, 32, 32, 6), dtype=np.float32))
    finally:
        keras.config.set_image_data_format(previous)
    assert encoder.count_params() == 574_528
    assert encoder.output.shape == (None, 128)
    image_layers = [
        layer
        for layer in encoder.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D, keras.layers.Flatten))
    ]
    assert len(image_layers) == 7
    assert {layer.data_format for layer in image_layers} == {"channels_last"}
    assert z.shape == (2, 128)


def test_tiny_encoder_output_shape(tiny_config, rng):
    encoder = build_encoder((8, 8, 6), tiny_config.model)
    z = encoder(rng.standard_normal((5, 8, 8, 6)).astype(np.float32))
    assert z.shape == (5, 16)
    assert not any(isinstance(layer, keras.layers.Dropout) for layer in encoder.layers)


def test_encoder_dropout_after_flatten(tiny_config):
    cfg = dataclasses.replace(tiny_config.model, dropout=0.5)
    names = [type(layer).__name__ for layer in build_encoder((8, 8, 6), cfg).layers]
    assert names[names.index("Flatten") + 1] == "Dropout"


def test_head_with_hidden_layer():
    head = build_head(16, 3, hidden=8, activation="elu", dropout=0.25, name="h")
    dense = [layer for layer in head.layers if isinstance(layer, keras.layers.Dense)]
    assert [d.units for d in dense] == [8, 3]
    assert dense[0].activation.__name__ == "elu" and dense[1].activation.__name__ == "linear"
    assert any(isinstance(layer, keras.layers.Dropout) for layer in head.layers)
    assert head.count_params() == 16 * 8 + 8 + 8 * 3 + 3


def test_head_without_hidden_layer_is_a_linear_readout():
    head = build_head(16, 1, hidden=0, activation="tanh", name="h")
    dense = [layer for layer in head.layers if isinstance(layer, keras.layers.Dense)]
    assert len(dense) == 1 and dense[0].units == 1
    assert head.count_params() == 16 + 1


def test_adversarial_net_submodels(tiny_config, rng):
    cfg = tiny_config.model
    net = AdversarialNet((8, 8, 6), 2, {"domain": 1, "identity": 7}, cfg)
    assert set(net.heads) == {"domain", "identity"}
    assert net.heads["domain"].output.shape == (None, 1)
    assert net.heads["identity"].output.shape == (None, 7)
    assert net.classifier.output.shape == (None, 2)
    hidden = net.classifier.layers[1]
    assert hidden.units == cfg.classifier_hidden and hidden.activation.__name__ == cfg.classifier_activation
    hidden = net.heads["domain"].layers[1]
    assert hidden.units == cfg.discriminator_hidden and hidden.activation.__name__ == cfg.discriminator_activation

    x = rng.standard_normal((4, 8, 8, 6)).astype(np.float32)
    assert net(x).shape == (4, 2)
    assert net.embed(x).shape == (4, cfg.dense_units[-1])
    n_params = sum(m.count_params() for m in [net.encoder, net.classifier, *net.heads.values()])
    assert net.count_params() == n_params == sum(int(np.prod(w.shape)) for w in net.trainable_weights)


def test_adversarial_net_without_adversaries(tiny_config):
    net = AdversarialNet((8, 8, 6), 2, {}, tiny_config.model)
    assert net.heads == {}
    assert len(net.trainable_weights) == len(net.encoder.trainable_weights) + len(net.classifier.trainable_weights)


def test_weights_round_trip(tiny_config, rng, tmp_path):
    keras.utils.set_random_seed(1)
    net = AdversarialNet((8, 8, 6), 2, {"group": 1, "domain": 1}, tiny_config.model)
    keras.utils.set_random_seed(2)
    other = AdversarialNet((8, 8, 6), 2, {"group": 1, "domain": 1}, tiny_config.model)
    x = rng.standard_normal((3, 8, 8, 6)).astype(np.float32)
    assert not np.allclose(net(x), other(x))

    path = tmp_path / "net.weights.h5"
    net.save_weights(path)
    other.load_weights(path)
    for a, b in zip(net.get_weights(), other.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(net(x), other(x))
    np.testing.assert_array_equal(net.heads["group"](net.embed(x)), other.heads["group"](other.embed(x)))


def test_weights_load_by_head_kind_whatever_the_dict_order(tiny_config, rng, tmp_path):
    keras.utils.set_random_seed(1)
    net = AdversarialNet((8, 8, 6), 2, {"group": 1, "domain": 1}, tiny_config.model)
    other = AdversarialNet((8, 8, 6), 2, {"domain": 1, "group": 1}, tiny_config.model)
    path = tmp_path / "net.weights.h5"
    net.save_weights(path)
    other.load_weights(path)
    z = net.embed(rng.standard_normal((3, 8, 8, 6)).astype(np.float32))
    for kind in ("group", "domain"):
        np.testing.assert_array_equal(net.heads[kind](z), other.heads[kind](z))
