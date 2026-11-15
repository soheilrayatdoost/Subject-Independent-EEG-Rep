"""Spectral topographic maps (paper Sec. 5.1; docs/DESIGN.md 6.3 and D6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from sieeg.config import BIOSEMI32_CHANNELS, apply_overrides
from sieeg.features import topomap
from sieeg.features.topomap import (
    TopomapProjector,
    azimuthal_equidistant,
    build_images,
    images_from_config,
    montage_center,
    montage_positions,
)

CHANNELS = list(BIOSEMI32_CHANNELS)
METHODS = ("cubic", "linear")


@pytest.fixture(scope="module", params=METHODS)
def projector(request) -> TopomapProjector:
    return TopomapProjector(CHANNELS, method=request.param)


def _grid(p: TopomapProjector) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of every pixel: columns follow xs, rows follow ys."""
    return np.meshgrid(p.xs, p.ys)


def _pixel(p: TopomapProjector, name: str) -> tuple[float, float]:
    """Fractional (row, col) of an electrode."""
    x, y = p.positions_2d[CHANNELS.index(name)]
    return (p.ys[0] - y) / (p.ys[0] - p.ys[1]), (x - p.xs[0]) / (p.xs[1] - p.xs[0])


# ---------------------------------------------------------------- geometry
def test_montage_positions_are_case_insensitive_metres():
    pos = montage_positions(["fp1", "CZ", "oz"])
    np.testing.assert_array_equal(pos, montage_positions(["Fp1", "Cz", "Oz"]))
    assert pos.shape == (3, 3)
    assert 0.05 < pos[1, 2] < 0.15  # Cz is about 10 cm above the origin
    # MNE head frame: +x right, +y anterior, +z up
    fp1, t8 = montage_positions(["Fp1", "T8"])
    assert fp1[0] < 0 < fp1[1] and t8[0] > 0


def test_montage_positions_missing_channels_raise():
    with pytest.raises(KeyError, match=r"\['Nope', 'EXG1'\]"):
        montage_positions(["Cz", "Nope", "EXG1"])


def test_montage_center_fits_a_sphere():
    import mne

    # biosemi32 template positions lie on a 95 mm sphere around the origin.
    np.testing.assert_allclose(montage_center("biosemi32"), 0.0, atol=1e-9)
    ch_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    radii = np.linalg.norm(np.array(list(ch_pos.values())) - montage_center("standard_1005"), axis=1)
    assert radii.std() / radii.mean() < 0.1


def test_azimuthal_equidistant_maps_colatitude_to_plane_distance():
    center = np.array([0.01, -0.02, 0.03])
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    directions = np.array(
        [
            [0.0, 0.0, 1.0],  # vertex
            [1.0, 0.0, 0.0],  # right
            [0.0, 1.0, 0.0],  # anterior
            [-1.0, 0.0, 0.0],  # left
            [0.0, -np.cos(0.2), -np.sin(0.2)],  # posterior, below the equator
            [s * np.cos(0.5), s * np.sin(0.5), c],  # 45 degrees from the vertex
        ]
    )
    expected = np.array(
        [
            [0.0, 0.0],
            [np.pi / 2, 0.0],
            [0.0, np.pi / 2],
            [-np.pi / 2, 0.0],
            [0.0, -(np.pi / 2 + 0.2)],
            [np.pi / 4 * np.cos(0.5), np.pi / 4 * np.sin(0.5)],
        ]
    )
    for radius in (0.09, 2.0):  # only the direction matters
        np.testing.assert_allclose(azimuthal_equidistant(center + radius * directions, center), expected, atol=1e-12)


def test_azimuthal_equidistant_equals_eeglearn_azim_proj(rng):
    pos = rng.standard_normal((50, 3))
    x, y, z = pos.T
    elev = np.arctan2(z, np.sqrt(x**2 + y**2))  # EEGLearn cart2sph
    az = np.arctan2(y, x)
    eeglearn = np.column_stack([(np.pi / 2 - elev) * np.cos(az), (np.pi / 2 - elev) * np.sin(az)])  # pol2cart
    np.testing.assert_allclose(azimuthal_equidistant(pos, np.zeros(3)), eeglearn, atol=1e-12)


def test_grid_spans_bounding_box_with_anterior_rows_first(projector):
    x, y = projector.positions_2d.T
    assert projector.positions_2d.shape == (32, 2)
    np.testing.assert_allclose(projector.xs, np.linspace(x.min(), x.max(), 32))
    np.testing.assert_allclose(projector.ys, np.linspace(y.max(), y.min(), 32))
    pos = dict(zip(CHANNELS, projector.positions_2d))
    assert pos["Fp1"][1] > pos["Fz"][1] > pos["Cz"][1] > pos["Pz"][1] > pos["Oz"][1]
    assert pos["T7"][0] < pos["C3"][0] < pos["Cz"][0] < pos["C4"][0] < pos["T8"][0]


def test_mask_is_the_convex_hull(projector):
    assert projector.mask.shape == (32, 32) and projector.mask.dtype == bool
    assert 0.6 < projector.mask.mean() < 0.8
    x, y = _grid(projector)
    hull = Delaunay(projector.positions_2d).find_simplex(np.column_stack([x.ravel(), y.ravel()])) >= 0
    np.testing.assert_array_equal(projector.mask, hull.reshape(32, 32))


# ------------------------------------------------------------- interpolation
def test_constant_field_is_constant_inside_hull(projector):
    image = projector.transform(np.full(32, 3.5))
    assert image.shape == (32, 32) and image.dtype == np.float32
    np.testing.assert_allclose(image[projector.mask], 3.5, rtol=1e-6)
    assert np.all(image[~projector.mask] == 0.0)


def test_linear_field_in_y_is_larger_in_top_rows(projector):
    image = projector.transform(projector.positions_2d[:, 1])
    _, y = _grid(projector)
    np.testing.assert_allclose(image[projector.mask], y[projector.mask], atol=1e-5)
    rows = np.ma.masked_array(image, ~projector.mask).mean(axis=1)
    assert rows[1:8].min() > rows[24:31].max()


def test_linear_field_in_x_increases_left_to_right(projector):
    image = projector.transform(projector.positions_2d[:, 0])
    x, _ = _grid(projector)
    np.testing.assert_allclose(image[projector.mask], x[projector.mask], atol=1e-5)
    for row, inside in zip(image, projector.mask):
        assert np.all(np.diff(row[inside]) > 0)


def test_electrode_values_appear_at_their_pixels_anterior_up(projector):
    images = projector.transform(np.eye(32))  # one unit impulse per electrode
    for i, name in enumerate(CHANNELS):
        row, col = _pixel(projector, name)
        peak = np.unravel_index(images[i].argmax(), images[i].shape)
        assert np.hypot(peak[0] - row, peak[1] - col) < 2.0, name
        r, c = round(row), round(col)
        if projector.mask[r, c]:  # electrodes on the hull may fall just outside the grid's hull pixels
            assert images[i, r, c] > 0.8, name
            assert np.delete(images[:, r, c], i).max() < 0.2, name
    peak_row = {n: np.unravel_index(images[CHANNELS.index(n)].argmax(), (32, 32))[0] for n in ("Fz", "Cz", "Pz")}
    assert peak_row["Fz"] < peak_row["Cz"] < peak_row["Pz"]


def test_outside_hull_and_nan_become_fill_value():
    p = TopomapProjector(CHANNELS, method="linear", fill_value=-7.0)
    image = p.transform(np.ones(32))
    assert np.all(image[~p.mask] == -7.0)
    assert np.all(image[p.mask] == 1.0)
    values = np.ones(32)
    values[CHANNELS.index("Cz")] = np.nan
    image = p.transform(values)
    assert not np.isnan(image).any()
    assert (image == -7.0).sum() > (~p.mask).sum()  # triangles touching Cz are filled too


def test_batched_transform_equals_per_sample(projector, rng, monkeypatch):
    values = rng.standard_normal((2, 3, 32))
    monkeypatch.setattr(topomap, "_CHUNK", 4)  # 6 maps -> chunks of 4 and 2
    batched = projector.transform(values)
    assert batched.shape == (2, 3, 32, 32) and batched.dtype == np.float32
    single = np.array([[projector.transform(v) for v in row] for row in values])
    np.testing.assert_allclose(batched, single, atol=1e-6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="interpolation"):
        TopomapProjector(CHANNELS, method="nearest")
    with pytest.raises(ValueError, match="32 channels"):
        TopomapProjector(CHANNELS, method="linear").transform(np.zeros((4, 31)))


# --------------------------------------------------------------------- images
def test_build_images_is_nhwc_float32_of_normalized_features(small_featureset):
    fs = small_featureset
    images = build_images(fs)
    assert images.shape == (len(fs), 32, 32, 6)
    assert images.dtype == np.float32 and images.flags["C_CONTIGUOUS"]
    p = TopomapProjector(fs.ch_names)
    np.testing.assert_allclose(images[..., 3], p.transform(fs.normalized("subject")[:, :, 3]), atol=1e-6)
    assert np.all(images[:, ~p.mask, :] == 0.0)
    # Features are z-scored per subject before interpolation, so each subject's mean image is ~0.
    for subject in fs.subject_ids:
        np.testing.assert_allclose(images[fs.subjects == subject].mean(axis=0), 0.0, atol=1e-4)


def test_build_images_with_given_projector(small_featureset):
    fs = small_featureset.subset(slice(0, 5))
    p = TopomapProjector([c.upper() for c in fs.ch_names], grid_size=16, method="linear")
    images = build_images(fs, normalization="none", projector=p)
    np.testing.assert_allclose(images, build_images(fs, normalization="none", grid_size=16, method="linear"))
    with pytest.raises(ValueError, match="do not match"):
        build_images(fs.pick_channels(fs.ch_names[::-1]), projector=p)


def test_images_from_config(small_featureset, tiny_config):
    cfg = apply_overrides(
        tiny_config, ["features.grid_size=16", "features.interpolation=linear", "protocol.normalization=none"]
    )
    fs = small_featureset.subset(slice(0, 4))
    images = images_from_config(fs, cfg)
    assert images.shape == (4, 16, 16, 6)
    np.testing.assert_array_equal(images, build_images(fs, normalization="none", grid_size=16, method="linear"))
