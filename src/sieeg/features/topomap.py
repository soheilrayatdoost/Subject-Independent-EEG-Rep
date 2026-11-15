"""Spectral topographic maps: per-channel band power -> 32x32 images (paper Sec. 5.1).

The paper converts "the PSD of EEG signals ... to 32x32 images, or spectral topographic
maps, by interpolating the PSD with respect to channel locations for each frequency band".
We follow EEGLearn (Bashivan et al., ICLR 2016; docs/DESIGN.md D6):

* electrodes come from an MNE standard montage, centred on a sphere fitted to *all* of
  its points, and are flattened by EEGLearn's azimuthal-equidistant ``azim_proj``;
* the image samples the projected electrodes' bounding box (x and y scaled separately);
* values are interpolated with Clough-Tocher (``"cubic"``) or barycentric (``"linear"``)
  interpolation on one Delaunay triangulation shared by every map; pixels outside the
  electrodes' convex hull get ``fill_value``.

Unlike EEGLearn's transposed ``mgrid`` layout, images are a top view: row 0 is anterior
and columns run from left to right.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay

from ..config import ExperimentConfig
from .featureset import FeatureSet

_INTERPOLATORS = {"cubic": CloughTocher2DInterpolator, "linear": LinearNDInterpolator}
_CHUNK = 2048  # maps interpolated per scipy call; bounds the float64 work arrays


@lru_cache(maxsize=None)
def _montage_points(montage: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Channel names and read-only ``(n, 3)`` positions (metres) of an MNE standard montage."""
    import mne

    ch_pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    positions = np.array(list(ch_pos.values()), dtype=np.float64)
    positions.setflags(write=False)
    return tuple(ch_pos), positions


def montage_positions(ch_names: Sequence[str], montage: str = "standard_1005") -> np.ndarray:
    """``(n, 3)`` positions in metres of ``ch_names`` in an MNE standard montage.

    MNE frame: +x right, +y anterior (nose), +z up. Matching is case-insensitive.

    Raises:
        KeyError: listing the channels that the montage lacks.
    """
    names, positions = _montage_points(montage)
    lookup = {name.lower(): i for i, name in enumerate(names)}
    missing = [c for c in ch_names if c.lower() not in lookup]
    if missing:
        raise KeyError(f"Channels not in montage {montage!r}: {missing}")
    return positions[[lookup[c.lower()] for c in ch_names]]


def montage_center(montage: str = "standard_1005") -> np.ndarray:
    """``(3,)`` centre in metres of the least-squares sphere through all points of ``montage``.

    Fitting every montage point keeps the projection independent of the channel subset.
    Algebraic fit: ``|p|^2 = 2 c.p + (r^2 - |c|^2)`` is linear in ``c`` and ``r^2 - |c|^2``.
    """
    _, positions = _montage_points(montage)
    design = np.column_stack([2.0 * positions, np.ones(len(positions))])
    solution, *_ = np.linalg.lstsq(design, np.sum(positions**2, axis=1), rcond=None)
    return solution[:3]


def azimuthal_equidistant(pos3d: np.ndarray, center: np.ndarray) -> np.ndarray:
    """EEGLearn ``azim_proj`` of ``(n, 3)`` positions about ``center`` -> ``(n, 2)``.

    With ``(x, y, z) = pos3d - center``, colatitude ``theta = arccos(z / r)`` and azimuth
    ``phi = atan2(y, x)``: ``(X, Y) = theta * (cos phi, sin phi)``. The distance from the
    vertex (+z) in the plane equals the angle in radians; X follows x (right), Y follows
    y (anterior).
    """
    x, y, z = (np.asarray(pos3d, dtype=np.float64) - np.asarray(center, dtype=np.float64)).T
    theta = np.arctan2(np.hypot(x, y), z)  # arccos(z / r), without rounding trouble near the poles
    phi = np.arctan2(y, x)
    return np.column_stack([theta * np.cos(phi), theta * np.sin(phi)])


class TopomapProjector:
    """Interpolates per-channel values onto a ``grid_size`` x ``grid_size`` image (paper Sec. 5.1).

    Attributes:
        ch_names: channels, in the order of the last axis of :meth:`transform` inputs.
        positions_2d: ``(n_channels, 2)`` projected electrodes ``(X, Y)``.
        xs: ``(G,)`` x of the image columns, ``linspace(min X, max X, G)`` (left -> right).
        ys: ``(G,)`` y of the image rows, ``linspace(max Y, min Y, G)`` (row 0 = anterior).
        mask: ``(G, G)`` bool, True inside the electrodes' convex hull.
    """

    def __init__(
        self,
        ch_names: Sequence[str],
        *,
        grid_size: int = 32,
        method: str = "cubic",
        montage: str = "standard_1005",
        fill_value: float = 0.0,
    ) -> None:
        if method not in _INTERPOLATORS:
            raise ValueError(f"Unknown interpolation {method!r}; use one of {sorted(_INTERPOLATORS)}")
        self.ch_names = [str(c) for c in ch_names]
        self.grid_size = int(grid_size)
        self.method = method
        self.montage = montage
        self.fill_value = float(fill_value)
        self.positions_2d = azimuthal_equidistant(montage_positions(self.ch_names, montage), montage_center(montage))
        x, y = self.positions_2d.T
        self.xs = np.linspace(x.min(), x.max(), self.grid_size)
        self.ys = np.linspace(y.max(), y.min(), self.grid_size)
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)  # "xy" indexing: [row, col] -> (xs[col], ys[row])
        self._points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        self._tri = Delaunay(self.positions_2d)
        # The interpolator itself decides which pixels are inside the hull, so the mask
        # matches the NaN pattern of every later call exactly.
        inside = np.isfinite(self._interpolate(np.ones((len(self.ch_names), 1))))
        self.mask = inside[:, 0].reshape(self.grid_size, self.grid_size)

    def _interpolate(self, columns: np.ndarray) -> np.ndarray:
        """``(n_channels, k)`` values -> ``(G * G, k)`` float64 maps, NaN outside the hull."""
        return _INTERPOLATORS[self.method](self._tri, columns)(self._points)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """``(..., n_channels)`` values -> ``(..., G, G)`` float32 images; NaN becomes ``fill_value``.

        All maps share the triangulation and are interpolated together, ``_CHUNK`` at a time.
        """
        values = np.asarray(values, dtype=np.float64)
        n_channels = len(self.ch_names)
        if values.ndim == 0 or values.shape[-1] != n_channels:
            raise ValueError(f"Expected values with {n_channels} channels on the last axis, got shape {values.shape}")
        columns = values.reshape(-1, n_channels).T
        images = np.empty((columns.shape[1], self.grid_size * self.grid_size), dtype=np.float32)
        for start in range(0, columns.shape[1], _CHUNK):
            images[start : start + _CHUNK] = self._interpolate(columns[:, start : start + _CHUNK]).T
        images[np.isnan(images)] = self.fill_value
        return images.reshape(values.shape[:-1] + (self.grid_size, self.grid_size))


def build_images(
    fs: FeatureSet,
    *,
    normalization: str = "subject",
    grid_size: int = 32,
    method: str = "cubic",
    montage: str = "standard_1005",
    projector: TopomapProjector | None = None,
) -> np.ndarray:
    """``(n_trials, G, G, n_bands)`` float32 images (NHWC), one image channel per band (Sec. 5.1).

    The band power is normalised *before* interpolation (``"subject"``: per-participant
    z-score of every (channel, band) feature, paper Sec. 5.3), so the default fill value 0
    outside the hull equals each feature's subject mean. A given ``projector`` replaces
    ``grid_size``, ``method`` and ``montage``; its channels must match ``fs.ch_names``.
    """
    features = fs.normalized(normalization)  # (n, n_channels, n_bands)
    if projector is None:
        projector = TopomapProjector(fs.ch_names, grid_size=grid_size, method=method, montage=montage)
    elif [c.lower() for c in projector.ch_names] != [c.lower() for c in fs.ch_names]:
        raise ValueError(f"Projector channels {projector.ch_names} do not match the feature channels {fs.ch_names}")
    images = projector.transform(np.moveaxis(features, 1, -1))  # (n, n_bands, G, G)
    return np.ascontiguousarray(np.moveaxis(images, 1, -1), dtype=np.float32)


def images_from_config(fs: FeatureSet, cfg: ExperimentConfig) -> np.ndarray:
    """:func:`build_images` with ``cfg.features`` and ``cfg.protocol.normalization``."""
    return build_images(
        fs,
        normalization=cfg.protocol.normalization,
        grid_size=cfg.features.grid_size,
        method=cfg.features.interpolation,
        montage=cfg.features.montage,
    )
