"""Lambda schedules (decision D11)."""

from __future__ import annotations

import numpy as np
import pytest

from sieeg.training.schedules import lambda_factor


@pytest.mark.parametrize("progress", [0.0, 0.3, 1.0])
def test_constant_is_one(progress):
    assert lambda_factor(progress) == 1.0
    assert lambda_factor(progress, "constant") == 1.0


def test_dann_ramp_values():
    assert lambda_factor(0.0, "dann") == 0.0
    assert lambda_factor(0.5, "dann") == pytest.approx(2 / (1 + np.exp(-5)) - 1)
    assert lambda_factor(0.5, "dann") == pytest.approx(0.98661, abs=1e-5)
    assert lambda_factor(1.0, "dann") == pytest.approx(0.99991, abs=1e-5)
    ramp = [lambda_factor(p, "dann") for p in np.linspace(0, 1, 11)]
    assert np.all(np.diff(ramp) > 0) and 0.0 <= min(ramp) and max(ramp) < 1.0


def test_unknown_schedule():
    with pytest.raises(ValueError, match="linear"):
        lambda_factor(0.5, "linear")
