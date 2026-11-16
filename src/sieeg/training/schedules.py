"""Schedules for the gradient-reversal coefficient lambda (decision D11).

The paper reports fixed lambdas (Sec. 6), so the default is ``"constant"``.
``"dann"`` is the ramp of Ganin et al. (2016): ``2 / (1 + exp(-10 p)) - 1``,
which suppresses the noisy adversarial signal early in training.
"""

from __future__ import annotations

import math


def lambda_factor(progress: float, schedule: str = "constant") -> float:
    """Multiplier of an adversary's lambda at training progress ``p = step / total_steps``."""
    if schedule == "constant":
        return 1.0
    if schedule == "dann":
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    raise ValueError(f"Unknown lambda schedule {schedule!r}; use 'constant' or 'dann'")
