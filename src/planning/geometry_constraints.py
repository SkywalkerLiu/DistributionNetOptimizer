from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.planning.common import sample_array, xy_to_cell


def segment_is_feasible(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    allowed_mask: np.ndarray,
    profile: dict[str, Any],
    sample_step_m: float,
) -> bool:
    """Return true when every sampled point of a segment stays inside the allowed mask."""

    distance = math.hypot(x2 - x1, y2 - y1)
    sample_count = max(2, int(math.ceil(distance / max(sample_step_m, 0.5))) + 1)
    for fraction in np.linspace(0.0, 1.0, sample_count):
        x = x1 + (x2 - x1) * float(fraction)
        y = y1 + (y2 - y1) * float(fraction)
        row, col = xy_to_cell(profile, x, y, shape=allowed_mask.shape)
        if allowed_mask[row, col] <= 0:
            return False
    return True


def segment_min_clearance(
    *,
    start: dict[str, Any],
    end: dict[str, Any],
    dtm: np.ndarray,
    profile: dict[str, Any],
    sample_step_m: float,
) -> float:
    """Return the minimum terrain clearance of a straight line segment."""

    distance = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    sample_count = max(2, int(math.ceil(distance / max(sample_step_m, 0.5))) + 1)
    min_clearance = float("inf")
    z_start = float(start["support_top_z"])
    z_end = float(end["support_top_z"])

    for fraction in np.linspace(0.0, 1.0, sample_count):
        x = float(start["x"]) + (float(end["x"]) - float(start["x"])) * float(fraction)
        y = float(start["y"]) + (float(end["y"]) - float(start["y"])) * float(fraction)
        terrain_z = sample_array(dtm, profile, x, y)
        line_z = z_start + (z_end - z_start) * float(fraction)
        min_clearance = min(min_clearance, float(line_z - terrain_z))
    return float(min_clearance)

