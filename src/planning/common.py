from __future__ import annotations

import math
from typing import Any

import numpy as np


def cell_to_xy(profile: dict[str, Any], row: int, col: int) -> tuple[float, float]:
    """Convert raster row and column to cell-center XY coordinates."""

    transform = profile["transform"]
    return (
        float(transform.c + (col + 0.5) * transform.a),
        float(transform.f + (row + 0.5) * transform.e),
    )


def xy_to_cell(
    profile: dict[str, Any],
    x: float,
    y: float,
    *,
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert XY coordinates to a clipped raster row and column."""

    transform = profile["transform"]
    col = int(math.floor((x - transform.c) / transform.a))
    row = int(math.floor((y - transform.f) / transform.e))
    row = int(np.clip(row, 0, shape[0] - 1))
    col = int(np.clip(col, 0, shape[1] - 1))
    return row, col


def sample_array(array: np.ndarray, profile: dict[str, Any], x: float, y: float) -> float:
    """Sample a raster array with nearest-neighbor lookup."""

    row, col = xy_to_cell(profile, x, y, shape=array.shape)
    return float(array[row, col])


def point_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Return horizontal length, 3D length, elevation delta, and slope."""

    dx = float(b["x"]) - float(a["x"])
    dy = float(b["y"]) - float(a["y"])
    dz = float(b.get("z", b.get("ground_z", 0.0))) - float(a.get("z", a.get("ground_z", 0.0)))
    horizontal = math.hypot(dx, dy)
    length_3d = math.sqrt(horizontal**2 + dz**2)
    slope_deg = math.degrees(math.atan2(abs(dz), horizontal)) if horizontal > 0 else 0.0
    return {
        "horizontal_length_m": float(horizontal),
        "length_3d_m": float(length_3d),
        "dz_m": float(dz),
        "slope_deg": float(slope_deg),
    }


def nearest_passable_cell(
    passable: np.ndarray,
    *,
    row: int,
    col: int,
    search_radius: int,
) -> tuple[int, int] | None:
    """Find the nearest passable cell around a target row and column."""

    row = int(np.clip(row, 0, passable.shape[0] - 1))
    col = int(np.clip(col, 0, passable.shape[1] - 1))
    if bool(passable[row, col]):
        return row, col

    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for radius in range(1, max(1, search_radius) + 1):
        row_min = max(0, row - radius)
        row_max = min(passable.shape[0] - 1, row + radius)
        col_min = max(0, col - radius)
        col_max = min(passable.shape[1] - 1, col + radius)
        for rr in range(row_min, row_max + 1):
            for cc in range(col_min, col_max + 1):
                if not bool(passable[rr, cc]):
                    continue
                distance = math.hypot(rr - row, cc - col)
                if distance < best_distance:
                    best_distance = distance
                    best = (rr, cc)
        if best is not None:
            return best
    return None

