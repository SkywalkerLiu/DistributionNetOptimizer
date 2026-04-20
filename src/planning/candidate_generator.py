from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def generate_candidate_layers(
    config: dict[str, Any],
    *,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Generate candidate transformer and pole points on regular grids."""

    crs = profile["crs"]
    valid_mask = (buildable_mask > 0) & (forbidden_mask == 0)

    transformer = _grid_candidates(
        valid_mask=valid_mask,
        profile=profile,
        step_m=float(config["planning"]["transformer_candidate_step_m"]),
        crs=crs,
    )
    poles = _grid_candidates(
        valid_mask=valid_mask,
        profile=profile,
        step_m=float(config["planning"]["pole_candidate_step_m"]),
        crs=crs,
    )

    return transformer, poles


def _grid_candidates(
    *,
    valid_mask: np.ndarray,
    profile: dict[str, Any],
    step_m: float,
    crs: Any,
) -> gpd.GeoDataFrame:
    """Sample valid cells at a fixed spatial step and convert them to points."""

    step_cells = max(1, int(round(step_m / abs(profile["transform"].a))))
    geometries: list[Point] = []
    candidate_ids: list[int] = []

    for row in range(0, valid_mask.shape[0], step_cells):
        for col in range(0, valid_mask.shape[1], step_cells):
            if not bool(valid_mask[row, col]):
                continue
            x = float(profile["transform"].c + (col + 0.5) * profile["transform"].a)
            y = float(profile["transform"].f + (row + 0.5) * profile["transform"].e)
            candidate_ids.append(len(candidate_ids) + 1)
            geometries.append(Point(x, y))

    return gpd.GeoDataFrame(
        {"candidate_id": np.array(candidate_ids, dtype=np.int64)},
        geometry=geometries,
        crs=crs,
    )

