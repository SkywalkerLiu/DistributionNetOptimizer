from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize


def build_cost_surface(
    *,
    slope: np.ndarray,
    roughness: np.ndarray,
    forbidden_mask: np.ndarray,
    forest: gpd.GeoDataFrame,
    profile: dict[str, Any],
    planning_config: dict[str, Any],
) -> np.ndarray:
    """Combine terrain and obstacle penalties into a base cost raster."""

    slope_weight = float(planning_config.get("slope_weight", 2.5))
    roughness_weight = float(planning_config.get("roughness_weight", 1.5))
    forbidden_cost = float(planning_config.get("forbidden_cost", 1_000_000.0))

    slope_norm = slope / max(float(np.percentile(slope, 99)), 1.0)
    roughness_norm = roughness / max(float(np.percentile(roughness, 99)), 1.0)
    cost = 1.0 + slope_weight * slope_norm + roughness_weight * roughness_norm

    if not forest.empty:
        forest_shapes = [
            (geom, float(pass_cost))
            for geom, pass_cost in zip(forest.geometry, forest["pass_cost"])
            if geom and not geom.is_empty
        ]
        if forest_shapes:
            forest_cost = rasterize(
                forest_shapes,
                out_shape=(profile["height"], profile["width"]),
                transform=profile["transform"],
                fill=0.0,
                dtype="float32",
            )
            cost += forest_cost

    cost = cost.astype(np.float32)
    cost[forbidden_mask > 0] = forbidden_cost
    return cost

