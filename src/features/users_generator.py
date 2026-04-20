from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Point


PHASE_TYPES = ("A", "B", "C", "ABC")


def generate_users(
    config: dict[str, Any],
    *,
    dtm: np.ndarray,
    valid_mask: np.ndarray,
    transform: Affine,
    crs: str,
) -> gpd.GeoDataFrame:
    """Generate user points while respecting spacing and validity constraints."""

    users_cfg = config["users"]
    scene_cfg = config["scene"]
    count = int(users_cfg["count"])
    min_spacing_m = float(users_cfg["min_spacing_m"])
    distribution_mode = str(users_cfg.get("distribution_mode", "uniform"))
    cluster_count = max(1, int(users_cfg.get("cluster_count", 3)))
    cluster_radius_m = float(users_cfg.get("cluster_radius_m", 200.0))
    resolution_m = float(scene_cfg["resolution_m"])
    rng = np.random.default_rng(int(scene_cfg["seed"]) + 101)

    valid_indices = np.argwhere(valid_mask.astype(bool))
    if len(valid_indices) == 0:
        raise ValueError("No valid cells available for user generation.")

    cluster_centers: list[tuple[int, int]] = []
    if distribution_mode == "clustered":
        chosen = rng.choice(
            len(valid_indices),
            size=min(cluster_count, len(valid_indices)),
            replace=False,
        )
        cluster_centers = [
            tuple(map(int, valid_indices[index])) for index in np.atleast_1d(chosen)
        ]

    rows: list[int] = []
    cols: list[int] = []
    geometries: list[Point] = []
    protected_coords: list[tuple[float, float]] = []

    attempts = 0
    max_attempts = count * 1000
    while len(geometries) < count and attempts < max_attempts:
        attempts += 1
        if distribution_mode == "clustered" and cluster_centers:
            center_row, center_col = cluster_centers[attempts % len(cluster_centers)]
            row = int(round(rng.normal(center_row, cluster_radius_m / resolution_m)))
            col = int(round(rng.normal(center_col, cluster_radius_m / resolution_m)))
            if not (0 <= row < valid_mask.shape[0] and 0 <= col < valid_mask.shape[1]):
                continue
            if not bool(valid_mask[row, col]):
                continue
        else:
            row, col = map(int, valid_indices[rng.integers(0, len(valid_indices))])

        x, y = _cell_center(transform, row=row, col=col)
        if any(np.hypot(x - px, y - py) < min_spacing_m for px, py in protected_coords):
            continue

        rows.append(row)
        cols.append(col)
        geometries.append(Point(x, y))
        protected_coords.append((x, y))

    if len(geometries) != count:
        raise ValueError(
            f"Unable to generate {count} users after {attempts} attempts. "
            "Relax spacing or expand the valid mask."
        )

    load_low, load_high = map(float, users_cfg["load_kw_range"])
    importance_low, importance_high = map(int, users_cfg["importance_range"])
    data = {
        "user_id": np.arange(1, count + 1, dtype=np.int64),
        "load_kw": rng.uniform(load_low, load_high, size=count).round(3),
        "phase_type": rng.choice(PHASE_TYPES, size=count, replace=True),
        "importance": rng.integers(
            importance_low,
            importance_high + 1,
            size=count,
            endpoint=False,
        ),
        "elev_m": np.array(
            [float(dtm[row, col]) for row, col in zip(rows, cols)],
            dtype=np.float64,
        ),
    }
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)


def _cell_center(transform: Affine, *, row: int, col: int) -> tuple[float, float]:
    """Convert raster row and column indices into cell-center coordinates."""

    x = float(transform.c + (col + 0.5) * transform.a)
    y = float(transform.f + (row + 0.5) * transform.e)
    return x, y

