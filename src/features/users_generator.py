from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Point


PHASE_TYPES = ("single", "three_phase", "A", "B", "C", "ABC")


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
    count = _resolve_user_count(users_cfg)
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

    load_kw, power_factor, phase_type = _build_load_profile(
        users_cfg=users_cfg,
        count=count,
        rng=rng,
    )
    importance_low, importance_high = map(int, users_cfg["importance_range"])
    data = {
        "user_id": np.arange(1, count + 1, dtype=np.int64),
        "load_kw": load_kw,
        "power_factor": power_factor,
        "phase_type": phase_type,
        "assigned_phase": np.full(count, "", dtype=object),
        "apparent_kva": np.round(load_kw / np.maximum(power_factor, 0.001), 3),
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


def _resolve_user_count(users_cfg: dict[str, Any]) -> int:
    """Resolve the number of users, preferring explicit load groups when present."""

    load_groups = users_cfg.get("load_groups")
    if load_groups:
        return int(sum(int(group["count"]) for group in load_groups))
    return int(users_cfg["count"])


def _build_load_profile(
    *,
    users_cfg: dict[str, Any],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-user load, power-factor, and phase-type arrays."""

    load_groups = users_cfg.get("load_groups")
    if load_groups:
        loads: list[float] = []
        power_factors: list[float] = []
        phase_types: list[str] = []
        for group in load_groups:
            group_count = int(group["count"])
            loads.extend([float(group["load_kw"])] * group_count)
            power_factors.extend([float(group.get("power_factor", 0.85))] * group_count)
            phase_types.extend([str(group.get("phase_type", "single"))] * group_count)

        if len(loads) != count:
            raise ValueError("load_groups count does not match resolved user count.")

        order = rng.permutation(count)
        return (
            np.asarray(loads, dtype=np.float64)[order],
            np.asarray(power_factors, dtype=np.float64)[order],
            np.asarray(phase_types, dtype=object)[order],
        )

    load_low, load_high = map(float, users_cfg.get("load_kw_range", [12.0, 12.0]))
    pf_low, pf_high = map(float, users_cfg.get("power_factor_range", [0.85, 0.85]))
    default_phase_type = str(users_cfg.get("default_phase_type", "single"))

    load_kw = rng.uniform(load_low, load_high, size=count).round(3)
    power_factor = rng.uniform(pf_low, pf_high, size=count).round(3)
    phase_distribution = users_cfg.get("phase_type_distribution")
    if phase_distribution:
        phase_names = np.asarray(list(phase_distribution.keys()), dtype=object)
        weights = np.asarray(list(phase_distribution.values()), dtype=np.float64)
        weights = weights / weights.sum()
        phase_type = rng.choice(phase_names, size=count, replace=True, p=weights)
    else:
        phase_type = np.full(count, default_phase_type, dtype=object)

    return load_kw, power_factor, phase_type


def _cell_center(transform: Affine, *, row: int, col: int) -> tuple[float, float]:
    """Convert raster row and column indices into cell-center coordinates."""

    x = float(transform.c + (col + 0.5) * transform.a)
    y = float(transform.f + (row + 0.5) * transform.e)
    return x, y
