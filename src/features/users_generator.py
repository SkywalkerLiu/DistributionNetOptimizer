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
    distribution_mode = str(users_cfg.get("distribution_mode", "uniform"))
    rng = np.random.default_rng(int(scene_cfg["seed"]) + 101)

    valid_indices = np.argwhere(valid_mask.astype(bool))
    if len(valid_indices) == 0:
        raise ValueError("No valid cells available for user generation.")

    if distribution_mode == "clustered_with_scattered":
        rows, cols, geometries, settlement_types, cluster_ids = _generate_clustered_with_scattered_points(
            users_cfg=users_cfg,
            count=count,
            valid_mask=valid_mask.astype(bool),
            valid_indices=valid_indices,
            transform=transform,
            rng=rng,
        )
    else:
        rows, cols, geometries = _generate_legacy_points(
            users_cfg=users_cfg,
            scene_cfg=scene_cfg,
            count=count,
            valid_mask=valid_mask.astype(bool),
            valid_indices=valid_indices,
            transform=transform,
            distribution_mode=distribution_mode,
            rng=rng,
        )
        settlement_types = [distribution_mode] * count
        cluster_ids = [None] * count

    load_kw, power_factor, phase_type, user_type = _build_load_profile(
        users_cfg=users_cfg,
        count=count,
        rng=rng,
    )
    importance_low, importance_high = map(int, users_cfg["importance_range"])
    data = {
        "user_id": np.arange(1, count + 1, dtype=np.int64),
        "user_type": user_type,
        "settlement_type": np.asarray(settlement_types, dtype=object),
        "cluster_id": np.asarray(cluster_ids, dtype=object),
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
    """Resolve and validate the configured number of users."""

    explicit_count = users_cfg.get("count")
    load_groups = users_cfg.get("load_groups")
    if explicit_count is None:
        if load_groups:
            return int(sum(int(group["count"]) for group in load_groups))
        raise KeyError("users.count is required when load_groups is not configured.")

    count = int(explicit_count)
    if load_groups:
        grouped_count = int(sum(int(group["count"]) for group in load_groups))
        if grouped_count != count:
            raise ValueError(
                f"users.count ({count}) must equal sum(load_groups.count) ({grouped_count})."
            )
    return count


def _generate_clustered_with_scattered_points(
    *,
    users_cfg: dict[str, Any],
    count: int,
    valid_mask: np.ndarray,
    valid_indices: np.ndarray,
    transform: Affine,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[Point], list[str], list[str | None]]:
    """Generate clustered rural residents plus isolated scattered households."""

    clustered_count = int(users_cfg["clustered_count"])
    scattered_count = int(users_cfg["scattered_count"])
    if clustered_count + scattered_count != count:
        raise ValueError(
            "users.clustered_count + users.scattered_count must equal users.count."
        )

    cluster_size_min = int(users_cfg.get("cluster_size_min", 3))
    cluster_size_max = int(users_cfg.get("cluster_size_max", 8))
    cluster_diameter_m = float(users_cfg.get("cluster_diameter_m", 10.0))
    cluster_radius_m = cluster_diameter_m / 2.0
    center_spacing_m = float(users_cfg.get("cluster_center_min_spacing_m", 40.0))

    cluster_sizes = _build_cluster_sizes(
        total=clustered_count,
        size_min=cluster_size_min,
        size_max=cluster_size_max,
        rng=rng,
    )

    rows: list[int] = []
    cols: list[int] = []
    geometries: list[Point] = []
    settlement_types: list[str] = []
    cluster_ids: list[str | None] = []
    cluster_centers: list[tuple[float, float]] = []

    for cluster_index, cluster_size in enumerate(cluster_sizes, start=1):
        cluster_id = f"C{cluster_index:03d}"
        center_xy, cluster_points = _place_cluster_points(
            cluster_size=cluster_size,
            cluster_id=cluster_id,
            cluster_radius_m=cluster_radius_m,
            center_spacing_m=center_spacing_m,
            existing_centers=cluster_centers,
            valid_mask=valid_mask,
            valid_indices=valid_indices,
            transform=transform,
            rng=rng,
        )
        cluster_centers.append(center_xy)
        for row, col, point in cluster_points:
            rows.append(row)
            cols.append(col)
            geometries.append(point)
            settlement_types.append("clustered")
            cluster_ids.append(cluster_id)

    scattered_points = _place_scattered_points(
        scattered_count=scattered_count,
        cluster_centers=cluster_centers,
        min_distance_from_cluster_m=float(
            users_cfg.get("scattered_min_distance_from_cluster_m", 30.0)
        ),
        scattered_min_spacing_m=float(users_cfg.get("scattered_min_spacing_m", 30.0)),
        valid_indices=valid_indices,
        transform=transform,
        rng=rng,
    )
    for row, col, point in scattered_points:
        rows.append(row)
        cols.append(col)
        geometries.append(point)
        settlement_types.append("scattered")
        cluster_ids.append(None)

    return rows, cols, geometries, settlement_types, cluster_ids


def _build_cluster_sizes(
    *,
    total: int,
    size_min: int,
    size_max: int,
    rng: np.random.Generator,
) -> list[int]:
    """Split clustered users into valid household-cluster sizes."""

    if total == 0:
        return []
    if size_min <= 0 or size_max < size_min:
        raise ValueError("cluster_size_min and cluster_size_max must define a positive range.")
    if total < size_min:
        raise ValueError("clustered_count must be at least cluster_size_min.")

    remaining = total
    sizes: list[int] = []
    while remaining > 0:
        if remaining <= size_max:
            if remaining < size_min:
                for index in range(len(sizes)):
                    if sizes[index] + remaining <= size_max:
                        sizes[index] += remaining
                        remaining = 0
                        break
                if remaining:
                    raise ValueError("Unable to split clustered_count into valid cluster sizes.")
            else:
                sizes.append(remaining)
                remaining = 0
            break

        max_size = min(size_max, remaining - size_min)
        cluster_size = int(rng.integers(size_min, max_size + 1))
        sizes.append(cluster_size)
        remaining -= cluster_size
    return sizes


def _place_cluster_points(
    *,
    cluster_size: int,
    cluster_id: str,
    cluster_radius_m: float,
    center_spacing_m: float,
    existing_centers: list[tuple[float, float]],
    valid_mask: np.ndarray,
    valid_indices: np.ndarray,
    transform: Affine,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], list[tuple[int, int, Point]]]:
    """Place one residential cluster around a valid center."""

    max_center_attempts = 5000
    max_point_attempts = max(500, cluster_size * 300)
    for _ in range(max_center_attempts):
        center_row, center_col = map(int, valid_indices[rng.integers(0, len(valid_indices))])
        center_x, center_y = _cell_center(transform, row=center_row, col=center_col)
        if any(
            np.hypot(center_x - other_x, center_y - other_y) < center_spacing_m
            for other_x, other_y in existing_centers
        ):
            continue

        points: list[tuple[int, int, Point]] = []
        point_attempts = 0
        while len(points) < cluster_size and point_attempts < max_point_attempts:
            point_attempts += 1
            x, y = _sample_point_in_circle(
                center_x=center_x,
                center_y=center_y,
                radius_m=cluster_radius_m,
                rng=rng,
            )
            cell = _valid_cell_for_xy(
                x=x,
                y=y,
                transform=transform,
                valid_mask=valid_mask,
            )
            if cell is None:
                continue
            row, col = cell
            points.append((row, col, Point(x, y)))

        if len(points) == cluster_size:
            return (center_x, center_y), points

    raise ValueError(
        f"Unable to place clustered users for {cluster_id} under current terrain/buildable constraints. "
        "Try reducing cluster_diameter_m or cluster_center_min_spacing_m."
    )


def _sample_point_in_circle(
    *,
    center_x: float,
    center_y: float,
    radius_m: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample a point uniformly inside a circle."""

    radius = radius_m * float(np.sqrt(rng.random()))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return center_x + radius * float(np.cos(angle)), center_y + radius * float(np.sin(angle))


def _place_scattered_points(
    *,
    scattered_count: int,
    cluster_centers: list[tuple[float, float]],
    min_distance_from_cluster_m: float,
    scattered_min_spacing_m: float,
    valid_indices: np.ndarray,
    transform: Affine,
    rng: np.random.Generator,
) -> list[tuple[int, int, Point]]:
    """Place isolated households away from clusters and one another."""

    scattered: list[tuple[int, int, Point]] = []
    scattered_coords: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(5000, scattered_count * 5000)
    while len(scattered) < scattered_count and attempts < max_attempts:
        attempts += 1
        row, col = map(int, valid_indices[rng.integers(0, len(valid_indices))])
        x, y = _cell_center(transform, row=row, col=col)
        if any(
            np.hypot(x - center_x, y - center_y) < min_distance_from_cluster_m
            for center_x, center_y in cluster_centers
        ):
            continue
        if any(
            np.hypot(x - other_x, y - other_y) < scattered_min_spacing_m
            for other_x, other_y in scattered_coords
        ):
            continue
        scattered.append((row, col, Point(x, y)))
        scattered_coords.append((x, y))

    if len(scattered) != scattered_count:
        raise ValueError(
            "Unable to place scattered users under current terrain/buildable constraints. "
            "Try reducing scattered_min_spacing_m or scattered_min_distance_from_cluster_m."
        )
    return scattered


def _generate_legacy_points(
    *,
    users_cfg: dict[str, Any],
    scene_cfg: dict[str, Any],
    count: int,
    valid_mask: np.ndarray,
    valid_indices: np.ndarray,
    transform: Affine,
    distribution_mode: str,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[Point]]:
    """Generate users with the original uniform/clustered spacing model."""

    min_spacing_m = float(users_cfg.get("min_spacing_m", 0.0))
    cluster_count = max(1, int(users_cfg.get("cluster_count", 3)))
    cluster_radius_m = float(users_cfg.get("cluster_radius_m", 200.0))
    resolution_m = float(scene_cfg["resolution_m"])

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
    return rows, cols, geometries


def _build_load_profile(
    *,
    users_cfg: dict[str, Any],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-user load, power-factor, phase-type, and user-type arrays."""

    load_groups = users_cfg.get("load_groups")
    if load_groups:
        loads: list[float] = []
        power_factors: list[float] = []
        phase_types: list[str] = []
        user_types: list[str] = []
        for index, group in enumerate(load_groups, start=1):
            group_count = int(group["count"])
            loads.extend([float(group["load_kw"])] * group_count)
            power_factors.extend([float(group.get("power_factor", 0.85))] * group_count)
            phase_types.extend([str(group.get("phase_type", "single"))] * group_count)
            user_types.extend([str(group.get("name", f"load_group_{index}"))] * group_count)

        if len(loads) != count:
            raise ValueError("load_groups count does not match resolved user count.")

        order = rng.permutation(count)
        return (
            np.asarray(loads, dtype=np.float64)[order],
            np.asarray(power_factors, dtype=np.float64)[order],
            np.asarray(phase_types, dtype=object)[order],
            np.asarray(user_types, dtype=object)[order],
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

    user_type = np.full(count, "residential", dtype=object)
    return load_kw, power_factor, phase_type, user_type


def _cell_center(transform: Affine, *, row: int, col: int) -> tuple[float, float]:
    """Convert raster row and column indices into cell-center coordinates."""

    x = float(transform.c + (col + 0.5) * transform.a)
    y = float(transform.f + (row + 0.5) * transform.e)
    return x, y


def _valid_cell_for_xy(
    *,
    x: float,
    y: float,
    transform: Affine,
    valid_mask: np.ndarray,
) -> tuple[int, int] | None:
    """Return the valid raster cell containing an XY point, if one exists."""

    col = int(np.floor((x - transform.c) / transform.a))
    row = int(np.floor((y - transform.f) / transform.e))
    if not (0 <= row < valid_mask.shape[0] and 0 <= col < valid_mask.shape[1]):
        return None
    if not bool(valid_mask[row, col]):
        return None
    return row, col
