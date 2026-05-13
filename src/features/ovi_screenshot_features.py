from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine, rowcol
from shapely.geometry import Point, Polygon

from src.io.vector_io import FEATURE_LAYER_DEFINITIONS


EXPECTED_WIDTH_M = 428.45
EXPECTED_HEIGHT_M = 909.30

OVI_USER_POINTS_M = [
    (127.7, 899.9),
    (222.6, 826.0),
    (241.3, 781.2),
    (396.9, 686.4),
    (158.3, 658.3),
    (174.9, 643.7),
    (388.6, 640.6),
    (118.9, 636.4),
    (237.1, 559.3),
    (35.9, 518.7),
    (253.7, 506.2),
    (50.4, 500.0),
    (17.2, 468.7),
    (314.9, 394.8),
    (349.2, 357.3),
    (288.0, 334.3),
    (187.3, 327.1),
    (199.8, 266.6),
    (301.5, 238.5),
    (289.4, 213.5),
    (184.2, 209.4),
    (197.7, 123.9),
    (242.3, 116.7),
    (177.0, 35.4),
    (298.3, 21.9),
]

HIGH_POWER_USER_IDS = {4, 9, 11, 15, 19}

WATER_POLYGONS_M = [
    [(318, 790), (365, 790), (370, 750), (325, 742)],
    [(340, 705), (380, 705), (385, 670), (350, 660)],
]

FOREST_POLYGONS_M = [
    [(0, 0), (45, 0), (38, 180), (18, 320), (5, 470), (0, 909)],
    [(0, 620), (28, 650), (34, 740), (0, 760)],
    [(270, 535), (320, 570), (330, 650), (265, 690), (250, 620)],
]

MANUAL_NO_BUILD_POLYGONS_M = [
    [(0, 780), (70, 790), (90, 909), (0, 909)],
    [(405, 650), (428, 655), (428, 760), (405, 735)],
]


def generate_ovi_users(
    config: dict[str, Any],
    *,
    dtm: np.ndarray,
    transform: Affine,
    crs: str,
) -> gpd.GeoDataFrame:
    """Generate the fixed 25 Ovi-screenshot user points."""

    _validate_ovi_scene_config(config)
    users_cfg = config["users"]
    expected_count = len(OVI_USER_POINTS_M)
    if users_cfg.get("source") != "ovi_screenshot":
        raise ValueError("users.source must be 'ovi_screenshot'.")
    if int(users_cfg.get("count", -1)) != expected_count:
        raise ValueError(f"users.count must be {expected_count} for ovi_screenshot.")
    _validate_load_groups(users_cfg)

    user_ids = np.arange(1, expected_count + 1, dtype=np.int64)
    load_kw = np.asarray(
        [10.0 if int(user_id) in HIGH_POWER_USER_IDS else 7.0 for user_id in user_ids],
        dtype=np.float64,
    )
    power_factor = np.full(expected_count, 0.85, dtype=np.float64)
    geometries = [Point(x, y) for x, y in OVI_USER_POINTS_M]
    elev_m = np.asarray(
        [_sample_dtm(dtm=dtm, transform=transform, x=x, y=y) for x, y in OVI_USER_POINTS_M],
        dtype=np.float64,
    )

    data = {
        "user_id": user_ids,
        "user_type": np.asarray(
            [
                "high_power_user" if int(user_id) in HIGH_POWER_USER_IDS else "normal_residential"
                for user_id in user_ids
            ],
            dtype=object,
        ),
        "settlement_type": np.full(expected_count, "ovi_village", dtype=object),
        "cluster_id": np.full(expected_count, "", dtype=object),
        "load_kw": load_kw,
        "power_factor": power_factor,
        "phase_type": np.full(expected_count, "single", dtype=object),
        "assigned_phase": np.full(expected_count, "", dtype=object),
        "apparent_kva": np.round(load_kw / power_factor, 3),
        "importance": np.ones(expected_count, dtype=np.int64),
        "elev_m": elev_m,
        "connected_node_id": np.full(expected_count, "", dtype=object),
        "voltage_drop_pct": np.zeros(expected_count, dtype=np.float64),
        "service_group_id": np.full(expected_count, "", dtype=object),
        "service_group_size": np.zeros(expected_count, dtype=np.int64),
        "is_service_singleton": np.zeros(expected_count, dtype=np.int64),
    }
    return _standard_gdf(data=data, geometries=geometries, layer="users", crs=crs)


def generate_ovi_obstacles(
    config: dict[str, Any],
    *,
    crs: str,
) -> dict[str, gpd.GeoDataFrame]:
    """Generate fixed forest, water, and no-build layers for the Ovi scene."""

    _validate_ovi_scene_config(config)
    obstacles_cfg = config["obstacles"]
    if obstacles_cfg.get("source") != "ovi_screenshot":
        raise ValueError("obstacles.source must be 'ovi_screenshot'.")

    return {
        "forest": _standard_gdf(
            data={
                "obs_id": np.arange(1, len(FOREST_POLYGONS_M) + 1, dtype=np.int64),
                "density": np.full(len(FOREST_POLYGONS_M), 0.8, dtype=np.float64),
                "pass_cost": np.full(len(FOREST_POLYGONS_M), 8.0, dtype=np.float64),
                "forbidden": np.ones(len(FOREST_POLYGONS_M), dtype=np.int64),
            },
            geometries=[Polygon(points) for points in FOREST_POLYGONS_M],
            layer="forest",
            crs=crs,
        ),
        "water": _standard_gdf(
            data={
                "obs_id": np.arange(1, len(WATER_POLYGONS_M) + 1, dtype=np.int64),
                "water_type": np.full(len(WATER_POLYGONS_M), "pond", dtype=object),
                "forbidden": np.ones(len(WATER_POLYGONS_M), dtype=np.int64),
            },
            geometries=[Polygon(points) for points in WATER_POLYGONS_M],
            layer="water",
            crs=crs,
        ),
        "manual_no_build": _standard_gdf(
            data={
                "obs_id": np.arange(1, len(MANUAL_NO_BUILD_POLYGONS_M) + 1, dtype=np.int64),
                "source": np.full(len(MANUAL_NO_BUILD_POLYGONS_M), "ovi_screenshot", dtype=object),
                "reason": np.full(
                    len(MANUAL_NO_BUILD_POLYGONS_M),
                    "steep_or_pond_buffer",
                    dtype=object,
                ),
                "forbidden": np.ones(len(MANUAL_NO_BUILD_POLYGONS_M), dtype=np.int64),
            },
            geometries=[Polygon(points) for points in MANUAL_NO_BUILD_POLYGONS_M],
            layer="manual_no_build",
            crs=crs,
        ),
    }


def _sample_dtm(
    *,
    dtm: np.ndarray,
    transform: Affine,
    x: float,
    y: float,
) -> float:
    row, col = rowcol(transform, x, y)
    row = int(row)
    col = int(col)
    if not (0 <= row < dtm.shape[0] and 0 <= col < dtm.shape[1]):
        raise ValueError(f"Ovi user point ({x}, {y}) falls outside the DTM extent.")
    return float(dtm[row, col])


def _standard_gdf(
    *,
    data: dict[str, Any],
    geometries: list,
    layer: str,
    crs: str,
) -> gpd.GeoDataFrame:
    _, schema = FEATURE_LAYER_DEFINITIONS[layer]
    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=crs)
    for column, dtype in schema.items():
        if column not in gdf.columns:
            gdf[column] = _empty_value_for_dtype(dtype, len(gdf))
        gdf[column] = gdf[column].astype(dtype)
    ordered = list(schema) + ["geometry"]
    return gdf[ordered]


def _empty_value_for_dtype(dtype: str, length: int) -> np.ndarray:
    if dtype.startswith("float"):
        return np.zeros(length, dtype=np.float64)
    if dtype.startswith("int"):
        return np.zeros(length, dtype=np.int64)
    return np.full(length, "", dtype=object)


def _validate_ovi_scene_config(config: dict[str, Any]) -> None:
    scene_cfg = config.get("scene", {})
    if not np.isclose(float(scene_cfg.get("width_m", -1.0)), EXPECTED_WIDTH_M):
        raise ValueError("ovi_screenshot features require scene.width_m = 428.45.")
    if not np.isclose(float(scene_cfg.get("height_m", -1.0)), EXPECTED_HEIGHT_M):
        raise ValueError("ovi_screenshot features require scene.height_m = 909.30.")
    if float(scene_cfg.get("resolution_m", 0.0)) != 1.0:
        raise ValueError("ovi_screenshot features require scene.resolution_m = 1.0.")
    if str(scene_cfg.get("crs", "")) != "EPSG:3857":
        raise ValueError("ovi_screenshot features require scene.crs = EPSG:3857.")

    ovi_cfg = config.get("ovi_screenshot")
    if not isinstance(ovi_cfg, dict):
        raise KeyError("ovi_screenshot config block is required.")
    roi_size = ovi_cfg.get("roi_size_m", {})
    if not np.isclose(float(roi_size.get("width", -1.0)), EXPECTED_WIDTH_M):
        raise ValueError("ovi_screenshot.roi_size_m.width must be 428.45.")
    if not np.isclose(float(roi_size.get("height", -1.0)), EXPECTED_HEIGHT_M):
        raise ValueError("ovi_screenshot.roi_size_m.height must be 909.30.")


def _validate_load_groups(users_cfg: dict[str, Any]) -> None:
    load_groups = users_cfg.get("load_groups", [])
    group_counts = {str(group.get("name")): int(group.get("count", 0)) for group in load_groups}
    if group_counts.get("normal_residential") != 20:
        raise ValueError("ovi_screenshot users require 20 normal_residential load-group users.")
    if group_counts.get("high_power_user") != 5:
        raise ValueError("ovi_screenshot users require 5 high_power_user load-group users.")
