from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon


EXPECTED_WIDTH_M = 428.45
EXPECTED_HEIGHT_M = 909.30

ELEVATION_CONTROL_POINTS = [
    (0, 0, 350),
    (0, 200, 380),
    (0, 450, 400),
    (0, 700, 360),
    (0, 909, 310),
    (60, 120, 340),
    (70, 350, 370),
    (80, 600, 350),
    (90, 850, 320),
    (150, 80, 310),
    (170, 250, 315),
    (180, 450, 320),
    (190, 650, 305),
    (210, 850, 300),
    (240, 120, 300),
    (250, 350, 290),
    (250, 550, 285),
    (250, 750, 285),
    (330, 100, 285),
    (330, 300, 270),
    (340, 500, 260),
    (360, 700, 250),
    (380, 850, 250),
    (428, 0, 300),
    (428, 250, 260),
    (428, 550, 250),
    (428, 909, 270),
    (330, 760, 248),
    (370, 775, 248),
    (385, 690, 250),
]

ROAD_CENTERLINES_M = [
    [(125, 900), (150, 760), (170, 630), (210, 520), (245, 380), (285, 220), (300, 20)],
    [(35, 520), (70, 560), (120, 635), (165, 660)],
    [(250, 560), (310, 610), (380, 650), (410, 690)],
    [(180, 210), (210, 160), (245, 120), (300, 50)],
]

WATER_POLYGONS_M = [
    [(318, 790), (365, 790), (370, 750), (325, 742)],
    [(340, 705), (380, 705), (385, 670), (350, 660)],
]


def generate_ovi_screenshot_terrain(config: dict[str, Any]) -> np.ndarray:
    """Generate the deterministic Ovi-screenshot terrain surface."""

    scene_cfg = config["scene"]
    terrain_cfg = config["terrain"]
    _validate_ovi_config(config)

    resolution = float(scene_cfg["resolution_m"])
    width = int(round(float(scene_cfg["width_m"]) / resolution))
    height = int(round(float(scene_cfg["height_m"]) / resolution))
    origin_x = float(scene_cfg.get("origin_x_m", 0.0))
    origin_y = float(scene_cfg.get("origin_y_m", scene_cfg["height_m"]))

    x_coords = origin_x + (np.arange(width, dtype=np.float32) + 0.5) * resolution
    y_coords = origin_y - (np.arange(height, dtype=np.float32) + 0.5) * resolution
    xx, yy = np.meshgrid(x_coords, y_coords)

    dtm = _interpolate_control_surface(xx=xx, yy=yy)
    dtm += _road_grade_effect(xx=xx, yy=yy)
    dtm += 1.2 * np.sin(xx / 35.0) * np.sin(yy / 47.0)

    water_mask = _rasterize_polygons(
        polygons=WATER_POLYGONS_M,
        out_shape=(height, width),
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
    )
    dtm[water_mask] = np.minimum(dtm[water_mask], 252.0)

    smooth_sigma = float(terrain_cfg.get("smooth_sigma", 4.0))
    if smooth_sigma > 0:
        dtm = gaussian_filter(dtm, sigma=smooth_sigma)
    dtm[water_mask] = np.minimum(dtm[water_mask], 252.0)

    return np.clip(
        dtm,
        float(terrain_cfg["clip_min"]),
        float(terrain_cfg["clip_max"]),
    ).astype(np.float32)


def _interpolate_control_surface(*, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    control = np.asarray(ELEVATION_CONTROL_POINTS, dtype=np.float64)
    rbf = RBFInterpolator(
        control[:, :2],
        control[:, 2],
        kernel="thin_plate_spline",
        smoothing=1.5,
    )

    target = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.empty(len(target), dtype=np.float64)
    chunk_size = 50000
    for start in range(0, len(target), chunk_size):
        end = min(start + chunk_size, len(target))
        values[start:end] = rbf(target[start:end])
    return values.reshape(xx.shape).astype(np.float32)


def _road_grade_effect(*, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    distance = _distance_to_polylines(xx=xx, yy=yy, polylines=ROAD_CENTERLINES_M)
    return (-4.0 * np.exp(-((distance / 12.0) ** 2))).astype(np.float32)


def _distance_to_polylines(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    polylines: list[list[tuple[float, float]]],
) -> np.ndarray:
    x_flat = xx.ravel().astype(np.float64)
    y_flat = yy.ravel().astype(np.float64)
    min_distance = np.full_like(x_flat, np.inf, dtype=np.float64)

    for polyline in polylines:
        for start, end in zip(polyline[:-1], polyline[1:]):
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq <= 0:
                continue
            t = ((x_flat - x1) * dx + (y_flat - y1) * dy) / length_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            distance = np.hypot(x_flat - proj_x, y_flat - proj_y)
            min_distance = np.minimum(min_distance, distance)

    return min_distance.reshape(xx.shape).astype(np.float32)


def _rasterize_polygons(
    *,
    polygons: list[list[tuple[float, float]]],
    out_shape: tuple[int, int],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> np.ndarray:
    transform = from_origin(origin_x, origin_y, resolution, resolution)
    shapes = [(Polygon(points), 1) for points in polygons]
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _validate_ovi_config(config: dict[str, Any]) -> None:
    terrain_cfg = config.get("terrain", {})
    if terrain_cfg.get("base_type") != "ovi_screenshot":
        raise ValueError("terrain.base_type must be 'ovi_screenshot'.")

    scene_cfg = config.get("scene", {})
    resolution = float(scene_cfg.get("resolution_m", 0.0))
    if resolution != 1.0:
        raise ValueError("ovi_screenshot terrain requires scene.resolution_m = 1.0.")
    if not np.isclose(float(scene_cfg.get("width_m", -1.0)), EXPECTED_WIDTH_M):
        raise ValueError("ovi_screenshot terrain requires scene.width_m = 428.45.")
    if not np.isclose(float(scene_cfg.get("height_m", -1.0)), EXPECTED_HEIGHT_M):
        raise ValueError("ovi_screenshot terrain requires scene.height_m = 909.30.")

    ovi_cfg = config.get("ovi_screenshot")
    if not isinstance(ovi_cfg, dict):
        raise KeyError("ovi_screenshot config block is required.")
    roi_size = ovi_cfg.get("roi_size_m", {})
    if not np.isclose(float(roi_size.get("width", -1.0)), EXPECTED_WIDTH_M):
        raise ValueError("ovi_screenshot.roi_size_m.width must be 428.45.")
    if not np.isclose(float(roi_size.get("height", -1.0)), EXPECTED_HEIGHT_M):
        raise ValueError("ovi_screenshot.roi_size_m.height must be 909.30.")
