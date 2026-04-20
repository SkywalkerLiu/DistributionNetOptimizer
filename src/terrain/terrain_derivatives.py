from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter


def derive_terrain_layers(
    dtm: np.ndarray,
    *,
    resolution_m: float,
    terrain_config: dict[str, Any],
    forbidden_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute terrain derivatives required by the planning workflow."""

    slope = compute_slope(dtm, resolution_m=resolution_m)
    aspect = compute_aspect(dtm, resolution_m=resolution_m)
    roughness = compute_roughness(
        dtm,
        window_size=int(terrain_config.get("roughness_window", 5)),
    )
    buildable = build_buildable_mask(
        slope=slope,
        roughness=roughness,
        max_slope_deg=float(terrain_config.get("max_buildable_slope_deg", 24.0)),
        max_roughness_m=float(terrain_config.get("max_buildable_roughness_m", 10.0)),
        forbidden_mask=forbidden_mask,
    )
    return {
        "slope": slope,
        "aspect": aspect,
        "roughness": roughness,
        "buildable_mask": buildable,
    }


def compute_slope(dtm: np.ndarray, *, resolution_m: float) -> np.ndarray:
    """Compute slope in degrees from a DTM using first derivatives."""

    grad_y, grad_x = np.gradient(dtm.astype(np.float32), resolution_m, resolution_m)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    return np.degrees(slope_rad).astype(np.float32)


def compute_aspect(dtm: np.ndarray, *, resolution_m: float) -> np.ndarray:
    """Compute aspect in degrees within the range [0, 360)."""

    grad_y, grad_x = np.gradient(dtm.astype(np.float32), resolution_m, resolution_m)
    aspect = np.degrees(np.arctan2(-grad_y, grad_x))
    aspect = (90.0 - aspect) % 360.0
    return aspect.astype(np.float32)


def compute_roughness(dtm: np.ndarray, *, window_size: int = 5) -> np.ndarray:
    """Estimate local roughness as the moving-window elevation standard deviation."""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("roughness window_size must be a positive odd integer")

    surface = dtm.astype(np.float32)
    mean = uniform_filter(surface, size=window_size, mode="nearest")
    mean_sq = uniform_filter(surface**2, size=window_size, mode="nearest")
    variance = np.clip(mean_sq - mean**2, 0.0, None)
    return np.sqrt(variance).astype(np.float32)


def build_buildable_mask(
    *,
    slope: np.ndarray,
    roughness: np.ndarray,
    max_slope_deg: float,
    max_roughness_m: float,
    forbidden_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a binary buildable mask from terrain thresholds and constraints."""

    buildable = (slope <= max_slope_deg) & (roughness <= max_roughness_m)
    if forbidden_mask is not None:
        buildable &= forbidden_mask == 0
    return buildable.astype(np.uint8)

