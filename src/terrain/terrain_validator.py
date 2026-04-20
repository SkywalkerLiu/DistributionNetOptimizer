from __future__ import annotations

import numpy as np


def validate_terrain_array(
    dtm: np.ndarray,
    *,
    clip_min: float,
    clip_max: float,
) -> None:
    """Validate terrain dimensions and elevation range."""

    if dtm.ndim != 2:
        raise ValueError("Terrain must be a 2D raster.")
    if not np.isfinite(dtm).all():
        raise ValueError("Terrain contains non-finite values.")
    if float(dtm.min()) < clip_min - 1e-6:
        raise ValueError("Terrain minimum elevation is below clip_min.")
    if float(dtm.max()) > clip_max + 1e-6:
        raise ValueError("Terrain maximum elevation is above clip_max.")


def terrain_statistics(dtm: np.ndarray) -> dict[str, float]:
    """Return basic descriptive statistics for a terrain raster."""

    return {
        "min_elevation_m": float(dtm.min()),
        "max_elevation_m": float(dtm.max()),
        "mean_elevation_m": float(dtm.mean()),
        "std_elevation_m": float(dtm.std()),
    }
