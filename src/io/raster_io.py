from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin


REQUIRED_PROFILE_KEYS = {
    "driver",
    "width",
    "height",
    "count",
    "dtype",
    "crs",
    "transform",
    "nodata",
}


def build_raster_profile(
    *,
    width: int,
    height: int,
    resolution: float,
    crs: str,
    origin_x: float = 0.0,
    origin_y: float | None = None,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Build a stable single-band GeoTIFF profile."""

    if origin_y is None:
        origin_y = float(height) * float(resolution)

    transform = from_origin(origin_x, origin_y, resolution, resolution)
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band GeoTIFF and return array plus metadata profile."""

    raster_path = Path(path)
    with rasterio.open(raster_path) as dataset:
        array = dataset.read(1)
        profile = dataset.profile.copy()
    return array, profile


def read_raster_metadata(path: str | Path) -> dict[str, Any]:
    """Read raster metadata without loading the full array twice."""

    raster_path = Path(path)
    with rasterio.open(raster_path) as dataset:
        profile = dataset.profile.copy()
        profile["bounds"] = dataset.bounds
        profile["res"] = dataset.res
    return profile


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    *,
    build_overviews: bool = True,
) -> Path:
    """Write a single-band GeoTIFF with stable metadata handling."""

    raster_path = Path(path)
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    raster = np.asarray(array)
    if raster.ndim != 2:
        raise ValueError("Only single-band 2D arrays are supported.")

    if raster.dtype == np.bool_:
        raster = raster.astype(np.uint8)

    output_profile = profile.copy()
    output_profile.update(
        {
            "driver": "GTiff",
            "width": int(raster.shape[1]),
            "height": int(raster.shape[0]),
            "count": 1,
            "dtype": str(raster.dtype),
        }
    )

    missing_keys = REQUIRED_PROFILE_KEYS - set(output_profile)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Raster profile missing keys: {missing}")

    if np.issubdtype(raster.dtype, np.integer):
        dtype_info = np.iinfo(raster.dtype)
        nodata = output_profile.get("nodata")
        if nodata is None or nodata < dtype_info.min or nodata > dtype_info.max:
            output_profile["nodata"] = dtype_info.max

    blockxsize = _tile_size(output_profile["width"])
    blockysize = _tile_size(output_profile["height"])
    if blockxsize and blockysize:
        output_profile.update(
            {"compress": "deflate", "predictor": 2, "tiled": True}
        )
        output_profile["blockxsize"] = blockxsize
        output_profile["blockysize"] = blockysize
    else:
        output_profile.update({"compress": "deflate", "predictor": 2})
        output_profile.pop("tiled", None)
        output_profile.pop("blockxsize", None)
        output_profile.pop("blockysize", None)

    with rasterio.open(raster_path, "w", **output_profile) as dataset:
        dataset.write(raster, 1)
        if build_overviews and min(raster.shape) >= 256:
            overview_levels = [2, 4, 8, 16]
            dataset.build_overviews(overview_levels, Resampling.average)
            dataset.update_tags(ns="rio_overview", resampling="average")

    return raster_path


def array_bounds(
    transform: Affine, height: int, width: int
) -> tuple[float, float, float, float]:
    """Return raster bounds as left, bottom, right, top."""

    left = float(transform.c)
    top = float(transform.f)
    right = left + float(width) * float(transform.a)
    bottom = top + float(height) * float(transform.e)
    return left, bottom, right, top


def _tile_size(length: int) -> int | None:
    """Pick a GeoTIFF tile size that is valid for the raster dimension."""

    if length < 16:
        return None

    size = min(256, length)
    while size >= 16 and size % 16 != 0:
        size -= 1
    return size if size >= 16 else None
