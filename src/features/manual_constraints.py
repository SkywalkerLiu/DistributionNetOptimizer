from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.features.obstacles_generator import rasterize_forbidden_mask
from src.io.vector_io import overwrite_layer, read_layer


def load_manual_constraints(path: str | Path, *, crs: str) -> gpd.GeoDataFrame:
    """Load manual no-build polygons from an external GeoJSON file."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Manual constraints file not found: {source_path}")

    gdf = gpd.read_file(source_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        gdf = gdf.to_crs(crs)

    if "forbidden" not in gdf.columns:
        gdf["forbidden"] = 1
    if "source" not in gdf.columns:
        gdf["source"] = "external"
    if "reason" not in gdf.columns:
        gdf["reason"] = "manual_import"
    if "obs_id" not in gdf.columns:
        gdf["obs_id"] = range(1, len(gdf) + 1)
    return gdf[["obs_id", "source", "reason", "forbidden", "geometry"]]


def merge_manual_constraints(
    existing: gpd.GeoDataFrame,
    imported: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Merge imported polygons into the manual no-build layer."""

    frames = [
        frame for frame in (existing, imported) if frame is not None and not frame.empty
    ]
    if not frames:
        raise ValueError("No manual constraints were provided for merging.")

    crs = frames[0].crs
    combined = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    merged["obs_id"] = range(1, len(merged) + 1)
    return merged[["obs_id", "source", "reason", "forbidden", "geometry"]]


def refresh_manual_constraints(
    *,
    gpkg_path: str | Path,
    profile: dict[str, Any],
    external_geojson: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Update manual no-build constraints and return the refreshed mask."""

    manual = read_layer(gpkg_path, "manual_no_build")
    if external_geojson is not None:
        imported = load_manual_constraints(external_geojson, crs=str(manual.crs))
        manual = merge_manual_constraints(manual, imported)
        overwrite_layer(gpkg_path, "manual_no_build", manual)

    forest = read_layer(gpkg_path, "forest")
    water = read_layer(gpkg_path, "water")
    forbidden = rasterize_forbidden_mask(
        profile=profile,
        forest=forest,
        water=water,
        manual_no_build=manual,
    )
    return manual, forbidden

