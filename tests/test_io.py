from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from src.io.raster_io import build_raster_profile, read_geotiff, read_raster_metadata, write_geotiff
from src.io.vector_io import append_layer, create_empty_layer, list_layers, overwrite_layer, read_layer


def test_geotiff_roundtrip() -> None:
    tmp_path = _workspace_tmpdir("geotiff")
    profile = build_raster_profile(
        width=8,
        height=6,
        resolution=2.0,
        crs="EPSG:3857",
        origin_x=0.0,
        origin_y=12.0,
    )
    array = np.arange(48, dtype=np.float32).reshape(6, 8)
    path = tmp_path / "sample.tif"

    write_geotiff(path, array, profile, build_overviews=False)
    loaded, loaded_profile = read_geotiff(path)
    metadata = read_raster_metadata(path)

    assert np.allclose(loaded, array)
    assert loaded_profile["transform"] == profile["transform"]
    assert str(loaded_profile["crs"]) == "EPSG:3857"
    assert float(loaded_profile["nodata"]) == -9999.0
    assert metadata["width"] == 8
    assert metadata["height"] == 6
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_gpkg_layer_create_overwrite_append() -> None:
    tmp_path = _workspace_tmpdir("gpkg")
    gpkg_path = tmp_path / "features.gpkg"
    create_empty_layer(
        gpkg_path,
        "planned_lines",
        geometry_type="LineString",
        columns={"line_id": "int64"},
        crs="EPSG:3857",
    )

    users_a = gpd.GeoDataFrame(
        {
            "user_id": [1],
            "load_kw": [3.5],
            "phase_type": ["A"],
            "importance": [1],
            "elev_m": [11.0],
        },
        geometry=[Point(1.0, 1.0)],
        crs="EPSG:3857",
    )
    users_b = gpd.GeoDataFrame(
        {
            "user_id": [2],
            "load_kw": [5.0],
            "phase_type": ["ABC"],
            "importance": [2],
            "elev_m": [13.0],
        },
        geometry=[Point(2.0, 2.0)],
        crs="EPSG:3857",
    )

    overwrite_layer(gpkg_path, "users", users_a)
    append_layer(gpkg_path, "users", users_b)

    users = read_layer(gpkg_path, "users")
    planned_lines = read_layer(gpkg_path, "planned_lines")

    assert len(users) == 2
    assert len(planned_lines) == 0
    assert set(list_layers(gpkg_path)) == {"planned_lines", "users"}
    shutil.rmtree(tmp_path, ignore_errors=True)


def _workspace_tmpdir(name: str) -> Path:
    """Create a temporary directory inside the current workspace."""

    base = Path.cwd() / ".tmp_test_runs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path
