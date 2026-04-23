from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.io.raster_io import build_raster_profile
from src.viz.plot_terrain_3d import downsample_terrain_surface, generate_terrain_3d_previews


def test_downsample_terrain_surface_limits_grid_size() -> None:
    dtm = np.arange(600, dtype=np.float32).reshape(20, 30)
    profile = build_raster_profile(
        width=30,
        height=20,
        resolution=2.0,
        crs="EPSG:3857",
        origin_x=0.0,
        origin_y=40.0,
    )

    surface = downsample_terrain_surface(dtm=dtm, profile=profile, max_grid_size=8)

    assert surface.z.shape[0] <= 8
    assert surface.z.shape[1] <= 8
    assert surface.original_height == 20
    assert surface.original_width == 30


def test_generate_terrain_3d_previews_writes_png_and_html() -> None:
    tmp_path = _workspace_tmpdir("viz3d")
    dtm = np.linspace(0.0, 100.0, num=24 * 18, dtype=np.float32).reshape(24, 18)
    profile = build_raster_profile(
        width=18,
        height=24,
        resolution=5.0,
        crs="EPSG:3857",
        origin_x=0.0,
        origin_y=120.0,
    )
    crs = "EPSG:3857"
    users = gpd.GeoDataFrame(
        {"user_id": [1, 2], "elev_m": [20.0, 30.0]},
        geometry=[Point(15.0, 80.0), Point(35.0, 40.0)],
        crs=crs,
    )
    forest = gpd.GeoDataFrame(
        {"obs_id": [1]},
        geometry=[Polygon([(10.0, 70.0), (30.0, 70.0), (30.0, 55.0), (10.0, 55.0)])],
        crs=crs,
    )
    water = gpd.GeoDataFrame(
        {"obs_id": [1]},
        geometry=[Polygon([(45.0, 95.0), (65.0, 95.0), (65.0, 75.0), (45.0, 75.0)])],
        crs=crs,
    )
    manual = gpd.GeoDataFrame(
        {"obs_id": [1]},
        geometry=[Polygon([(50.0, 35.0), (70.0, 35.0), (70.0, 20.0), (50.0, 20.0)])],
        crs=crs,
    )
    planned_lines = gpd.GeoDataFrame(
        {"line_id": [1]},
        geometry=[LineString([(20.0, 30.0), (40.0, 30.0), (55.0, 50.0)])],
        crs=crs,
    )

    outputs = generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=tmp_path,
        visualization_config={"terrain_3d_max_grid_size": 12},
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual,
        planned_lines=planned_lines,
    )

    assert outputs["terrain_3d_png"].exists()
    assert outputs["terrain_3d_html"].exists()
    assert outputs["terrain_3d_png"].stat().st_size > 0
    assert outputs["terrain_3d_html"].stat().st_size > 0
    html = outputs["terrain_3d_html"].read_text(encoding="utf-8")
    assert "Users" in html
    assert "Forest" in html
    assert "Water" in html
    assert "Manual No-Build" in html

    shutil.rmtree(tmp_path, ignore_errors=True)


def _workspace_tmpdir(name: str) -> Path:
    """Create a temporary directory inside the current workspace."""

    base = Path.cwd() / ".tmp_test_runs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path
