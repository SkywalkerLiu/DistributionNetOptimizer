from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.io.raster_io import build_raster_profile
from src.viz.plot_terrain_3d import (
    downsample_terrain_surface,
    generate_optimized_plan_3d_previews,
    generate_terrain_3d_previews,
)


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


def test_downsample_terrain_surface_can_use_original_grid() -> None:
    dtm = np.arange(600, dtype=np.float32).reshape(20, 30)
    profile = build_raster_profile(
        width=30,
        height=20,
        resolution=2.0,
        crs="EPSG:3857",
        origin_x=0.0,
        origin_y=40.0,
    )

    surface = downsample_terrain_surface(dtm=dtm, profile=profile, max_grid_size=0)

    assert surface.z.shape == dtm.shape
    np.testing.assert_array_equal(surface.z, dtm)


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
        {
            "user_id": [1, 2],
            "user_type": ["normal_residential", "high_power_user"],
            "settlement_type": ["clustered", "scattered"],
            "cluster_id": ["C001", None],
            "load_kw": [7.0, 10.0],
            "phase_type": ["single", "single"],
            "assigned_phase": ["", ""],
            "elev_m": [20.0, 30.0],
        },
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
    outputs = generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=tmp_path,
        visualization_config={"terrain_3d_max_grid_size": 12},
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual,
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
    assert "用户编号" in html
    assert "负荷" in html
    assert "LV Poles" not in html
    assert "Planned Lines" not in html
    assert "Transformer" not in html
    assert "Service Drop" not in html

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_terrain_3d_preview_does_not_include_optimized_plan_layers() -> None:
    tmp_path = _workspace_tmpdir("viz3d_terrain_only")
    dtm, profile, layers = _preview_layers()

    outputs = generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=tmp_path,
        visualization_config={"terrain_3d_max_grid_size": 12},
        users=layers["users"],
        forest=layers["forest"],
        water=layers["water"],
        manual_no_build=layers["manual"],
    )

    html = outputs["terrain_3d_html"].read_text(encoding="utf-8")
    assert "Users" in html
    assert "Forest" in html
    assert "Water" in html
    assert "Manual No-Build" in html
    assert "LV Poles" not in html
    assert "Planned Lines" not in html
    assert "Transformer" not in html
    assert "Service Drop" not in html

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_optimized_plan_3d_dynamic_includes_plan_layers() -> None:
    tmp_path = _workspace_tmpdir("viz3d_optimized")
    dtm, profile, layers = _preview_layers()

    outputs = generate_optimized_plan_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=tmp_path,
        visualization_config={"terrain_3d_max_grid_size": 12},
        users=layers["users"],
        forest=layers["forest"],
        water=layers["water"],
        manual_no_build=layers["manual"],
        planned_lines=layers["planned_lines"],
        transformer=layers["transformer"],
        poles=layers["poles"],
    )

    assert outputs["optimized_plan_3d_static"].exists()
    assert outputs["optimized_plan_3d_dynamic"].exists()
    html = outputs["optimized_plan_3d_dynamic"].read_text(encoding="utf-8")
    assert "LV Poles" in html
    assert "Planned Lines" in html
    assert "Transformer" in html
    assert "Service Drop" in html

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_terrain_3d_user_hover_contains_load_and_user_id() -> None:
    tmp_path = _workspace_tmpdir("viz3d_hover")
    dtm, profile, layers = _preview_layers()

    outputs = generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=tmp_path,
        visualization_config={"terrain_3d_max_grid_size": 12},
        users=layers["users"],
        forest=layers["forest"],
        water=layers["water"],
        manual_no_build=layers["manual"],
    )

    html = outputs["terrain_3d_html"].read_text(encoding="utf-8")
    assert "用户编号" in html
    assert "负荷" in html
    assert "U001" in html
    assert "7.0" in html

    shutil.rmtree(tmp_path, ignore_errors=True)


def _preview_layers():
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
        {
            "user_id": [1, 2],
            "user_type": ["normal_residential", "high_power_user"],
            "settlement_type": ["clustered", "scattered"],
            "cluster_id": ["C001", None],
            "load_kw": [7.0, 10.0],
            "phase_type": ["single", "single"],
            "assigned_phase": ["", ""],
            "elev_m": [20.0, 30.0],
        },
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
        {"line_id": [1, 2], "line_type": ["lv_line", "service_drop"]},
        geometry=[
            LineString([(20.0, 30.0), (40.0, 30.0), (55.0, 50.0)]),
            LineString([(35.0, 40.0), (55.0, 50.0)]),
        ],
        crs=crs,
    )
    transformer = gpd.GeoDataFrame(
        {"transformer_id": ["TX1"], "elev_m": [40.0]},
        geometry=[Point(55.0, 50.0)],
        crs=crs,
    )
    poles = gpd.GeoDataFrame(
        {
            "pole_id": ["P0001", "P0002"],
            "pole_type": ["lv_pole", "lv_pole"],
            "pole_height_m": [9.5, 9.5],
            "elev_m": [22.0, 35.0],
        },
        geometry=[Point(20.0, 30.0), Point(55.0, 50.0)],
        crs=crs,
    )
    return dtm, profile, {
        "users": users,
        "forest": forest,
        "water": water,
        "manual": manual,
        "planned_lines": planned_lines,
        "transformer": transformer,
        "poles": poles,
    }


def _workspace_tmpdir(name: str) -> Path:
    """Create a temporary directory inside the current workspace."""

    base = Path.cwd() / ".tmp_test_runs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path
