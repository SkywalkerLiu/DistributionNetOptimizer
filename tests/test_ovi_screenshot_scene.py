from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

from src.features.obstacles_generator import generate_obstacle_layers
from src.features.users_generator import generate_users
from src.io.vector_io import FEATURE_LAYER_DEFINITIONS, read_layer
from src.main import build_profile, derive_terrain, generate_scene, load_config, profile_bounds
from src.terrain.terrain_derivatives import derive_terrain_layers
from src.terrain.terrain_generator import generate_terrain


CONFIG_PATH = Path("configs/ovi_screenshot_config.yaml")


def test_ovi_terrain_shape_and_elevation_trend() -> None:
    config = load_config(CONFIG_PATH)
    dtm = generate_terrain(config)

    assert dtm.shape == (909, 428)
    assert float(dtm.min()) >= 220.0
    assert float(dtm.max()) <= 420.0
    assert float(dtm[:, :60].mean()) > float(dtm[:, -80:].mean())
    assert np.isfinite(dtm).all()


def test_ovi_users_keep_standard_schema() -> None:
    config = load_config(CONFIG_PATH)
    profile = build_profile(config)
    dtm = generate_terrain(config)
    terrain = derive_terrain_layers(
        dtm,
        resolution_m=float(config["scene"]["resolution_m"]),
        terrain_config=config["terrain"],
    )
    users = generate_users(
        config,
        dtm=dtm,
        valid_mask=terrain["buildable_mask"].astype(bool),
        transform=profile["transform"],
        crs=profile["crs"],
    )

    assert len(users) == 25
    assert list(FEATURE_LAYER_DEFINITIONS["users"][1]) == [
        column for column in users.columns if column != "geometry"
    ]
    assert set(users["user_id"]) == set(range(1, 26))
    assert (users["user_type"] == "high_power_user").sum() == 5


def test_ovi_obstacles_keep_standard_layers() -> None:
    config = load_config(CONFIG_PATH)
    profile = build_profile(config)
    dtm = generate_terrain(config)
    users = generate_users(
        config,
        dtm=dtm,
        valid_mask=np.ones_like(dtm, dtype=bool),
        transform=profile["transform"],
        crs=profile["crs"],
    )
    obstacles = generate_obstacle_layers(
        config,
        scene_bounds=profile_bounds(profile),
        crs=profile["crs"],
        users=users,
    )

    assert len(obstacles["water"]) == 2
    assert len(obstacles["forest"]) == 3
    assert len(obstacles["manual_no_build"]) == 2
    for layer_name, gdf in obstacles.items():
        assert list(FEATURE_LAYER_DEFINITIONS[layer_name][1]) == [
            column for column in gdf.columns if column != "geometry"
        ]
    forbidden_union = unary_union(
        list(obstacles["forest"].geometry)
        + list(obstacles["water"].geometry)
        + list(obstacles["manual_no_build"].geometry)
    )
    assert not users.geometry.intersects(forbidden_union).any()


def test_ovi_generate_scene_and_derive_smoke() -> None:
    config = load_config(CONFIG_PATH)
    config["outputs"]["create_plots"] = False
    tmp_path = _workspace_tmpdir("ovi_scene")

    try:
        paths = generate_scene(config=config, project_root=tmp_path)
        derive_terrain(config=config, project_root=tmp_path)

        users = read_layer(paths["features"], "users")
        water = read_layer(paths["features"], "water")
        forest = read_layer(paths["features"], "forest")
        manual = read_layer(paths["features"], "manual_no_build")

        assert paths["dtm"].exists()
        assert paths["buildable_mask"].exists()
        assert len(users) == 25
        assert len(water) == 2
        assert len(forest) == 3
        assert len(manual) == 2
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _workspace_tmpdir(name: str) -> Path:
    base = Path.cwd() / ".tmp_test_runs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}_{uuid.uuid4().hex}"
    path.mkdir()
    return path
