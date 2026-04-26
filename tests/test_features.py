from __future__ import annotations

import numpy as np
from rasterio.transform import rowcol
from shapely.ops import unary_union

from src.features.obstacles_generator import generate_obstacle_layers, rasterize_forbidden_mask
from src.features.users_generator import generate_users
from src.main import build_profile, profile_bounds
from src.terrain.terrain_derivatives import derive_terrain_layers
from src.terrain.terrain_generator import generate_terrain


def _config() -> dict:
    return {
        "scene": {
            "width_m": 160,
            "height_m": 160,
            "max_elevation_m": 100,
            "resolution_m": 4,
            "origin_x_m": 0,
            "origin_y_m": 160,
            "crs": "EPSG:3857",
            "seed": 23,
        },
        "terrain": {
            "base_type": "saddle",
            "add_perlin_noise": True,
            "noise_scale": 0.1,
            "noise_amplitude": 7.0,
            "noise_octaves": 3,
            "add_gaussian_hills": True,
            "hill_count": 4,
            "valley_ratio": 0.2,
            "smooth_sigma": 1.0,
            "clip_min": 0,
            "clip_max": 100,
            "max_buildable_slope_deg": 35.0,
            "max_buildable_roughness_m": 10.0,
            "roughness_window": 3,
        },
        "users": {
            "count": 10,
            "min_spacing_m": 8,
            "distribution_mode": "clustered",
            "cluster_count": 3,
            "cluster_radius_m": 12,
            "load_kw_range": [1.0, 5.0],
            "importance_range": [1, 3],
        },
        "obstacles": {
            "forest_count": 2,
            "water_count": 1,
            "manual_no_build_count": 1,
            "min_area_m2": 60,
            "max_area_m2": 500,
            "buffer_from_users_m": 4,
        },
        "planning": {
            "transformer_candidate_step_m": 16,
            "pole_candidate_step_m": 8,
        },
    }


def test_users_and_obstacles_are_spatially_consistent() -> None:
    config = _config()
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
    obstacles = generate_obstacle_layers(
        config,
        scene_bounds=profile_bounds(profile),
        crs=profile["crs"],
        users=users,
    )
    forbidden_mask = rasterize_forbidden_mask(
        profile=profile,
        forest=obstacles["forest"],
        water=obstacles["water"],
        manual_no_build=obstacles["manual_no_build"],
    )

    geoms = list(obstacles["water"].geometry) + list(obstacles["manual_no_build"].geometry)
    forbidden_forest = obstacles["forest"].loc[obstacles["forest"]["forbidden"] == 1]
    geoms.extend(list(forbidden_forest.geometry))
    forbidden_union = unary_union(geoms)

    assert len(users) == config["users"]["count"]
    assert users["elev_m"].notna().all()
    assert forbidden_mask.sum() > 0
    if not forbidden_union.is_empty:
        assert not users.geometry.intersects(forbidden_union).any()

    for geometry in geoms:
        point = geometry.representative_point()
        row, col = rowcol(profile["transform"], point.x, point.y)
        assert forbidden_mask[row, col] == 1


def test_clustered_with_scattered_user_count() -> None:
    users = _clustered_with_scattered_users()

    assert len(users) == 50
    assert (users["settlement_type"] == "clustered").sum() == 45
    assert (users["settlement_type"] == "scattered").sum() == 5


def test_cluster_size_between_3_and_8() -> None:
    users = _clustered_with_scattered_users()
    clustered = users.loc[users["settlement_type"] == "clustered"]

    for _, group in clustered.groupby("cluster_id"):
        assert 3 <= len(group) <= 8


def test_cluster_pairwise_distance_not_exceed_10m() -> None:
    users = _clustered_with_scattered_users()
    clustered = users.loc[users["settlement_type"] == "clustered"]

    for _, group in clustered.groupby("cluster_id"):
        for p1 in group.geometry:
            for p2 in group.geometry:
                assert p1.distance(p2) <= 10.0 + 1e-9


def test_scattered_user_count_is_5() -> None:
    users = _clustered_with_scattered_users()

    assert (users["settlement_type"] == "scattered").sum() == 5


def test_scattered_users_have_no_cluster_id() -> None:
    users = _clustered_with_scattered_users()
    scattered = users.loc[users["settlement_type"] == "scattered"]

    assert scattered["cluster_id"].isna().all()


def test_load_group_counts_are_40_and_10() -> None:
    users = _clustered_with_scattered_users()

    assert (users["load_kw"] == 7.0).sum() == 40
    assert (users["load_kw"] == 10.0).sum() == 10
    assert (users["user_type"] == "normal_residential").sum() == 40
    assert (users["user_type"] == "high_power_user").sum() == 10
    assert (users["power_factor"] == 0.85).all()


def test_all_users_are_single_phase() -> None:
    users = _clustered_with_scattered_users()

    assert (users["phase_type"] == "single").all()


def _clustered_with_scattered_users():
    config = _clustered_with_scattered_config()
    profile = build_profile(config)
    dtm = np.zeros((profile["height"], profile["width"]), dtype=np.float32)
    valid_mask = np.ones_like(dtm, dtype=bool)
    return generate_users(
        config,
        dtm=dtm,
        valid_mask=valid_mask,
        transform=profile["transform"],
        crs=profile["crs"],
    )


def _clustered_with_scattered_config() -> dict:
    return {
        "scene": {
            "width_m": 400,
            "height_m": 600,
            "max_elevation_m": 100,
            "resolution_m": 1,
            "origin_x_m": 0,
            "origin_y_m": 600,
            "crs": "EPSG:3857",
            "seed": 66,
        },
        "users": {
            "count": 50,
            "distribution_mode": "clustered_with_scattered",
            "clustered_count": 45,
            "scattered_count": 5,
            "cluster_size_min": 3,
            "cluster_size_max": 8,
            "cluster_diameter_m": 10.0,
            "cluster_center_min_spacing_m": 40.0,
            "scattered_min_distance_from_cluster_m": 30.0,
            "scattered_min_spacing_m": 30.0,
            "load_groups": [
                {
                    "name": "normal_residential",
                    "count": 40,
                    "load_kw": 7.0,
                    "power_factor": 0.85,
                    "phase_type": "single",
                },
                {
                    "name": "high_power_user",
                    "count": 10,
                    "load_kw": 10.0,
                    "power_factor": 0.85,
                    "phase_type": "single",
                },
            ],
            "default_phase_type": "single",
            "importance_range": [1, 3],
        },
    }
