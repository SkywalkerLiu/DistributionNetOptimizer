from __future__ import annotations

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
