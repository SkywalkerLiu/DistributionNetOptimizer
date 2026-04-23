from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from src.features.users_generator import generate_users
from src.io.raster_io import build_raster_profile
from src.planning.optimizer import optimize_distribution_network


def test_load_groups_generate_default_phase_interfaces() -> None:
    config = _scene_config()
    config["users"] = {
        "count": 5,
        "min_spacing_m": 2,
        "distribution_mode": "uniform",
        "load_groups": [
            {"count": 3, "load_kw": 7.0, "power_factor": 0.85, "phase_type": "single"},
            {"count": 2, "load_kw": 12.0, "power_factor": 0.85, "phase_type": "single"},
        ],
        "importance_range": [1, 3],
    }
    profile = build_raster_profile(width=20, height=20, resolution=5.0, crs="EPSG:3857", origin_y=100.0)
    dtm = np.zeros((20, 20), dtype=np.float32)
    valid_mask = np.ones((20, 20), dtype=np.uint8)

    users = generate_users(config, dtm=dtm, valid_mask=valid_mask, transform=profile["transform"], crs="EPSG:3857")

    assert len(users) == 5
    assert sorted(users["load_kw"].tolist()) == [7.0, 7.0, 7.0, 12.0, 12.0]
    assert set(users["power_factor"].round(2)) == {0.85}
    assert set(users["phase_type"]) == {"single"}
    assert set(users["assigned_phase"]) == {""}
    assert users["apparent_kva"].notna().all()


def test_optimizer_outputs_radial_3d_plan_with_assigned_phases() -> None:
    config = _scene_config()
    profile = build_raster_profile(width=30, height=30, resolution=4.0, crs="EPSG:3857", origin_y=120.0)
    dtm = np.tile(np.linspace(0.0, 20.0, 30, dtype=np.float32), (30, 1))
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = gpd.GeoDataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "load_kw": [7.0, 7.0, 7.0, 12.0, 12.0, 7.0],
            "power_factor": [0.85] * 6,
            "phase_type": ["single"] * 6,
            "assigned_phase": [""] * 6,
            "apparent_kva": [7.0 / 0.85, 7.0 / 0.85, 7.0 / 0.85, 12.0 / 0.85, 12.0 / 0.85, 7.0 / 0.85],
            "importance": [1] * 6,
            "elev_m": [0.0] * 6,
        },
        geometry=[
            Point(18.0, 90.0),
            Point(38.0, 82.0),
            Point(70.0, 78.0),
            Point(90.0, 50.0),
            Point(58.0, 34.0),
            Point(26.0, 42.0),
        ],
        crs="EPSG:3857",
    )

    optimized = optimize_distribution_network(
        config=config,
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
    )

    assert len(optimized.transformer) == 1
    assert not optimized.poles.empty
    assert not optimized.planned_lines.empty
    assert set(optimized.users["assigned_phase"]).issubset({"A", "B", "C"})
    assert (optimized.planned_lines["horizontal_length_m"] <= config["planning"]["max_pole_span_m"] + 1e-9).all()
    service = optimized.planned_lines.loc[optimized.planned_lines["line_type"] == "service_drop"]
    assert (service["horizontal_length_m"] <= config["planning"]["max_service_drop_m"] + 1e-9).all()
    assert len(service) == len(users)
    non_service = optimized.planned_lines.loc[optimized.planned_lines["line_type"] != "service_drop"]
    assert not non_service["from_node"].astype(str).str.startswith("user_").any()
    assert not non_service["to_node"].astype(str).str.startswith("user_").any()
    assert (optimized.planned_lines["length_3d_m"] >= optimized.planned_lines["horizontal_length_m"]).all()
    assert "phase_balance" in optimized.summary


def _scene_config() -> dict:
    return {
        "scene": {"seed": 11, "resolution_m": 4.0},
        "users": {},
        "planning": {
            "transformer_candidate_step_m": 12.0,
            "path_search_step_m": 8.0,
            "transformer_capacity_kva": 630.0,
            "max_loading_ratio": 1.0,
            "transformer_fixed_cost": 100000.0,
            "pole_fixed_cost": 1000.0,
            "hv_pole_height_m": 13.0,
            "lv_pole_height_m": 10.0,
            "transformer_hv_connection_height_m": 12.0,
            "transformer_lv_connection_height_m": 9.0,
            "source_connection_height_m": 12.0,
            "user_attachment_height_m": 4.0,
            "line_cost_per_m": 50.0,
            "service_line_cost_per_m": 30.0,
            "hv_line_cost_per_m": 100.0,
            "max_pole_span_m": 50.0,
            "max_service_drop_m": 25.0,
            "source_point_xy": [0.0, 60.0],
            "phase_balance_max_ratio": 0.25,
            "voltage_drop_max_pct": 20.0,
            "low_voltage_phase_v": 230.0,
            "line_resistance_ohm_per_km": 0.642,
            "line_reactance_ohm_per_km": 0.083,
            "hv_ground_clearance_m": 6.5,
            "lv_ground_clearance_m": 6.0,
            "service_ground_clearance_m": 2.5,
            "clearance_sample_step_m": 2.0,
            "clearance_search_radius_m": 10.0,
            "clearance_max_repair_depth": 6,
            "terrain_length_weight": 1.0,
            "terrain_slope_weight": 0.0,
            "terrain_roughness_weight": 0.0,
            "forbidden_edge_sample_step_m": 4.0,
        },
    }
