from __future__ import annotations

import copy

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.io.raster_io import build_raster_profile
from src.planning.corridor_graph import build_corridor_graph
from src.planning.optimizer import optimize_distribution_network


@pytest.fixture(scope="module")
def optimized_plan():
    config = _config()
    profile = build_raster_profile(width=28, height=28, resolution=4.0, crs="EPSG:3857", origin_y=112.0)
    dtm = np.zeros((28, 28), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = _users(
        [
            (1, 16.0, 88.0, 7.0),
            (2, 36.0, 80.0, 7.0),
            (3, 72.0, 84.0, 12.0),
            (4, 88.0, 56.0, 7.0),
            (5, 60.0, 28.0, 12.0),
            (6, 24.0, 40.0, 7.0),
        ]
    )

    return optimize_distribution_network(
        config=config,
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
    )


def test_path_diagnostics_are_reported(optimized_plan) -> None:
    assert "path_diagnostics" in optimized_plan.summary
    diagnostics = optimized_plan.summary["path_diagnostics"]
    assert "max_user_path_length_m" in diagnostics
    assert "top_long_user_paths" in diagnostics


def test_root_feeder_diagnostics_are_reported(optimized_plan) -> None:
    assert "root_feeder_diagnostics" in optimized_plan.summary
    diagnostics = optimized_plan.summary["root_feeder_diagnostics"]
    assert "root_feeder_count" in diagnostics


def test_corridor_neighbor_count_controls_graph_density() -> None:
    profile = build_raster_profile(width=36, height=36, resolution=4.0, crs="EPSG:3857", origin_y=144.0)
    dtm = np.zeros((36, 36), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = _users(
        [
            (1, 16.0, 124.0, 7.0),
            (2, 36.0, 116.0, 7.0),
            (3, 60.0, 100.0, 7.0),
            (4, 88.0, 84.0, 12.0),
            (5, 116.0, 64.0, 12.0),
            (6, 104.0, 32.0, 7.0),
            (7, 64.0, 24.0, 7.0),
            (8, 24.0, 44.0, 7.0),
        ]
    )
    cfg_low = _planning_cfg()
    cfg_high = copy.deepcopy(cfg_low)
    cfg_low["corridor_neighbor_count"] = 6
    cfg_high["corridor_neighbor_count"] = 16

    graph_low = build_corridor_graph(
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
        planning_cfg=cfg_low,
        seed=13,
    )
    graph_high = build_corridor_graph(
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
        planning_cfg=cfg_high,
        seed=13,
    )

    assert len(graph_high.edges) >= len(graph_low.edges)


def _users(specs: list[tuple[int, float, float, float]]) -> gpd.GeoDataFrame:
    rows = []
    geometry = []
    for user_id, x, y, load_kw in specs:
        rows.append(
            {
                "user_id": int(user_id),
                "load_kw": float(load_kw),
                "power_factor": 0.85,
                "phase_type": "single",
                "assigned_phase": "",
                "apparent_kva": float(load_kw / 0.85),
                "importance": 1,
                "elev_m": 0.0,
            }
        )
        geometry.append(Point(float(x), float(y)))
    return gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:3857")


def _planning_cfg() -> dict:
    return {
        "corridor_safe_margin_m": 4.0,
        "corridor_cluster_count": 3,
        "corridor_edge_max_length_m": 56.0,
        "corridor_neighbor_count": 12,
        "corridor_boundary_penalty_weight": 16.0,
        "max_service_drop_m": 25.0,
        "candidate_solution_pool_size": 4,
        "max_pole_span_m": 50.0,
        "pole_user_clearance_m": 0.0,
        "line_user_clearance_m": 0.0,
        "line_cost_per_m": 50.0,
        "service_line_cost_per_m": 30.0,
    }


def _config() -> dict:
    planning = {
        "transformer_capacity_kva": 630.0,
        "max_loading_ratio": 1.0,
        "transformer_fixed_cost": 100000.0,
        "pole_fixed_cost": 1000.0,
        "lv_pole_height_m": 10.0,
        "transformer_lv_connection_height_m": 9.0,
        "user_attachment_height_m": 4.0,
        "line_cost_per_m": 50.0,
        "service_line_cost_per_m": 30.0,
        "max_pole_span_m": 50.0,
        "max_service_drop_m": 25.0,
        "phase_balance_max_ratio": 0.25,
        "voltage_drop_max_pct": 20.0,
        "low_voltage_phase_v": 230.0,
        "lv_ground_clearance_m": 2.5,
        "service_ground_clearance_m": 2.0,
        "clearance_sample_step_m": 2.0,
    }
    planning_v2 = {
        "enable_v2_optimizer": True,
        "tx_candidate_count": 8,
        "tx_prefilter_top_k": 3,
        "corridor_safe_margin_m": 8.0,
        "corridor_cluster_count": 3,
        "corridor_edge_max_length_m": 48.0,
        "corridor_neighbor_count": 12,
        "corridor_boundary_penalty_weight": 16.0,
        "build_cost_weight": 1.0,
        "path_length_penalty_weight": 20.0,
        "max_user_path_length_m": 300.0,
        "max_user_path_penalty_weight": 1000.0,
        "load_weighted_path_penalty_weight": 1.0,
        "root_feeder_min_count": 3,
        "root_feeder_count_penalty_weight": 50000.0,
        "phase_unbalance_weight": 1.0,
        "tx_unbalance_weight": 1.0,
        "segment_unbalance_weight": 0.5,
        "max_service_drop_m": 25.0,
        "max_pole_span_m": 50.0,
        "pole_user_clearance_m": 5.0,
        "line_user_clearance_m": 1.0,
        "voltage_drop_max_pct": 20.0,
        "phase_balance_target_ratio": 0.15,
        "phase_balance_max_ratio": 0.25,
        "solver_backend": "highs",
        "mip_gap": 0.05,
        "parallel_candidate_eval": True,
        "parallel_workers": 1,
        "highs_threads_per_worker": 1,
        "local_search_top_k": 1,
        "emit_performance_metrics": True,
        "alns_max_iter": 1,
        "alns_destroy_ratio": 0.2,
        "candidate_solution_pool_size": 4,
        "show_progress": False,
        "progress_bar_width": 24,
    }
    return {
        "scene": {"seed": 13, "resolution_m": 4.0},
        "users": {},
        "planning": planning,
        "planning_v2": planning_v2,
    }
