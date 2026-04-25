from __future__ import annotations

import networkx as nx
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from src.io.raster_io import build_raster_profile
from src.planning.bfs_power_flow import run_backward_forward_sweep
from src.planning.corridor_graph import build_corridor_graph
from src.planning.geometry_constraints import segment_is_feasible
from src.planning.models import AttachmentOption, CorridorEdge, CorridorGraph, CorridorNode, PowerFlowResult, RadialTreeResult
from src.planning.optimizer import optimize_distribution_network
from src.planning.optimizer_v2 import _resolve_parallel_workers
from src.planning.voltage_eval import evaluate_solution_feasibility


def test_corridor_graph_respects_forbidden_mask() -> None:
    config = _config()
    profile = build_raster_profile(width=24, height=24, resolution=4.0, crs="EPSG:3857", origin_y=96.0)
    dtm = np.zeros((24, 24), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    forbidden[9:15, 10:14] = 1
    users = _users(
        [
            (1, 12.0, 84.0, 7.0),
            (2, 28.0, 72.0, 7.0),
            (3, 64.0, 36.0, 12.0),
            (4, 80.0, 20.0, 7.0),
        ]
    )

    corridor = build_corridor_graph(
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
        planning_cfg=_planning_cfg(config),
        seed=11,
    )

    assert corridor.nodes
    assert corridor.edges
    user_union = users.geometry.union_all()
    for edge in corridor.edges.values():
        assert segment_is_feasible(
            edge.geometry.coords[0][0],
            edge.geometry.coords[0][1],
            edge.geometry.coords[-1][0],
            edge.geometry.coords[-1][1],
            allowed_mask=corridor.corridor_mask,
            profile=profile,
            sample_step_m=4.0,
        )
        assert edge.geometry.distance(user_union) >= config["planning_v2"]["line_user_clearance_m"] - 1e-9
    for node in corridor.nodes.values():
        assert Point(float(node.x), float(node.y)).distance(user_union) >= config["planning_v2"]["pole_user_clearance_m"] - 1e-9


def test_optimizer_outputs_v2_radial_plan() -> None:
    config = _config()
    profile = build_raster_profile(width=30, height=30, resolution=4.0, crs="EPSG:3857", origin_y=120.0)
    dtm = np.zeros((30, 30), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = _users(
        [
            (1, 18.0, 90.0, 7.0),
            (2, 38.0, 82.0, 7.0),
            (3, 70.0, 78.0, 7.0),
            (4, 90.0, 50.0, 12.0),
            (5, 58.0, 34.0, 12.0),
            (6, 26.0, 42.0, 7.0),
        ]
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

    assert optimized.summary["algorithm"] == "planning_v2"
    assert optimized.summary["infeasible_reasons"] == []
    assert optimized.summary["infeasible_reason"] == []
    assert "top_diagnostics" in optimized.summary
    assert "violation_examples" in optimized.summary
    assert optimized.summary["final_geometry_checked_solution_count"] >= 1
    assert optimized.summary["performance"]["initial_candidate_count"] == config["planning_v2"]["tx_prefilter_top_k"]
    assert optimized.summary["performance"]["local_search_candidate_count"] == config["planning_v2"]["local_search_top_k"]
    assert optimized.summary["performance"]["milp_solve_count"] >= config["planning_v2"]["tx_prefilter_top_k"]
    assert "local_search_full_eval_count" in optimized.summary["performance"]
    assert len(optimized.transformer) == 1
    assert not optimized.poles.empty
    assert not optimized.planned_lines.empty
    assert set(optimized.users["assigned_phase"]).issubset({"A", "B", "C"})
    service = optimized.planned_lines.loc[optimized.planned_lines["line_type"] == "service_drop"]
    assert len(service) == len(users)
    assert (service["horizontal_length_m"] <= config["planning_v2"]["max_service_drop_m"] + 1e-9).all()
    assert optimized.users["connected_node_id"].astype(str).ne("").all()

    lv_lines = optimized.planned_lines.loc[optimized.planned_lines["line_type"] == "lv_line"]
    assert (optimized.poles["user_clearance_m"] >= config["planning_v2"]["pole_user_clearance_m"] - 1e-9).all()
    assert (lv_lines["user_clearance_m"] >= config["planning_v2"]["line_user_clearance_m"] - 1e-9).all()
    graph = nx.Graph()
    for row in lv_lines.itertuples():
        graph.add_edge(str(row.from_node), str(row.to_node))
    assert nx.is_forest(graph)


def test_optimizer_reports_balancing_loss_and_voltage_metrics() -> None:
    config = _config()
    profile = build_raster_profile(width=26, height=26, resolution=4.0, crs="EPSG:3857", origin_y=104.0)
    dtm = np.zeros((26, 26), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = _users(
        [
            (1, 16.0, 80.0, 7.0),
            (2, 24.0, 72.0, 7.0),
            (3, 36.0, 60.0, 7.0),
            (4, 56.0, 52.0, 12.0),
            (5, 68.0, 44.0, 12.0),
            (6, 76.0, 28.0, 7.0),
        ]
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

    assert optimized.summary["losses"]["total_loss_kw"] >= 0.0
    assert optimized.summary["losses"]["loss_penalty"] >= 0.0
    assert "loss_penalty" in optimized.summary["objective"]
    assert "voltage_drop_penalty" in optimized.summary["objective"]
    assert optimized.summary["voltage"]["hard_constraint_enabled"] is False
    assert optimized.summary["voltage"]["used_as_soft_penalty"] is True
    assert optimized.summary["voltage"]["max_total_voltage_drop_pct"] >= 0.0
    assert "top_voltage_drop_users" in optimized.summary["voltage"]
    assert "max_phase_current_a" in optimized.summary["max_current_line"]
    assert "loss_kw" in optimized.summary["max_current_line"]
    assert optimized.summary["phase_balance"]["transformer_unbalance_ratio"] <= config["planning_v2"]["phase_balance_max_ratio"] + 1e-9


def test_optimizer_progress_bar_writes_terminal_status(capsys) -> None:
    config = _config()
    config["planning_v2"]["show_progress"] = True
    config["planning_v2"]["alns_max_iter"] = 2
    profile = build_raster_profile(width=22, height=22, resolution=4.0, crs="EPSG:3857", origin_y=88.0)
    dtm = np.zeros((22, 22), dtype=np.float32)
    slope = np.zeros_like(dtm, dtype=np.float32)
    roughness = np.zeros_like(dtm, dtype=np.float32)
    buildable = np.ones_like(dtm, dtype=np.uint8)
    forbidden = np.zeros_like(dtm, dtype=np.uint8)
    users = _users(
        [
            (1, 16.0, 72.0, 7.0),
            (2, 32.0, 64.0, 7.0),
            (3, 60.0, 40.0, 12.0),
        ]
    )

    optimize_distribution_network(
        config=config,
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable,
        forbidden_mask=forbidden,
        profile=profile,
        users=users,
    )

    captured = capsys.readouterr()
    assert "优化进度 [" in captured.out
    assert "候选评估" in captured.out
    assert "完成" in captured.out


def test_feasibility_reports_machine_readable_reason_codes() -> None:
    power_flow = PowerFlowResult(
        edge_phase_loads={},
        edge_phase_kw={},
        edge_losses_kw={},
        edge_voltage_drop_pct={},
        edge_phase_currents_a={},
        transformer_phase_loads=np.asarray([30.0, 0.0, 0.0], dtype=float),
        user_voltage_drop_pct={1: 12.0},
        user_service_drop_pct={1: 0.5},
        user_connection_nodes={1: "n1"},
        total_loss_kw=0.0,
        max_voltage_drop_pct=12.0,
    )

    ok, diagnostics, reasons = evaluate_solution_feasibility(
        users=gpd.GeoDataFrame(),
        power_flow=power_flow,
        planning_cfg={
            "transformer_capacity_kva": 630.0,
            "max_loading_ratio": 1.0,
            "voltage_drop_max_pct": 7.0,
            "phase_balance_max_ratio": 0.15,
        },
    )

    assert not ok
    assert diagnostics
    assert reasons == ["phase_unbalance_exceeded"]


def test_power_flow_user_voltage_drop_uses_only_service_drop() -> None:
    graph = nx.Graph()
    graph.add_edge("root", "attach", edge_id="e1")
    corridor = CorridorGraph(
        graph=graph,
        nodes={
            "root": CorridorNode("root", 0.0, 0.0, 0.0, 0, 0, "tx"),
            "attach": CorridorNode("attach", 1000.0, 0.0, 0.0, 0, 1000, "pole"),
        },
        edges={
            "e1": CorridorEdge(
                edge_id="e1",
                u="root",
                v="attach",
                geometry=LineString([(0.0, 0.0), (1000.0, 0.0)]),
                horizontal_length_m=1000.0,
                length_3d_m=1000.0,
                build_cost=1000.0,
                terrain_cost=0.0,
                risk_cost=0.0,
                max_span_feasible=True,
                is_forbidden=False,
                slope_deg=0.0,
                boundary_clearance_m=20.0,
            )
        },
        corridor_mask=np.ones((2, 2), dtype=np.uint8),
        boundary_distance_m=np.ones((2, 2), dtype=np.float32),
        resolution_m=1.0,
    )
    tree = RadialTreeResult(
        root_node_id="root",
        selected_edge_ids=["e1"],
        parent_by_node={"attach": "root"},
        depth_by_node={"root": 0, "attach": 1},
        terminal_nodes={"attach"},
    )
    users = _users([(1, 1010.0, 0.0, 7.0)])
    option = AttachmentOption(
        user_id=1,
        attach_node_id="attach",
        horizontal_length_m=10.0,
        length_3d_m=10.0,
        cost=10.0,
    )

    result = run_backward_forward_sweep(
        corridor=corridor,
        tree=tree,
        attachment_choices={1: option},
        assignments={1: "A"},
        edge_phase_loads={("root", "attach"): np.asarray([100.0, 0.0, 0.0], dtype=float)},
        edge_phase_kw={("root", "attach"): np.asarray([85.0, 0.0, 0.0], dtype=float)},
        users=users,
        planning_cfg={},
    )

    np.testing.assert_allclose(
        result.edge_phase_currents_a[("root", "attach")],
        np.asarray([100.0 * 1000.0 / 230.0, 0.0, 0.0], dtype=float),
    )
    assert result.edge_voltage_drop_pct[("root", "attach")][0] > 0.0
    assert result.edge_losses_kw[("root", "attach")] > 0.0
    assert result.total_loss_kw > 0.0
    assert result.user_voltage_drop_pct[1] > result.user_service_drop_pct[1]
    assert np.isclose(result.user_service_drop_pct[1], 15.0)
    assert result.max_voltage_drop_pct == result.user_voltage_drop_pct[1]


def test_parallel_worker_auto_resolution(monkeypatch) -> None:
    monkeypatch.setattr("src.planning.optimizer_v2.os.cpu_count", lambda: 8)
    assert _resolve_parallel_workers(
        planning_cfg={"parallel_candidate_eval": True, "parallel_workers": 0},
        task_count=20,
    ) == 6
    assert _resolve_parallel_workers(
        planning_cfg={"parallel_candidate_eval": True, "parallel_workers": 3},
        task_count=20,
    ) == 3
    assert _resolve_parallel_workers(
        planning_cfg={"parallel_candidate_eval": False, "parallel_workers": 0},
        task_count=20,
    ) == 1


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


def _planning_cfg(config: dict) -> dict:
    merged = {
        "corridor_safe_margin_m": 8.0,
        "corridor_cluster_count": 3,
        "corridor_edge_max_length_m": 40.0,
        "corridor_boundary_penalty_weight": 16.0,
        "max_service_drop_m": 25.0,
        "candidate_solution_pool_size": 6,
        "max_pole_span_m": 50.0,
        "line_cost_per_m": 50.0,
        "service_line_cost_per_m": 30.0,
    }
    merged.update(config["planning"])
    merged.update(config["planning_v2"])
    return merged


def _config() -> dict:
    return {
        "scene": {"seed": 11, "resolution_m": 4.0},
        "users": {},
        "planning": {
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
        },
        "planning_v2": {
            "enable_v2_optimizer": True,
            "tx_candidate_count": 8,
            "tx_prefilter_top_k": 3,
            "corridor_safe_margin_m": 8.0,
            "corridor_cluster_count": 3,
            "corridor_edge_max_length_m": 40.0,
            "corridor_boundary_penalty_weight": 16.0,
            "build_cost_weight": 1.0,
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
        },
    }
