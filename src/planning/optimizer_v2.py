from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np

from src.planning.attachment_model import build_attachment_options, select_initial_attachments
from src.planning.bfs_power_flow import run_backward_forward_sweep
from src.planning.common import sample_array
from src.planning.corridor_graph import build_corridor_graph
from src.planning.loss_eval import loss_cost_from_power
from src.planning.models import EvaluatedSolution, OptimizedPlan
from src.planning.neighborhood_search import improve_attachment_choices
from src.planning.phase_assignment import optimize_phase_assignment
from src.planning.pole_generation import generate_plan_layers
from src.planning.progress import OptimizationProgress
from src.planning.radial_tree_milp import solve_radial_tree
from src.planning.summary_v2 import build_summary_v2
from src.planning.transformer_candidates import generate_transformer_candidates
from src.planning.voltage_eval import evaluate_solution_feasibility, phase_unbalance_penalty


def optimize_distribution_network_v2(
    *,
    config: dict[str, Any],
    dtm: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
) -> OptimizedPlan:
    """Run the V2 corridor-graph radial planning optimizer."""

    planning_cfg = _resolve_planning_v2_config(config)
    users = _normalize_users(users=users, dtm=dtm, profile=profile)
    progress = OptimizationProgress(
        enabled=bool(planning_cfg.get("show_progress", True)),
        total_candidates=max(1, min(
            int(planning_cfg.get("tx_prefilter_top_k", 6)),
            int(planning_cfg.get("candidate_solution_pool_size", 10)),
        )),
        max_search_iter=int(planning_cfg.get("alns_max_iter", 40)),
        bar_width=int(planning_cfg.get("progress_bar_width", 32)),
    )
    progress.stage(
        progress=0.02,
        label="准备输入",
        detail=f"用户 {len(users)} | 栅格 {dtm.shape[1]}x{dtm.shape[0]}",
    )
    corridor = build_corridor_graph(
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        users=users,
        planning_cfg=planning_cfg,
        seed=int(config.get("scene", {}).get("seed", 0)),
    )
    progress.stage(
        progress=0.15,
        label="构建候选走廊图",
        detail=f"节点 {len(corridor.nodes)} | 边 {len(corridor.edges)}",
    )
    transformer_candidates = generate_transformer_candidates(
        corridor=corridor,
        users=users,
        slope=slope,
        roughness=roughness,
        planning_cfg=planning_cfg,
    )
    progress.stage(
        progress=0.20,
        label="配变候选预筛",
        detail=f"保留 {len(transformer_candidates)} 个候选点",
    )
    attachment_options = build_attachment_options(
        corridor=corridor,
        users=users,
        planning_cfg=planning_cfg,
        profile=profile,
    )
    base_choices = select_initial_attachments(attachment_options)
    progress.stage(
        progress=0.25,
        label="构建接入候选",
        detail=f"用户 {len(attachment_options)} | 初始接入完成",
    )

    candidate_pool = transformer_candidates[: max(1, int(planning_cfg.get("candidate_solution_pool_size", 10)))]
    progress.total_candidates = max(1, len(candidate_pool))
    solutions: list[EvaluatedSolution] = []
    for candidate_index, candidate in enumerate(candidate_pool, start=1):
        progress.candidate_started(
            index=candidate_index,
            total=len(candidate_pool),
            detail=f"rank={candidate.rank}",
        )

        def evaluate(choices: dict[int, Any]) -> EvaluatedSolution:
            return _evaluate_candidate_solution(
                candidate=candidate,
                choices=choices,
                corridor=corridor,
                users=users,
                planning_cfg=planning_cfg,
            )

        initial = evaluate(base_choices)
        progress.candidate_initial(
            index=candidate_index,
            total=len(candidate_pool),
            objective=initial.objective,
            feasible=initial.feasible,
        )
        improved = improve_attachment_choices(
            options_by_user=attachment_options,
            initial_solution=initial,
            evaluate=evaluate,
            max_iter=int(planning_cfg.get("alns_max_iter", 40)),
            destroy_ratio=float(planning_cfg.get("alns_destroy_ratio", 0.15)),
            progress_callback=lambda iteration, max_iter, changed, current_best: progress.candidate_iteration(
                index=candidate_index,
                total=len(candidate_pool),
                iteration=iteration,
                max_iter=max_iter,
                objective=current_best.objective,
                improved=changed,
            ),
        )
        progress.candidate_finished(
            index=candidate_index,
            total=len(candidate_pool),
            objective=improved.objective,
            feasible=improved.feasible,
        )
        solutions.append(improved)

    if not solutions:
        raise ValueError("The V2 optimizer could not generate any candidate solutions.")
    solutions.sort(key=lambda item: (not item.feasible, item.objective))
    best = solutions[0]

    progress.stage(
        progress=0.96,
        label="生成输出图层",
        detail="杆塔、线路和 summary 落图中",
    )
    transformer_layer, poles_layer, planned_lines, user_connection_public = generate_plan_layers(
        corridor=corridor,
        solution=best,
        users=users,
        dtm=dtm,
        profile=profile,
        planning_cfg=planning_cfg,
        crs=profile["crs"],
    )
    users = users.copy()
    users["assigned_phase"] = users["user_id"].map(lambda value: best.phase_assignment.get(int(value), ""))
    users["connected_node_id"] = users["user_id"].map(lambda value: user_connection_public.get(int(value), ""))
    users["voltage_drop_pct"] = users["user_id"].map(
        lambda value: best.power_flow.user_voltage_drop_pct.get(int(value), 0.0)
    )
    summary = build_summary_v2(
        corridor=corridor,
        solution=best,
        users=users,
        poles=poles_layer,
        planned_lines=planned_lines,
        planning_cfg=planning_cfg,
    )
    progress.finish(
        detail=f"最优 obj={best.objective:.1f} | {'feasible' if best.feasible else 'infeasible'}",
    )
    return OptimizedPlan(
        users=users,
        transformer=transformer_layer,
        poles=poles_layer,
        planned_lines=planned_lines,
        summary=summary,
    )


def _evaluate_candidate_solution(
    *,
    candidate: Any,
    choices: dict[int, Any],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
) -> EvaluatedSolution:
    """Evaluate one transformer candidate plus one set of user attachments."""

    diagnostics: list[str] = []
    for option in choices.values():
        if option.horizontal_length_m > float(planning_cfg.get("max_service_drop_m", 25.0)) + corridor.resolution_m * 2.0:
            diagnostics.append(
                f"Service drop for user {option.user_id} exceeds max length: {option.horizontal_length_m:.2f} m."
            )
    tree = solve_radial_tree(
        corridor=corridor,
        root_node_id=candidate.node_id,
        attachment_node_ids=[option.attach_node_id for option in choices.values()],
    )
    phase_assignment, edge_phase_loads, edge_phase_kw = optimize_phase_assignment(
        tree=tree,
        attachment_choices=choices,
        users=users,
        planning_cfg=planning_cfg,
    )
    power_flow = run_backward_forward_sweep(
        corridor=corridor,
        tree=tree,
        attachment_choices=choices,
        assignments=phase_assignment,
        edge_phase_loads=edge_phase_loads,
        edge_phase_kw=edge_phase_kw,
        users=users,
        planning_cfg=planning_cfg,
    )
    voltage_ok, voltage_diagnostics = evaluate_solution_feasibility(
        users=users,
        power_flow=power_flow,
        planning_cfg=planning_cfg,
    )
    diagnostics.extend(voltage_diagnostics)

    build_cost = _estimate_build_cost(
        tree=tree,
        choices=choices,
        corridor=corridor,
        planning_cfg=planning_cfg,
    )
    loss_cost = loss_cost_from_power(total_loss_kw=power_flow.total_loss_kw, planning_cfg=planning_cfg)
    unbalance_penalty = phase_unbalance_penalty(power_flow=power_flow, planning_cfg=planning_cfg)
    objective = (
        float(planning_cfg.get("build_cost_weight", 1.0)) * build_cost
        + float(planning_cfg.get("loss_cost_weight", 2.0)) * loss_cost
        + unbalance_penalty
        + 1_000_000.0 * len(diagnostics)
    )
    return EvaluatedSolution(
        transformer_candidate=candidate,
        radial_tree=tree,
        attachment_choices=dict(choices),
        phase_assignment=phase_assignment,
        power_flow=power_flow,
        build_cost=float(build_cost),
        loss_cost=float(loss_cost),
        total_unbalance_penalty=float(unbalance_penalty),
        objective=float(objective),
        feasible=bool(voltage_ok and not diagnostics),
        voltage_ok=bool(voltage_ok),
        diagnostics=diagnostics,
    )


def _estimate_build_cost(
    *,
    tree: Any,
    choices: dict[int, Any],
    corridor: Any,
    planning_cfg: dict[str, Any],
) -> float:
    """Estimate the construction cost before final pole generation."""

    backbone_cost = sum(float(corridor.edges[edge_id].build_cost) for edge_id in tree.selected_edge_ids)
    service_cost = sum(float(option.cost) for option in choices.values())
    max_span = max(float(planning_cfg.get("max_pole_span_m", 50.0)), 1.0)
    estimated_poles = max(0, len(tree.depth_by_node) - 1)
    for edge_id in tree.selected_edge_ids:
        estimated_poles += max(0, int(math.ceil(float(corridor.edges[edge_id].horizontal_length_m) / max_span)) - 1)
    pole_cost = estimated_poles * float(planning_cfg.get("pole_fixed_cost", 1800.0))
    transformer_cost = float(planning_cfg.get("transformer_fixed_cost", 120000.0))
    return float(backbone_cost + service_cost + pole_cost + transformer_cost)


def _normalize_users(
    *,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Ensure the V2 optimizer receives the expected user schema."""

    normalized = users.copy()
    if "power_factor" not in normalized.columns:
        normalized["power_factor"] = 0.85
    normalized["power_factor"] = normalized["power_factor"].fillna(0.85).astype(float)

    if "phase_type" not in normalized.columns:
        normalized["phase_type"] = "single"
    normalized["phase_type"] = normalized["phase_type"].fillna("single").astype(str)
    normalized.loc[normalized["phase_type"].isin(["A", "B", "C"]), "phase_type"] = "single"
    normalized.loc[normalized["phase_type"].isin(["ABC"]), "phase_type"] = "three_phase"

    if "assigned_phase" not in normalized.columns:
        normalized["assigned_phase"] = ""
    normalized["assigned_phase"] = normalized["assigned_phase"].fillna("").astype(str)

    if "apparent_kva" not in normalized.columns:
        normalized["apparent_kva"] = normalized["load_kw"] / normalized["power_factor"]
    normalized["apparent_kva"] = normalized["apparent_kva"].astype(float)

    if "elev_m" not in normalized.columns or normalized["elev_m"].isna().any():
        normalized["elev_m"] = [
            sample_array(dtm, profile, float(point.x), float(point.y))
            for point in normalized.geometry
        ]
    if "connected_node_id" not in normalized.columns:
        normalized["connected_node_id"] = ""
    if "voltage_drop_pct" not in normalized.columns:
        normalized["voltage_drop_pct"] = 0.0
    return normalized


def _resolve_planning_v2_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge legacy cost/electrical constants with the new V2 control block."""

    planning_defaults = {
        "enable_v2_optimizer": True,
        "tx_candidate_count": 20,
        "tx_prefilter_top_k": 6,
        "corridor_safe_margin_m": 12.0,
        "corridor_cluster_count": 6,
        "corridor_edge_max_length_m": 180.0,
        "corridor_boundary_penalty_weight": 20.0,
        "build_cost_weight": 1.0,
        "loss_cost_weight": 2.0,
        "phase_unbalance_weight": 3.0,
        "tx_unbalance_weight": 2.0,
        "segment_unbalance_weight": 1.0,
        "max_service_drop_m": 25.0,
        "max_pole_span_m": 50.0,
        "voltage_drop_max_pct": 7.0,
        "phase_balance_target_ratio": 0.10,
        "phase_balance_max_ratio": 0.15,
        "milp_time_limit_s": 180,
        "alns_max_iter": 40,
        "alns_destroy_ratio": 0.15,
        "candidate_solution_pool_size": 10,
        "show_progress": True,
        "progress_bar_width": 32,
    }
    merged = dict(planning_defaults)
    merged.update(config.get("planning", {}))
    merged.update(config.get("planning_v2", {}))
    return merged
