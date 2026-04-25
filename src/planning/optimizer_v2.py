from __future__ import annotations

import os
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from src.planning.radial_tree_milp import RadialTreeModelData, build_radial_tree_model_data, solve_radial_tree
from src.planning.summary_v2 import build_summary_v2
from src.planning.transformer_candidates import generate_transformer_candidates
from src.planning.voltage_eval import evaluate_solution_feasibility, phase_unbalance_penalty


@dataclass(slots=True)
class CandidateEvaluationResult:
    """One evaluated transformer candidate plus lightweight performance data."""

    candidate_index: int
    solution: EvaluatedSolution
    duration_s: float
    milp_solve_count: int
    milp_cache_hit_count: int
    local_search_trial_count: int = 0
    local_search_full_eval_count: int = 0


@dataclass(slots=True)
class FinalGeometrySelection:
    """A solution after final layer generation and geometry summary checks."""

    solution: EvaluatedSolution
    users: gpd.GeoDataFrame
    transformer_layer: gpd.GeoDataFrame
    poles_layer: gpd.GeoDataFrame
    planned_lines: gpd.GeoDataFrame
    user_connection_public: dict[int, str]
    summary: dict[str, Any]
    checked_index: int


_WORKER_CONTEXT: dict[str, Any] = {}


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

    total_started_at = time.perf_counter()
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
    corridor_started_at = time.perf_counter()
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
    corridor_duration_s = time.perf_counter() - corridor_started_at
    progress.stage(
        progress=0.15,
        label="构建候选走廊图",
        detail=f"节点 {len(corridor.nodes)} | 边 {len(corridor.edges)}",
    )
    prefilter_started_at = time.perf_counter()
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
    prefilter_duration_s = time.perf_counter() - prefilter_started_at
    progress.stage(
        progress=0.25,
        label="构建接入候选",
        detail=f"用户 {len(attachment_options)} | 初始接入完成",
    )

    candidate_pool = transformer_candidates[: max(1, int(planning_cfg.get("candidate_solution_pool_size", 10)))]
    progress.total_candidates = max(1, len(candidate_pool))
    worker_count = _resolve_parallel_workers(
        planning_cfg=planning_cfg,
        task_count=len(candidate_pool),
    )
    radial_model_data = build_radial_tree_model_data(corridor)

    initial_started_at = time.perf_counter()
    initial_results = _evaluate_initial_candidates(
        candidate_pool=candidate_pool,
        base_choices=base_choices,
        corridor=corridor,
        users=users,
        planning_cfg=planning_cfg,
        radial_model_data=radial_model_data,
        worker_count=worker_count,
        progress=progress,
    )
    initial_duration_s = time.perf_counter() - initial_started_at
    initial_results.sort(key=lambda item: item.candidate_index)

    local_search_top_k = max(0, min(int(planning_cfg.get("local_search_top_k", 8)), len(initial_results)))
    ranked_for_search = sorted(initial_results, key=lambda item: (not item.solution.feasible, item.solution.objective))[
        :local_search_top_k
    ]
    search_started_at = time.perf_counter()
    improved_results = _improve_top_candidates(
        initial_results=ranked_for_search,
        attachment_options=attachment_options,
        corridor=corridor,
        users=users,
        planning_cfg=planning_cfg,
        radial_model_data=radial_model_data,
        worker_count=worker_count,
        progress=progress,
    )
    local_search_duration_s = time.perf_counter() - search_started_at

    solutions: list[EvaluatedSolution] = [result.solution for result in initial_results]
    solutions.extend(result.solution for result in improved_results)

    if not solutions:
        raise ValueError("The V2 optimizer could not generate any candidate solutions.")
    progress.stage(
        progress=0.96,
        label="输出复核",
        detail="geometry check and summary",
    )
    output_started_at = time.perf_counter()
    geometry_selection = _select_final_solution_with_geometry_check(
        solutions=solutions,
        corridor=corridor,
        users=users,
        dtm=dtm,
        profile=profile,
        planning_cfg=planning_cfg,
        crs=profile["crs"],
        top_k=int(planning_cfg.get("final_geometry_check_top_k", 8)),
    )
    best = geometry_selection.solution
    users = geometry_selection.users
    transformer_layer = geometry_selection.transformer_layer
    poles_layer = geometry_selection.poles_layer
    planned_lines = geometry_selection.planned_lines
    summary = geometry_selection.summary
    output_duration_s = time.perf_counter() - output_started_at
    if bool(planning_cfg.get("emit_performance_metrics", True)):
        summary["performance"] = _build_performance_summary(
            total_duration_s=time.perf_counter() - total_started_at,
            corridor_duration_s=corridor_duration_s,
            prefilter_duration_s=prefilter_duration_s,
            initial_duration_s=initial_duration_s,
            local_search_duration_s=local_search_duration_s,
            output_duration_s=output_duration_s,
            worker_count=worker_count,
            initial_results=initial_results,
            improved_results=improved_results,
            local_search_top_k=local_search_top_k,
            planning_cfg=planning_cfg,
        )
    progress.finish(detail=_finish_detail(objective=best.objective, summary=summary))
    return OptimizedPlan(
        users=users,
        transformer=transformer_layer,
        poles=poles_layer,
        planned_lines=planned_lines,
        summary=summary,
    )


def _evaluate_initial_candidates(
    *,
    candidate_pool: list[Any],
    base_choices: dict[int, Any],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
    radial_model_data: RadialTreeModelData,
    worker_count: int,
    progress: OptimizationProgress,
) -> list[CandidateEvaluationResult]:
    """Evaluate every initial transformer candidate, in parallel when configured."""

    total = len(candidate_pool)
    if _use_parallel(planning_cfg=planning_cfg, worker_count=worker_count):
        context = {
            "corridor": corridor,
            "users": users,
            "planning_cfg": planning_cfg,
            "radial_model_data": radial_model_data,
            "base_choices": base_choices,
        }
        try:
            results: list[CandidateEvaluationResult] = []
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_evaluation_worker,
                initargs=(context,),
            ) as executor:
                futures = {
                    executor.submit(_evaluate_initial_candidate_worker, index, candidate): index
                    for index, candidate in enumerate(candidate_pool, start=1)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    results.append(result)
                    progress.stage(
                        progress=0.25 + 0.35 * completed / max(total, 1),
                        label="候选评估",
                        detail=(
                            f"初始候选并行评估 {completed}/{total} | workers={worker_count} | "
                            f"rank={result.solution.transformer_candidate.rank} | obj={result.solution.objective:.1f}"
                        ),
                    )
            return results
        except (OSError, PermissionError, RuntimeError) as exc:
            planning_cfg["parallel_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            progress.stage(
                progress=0.25,
                label="候选评估",
                detail="并行评估不可用，已自动回退串行",
            )

    results: list[CandidateEvaluationResult] = []
    for index, candidate in enumerate(candidate_pool, start=1):
        progress.candidate_started(index=index, total=total, detail=f"rank={candidate.rank}")
        result = _evaluate_initial_candidate_direct(
            candidate_index=index,
            candidate=candidate,
            base_choices=base_choices,
            corridor=corridor,
            users=users,
            planning_cfg=planning_cfg,
            radial_model_data=radial_model_data,
        )
        progress.candidate_initial(
            index=index,
            total=total,
            objective=result.solution.objective,
            feasible=result.solution.feasible,
        )
        results.append(result)
    return results


def _improve_top_candidates(
    *,
    initial_results: list[CandidateEvaluationResult],
    attachment_options: dict[int, list[Any]],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
    radial_model_data: RadialTreeModelData,
    worker_count: int,
    progress: OptimizationProgress,
) -> list[CandidateEvaluationResult]:
    """Run local reattachment search for the ranked top-k candidates."""

    if not initial_results:
        return []

    total = len(initial_results)
    if _use_parallel(planning_cfg=planning_cfg, worker_count=worker_count) and total > 1:
        context = {
            "corridor": corridor,
            "users": users,
            "planning_cfg": planning_cfg,
            "radial_model_data": radial_model_data,
            "attachment_options": attachment_options,
        }
        try:
            results: list[CandidateEvaluationResult] = []
            with ProcessPoolExecutor(
                max_workers=min(worker_count, total),
                initializer=_init_evaluation_worker,
                initargs=(context,),
            ) as executor:
                futures = {
                    executor.submit(_improve_candidate_worker, result): result.candidate_index
                    for result in initial_results
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    results.append(result)
                    progress.stage(
                        progress=0.60 + 0.30 * completed / max(total, 1),
                        label="候选评估",
                        detail=(
                            f"局部搜索并行评估 {completed}/{total} | workers={min(worker_count, total)} | "
                            f"候选 {result.candidate_index} | obj={result.solution.objective:.1f}"
                        ),
                    )
            return results
        except (OSError, PermissionError, RuntimeError) as exc:
            planning_cfg["parallel_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            progress.stage(
                progress=0.60,
                label="候选评估",
                detail="局部搜索并行不可用，已自动回退串行",
            )

    results = []
    for ranked_index, result in enumerate(initial_results, start=1):
        improved = _improve_candidate_direct(
            initial_result=result,
            attachment_options=attachment_options,
            corridor=corridor,
            users=users,
            planning_cfg=planning_cfg,
            radial_model_data=radial_model_data,
            progress=progress,
            progress_index=ranked_index,
            progress_total=total,
        )
        progress.candidate_finished(
            index=ranked_index,
            total=total,
            objective=improved.solution.objective,
            feasible=improved.solution.feasible,
        )
        results.append(improved)
    return results


def _evaluate_initial_candidate_direct(
    *,
    candidate_index: int,
    candidate: Any,
    base_choices: dict[int, Any],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
    radial_model_data: RadialTreeModelData,
) -> CandidateEvaluationResult:
    """Evaluate one initial candidate in the current process."""

    started_at = time.perf_counter()
    counters: dict[str, int] = {}
    solution = _evaluate_candidate_solution(
        candidate=candidate,
        choices=base_choices,
        corridor=corridor,
        users=users,
        planning_cfg=planning_cfg,
        tree_cache={},
        radial_model_data=radial_model_data,
        performance_counters=counters,
    )
    return CandidateEvaluationResult(
        candidate_index=candidate_index,
        solution=solution,
        duration_s=time.perf_counter() - started_at,
        milp_solve_count=int(counters.get("milp_solve_count", 0)),
        milp_cache_hit_count=int(counters.get("milp_cache_hit_count", 0)),
    )


def _improve_candidate_direct(
    *,
    initial_result: CandidateEvaluationResult,
    attachment_options: dict[int, list[Any]],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
    radial_model_data: RadialTreeModelData,
    progress: OptimizationProgress | None = None,
    progress_index: int = 1,
    progress_total: int = 1,
) -> CandidateEvaluationResult:
    """Improve one candidate in the current process."""

    started_at = time.perf_counter()
    counters: dict[str, int] = {}
    tree_cache = {
        _tree_cache_key(
            candidate_node_id=initial_result.solution.transformer_candidate.node_id,
            choices=initial_result.solution.attachment_choices,
        ): initial_result.solution.radial_tree
    }

    def evaluate(choices: dict[int, Any]) -> EvaluatedSolution:
        return _evaluate_candidate_solution(
            candidate=initial_result.solution.transformer_candidate,
            choices=choices,
            corridor=corridor,
            users=users,
            planning_cfg=planning_cfg,
            tree_cache=tree_cache,
            radial_model_data=radial_model_data,
            performance_counters=counters,
        )

    improved = improve_attachment_choices(
        options_by_user=attachment_options,
        initial_solution=initial_result.solution,
        evaluate=evaluate,
        max_iter=int(planning_cfg.get("alns_max_iter", 40)),
        destroy_ratio=float(planning_cfg.get("alns_destroy_ratio", 0.15)),
        patience=int(planning_cfg.get("local_search_patience", 8)),
        top_options=int(planning_cfg.get("local_search_top_options", 5)),
        max_full_evals=int(planning_cfg.get("local_search_max_full_evals", 200)),
        performance_counters=counters,
        progress_callback=None
        if progress is None
        else lambda iteration, max_iter, changed, current_best: progress.candidate_iteration(
            index=progress_index,
            total=progress_total,
            iteration=iteration,
            max_iter=max_iter,
            objective=current_best.objective,
            improved=changed,
        ),
    )
    return CandidateEvaluationResult(
        candidate_index=initial_result.candidate_index,
        solution=improved,
        duration_s=time.perf_counter() - started_at,
        milp_solve_count=int(counters.get("milp_solve_count", 0)),
        milp_cache_hit_count=int(counters.get("milp_cache_hit_count", 0)),
        local_search_trial_count=int(counters.get("local_search_trial_count", 0)),
        local_search_full_eval_count=int(counters.get("local_search_full_eval_count", 0)),
    )


def _init_evaluation_worker(context: dict[str, Any]) -> None:
    """Set process-local immutable evaluation context."""

    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(context)


def _evaluate_initial_candidate_worker(candidate_index: int, candidate: Any) -> CandidateEvaluationResult:
    """Evaluate one initial candidate inside a process worker."""

    return _evaluate_initial_candidate_direct(
        candidate_index=candidate_index,
        candidate=candidate,
        base_choices=_WORKER_CONTEXT["base_choices"],
        corridor=_WORKER_CONTEXT["corridor"],
        users=_WORKER_CONTEXT["users"],
        planning_cfg=_WORKER_CONTEXT["planning_cfg"],
        radial_model_data=_WORKER_CONTEXT["radial_model_data"],
    )


def _improve_candidate_worker(initial_result: CandidateEvaluationResult) -> CandidateEvaluationResult:
    """Improve one candidate inside a process worker."""

    return _improve_candidate_direct(
        initial_result=initial_result,
        attachment_options=_WORKER_CONTEXT["attachment_options"],
        corridor=_WORKER_CONTEXT["corridor"],
        users=_WORKER_CONTEXT["users"],
        planning_cfg=_WORKER_CONTEXT["planning_cfg"],
        radial_model_data=_WORKER_CONTEXT["radial_model_data"],
        progress=None,
    )


def _select_final_solution_with_geometry_check(
    *,
    solutions: list[EvaluatedSolution],
    corridor: Any,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    crs: Any,
    top_k: int,
) -> FinalGeometrySelection:
    """Generate final layers for top solutions and pick one that passes geometry checks."""

    ranked = sorted(solutions, key=lambda item: (not item.feasible, item.objective))
    bounded_top_k = max(1, min(int(top_k), len(ranked)))
    checked: list[FinalGeometrySelection] = []
    first_solution = ranked[0]
    for checked_index, solution in enumerate(ranked[:bounded_top_k], start=1):
        selection = _build_geometry_selection(
            solution=solution,
            corridor=corridor,
            users=users,
            dtm=dtm,
            profile=profile,
            planning_cfg=planning_cfg,
            crs=crs,
            checked_index=checked_index,
        )
        checked.append(selection)
        if bool(selection.summary.get("feasible", False)):
            _annotate_geometry_check_summary(
                summary=selection.summary,
                checked_count=len(checked),
                top_k=bounded_top_k,
                selected_after_geometry_check=solution is not first_solution,
                selected_geometry_check_index=checked_index,
            )
            return selection

    checked.sort(key=_geometry_fallback_key)
    selected = checked[0]
    _annotate_geometry_check_summary(
        summary=selected.summary,
        checked_count=len(checked),
        top_k=bounded_top_k,
        selected_after_geometry_check=selected.solution is not first_solution,
        selected_geometry_check_index=selected.checked_index,
    )
    return selected


def _build_geometry_selection(
    *,
    solution: EvaluatedSolution,
    corridor: Any,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    crs: Any,
    checked_index: int,
) -> FinalGeometrySelection:
    transformer_layer, poles_layer, planned_lines, user_connection_public = generate_plan_layers(
        corridor=corridor,
        solution=solution,
        users=users,
        dtm=dtm,
        profile=profile,
        planning_cfg=planning_cfg,
        crs=crs,
    )
    annotated_users = _annotate_users_for_solution(
        users=users,
        solution=solution,
        user_connection_public=user_connection_public,
    )
    summary = build_summary_v2(
        corridor=corridor,
        solution=solution,
        users=annotated_users,
        poles=poles_layer,
        planned_lines=planned_lines,
        planning_cfg=planning_cfg,
    )
    return FinalGeometrySelection(
        solution=solution,
        users=annotated_users,
        transformer_layer=transformer_layer,
        poles_layer=poles_layer,
        planned_lines=planned_lines,
        user_connection_public=user_connection_public,
        summary=summary,
        checked_index=checked_index,
    )


def _annotate_users_for_solution(
    *,
    users: gpd.GeoDataFrame,
    solution: EvaluatedSolution,
    user_connection_public: dict[int, str],
) -> gpd.GeoDataFrame:
    annotated = users.copy()
    annotated["assigned_phase"] = annotated["user_id"].map(
        lambda value: solution.phase_assignment.get(int(value), "")
    )
    annotated["connected_node_id"] = annotated["user_id"].map(
        lambda value: user_connection_public.get(int(value), "")
    )
    annotated["voltage_drop_pct"] = annotated["user_id"].map(
        lambda value: solution.power_flow.user_voltage_drop_pct.get(int(value), 0.0)
    )
    return annotated


def _annotate_geometry_check_summary(
    *,
    summary: dict[str, Any],
    checked_count: int,
    top_k: int,
    selected_after_geometry_check: bool,
    selected_geometry_check_index: int,
) -> None:
    summary["final_geometry_checked_solution_count"] = int(checked_count)
    summary["final_geometry_check_top_k"] = int(top_k)
    summary["selected_after_geometry_check"] = bool(selected_after_geometry_check)
    summary["selected_geometry_check_index"] = int(selected_geometry_check_index)


def _geometry_fallback_key(selection: FinalGeometrySelection) -> tuple[int, int, int, int, int, float]:
    summary = selection.summary
    reasons = summary.get("infeasible_reasons") or summary.get("infeasible_reason") or []
    line_violation_count = int(summary.get("line_totals", {}).get("violation_count", 0))
    pole_violation_count = int(summary.get("poles", {}).get("user_clearance_violation_count", 0))
    return (
        line_violation_count + pole_violation_count,
        line_violation_count,
        pole_violation_count,
        int(not selection.solution.feasible),
        len(reasons),
        float(selection.solution.objective),
    )


def _finish_detail(*, objective: float, summary: dict[str, Any]) -> str:
    if bool(summary.get("feasible", False)):
        return f"best obj={objective:.1f} | feasible"
    reasons = summary.get("infeasible_reasons") or summary.get("infeasible_reason") or []
    reason_text = ",".join(str(reason) for reason in reasons[:3]) if reasons else "unknown_reason"
    return f"best obj={objective:.1f} | infeasible | reason={reason_text}"


def _evaluate_candidate_solution(
    *,
    candidate: Any,
    choices: dict[int, Any],
    corridor: Any,
    users: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
    tree_cache: dict[tuple[str, tuple[str, ...]], Any] | None = None,
    radial_model_data: RadialTreeModelData | None = None,
    performance_counters: dict[str, int] | None = None,
) -> EvaluatedSolution:
    """Evaluate one transformer candidate plus one set of user attachments."""

    diagnostics: list[str] = []
    infeasible_reasons: list[str] = []
    for option in choices.values():
        if not option.feasible:
            diagnostics.append(f"No feasible service corridor was found for user {option.user_id}.")
            infeasible_reasons.extend(option.infeasible_reasons)
        if option.horizontal_length_m > float(planning_cfg.get("max_service_drop_m", 25.0)) + corridor.resolution_m * 2.0:
            diagnostics.append(
                f"Service drop for user {option.user_id} exceeds max length: {option.horizontal_length_m:.2f} m."
            )
            infeasible_reasons.append("service_drop_too_long")
    tree_key = _tree_cache_key(candidate_node_id=candidate.node_id, choices=choices)
    try:
        if tree_cache is not None and tree_key in tree_cache:
            tree = tree_cache[tree_key]
            if performance_counters is not None:
                performance_counters["milp_cache_hit_count"] = performance_counters.get("milp_cache_hit_count", 0) + 1
        else:
            if performance_counters is not None:
                performance_counters["milp_solve_count"] = performance_counters.get("milp_solve_count", 0) + 1
            tree = solve_radial_tree(
                corridor=corridor,
                root_node_id=candidate.node_id,
                attachment_node_ids=[option.attach_node_id for option in choices.values()],
                planning_cfg=planning_cfg,
                model_data=radial_model_data,
            )
            if tree_cache is not None:
                tree_cache[tree_key] = tree
    except ValueError as exc:
        from src.planning.models import PowerFlowResult, RadialTreeResult

        diagnostics.append(str(exc))
        infeasible_reasons.append("radial_tree_infeasible")
        tree = RadialTreeResult(
            root_node_id=candidate.node_id,
            selected_edge_ids=[],
            parent_by_node={},
            depth_by_node={candidate.node_id: 0},
            terminal_nodes={candidate.node_id},
        )
        power_flow = PowerFlowResult(
            edge_phase_loads={},
            edge_phase_kw={},
            edge_losses_kw={},
            edge_voltage_drop_pct={},
            transformer_phase_loads=np.zeros(3, dtype=float),
            user_voltage_drop_pct={},
            user_service_drop_pct={},
            user_connection_nodes={},
            total_loss_kw=0.0,
            max_voltage_drop_pct=0.0,
        )
        return EvaluatedSolution(
            transformer_candidate=candidate,
            radial_tree=tree,
            attachment_choices=dict(choices),
            phase_assignment={},
            power_flow=power_flow,
            build_cost=float("inf"),
            loss_cost=0.0,
            total_unbalance_penalty=0.0,
            objective=float("inf"),
            feasible=False,
            voltage_ok=False,
            diagnostics=diagnostics,
            infeasible_reasons=_dedupe_reasons(infeasible_reasons),
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
    voltage_ok, voltage_diagnostics, voltage_reasons = evaluate_solution_feasibility(
        users=users,
        power_flow=power_flow,
        planning_cfg=planning_cfg,
    )
    diagnostics.extend(voltage_diagnostics)
    infeasible_reasons.extend(voltage_reasons)

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
        infeasible_reasons=_dedupe_reasons(infeasible_reasons),
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
        "tx_candidate_count": 60,
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
        "pole_user_clearance_m": 5.0,
        "line_user_clearance_m": 1.0,
        "voltage_drop_max_pct": 7.0,
        "phase_balance_target_ratio": 0.10,
        "phase_balance_max_ratio": 0.15,
        "milp_time_limit_s": 8,
        "solver_backend": "highs",
        "mip_gap": 0.15,
        "parallel_candidate_eval": False,
        "parallel_workers": 1,
        "highs_threads_per_worker": 1,
        "local_search_top_k": 1,
        "local_search_patience": 2,
        "local_search_top_options": 1,
        "local_search_max_full_evals": 8,
        "emit_performance_metrics": True,
        "alns_max_iter": 20,
        "alns_destroy_ratio": 0.15,
        "candidate_solution_pool_size": 6,
        "final_geometry_check_top_k": 6,
        "show_progress": True,
        "progress_bar_width": 32,
    }
    merged = dict(planning_defaults)
    merged.update(config.get("planning", {}))
    merged.update(config.get("planning_v2", {}))
    return merged


def _tree_cache_key(*, candidate_node_id: str, choices: dict[int, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the stable cache key used for one root plus attachment node set."""

    return (str(candidate_node_id), tuple(sorted({str(option.attach_node_id) for option in choices.values()})))


def _resolve_parallel_workers(*, planning_cfg: dict[str, Any], task_count: int) -> int:
    """Resolve the candidate-evaluation worker count from config and CPU count."""

    if task_count <= 1 or not bool(planning_cfg.get("parallel_candidate_eval", True)):
        return 1
    configured = int(planning_cfg.get("parallel_workers", 0))
    if configured > 0:
        return max(1, min(int(task_count), configured))
    cpu_count = os.cpu_count() or 1
    auto_workers = max(1, cpu_count - 2)
    return max(1, min(int(task_count), auto_workers))


def _use_parallel(*, planning_cfg: dict[str, Any], worker_count: int) -> bool:
    """Return whether process-based candidate parallelism should be used."""

    return (
        bool(planning_cfg.get("parallel_candidate_eval", True))
        and int(worker_count) > 1
        and "parallel_fallback_reason" not in planning_cfg
    )


def _build_performance_summary(
    *,
    total_duration_s: float,
    corridor_duration_s: float,
    prefilter_duration_s: float,
    initial_duration_s: float,
    local_search_duration_s: float,
    output_duration_s: float,
    worker_count: int,
    initial_results: list[CandidateEvaluationResult],
    improved_results: list[CandidateEvaluationResult],
    local_search_top_k: int,
    planning_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build stable timing and solver-count metrics for optimization_summary.json."""

    all_results = [*initial_results, *improved_results]
    return {
        "total_duration_s": round(float(total_duration_s), 3),
        "corridor_build_duration_s": round(float(corridor_duration_s), 3),
        "candidate_prefilter_duration_s": round(float(prefilter_duration_s), 3),
        "initial_candidate_eval_duration_s": round(float(initial_duration_s), 3),
        "local_search_duration_s": round(float(local_search_duration_s), 3),
        "output_generation_duration_s": round(float(output_duration_s), 3),
        "parallel_candidate_eval": bool(planning_cfg.get("parallel_candidate_eval", True)),
        "worker_count": int(worker_count),
        "parallel_fallback_reason": str(planning_cfg.get("parallel_fallback_reason", "")),
        "highs_threads_per_worker": int(planning_cfg.get("highs_threads_per_worker", 1)),
        "initial_candidate_count": int(len(initial_results)),
        "local_search_candidate_count": int(len(improved_results)),
        "local_search_top_k": int(local_search_top_k),
        "milp_solve_count": int(sum(result.milp_solve_count for result in all_results)),
        "milp_cache_hit_count": int(sum(result.milp_cache_hit_count for result in all_results)),
        "local_search_trial_count": int(sum(result.local_search_trial_count for result in all_results)),
        "local_search_full_eval_count": int(sum(result.local_search_full_eval_count for result in all_results)),
        "local_search_patience": int(planning_cfg.get("local_search_patience", 8)),
        "local_search_top_options": int(planning_cfg.get("local_search_top_options", 5)),
        "local_search_max_full_evals": int(planning_cfg.get("local_search_max_full_evals", 200)),
        "slowest_candidate_duration_s": round(max((result.duration_s for result in all_results), default=0.0), 3),
    }


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    """Return reason codes once while preserving their original order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if not reason or reason in seen:
            continue
        seen.add(reason)
        ordered.append(reason)
    return ordered
