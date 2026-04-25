from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np

from src.planning.models import CorridorGraph, EvaluatedSolution
from src.planning.voltage_eval import phase_unbalance_ratio


def build_summary_v2(
    *,
    corridor: CorridorGraph,
    solution: EvaluatedSolution,
    users: gpd.GeoDataFrame,
    poles: gpd.GeoDataFrame,
    planned_lines: gpd.GeoDataFrame,
    planning_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the optimization summary emitted by the V2 pipeline."""

    phase_load = solution.power_flow.transformer_phase_loads
    non_service = planned_lines.loc[planned_lines["line_type"] == "lv_line"]
    shared_unbalance = []
    for row in non_service.itertuples():
        load = np.asarray([float(row.load_a_kva), float(row.load_b_kva), float(row.load_c_kva)], dtype=float)
        if float(load.sum()) <= 0.0:
            continue
        shared_unbalance.append(phase_unbalance_ratio(load))
    diagnostics = list(solution.diagnostics)
    infeasible_reasons = _dedupe_reasons(list(solution.infeasible_reasons))
    line_violation_count = int(planned_lines["is_violation"].sum()) if not planned_lines.empty else 0
    line_vertical_violation_count = _line_vertical_violation_count(planned_lines)
    line_user_violation_count = _line_user_violation_count(planned_lines)
    pole_user_violation_count = _point_user_violation_count(poles)
    if line_vertical_violation_count:
        diagnostics.append(f"{line_vertical_violation_count} line spans violate configured vertical clearance limits.")
        infeasible_reasons.append("line_vertical_clearance_exceeded")
    if line_user_violation_count:
        diagnostics.append(f"{line_user_violation_count} line spans violate configured user horizontal clearance limits.")
        infeasible_reasons.append("line_user_clearance_exceeded")
    if pole_user_violation_count:
        diagnostics.append(f"{pole_user_violation_count} poles violate configured user horizontal clearance limits.")
        infeasible_reasons.append("pole_user_clearance_exceeded")
    infeasible_reasons = _dedupe_reasons(infeasible_reasons)
    feasible = bool(
        solution.feasible
        and line_violation_count == 0
        and pole_user_violation_count == 0
    )
    emitted_reasons = [] if feasible else infeasible_reasons

    return {
        "algorithm": "planning_v2",
        "feasible": feasible,
        "infeasible_reasons": emitted_reasons,
        "infeasible_reason": emitted_reasons,
        "diagnostics": diagnostics,
        "top_diagnostics": diagnostics[:5],
        "violation_examples": _violation_examples(planned_lines=planned_lines, poles=poles),
        "objective": {
            "total": round(float(solution.objective), 3),
            "build_cost": round(float(solution.build_cost), 3),
            "loss_cost": round(float(solution.loss_cost), 3),
            "unbalance_penalty": round(float(solution.total_unbalance_penalty), 3),
        },
        "corridor": {
            "node_count": int(len(corridor.nodes)),
            "edge_count": int(len(corridor.edges)),
            "selected_edge_count": int(len(solution.radial_tree.selected_edge_ids)),
        },
        "transformer": {
            "id": "TX1",
            "candidate_rank": int(solution.transformer_candidate.rank),
            "capacity_kva": float(planning_cfg.get("transformer_capacity_kva", 630.0)),
            "loading_kva": round(float(phase_load.sum()), 3),
            "loading_ratio": round(
                float(phase_load.sum()) / max(float(planning_cfg.get("transformer_capacity_kva", 630.0)), 1.0),
                5,
            ),
            "x": round(float(solution.transformer_candidate.x), 3),
            "y": round(float(solution.transformer_candidate.y), 3),
            "elev_m": round(float(solution.transformer_candidate.z), 3),
        },
        "line_totals": {
            "count": int(len(planned_lines)),
            "total_3d_length_m": round(float(planned_lines["length_3d_m"].sum()) if not planned_lines.empty else 0.0, 3),
            "total_horizontal_length_m": round(
                float(planned_lines["horizontal_length_m"].sum()) if not planned_lines.empty else 0.0,
                3,
            ),
            "violation_count": line_violation_count,
            "vertical_clearance_violation_count": line_vertical_violation_count,
            "user_clearance_violation_count": line_user_violation_count,
        },
        "poles": {
            "count": int(len(poles)),
            "user_clearance_violation_count": pole_user_violation_count,
        },
        "phase_balance": {
            "load_a_kva": round(float(phase_load[0]), 3),
            "load_b_kva": round(float(phase_load[1]), 3),
            "load_c_kva": round(float(phase_load[2]), 3),
            "transformer_unbalance_ratio": round(float(phase_unbalance_ratio(phase_load)), 5),
            "mean_shared_lv_line_unbalance_ratio": round(float(np.mean(shared_unbalance)) if shared_unbalance else 0.0, 5),
            "max_shared_lv_line_unbalance_ratio": round(float(np.max(shared_unbalance)) if shared_unbalance else 0.0, 5),
        },
        "losses": {
            "total_loss_kw": round(float(solution.power_flow.total_loss_kw), 5),
        },
        "voltage": {
            "max_voltage_drop_pct": round(float(solution.power_flow.max_voltage_drop_pct), 5),
        },
        "path_diagnostics": solution.extra_metrics.get("path_diagnostics", {}),
        "root_feeder_diagnostics": solution.extra_metrics.get("root_feeder_diagnostics", {}),
        "users": {
            "count": int(len(users)),
            "connected_count": int(users["connected_node_id"].astype(str).ne("").sum()) if "connected_node_id" in users.columns else 0,
        },
    }


def _line_vertical_violation_count(planned_lines: gpd.GeoDataFrame) -> int:
    """Count line spans whose vertical clearance is below the configured requirement."""

    if planned_lines.empty or "min_clearance_m" not in planned_lines.columns:
        return 0
    return int((planned_lines["min_clearance_m"] + 1e-9 < planned_lines["required_clearance_m"]).sum())


def _line_user_violation_count(planned_lines: gpd.GeoDataFrame) -> int:
    """Count line spans whose horizontal clearance to users is below the configured requirement."""

    if planned_lines.empty or "user_clearance_m" not in planned_lines.columns:
        return 0
    clearance = planned_lines["user_clearance_m"].astype(float)
    required = planned_lines["required_user_clearance_m"].astype(float)
    applicable = clearance >= 0.0
    return int(((clearance + 1e-9 < required) & applicable).sum())


def _point_user_violation_count(points: gpd.GeoDataFrame) -> int:
    """Count poles/point assets whose horizontal clearance to users is too small."""

    if points.empty or "user_clearance_m" not in points.columns:
        return 0
    clearance = points["user_clearance_m"].astype(float)
    required = points["required_user_clearance_m"].astype(float)
    applicable = clearance >= 0.0
    return int(((clearance + 1e-9 < required) & applicable).sum())


def _violation_examples(
    *,
    planned_lines: gpd.GeoDataFrame,
    poles: gpd.GeoDataFrame,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Return compact examples that help identify final geometry violations."""

    return {
        "line_vertical_clearance": _line_vertical_violation_examples(planned_lines, limit=limit),
        "line_user_clearance": _line_user_violation_examples(planned_lines, limit=limit),
        "pole_user_clearance": _pole_user_violation_examples(poles, limit=limit),
    }


def _line_vertical_violation_examples(planned_lines: gpd.GeoDataFrame, *, limit: int) -> list[dict[str, Any]]:
    if planned_lines.empty or not {"line_id", "min_clearance_m", "required_clearance_m"}.issubset(planned_lines.columns):
        return []
    rows = planned_lines.loc[
        planned_lines["min_clearance_m"].astype(float) + 1e-9
        < planned_lines["required_clearance_m"].astype(float)
    ].head(limit)
    return [
        {
            "line_id": _json_scalar(row.line_id),
            "line_type": str(getattr(row, "line_type", "")),
            "min_clearance_m": round(float(row.min_clearance_m), 3),
            "required_clearance_m": round(float(row.required_clearance_m), 3),
        }
        for row in rows.itertuples()
    ]


def _line_user_violation_examples(planned_lines: gpd.GeoDataFrame, *, limit: int) -> list[dict[str, Any]]:
    required_columns = {"line_id", "user_clearance_m", "required_user_clearance_m"}
    if planned_lines.empty or not required_columns.issubset(planned_lines.columns):
        return []
    clearance = planned_lines["user_clearance_m"].astype(float)
    required = planned_lines["required_user_clearance_m"].astype(float)
    rows = planned_lines.loc[((clearance + 1e-9 < required) & (clearance >= 0.0))].head(limit)
    return [
        {
            "line_id": _json_scalar(row.line_id),
            "line_type": str(getattr(row, "line_type", "")),
            "user_clearance_m": round(float(row.user_clearance_m), 3),
            "required_user_clearance_m": round(float(row.required_user_clearance_m), 3),
        }
        for row in rows.itertuples()
    ]


def _pole_user_violation_examples(poles: gpd.GeoDataFrame, *, limit: int) -> list[dict[str, Any]]:
    required_columns = {"pole_id", "user_clearance_m", "required_user_clearance_m"}
    if poles.empty or not required_columns.issubset(poles.columns):
        return []
    clearance = poles["user_clearance_m"].astype(float)
    required = poles["required_user_clearance_m"].astype(float)
    rows = poles.loc[((clearance + 1e-9 < required) & (clearance >= 0.0))].head(limit)
    return [
        {
            "pole_id": str(row.pole_id),
            "user_clearance_m": round(float(row.user_clearance_m), 3),
            "required_user_clearance_m": round(float(row.required_user_clearance_m), 3),
        }
        for row in rows.itertuples()
    ]


def _json_scalar(value: Any) -> int | float | str:
    """Convert pandas/numpy scalar values into JSON-friendly primitives."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


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
