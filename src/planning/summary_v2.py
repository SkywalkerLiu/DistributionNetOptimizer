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
    violation_count = int(planned_lines["is_violation"].sum()) if not planned_lines.empty else 0
    if violation_count:
        diagnostics.append(f"{violation_count} line spans violate configured clearance limits.")

    return {
        "algorithm": "planning_v2",
        "feasible": bool(solution.feasible and violation_count == 0),
        "diagnostics": diagnostics,
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
            "violation_count": violation_count,
        },
        "poles": {"count": int(len(poles))},
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
        "users": {
            "count": int(len(users)),
            "connected_count": int(users["connected_node_id"].astype(str).ne("").sum()) if "connected_node_id" in users.columns else 0,
        },
    }

