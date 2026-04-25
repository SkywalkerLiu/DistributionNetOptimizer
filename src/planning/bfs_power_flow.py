from __future__ import annotations

from typing import Any

import numpy as np

from src.planning.models import AttachmentOption, CorridorGraph, PHASE_INDEX, PowerFlowResult, RadialTreeResult


def run_backward_forward_sweep(
    *,
    corridor: CorridorGraph,
    tree: RadialTreeResult,
    attachment_choices: dict[int, AttachmentOption],
    assignments: dict[int, str],
    edge_phase_loads: dict[tuple[str, str], np.ndarray],
    edge_phase_kw: dict[tuple[str, str], np.ndarray],
    users: Any,
    planning_cfg: dict[str, Any],
) -> PowerFlowResult:
    """Evaluate the radial tree with a planning-grade backward/forward sweep."""

    edge_losses_kw: dict[tuple[str, str], float] = {}
    edge_voltage_drop_pct: dict[tuple[str, str], np.ndarray] = {}
    total_loss_kw = 0.0

    for edge in edge_phase_loads:
        edge_losses_kw[edge] = 0.0
        edge_voltage_drop_pct[edge] = np.zeros(3, dtype=float)

    user_by_id = {int(row.user_id): row for row in users.itertuples()}
    transformer_phase_loads = np.zeros(3, dtype=float)
    user_voltage_drop_pct: dict[int, float] = {}
    user_service_drop_pct: dict[int, float] = {}
    user_connection_nodes: dict[int, str] = {}

    for user_id, option in attachment_choices.items():
        row = user_by_id[user_id]
        phase = assignments[user_id]
        if phase == "ABC":
            transformer_phase_loads += float(row.apparent_kva) / 3.0
        else:
            transformer_phase_loads[PHASE_INDEX[phase]] += float(row.apparent_kva)

        if phase == "ABC":
            service_drop = _service_drop_pct(
                apparent_kva=float(row.apparent_kva) / 3.0,
                load_kw=float(row.load_kw) / 3.0,
            )
        else:
            service_drop = _service_drop_pct(
                apparent_kva=float(row.apparent_kva),
                load_kw=float(row.load_kw),
            )
        user_service_drop_pct[user_id] = float(service_drop)
        user_voltage_drop_pct[user_id] = float(service_drop)
        user_connection_nodes[user_id] = option.attach_node_id

    max_voltage_drop_pct = max(user_voltage_drop_pct.values(), default=0.0)
    return PowerFlowResult(
        edge_phase_loads=edge_phase_loads,
        edge_phase_kw=edge_phase_kw,
        edge_losses_kw=edge_losses_kw,
        edge_voltage_drop_pct=edge_voltage_drop_pct,
        transformer_phase_loads=transformer_phase_loads,
        user_voltage_drop_pct=user_voltage_drop_pct,
        user_service_drop_pct=user_service_drop_pct,
        user_connection_nodes=user_connection_nodes,
        total_loss_kw=float(total_loss_kw),
        max_voltage_drop_pct=float(max_voltage_drop_pct),
    )


def _service_drop_pct(
    *,
    apparent_kva: float,
    load_kw: float,
) -> float:
    """Return the user power-factor voltage-drop percentage."""

    pf = 0.0 if apparent_kva <= 0.0 else min(max(load_kw / apparent_kva, 0.0), 1.0)
    return float((1.0 - pf) * 100.0)
