from __future__ import annotations

from typing import Any

import numpy as np

from src.planning.models import AttachmentOption, CorridorGraph, PHASE_INDEX, PowerFlowResult, RadialTreeResult
from src.planning.radial_tree_milp import path_to_root


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
    edge_phase_currents_a: dict[tuple[str, str], np.ndarray] = {}
    total_loss_kw = 0.0
    phase_voltage = float(planning_cfg.get("low_voltage_phase_v", 230.0))
    resistance_ohm_per_km = float(planning_cfg.get("line_resistance_ohm_per_km", 0.642))
    reactance_ohm_per_km = float(planning_cfg.get("line_reactance_ohm_per_km", 0.083))

    for edge in edge_phase_loads:
        load = np.asarray(edge_phase_loads[edge], dtype=float)
        kw = np.asarray(edge_phase_kw.get(edge, np.zeros(3, dtype=float)), dtype=float)
        currents = load * 1000.0 / max(phase_voltage, 1.0)
        edge_phase_currents_a[edge] = currents.astype(float)

        parent, child = edge
        edge_id = str(corridor.graph[parent][child]["edge_id"])
        length_km = float(corridor.edges[edge_id].length_3d_m) / 1000.0
        resistance_ohm = resistance_ohm_per_km * length_km
        reactance_ohm = reactance_ohm_per_km * length_km

        power_factor = np.divide(
            kw,
            load,
            out=np.zeros(3, dtype=float),
            where=load > 1e-9,
        )
        power_factor = np.clip(power_factor, 0.0, 1.0)
        reactive_factor = np.sqrt(np.maximum(1.0 - power_factor**2, 0.0))

        voltage_drop_v = currents * (resistance_ohm * power_factor + reactance_ohm * reactive_factor)
        edge_voltage_drop_pct[edge] = 100.0 * voltage_drop_v / max(phase_voltage, 1.0)
        edge_losses_kw[edge] = float(np.sum(currents**2 * resistance_ohm) / 1000.0)
        total_loss_kw += edge_losses_kw[edge]

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
        line_drop = 0.0
        for edge in path_to_root(tree, option.attach_node_id):
            edge_drop = edge_voltage_drop_pct.get(edge, np.zeros(3, dtype=float))
            if phase == "ABC":
                line_drop += float(np.max(edge_drop))
            else:
                line_drop += float(edge_drop[PHASE_INDEX[phase]])

        user_service_drop_pct[user_id] = float(service_drop)
        user_voltage_drop_pct[user_id] = float(line_drop + service_drop)
        user_connection_nodes[user_id] = option.attach_node_id

    max_voltage_drop_pct = max(user_voltage_drop_pct.values(), default=0.0)
    return PowerFlowResult(
        edge_phase_loads=edge_phase_loads,
        edge_phase_kw=edge_phase_kw,
        edge_losses_kw=edge_losses_kw,
        edge_voltage_drop_pct=edge_voltage_drop_pct,
        edge_phase_currents_a=edge_phase_currents_a,
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
