from __future__ import annotations

import math
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

    phase_voltage = float(planning_cfg.get("low_voltage_phase_v", 230.0))
    line_r_ohm_per_km = float(planning_cfg.get("line_resistance_ohm_per_km", 0.6))
    line_x_ohm_per_km = float(planning_cfg.get("line_reactance_ohm_per_km", 0.35))
    edge_losses_kw: dict[tuple[str, str], float] = {}
    edge_voltage_drop_pct: dict[tuple[str, str], np.ndarray] = {}
    total_loss_kw = 0.0

    for edge, load in edge_phase_loads.items():
        parent, child = edge
        edge_id = corridor.graph[parent][child]["edge_id"]
        corridor_edge = corridor.edges[edge_id]
        length_km = corridor_edge.length_3d_m / 1000.0
        resistance = line_r_ohm_per_km * length_km
        reactance = line_x_ohm_per_km * length_km
        phase_kw = edge_phase_kw.get(edge, np.zeros(3, dtype=float))
        phase_pf = np.divide(phase_kw, np.maximum(load, 1e-9), out=np.zeros(3, dtype=float), where=np.maximum(load, 1e-9) > 0)
        phase_pf = np.clip(phase_pf, 0.0, 1.0)
        phase_sin = np.sqrt(np.maximum(1.0 - phase_pf**2, 0.0))
        currents = load * 1000.0 / max(phase_voltage, 1.0)
        edge_losses_kw[edge] = float(np.sum((currents**2) * resistance) / 1000.0)
        edge_voltage_drop_pct[edge] = (
            currents * (resistance * phase_pf + reactance * phase_sin) * 100.0 / max(phase_voltage, 1.0)
        )
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

        path_edges = path_to_root(tree, option.attach_node_id)
        if phase == "ABC":
            path_drop = sum(float(np.max(edge_voltage_drop_pct.get(edge, np.zeros(3, dtype=float)))) for edge in path_edges)
            service_drop = _service_drop_pct(
                apparent_kva=float(row.apparent_kva) / 3.0,
                load_kw=float(row.load_kw) / 3.0,
                length_m=option.length_3d_m,
                phase_voltage=phase_voltage,
                line_r_ohm_per_km=line_r_ohm_per_km,
                line_x_ohm_per_km=line_x_ohm_per_km,
            )
        else:
            phase_index = PHASE_INDEX[phase]
            path_drop = sum(float(edge_voltage_drop_pct.get(edge, np.zeros(3, dtype=float))[phase_index]) for edge in path_edges)
            service_drop = _service_drop_pct(
                apparent_kva=float(row.apparent_kva),
                load_kw=float(row.load_kw),
                length_m=option.length_3d_m,
                phase_voltage=phase_voltage,
                line_r_ohm_per_km=line_r_ohm_per_km,
                line_x_ohm_per_km=line_x_ohm_per_km,
            )
        user_service_drop_pct[user_id] = float(service_drop)
        user_voltage_drop_pct[user_id] = float(path_drop + service_drop)
        user_connection_nodes[user_id] = option.attach_node_id
        total_loss_kw += _service_drop_loss_kw(
            apparent_kva=float(row.apparent_kva if phase != "ABC" else row.apparent_kva / 3.0),
            length_m=option.length_3d_m,
            phase_voltage=phase_voltage,
            line_r_ohm_per_km=line_r_ohm_per_km,
        )

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
    length_m: float,
    phase_voltage: float,
    line_r_ohm_per_km: float,
    line_x_ohm_per_km: float,
) -> float:
    """Return the voltage drop percentage of one service drop."""

    current = apparent_kva * 1000.0 / max(phase_voltage, 1.0)
    pf = 0.0 if apparent_kva <= 0.0 else min(max(load_kw / apparent_kva, 0.0), 1.0)
    sin_phi = math.sqrt(max(1.0 - pf**2, 0.0))
    resistance = line_r_ohm_per_km * (length_m / 1000.0)
    reactance = line_x_ohm_per_km * (length_m / 1000.0)
    return float(current * (resistance * pf + reactance * sin_phi) * 100.0 / max(phase_voltage, 1.0))


def _service_drop_loss_kw(
    *,
    apparent_kva: float,
    length_m: float,
    phase_voltage: float,
    line_r_ohm_per_km: float,
) -> float:
    """Return the ohmic loss of one service drop."""

    current = apparent_kva * 1000.0 / max(phase_voltage, 1.0)
    resistance = line_r_ohm_per_km * (length_m / 1000.0)
    return float((current**2) * resistance / 1000.0)
