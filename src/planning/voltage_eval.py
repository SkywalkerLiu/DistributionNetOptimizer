from __future__ import annotations

from typing import Any

import numpy as np

from src.planning.models import PowerFlowResult


def evaluate_solution_feasibility(
    *,
    users: Any,
    power_flow: PowerFlowResult,
    planning_cfg: dict[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """Evaluate hard planning constraints on the candidate solution."""

    diagnostics: list[str] = []
    infeasible_reasons: list[str] = []
    capacity_limit = float(planning_cfg.get("transformer_capacity_kva", 630.0)) * float(
        planning_cfg.get("max_loading_ratio", 1.0)
    )
    transformer_load = float(power_flow.transformer_phase_loads.sum())
    if transformer_load > capacity_limit + 1e-9:
        diagnostics.append(
            f"Transformer loading {transformer_load:.2f} kVA exceeds limit {capacity_limit:.2f} kVA."
        )
        infeasible_reasons.append("transformer_overloaded")

    voltage_limit = float(planning_cfg.get("voltage_drop_max_pct", 7.0))
    if power_flow.max_voltage_drop_pct > voltage_limit + 1e-9:
        diagnostics.append(
            f"Maximum voltage drop {power_flow.max_voltage_drop_pct:.2f}% exceeds limit {voltage_limit:.2f}%."
        )
        infeasible_reasons.append("voltage_drop_exceeded")

    balance_limit = float(planning_cfg.get("phase_balance_max_ratio", 0.15))
    imbalance = phase_unbalance_ratio(power_flow.transformer_phase_loads)
    if imbalance > balance_limit + 1e-9:
        diagnostics.append(
            f"Transformer phase imbalance {imbalance:.4f} exceeds limit {balance_limit:.4f}."
        )
        infeasible_reasons.append("phase_unbalance_exceeded")
    return len(diagnostics) == 0, diagnostics, infeasible_reasons


def phase_unbalance_ratio(load: np.ndarray) -> float:
    """Return the standard max-deviation phase imbalance ratio."""

    total = float(load.sum())
    if total <= 0.0:
        return 0.0
    average = total / 3.0
    return float(np.max(np.abs(load - average)) / max(average, 1e-9))


def phase_unbalance_penalty(
    *,
    power_flow: PowerFlowResult,
    planning_cfg: dict[str, Any],
) -> float:
    """Return the weighted V2 phase-balance penalty."""

    tx_ratio = phase_unbalance_ratio(power_flow.transformer_phase_loads)
    edge_ratios = [
        phase_unbalance_ratio(load)
        for load in power_flow.edge_phase_loads.values()
        if float(load.sum()) > 0.0
    ]
    segment_ratio = float(np.mean(edge_ratios)) if edge_ratios else 0.0
    combined = (
        float(planning_cfg.get("tx_unbalance_weight", 2.0)) * tx_ratio
        + float(planning_cfg.get("segment_unbalance_weight", 1.0)) * segment_ratio
    )
    return float(planning_cfg.get("phase_unbalance_weight", 3.0)) * combined * 10000.0
