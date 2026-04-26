from __future__ import annotations

from typing import Any

import numpy as np


def recommend_transformer_capacity(
    *,
    transformer_phase_loads: np.ndarray,
    planning_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Recommend transformer capacity after optimization."""

    raw_loading_kva = float(np.asarray(transformer_phase_loads, dtype=float).sum())
    demand_factor = float(planning_cfg.get("demand_factor", 1.0))
    target_loading_ratio = float(planning_cfg.get("transformer_target_loading_ratio", 0.80))

    effective_loading_kva = raw_loading_kva * demand_factor
    required_capacity_kva = effective_loading_kva / max(target_loading_ratio, 1e-9)

    standard_capacities = [
        float(value)
        for value in planning_cfg.get(
            "transformer_standard_capacities_kva",
            [100, 160, 200, 250, 315, 400, 500, 630, 800],
        )
    ]

    recommended_capacity = None
    for capacity in sorted(standard_capacities):
        if capacity + 1e-9 >= required_capacity_kva:
            recommended_capacity = capacity
            break

    if recommended_capacity is None:
        recommended_capacity = max(standard_capacities) if standard_capacities else required_capacity_kva

    recommended_loading_ratio = effective_loading_kva / max(recommended_capacity, 1e-9)

    return {
        "capacity_enforced_during_optimization": False,
        "raw_loading_kva": round(raw_loading_kva, 3),
        "demand_factor": round(demand_factor, 4),
        "effective_loading_kva": round(effective_loading_kva, 3),
        "target_loading_ratio": round(target_loading_ratio, 4),
        "required_capacity_kva": round(required_capacity_kva, 3),
        "recommended_capacity_kva": round(recommended_capacity, 3),
        "recommended_loading_ratio": round(recommended_loading_ratio, 5),
        "standard_capacities_kva": standard_capacities,
    }
