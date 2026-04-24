from __future__ import annotations

from typing import Any


def loss_cost_from_power(
    *,
    total_loss_kw: float,
    planning_cfg: dict[str, Any],
) -> float:
    """Convert instantaneous loss power into an annualized planning cost proxy."""

    equivalent_hours = float(planning_cfg.get("loss_cost_hours", 8760.0))
    energy_price = float(planning_cfg.get("loss_energy_price_per_kwh", 1.0))
    return float(total_loss_kw) * equivalent_hours * energy_price

