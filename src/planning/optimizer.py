from __future__ import annotations

from src.planning.optimizer_v2 import optimize_distribution_network_v2


def optimize_distribution_network(**kwargs):
    """Project-level optimizer entrypoint backed exclusively by the V2 algorithm."""

    return optimize_distribution_network_v2(**kwargs)
