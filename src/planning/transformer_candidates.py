from __future__ import annotations

from typing import Any

import numpy as np

from src.planning.models import CorridorGraph, TransformerCandidate


def generate_transformer_candidates(
    *,
    corridor: CorridorGraph,
    users: Any,
    slope: np.ndarray,
    roughness: np.ndarray,
    planning_cfg: dict[str, Any],
) -> list[TransformerCandidate]:
    """Rank corridor nodes and keep the best transformer root candidates."""

    candidate_nodes = [
        node for node in corridor.nodes.values() if node.kind in {"junction", "cluster", "attach"}
    ]
    if not candidate_nodes:
        raise ValueError("The corridor graph does not contain transformer candidate nodes.")

    user_xy = np.column_stack([users.geometry.x.to_numpy(dtype=float), users.geometry.y.to_numpy(dtype=float)])
    user_weights = users["apparent_kva"].to_numpy(dtype=float)
    user_weight_total = max(float(user_weights.sum()), 1.0)
    ranked: list[TransformerCandidate] = []

    for node in candidate_nodes:
        distance = np.sqrt((user_xy[:, 0] - node.x) ** 2 + (user_xy[:, 1] - node.y) ** 2)
        weighted_distance = float((distance * user_weights).sum() / user_weight_total)
        terrain_score = float(slope[node.row, node.col]) * 3.0 + float(roughness[node.row, node.col]) * 2.0
        corridor_score = 0.0
        if corridor.graph.degree(node.node_id) > 0:
            corridor_score = 20.0 / float(corridor.graph.degree(node.node_id))
        boundary_distance = float(corridor.boundary_distance_m[node.row, node.col])
        boundary_penalty = max(0.0, float(planning_cfg.get("corridor_safe_margin_m", 12.0)) - boundary_distance)
        ranked.append(
            TransformerCandidate(
                node_id=node.node_id,
                x=node.x,
                y=node.y,
                z=node.z,
                score=weighted_distance + terrain_score + corridor_score + boundary_penalty,
                rank=0,
            )
        )

    ranked.sort(key=lambda item: item.score)
    keep = max(
        1,
        min(
            int(planning_cfg.get("tx_prefilter_top_k", 6)),
            int(planning_cfg.get("tx_candidate_count", 20)),
            len(ranked),
        ),
    )
    selected = ranked[:keep]
    for rank, candidate in enumerate(selected, start=1):
        candidate.rank = rank
    return selected
