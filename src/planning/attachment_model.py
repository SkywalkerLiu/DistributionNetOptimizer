from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from src.planning.common import point_metrics
from src.planning.geometry_constraints import build_user_point_map, line_min_user_clearance, segment_is_feasible
from src.planning.models import AttachmentOption, CorridorGraph


def build_attachment_options(
    *,
    corridor: CorridorGraph,
    users: Any,
    planning_cfg: dict[str, Any],
    profile: dict[str, Any],
) -> dict[int, list[AttachmentOption]]:
    """Build feasible user-to-corridor attachment options."""

    node_ids = list(corridor.nodes)
    if not node_ids:
        return {int(row.user_id): [] for row in users.itertuples()}
    node_xy = np.asarray([(corridor.nodes[node_id].x, corridor.nodes[node_id].y) for node_id in node_ids], dtype=float)
    tree = cKDTree(node_xy)
    allowed_mask = corridor.corridor_mask.astype(np.uint8)
    max_service_drop = float(planning_cfg.get("max_service_drop_m", 25.0))
    line_user_clearance_m = float(planning_cfg.get("line_user_clearance_m", 1.0))
    user_points = build_user_point_map(users)
    query_k = min(len(node_ids), max(6, int(planning_cfg.get("candidate_solution_pool_size", 10))))
    options_by_user: dict[int, list[AttachmentOption]] = {}

    for row in users.itertuples():
        user_id = int(row.user_id)
        distances, indices = tree.query([float(row.geometry.x), float(row.geometry.y)], k=query_k)
        if np.isscalar(indices):
            indices = np.asarray([indices], dtype=int)
            distances = np.asarray([distances], dtype=float)
        candidates: list[AttachmentOption] = []
        for index in np.asarray(indices, dtype=int):
            node_id = node_ids[int(index)]
            node = corridor.nodes[node_id]
            line = LineString([(float(row.geometry.x), float(row.geometry.y)), (node.x, node.y)])
            if not segment_is_feasible(
                float(row.geometry.x),
                float(row.geometry.y),
                node.x,
                node.y,
                allowed_mask=allowed_mask,
                profile=profile,
                sample_step_m=max(corridor.resolution_m, 2.0),
            ):
                continue
            if (
                line_user_clearance_m > 0.0
                and line_min_user_clearance(line=line, user_points=user_points, exclude_user_id=user_id)
                + 1e-9
                < line_user_clearance_m
            ):
                continue
            metrics = point_metrics(
                {"x": float(row.geometry.x), "y": float(row.geometry.y), "z": float(row.elev_m)},
                {"x": node.x, "y": node.y, "z": node.z},
            )
            if metrics["horizontal_length_m"] > max_service_drop + corridor.resolution_m * 2.0:
                continue
            candidates.append(
                AttachmentOption(
                    user_id=user_id,
                    attach_node_id=node_id,
                    horizontal_length_m=float(metrics["horizontal_length_m"]),
                    length_3d_m=float(metrics["length_3d_m"]),
                    cost=float(metrics["length_3d_m"]) * float(planning_cfg.get("service_line_cost_per_m", 35.0)),
                )
            )
        if not candidates:
            node_id = node_ids[int(np.asarray(indices, dtype=int)[0])]
            node = corridor.nodes[node_id]
            line = LineString([(float(row.geometry.x), float(row.geometry.y)), (node.x, node.y)])
            metrics = point_metrics(
                {"x": float(row.geometry.x), "y": float(row.geometry.y), "z": float(row.elev_m)},
                {"x": node.x, "y": node.y, "z": node.z},
            )
            reasons = [f"no_corridor_to_user_{user_id}"]
            if metrics["horizontal_length_m"] > max_service_drop + corridor.resolution_m * 2.0:
                reasons.append("service_drop_too_long")
            if (
                line_user_clearance_m > 0.0
                and line_min_user_clearance(line=line, user_points=user_points, exclude_user_id=user_id)
                + 1e-9
                < line_user_clearance_m
            ):
                reasons.append("line_user_clearance_exceeded")
            candidates.append(
                AttachmentOption(
                    user_id=user_id,
                    attach_node_id=node_id,
                    horizontal_length_m=float(metrics["horizontal_length_m"]),
                    length_3d_m=float(metrics["length_3d_m"]),
                    cost=float(metrics["length_3d_m"]) * float(planning_cfg.get("service_line_cost_per_m", 35.0)),
                    feasible=False,
                    infeasible_reasons=tuple(reasons),
                )
            )
        candidates.sort(key=lambda item: (item.cost, item.horizontal_length_m))
        options_by_user[user_id] = candidates[: max(1, min(3, len(candidates)))]
    return options_by_user


def select_initial_attachments(
    options_by_user: dict[int, list[AttachmentOption]],
) -> dict[int, AttachmentOption]:
    """Pick the cheapest available attachment option for each user."""

    return {user_id: options[0] for user_id, options in options_by_user.items() if options}
