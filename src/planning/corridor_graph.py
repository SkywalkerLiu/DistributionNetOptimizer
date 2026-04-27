from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from src.planning.common import cell_to_xy, nearest_passable_cell, point_metrics, xy_to_cell
from src.planning.geometry_constraints import (
    build_user_point_map,
    line_min_user_clearance,
    point_min_user_clearance,
    segment_is_feasible,
)
from src.planning.models import CorridorEdge, CorridorGraph, CorridorNode
from src.planning.service_grouping import ServiceGroup, _split_component_to_bounded_groups, build_service_groups


def build_corridor_graph(
    *,
    dtm: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    users: Any,
    planning_cfg: dict[str, Any],
    seed: int,
) -> CorridorGraph:
    """Build the V2 candidate corridor graph from passable terrain and user clusters."""

    resolution = abs(float(profile["transform"].a))
    base_passable = ((buildable_mask > 0) & (forbidden_mask == 0)).astype(np.uint8)
    safe_margin_cells = max(0, int(round(float(planning_cfg.get("corridor_safe_margin_m", 12.0)) / resolution)))
    inflated_forbidden = ndimage.binary_dilation(forbidden_mask > 0, iterations=safe_margin_cells)
    corridor_mask = (base_passable > 0) & ~inflated_forbidden
    if not corridor_mask.any():
        corridor_mask = base_passable > 0
    if not corridor_mask.any():
        raise ValueError("No feasible corridor cells are available for the V2 optimizer.")

    user_points = build_user_point_map(users)
    pole_user_clearance_m = float(planning_cfg.get("pole_user_clearance_m", 5.0))
    line_user_clearance_m = float(planning_cfg.get("line_user_clearance_m", 1.0))
    support_mask = _support_mask_outside_users(
        corridor_mask=corridor_mask,
        profile=profile,
        users=users,
        min_clearance_m=pole_user_clearance_m,
    )
    if not support_mask.any():
        raise ValueError("No feasible pole support cells remain outside the configured user clearance.")

    boundary_distance_m = ndimage.distance_transform_edt(corridor_mask) * resolution
    graph = nx.Graph()
    nodes: dict[str, CorridorNode] = {}
    node_by_cell: dict[tuple[int, int], str] = {}
    kind_priority = {
        "junction": 0,
        "cluster": 1,
        "attach": 2,
        "service_group_attach": 3,
    }

    def add_node(
        *,
        row: int,
        col: int,
        kind: str,
        prefix: str,
        service_group_id: str = "",
    ) -> str:
        key = (int(row), int(col))
        if key in node_by_cell:
            existing_id = node_by_cell[key]
            existing = nodes[existing_id]
            if kind_priority.get(kind, 0) > kind_priority.get(existing.kind, 0):
                existing.kind = kind
                graph.nodes[existing_id]["kind"] = kind
            if service_group_id and (not existing.service_group_id or kind == "service_group_attach"):
                existing.service_group_id = service_group_id
                graph.nodes[existing_id]["service_group_id"] = service_group_id
            return existing_id
        x, y = cell_to_xy(profile, row, col)
        node_id = f"{prefix}_{len(nodes) + 1:04d}"
        node = CorridorNode(
            node_id=node_id,
            x=float(x),
            y=float(y),
            z=float(dtm[row, col]),
            row=int(row),
            col=int(col),
            kind=kind,
            service_group_id=service_group_id,
        )
        nodes[node_id] = node
        node_by_cell[key] = node_id
        graph.add_node(
            node_id,
            kind=kind,
            x=node.x,
            y=node.y,
            z=node.z,
            service_group_id=service_group_id,
        )
        return node_id

    edge_max_length_m = float(planning_cfg.get("corridor_edge_max_length_m", 180.0))
    stride = max(1, int(round((edge_max_length_m / 2.0) / resolution)))
    for row in range(0, corridor_mask.shape[0], stride):
        for col in range(0, corridor_mask.shape[1], stride):
            nearest = nearest_passable_cell(support_mask, row=row, col=col, search_radius=max(1, stride))
            if nearest is None:
                continue
            add_node(row=nearest[0], col=nearest[1], kind="junction", prefix="j")

    cluster_count = max(1, min(int(planning_cfg.get("corridor_cluster_count", 6)), len(users)))
    cluster_centers = _kmeans_centers(
        points=np.column_stack([users.geometry.x.to_numpy(dtype=float), users.geometry.y.to_numpy(dtype=float)]),
        count=cluster_count,
        seed=seed,
    )
    for center_x, center_y in cluster_centers:
        row, col = xy_to_cell(profile, float(center_x), float(center_y), shape=corridor_mask.shape)
        nearest = nearest_passable_cell(support_mask, row=row, col=col, search_radius=max(2, stride * 2))
        if nearest is not None:
            add_node(row=nearest[0], col=nearest[1], kind="cluster", prefix="c")

    service_groups = _refine_groups_for_attach_feasibility(
        groups=build_service_groups(
            users=users,
            planning_cfg=planning_cfg,
        ),
        users=users,
        support_mask=support_mask,
        profile=profile,
        planning_cfg=planning_cfg,
    )
    service_group_by_user: dict[int, str] = {}
    service_group_members: dict[str, list[int]] = {}
    service_group_attach_node: dict[str, str] = {}

    for group in service_groups:
        service_group_members[group.group_id] = list(group.user_ids)
        for user_id in group.user_ids:
            service_group_by_user[int(user_id)] = group.group_id

        if group.is_singleton:
            user_id = int(group.user_ids[0])
            user_point = user_points[user_id]
            rr, cc = xy_to_cell(profile, float(user_point.x), float(user_point.y), shape=corridor_mask.shape)
            nearest = nearest_passable_cell(
                support_mask,
                row=rr,
                col=cc,
                search_radius=max(
                    2,
                    int(math.ceil(float(planning_cfg.get("max_service_drop_m", 25.0)) / resolution)),
                ),
            )
            if nearest is None:
                raise ValueError(f"No feasible attach node was generated for singleton service group {group.group_id}.")
            node_id = add_node(
                row=nearest[0],
                col=nearest[1],
                kind="attach",
                prefix="a",
                service_group_id=group.group_id,
            )
        else:
            attach_cell = _find_service_group_attach_cell(
                group=group,
                users=users,
                support_mask=support_mask,
                profile=profile,
                max_service_drop_m=float(
                    planning_cfg.get(
                        "service_group_max_service_drop_m",
                        planning_cfg.get("max_service_drop_m", 25.0),
                    )
                ),
                pole_user_clearance_m=pole_user_clearance_m,
                search_radius_m=float(planning_cfg.get("service_group_attach_search_radius_m", 35.0)),
            )
            if attach_cell is None:
                raise ValueError(f"No feasible shared attach node was generated for service group {group.group_id}.")
            node_id = add_node(
                row=attach_cell[0],
                col=attach_cell[1],
                kind="service_group_attach",
                prefix="sg",
                service_group_id=group.group_id,
            )
        service_group_attach_node[group.group_id] = node_id

    node_ids = list(nodes)
    if not node_ids:
        raise ValueError("No feasible corridor graph nodes remain outside the configured user clearance.")
    node_xy = np.asarray([(nodes[node_id].x, nodes[node_id].y) for node_id in node_ids], dtype=float)
    tree = cKDTree(node_xy)
    allowed_mask = corridor_mask.astype(np.uint8)
    sample_step_m = max(resolution, edge_max_length_m / 18.0)
    configured_neighbors = int(planning_cfg.get("corridor_neighbor_count", 12))
    max_neighbors = min(
        max(2, configured_neighbors),
        max(1, len(node_ids) - 1),
    )
    added_pairs: set[tuple[str, str]] = set()

    for index, node_id in enumerate(node_ids):
        distances, indices = tree.query(node_xy[index], k=min(max_neighbors + 1, len(node_ids)))
        if np.isscalar(indices):
            indices = np.asarray([indices], dtype=int)
        for neighbor_index in np.asarray(indices, dtype=int):
            if neighbor_index == index:
                continue
            neighbor_id = node_ids[int(neighbor_index)]
            pair = tuple(sorted((node_id, neighbor_id)))
            if pair in added_pairs:
                continue
            added_pairs.add(pair)
            _try_add_edge(
                graph=graph,
                nodes=nodes,
                edge_store=None,
                edge_ids=None,
                u=node_id,
                v=neighbor_id,
                slope=slope,
                roughness=roughness,
                boundary_distance_m=boundary_distance_m,
                allowed_mask=allowed_mask,
                profile=profile,
                planning_cfg=planning_cfg,
                sample_step_m=sample_step_m,
                user_points=user_points,
                line_user_clearance_m=line_user_clearance_m,
            )

    _bridge_components(
        graph=graph,
        nodes=nodes,
        slope=slope,
        roughness=roughness,
        boundary_distance_m=boundary_distance_m,
        allowed_mask=allowed_mask,
        profile=profile,
        planning_cfg=planning_cfg,
        sample_step_m=sample_step_m,
        user_points=user_points,
        line_user_clearance_m=line_user_clearance_m,
    )

    edge_ids: dict[tuple[str, str], str] = {}
    edges: dict[str, CorridorEdge] = {}
    for index, (u, v, data) in enumerate(graph.edges(data=True), start=1):
        edge_id = f"e_{index:05d}"
        edge_ids[(u, v)] = edge_id
        edge_ids[(v, u)] = edge_id
        edges[edge_id] = CorridorEdge(
            edge_id=edge_id,
            u=u,
            v=v,
            geometry=LineString([(nodes[u].x, nodes[u].y), (nodes[v].x, nodes[v].y)]),
            horizontal_length_m=float(data["horizontal_length_m"]),
            length_3d_m=float(data["length_3d_m"]),
            build_cost=float(data["weight"]),
            terrain_cost=float(data["terrain_cost"]),
            risk_cost=float(data["risk_cost"]),
            max_span_feasible=bool(data["horizontal_length_m"] <= float(planning_cfg.get("max_pole_span_m", 50.0))),
            is_forbidden=False,
            slope_deg=float(data["slope_deg"]),
            boundary_clearance_m=float(data["boundary_clearance_m"]),
        )
        data["edge_id"] = edge_id

    return CorridorGraph(
        graph=graph,
        nodes=nodes,
        edges=edges,
        corridor_mask=allowed_mask.astype(np.uint8),
        boundary_distance_m=boundary_distance_m.astype(float),
        resolution_m=resolution,
        service_group_by_user=service_group_by_user,
        service_group_members=service_group_members,
        service_group_attach_node=service_group_attach_node,
    )


def _try_add_edge(
    *,
    graph: nx.Graph,
    nodes: dict[str, CorridorNode],
    edge_store: dict[str, CorridorEdge] | None,
    edge_ids: dict[tuple[str, str], str] | None,
    u: str,
    v: str,
    slope: np.ndarray,
    roughness: np.ndarray,
    boundary_distance_m: np.ndarray,
    allowed_mask: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    sample_step_m: float,
    user_points: dict[int, Any] | None = None,
    line_user_clearance_m: float = 1.0,
) -> bool:
    """Add one feasible corridor edge to the graph when possible."""

    node_u = nodes[u]
    node_v = nodes[v]
    metrics = point_metrics(
        {"x": node_u.x, "y": node_u.y, "z": node_u.z},
        {"x": node_v.x, "y": node_v.y, "z": node_v.z},
    )
    edge_max_length_m = float(planning_cfg.get("corridor_edge_max_length_m", 180.0))
    if metrics["horizontal_length_m"] > edge_max_length_m * 1.75:
        return False
    if not segment_is_feasible(
        node_u.x,
        node_u.y,
        node_v.x,
        node_v.y,
        allowed_mask=allowed_mask,
        profile=profile,
        sample_step_m=sample_step_m,
    ):
        return False
    if user_points and line_user_clearance_m > 0.0:
        clearance = line_min_user_clearance(
            line=LineString([(node_u.x, node_u.y), (node_v.x, node_v.y)]),
            user_points=user_points,
        )
        if clearance + 1e-9 < line_user_clearance_m:
            return False

    terrain_cost = _terrain_cost(node_u=node_u, node_v=node_v, slope=slope, roughness=roughness)
    boundary_clearance = min(
        float(boundary_distance_m[node_u.row, node_u.col]),
        float(boundary_distance_m[node_v.row, node_v.col]),
    )
    safe_margin = float(planning_cfg.get("corridor_safe_margin_m", 12.0))
    boundary_ratio = 0.0 if safe_margin <= 0 else max(0.0, safe_margin - boundary_clearance) / safe_margin
    risk_cost = metrics["horizontal_length_m"] * float(planning_cfg.get("corridor_boundary_penalty_weight", 20.0)) * boundary_ratio
    line_cost = metrics["length_3d_m"] * float(planning_cfg.get("line_cost_per_m", 55.0))
    weight = line_cost + terrain_cost + risk_cost
    graph.add_edge(
        u,
        v,
        weight=float(weight),
        horizontal_length_m=float(metrics["horizontal_length_m"]),
        length_3d_m=float(metrics["length_3d_m"]),
        slope_deg=float(metrics["slope_deg"]),
        terrain_cost=float(terrain_cost),
        risk_cost=float(risk_cost),
        boundary_clearance_m=float(boundary_clearance),
    )
    return True


def _bridge_components(
    *,
    graph: nx.Graph,
    nodes: dict[str, CorridorNode],
    slope: np.ndarray,
    roughness: np.ndarray,
    boundary_distance_m: np.ndarray,
    allowed_mask: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    sample_step_m: float,
    user_points: dict[int, Any] | None = None,
    line_user_clearance_m: float = 1.0,
) -> None:
    """Bridge disconnected corridor components with the shortest feasible links."""

    while nx.number_connected_components(graph) > 1:
        components = [sorted(component) for component in nx.connected_components(graph)]
        best_pair: tuple[str, str] | None = None
        best_distance = float("inf")
        for left_index, left_component in enumerate(components[:-1]):
            for right_component in components[left_index + 1 :]:
                for left in left_component:
                    for right in right_component:
                        distance = math.hypot(nodes[left].x - nodes[right].x, nodes[left].y - nodes[right].y)
                        if distance >= best_distance:
                            continue
                        if not segment_is_feasible(
                            nodes[left].x,
                            nodes[left].y,
                            nodes[right].x,
                            nodes[right].y,
                            allowed_mask=allowed_mask,
                            profile=profile,
                            sample_step_m=sample_step_m,
                        ):
                            continue
                        if user_points and line_user_clearance_m > 0.0:
                            clearance = line_min_user_clearance(
                                line=LineString([(nodes[left].x, nodes[left].y), (nodes[right].x, nodes[right].y)]),
                                user_points=user_points,
                            )
                            if clearance + 1e-9 < line_user_clearance_m:
                                continue
                        best_distance = distance
                        best_pair = (left, right)
        if best_pair is None:
            break
        _try_add_edge(
            graph=graph,
            nodes=nodes,
            edge_store=None,
            edge_ids=None,
            u=best_pair[0],
            v=best_pair[1],
            slope=slope,
            roughness=roughness,
            boundary_distance_m=boundary_distance_m,
            allowed_mask=allowed_mask,
            profile=profile,
            planning_cfg=planning_cfg,
            sample_step_m=sample_step_m,
            user_points=user_points,
            line_user_clearance_m=line_user_clearance_m,
        )


def _terrain_cost(
    *,
    node_u: CorridorNode,
    node_v: CorridorNode,
    slope: np.ndarray,
    roughness: np.ndarray,
) -> float:
    """Compute terrain-dependent cost between two corridor nodes."""

    avg_slope = float(slope[node_u.row, node_u.col] + slope[node_v.row, node_v.col]) / 2.0
    avg_roughness = float(roughness[node_u.row, node_u.col] + roughness[node_v.row, node_v.col]) / 2.0
    return avg_slope * 2.0 + avg_roughness * 1.5


def _support_mask_outside_users(
    *,
    corridor_mask: np.ndarray,
    profile: dict[str, Any],
    users: Any,
    min_clearance_m: float,
) -> np.ndarray:
    """Return corridor cells whose centers satisfy the user-to-pole clearance."""

    support_mask = corridor_mask.astype(bool, copy=True)
    if min_clearance_m <= 0.0 or users is None or len(users) == 0:
        return support_mask

    rows, cols = np.nonzero(support_mask)
    if len(rows) == 0:
        return support_mask
    cell_xy = np.asarray([cell_to_xy(profile, int(row), int(col)) for row, col in zip(rows, cols)], dtype=float)
    user_xy = np.column_stack([users.geometry.x.to_numpy(dtype=float), users.geometry.y.to_numpy(dtype=float)])
    distances, _ = cKDTree(user_xy).query(cell_xy, k=1)
    keep = np.asarray(distances, dtype=float) + 1e-9 >= min_clearance_m
    filtered = np.zeros_like(support_mask, dtype=bool)
    filtered[rows[keep], cols[keep]] = True

    # A raster cell-center mask can be slightly conservative near boundaries.
    # If it removes every cell, keep exact point checks in add_node as a guard.
    if not filtered.any():
        for row, col in zip(rows, cols):
            x, y = cell_to_xy(profile, int(row), int(col))
            if point_min_user_clearance(x=x, y=y, user_points=build_user_point_map(users)) + 1e-9 >= min_clearance_m:
                filtered[int(row), int(col)] = True
    return filtered


def _find_service_group_attach_cell(
    *,
    group: ServiceGroup,
    users: Any,
    support_mask: np.ndarray,
    profile: dict[str, Any],
    max_service_drop_m: float,
    pole_user_clearance_m: float,
    search_radius_m: float,
) -> tuple[int, int] | None:
    """Find the best feasible shared attach cell for one service group."""

    resolution = abs(float(profile["transform"].a))
    center_row, center_col = xy_to_cell(
        profile,
        float(group.centroid_x),
        float(group.centroid_y),
        shape=support_mask.shape,
    )
    search_radius_cells = max(1, int(math.ceil(float(search_radius_m) / max(resolution, 1e-9))))
    xy_by_user = _xy_by_user(users)
    group_points = [xy_by_user[int(user_id)] for user_id in group.user_ids if int(user_id) in xy_by_user]
    all_points = list(xy_by_user.values())
    if not group_points:
        return None

    best: tuple[float, int, int] | None = None
    row_min = max(0, center_row - search_radius_cells)
    row_max = min(support_mask.shape[0] - 1, center_row + search_radius_cells)
    col_min = max(0, center_col - search_radius_cells)
    col_max = min(support_mask.shape[1] - 1, center_col + search_radius_cells)

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if not bool(support_mask[row, col]):
                continue
            x, y = cell_to_xy(profile, row, col)
            distance_to_centroid = math.hypot(float(x) - float(group.centroid_x), float(y) - float(group.centroid_y))
            if distance_to_centroid > float(search_radius_m) + resolution:
                continue
            group_distances = [math.hypot(float(x) - user_x, float(y) - user_y) for user_x, user_y in group_points]
            if max(group_distances, default=float("inf")) > float(max_service_drop_m) + 1e-9:
                continue
            if any(
                math.hypot(float(x) - user_x, float(y) - user_y) + 1e-9 < float(pole_user_clearance_m)
                for user_x, user_y in all_points
            ):
                continue
            score = (
                max(group_distances) * 3.0
                + float(np.mean(group_distances))
                + distance_to_centroid * 0.5
            )
            candidate = (float(score), int(row), int(col))
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return None
    return best[1], best[2]


def _refine_groups_for_attach_feasibility(
    *,
    groups: list[ServiceGroup],
    users: Any,
    support_mask: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> list[ServiceGroup]:
    """Split service groups until every multi-user group has a feasible shared attach cell."""

    xy_by_user = _xy_by_user(users)
    max_users = max(1, int(planning_cfg.get("service_group_max_users", 8)))
    max_diameter_m = float(planning_cfg.get("service_group_max_diameter_m", 20.0))
    max_service_drop_m = float(
        planning_cfg.get(
            "service_group_max_service_drop_m",
            planning_cfg.get("max_service_drop_m", 25.0),
        )
    )
    pole_user_clearance_m = float(planning_cfg.get("pole_user_clearance_m", 5.0))
    search_radius_m = float(planning_cfg.get("service_group_attach_search_radius_m", 35.0))

    pending = [list(group.user_ids) for group in groups]
    accepted: list[list[int]] = []
    while pending:
        user_ids = sorted((int(user_id) for user_id in pending.pop(0)), key=lambda item: _user_sort_key(item, xy_by_user))
        if len(user_ids) <= 1:
            accepted.append(user_ids)
            continue

        group = _service_group_from_user_ids(
            group_id="pending",
            user_ids=user_ids,
            xy_by_user=xy_by_user,
        )
        attach_cell = _find_service_group_attach_cell(
            group=group,
            users=users,
            support_mask=support_mask,
            profile=profile,
            max_service_drop_m=max_service_drop_m,
            pole_user_clearance_m=pole_user_clearance_m,
            search_radius_m=search_radius_m,
        )
        if attach_cell is not None:
            accepted.append(user_ids)
            continue

        forced_max_users = max(1, min(max_users, int(math.ceil(len(user_ids) / 2.0))))
        if forced_max_users >= len(user_ids):
            forced_max_users = len(user_ids) - 1
        split_groups = _split_component_to_bounded_groups(
            user_ids=user_ids,
            xy_by_user=xy_by_user,
            max_users=forced_max_users,
            max_diameter_m=max_diameter_m,
        )
        if len(split_groups) <= 1 and len(split_groups[0]) == len(user_ids):
            split_groups = [[user_id] for user_id in user_ids]
        pending = [*split_groups, *pending]

    accepted.sort(key=lambda group: _group_sort_key(group, xy_by_user))
    return [
        _service_group_from_user_ids(
            group_id=f"sg_{index:04d}",
            user_ids=user_ids,
            xy_by_user=xy_by_user,
        )
        for index, user_ids in enumerate(accepted, start=1)
    ]


def _xy_by_user(users: Any) -> dict[int, tuple[float, float]]:
    xy_by_user: dict[int, tuple[float, float]] = {}
    if users is None:
        return xy_by_user
    for row in users.itertuples():
        geometry = getattr(row, "geometry", None)
        if geometry is None or geometry.is_empty:
            continue
        xy_by_user[int(row.user_id)] = (float(geometry.x), float(geometry.y))
    return xy_by_user


def _service_group_from_user_ids(
    *,
    group_id: str,
    user_ids: list[int],
    xy_by_user: dict[int, tuple[float, float]],
) -> ServiceGroup:
    sorted_user_ids = sorted((int(user_id) for user_id in user_ids), key=lambda item: _user_sort_key(item, xy_by_user))
    xs = [xy_by_user[user_id][0] for user_id in sorted_user_ids]
    ys = [xy_by_user[user_id][1] for user_id in sorted_user_ids]
    return ServiceGroup(
        group_id=group_id,
        user_ids=sorted_user_ids,
        centroid_x=float(np.mean(xs)),
        centroid_y=float(np.mean(ys)),
        max_pairwise_distance_m=_max_pairwise_distance(user_ids=sorted_user_ids, xy_by_user=xy_by_user),
        is_singleton=len(sorted_user_ids) <= 1,
    )


def _max_pairwise_distance(
    *,
    user_ids: list[int],
    xy_by_user: dict[int, tuple[float, float]],
) -> float:
    max_distance = 0.0
    for left_index, left in enumerate(user_ids[:-1]):
        left_x, left_y = xy_by_user[int(left)]
        for right in user_ids[left_index + 1 :]:
            right_x, right_y = xy_by_user[int(right)]
            max_distance = max(max_distance, math.hypot(left_x - right_x, left_y - right_y))
    return float(max_distance)


def _user_sort_key(user_id: int, xy_by_user: dict[int, tuple[float, float]]) -> tuple[float, float, int]:
    x, y = xy_by_user[int(user_id)]
    return (float(x), float(y), int(user_id))


def _group_sort_key(group: list[int], xy_by_user: dict[int, tuple[float, float]]) -> tuple[float, float, int]:
    first = sorted(group, key=lambda user_id: _user_sort_key(user_id, xy_by_user))[0]
    x, y = xy_by_user[int(first)]
    return (float(x), float(y), int(first))


def _kmeans_centers(*, points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Return deterministic K-means style cluster centers without external ML deps."""

    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(points) <= count:
        return points.astype(float, copy=True)

    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=count, replace=False)].astype(float)
    for _ in range(12):
        assignments = np.argmin(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        updated = centers.copy()
        for index in range(count):
            members = points[assignments == index]
            if len(members) == 0:
                continue
            updated[index] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers
