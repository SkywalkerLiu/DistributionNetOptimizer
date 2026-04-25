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
    kind_priority = {"junction": 0, "cluster": 1, "attach": 2}

    def add_node(*, row: int, col: int, kind: str, prefix: str) -> str:
        key = (int(row), int(col))
        if key in node_by_cell:
            existing_id = node_by_cell[key]
            existing = nodes[existing_id]
            if kind_priority.get(kind, 0) > kind_priority.get(existing.kind, 0):
                existing.kind = kind
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
        )
        nodes[node_id] = node
        node_by_cell[key] = node_id
        graph.add_node(node_id, kind=kind, x=node.x, y=node.y, z=node.z)
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

    for row in users.itertuples():
        rr, cc = xy_to_cell(profile, float(row.geometry.x), float(row.geometry.y), shape=corridor_mask.shape)
        nearest = nearest_passable_cell(
            support_mask,
            row=rr,
            col=cc,
            search_radius=max(2, int(math.ceil(float(planning_cfg.get("max_service_drop_m", 25.0)) / resolution))),
        )
        if nearest is not None:
            add_node(row=nearest[0], col=nearest[1], kind="attach", prefix="a")

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
