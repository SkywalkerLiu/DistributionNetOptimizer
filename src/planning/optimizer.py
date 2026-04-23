from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


PHASES = ("A", "B", "C")
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}


@dataclass(slots=True)
class OptimizedPlan:
    """Optimized network layers and summary metrics."""

    users: gpd.GeoDataFrame
    transformer: gpd.GeoDataFrame
    poles: gpd.GeoDataFrame
    planned_lines: gpd.GeoDataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class _RoutingContext:
    """Route graph plus node coordinate metadata."""

    graph: nx.Graph
    node_data: dict[str, dict[str, Any]]
    grid_node_ids: list[str]
    grid_x: np.ndarray
    grid_y: np.ndarray


def optimize_distribution_network(
    *,
    config: dict[str, Any],
    dtm: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
) -> OptimizedPlan:
    """Build a first-pass 3D radial plan with phase balancing and checks."""

    planning_cfg = config["planning"]
    crs = profile["crs"]
    users = _normalize_users(users=users, dtm=dtm, profile=profile)
    blocked_mask = np.where((buildable_mask > 0) & (forbidden_mask == 0), 0, 1).astype(np.uint8)

    capacity_kva = float(planning_cfg.get("transformer_capacity_kva", 630.0))
    max_loading_ratio = float(planning_cfg.get("max_loading_ratio", 1.0))
    total_kva = float(users["apparent_kva"].sum())
    capacity_limit_kva = capacity_kva * max_loading_ratio
    diagnostics: list[str] = []
    if total_kva > capacity_limit_kva:
        diagnostics.append(
            f"Total load {total_kva:.2f} kVA exceeds transformer limit "
            f"{capacity_limit_kva:.2f} kVA."
        )

    transformer_node = _select_transformer(
        config=config,
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        users=users,
    )
    source_node = _source_node(config=config, dtm=dtm, profile=profile)
    route_context = _build_routing_graph(
        config=config,
        dtm=dtm,
        slope=slope,
        roughness=roughness,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        source_node=source_node,
        transformer_node=transformer_node,
        users=users,
    )

    _attach_standard_terminals(
        context=route_context,
        source_node=source_node,
        transformer_node=transformer_node,
        users=users,
        blocked_mask=blocked_mask,
        profile=profile,
        config=config,
    )
    hv_path = _shortest_path_or_diagnostic(
        graph=route_context.graph,
        source=source_node["node_id"],
        target=transformer_node["node_id"],
        diagnostics=diagnostics,
        label="high-voltage source to transformer",
    )
    low_tree = _build_low_voltage_tree(
        context=route_context,
        graph=route_context.graph,
        root=transformer_node["node_id"],
        users=users,
        blocked_mask=blocked_mask,
        profile=profile,
        config=config,
        diagnostics=diagnostics,
    )
    hv_path = _expand_path_with_supports(
        context=route_context,
        path=hv_path,
        line_type="hv_line",
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
    )
    low_tree = _expand_tree_with_supports(
        context=route_context,
        tree=low_tree,
        root=transformer_node["node_id"],
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
    )

    phase_assignment = _optimize_phases(
        tree=low_tree,
        root=transformer_node["node_id"],
        users=users,
    )
    users["assigned_phase"] = users["user_id"].map(
        lambda value: phase_assignment.get(int(value), "")
    )

    electrical = _compute_electrical_metrics(
        tree=low_tree,
        root=transformer_node["node_id"],
        users=users,
        config=config,
    )
    users["connected_node_id"] = users["user_id"].map(
        lambda value: electrical["user_connections"].get(int(value), "")
    )
    users["voltage_drop_pct"] = users["user_id"].map(
        lambda value: electrical["user_voltage_drop_pct"].get(int(value), 0.0)
    )

    transformer_layer = _build_transformer_layer(
        transformer_node=transformer_node,
        slope=slope,
        buildable_mask=buildable_mask,
        profile=profile,
        config=config,
        crs=crs,
    )
    pole_layer, pole_id_by_node = _build_pole_layer(
        context=route_context,
        low_tree=low_tree,
        hv_path=hv_path,
        slope=slope,
        profile=profile,
        config=config,
        crs=crs,
    )
    planned_lines = _build_line_layer(
        context=route_context,
        low_tree=low_tree,
        hv_path=hv_path,
        root=transformer_node["node_id"],
        pole_id_by_node=pole_id_by_node,
        electrical=electrical,
        users=users,
        config=config,
        crs=crs,
    )
    pole_layer, planned_lines = _enforce_line_clearance(
        pole_layer=pole_layer,
        planned_lines=planned_lines,
        transformer=transformer_layer,
        users=users,
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
        diagnostics=diagnostics,
        crs=crs,
    )
    summary = _build_summary(
        users=users,
        transformer_node=transformer_node,
        planned_lines=planned_lines,
        pole_layer=pole_layer,
        total_kva=total_kva,
        capacity_kva=capacity_kva,
        capacity_limit_kva=capacity_limit_kva,
        electrical=electrical,
        diagnostics=diagnostics,
        config=config,
    )
    return OptimizedPlan(users, transformer_layer, pole_layer, planned_lines, summary)


def _attach_standard_terminals(
    *,
    context: _RoutingContext,
    source_node: dict[str, Any],
    transformer_node: dict[str, Any],
    users: gpd.GeoDataFrame,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Attach source/transformer to the route graph and register user terminals."""

    planning_cfg = config["planning"]
    sample_step = float(planning_cfg.get("forbidden_edge_sample_step_m", 5.0))
    max_span = float(planning_cfg.get("max_pole_span_m", 50.0))
    for node, kind, distance in (
        (source_node, "source", max_span),
        (transformer_node, "transformer", max_span),
    ):
        _add_terminal_node(
            context=context,
            node_id=node["node_id"],
            x=node["x"],
            y=node["y"],
            z=node["z"],
            kind=kind,
            max_connect_m=distance,
            blocked_mask=blocked_mask,
            profile=profile,
            sample_step_m=sample_step,
        )
    for row in users.itertuples():
        _register_terminal_node(
            context=context,
            node_id=f"user_{int(row.user_id)}",
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            z=float(row.elev_m),
            kind="user",
        )


def _normalize_users(
    *,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Ensure optimizer-required user columns exist."""

    normalized = users.copy()
    if "power_factor" not in normalized.columns:
        normalized["power_factor"] = 0.85
    normalized["power_factor"] = normalized["power_factor"].fillna(0.85).astype(float)

    if "phase_type" not in normalized.columns:
        normalized["phase_type"] = "single"
    normalized["phase_type"] = normalized["phase_type"].fillna("single").astype(str)
    normalized.loc[normalized["phase_type"].isin(["A", "B", "C"]), "phase_type"] = "single"
    normalized.loc[normalized["phase_type"].isin(["ABC"]), "phase_type"] = "three_phase"

    if "assigned_phase" not in normalized.columns:
        normalized["assigned_phase"] = ""
    normalized["assigned_phase"] = normalized["assigned_phase"].fillna("").astype(str)

    if "apparent_kva" not in normalized.columns:
        normalized["apparent_kva"] = normalized["load_kw"] / normalized["power_factor"]
    normalized["apparent_kva"] = normalized["apparent_kva"].astype(float)

    if "elev_m" not in normalized.columns or normalized["elev_m"].isna().any():
        normalized["elev_m"] = [
            _sample_array(dtm, profile, float(point.x), float(point.y))
            for point in normalized.geometry
        ]
    if "connected_node_id" not in normalized.columns:
        normalized["connected_node_id"] = ""
    if "voltage_drop_pct" not in normalized.columns:
        normalized["voltage_drop_pct"] = 0.0
    return normalized


def _select_transformer(
    *,
    config: dict[str, Any],
    dtm: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Select a transformer location from the continuous feasible region."""

    planning_cfg = config["planning"]
    resolution = abs(float(profile["transform"].a))
    step_m = float(planning_cfg.get("transformer_candidate_step_m", 100.0))
    stride = max(1, int(round(step_m / resolution)))
    passable = (buildable_mask > 0) & (forbidden_mask == 0)
    rows, cols = np.where(passable[::stride, ::stride])
    if len(rows) == 0:
        raise ValueError("No feasible transformer search cells are available.")
    rows = rows * stride
    cols = cols * stride
    xs, ys = _cells_to_xy(profile, rows, cols)

    user_x = users.geometry.x.to_numpy(dtype=np.float64)
    user_y = users.geometry.y.to_numpy(dtype=np.float64)
    weights = users["apparent_kva"].to_numpy(dtype=np.float64)
    source_x, source_y = _source_xy(config=config, profile=profile)
    dist_to_users = np.sqrt((xs[:, None] - user_x[None, :]) ** 2 + (ys[:, None] - user_y[None, :]) ** 2)
    weighted_distance = (dist_to_users * weights[None, :]).sum(axis=1) / max(float(weights.sum()), 1.0)
    hv_distance = np.hypot(xs - source_x, ys - source_y)
    slope_score = slope[rows, cols] / max(float(np.percentile(slope, 95)), 1.0)
    roughness_score = roughness[rows, cols] / max(float(np.percentile(roughness, 95)), 1.0)
    score = weighted_distance + 0.35 * hv_distance + 25.0 * slope_score + 15.0 * roughness_score
    best = int(np.argmin(score))
    best_row = int(rows[best])
    best_col = int(cols[best])
    refine_radius = max(1, stride)
    row_min = max(0, best_row - refine_radius)
    row_max = min(passable.shape[0] - 1, best_row + refine_radius)
    col_min = max(0, best_col - refine_radius)
    col_max = min(passable.shape[1] - 1, best_col + refine_radius)
    refine_rows, refine_cols = np.where(passable[row_min : row_max + 1, col_min : col_max + 1])
    if len(refine_rows) > 0:
        refine_rows = refine_rows + row_min
        refine_cols = refine_cols + col_min
        refine_xs, refine_ys = _cells_to_xy(profile, refine_rows, refine_cols)
        refine_dist_to_users = np.sqrt(
            (refine_xs[:, None] - user_x[None, :]) ** 2 + (refine_ys[:, None] - user_y[None, :]) ** 2
        )
        refine_weighted_distance = (refine_dist_to_users * weights[None, :]).sum(axis=1) / max(float(weights.sum()), 1.0)
        refine_hv_distance = np.hypot(refine_xs - source_x, refine_ys - source_y)
        refine_slope_score = slope[refine_rows, refine_cols] / max(float(np.percentile(slope, 95)), 1.0)
        refine_roughness_score = roughness[refine_rows, refine_cols] / max(float(np.percentile(roughness, 95)), 1.0)
        refine_score = (
            refine_weighted_distance
            + 0.35 * refine_hv_distance
            + 25.0 * refine_slope_score
            + 15.0 * refine_roughness_score
        )
        refine_best = int(np.argmin(refine_score))
        rows = refine_rows
        cols = refine_cols
        xs = refine_xs
        ys = refine_ys
        score = refine_score
        best = refine_best
    return {
        "node_id": "TX1",
        "x": float(xs[best]),
        "y": float(ys[best]),
        "z": float(dtm[rows[best], cols[best]]),
        "row": int(rows[best]),
        "col": int(cols[best]),
        "score": float(score[best]),
    }


def _source_node(
    *,
    config: dict[str, Any],
    dtm: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the configured high-voltage source terminal."""

    x, y = _source_xy(config=config, profile=profile)
    return {"node_id": "SOURCE", "x": x, "y": y, "z": _sample_array(dtm, profile, x, y)}


def _source_xy(
    *,
    config: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[float, float]:
    """Return configured source XY, defaulting to the left-middle boundary."""

    planning_cfg = config.get("planning", {})
    if "source_point_xy" in planning_cfg:
        x, y = planning_cfg["source_point_xy"]
        return float(x), float(y)
    transform = profile["transform"]
    left = float(transform.c)
    top = float(transform.f)
    bottom = top + float(profile["height"]) * float(transform.e)
    return left, (top + bottom) / 2.0


def _build_routing_graph(
    *,
    config: dict[str, Any],
    dtm: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    source_node: dict[str, Any] | None = None,
    transformer_node: dict[str, Any] | None = None,
    users: gpd.GeoDataFrame | None = None,
) -> _RoutingContext:
    """Build a sparse visibility graph over the continuous feasible domain."""

    planning_cfg = config["planning"]
    resolution = abs(float(profile["transform"].a))
    step_m = float(planning_cfg.get("path_search_step_m", 20.0))
    stride = max(1, int(round(step_m / resolution)))
    graph = nx.Graph()
    node_data: dict[str, dict[str, Any]] = {}
    grid_node_ids: list[str] = []
    grid_x: list[float] = []
    grid_y: list[float] = []

    passable = (buildable_mask > 0) & (forbidden_mask == 0)
    anchor_cells = _collect_route_anchor_cells(
        passable=passable,
        profile=profile,
        stride=stride,
        source_node=source_node,
        transformer_node=transformer_node,
        users=users,
        config=config,
    )
    if not anchor_cells:
        raise ValueError("No feasible continuous-domain route anchors are available.")

    for row, col in sorted(anchor_cells):
        node_id = _grid_node_id(row, col)
        x, y = _cell_to_xy(profile, row, col)
        data = {
            "node_id": node_id,
            "kind": "grid",
            "x": x,
            "y": y,
            "z": float(dtm[row, col]),
            "row": row,
            "col": col,
        }
        node_data[node_id] = data
        graph.add_node(node_id)
        grid_node_ids.append(node_id)
        grid_x.append(x)
        grid_y.append(y)

    _add_visibility_edges(
        graph=graph,
        node_data=node_data,
        node_ids=grid_node_ids,
        node_x=np.asarray(grid_x, dtype=np.float64),
        node_y=np.asarray(grid_y, dtype=np.float64),
        slope=slope,
        roughness=roughness,
        blocked_mask=np.where(passable, 0, 1).astype(np.uint8),
        profile=profile,
        planning_cfg=planning_cfg,
        step_m=step_m,
    )

    return _RoutingContext(
        graph=graph,
        node_data=node_data,
        grid_node_ids=grid_node_ids,
        grid_x=np.asarray(grid_x, dtype=np.float64),
        grid_y=np.asarray(grid_y, dtype=np.float64),
    )


def _collect_route_anchor_cells(
    *,
    passable: np.ndarray,
    profile: dict[str, Any],
    stride: int,
    source_node: dict[str, Any] | None,
    transformer_node: dict[str, Any] | None,
    users: gpd.GeoDataFrame | None,
    config: dict[str, Any],
) -> set[tuple[int, int]]:
    """Collect sparse anchor cells for continuous-domain visibility routing."""

    anchor_cells: set[tuple[int, int]] = set()
    row_values = list(range(0, passable.shape[0], stride))
    for row_index, row in enumerate(row_values):
        col_offset = stride // 2 if row_index % 2 else 0
        for col in range(col_offset, passable.shape[1], stride):
            nearest = _nearest_passable_cell(
                passable=passable,
                row=row,
                col=col,
                search_radius=max(1, stride // 2),
            )
            if nearest is not None:
                anchor_cells.add(nearest)

    boundary_stride = max(1, stride // 2)
    boundary_mask = _route_boundary_mask(passable)
    boundary_rows, boundary_cols = np.where(boundary_mask)
    for row, col in zip(boundary_rows.tolist(), boundary_cols.tolist()):
        if row % boundary_stride == 0 or col % boundary_stride == 0:
            anchor_cells.add((int(row), int(col)))

    point_specs: list[tuple[float, float, int]] = []
    local_radius = max(1, stride)
    if source_node is not None:
        point_specs.append((float(source_node["x"]), float(source_node["y"]), local_radius))
    if transformer_node is not None:
        point_specs.append((float(transformer_node["x"]), float(transformer_node["y"]), local_radius))
    if users is not None and not users.empty:
        user_radius_m = min(
            float(config["planning"].get("max_service_drop_m", 25.0)),
            float(config["planning"].get("path_search_step_m", 20.0)) * 1.5,
        )
        user_radius = max(1, int(round(user_radius_m / abs(float(profile["transform"].a)))))
        for row in users.itertuples():
            point_specs.append((float(row.geometry.x), float(row.geometry.y), user_radius))

    for x, y, radius_cells in point_specs:
        _add_local_route_seed_cells(
            anchor_cells=anchor_cells,
            passable=passable,
            profile=profile,
            x=x,
            y=y,
            radius_cells=radius_cells,
        )

    return anchor_cells


def _route_boundary_mask(passable: np.ndarray) -> np.ndarray:
    """Return passable cells that border any blocked cell in the 8-neighborhood."""

    padded = np.pad(passable.astype(bool), 1, mode="constant", constant_values=False)
    boundary = np.zeros_like(passable, dtype=bool)
    height, width = passable.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
            boundary |= passable & ~neighbor
    return boundary


def _add_local_route_seed_cells(
    *,
    anchor_cells: set[tuple[int, int]],
    passable: np.ndarray,
    profile: dict[str, Any],
    x: float,
    y: float,
    radius_cells: int,
) -> None:
    """Seed route anchors near source, transformer, and users."""

    row, col = _xy_to_cell(profile, x, y, shape=passable.shape)
    search_radius = max(1, radius_cells // 2)
    radii = sorted({0, max(1, radius_cells // 2), radius_cells})
    for radius in radii:
        offsets = [
            (0, 0),
            (radius, 0),
            (-radius, 0),
            (0, radius),
            (0, -radius),
            (radius, radius),
            (radius, -radius),
            (-radius, radius),
            (-radius, -radius),
        ]
        for dr, dc in offsets:
            nearest = _nearest_passable_cell(
                passable=passable,
                row=int(np.clip(row + dr, 0, passable.shape[0] - 1)),
                col=int(np.clip(col + dc, 0, passable.shape[1] - 1)),
                search_radius=search_radius,
            )
            if nearest is not None:
                anchor_cells.add(nearest)


def _nearest_passable_cell(
    *,
    passable: np.ndarray,
    row: int,
    col: int,
    search_radius: int,
) -> tuple[int, int] | None:
    """Find the nearest passable raster cell near a target row/column."""

    row = int(np.clip(row, 0, passable.shape[0] - 1))
    col = int(np.clip(col, 0, passable.shape[1] - 1))
    if bool(passable[row, col]):
        return row, col
    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for radius in range(1, max(1, search_radius) + 1):
        row_min = max(0, row - radius)
        row_max = min(passable.shape[0] - 1, row + radius)
        col_min = max(0, col - radius)
        col_max = min(passable.shape[1] - 1, col + radius)
        for rr in range(row_min, row_max + 1):
            for cc in range(col_min, col_max + 1):
                if not bool(passable[rr, cc]):
                    continue
                distance = math.hypot(rr - row, cc - col)
                if distance < best_distance:
                    best_distance = distance
                    best = (rr, cc)
        if best is not None:
            return best
    return None


def _add_visibility_edges(
    *,
    graph: nx.Graph,
    node_data: dict[str, dict[str, Any]],
    node_ids: list[str],
    node_x: np.ndarray,
    node_y: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    step_m: float,
) -> None:
    """Connect sparse anchors with line-of-sight edges in the feasible domain."""

    if len(node_ids) <= 1:
        return
    max_neighbors = int(planning_cfg.get("visibility_neighbor_count", 14))
    visibility_radius = float(
        planning_cfg.get(
            "visibility_radius_m",
            max(step_m * 6.0, float(planning_cfg.get("max_pole_span_m", 50.0)) * 3.0),
        )
    )
    sample_step = float(planning_cfg.get("forbidden_edge_sample_step_m", 5.0))
    radius_sq = visibility_radius * visibility_radius

    for index, node_id in enumerate(node_ids):
        dx = node_x - node_x[index]
        dy = node_y - node_y[index]
        dist_sq = dx * dx + dy * dy
        nearby = np.where((dist_sq > 1e-9) & (dist_sq <= radius_sq))[0]
        if len(nearby) < max_neighbors:
            nearby = np.where(dist_sq > 1e-9)[0]
        if len(nearby) == 0:
            continue
        ordered = nearby[np.argsort(dist_sq[nearby])[:max_neighbors]]
        for neighbor_index in ordered:
            if int(neighbor_index) <= index:
                continue
            target_id = node_ids[int(neighbor_index)]
            if graph.has_edge(node_id, target_id):
                continue
            if _line_crosses_blocked(
                float(node_x[index]),
                float(node_y[index]),
                float(node_x[int(neighbor_index)]),
                float(node_y[int(neighbor_index)]),
                blocked_mask=blocked_mask,
                profile=profile,
                sample_step_m=sample_step,
            ):
                continue
            metrics = _edge_metrics(
                a=node_data[node_id],
                b=node_data[target_id],
                slope=slope,
                roughness=roughness,
                profile=profile,
                planning_cfg=planning_cfg,
            )
            graph.add_edge(node_id, target_id, **metrics)


def _add_terminal_node(
    *,
    context: _RoutingContext,
    node_id: str,
    x: float,
    y: float,
    z: float,
    kind: str,
    max_connect_m: float,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    sample_step_m: float,
) -> None:
    """Attach a source, transformer, or user terminal to nearby route nodes."""

    context.graph.add_node(node_id)
    context.node_data[node_id] = {
        "node_id": node_id,
        "kind": kind,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "row": None,
        "col": None,
    }
    if len(context.grid_node_ids) == 0:
        raise ValueError("No route graph nodes are available for terminal attachment.")

    distances = np.hypot(context.grid_x - x, context.grid_y - y)
    nearby = np.where(distances <= max_connect_m + 1e-9)[0]
    if len(nearby) == 0:
        nearest = int(np.argmin(distances))
        raise ValueError(
            f"Terminal {node_id} has no feasible route node within {max_connect_m:.1f} m "
            f"(nearest {distances[nearest]:.1f} m)."
        )

    connected = False
    max_edges = 1 if kind == "user" else 12
    for index in nearby[np.argsort(distances[nearby])[:max_edges]]:
        target_id = context.grid_node_ids[int(index)]
        target = context.node_data[target_id]
        if _line_crosses_blocked(
            x,
            y,
            target["x"],
            target["y"],
            blocked_mask=blocked_mask,
            profile=profile,
            sample_step_m=sample_step_m,
        ):
            continue
        metrics = _metrics_between_points(context.node_data[node_id], target)
        context.graph.add_edge(node_id, target_id, **metrics)
        connected = True
        if kind == "user":
            break

    if not connected:
        raise ValueError(f"Terminal {node_id} cannot connect without crossing a hard constraint.")


def _register_terminal_node(
    *,
    context: _RoutingContext,
    node_id: str,
    x: float,
    y: float,
    z: float,
    kind: str,
) -> None:
    """Register a terminal node without connecting it into the route graph."""

    context.graph.add_node(node_id)
    context.node_data[node_id] = {
        "node_id": node_id,
        "kind": kind,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "row": None,
        "col": None,
    }


def _shortest_path_or_diagnostic(
    *,
    graph: nx.Graph,
    source: str,
    target: str,
    diagnostics: list[str],
    label: str,
) -> list[str]:
    """Find a shortest path and append diagnostics when missing."""

    try:
        return list(nx.shortest_path(graph, source, target, weight="cost"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        diagnostics.append(f"No feasible route for {label}.")
        return []


def _build_low_voltage_tree(
    *,
    context: _RoutingContext,
    graph: nx.Graph,
    root: str,
    users: gpd.GeoDataFrame,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    diagnostics: list[str],
) -> nx.Graph:
    """Build a shared LV backbone and attach users as service-drop leaves only."""

    planning_cfg = config["planning"]
    service_cost_per_m = float(planning_cfg.get("service_line_cost_per_m", 35.0))
    lv_graph = graph.copy()
    if "SOURCE" in lv_graph:
        lv_graph.remove_node("SOURCE")
    attachment_candidates = _user_attachment_candidates(
        context=context,
        root=root,
        users=users,
        blocked_mask=blocked_mask,
        profile=profile,
        config=config,
        diagnostics=diagnostics,
    )
    if not attachment_candidates:
        raise ValueError("Unable to find any feasible low-voltage attachment candidates.")

    root_lengths = nx.single_source_dijkstra_path_length(lv_graph, root, weight="cost")
    ordered_users = sorted(
        users.itertuples(),
        key=lambda row: (
            -_best_root_attachment_cost(
                candidates=attachment_candidates.get(int(row.user_id), []),
                root_lengths=root_lengths,
            ),
            -float(row.apparent_kva),
        ),
    )

    tree = nx.Graph()
    tree.add_node(root)
    backbone_nodes = {root}

    for row in ordered_users:
        user_id = int(row.user_id)
        user_node = f"user_{user_id}"
        candidates = attachment_candidates.get(user_id, [])
        if not candidates:
            diagnostics.append(f"User {user_id} has no feasible service attachment candidate.")
            continue

        best = _select_best_attachment_to_backbone(
            graph=lv_graph,
            backbone_nodes=backbone_nodes,
            candidates=candidates,
            service_cost_per_m=service_cost_per_m,
        )
        if best is None:
            diagnostics.append(f"User {user_id} cannot reach the shared low-voltage backbone.")
            continue

        trimmed_path = _trim_path_to_new_backbone_segment(best["path"], backbone_nodes)
        for node_id in trimmed_path:
            tree.add_node(node_id)
        for from_node, to_node in zip(trimmed_path[:-1], trimmed_path[1:]):
            if not tree.has_edge(from_node, to_node):
                tree.add_edge(from_node, to_node, **graph.edges[from_node, to_node])
        backbone_nodes.update(trimmed_path)

        access_node = str(best["access_node"])
        service_metrics = _metrics_between_points(context.node_data[access_node], context.node_data[user_node])
        service_metrics["cost"] = service_metrics["length_3d_m"] * service_cost_per_m
        tree.add_node(user_node)
        tree.add_edge(access_node, user_node, **service_metrics)

    if tree.number_of_edges() == 0:
        raise ValueError("Unable to build any shared low-voltage route.")
    return tree


def _expand_path_with_supports(
    *,
    context: _RoutingContext,
    path: list[str],
    line_type: str,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Insert intermediate support nodes so routed segments respect span limits."""

    if len(path) <= 1:
        return path
    expanded: list[str] = [path[0]]
    for from_node, to_node in zip(path[:-1], path[1:]):
        chain = _edge_support_chain(
            context=context,
            from_node=from_node,
            to_node=to_node,
            line_type=line_type,
            dtm=dtm,
            buildable_mask=buildable_mask,
            forbidden_mask=forbidden_mask,
            profile=profile,
            config=config,
        )
        expanded.extend(chain[1:])
    return expanded


def _expand_tree_with_supports(
    *,
    context: _RoutingContext,
    tree: nx.Graph,
    root: str,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> nx.Graph:
    """Expand every backbone edge with intermediate support nodes when needed."""

    if tree.number_of_edges() == 0:
        return tree
    expanded = nx.Graph()
    expanded.add_node(root)
    for parent, child in nx.bfs_edges(tree, root):
        line_type = "service_drop" if child.startswith("user_") or parent.startswith("user_") else "lv_line"
        chain = _edge_support_chain(
            context=context,
            from_node=parent,
            to_node=child,
            line_type=line_type,
            dtm=dtm,
            buildable_mask=buildable_mask,
            forbidden_mask=forbidden_mask,
            profile=profile,
            config=config,
        )
        for node_id in chain:
            expanded.add_node(node_id)
        load_metrics = tree.edges[parent, child]
        for from_node, to_node in zip(chain[:-1], chain[1:]):
            metrics = _metrics_between_points(context.node_data[from_node], context.node_data[to_node])
            metrics["cost"] = metrics["length_3d_m"] * float(
                config["planning"].get(
                    "service_line_cost_per_m" if line_type == "service_drop" else "line_cost_per_m",
                    35.0 if line_type == "service_drop" else 55.0,
                )
            )
            for key in ("load_a_kva", "load_b_kva", "load_c_kva"):
                if key in load_metrics:
                    metrics[key] = load_metrics[key]
            expanded.add_edge(from_node, to_node, **metrics)
    return expanded


def _edge_support_chain(
    *,
    context: _RoutingContext,
    from_node: str,
    to_node: str,
    line_type: str,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Return one edge split by intermediate supports when span limits require it."""

    if from_node.startswith("user_") or to_node.startswith("user_"):
        span_limit = float(config["planning"].get("max_service_drop_m", 25.0))
    else:
        span_limit = float(config["planning"].get("max_pole_span_m", 50.0))
    start = context.node_data[from_node]
    end = context.node_data[to_node]
    metrics = _metrics_between_points(start, end)
    if metrics["horizontal_length_m"] <= span_limit + 1e-9:
        return [from_node, to_node]

    segment_count = max(2, int(math.ceil(metrics["horizontal_length_m"] / max(span_limit, 1.0))))
    chain = [from_node]
    for index in range(1, segment_count):
        fraction = float(index) / float(segment_count)
        x = float(start["x"]) + (float(end["x"]) - float(start["x"])) * fraction
        y = float(start["y"]) + (float(end["y"]) - float(start["y"])) * fraction
        support_id = _create_inline_support_node(
            context=context,
            x=x,
            y=y,
            line_type=line_type,
            dtm=dtm,
            buildable_mask=buildable_mask,
            forbidden_mask=forbidden_mask,
            profile=profile,
            config=config,
        )
        chain.append(support_id)
    chain.append(to_node)
    return chain


def _create_inline_support_node(
    *,
    context: _RoutingContext,
    x: float,
    y: float,
    line_type: str,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Create one inline support snapped onto feasible terrain near a routed segment."""

    resolution = abs(float(profile["transform"].a))
    row, col = _xy_to_cell(profile, x, y, shape=buildable_mask.shape)
    nearest = _nearest_passable_cell(
        passable=(buildable_mask > 0) & (forbidden_mask == 0),
        row=row,
        col=col,
        search_radius=max(1, int(math.ceil(float(config["planning"].get("path_search_step_m", 20.0)) / resolution))),
    )
    if nearest is not None:
        row, col = nearest
        x, y = _cell_to_xy(profile, row, col)
    node_id = f"seg_{len(context.node_data) + 1:06d}"
    context.node_data[node_id] = {
        "node_id": node_id,
        "kind": "grid",
        "x": float(x),
        "y": float(y),
        "z": float(_sample_array(dtm, profile, float(x), float(y))),
        "row": int(row),
        "col": int(col),
        "line_hint": line_type,
    }
    context.graph.add_node(node_id)
    return node_id


def _user_attachment_candidates(
    *,
    context: _RoutingContext,
    root: str,
    users: gpd.GeoDataFrame,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    diagnostics: list[str],
) -> dict[int, list[dict[str, Any]]]:
    """Enumerate feasible service-drop attachment nodes for each user."""

    planning_cfg = config["planning"]
    max_service = float(planning_cfg.get("max_service_drop_m", 25.0))
    sample_step = float(planning_cfg.get("forbidden_edge_sample_step_m", 5.0))
    candidates_by_user: dict[int, list[dict[str, Any]]] = {}
    root_data = context.node_data[root]

    for row in users.itertuples():
        user_id = int(row.user_id)
        x = float(row.geometry.x)
        y = float(row.geometry.y)
        distances = np.hypot(context.grid_x - x, context.grid_y - y)
        nearby = np.where(distances <= max_service + 1e-9)[0]
        candidates: list[dict[str, Any]] = []
        for index in nearby[np.argsort(distances[nearby])]:
            node_id = context.grid_node_ids[int(index)]
            node = context.node_data[node_id]
            if _line_crosses_blocked(
                x,
                y,
                float(node["x"]),
                float(node["y"]),
                blocked_mask=blocked_mask,
                profile=profile,
                sample_step_m=sample_step,
            ):
                continue
            candidates.append(
                {
                    "access_node": node_id,
                    "service_length_m": float(distances[int(index)]),
                }
            )

        root_distance = math.hypot(x - float(root_data["x"]), y - float(root_data["y"]))
        if root_distance <= max_service + 1e-9 and not _line_crosses_blocked(
            x,
            y,
            float(root_data["x"]),
            float(root_data["y"]),
            blocked_mask=blocked_mask,
            profile=profile,
            sample_step_m=sample_step,
        ):
            candidates.append({"access_node": root, "service_length_m": float(root_distance)})

        if not candidates:
            diagnostics.append(
                f"User {user_id} has no feasible pole/transformer attachment within {max_service:.1f} m."
            )
        candidates_by_user[user_id] = candidates
    return candidates_by_user


def _best_root_attachment_cost(
    *,
    candidates: list[dict[str, Any]],
    root_lengths: dict[str, float],
) -> float:
    """Return the cheapest root-to-candidate path cost for ordering users."""

    if not candidates:
        return -1.0
    best = float("inf")
    for candidate in candidates:
        access_node = str(candidate["access_node"])
        if access_node in root_lengths:
            best = min(best, float(root_lengths[access_node]))
    return best if math.isfinite(best) else -1.0


def _select_best_attachment_to_backbone(
    *,
    graph: nx.Graph,
    backbone_nodes: set[str],
    candidates: list[dict[str, Any]],
    service_cost_per_m: float,
) -> dict[str, Any] | None:
    """Choose the access node that adds the least new backbone plus service cost."""

    if not candidates:
        return None
    lengths, paths = nx.multi_source_dijkstra(graph, list(backbone_nodes), weight="cost")
    best: dict[str, Any] | None = None
    best_score = float("inf")
    for candidate in candidates:
        access_node = str(candidate["access_node"])
        if access_node not in paths:
            continue
        path = list(paths[access_node])
        trimmed = _trim_path_to_new_backbone_segment(path, backbone_nodes)
        incremental_cost = 0.0
        for from_node, to_node in zip(trimmed[:-1], trimmed[1:]):
            incremental_cost += float(graph.edges[from_node, to_node].get("cost", 0.0))
        score = incremental_cost + float(candidate["service_length_m"]) * service_cost_per_m
        if score + 1e-9 < best_score:
            best_score = score
            best = {
                **candidate,
                "path": path,
                "trimmed_path": trimmed,
                "incremental_cost": incremental_cost,
                "score": score,
            }
    return best


def _trim_path_to_new_backbone_segment(path: list[str], backbone_nodes: set[str]) -> list[str]:
    """Trim a graph path so only the new segment beyond the shared backbone remains."""

    if not path:
        return []
    last_backbone_index = 0
    for index, node_id in enumerate(path):
        if node_id in backbone_nodes:
            last_backbone_index = index
    return path[last_backbone_index:]


def _optimize_phases(
    *,
    tree: nx.Graph,
    root: str,
    users: gpd.GeoDataFrame,
) -> dict[int, str]:
    """Assign single-phase users to A/B/C using greedy search plus local swaps."""

    user_kva = {int(row.user_id): float(row.apparent_kva) for row in users.itertuples()}
    user_phase_type = {int(row.user_id): str(row.phase_type) for row in users.itertuples()}
    single_users = [
        user_id
        for user_id, phase_type in user_phase_type.items()
        if phase_type not in {"three_phase", "ABC"}
    ]
    assignment: dict[int, str] = {}
    phase_load = {phase: 0.0 for phase in PHASES}
    for user_id in sorted(single_users, key=lambda value: user_kva[value], reverse=True):
        phase = min(PHASES, key=lambda name: phase_load[name])
        assignment[user_id] = phase
        phase_load[phase] += user_kva[user_id]
    for user_id, phase_type in user_phase_type.items():
        if phase_type in {"three_phase", "ABC"}:
            assignment[user_id] = "ABC"

    best_score = _phase_score(tree=tree, root=root, assignment=assignment, user_kva=user_kva)
    improved = True
    while improved:
        improved = False
        for user_id in single_users:
            current = assignment[user_id]
            for phase in PHASES:
                if phase == current:
                    continue
                trial = assignment.copy()
                trial[user_id] = phase
                score = _phase_score(tree=tree, root=root, assignment=trial, user_kva=user_kva)
                if score + 1e-9 < best_score:
                    assignment = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return assignment


def _phase_score(
    *,
    tree: nx.Graph,
    root: str,
    assignment: dict[int, str],
    user_kva: dict[int, float],
) -> float:
    """Score transformer and shared-LV-line phase balance for a phase assignment."""

    downstream = _downstream_loads(tree=tree, root=root, assignment=assignment, user_kva=user_kva)
    root_load = downstream["node_loads"].get(root, np.zeros(3, dtype=np.float64))
    line_scores = _shared_lv_line_unbalance_scores(downstream["edge_loads"])
    mean_line_score = float(np.mean(line_scores)) if line_scores else 0.0
    max_line_score = float(np.max(line_scores)) if line_scores else 0.0
    return _unbalance_ratio(root_load) + 0.55 * mean_line_score + 0.90 * max_line_score


def _compute_electrical_metrics(
    *,
    tree: nx.Graph,
    root: str,
    users: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize downstream phase loads, neutral current, and voltage drop."""

    planning_cfg = config["planning"]
    user_kva = {int(row.user_id): float(row.apparent_kva) for row in users.itertuples()}
    assignment = {int(row.user_id): str(row.assigned_phase) for row in users.itertuples()}
    downstream = _downstream_loads(tree=tree, root=root, assignment=assignment, user_kva=user_kva)

    parent = _parent_map(tree, root)
    children = {node: [] for node in tree.nodes}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)

    voltage_phase_v = float(planning_cfg.get("low_voltage_phase_v", 230.0))
    resistance = float(planning_cfg.get("line_resistance_ohm_per_km", 0.642))
    reactance = float(planning_cfg.get("line_reactance_ohm_per_km", 0.083))
    cos_phi = float(users["power_factor"].mean()) if len(users) else 0.85
    sin_phi = math.sqrt(max(1.0 - cos_phi**2, 0.0))

    cumulative_drop: dict[str, np.ndarray] = {root: np.zeros(3, dtype=np.float64)}
    edge_voltage_drop_pct: dict[tuple[str, str], float] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            load = downstream["edge_loads"].get((node, child), np.zeros(3, dtype=np.float64))
            current = (load * 1000.0) / max(voltage_phase_v, 1.0)
            length_km = float(tree.edges[node, child].get("length_3d_m", 0.0)) / 1000.0
            drop_v = current * (resistance * cos_phi + reactance * sin_phi) * length_km
            cumulative_drop[child] = cumulative_drop[node] + (drop_v / max(voltage_phase_v, 1.0) * 100.0)
            edge_voltage_drop_pct[(node, child)] = float(cumulative_drop[child].max())
            stack.append(child)

    user_voltage_drop_pct: dict[int, float] = {}
    user_connections: dict[int, str] = {}
    for row in users.itertuples():
        user_id = int(row.user_id)
        node_id = f"user_{user_id}"
        phase = str(row.assigned_phase)
        if phase == "ABC":
            user_voltage_drop_pct[user_id] = float(cumulative_drop.get(node_id, np.zeros(3)).max())
        else:
            user_voltage_drop_pct[user_id] = float(
                cumulative_drop.get(node_id, np.zeros(3))[PHASE_INDEX.get(phase, 0)]
            )
        user_connections[user_id] = parent.get(node_id, "")

    return {
        **downstream,
        "edge_voltage_drop_pct": edge_voltage_drop_pct,
        "user_voltage_drop_pct": user_voltage_drop_pct,
        "user_connections": user_connections,
    }


def _downstream_loads(
    *,
    tree: nx.Graph,
    root: str,
    assignment: dict[int, str],
    user_kva: dict[int, float],
) -> dict[str, dict[Any, np.ndarray]]:
    """Compute downstream A/B/C kVA for every node and oriented edge."""

    parent = _parent_map(tree, root)
    children = {node: [] for node in tree.nodes}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)

    node_loads: dict[str, np.ndarray] = {}
    edge_loads: dict[tuple[str, str], np.ndarray] = {}

    def visit(node: str) -> np.ndarray:
        load = np.zeros(3, dtype=np.float64)
        if node.startswith("user_"):
            user_id = int(node.split("_", 1)[1])
            phase = assignment.get(user_id, "A")
            if phase == "ABC":
                load += user_kva.get(user_id, 0.0) / 3.0
            else:
                load[PHASE_INDEX.get(phase, 0)] += user_kva.get(user_id, 0.0)
        for child in children.get(node, []):
            child_load = visit(child)
            edge_loads[(node, child)] = child_load
            load += child_load
        node_loads[node] = load
        return load

    visit(root)
    return {"node_loads": node_loads, "edge_loads": edge_loads}


def _parent_map(tree: nx.Graph, root: str) -> dict[str, str]:
    """Return a BFS parent map for a tree."""

    parent: dict[str, str] = {}
    for par, child in nx.bfs_edges(tree, root):
        parent[child] = par
    return parent


def _build_transformer_layer(
    *,
    transformer_node: dict[str, Any],
    slope: np.ndarray,
    buildable_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    crs: Any,
) -> gpd.GeoDataFrame:
    """Create the selected transformer point layer."""

    row, col = _xy_to_cell(profile, transformer_node["x"], transformer_node["y"], shape=slope.shape)
    planning_cfg = config["planning"]
    return gpd.GeoDataFrame(
        {
            "transformer_id": ["TX1"],
            "candidate_id": [1],
            "capacity_kva": [float(planning_cfg.get("transformer_capacity_kva", 630.0))],
            "fixed_cost": [float(planning_cfg.get("transformer_fixed_cost", 120000.0))],
            "elev_m": [float(transformer_node["z"])],
            "ground_slope_deg": [float(slope[row, col])],
            "buildable_score": [float(buildable_mask[row, col])],
            "source": ["optimized"],
        },
        geometry=[Point(transformer_node["x"], transformer_node["y"])],
        crs=crs,
    )


def _build_pole_layer(
    *,
    context: _RoutingContext,
    low_tree: nx.Graph,
    hv_path: list[str],
    slope: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    crs: Any,
) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """Create a pole point for every dynamic route node that needs one."""

    hv_nodes = {node for node in hv_path if node in context.node_data and context.node_data[node]["kind"] == "grid"}
    lv_nodes = {node for node in low_tree.nodes if context.node_data[node]["kind"] == "grid"}
    node_ids = set(hv_nodes) | set(lv_nodes)
    rows = []
    geometries = []
    pole_id_by_node: dict[str, str] = {}
    planning_cfg = config["planning"]
    for index, node_id in enumerate(sorted(node_ids), start=1):
        data = context.node_data[node_id]
        row, col = _xy_to_cell(profile, data["x"], data["y"], shape=slope.shape)
        pole_id = f"P{index:04d}"
        pole_id_by_node[node_id] = pole_id
        if node_id in hv_nodes and node_id in lv_nodes:
            pole_type = "hv_lv_shared"
            pole_height = max(
                float(planning_cfg.get("hv_pole_height_m", 12.0)),
                float(planning_cfg.get("lv_pole_height_m", 10.0)),
            )
        elif node_id in hv_nodes:
            pole_type = "hv_pole"
            pole_height = float(planning_cfg.get("hv_pole_height_m", 12.0))
        else:
            pole_type = "lv_pole"
            pole_height = float(planning_cfg.get("lv_pole_height_m", 10.0))
        rows.append(
            {
                "pole_id": pole_id,
                "candidate_id": index,
                "pole_type": pole_type,
                "pole_height_m": pole_height,
                "fixed_cost": float(planning_cfg.get("pole_fixed_cost", 1800.0)),
                "elev_m": float(data["z"]),
                "ground_slope_deg": float(slope[row, col]),
                "source": "dynamic_path",
            }
        )
        geometries.append(Point(data["x"], data["y"]))
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs), pole_id_by_node


def _build_line_layer(
    *,
    context: _RoutingContext,
    low_tree: nx.Graph,
    hv_path: list[str],
    root: str,
    pole_id_by_node: dict[str, str],
    electrical: dict[str, Any],
    users: gpd.GeoDataFrame,
    config: dict[str, Any],
    crs: Any,
) -> gpd.GeoDataFrame:
    """Build high-voltage, low-voltage, and service-drop line records."""

    records: list[dict[str, Any]] = []
    geometries: list[LineString] = []
    planning_cfg = config["planning"]
    total_load = float(users["apparent_kva"].sum())
    max_vdrop = float(planning_cfg.get("voltage_drop_max_pct", 7.0))

    def add_line(from_node: str, to_node: str, line_type: str, load: np.ndarray, voltage_drop_pct: float) -> None:
        data_a = context.node_data[from_node]
        data_b = context.node_data[to_node]
        metrics = _metrics_between_points(data_a, data_b)
        support_z_start = _support_top_elevation(node_id=from_node, node_data=data_a, line_type=line_type, config=config)
        support_z_end = _support_top_elevation(node_id=to_node, node_data=data_b, line_type=line_type, config=config)
        service_phase = ""
        phase_set = "ABC" if line_type == "hv_line" else "ABCN"
        if line_type == "service_drop":
            user_id = _user_id_from_edge(from_node, to_node)
            if user_id is not None:
                match = users.loc[users["user_id"] == user_id]
                service_phase = str(match.iloc[0]["assigned_phase"]) if not match.empty else ""
                phase_set = f"{service_phase}N" if service_phase in PHASES else "ABCN"

        line_cost_key = {
            "hv_line": "hv_line_cost_per_m",
            "service_drop": "service_line_cost_per_m",
        }.get(line_type, "line_cost_per_m")
        is_violation = int(
            voltage_drop_pct > max_vdrop
            or metrics["horizontal_length_m"] > float(planning_cfg.get("max_pole_span_m", 50.0)) + 1e-9
        )
        if line_type == "service_drop":
            is_violation = int(
                is_violation
                or metrics["horizontal_length_m"] > float(planning_cfg.get("max_service_drop_m", 25.0)) + 1e-9
            )
        records.append(
            {
                "line_id": len(records) + 1,
                "line_type": line_type,
                "from_node": _public_node_id(from_node, pole_id_by_node),
                "to_node": _public_node_id(to_node, pole_id_by_node),
                "phase_set": phase_set,
                "service_phase": service_phase,
                "horizontal_length_m": metrics["horizontal_length_m"],
                "length_3d_m": metrics["length_3d_m"],
                "dz_m": metrics["dz_m"],
                "slope_deg": metrics["slope_deg"],
                "cost": metrics["length_3d_m"] * float(planning_cfg.get(line_cost_key, 55.0)),
                "load_a_kva": float(load[0]),
                "load_b_kva": float(load[1]),
                "load_c_kva": float(load[2]),
                "neutral_current_a": _neutral_current_a(load, float(planning_cfg.get("low_voltage_phase_v", 230.0))),
                "voltage_drop_pct": voltage_drop_pct,
                "support_z_start_m": support_z_start,
                "support_z_end_m": support_z_end,
                "min_clearance_m": np.nan,
                "required_clearance_m": np.nan,
                "is_violation": is_violation,
            }
        )
        geometries.append(LineString([(data_a["x"], data_a["y"]), (data_b["x"], data_b["y"])]))

    hv_load = np.asarray([total_load / 3.0, total_load / 3.0, total_load / 3.0], dtype=np.float64)
    for from_node, to_node in zip(hv_path[:-1], hv_path[1:]):
        add_line(from_node, to_node, "hv_line", hv_load, 0.0)

    parent = _parent_map(low_tree, root)
    for child, par in parent.items():
        line_type = "service_drop" if child.startswith("user_") or par.startswith("user_") else "lv_line"
        add_line(
            par,
            child,
            line_type,
            electrical["edge_loads"].get((par, child), np.zeros(3, dtype=np.float64)),
            float(electrical["edge_voltage_drop_pct"].get((par, child), 0.0)),
        )

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    gdf.attrs["geometry_type"] = "LineString"
    return gdf


def _enforce_line_clearance(
    *,
    pole_layer: gpd.GeoDataFrame,
    planned_lines: gpd.GeoDataFrame,
    transformer: gpd.GeoDataFrame,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    diagnostics: list[str],
    crs: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Repair line segments that violate terrain clearance by adding poles."""

    registry = _build_support_registry(
        pole_layer=pole_layer,
        transformer=transformer,
        users=users,
        dtm=dtm,
        profile=profile,
        config=config,
    )
    added_pole_rows: list[dict[str, Any]] = []
    added_pole_geometries: list[Point] = []
    counters = {"value": len(pole_layer) + 1}
    refined_records: list[dict[str, Any]] = []
    refined_geometries: list[LineString] = []

    for base_record, geometry in zip(planned_lines.drop(columns="geometry").to_dict("records"), planned_lines.geometry):
        record = dict(base_record)
        record["geometry"] = geometry
        pieces = _repair_line_segment(
            record=record,
            registry=registry,
            dtm=dtm,
            buildable_mask=buildable_mask,
            forbidden_mask=forbidden_mask,
            profile=profile,
            config=config,
            counters=counters,
            added_pole_rows=added_pole_rows,
            added_pole_geometries=added_pole_geometries,
            diagnostics=diagnostics,
            depth=0,
        )
        for piece in pieces:
            geometry_piece = piece.pop("geometry")
            refined_records.append(piece)
            refined_geometries.append(geometry_piece)

    if added_pole_rows:
        extra_poles = gpd.GeoDataFrame(added_pole_rows, geometry=added_pole_geometries, crs=crs)
        pole_layer = gpd.GeoDataFrame(
            pd.concat([pole_layer, extra_poles], ignore_index=True),
            geometry="geometry",
            crs=crs,
        )
    refined_lines = gpd.GeoDataFrame(refined_records, geometry=refined_geometries, crs=crs)
    refined_lines.attrs["geometry_type"] = "LineString"
    if not refined_lines.empty:
        refined_lines["line_id"] = np.arange(1, len(refined_lines) + 1, dtype=np.int64)
    return pole_layer, refined_lines


def _build_support_registry(
    *,
    pole_layer: gpd.GeoDataFrame,
    transformer: gpd.GeoDataFrame,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, float | str]]:
    """Build support nodes with ground and attachment elevations."""

    planning_cfg = config["planning"]
    registry: dict[str, dict[str, float | str]] = {}

    source_x, source_y = _source_xy(config=config, profile=profile)
    source_ground = _sample_array(dtm, profile, source_x, source_y)
    registry["SOURCE"] = {
        "node_id": "SOURCE",
        "x": source_x,
        "y": source_y,
        "ground_z": source_ground,
        "pole_height_m": float(planning_cfg.get("source_connection_height_m", 12.0)),
        "kind": "source",
    }

    for row in transformer.itertuples():
        ground = float(row.elev_m)
        registry[str(row.transformer_id)] = {
            "node_id": str(row.transformer_id),
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "ground_z": ground,
            "pole_height_m": float(
                max(
                    _transformer_connection_height(line_type="hv_line", config=config),
                    _transformer_connection_height(line_type="lv_line", config=config),
                )
            ),
            "kind": "transformer",
        }

    for row in pole_layer.itertuples():
        pole_height = float(
            getattr(
                row,
                "pole_height_m",
                max(
                    float(planning_cfg.get("hv_pole_height_m", 13.0)),
                    float(planning_cfg.get("lv_pole_height_m", 10.0)),
                ),
            )
        )
        ground = float(row.elev_m)
        registry[str(row.pole_id)] = {
            "node_id": str(row.pole_id),
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "ground_z": ground,
            "pole_height_m": pole_height,
            "pole_type": str(getattr(row, "pole_type", "lv_pole")),
            "kind": "pole",
        }

    for row in users.itertuples():
        ground = float(row.elev_m)
        registry[f"user_{int(row.user_id)}"] = {
            "node_id": f"user_{int(row.user_id)}",
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "ground_z": ground,
            "pole_height_m": float(planning_cfg.get("user_attachment_height_m", 4.0)),
            "kind": "user",
        }
    return registry


def _repair_line_segment(
    *,
    record: dict[str, Any],
    registry: dict[str, dict[str, float | str]],
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    counters: dict[str, int],
    added_pole_rows: list[dict[str, Any]],
    added_pole_geometries: list[Point],
    diagnostics: list[str],
    depth: int,
) -> list[dict[str, Any]]:
    """Recursively split a segment until terrain clearance is satisfied."""

    planning_cfg = config["planning"]
    start = registry[str(record["from_node"])]
    end = registry[str(record["to_node"])]
    required_clearance = _required_clearance_m(str(record["line_type"]), config)
    clearance = _segment_clearance(
        start=start,
        end=end,
        line_type=str(record["line_type"]),
        dtm=dtm,
        profile=profile,
        config=config,
        sample_step_m=float(planning_cfg.get("clearance_sample_step_m", 2.0)),
    )
    if clearance["min_clearance_m"] >= required_clearance:
        return [
            _finalize_segment_record(
                record=record,
                start=start,
                end=end,
                clearance=clearance,
                required_clearance=required_clearance,
                config=config,
            )
        ]

    if depth >= int(planning_cfg.get("clearance_max_repair_depth", 6)):
        diagnostics.append(
            f"Clearance repair limit reached for segment {record['from_node']} -> {record['to_node']}."
        )
        finalized = _finalize_segment_record(
            record=record,
            start=start,
            end=end,
            clearance=clearance,
            required_clearance=required_clearance,
            config=config,
        )
        finalized["is_violation"] = 1
        return [finalized]

    support = _create_repair_support(
        worst_x=float(clearance["worst_x"]),
        worst_y=float(clearance["worst_y"]),
        line_type=str(record["line_type"]),
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
        counters=counters,
    )
    if support is None:
        diagnostics.append(
            f"Unable to place clearance repair pole for segment {record['from_node']} -> {record['to_node']}."
        )
        finalized = _finalize_segment_record(
            record=record,
            start=start,
            end=end,
            clearance=clearance,
            required_clearance=required_clearance,
            config=config,
        )
        finalized["is_violation"] = 1
        return [finalized]

    if (
        math.hypot(float(support["x"]) - float(start["x"]), float(support["y"]) - float(start["y"])) < 1.0
        or math.hypot(float(support["x"]) - float(end["x"]), float(support["y"]) - float(end["y"])) < 1.0
    ):
        diagnostics.append(
            f"Repair support collapsed onto existing node for segment {record['from_node']} -> {record['to_node']}."
        )
        finalized = _finalize_segment_record(
            record=record,
            start=start,
            end=end,
            clearance=clearance,
            required_clearance=required_clearance,
            config=config,
        )
        finalized["is_violation"] = 1
        return [finalized]

    registry[str(support["node_id"])] = support
    added_pole_rows.append(
        {
            "pole_id": str(support["node_id"]),
            "candidate_id": counters["value"] - 1,
            "pole_type": "clearance_repair",
            "pole_height_m": float(support["pole_height_m"]),
            "fixed_cost": float(config["planning"].get("pole_fixed_cost", 1800.0)),
            "elev_m": float(support["ground_z"]),
            "ground_slope_deg": 0.0,
            "source": "clearance_repair",
        }
    )
    added_pole_geometries.append(Point(float(support["x"]), float(support["y"])))

    first_record = dict(record)
    second_record = dict(record)
    first_record["to_node"] = str(support["node_id"])
    second_record["from_node"] = str(support["node_id"])

    total_length = max(float(record["length_3d_m"]), 1e-6)
    first_length = _metrics_between_points(start, support)["length_3d_m"]
    second_length = _metrics_between_points(support, end)["length_3d_m"]
    first_record["voltage_drop_pct"] = float(record["voltage_drop_pct"]) * first_length / total_length
    second_record["voltage_drop_pct"] = float(record["voltage_drop_pct"]) * second_length / total_length

    if str(record["line_type"]) == "service_drop":
        if str(record["to_node"]).startswith("user_"):
            first_record["line_type"] = "lv_line"
            second_record["line_type"] = "service_drop"
        elif str(record["from_node"]).startswith("user_"):
            first_record["line_type"] = "service_drop"
            second_record["line_type"] = "lv_line"

    return _repair_line_segment(
        record=first_record,
        registry=registry,
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
        counters=counters,
        added_pole_rows=added_pole_rows,
        added_pole_geometries=added_pole_geometries,
        diagnostics=diagnostics,
        depth=depth + 1,
    ) + _repair_line_segment(
        record=second_record,
        registry=registry,
        dtm=dtm,
        buildable_mask=buildable_mask,
        forbidden_mask=forbidden_mask,
        profile=profile,
        config=config,
        counters=counters,
        added_pole_rows=added_pole_rows,
        added_pole_geometries=added_pole_geometries,
        diagnostics=diagnostics,
        depth=depth + 1,
    )


def _finalize_segment_record(
    *,
    record: dict[str, Any],
    start: dict[str, float | str],
    end: dict[str, float | str],
    clearance: dict[str, float],
    required_clearance: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Finalize one line segment after clearance checking."""

    metrics = _metrics_between_points(start, end)
    line_type = str(record["line_type"])
    finalized = {key: value for key, value in record.items() if key != "geometry"}
    finalized["horizontal_length_m"] = metrics["horizontal_length_m"]
    finalized["length_3d_m"] = metrics["length_3d_m"]
    finalized["dz_m"] = metrics["dz_m"]
    finalized["slope_deg"] = metrics["slope_deg"]
    finalized["support_z_start_m"] = _support_top_elevation_from_support(start, line_type=line_type, config=config)
    finalized["support_z_end_m"] = _support_top_elevation_from_support(end, line_type=line_type, config=config)
    finalized["min_clearance_m"] = float(clearance["min_clearance_m"])
    finalized["required_clearance_m"] = float(required_clearance)
    finalized["is_violation"] = int(
        bool(finalized.get("is_violation", 0)) or float(clearance["min_clearance_m"]) < float(required_clearance) - 1e-9
    )
    finalized["geometry"] = LineString([(float(start["x"]), float(start["y"])), (float(end["x"]), float(end["y"]))])
    return finalized


def _segment_clearance(
    *,
    start: dict[str, float | str],
    end: dict[str, float | str],
    line_type: str,
    dtm: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    sample_step_m: float,
) -> dict[str, float]:
    """Sample conductor clearance above terrain along one segment."""

    horizontal = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    sample_count = max(2, int(math.ceil(horizontal / max(sample_step_m, 0.5))) + 1)
    worst_clearance = float("inf")
    worst_x = float(start["x"])
    worst_y = float(start["y"])
    z_start = _support_top_elevation_from_support(start, line_type=line_type, config=config)
    z_end = _support_top_elevation_from_support(end, line_type=line_type, config=config)
    for fraction in np.linspace(0.0, 1.0, sample_count):
        x = float(start["x"]) + (float(end["x"]) - float(start["x"])) * float(fraction)
        y = float(start["y"]) + (float(end["y"]) - float(start["y"])) * float(fraction)
        line_z = z_start + (z_end - z_start) * float(fraction)
        terrain_z = _sample_array(dtm, profile, x, y)
        clearance = line_z - terrain_z
        if clearance < worst_clearance:
            worst_clearance = clearance
            worst_x = x
            worst_y = y
    return {"min_clearance_m": worst_clearance, "worst_x": worst_x, "worst_y": worst_y}


def _create_repair_support(
    *,
    worst_x: float,
    worst_y: float,
    line_type: str,
    dtm: np.ndarray,
    buildable_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    config: dict[str, Any],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    """Place a pole near the clearance violation point on feasible terrain."""

    planning_cfg = config["planning"]
    row, col = _xy_to_cell(profile, worst_x, worst_y, shape=buildable_mask.shape)
    resolution = abs(float(profile["transform"].a))
    radius_cells = max(1, int(math.ceil(float(planning_cfg.get("clearance_search_radius_m", 12.0)) / resolution)))
    best: dict[str, Any] | None = None
    best_distance = float("inf")
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            rr = int(np.clip(row + dr, 0, buildable_mask.shape[0] - 1))
            cc = int(np.clip(col + dc, 0, buildable_mask.shape[1] - 1))
            if buildable_mask[rr, cc] <= 0 or forbidden_mask[rr, cc] > 0:
                continue
            x, y = _cell_to_xy(profile, rr, cc)
            distance = math.hypot(x - worst_x, y - worst_y)
            if distance < best_distance:
                ground_z = float(dtm[rr, cc])
                support_id = f"AP{counters['value']:04d}"
                best = {
                    "node_id": support_id,
                    "x": float(x),
                    "y": float(y),
                    "ground_z": ground_z,
                    "pole_height_m": _repair_pole_height(line_type=line_type, config=config),
                    "pole_type": "hv_pole" if line_type == "hv_line" else "lv_pole",
                    "kind": "pole",
                }
                best_distance = distance
    if best is not None:
        counters["value"] += 1
    return best


def _required_clearance_m(line_type: str, config: dict[str, Any]) -> float:
    """Return the configured minimum ground clearance for a line type."""

    planning_cfg = config["planning"]
    if line_type == "hv_line":
        return float(planning_cfg.get("hv_ground_clearance_m", 6.5))
    if line_type == "service_drop":
        return float(planning_cfg.get("service_ground_clearance_m", 2.5))
    return float(planning_cfg.get("lv_ground_clearance_m", 6.0))


def _support_top_elevation(
    *,
    node_id: str,
    node_data: dict[str, Any],
    line_type: str,
    config: dict[str, Any],
) -> float:
    """Return support top elevation used for line rendering."""

    planning_cfg = config["planning"]
    ground = float(node_data["z"])
    if node_id == "SOURCE":
        return ground + float(planning_cfg.get("source_connection_height_m", 12.0))
    if node_id == "TX1":
        return ground + _transformer_connection_height(line_type=line_type, config=config)
    if str(node_id).startswith("user_"):
        return ground + float(planning_cfg.get("user_attachment_height_m", 4.0))
    if line_type == "hv_line":
        return ground + float(planning_cfg.get("hv_pole_height_m", 13.0))
    return ground + float(planning_cfg.get("lv_pole_height_m", 10.0))


def _support_top_elevation_from_support(
    support: dict[str, float | str],
    *,
    line_type: str,
    config: dict[str, Any],
) -> float:
    """Return support top elevation from registry support data."""

    planning_cfg = config["planning"]
    ground = float(support["ground_z"])
    kind = str(support.get("kind", "pole"))
    if kind == "source":
        return ground + float(planning_cfg.get("source_connection_height_m", 12.0))
    if kind == "transformer":
        return ground + _transformer_connection_height(line_type=line_type, config=config)
    if kind == "user":
        return ground + float(planning_cfg.get("user_attachment_height_m", 4.0))

    pole_type = str(support.get("pole_type", "lv_pole"))
    if pole_type == "hv_lv_shared":
        return ground + (
            float(planning_cfg.get("hv_pole_height_m", 13.0))
            if line_type == "hv_line"
            else float(planning_cfg.get("lv_pole_height_m", 10.0))
        )
    if pole_type == "hv_pole" or line_type == "hv_line":
        return ground + float(support.get("pole_height_m", planning_cfg.get("hv_pole_height_m", 13.0)))
    return ground + float(support.get("pole_height_m", planning_cfg.get("lv_pole_height_m", 10.0)))


def _repair_pole_height(*, line_type: str, config: dict[str, Any]) -> float:
    """Return height for a clearance-repair support."""

    planning_cfg = config["planning"]
    if line_type == "hv_line":
        return float(planning_cfg.get("hv_pole_height_m", 13.0))
    return float(planning_cfg.get("lv_pole_height_m", 10.0))


def _transformer_connection_height(*, line_type: str, config: dict[str, Any]) -> float:
    """Return transformer-side conductor attachment height by line type."""

    planning_cfg = config["planning"]
    if line_type == "hv_line":
        return float(
            planning_cfg.get(
                "transformer_hv_connection_height_m",
                planning_cfg.get("transformer_connection_height_m", planning_cfg.get("hv_pole_height_m", 13.0)),
            )
        )
    return float(
        planning_cfg.get(
            "transformer_lv_connection_height_m",
            planning_cfg.get("transformer_connection_height_m", planning_cfg.get("lv_pole_height_m", 10.0)),
        )
    )


def _build_summary(
    *,
    users: gpd.GeoDataFrame,
    transformer_node: dict[str, Any],
    planned_lines: gpd.GeoDataFrame,
    pole_layer: gpd.GeoDataFrame,
    total_kva: float,
    capacity_kva: float,
    capacity_limit_kva: float,
    electrical: dict[str, Any],
    diagnostics: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable optimization summary."""

    planning_cfg = config["planning"]
    total_phase_load = electrical["node_loads"].get("TX1", np.zeros(3, dtype=np.float64))
    max_voltage_drop = float(users["voltage_drop_pct"].max()) if len(users) else 0.0
    shared_lv_line_scores = _shared_lv_line_unbalance_scores(electrical["edge_loads"])
    max_line_unbalance = float(np.max(shared_lv_line_scores)) if shared_lv_line_scores else 0.0
    mean_line_unbalance = float(np.mean(shared_lv_line_scores)) if shared_lv_line_scores else 0.0

    total_cost = float(planned_lines["cost"].sum() if not planned_lines.empty else 0.0)
    total_cost += float(planning_cfg.get("transformer_fixed_cost", 120000.0))
    total_cost += len(pole_layer) * float(planning_cfg.get("pole_fixed_cost", 1800.0))

    violation_count = int(planned_lines["is_violation"].sum()) if not planned_lines.empty else 0
    if max_voltage_drop > float(planning_cfg.get("voltage_drop_max_pct", 7.0)):
        diagnostics.append(f"Maximum voltage drop {max_voltage_drop:.2f}% exceeds configured limit.")
    if _unbalance_ratio(total_phase_load) > float(planning_cfg.get("phase_balance_max_ratio", 0.15)):
        diagnostics.append("Transformer phase balance exceeds configured hard threshold.")
    if max_line_unbalance > float(planning_cfg.get("phase_balance_max_ratio", 0.15)):
        diagnostics.append("Shared LV line phase balance exceeds configured hard threshold.")
    return {
        "feasible": len(diagnostics) == 0 and violation_count == 0 and total_kva <= capacity_limit_kva,
        "diagnostics": diagnostics,
        "total_cost": round(total_cost, 3),
        "transformer": {
            "id": "TX1",
            "capacity_kva": capacity_kva,
            "loading_kva": round(total_kva, 3),
            "loading_ratio": round(total_kva / max(capacity_kva, 1.0), 4),
            "x": round(float(transformer_node["x"]), 3),
            "y": round(float(transformer_node["y"]), 3),
            "elev_m": round(float(transformer_node["z"]), 3),
        },
        "line_totals": {
            "count": int(len(planned_lines)),
            "total_3d_length_m": round(float(planned_lines["length_3d_m"].sum()), 3) if not planned_lines.empty else 0.0,
            "total_horizontal_length_m": round(float(planned_lines["horizontal_length_m"].sum()), 3) if not planned_lines.empty else 0.0,
            "max_dz_m": round(float(planned_lines["dz_m"].abs().max()), 3) if not planned_lines.empty else 0.0,
            "max_slope_deg": round(float(planned_lines["slope_deg"].max()), 3) if not planned_lines.empty else 0.0,
        },
        "poles": {"count": int(len(pole_layer))},
        "phase_balance": {
            "load_a_kva": round(float(total_phase_load[0]), 3),
            "load_b_kva": round(float(total_phase_load[1]), 3),
            "load_c_kva": round(float(total_phase_load[2]), 3),
            "transformer_unbalance_ratio": round(_unbalance_ratio(total_phase_load), 5),
            "max_shared_lv_line_unbalance_ratio": round(max_line_unbalance, 5),
            "mean_shared_lv_line_unbalance_ratio": round(mean_line_unbalance, 5),
        },
        "voltage": {"max_voltage_drop_pct": round(max_voltage_drop, 4)},
    }


def _edge_metrics(
    *,
    a: dict[str, Any],
    b: dict[str, Any],
    slope: np.ndarray,
    roughness: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute route graph cost metrics between raster-backed nodes."""

    metrics = _metrics_between_points(a, b)
    row_a, col_a = _xy_to_cell(profile, a["x"], a["y"], shape=slope.shape)
    row_b, col_b = _xy_to_cell(profile, b["x"], b["y"], shape=slope.shape)
    slope_penalty = float((slope[row_a, col_a] + slope[row_b, col_b]) / 2.0)
    roughness_penalty = float((roughness[row_a, col_a] + roughness[row_b, col_b]) / 2.0)
    cost = metrics["length_3d_m"] * float(planning_cfg.get("terrain_length_weight", 1.0))
    cost *= 1.0 + float(planning_cfg.get("terrain_slope_weight", 2.0)) * slope_penalty / 45.0
    cost *= 1.0 + float(planning_cfg.get("terrain_roughness_weight", 1.0)) * roughness_penalty / 10.0
    metrics["cost"] = float(cost)
    return metrics


def _metrics_between_points(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Compute horizontal length, 3D length, elevation change, and slope."""

    dx = float(b["x"] - a["x"])
    dy = float(b["y"] - a["y"])
    z_a = float(a.get("z", a.get("ground_z", 0.0)))
    z_b = float(b.get("z", b.get("ground_z", 0.0)))
    dz = float(z_b - z_a)
    horizontal = math.hypot(dx, dy)
    length_3d = math.sqrt(horizontal**2 + dz**2)
    slope_deg = math.degrees(math.atan2(abs(dz), horizontal)) if horizontal > 0 else 0.0
    return {
        "horizontal_length_m": float(horizontal),
        "length_3d_m": float(length_3d),
        "dz_m": float(dz),
        "slope_deg": float(slope_deg),
        "cost": float(length_3d),
    }


def _line_crosses_blocked(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    blocked_mask: np.ndarray,
    profile: dict[str, Any],
    sample_step_m: float,
) -> bool:
    """Return true when a segment samples any blocked raster cell."""

    distance = math.hypot(x2 - x1, y2 - y1)
    sample_count = max(2, int(math.ceil(distance / max(sample_step_m, 0.5))) + 1)
    for fraction in np.linspace(0.0, 1.0, sample_count):
        x = x1 + (x2 - x1) * float(fraction)
        y = y1 + (y2 - y1) * float(fraction)
        row, col = _xy_to_cell(profile, x, y, shape=blocked_mask.shape)
        if blocked_mask[row, col] > 0:
            return True
    return False


def _line_crosses_forbidden(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    sample_step_m: float,
) -> bool:
    """Backward-compatible helper for checks against hard-forbidden areas only."""

    return _line_crosses_blocked(
        x1,
        y1,
        x2,
        y2,
        blocked_mask=forbidden_mask,
        profile=profile,
        sample_step_m=sample_step_m,
    )


def _grid_node_id(row: int, col: int) -> str:
    """Return a stable route-grid node id."""

    return f"g_{row}_{col}"


def _cell_to_xy(profile: dict[str, Any], row: int, col: int) -> tuple[float, float]:
    """Convert raster row/column to cell-center XY."""

    transform = profile["transform"]
    return (
        float(transform.c + (col + 0.5) * transform.a),
        float(transform.f + (row + 0.5) * transform.e),
    )


def _cells_to_xy(
    profile: dict[str, Any],
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized cell-center conversion."""

    transform = profile["transform"]
    return (
        transform.c + (cols + 0.5) * transform.a,
        transform.f + (rows + 0.5) * transform.e,
    )


def _xy_to_cell(
    profile: dict[str, Any],
    x: float,
    y: float,
    *,
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert XY to a clipped raster row/column."""

    transform = profile["transform"]
    col = int(math.floor((x - transform.c) / transform.a))
    row = int(math.floor((y - transform.f) / transform.e))
    row = int(np.clip(row, 0, shape[0] - 1))
    col = int(np.clip(col, 0, shape[1] - 1))
    return row, col


def _sample_array(array: np.ndarray, profile: dict[str, Any], x: float, y: float) -> float:
    """Nearest-neighbor sample from a raster array."""

    row, col = _xy_to_cell(profile, x, y, shape=array.shape)
    return float(array[row, col])


def _user_id_from_edge(from_node: str, to_node: str) -> int | None:
    """Return the user id carried by a service edge when present."""

    for node in (from_node, to_node):
        if node.startswith("user_"):
            return int(node.split("_", 1)[1])
    return None


def _public_node_id(node_id: str, pole_id_by_node: dict[str, str]) -> str:
    """Map internal route node ids to public transformer, pole, or user ids."""

    if node_id in pole_id_by_node:
        return pole_id_by_node[node_id]
    return node_id


def _neutral_current_a(load: np.ndarray, voltage_phase_v: float) -> float:
    """Estimate neutral current from three phase currents."""

    currents = (load * 1000.0) / max(voltage_phase_v, 1.0)
    ia, ib, ic = map(float, currents)
    neutral_sq = max(ia**2 + ib**2 + ic**2 - ia * ib - ib * ic - ic * ia, 0.0)
    return float(math.sqrt(neutral_sq))


def _unbalance_ratio(load: np.ndarray) -> float:
    """Return max phase deviation divided by average phase load."""

    total = float(load.sum())
    if total <= 0.0:
        return 0.0
    avg = total / 3.0
    return float(np.max(np.abs(load - avg)) / max(avg, 1e-9))


def _shared_lv_line_unbalance_scores(edge_loads: dict[tuple[str, str], np.ndarray]) -> list[float]:
    """Return unbalance scores for shared LV backbone segments only.

    Service drops to individual users are excluded because they are single-phase by design
    and are not meaningful ABC balance targets.
    """

    scores: list[float] = []
    for (_, child), load in edge_loads.items():
        if str(child).startswith("user_"):
            continue
        if float(load.sum()) <= 0.0:
            continue
        scores.append(_unbalance_ratio(load))
    return scores
