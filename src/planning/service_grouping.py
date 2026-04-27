from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class ServiceGroup:
    group_id: str
    user_ids: list[int]
    centroid_x: float
    centroid_y: float
    max_pairwise_distance_m: float
    is_singleton: bool


def build_service_groups(
    *,
    users: Any,
    planning_cfg: dict,
) -> list[ServiceGroup]:
    """Build service groups from user coordinates only.

    Do not use cluster_id, settlement_type, or any simulated grouping field.
    """

    xy_by_user = _xy_by_user(users)
    if not xy_by_user:
        return []

    if not bool(planning_cfg.get("service_grouping_enabled", True)):
        return _make_service_groups([[user_id] for user_id in _sorted_user_ids(xy_by_user)], xy_by_user)

    user_ids = _sorted_user_ids(xy_by_user)
    neighbor_radius_m = float(planning_cfg.get("service_group_neighbor_radius_m", 12.0))
    min_users = max(1, int(planning_cfg.get("service_group_min_users", 2)))
    max_users = max(1, int(planning_cfg.get("service_group_max_users", 8)))
    max_diameter_m = float(planning_cfg.get("service_group_max_diameter_m", 20.0))

    graph = nx.Graph()
    graph.add_nodes_from(user_ids)
    points = np.asarray([xy_by_user[user_id] for user_id in user_ids], dtype=float)
    if len(user_ids) > 1 and neighbor_radius_m >= 0.0:
        tree = cKDTree(points)
        for left, right in tree.query_pairs(r=neighbor_radius_m + 1e-9):
            graph.add_edge(user_ids[int(left)], user_ids[int(right)])

    bounded_groups: list[list[int]] = []
    for component in nx.connected_components(graph):
        component_user_ids = sorted((int(user_id) for user_id in component), key=lambda item: _user_sort_key(item, xy_by_user))
        if len(component_user_ids) < min_users:
            bounded_groups.extend([[user_id] for user_id in component_user_ids])
            continue
        bounded_groups.extend(
            _split_component_to_bounded_groups(
                user_ids=component_user_ids,
                xy_by_user=xy_by_user,
                max_users=max_users,
                max_diameter_m=max_diameter_m,
            )
        )

    bounded_groups.sort(key=lambda group: _group_sort_key(group, xy_by_user))
    return _make_service_groups(bounded_groups, xy_by_user)


def _split_component_to_bounded_groups(
    *,
    user_ids: list[int],
    xy_by_user: dict[int, tuple[float, float]],
    max_users: int,
    max_diameter_m: float,
) -> list[list[int]]:
    """Split one connected component into deterministic bounded service groups."""

    max_users = max(1, int(max_users))
    remaining = sorted((int(user_id) for user_id in user_ids), key=lambda item: _user_sort_key(item, xy_by_user))
    groups: list[list[int]] = []

    while remaining:
        seed = remaining[0]
        current = [seed]
        seed_x, seed_y = xy_by_user[seed]
        candidates = sorted(
            remaining[1:],
            key=lambda user_id: (
                math.hypot(xy_by_user[user_id][0] - seed_x, xy_by_user[user_id][1] - seed_y),
                *_user_sort_key(user_id, xy_by_user),
            ),
        )
        for candidate in candidates:
            if len(current) >= max_users:
                break
            trial = [*current, candidate]
            if _max_pairwise_distance(trial, xy_by_user) <= max_diameter_m + 1e-9:
                current.append(candidate)
        selected = set(current)
        groups.append(sorted(current, key=lambda item: _user_sort_key(item, xy_by_user)))
        remaining = [user_id for user_id in remaining if user_id not in selected]

    return groups


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


def _make_service_groups(
    groups: list[list[int]],
    xy_by_user: dict[int, tuple[float, float]],
) -> list[ServiceGroup]:
    service_groups: list[ServiceGroup] = []
    for index, user_ids in enumerate(groups, start=1):
        sorted_user_ids = sorted((int(user_id) for user_id in user_ids), key=lambda item: _user_sort_key(item, xy_by_user))
        xs = [xy_by_user[user_id][0] for user_id in sorted_user_ids]
        ys = [xy_by_user[user_id][1] for user_id in sorted_user_ids]
        service_groups.append(
            ServiceGroup(
                group_id=f"sg_{index:04d}",
                user_ids=sorted_user_ids,
                centroid_x=float(np.mean(xs)),
                centroid_y=float(np.mean(ys)),
                max_pairwise_distance_m=float(_max_pairwise_distance(sorted_user_ids, xy_by_user)),
                is_singleton=len(sorted_user_ids) <= 1,
            )
        )
    return service_groups


def _max_pairwise_distance(
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


def _sorted_user_ids(xy_by_user: dict[int, tuple[float, float]]) -> list[int]:
    return sorted(xy_by_user, key=lambda user_id: _user_sort_key(user_id, xy_by_user))


def _user_sort_key(user_id: int, xy_by_user: dict[int, tuple[float, float]]) -> tuple[float, float, int]:
    x, y = xy_by_user[int(user_id)]
    return (float(x), float(y), int(user_id))


def _group_sort_key(group: list[int], xy_by_user: dict[int, tuple[float, float]]) -> tuple[float, float, int]:
    ordered = sorted(group, key=lambda user_id: _user_sort_key(user_id, xy_by_user))
    first = ordered[0]
    x, y = xy_by_user[first]
    return (float(x), float(y), int(first))
