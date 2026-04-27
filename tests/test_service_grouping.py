from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from src.planning.service_grouping import build_service_groups


def test_service_grouping_groups_nearby_users_without_cluster_id() -> None:
    cfg = _service_group_cfg()
    users_without_cluster_id = _users(
        [
            (1, 10.0, 10.0),
            (2, 14.0, 11.0),
            (3, 11.0, 15.0),
            (4, 16.0, 15.0),
            (5, 18.0, 12.0),
        ]
    )

    groups = build_service_groups(users=users_without_cluster_id, planning_cfg=cfg)
    multi_groups = [group for group in groups if not group.is_singleton]

    assert len(multi_groups) == 1
    assert len(multi_groups[0].user_ids) == 5


def test_service_grouping_keeps_far_users_singleton() -> None:
    cfg = _service_group_cfg()
    users = _users(
        [
            (1, 10.0, 10.0),
            (2, 50.0, 10.0),
            (3, 10.0, 55.0),
        ]
    )

    groups = build_service_groups(users=users, planning_cfg=cfg)

    assert len(groups) == 3
    assert all(group.is_singleton for group in groups)


def test_service_grouping_splits_large_group_by_max_users() -> None:
    cfg = {
        **_service_group_cfg(),
        "service_group_max_users": 3,
        "service_group_max_diameter_m": 100.0,
    }
    users = _users([(user_id, 10.0 + user_id * 4.0, 10.0) for user_id in range(1, 10)])

    groups = build_service_groups(users=users, planning_cfg=cfg)

    assert max(len(group.user_ids) for group in groups) <= cfg["service_group_max_users"]
    assert len(groups) == 3


def _users(specs: list[tuple[int, float, float]]) -> gpd.GeoDataFrame:
    rows = [{"user_id": int(user_id)} for user_id, _x, _y in specs]
    geometry = [Point(float(x), float(y)) for _user_id, x, y in specs]
    return gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:3857")


def _service_group_cfg() -> dict:
    return {
        "service_grouping_enabled": True,
        "service_group_neighbor_radius_m": 12.0,
        "service_group_min_users": 2,
        "service_group_max_users": 8,
        "service_group_max_diameter_m": 20.0,
    }
