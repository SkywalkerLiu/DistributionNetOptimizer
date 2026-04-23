from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def generate_obstacle_layers(
    config: dict[str, Any],
    *,
    scene_bounds: tuple[float, float, float, float],
    crs: str,
    users: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """Generate forest, water, and manual no-build polygons."""

    obstacles_cfg = config["obstacles"]
    seed = int(config["scene"]["seed"]) + 303
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = scene_bounds
    scene_polygon = box(minx, miny, maxx, maxy)
    protected = (
        unary_union(users.geometry.buffer(float(obstacles_cfg["buffer_from_users_m"])))
        if not users.empty
        else None
    )

    forest = _make_obstacle_layer(
        rng=rng,
        count=int(obstacles_cfg["forest_count"]),
        kind="forest",
        scene_polygon=scene_polygon,
        protected=protected,
        min_area=float(obstacles_cfg["min_area_m2"]),
        max_area=float(obstacles_cfg["max_area_m2"]),
        crs=crs,
    )
    water = _make_obstacle_layer(
        rng=rng,
        count=int(obstacles_cfg["water_count"]),
        kind="water",
        scene_polygon=scene_polygon,
        protected=protected,
        min_area=float(obstacles_cfg["min_area_m2"]),
        max_area=float(obstacles_cfg["max_area_m2"]),
        crs=crs,
    )
    manual = _make_obstacle_layer(
        rng=rng,
        count=int(obstacles_cfg["manual_no_build_count"]),
        kind="manual_no_build",
        scene_polygon=scene_polygon,
        protected=protected,
        min_area=float(obstacles_cfg["min_area_m2"]),
        max_area=float(obstacles_cfg["max_area_m2"]),
        crs=crs,
    )

    return {
        "forest": forest,
        "water": water,
        "manual_no_build": manual,
    }


def rasterize_forbidden_mask(
    *,
    profile: dict[str, Any],
    forest: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    manual_no_build: gpd.GeoDataFrame,
) -> np.ndarray:
    """Rasterize the forbidden zones into a binary mask."""

    forbidden_shapes = []
    if not forest.empty:
        forbidden_shapes.extend(
            (geom, 1)
            for geom in forest.geometry
            if geom and not geom.is_empty
        )
    forbidden_shapes.extend((geom, 1) for geom in water.geometry if geom and not geom.is_empty)
    forbidden_shapes.extend(
        (geom, 1) for geom in manual_no_build.geometry if geom and not geom.is_empty
    )

    if not forbidden_shapes:
        return np.zeros((profile["height"], profile["width"]), dtype=np.uint8)

    mask = rasterize(
        forbidden_shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    )
    return mask.astype(np.uint8)


def _make_obstacle_layer(
    *,
    rng: np.random.Generator,
    count: int,
    kind: str,
    scene_polygon: Polygon,
    protected,
    min_area: float,
    max_area: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """Generate one obstacle layer with stable attributes."""

    geometries: list[Polygon] = []
    attributes: list[dict[str, Any]] = []

    attempts = 0
    max_attempts = max(50, count * 200)
    while len(geometries) < count and attempts < max_attempts:
        attempts += 1
        polygon = _random_blob_polygon(
            rng=rng,
            scene_polygon=scene_polygon,
            min_area=min_area,
            max_area=max_area,
        )
        if polygon is None or polygon.is_empty or not polygon.is_valid:
            continue
        if protected is not None and polygon.intersects(protected):
            continue

        geometries.append(polygon)
        obs_id = len(geometries)
        if kind == "forest":
            density = float(rng.uniform(0.4, 1.0))
            attributes.append(
                {
                    "obs_id": obs_id,
                    "density": density,
                    "pass_cost": round(1.2 + density * 3.2, 3),
                    "forbidden": 1,
                }
            )
        elif kind == "water":
            attributes.append(
                {
                    "obs_id": obs_id,
                    "water_type": str(rng.choice(["pond", "canal", "stream"])),
                    "forbidden": 1,
                }
            )
        else:
            attributes.append(
                {
                    "obs_id": obs_id,
                    "source": "auto",
                    "reason": "planning_reserve",
                    "forbidden": 1,
                }
            )

    if len(geometries) != count:
        raise ValueError(f"Unable to generate {count} polygons for layer {kind}.")

    return gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)


def _random_blob_polygon(
    *,
    rng: np.random.Generator,
    scene_polygon: Polygon,
    min_area: float,
    max_area: float,
) -> Polygon | None:
    """Create an organic polygon by unioning a few random ellipses."""

    minx, miny, maxx, maxy = scene_polygon.bounds
    center_x = float(rng.uniform(minx, maxx))
    center_y = float(rng.uniform(miny, maxy))
    target_area = float(rng.uniform(min_area, max_area))
    part_count = int(rng.integers(2, 5))

    parts = []
    for _ in range(part_count):
        radius = np.sqrt(target_area / np.pi) * float(rng.uniform(0.35, 0.8))
        blob = Point(0.0, 0.0).buffer(1.0, quad_segs=24)
        ellipse = affinity.scale(
            affinity.rotate(
                affinity.translate(
                    blob,
                    xoff=center_x + float(rng.normal(0.0, radius * 0.3)),
                    yoff=center_y + float(rng.normal(0.0, radius * 0.3)),
                ),
                angle=float(rng.uniform(0.0, 180.0)),
            ),
            xfact=radius * float(rng.uniform(0.6, 1.6)),
            yfact=radius * float(rng.uniform(0.4, 1.3)),
            origin="center",
        )
        parts.append(ellipse)

    polygon = unary_union(parts).intersection(scene_polygon).buffer(0)
    if polygon.is_empty:
        return None
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda geom: geom.area)

    area = float(polygon.area)
    if area < min_area or area > max_area * 1.2:
        return None
    return polygon
