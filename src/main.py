from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml

from src.features.manual_constraints import refresh_manual_constraints
from src.features.obstacles_generator import generate_obstacle_layers, rasterize_forbidden_mask
from src.features.users_generator import generate_users
from src.io.raster_io import build_raster_profile, read_geotiff, write_geotiff
from src.io.vector_io import (
    FEATURE_LAYER_DEFINITIONS,
    empty_geodataframe,
    initialize_features_gpkg,
    overwrite_layer,
    read_layer,
)
from src.planning.cost_surface import build_cost_surface
from src.terrain.terrain_derivatives import derive_terrain_layers
from src.terrain.terrain_generator import generate_terrain
from src.terrain.terrain_validator import terrain_statistics, validate_terrain_array
from src.viz.plot_scene import generate_scene_plots
from src.viz.plot_terrain_3d import generate_terrain_3d_previews


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the project entry points."""

    parser = argparse.ArgumentParser(description="Distribution scene generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("generate-scene", "derive-terrain", "plot-scene", "plot-terrain-3d"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--config",
            type=Path,
            default=Path("configs/default_config.yaml"),
            help="Path to the YAML configuration file.",
        )
        command_parser.add_argument(
            "--project-root",
            type=Path,
            default=Path("."),
            help="Project root used to resolve output paths.",
        )

    manual_parser = subparsers.add_parser("refresh-manual")
    manual_parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    manual_parser.add_argument("--project-root", type=Path, default=Path("."))
    manual_parser.add_argument(
        "--manual-geojson",
        type=Path,
        required=False,
        help="Optional external manual constraint GeoJSON file.",
    )
    return parser


def main() -> None:
    """Dispatch CLI commands."""

    parser = build_parser()
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config = load_config(args.config)

    if args.command == "generate-scene":
        generate_scene(config=config, project_root=project_root)
    elif args.command == "derive-terrain":
        derive_terrain(config=config, project_root=project_root)
    elif args.command == "plot-scene":
        plot_scene(config=config, project_root=project_root)
    elif args.command == "plot-terrain-3d":
        plot_terrain_3d(config=config, project_root=project_root)
    elif args.command == "refresh-manual":
        refresh_manual(
            config=config,
            project_root=project_root,
            manual_geojson=args.manual_geojson,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")


def generate_scene(*, config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Generate a complete synthetic planning scene."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    profile = build_profile(config)

    dtm = generate_terrain(config)
    validate_terrain_array(
        dtm,
        clip_min=float(config["terrain"]["clip_min"]),
        clip_max=float(config["terrain"]["clip_max"]),
    )
    write_geotiff(paths["dtm"], dtm, profile)
    write_stats(paths["terrain_stats"], terrain_statistics(dtm))

    terrain_layers = derive_terrain_layers(
        dtm,
        resolution_m=float(config["scene"]["resolution_m"]),
        terrain_config=config["terrain"],
        forbidden_mask=None,
    )

    users = generate_users(
        config,
        dtm=dtm,
        valid_mask=terrain_layers["buildable_mask"].astype(bool),
        transform=profile["transform"],
        crs=profile["crs"],
    )

    obstacles = generate_obstacle_layers(
        config,
        scene_bounds=profile_bounds(profile),
        crs=profile["crs"],
        users=users,
    )
    forbidden_mask = rasterize_forbidden_mask(
        profile=profile,
        forest=obstacles["forest"],
        water=obstacles["water"],
        manual_no_build=obstacles["manual_no_build"],
    )

    terrain_layers = derive_terrain_layers(
        dtm,
        resolution_m=float(config["scene"]["resolution_m"]),
        terrain_config=config["terrain"],
        forbidden_mask=forbidden_mask,
    )
    cost_base = build_cost_surface(
        slope=terrain_layers["slope"],
        roughness=terrain_layers["roughness"],
        forbidden_mask=forbidden_mask,
        forest=obstacles["forest"],
        profile=profile,
        planning_config=config["planning"],
    )

    write_geotiff(paths["slope"], terrain_layers["slope"], profile)
    write_geotiff(paths["aspect"], terrain_layers["aspect"], profile)
    write_geotiff(paths["roughness"], terrain_layers["roughness"], profile)
    write_geotiff(paths["buildable_mask"], terrain_layers["buildable_mask"], profile)
    write_geotiff(paths["forbidden_mask"], forbidden_mask, profile)
    write_geotiff(paths["cost_base"], cost_base, profile)

    initialize_features_gpkg(paths["features"], str(profile["crs"]))
    overwrite_layer(paths["features"], "users", users)
    overwrite_layer(paths["features"], "forest", obstacles["forest"])
    overwrite_layer(paths["features"], "water", obstacles["water"])
    overwrite_layer(paths["features"], "manual_no_build", obstacles["manual_no_build"])
    overwrite_layer(
        paths["features"],
        "planned_lines",
        empty_geodataframe(
            columns=FEATURE_LAYER_DEFINITIONS["planned_lines"][1],
            geometry_type="LineString",
            crs=str(profile["crs"]),
        ),
    )

    if bool(config.get("outputs", {}).get("create_plots", True)):
        generate_scene_plots(
            dtm=dtm,
            slope=terrain_layers["slope"],
            forbidden_mask=forbidden_mask,
            profile=profile,
            users=users,
            forest=obstacles["forest"],
            water=obstacles["water"],
            manual_no_build=obstacles["manual_no_build"],
            output_dir=paths["plots"],
        )
        generate_terrain_3d_previews(
            dtm=dtm,
            profile=profile,
            output_dir=paths["plots"],
            visualization_config=config.get("visualization", {}),
            users=users,
            forest=obstacles["forest"],
            water=obstacles["water"],
            manual_no_build=obstacles["manual_no_build"],
            planned_lines=_read_or_empty(paths["features"], "planned_lines"),
        )

    return paths


def derive_terrain(*, config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Rebuild terrain-derived rasters from existing terrain and vector data."""

    paths = resolve_paths(project_root)
    dtm, profile = read_geotiff(paths["dtm"])
    forest = _read_or_empty(paths["features"], "forest")
    water = _read_or_empty(paths["features"], "water")
    manual = _read_or_empty(paths["features"], "manual_no_build")

    forbidden_mask = rasterize_forbidden_mask(
        profile=profile,
        forest=forest,
        water=water,
        manual_no_build=manual,
    )
    terrain_layers = derive_terrain_layers(
        dtm,
        resolution_m=float(config["scene"]["resolution_m"]),
        terrain_config=config["terrain"],
        forbidden_mask=forbidden_mask,
    )
    cost_base = build_cost_surface(
        slope=terrain_layers["slope"],
        roughness=terrain_layers["roughness"],
        forbidden_mask=forbidden_mask,
        forest=forest,
        profile=profile,
        planning_config=config["planning"],
    )

    write_geotiff(paths["slope"], terrain_layers["slope"], profile)
    write_geotiff(paths["aspect"], terrain_layers["aspect"], profile)
    write_geotiff(paths["roughness"], terrain_layers["roughness"], profile)
    write_geotiff(paths["buildable_mask"], terrain_layers["buildable_mask"], profile)
    write_geotiff(paths["forbidden_mask"], forbidden_mask, profile)
    write_geotiff(paths["cost_base"], cost_base, profile)
    return paths


def plot_scene(*, config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Regenerate plot products from existing raster and vector outputs."""

    paths = resolve_paths(project_root)
    dtm, profile = read_geotiff(paths["dtm"])
    slope, _ = read_geotiff(paths["slope"])
    forbidden_mask, _ = read_geotiff(paths["forbidden_mask"])

    users = _read_or_empty(paths["features"], "users")
    forest = _read_or_empty(paths["features"], "forest")
    water = _read_or_empty(paths["features"], "water")
    manual = _read_or_empty(paths["features"], "manual_no_build")

    if bool(config.get("outputs", {}).get("create_plots", True)):
        generate_scene_plots(
            dtm=dtm,
            slope=slope,
            forbidden_mask=forbidden_mask,
            profile=profile,
            users=users,
            forest=forest,
            water=water,
            manual_no_build=manual,
            output_dir=paths["plots"],
        )
    return paths


def plot_terrain_3d(*, config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Generate 3D terrain preview outputs from the existing DTM raster."""

    paths = resolve_paths(project_root)
    dtm, profile = read_geotiff(paths["dtm"])
    users = _read_or_empty(paths["features"], "users")
    forest = _read_or_empty(paths["features"], "forest")
    water = _read_or_empty(paths["features"], "water")
    manual = _read_or_empty(paths["features"], "manual_no_build")
    planned_lines = _read_or_empty(paths["features"], "planned_lines")
    generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=paths["plots"],
        visualization_config=config.get("visualization", {}),
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual,
        planned_lines=planned_lines,
    )
    return paths


def refresh_manual(
    *,
    config: dict[str, Any],
    project_root: Path,
    manual_geojson: Path | None = None,
) -> dict[str, Path]:
    """Refresh manual no-build constraints and downstream derived outputs."""

    paths = resolve_paths(project_root)
    _, profile = read_geotiff(paths["dtm"])
    _, forbidden_mask = refresh_manual_constraints(
        gpkg_path=paths["features"],
        profile=profile,
        external_geojson=manual_geojson,
    )
    write_geotiff(paths["forbidden_mask"], forbidden_mask, profile)
    derive_terrain(config=config, project_root=project_root)
    if bool(config.get("outputs", {}).get("create_plots", True)):
        plot_scene(config=config, project_root=project_root)
    return paths


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Create the raster profile shared by all raster outputs."""

    scene_cfg = config["scene"]
    width = int(round(scene_cfg["width_m"] / scene_cfg["resolution_m"]))
    height = int(round(scene_cfg["height_m"] / scene_cfg["resolution_m"]))
    return build_raster_profile(
        width=width,
        height=height,
        resolution=float(scene_cfg["resolution_m"]),
        crs=str(scene_cfg["crs"]),
        origin_x=float(scene_cfg.get("origin_x_m", 0.0)),
        origin_y=float(scene_cfg.get("origin_y_m", scene_cfg["height_m"])),
        nodata=-9999.0,
        dtype="float32",
    )


def resolve_paths(project_root: Path) -> dict[str, Path]:
    """Resolve all stable output paths relative to the project root."""

    return {
        "dtm": project_root / "data/terrain/dtm.tif",
        "slope": project_root / "data/terrain/slope.tif",
        "aspect": project_root / "data/terrain/aspect.tif",
        "roughness": project_root / "data/terrain/roughness.tif",
        "cost_base": project_root / "data/terrain/cost_base.tif",
        "forbidden_mask": project_root / "data/masks/forbidden_mask.tif",
        "buildable_mask": project_root / "data/masks/buildable_mask.tif",
        "features": project_root / "data/vector/features.gpkg",
        "plots": project_root / "data/outputs/plots",
        "plans": project_root / "data/outputs/plans",
        "terrain_stats": project_root / "data/outputs/plans/terrain_stats.json",
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    """Create all parent directories used by the output paths."""

    for key, path in paths.items():
        if key in {"plots", "plans"}:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)


def profile_bounds(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return bounds in left, bottom, right, top order."""

    left = float(profile["transform"].c)
    top = float(profile["transform"].f)
    right = left + float(profile["width"]) * float(profile["transform"].a)
    bottom = top + float(profile["height"]) * float(profile["transform"].e)
    return left, bottom, right, top


def write_stats(path: Path, stats: dict[str, float]) -> None:
    """Write terrain statistics to a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _read_or_empty(features_path: Path, layer: str) -> gpd.GeoDataFrame:
    """Read a feature layer when present, otherwise return an empty template."""

    if features_path.exists():
        try:
            return read_layer(features_path, layer)
        except KeyError:
            pass
    geometry_type, columns = FEATURE_LAYER_DEFINITIONS[layer]
    return empty_geodataframe(
        columns=columns,
        geometry_type=geometry_type,
        crs="EPSG:3857",
    )


if __name__ == "__main__":
    main()
