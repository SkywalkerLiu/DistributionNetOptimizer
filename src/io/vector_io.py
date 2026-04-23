from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio


FEATURE_LAYER_DEFINITIONS: dict[str, tuple[str, dict[str, str]]] = {
    "users": (
        "Point",
        {
            "user_id": "int64",
            "load_kw": "float64",
            "power_factor": "float64",
            "phase_type": "object",
            "assigned_phase": "object",
            "apparent_kva": "float64",
            "importance": "int64",
            "elev_m": "float64",
            "connected_node_id": "object",
            "voltage_drop_pct": "float64",
        },
    ),
    "forest": (
        "Polygon",
        {
            "obs_id": "int64",
            "density": "float64",
            "pass_cost": "float64",
            "forbidden": "int64",
        },
    ),
    "water": (
        "Polygon",
        {
            "obs_id": "int64",
            "water_type": "object",
            "forbidden": "int64",
        },
    ),
    "manual_no_build": (
        "Polygon",
        {
            "obs_id": "int64",
            "source": "object",
            "reason": "object",
            "forbidden": "int64",
        },
    ),
    "candidate_transformer": (
        "Point",
        {
            "transformer_id": "object",
            "candidate_id": "int64",
            "capacity_kva": "float64",
            "fixed_cost": "float64",
            "elev_m": "float64",
            "ground_slope_deg": "float64",
            "buildable_score": "float64",
            "source": "object",
        },
    ),
    "candidate_poles": (
        "Point",
        {
            "pole_id": "object",
            "candidate_id": "int64",
            "pole_type": "object",
            "pole_height_m": "float64",
            "fixed_cost": "float64",
            "elev_m": "float64",
            "ground_slope_deg": "float64",
            "source": "object",
        },
    ),
    "planned_lines": (
        "LineString",
        {
            "line_id": "int64",
            "line_type": "object",
            "from_node": "object",
            "to_node": "object",
            "phase_set": "object",
            "service_phase": "object",
            "horizontal_length_m": "float64",
            "length_3d_m": "float64",
            "dz_m": "float64",
            "slope_deg": "float64",
            "cost": "float64",
            "load_a_kva": "float64",
            "load_b_kva": "float64",
            "load_c_kva": "float64",
            "neutral_current_a": "float64",
            "voltage_drop_pct": "float64",
            "support_z_start_m": "float64",
            "support_z_end_m": "float64",
            "min_clearance_m": "float64",
            "required_clearance_m": "float64",
            "is_violation": "int64",
        },
    ),
}


def list_layers(path: str | Path) -> list[str]:
    """List layers in a GeoPackage, returning an empty list if missing."""

    gpkg_path = Path(path)
    if not gpkg_path.exists():
        return []
    return [str(row[0]) for row in pyogrio.list_layers(gpkg_path)]


def read_layer(path: str | Path, layer: str) -> gpd.GeoDataFrame:
    """Read a single GeoPackage layer."""

    gpkg_path = Path(path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")
    if layer not in list_layers(gpkg_path):
        raise KeyError(f"Layer not found: {layer}")
    gdf = pyogrio.read_dataframe(gpkg_path, layer=layer)
    gdf.attrs["geometry_type"] = _read_geometry_type(gpkg_path, layer)
    return gdf


def write_layer(path: str | Path, layer: str, gdf: gpd.GeoDataFrame) -> Path:
    """Create a new GeoPackage layer and fail if it already exists."""

    if layer in list_layers(path):
        raise ValueError(f"Layer already exists: {layer}")
    return overwrite_layer(path, layer, gdf)


def append_layer(path: str | Path, layer: str, gdf: gpd.GeoDataFrame) -> Path:
    """Append records to an existing layer or create it when missing."""

    gpkg_path = Path(path)
    if layer not in list_layers(gpkg_path):
        return overwrite_layer(gpkg_path, layer, gdf)

    existing = read_layer(gpkg_path, layer)
    crs = existing.crs or gdf.crs
    incoming = _normalize_crs(gdf, crs)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
    geometry_type = _read_geometry_type(gpkg_path, layer)
    return _write_layer_specs(
        gpkg_path,
        _replace_layer_spec(gpkg_path, layer, combined_gdf, geometry_type),
    )


def overwrite_layer(path: str | Path, layer: str, gdf: gpd.GeoDataFrame) -> Path:
    """Replace a layer while preserving all other layers in the GeoPackage."""

    gpkg_path = Path(path)
    geometry_type = _geometry_type_for(gdf, _read_geometry_type(gpkg_path, layer))
    layer_specs = _replace_layer_spec(gpkg_path, layer, gdf, geometry_type)
    return _write_layer_specs(gpkg_path, layer_specs)


def create_empty_layer(
    path: str | Path,
    layer: str,
    *,
    geometry_type: str,
    columns: dict[str, str],
    crs: str,
) -> Path:
    """Create or replace an empty layer with a stable schema."""

    gdf = empty_geodataframe(columns=columns, geometry_type=geometry_type, crs=crs)
    return overwrite_layer(path, layer, gdf)


def initialize_features_gpkg(path: str | Path, crs: str) -> Path:
    """Create the standard feature layers defined by the planning document."""

    gpkg_path = Path(path)
    layer_specs: OrderedDict[str, tuple[gpd.GeoDataFrame, str]] = OrderedDict()
    for layer_name, (geometry_type, columns) in FEATURE_LAYER_DEFINITIONS.items():
        gdf = empty_geodataframe(
            columns=columns,
            geometry_type=geometry_type,
            crs=crs,
        )
        layer_specs[layer_name] = (gdf, geometry_type)
    return _write_layer_specs(gpkg_path, layer_specs)


def empty_geodataframe(
    *,
    columns: dict[str, str],
    geometry_type: str,
    crs: str,
) -> gpd.GeoDataFrame:
    """Build an empty GeoDataFrame with explicit dtypes and CRS."""

    data = {name: pd.Series(dtype=dtype) for name, dtype in columns.items()}
    geometry = gpd.GeoSeries([], dtype="geometry", crs=crs)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    gdf.attrs["geometry_type"] = geometry_type
    return gdf


def _replace_layer_spec(
    path: Path,
    layer: str,
    gdf: gpd.GeoDataFrame,
    geometry_type: str,
) -> OrderedDict[str, tuple[gpd.GeoDataFrame, str]]:
    """Replace one layer inside the in-memory layer spec map."""

    layer_specs = _read_layer_specs(path)
    normalized = gdf.copy()
    if normalized.crs is None:
        raise ValueError(f"Layer {layer} is missing CRS information.")

    if layer in layer_specs:
        existing_gdf, _ = layer_specs[layer]
        normalized = _normalize_crs(normalized, existing_gdf.crs)

    layer_specs[layer] = (normalized, geometry_type)
    return layer_specs


def _read_layer_specs(
    path: Path,
) -> OrderedDict[str, tuple[gpd.GeoDataFrame, str]]:
    """Load all layers and their geometry schemas from an existing GeoPackage."""

    layer_specs: OrderedDict[str, tuple[gpd.GeoDataFrame, str]] = OrderedDict()
    if not path.exists():
        return layer_specs

    for layer_name in list_layers(path):
        geometry_type = _read_geometry_type(path, layer_name)
        gdf = pyogrio.read_dataframe(path, layer=layer_name)
        gdf.attrs["geometry_type"] = geometry_type
        layer_specs[layer_name] = (gdf, geometry_type)
    return layer_specs


def _write_layer_specs(
    path: Path,
    layer_specs: OrderedDict[str, tuple[gpd.GeoDataFrame, str]],
) -> Path:
    """Rewrite a GeoPackage from a complete ordered set of layers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_gpkg_sidecars(path)
    if path.exists():
        path.unlink()

    mode = "w"
    for layer_name, (gdf, geometry_type) in layer_specs.items():
        normalized = gdf.copy()
        if normalized.crs is None:
            raise ValueError(f"Layer {layer_name} is missing CRS information.")

        pyogrio.write_dataframe(
            normalized,
            path,
            layer=layer_name,
            driver="GPKG",
            append=(mode == "a"),
            geometry_type=geometry_type,
        )
        mode = "a"
    return path


def _normalize_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Set or convert CRS to the target CRS."""

    normalized = gdf.copy()
    if target_crs is None:
        return normalized
    if normalized.crs is None:
        return normalized.set_crs(target_crs)
    return normalized.to_crs(target_crs)


def _read_geometry_type(path: Path, layer: str) -> str:
    """Read the geometry type recorded for a GeoPackage layer."""

    if not path.exists() or layer not in list_layers(path):
        return "Unknown"
    return str(pyogrio.read_info(path, layer=layer).get("geometry_type") or "Unknown")


def _geometry_type_for(gdf: gpd.GeoDataFrame, fallback: str = "Unknown") -> str:
    """Infer a Fiona-compatible geometry type string from a GeoDataFrame."""

    explicit = gdf.attrs.get("geometry_type")
    if explicit:
        return str(explicit)

    geom_types = sorted({geom_type for geom_type in gdf.geom_type if geom_type})
    if not geom_types:
        return fallback
    if len(geom_types) == 1:
        return geom_types[0]
    if set(geom_types).issubset({"Polygon", "MultiPolygon"}):
        return "MultiPolygon"
    if set(geom_types).issubset({"Point", "MultiPoint"}):
        return "MultiPoint"
    if set(geom_types).issubset({"LineString", "MultiLineString"}):
        return "MultiLineString"
    return "Unknown"


def _property_schema(gdf: gpd.GeoDataFrame) -> dict[str, str]:
    """Map pandas dtypes to Fiona schema field types."""

    properties: dict[str, str] = {}
    for column_name, dtype in gdf.drop(columns="geometry", errors="ignore").dtypes.items():
        kind = str(dtype)
        if kind.startswith("int"):
            properties[column_name] = "int"
        elif kind.startswith("float"):
            properties[column_name] = "float"
        else:
            properties[column_name] = "str"
    return properties


def _cleanup_gpkg_sidecars(path: Path) -> None:
    """Remove SQLite sidecars left from a previous GeoPackage write."""

    for suffix in ("-wal", "-shm", ".wal", ".shm"):
        if suffix.startswith("-"):
            candidate = path.with_name(path.name + suffix)
        else:
            candidate = path.with_suffix(path.suffix + suffix)
        if candidate.exists():
            candidate.unlink()
