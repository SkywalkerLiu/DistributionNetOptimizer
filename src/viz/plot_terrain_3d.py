from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _cyan_red_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.0, "#00ffff"),
        (0.25, "#00ff88"),
        (0.5, "#ffff00"),
        (0.75, "#ff8800"),
        (1.0, "#ff0000"),
    ]
    return LinearSegmentedColormap.from_list("cyan_red", colors)


@dataclass(slots=True)
class TerrainSurfacePreview:
    """Downsampled terrain surface prepared for 3D visualization."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sampled_height: int
    sampled_width: int
    original_height: int
    original_width: int


@dataclass(slots=True)
class OverlayLine3D:
    """A 3D line draped on the terrain surface."""

    label: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: str
    width: float


@dataclass(slots=True)
class OverlayPoint3D:
    """A 3D point set placed slightly above the terrain surface."""

    label: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: str
    size: float
    text: list[str] | None = None


def generate_terrain_3d_previews(
    *,
    dtm: np.ndarray,
    profile: dict[str, Any],
    output_dir: str | Path,
    visualization_config: dict[str, Any] | None = None,
    users: gpd.GeoDataFrame | None = None,
    forest: gpd.GeoDataFrame | None = None,
    water: gpd.GeoDataFrame | None = None,
    manual_no_build: gpd.GeoDataFrame | None = None,
    planned_lines: gpd.GeoDataFrame | None = None,
) -> dict[str, Path]:
    """Generate static and interactive 3D terrain preview files."""

    cfg = visualization_config or {}
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    max_grid_size = int(cfg.get("terrain_3d_max_grid_size", 200))
    vertical_exaggeration = float(cfg.get("terrain_3d_vertical_exaggeration", 3.0))
    elev_deg = float(cfg.get("terrain_3d_camera_elev_deg", 40.0))
    azim_deg = float(cfg.get("terrain_3d_camera_azim_deg", -60.0))

    surface = downsample_terrain_surface(
        dtm=dtm,
        profile=profile,
        max_grid_size=max_grid_size,
    )
    lines, points = build_scene_3d_overlays(
        dtm=dtm,
        profile=profile,
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual_no_build,
        planned_lines=planned_lines,
    )

    outputs = {
        "terrain_3d_png": plots_dir / "terrain_3d_preview.png",
        "terrain_3d_html": plots_dir / "terrain_3d_preview.html",
    }
    _save_matplotlib_surface(
        surface=surface,
        output_path=outputs["terrain_3d_png"],
        vertical_exaggeration=vertical_exaggeration,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        lines=lines,
        points=points,
    )
    _save_plotly_surface(
        surface=surface,
        output_path=outputs["terrain_3d_html"],
        vertical_exaggeration=vertical_exaggeration,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        lines=lines,
        points=points,
    )
    return outputs


def downsample_terrain_surface(
    *,
    dtm: np.ndarray,
    profile: dict[str, Any],
    max_grid_size: int,
) -> TerrainSurfacePreview:
    """Downsample a terrain raster into a manageable grid for 3D preview."""

    if dtm.ndim != 2:
        raise ValueError("DTM must be a 2D array for 3D preview.")
    if max_grid_size < 2:
        raise ValueError("max_grid_size must be at least 2.")

    height, width = dtm.shape
    sampled_height = min(height, max_grid_size)
    sampled_width = min(width, max_grid_size)

    row_idx = np.unique(np.linspace(0, height - 1, sampled_height, dtype=int))
    col_idx = np.unique(np.linspace(0, width - 1, sampled_width, dtype=int))
    sampled = dtm[np.ix_(row_idx, col_idx)].astype(np.float32)

    transform = profile["transform"]
    x_coords = transform.c + (col_idx + 0.5) * transform.a
    y_coords = transform.f + (row_idx + 0.5) * transform.e
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    return TerrainSurfacePreview(
        x=x_grid.astype(np.float32),
        y=y_grid.astype(np.float32),
        z=sampled,
        sampled_height=int(sampled.shape[0]),
        sampled_width=int(sampled.shape[1]),
        original_height=height,
        original_width=width,
    )


def build_scene_3d_overlays(
    *,
    dtm: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame | None = None,
    forest: gpd.GeoDataFrame | None = None,
    water: gpd.GeoDataFrame | None = None,
    manual_no_build: gpd.GeoDataFrame | None = None,
    planned_lines: gpd.GeoDataFrame | None = None,
) -> tuple[list[OverlayLine3D], list[OverlayPoint3D]]:
    """Build 3D overlay traces for points and boundaries on the terrain."""

    z_span = float(dtm.max() - dtm.min()) or 1.0
    line_offset = max(0.6, z_span * 0.003)
    point_offset = max(1.2, z_span * 0.006)

    lines: list[OverlayLine3D] = []
    points: list[OverlayPoint3D] = []

    lines.extend(
        _build_line_overlays(
            gdf=forest,
            dtm=dtm,
            profile=profile,
            label="Forest",
            color="#1d5e2e",
            width=3.0,
            z_offset=line_offset,
        )
    )
    lines.extend(
        _build_line_overlays(
            gdf=water,
            dtm=dtm,
            profile=profile,
            label="Water",
            color="#1a4a8f",
            width=3.4,
            z_offset=line_offset * 1.15,
        )
    )
    lines.extend(
        _build_line_overlays(
            gdf=manual_no_build,
            dtm=dtm,
            profile=profile,
            label="Manual No-Build",
            color="#e03c2a",
            width=3.6,
            z_offset=line_offset * 1.3,
        )
    )
    lines.extend(
        _build_line_overlays(
            gdf=planned_lines,
            dtm=dtm,
            profile=profile,
            label="Planned Lines",
            color="#f3a712",
            width=3.2,
            z_offset=line_offset * 1.45,
        )
    )

    user_points = _build_point_overlay(
        gdf=users,
        dtm=dtm,
        profile=profile,
        label="Users",
        color="#000000",
        size=3.0,
        z_offset=point_offset,
        text_column="user_id",
        text_prefix="user_id",
    )
    if user_points is not None:
        points.append(user_points)

    return lines, points


def _save_matplotlib_surface(
    *,
    surface: TerrainSurfacePreview,
    output_path: Path,
    vertical_exaggeration: float,
    elev_deg: float,
    azim_deg: float,
    lines: list[OverlayLine3D],
    points: list[OverlayPoint3D],
) -> None:
    """Save a static PNG terrain surface preview."""

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    plot = ax.plot_surface(
        surface.x,
        surface.y,
        surface.z,
        cmap=_cyan_red_colormap(),
        linewidth=0.0,
        antialiased=True,
        alpha=0.70,
    )
    colorbar = fig.colorbar(plot, ax=ax, shrink=0.7, pad=0.08)
    colorbar.set_label("Elevation (m)")

    seen_labels: set[str] = set()
    for line in lines:
        label = line.label if line.label not in seen_labels else "_nolegend_"
        ax.plot(
            line.x,
            line.y,
            line.z,
            color=line.color,
            linewidth=line.width,
            label=label,
            zorder=50,
        )
        seen_labels.add(line.label)

    for point in points:
        label = point.label if point.label not in seen_labels else "_nolegend_"
        ax.scatter(
            point.x,
            point.y,
            point.z,
            color=point.color,
            s=(point.size * 2) ** 2,
            marker="o",
            depthshade=False,
            edgecolors="white",
            linewidths=0.5,
            label=label,
            zorder=100,
        )
        seen_labels.add(point.label)

    ax.set_title(
        "Terrain 3D Preview With Scene Overlays\n"
        f"sampled {surface.sampled_width}x{surface.sampled_height} from "
        f"{surface.original_width}x{surface.original_height}"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Elevation (m)")
    ax.view_init(elev=elev_deg, azim=azim_deg)

    x_span = float(surface.x.max() - surface.x.min()) or 1.0
    y_span = float(surface.y.max() - surface.y.min()) or 1.0
    z_span = float(surface.z.max() - surface.z.min()) or 1.0
    try:
        ax.set_box_aspect((x_span, y_span, z_span * max(vertical_exaggeration, 1.0)))
    except AttributeError:
        pass

    if seen_labels:
        ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_plotly_surface(
    *,
    surface: TerrainSurfacePreview,
    output_path: Path,
    vertical_exaggeration: float,
    elev_deg: float,
    azim_deg: float,
    lines: list[OverlayLine3D],
    points: list[OverlayPoint3D],
) -> None:
    """Save an interactive HTML terrain surface preview."""

    x_span = float(surface.x.max() - surface.x.min()) or 1.0
    y_span = float(surface.y.max() - surface.y.min()) or 1.0
    z_span = float(surface.z.max() - surface.z.min()) or 1.0
    span = max(x_span, y_span, 1.0)
    aspect_z = max((z_span / span) * max(vertical_exaggeration, 1.0), 0.05)

    fig = go.Figure(
        data=[
            go.Surface(
                x=surface.x,
                y=surface.y,
                z=surface.z,
                colorscale=[
                    [0.0, "#00ffff"],
                    [0.25, "#00ff88"],
                    [0.5, "#ffff00"],
                    [0.75, "#ff8800"],
                    [1.0, "#ff0000"],
                ],
                colorbar={"title": "Elevation (m)"},
                hovertemplate="X=%{x:.1f} m<br>Y=%{y:.1f} m<br>Z=%{z:.2f} m<extra></extra>",
                name="Terrain",
                showscale=True,
            )
        ]
    )

    shown_labels: set[str] = set()
    for line in lines:
        fig.add_trace(
            go.Scatter3d(
                x=line.x,
                y=line.y,
                z=line.z,
                mode="lines",
                name=line.label,
                showlegend=line.label not in shown_labels,
                legendgroup=line.label,
                line={"color": line.color, "width": line.width},
                hovertemplate=f"{line.label}<br>X=%{{x:.1f}} m<br>Y=%{{y:.1f}} m<br>Z=%{{z:.2f}} m<extra></extra>",
            )
        )
        shown_labels.add(line.label)

    for point in points:
        fig.add_trace(
            go.Scatter3d(
                x=point.x,
                y=point.y,
                z=point.z,
                mode="markers",
                name=point.label,
                showlegend=point.label not in shown_labels,
                legendgroup=point.label,
                marker={
                    "size": point.size,
                    "color": point.color,
                    "symbol": "circle",
                    "line": {"color": "white", "width": 0.5},
                },
                text=point.text,
                hovertemplate=_point_hover_template(point.label, point.text),
            )
        )
        shown_labels.add(point.label)

    fig.update_layout(
        title=(
            "Terrain 3D Preview With Scene Overlays"
            f" (sampled {surface.sampled_width}x{surface.sampled_height} from "
            f"{surface.original_width}x{surface.original_height})"
        ),
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Elevation (m)",
            "aspectmode": "manual",
            "aspectratio": {"x": x_span / span, "y": y_span / span, "z": aspect_z},
            "camera": {"eye": _camera_eye(elev_deg=elev_deg, azim_deg=azim_deg)},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def _build_line_overlays(
    *,
    gdf: gpd.GeoDataFrame | None,
    dtm: np.ndarray,
    profile: dict[str, Any],
    label: str,
    color: str,
    width: float,
    z_offset: float,
) -> list[OverlayLine3D]:
    """Convert polygon or line layers into terrain-draped 3D line overlays."""

    if gdf is None or gdf.empty:
        return []

    overlays: list[OverlayLine3D] = []
    for geometry in gdf.geometry:
        for coords in _iter_line_coordinate_sequences(geometry):
            simplified = _thin_coordinates(coords, max_points=240)
            xs = simplified[:, 0].astype(np.float32)
            ys = simplified[:, 1].astype(np.float32)
            zs = _sample_surface_elevation(xs=xs, ys=ys, dtm=dtm, profile=profile) + z_offset
            overlays.append(
                OverlayLine3D(
                    label=label,
                    x=xs,
                    y=ys,
                    z=zs,
                    color=color,
                    width=width,
                )
            )
    return overlays


def _build_point_overlay(
    *,
    gdf: gpd.GeoDataFrame | None,
    dtm: np.ndarray,
    profile: dict[str, Any],
    label: str,
    color: str,
    size: float,
    z_offset: float,
    text_column: str | None = None,
    text_prefix: str | None = None,
) -> OverlayPoint3D | None:
    """Convert a point layer into a terrain-aware 3D point overlay."""

    if gdf is None or gdf.empty:
        return None

    xs = gdf.geometry.x.to_numpy(dtype=np.float32)
    ys = gdf.geometry.y.to_numpy(dtype=np.float32)
    if "elev_m" in gdf.columns:
        base_z = gdf["elev_m"].to_numpy(dtype=np.float32)
    else:
        base_z = _sample_surface_elevation(xs=xs, ys=ys, dtm=dtm, profile=profile)
    zs = base_z + z_offset

    text: list[str] | None = None
    if text_column is not None and text_column in gdf.columns:
        if text_prefix is None:
            text_prefix = text_column
        text = [f"{text_prefix}: {value}" for value in gdf[text_column].tolist()]

    return OverlayPoint3D(
        label=label,
        x=xs,
        y=ys,
        z=zs.astype(np.float32),
        color=color,
        size=size,
        text=text,
    )


def _iter_line_coordinate_sequences(geometry: Any):
    """Yield line coordinate arrays from polygon, multi-polygon, or line geometry."""

    if geometry is None or geometry.is_empty:
        return

    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        yield np.asarray(geometry.exterior.coords, dtype=np.float32)
        for ring in geometry.interiors:
            yield np.asarray(ring.coords, dtype=np.float32)
    elif geom_type == "MultiPolygon":
        for part in geometry.geoms:
            yield from _iter_line_coordinate_sequences(part)
    elif geom_type == "LineString":
        yield np.asarray(geometry.coords, dtype=np.float32)
    elif geom_type == "MultiLineString":
        for part in geometry.geoms:
            yield np.asarray(part.coords, dtype=np.float32)
    elif geom_type == "LinearRing":
        yield np.asarray(geometry.coords, dtype=np.float32)


def _thin_coordinates(coords: np.ndarray, *, max_points: int) -> np.ndarray:
    """Reduce coordinate count for overlays while preserving shape endpoints."""

    if len(coords) <= max_points:
        return coords
    keep = np.unique(np.linspace(0, len(coords) - 1, max_points, dtype=int))
    return coords[keep]


def _sample_surface_elevation(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    dtm: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    """Sample terrain elevation at XY coordinates using nearest raster cells."""

    transform = profile["transform"]
    cols = np.rint((xs - transform.c) / transform.a - 0.5).astype(int)
    rows = np.rint((ys - transform.f) / transform.e - 0.5).astype(int)
    rows = np.clip(rows, 0, dtm.shape[0] - 1)
    cols = np.clip(cols, 0, dtm.shape[1] - 1)
    return dtm[rows, cols].astype(np.float32)


def _point_hover_template(label: str, text: list[str] | None) -> str:
    """Build a hover template for point overlays."""

    if text:
        return (
            f"{label}<br>%{{text}}<br>X=%{{x:.1f}} m"
            "<br>Y=%{y:.1f} m<br>Z=%{z:.2f} m<extra></extra>"
        )
    return (
        f"{label}<br>X=%{{x:.1f}} m"
        "<br>Y=%{y:.1f} m<br>Z=%{z:.2f} m<extra></extra>"
    )


def _camera_eye(*, elev_deg: float, azim_deg: float) -> dict[str, float]:
    """Convert matplotlib-like view angles into a Plotly camera eye position."""

    elev_rad = np.deg2rad(elev_deg)
    azim_rad = np.deg2rad(azim_deg)
    radius = 1.8
    return {
        "x": float(radius * np.cos(elev_rad) * np.cos(azim_rad)),
        "y": float(radius * np.cos(elev_rad) * np.sin(azim_rad)),
        "z": float(radius * np.sin(elev_rad)),
    }
