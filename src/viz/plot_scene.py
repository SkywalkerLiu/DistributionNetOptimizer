from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.io.raster_io import array_bounds


def generate_scene_plots(
    *,
    dtm: np.ndarray,
    slope: np.ndarray,
    forbidden_mask: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
    forest: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    manual_no_build: gpd.GeoDataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate all standard visualization products for the scene."""

    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "terrain_preview": plots_dir / "terrain_preview.png",
        "slope_preview": plots_dir / "slope_preview.png",
        "scene_overview": plots_dir / "scene_overview.png",
        "terrain_with_features": plots_dir / "terrain_with_features.png",
        "forbidden_mask": plots_dir / "forbidden_mask.png",
    }

    _save_raster_plot(
        array=dtm,
        profile=profile,
        title="Terrain Preview",
        colorbar_label="Elevation (m)",
        cmap="terrain",
        output_path=outputs["terrain_preview"],
    )
    _save_raster_plot(
        array=slope,
        profile=profile,
        title="Slope Preview",
        colorbar_label="Slope (deg)",
        cmap="viridis",
        output_path=outputs["slope_preview"],
    )
    _save_raster_plot(
        array=forbidden_mask,
        profile=profile,
        title="Forbidden Mask",
        colorbar_label="Forbidden",
        cmap="gray_r",
        output_path=outputs["forbidden_mask"],
    )
    _save_overlay_plot(
        background=dtm,
        profile=profile,
        title="Terrain With Features",
        output_path=outputs["terrain_with_features"],
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual_no_build,
    )
    _save_overlay_plot(
        background=dtm,
        profile=profile,
        title="Scene Overview",
        output_path=outputs["scene_overview"],
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual_no_build,
    )
    return outputs


def _save_raster_plot(
    *,
    array: np.ndarray,
    profile: dict[str, Any],
    title: str,
    colorbar_label: str,
    cmap: str,
    output_path: Path,
) -> None:
    """Render a single raster preview image."""

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = _extent(profile)
    image = ax.imshow(array, extent=extent, origin="upper", cmap=cmap)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.text(
        0.01,
        0.01,
        f"Resolution: {abs(profile['transform'].a):.2f} m/pixel",
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_overlay_plot(
    *,
    background: np.ndarray,
    profile: dict[str, Any],
    title: str,
    output_path: Path,
    users: gpd.GeoDataFrame,
    forest: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    manual_no_build: gpd.GeoDataFrame,
) -> None:
    """Render a terrain background with vector feature overlays."""

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(background, extent=_extent(profile), origin="upper", cmap="terrain", alpha=0.9)
    if not forest.empty:
        forest.plot(ax=ax, facecolor="#4a9e5c", edgecolor="#1d5e2e", alpha=0.20, linewidth=1.4)
    if not water.empty:
        water.plot(ax=ax, facecolor="#5ba8e8", edgecolor="#1a4a8f", alpha=0.25, linewidth=1.4)
    if not manual_no_build.empty:
        manual_no_build.plot(ax=ax, facecolor="none", edgecolor="#e03c2a", linewidth=2.0, linestyle="--")
    if not users.empty:
        users.plot(ax=ax, marker=".", markersize=10, facecolor="#000000", edgecolor="white", linewidth=0.5, alpha=0.90)

    legend_handles = [
        Patch(facecolor="#4a9e5c", edgecolor="#1d5e2e", alpha=0.20, linewidth=1.4, label="Forest"),
        Patch(facecolor="#5ba8e8", edgecolor="#1a4a8f", alpha=0.25, linewidth=1.4, label="Water"),
        Patch(facecolor="none", edgecolor="#e03c2a", linewidth=2.0, linestyle="--", label="Manual No-Build"),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="#000000",
            markeredgecolor="white",
            markersize=7,
            linewidth=0.5,
            label="Users",
        ),
    ]

    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9, edgecolor="gray")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.text(
        0.01,
        0.01,
        f"Resolution: {abs(profile['transform'].a):.2f} m/pixel",
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extent(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert raster profile metadata into a Matplotlib extent."""

    left, bottom, right, top = array_bounds(
        profile["transform"],
        int(profile["height"]),
        int(profile["width"]),
    )
    return left, right, bottom, top
