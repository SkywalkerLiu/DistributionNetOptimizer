from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.io.raster_io import array_bounds
from src.viz.plot_terrain_3d import generate_terrain_3d_previews


PHASE_COLORS = {"A": "#ffd400", "B": "#2ca25f", "C": "#e53935", "ABC": "#984ea3", "": "#777777"}
POLE_STYLES = {
    "hv_pole": {"color": "#7b1fa2", "marker": "^", "size": 28, "label": "HV Pole"},
    "hv_lv_shared": {"color": "#ad8b00", "marker": "s", "size": 30, "label": "Shared HV/LV Pole"},
    "lv_pole": {"color": "#00c7d9", "marker": "D", "size": 22, "label": "LV Pole"},
    "clearance_repair": {"color": "#ff5ea8", "marker": "P", "size": 32, "label": "Clearance Repair Pole"},
}


def generate_optimized_plan_plots(
    *,
    dtm: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
    forest: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    manual_no_build: gpd.GeoDataFrame,
    transformer: gpd.GeoDataFrame,
    poles: gpd.GeoDataFrame,
    planned_lines: gpd.GeoDataFrame,
    output_dir: str | Path,
    visualization_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate 2D, static 3D, and interactive 3D optimized-plan plots."""

    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "optimized_plan_2d": plots_dir / "optimized_plan_2d.png",
        "optimized_plan_3d_static": plots_dir / "optimized_plan_3d_static.png",
        "optimized_plan_3d_dynamic": plots_dir / "optimized_plan_3d_dynamic.html",
    }

    _save_optimized_2d(
        dtm=dtm,
        profile=profile,
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual_no_build,
        transformer=transformer,
        poles=poles,
        planned_lines=planned_lines,
        output_path=outputs["optimized_plan_2d"],
    )
    preview_outputs = generate_terrain_3d_previews(
        dtm=dtm,
        profile=profile,
        output_dir=plots_dir,
        visualization_config=visualization_config or {},
        users=users,
        forest=forest,
        water=water,
        manual_no_build=manual_no_build,
        planned_lines=planned_lines,
        transformer=transformer,
        poles=poles,
    )
    shutil.copyfile(preview_outputs["terrain_3d_png"], outputs["optimized_plan_3d_static"])
    shutil.copyfile(preview_outputs["terrain_3d_html"], outputs["optimized_plan_3d_dynamic"])
    return outputs


def _save_optimized_2d(
    *,
    dtm: np.ndarray,
    profile: dict[str, Any],
    users: gpd.GeoDataFrame,
    forest: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    manual_no_build: gpd.GeoDataFrame,
    transformer: gpd.GeoDataFrame,
    poles: gpd.GeoDataFrame,
    planned_lines: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Render an optimized plan map with phase-colored users and service drops."""

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(dtm, extent=_extent(profile), origin="upper", cmap="terrain", alpha=0.88)
    if not forest.empty:
        forest.plot(ax=ax, facecolor="#2e7d32", edgecolor="#0b3d16", alpha=0.28, linewidth=1.2)
    if not water.empty:
        water.plot(ax=ax, facecolor="#64b5f6", edgecolor="#0d47a1", alpha=0.32, linewidth=1.2)
    if not manual_no_build.empty:
        manual_no_build.plot(ax=ax, facecolor="none", edgecolor="#e03c2a", linestyle="--", linewidth=1.8)

    if not planned_lines.empty:
        _plot_lines_by_type(ax=ax, planned_lines=planned_lines)
    if not poles.empty:
        _plot_poles_by_type(ax=ax, poles=poles)
    if not transformer.empty:
        transformer.plot(ax=ax, marker="s", markersize=80, facecolor="#d7191c", edgecolor="white", linewidth=1.0, zorder=10)
    if not users.empty:
        phase_values = users["assigned_phase"] if "assigned_phase" in users.columns else np.full(len(users), "")
        for phase, color in PHASE_COLORS.items():
            subset = users.loc[phase_values == phase]
            if subset.empty:
                continue
            subset.plot(ax=ax, marker="o", markersize=24, facecolor=color, edgecolor="white", linewidth=0.6, zorder=12)

    handles = [
        Patch(facecolor="#2e7d32", edgecolor="#0b3d16", alpha=0.28, label="Forest (Forbidden)"),
        Patch(facecolor="#64b5f6", edgecolor="#0d47a1", alpha=0.32, label="Water"),
        Patch(facecolor="none", edgecolor="#e03c2a", linestyle="--", label="Manual No-Build"),
        Line2D([0], [0], color="#c44e52", linewidth=2.4, label="HV Line"),
        Line2D([0], [0], color="#222222", linewidth=1.8, label="LV ABCN Line"),
        Line2D([0], [0], color="#ff7f00", linewidth=1.2, label="Service Drop"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d7191c", markeredgecolor="white", markersize=9, label="Transformer"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=POLE_STYLES["hv_pole"]["color"], markeredgecolor="white", markersize=7, label="HV Pole"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=POLE_STYLES["lv_pole"]["color"], markeredgecolor="white", markersize=7, label="LV Pole"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=POLE_STYLES["hv_lv_shared"]["color"], markeredgecolor="white", markersize=7, label="Shared HV/LV Pole"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLORS["A"], markeredgecolor="white", markersize=7, label="Phase A User"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLORS["B"], markeredgecolor="white", markersize=7, label="Phase B User"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLORS["C"], markeredgecolor="white", markersize=7, label="Phase C User"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title("Optimized Distribution Network Plan")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.text(
        0.01,
        0.01,
        f"Resolution: {abs(profile['transform'].a):.2f} m/pixel",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_lines_by_type(*, ax: Any, planned_lines: gpd.GeoDataFrame) -> None:
    """Plot planned lines by type and service phase."""

    styles = {
        "hv_line": {"color": "#c44e52", "linewidth": 2.4, "alpha": 0.95},
        "lv_line": {"color": "#222222", "linewidth": 1.8, "alpha": 0.88},
        "service_drop": {"color": "#ff7f00", "linewidth": 1.2, "alpha": 0.85},
    }
    for line_type, style in styles.items():
        subset = planned_lines.loc[planned_lines["line_type"] == line_type]
        if not subset.empty:
            subset.plot(ax=ax, **style, zorder=7 if line_type != "service_drop" else 9)


def _plot_poles_by_type(*, ax: Any, poles: gpd.GeoDataFrame) -> None:
    """Plot poles with different styles for HV, LV, and shared supports."""

    for pole_type, style in POLE_STYLES.items():
        subset = poles.loc[poles["pole_type"] == pole_type]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            marker=style["marker"],
            markersize=style["size"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=8,
        )


def _extent(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert raster profile metadata into a Matplotlib extent."""

    left, bottom, right, top = array_bounds(
        profile["transform"],
        int(profile["height"]),
        int(profile["width"]),
    )
    return left, right, bottom, top
