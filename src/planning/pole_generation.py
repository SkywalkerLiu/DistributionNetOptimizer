from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from src.planning.common import point_metrics, sample_array
from src.planning.geometry_constraints import segment_min_clearance
from src.planning.models import AttachmentOption, CorridorGraph, EvaluatedSolution, PHASE_INDEX


def generate_plan_layers(
    *,
    corridor: CorridorGraph,
    solution: EvaluatedSolution,
    users: gpd.GeoDataFrame,
    dtm: np.ndarray,
    profile: dict[str, Any],
    planning_cfg: dict[str, Any],
    crs: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[int, str]]:
    """Generate transformer, pole, and planned-line layers for the chosen V2 solution."""

    transformer_candidate = solution.transformer_candidate
    transformer_layer = gpd.GeoDataFrame(
        [
            {
                "transformer_id": "TX1",
                "candidate_id": int(transformer_candidate.rank),
                "capacity_kva": float(planning_cfg.get("transformer_capacity_kva", 630.0)),
                "fixed_cost": float(planning_cfg.get("transformer_fixed_cost", 120000.0)),
                "elev_m": float(transformer_candidate.z),
                "ground_slope_deg": float(sample_array(np.zeros_like(dtm), profile, transformer_candidate.x, transformer_candidate.y)),
                "buildable_score": float(transformer_candidate.score),
                "source": "planning_v2",
            }
        ],
        geometry=[Point(float(transformer_candidate.x), float(transformer_candidate.y))],
        crs=crs,
    )

    poles_rows: list[dict[str, Any]] = []
    pole_geometries: list[Point] = []
    public_node_id: dict[str, str] = {solution.radial_tree.root_node_id: "TX1"}
    support_registry = {
        "TX1": {
            "public_id": "TX1",
            "x": float(transformer_candidate.x),
            "y": float(transformer_candidate.y),
            "ground_z": float(transformer_candidate.z),
            "pole_height_m": float(planning_cfg.get("transformer_lv_connection_height_m", 8.5)),
            "support_top_z": float(transformer_candidate.z)
            + float(planning_cfg.get("transformer_lv_connection_height_m", 8.5)),
            "kind": "transformer",
        }
    }

    next_pole_id = 1
    for node_id in sorted(solution.radial_tree.depth_by_node, key=lambda item: solution.radial_tree.depth_by_node[item]):
        if node_id == solution.radial_tree.root_node_id:
            continue
        node = corridor.nodes[node_id]
        pole_id = f"P{next_pole_id:04d}"
        next_pole_id += 1
        public_node_id[node_id] = pole_id
        support_registry[pole_id] = {
            "public_id": pole_id,
            "x": float(node.x),
            "y": float(node.y),
            "ground_z": float(node.z),
            "pole_height_m": float(planning_cfg.get("lv_pole_height_m", 9.5)),
            "support_top_z": float(node.z) + float(planning_cfg.get("lv_pole_height_m", 9.5)),
            "kind": "pole",
        }
        poles_rows.append(
            {
                "pole_id": pole_id,
                "candidate_id": next_pole_id - 1,
                "pole_type": "lv_pole",
                "pole_height_m": float(planning_cfg.get("lv_pole_height_m", 9.5)),
                "fixed_cost": float(planning_cfg.get("pole_fixed_cost", 1800.0)),
                "elev_m": float(node.z),
                "ground_slope_deg": 0.0,
                "source": node.kind,
            }
        )
        pole_geometries.append(Point(float(node.x), float(node.y)))

    planned_line_rows: list[dict[str, Any]] = []
    planned_line_geometries: list[LineString] = []
    line_id = 1

    for child, parent in solution.radial_tree.parent_by_node.items():
        parent_public = public_node_id[parent]
        child_public = public_node_id[child]
        edge = corridor.edges[corridor.graph[parent][child]["edge_id"]]
        phase_load = solution.power_flow.edge_phase_loads.get((parent, child), np.zeros(3, dtype=float))
        neutral_current = _neutral_current_a(phase_load, float(planning_cfg.get("low_voltage_phase_v", 230.0)))
        voltage_drop_pct = float(np.max(solution.power_flow.edge_voltage_drop_pct.get((parent, child), np.zeros(3, dtype=float))))
        span_supports = _split_edge_supports(
            start=support_registry[parent_public],
            end=support_registry[child_public],
            max_span_m=float(planning_cfg.get("max_pole_span_m", 50.0)),
            next_pole_id_start=next_pole_id,
            planning_cfg=planning_cfg,
            dtm=dtm,
            profile=profile,
        )
        next_pole_id = span_supports["next_pole_id"]
        for support in span_supports["inserted"]:
            support_registry[support["public_id"]] = support
            poles_rows.append(
                {
                    "pole_id": support["public_id"],
                    "candidate_id": int(str(support["public_id"]).removeprefix("P") or "0"),
                    "pole_type": "lv_pole",
                    "pole_height_m": float(support["pole_height_m"]),
                    "fixed_cost": float(planning_cfg.get("pole_fixed_cost", 1800.0)),
                    "elev_m": float(support["ground_z"]),
                    "ground_slope_deg": 0.0,
                    "source": "span_split",
                }
            )
            pole_geometries.append(Point(float(support["x"]), float(support["y"])))

        segments = span_supports["supports"]
        total_length = max(float(edge.length_3d_m), 1e-9)
        for start_support, end_support in zip(segments[:-1], segments[1:]):
            metrics = point_metrics(start_support, end_support)
            clearance = segment_min_clearance(
                start=start_support,
                end=end_support,
                dtm=dtm,
                profile=profile,
                sample_step_m=float(planning_cfg.get("clearance_sample_step_m", 2.0)),
            )
            required_clearance = float(planning_cfg.get("lv_ground_clearance_m", 4.0))
            planned_line_rows.append(
                {
                    "line_id": line_id,
                    "line_type": "lv_line",
                    "from_node": start_support["public_id"],
                    "to_node": end_support["public_id"],
                    "phase_set": "ABCN",
                    "service_phase": "",
                    "horizontal_length_m": float(metrics["horizontal_length_m"]),
                    "length_3d_m": float(metrics["length_3d_m"]),
                    "dz_m": float(metrics["dz_m"]),
                    "slope_deg": float(metrics["slope_deg"]),
                    "cost": float(metrics["length_3d_m"]) * float(planning_cfg.get("line_cost_per_m", 55.0)),
                    "load_a_kva": float(phase_load[0]),
                    "load_b_kva": float(phase_load[1]),
                    "load_c_kva": float(phase_load[2]),
                    "neutral_current_a": float(neutral_current),
                    "voltage_drop_pct": float(voltage_drop_pct * metrics["length_3d_m"] / total_length),
                    "support_z_start_m": float(start_support["support_top_z"]),
                    "support_z_end_m": float(end_support["support_top_z"]),
                    "min_clearance_m": float(clearance),
                    "required_clearance_m": float(required_clearance),
                    "is_violation": int(clearance + 1e-9 < required_clearance),
                }
            )
            planned_line_geometries.append(
                LineString([(float(start_support["x"]), float(start_support["y"])), (float(end_support["x"]), float(end_support["y"]))])
            )
            line_id += 1

    user_connection_public: dict[int, str] = {}
    for row in users.itertuples():
        user_id = int(row.user_id)
        option = solution.attachment_choices[user_id]
        attach_public = public_node_id.get(option.attach_node_id, "TX1")
        attach_support = support_registry[attach_public]
        assigned_phase = solution.phase_assignment[user_id]
        phase_load = np.zeros(3, dtype=float)
        if assigned_phase == "ABC":
            phase_load[:] = float(row.apparent_kva) / 3.0
        else:
            phase_load[PHASE_INDEX[assigned_phase]] = float(row.apparent_kva)
        user_support = {
            "public_id": f"user_{user_id}",
            "x": float(row.geometry.x),
            "y": float(row.geometry.y),
            "ground_z": float(row.elev_m),
            "support_top_z": float(row.elev_m) + float(planning_cfg.get("user_attachment_height_m", 4.0)),
            "kind": "user",
        }
        metrics = point_metrics(attach_support, user_support)
        clearance = segment_min_clearance(
            start=attach_support,
            end=user_support,
            dtm=dtm,
            profile=profile,
            sample_step_m=float(planning_cfg.get("clearance_sample_step_m", 2.0)),
        )
        required_clearance = float(planning_cfg.get("service_ground_clearance_m", 2.3))
        planned_line_rows.append(
            {
                "line_id": line_id,
                "line_type": "service_drop",
                "from_node": attach_public,
                "to_node": f"user_{user_id}",
                "phase_set": assigned_phase if assigned_phase != "ABC" else "ABC",
                "service_phase": assigned_phase,
                "horizontal_length_m": float(metrics["horizontal_length_m"]),
                "length_3d_m": float(metrics["length_3d_m"]),
                "dz_m": float(metrics["dz_m"]),
                "slope_deg": float(metrics["slope_deg"]),
                "cost": float(metrics["length_3d_m"]) * float(planning_cfg.get("service_line_cost_per_m", 35.0)),
                "load_a_kva": float(phase_load[0]),
                "load_b_kva": float(phase_load[1]),
                "load_c_kva": float(phase_load[2]),
                "neutral_current_a": float(_neutral_current_a(phase_load, float(planning_cfg.get("low_voltage_phase_v", 230.0)))),
                "voltage_drop_pct": float(solution.power_flow.user_service_drop_pct.get(user_id, 0.0)),
                "support_z_start_m": float(attach_support["support_top_z"]),
                "support_z_end_m": float(user_support["support_top_z"]),
                "min_clearance_m": float(clearance),
                "required_clearance_m": float(required_clearance),
                "is_violation": int(clearance + 1e-9 < required_clearance),
            }
        )
        planned_line_geometries.append(
            LineString([(float(attach_support["x"]), float(attach_support["y"])), (float(row.geometry.x), float(row.geometry.y))])
        )
        line_id += 1
        user_connection_public[user_id] = attach_public

    poles_layer = gpd.GeoDataFrame(poles_rows, geometry=pole_geometries, crs=crs)
    planned_lines = gpd.GeoDataFrame(planned_line_rows, geometry=planned_line_geometries, crs=crs)
    return transformer_layer, poles_layer, planned_lines, user_connection_public


def _split_edge_supports(
    *,
    start: dict[str, Any],
    end: dict[str, Any],
    max_span_m: float,
    next_pole_id_start: int,
    planning_cfg: dict[str, Any],
    dtm: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Split one corridor edge into support-to-support spans."""

    metrics = point_metrics(start, end)
    segment_count = max(1, int(math.ceil(metrics["horizontal_length_m"] / max(max_span_m, 1.0))))
    supports = [start]
    inserted: list[dict[str, Any]] = []
    next_pole_id = next_pole_id_start
    for index in range(1, segment_count):
        fraction = index / segment_count
        x = float(start["x"]) + (float(end["x"]) - float(start["x"])) * fraction
        y = float(start["y"]) + (float(end["y"]) - float(start["y"])) * fraction
        ground_z = float(sample_array(dtm, profile, x, y))
        pole_id = f"P{next_pole_id:04d}"
        next_pole_id += 1
        support = {
            "public_id": pole_id,
            "x": float(x),
            "y": float(y),
            "ground_z": float(ground_z),
            "pole_height_m": float(planning_cfg.get("lv_pole_height_m", 9.5)),
            "support_top_z": float(ground_z) + float(planning_cfg.get("lv_pole_height_m", 9.5)),
            "kind": "pole",
        }
        inserted.append(support)
        supports.append(support)
    supports.append(end)
    return {"supports": supports, "inserted": inserted, "next_pole_id": next_pole_id}


def _neutral_current_a(load: np.ndarray, phase_voltage_v: float) -> float:
    """Estimate neutral current from phase apparent power."""

    currents = (load * 1000.0) / max(phase_voltage_v, 1.0)
    ia, ib, ic = map(float, currents)
    neutral_sq = max(ia**2 + ib**2 + ic**2 - ia * ib - ib * ic - ic * ia, 0.0)
    return float(math.sqrt(neutral_sq))
