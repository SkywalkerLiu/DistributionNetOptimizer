from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString


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
class CorridorNode:
    """One candidate node inside the corridor graph."""

    node_id: str
    x: float
    y: float
    z: float
    row: int
    col: int
    kind: str


@dataclass(slots=True)
class CorridorEdge:
    """One feasible corridor edge between candidate nodes."""

    edge_id: str
    u: str
    v: str
    geometry: LineString
    horizontal_length_m: float
    length_3d_m: float
    build_cost: float
    terrain_cost: float
    risk_cost: float
    max_span_feasible: bool
    is_forbidden: bool
    slope_deg: float
    boundary_clearance_m: float


@dataclass(slots=True)
class CorridorGraph:
    """Candidate engineering corridor graph used by the V2 optimizer."""

    graph: nx.Graph
    nodes: dict[str, CorridorNode]
    edges: dict[str, CorridorEdge]
    corridor_mask: np.ndarray
    boundary_distance_m: np.ndarray
    resolution_m: float


@dataclass(slots=True)
class TransformerCandidate:
    """Transformer root candidate ranked on the corridor graph."""

    node_id: str
    x: float
    y: float
    z: float
    score: float
    rank: int


@dataclass(slots=True)
class AttachmentOption:
    """One feasible user-to-corridor attachment choice."""

    user_id: int
    attach_node_id: str
    horizontal_length_m: float
    length_3d_m: float
    cost: float


@dataclass(slots=True)
class RadialTreeResult:
    """A rooted radial tree selected from the corridor graph."""

    root_node_id: str
    selected_edge_ids: list[str]
    parent_by_node: dict[str, str]
    depth_by_node: dict[str, int]
    terminal_nodes: set[str]


@dataclass(slots=True)
class PowerFlowResult:
    """Electrical evaluation results for one radial solution."""

    edge_phase_loads: dict[tuple[str, str], np.ndarray]
    edge_phase_kw: dict[tuple[str, str], np.ndarray]
    edge_losses_kw: dict[tuple[str, str], float]
    edge_voltage_drop_pct: dict[tuple[str, str], np.ndarray]
    transformer_phase_loads: np.ndarray
    user_voltage_drop_pct: dict[int, float]
    user_service_drop_pct: dict[int, float]
    user_connection_nodes: dict[int, str]
    total_loss_kw: float
    max_voltage_drop_pct: float


@dataclass(slots=True)
class EvaluatedSolution:
    """Full evaluated candidate solution used for ranking."""

    transformer_candidate: TransformerCandidate
    radial_tree: RadialTreeResult
    attachment_choices: dict[int, AttachmentOption]
    phase_assignment: dict[int, str]
    power_flow: PowerFlowResult
    build_cost: float
    loss_cost: float
    total_unbalance_penalty: float
    objective: float
    feasible: bool
    voltage_ok: bool
    diagnostics: list[str] = field(default_factory=list)

