from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pyomo.environ as pyo

from src.planning.models import CorridorGraph, RadialTreeResult


@dataclass(slots=True)
class RadialTreeModelData:
    """Precomputed graph data reused by repeated Pyomo radial-tree solves."""

    node_ids: list[str]
    undirected_edges: list[tuple[str, str, float, str]]
    arcs: list[tuple[int, str, str]]
    outgoing: dict[str, list[int]]
    incoming: dict[str, list[int]]
    component_by_node: dict[str, int]


def solve_radial_tree(
    *,
    corridor: CorridorGraph,
    root_node_id: str,
    attachment_node_ids: list[str],
    planning_cfg: dict[str, Any] | None = None,
    model_data: RadialTreeModelData | None = None,
) -> RadialTreeResult:
    """Select a minimum-cost rooted radial tree with Pyomo + HiGHS."""

    terminals = sorted({root_node_id, *attachment_node_ids})
    if len(terminals) == 1:
        return RadialTreeResult(
            root_node_id=root_node_id,
            selected_edge_ids=[],
            parent_by_node={},
            depth_by_node={root_node_id: 0},
            terminal_nodes=set(terminals),
        )

    precomputed = model_data or build_radial_tree_model_data(corridor)
    missing = [terminal for terminal in terminals if terminal not in precomputed.component_by_node]
    if missing:
        raise ValueError(f"Terminals are missing from the corridor graph: {', '.join(missing)}.")
    root_component = precomputed.component_by_node[root_node_id]
    for terminal in terminals:
        if precomputed.component_by_node[terminal] != root_component:
            raise ValueError(f"Terminal {terminal} is not reachable from transformer candidate {root_node_id}.")

    selected_graph = _solve_highs_flow_tree(
        corridor=corridor,
        root_node_id=root_node_id,
        terminals=terminals,
        planning_cfg=planning_cfg or {},
        model_data=precomputed,
    )
    if not selected_graph.nodes:
        raise ValueError(f"HiGHS returned an empty radial tree for transformer candidate {root_node_id}.")

    rooted_source = selected_graph
    if not nx.is_tree(selected_graph):
        rooted_source = nx.minimum_spanning_tree(selected_graph, weight="weight")
    rooted = nx.bfs_tree(rooted_source, root_node_id)
    parent_by_node = {child: parent for parent, child in rooted.edges()}
    depth_by_node = dict(nx.single_source_shortest_path_length(rooted, root_node_id))

    selected_edge_ids = [
        str(rooted_source[parent][child]["edge_id"])
        for parent, child in rooted.edges()
    ]

    return RadialTreeResult(
        root_node_id=root_node_id,
        selected_edge_ids=selected_edge_ids,
        parent_by_node=parent_by_node,
        depth_by_node=depth_by_node,
        terminal_nodes=set(terminals),
    )


def build_radial_tree_model_data(corridor: CorridorGraph) -> RadialTreeModelData:
    """Build reusable directed-flow index data for the corridor graph."""

    node_ids = sorted(str(node_id) for node_id in corridor.graph.nodes)
    undirected_edges = [
        (str(u), str(v), float(data.get("weight", 0.0)), str(data["edge_id"]))
        for u, v, data in corridor.graph.edges(data=True)
    ]
    arcs: list[tuple[int, str, str]] = []
    outgoing: dict[str, list[int]] = {node_id: [] for node_id in node_ids}
    incoming: dict[str, list[int]] = {node_id: [] for node_id in node_ids}
    for edge_index, (u, v, _weight, _edge_id) in enumerate(undirected_edges):
        forward_index = len(arcs)
        arcs.append((edge_index, u, v))
        outgoing[u].append(forward_index)
        incoming[v].append(forward_index)
        reverse_index = len(arcs)
        arcs.append((edge_index, v, u))
        outgoing[v].append(reverse_index)
        incoming[u].append(reverse_index)

    component_by_node: dict[str, int] = {}
    for component_index, component in enumerate(nx.connected_components(corridor.graph)):
        for node_id in component:
            component_by_node[str(node_id)] = component_index

    return RadialTreeModelData(
        node_ids=node_ids,
        undirected_edges=undirected_edges,
        arcs=arcs,
        outgoing=outgoing,
        incoming=incoming,
        component_by_node=component_by_node,
    )


def _solve_highs_flow_tree(
    *,
    corridor: CorridorGraph,
    root_node_id: str,
    terminals: list[str],
    planning_cfg: dict[str, Any],
    model_data: RadialTreeModelData,
) -> nx.Graph:
    """Solve a directed single-commodity flow model over the undirected corridor graph."""

    solver_backend = str(planning_cfg.get("solver_backend", "highs")).lower()
    if solver_backend != "highs":
        raise ValueError(f"Unsupported solver backend '{solver_backend}'. This project currently uses only HiGHS.")

    undirected_edges = model_data.undirected_edges
    if not undirected_edges:
        raise ValueError("The corridor graph has no candidate edges for the radial tree model.")

    terminal_set = set(terminals)
    sink_terminals = sorted(terminal for terminal in terminal_set if terminal != root_node_id)
    required_flow = len(sink_terminals)
    model = pyo.ConcreteModel()
    model.E = pyo.RangeSet(0, len(undirected_edges) - 1)
    arcs = model_data.arcs
    model.A = pyo.RangeSet(0, len(arcs) - 1)
    model.x = pyo.Var(model.E, domain=pyo.Binary)
    model.flow = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    model.objective = pyo.Objective(
        expr=sum(weight * model.x[index] for index, (_u, _v, weight, _edge_id) in enumerate(undirected_edges)),
        sense=pyo.minimize,
    )

    def flow_capacity_rule(m: pyo.ConcreteModel, arc_index: int) -> pyo.Constraint:
        edge_index, _u, _v = arcs[arc_index]
        return m.flow[arc_index] <= required_flow * m.x[edge_index]

    model.flow_capacity = pyo.Constraint(model.A, rule=flow_capacity_rule)

    def conservation_rule(m: pyo.ConcreteModel, node_id: str) -> pyo.Constraint:
        outflow = sum(m.flow[arc_index] for arc_index in model_data.outgoing.get(node_id, []))
        inflow = sum(m.flow[arc_index] for arc_index in model_data.incoming.get(node_id, []))
        if node_id == root_node_id:
            rhs = required_flow
        elif node_id in sink_terminals:
            rhs = -1
        else:
            rhs = 0
        return outflow - inflow == rhs

    model.N = pyo.Set(initialize=model_data.node_ids)
    model.conservation = pyo.Constraint(model.N, rule=conservation_rule)

    solver = pyo.SolverFactory("highs")
    if not solver.available(exception_flag=False):
        raise ValueError("Pyomo HiGHS solver is not available. Install dependencies with: python -m pip install pyomo highspy")
    time_limit = float(planning_cfg.get("milp_time_limit_s", 180.0))
    if time_limit > 0.0:
        solver.options["time_limit"] = time_limit
    mip_gap = float(planning_cfg.get("mip_gap", 0.05))
    if mip_gap >= 0.0:
        solver.options["mip_rel_gap"] = mip_gap
    highs_threads = int(planning_cfg.get("highs_threads_per_worker", 1))
    if highs_threads > 0:
        solver.options["threads"] = highs_threads

    result = solver.solve(model, tee=False)
    termination = str(result.solver.termination_condition).lower()
    if termination not in {"optimal", "feasible", "maxtimelimit"}:
        raise ValueError(f"HiGHS radial tree model ended with termination condition: {termination}.")

    selected = nx.Graph()
    for terminal in terminals:
        selected.add_node(terminal)
    for edge_index, (u, v, weight, edge_id) in enumerate(undirected_edges):
        value = pyo.value(model.x[edge_index])
        if value is None or float(value) < 0.5:
            continue
        selected.add_edge(
            u,
            v,
            weight=float(weight),
            edge_id=str(edge_id),
        )

    if not all(nx.has_path(selected, root_node_id, terminal) for terminal in terminals):
        raise ValueError(f"HiGHS radial tree result does not connect all terminals for transformer candidate {root_node_id}.")
    connected_nodes = nx.node_connected_component(selected, root_node_id)
    return selected.subgraph(connected_nodes).copy()


def path_to_root(tree: RadialTreeResult, node_id: str) -> list[tuple[str, str]]:
    """Return the ordered parent-child path from the root to one node."""

    edges: list[tuple[str, str]] = []
    current = node_id
    while current in tree.parent_by_node:
        parent = tree.parent_by_node[current]
        edges.append((parent, current))
        current = parent
    edges.reverse()
    return edges
