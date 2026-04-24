from __future__ import annotations

from typing import Any

import networkx as nx

from src.planning.models import CorridorGraph, RadialTreeResult


def solve_radial_tree(
    *,
    corridor: CorridorGraph,
    root_node_id: str,
    attachment_node_ids: list[str],
) -> RadialTreeResult:
    """Approximate the rooted radial tree using a Steiner-style MST expansion."""

    terminals = sorted({root_node_id, *attachment_node_ids})
    if len(terminals) == 1:
        return RadialTreeResult(
            root_node_id=root_node_id,
            selected_edge_ids=[],
            parent_by_node={},
            depth_by_node={root_node_id: 0},
            terminal_nodes=set(terminals),
        )

    metric_closure = nx.Graph()
    shortest_paths: dict[tuple[str, str], list[str]] = {}
    for terminal in terminals:
        lengths, paths = nx.single_source_dijkstra(corridor.graph, terminal, weight="weight")
        for other in terminals:
            if other == terminal:
                continue
            if other not in lengths:
                raise ValueError(f"Terminal {other} is not reachable from transformer candidate {root_node_id}.")
            metric_closure.add_edge(terminal, other, weight=float(lengths[other]))
            shortest_paths[(terminal, other)] = list(paths[other])

    terminal_tree = nx.minimum_spanning_tree(metric_closure, weight="weight")
    expanded = nx.Graph()
    for u, v in terminal_tree.edges():
        path = shortest_paths[(u, v)]
        expanded.add_nodes_from(path)
        expanded.add_edges_from(zip(path[:-1], path[1:]))

    weighted_subgraph = corridor.graph.subgraph(expanded.nodes()).copy()
    steiner_tree = nx.minimum_spanning_tree(weighted_subgraph, weight="weight")
    rooted = nx.bfs_tree(steiner_tree, root_node_id)
    parent_by_node = {child: parent for parent, child in rooted.edges()}
    depth_by_node = dict(nx.single_source_shortest_path_length(rooted, root_node_id))

    selected_edge_ids: list[str] = []
    for parent, child in rooted.edges():
        edge_data = corridor.graph[parent][child]
        selected_edge_ids.append(str(edge_data["edge_id"]))

    return RadialTreeResult(
        root_node_id=root_node_id,
        selected_edge_ids=selected_edge_ids,
        parent_by_node=parent_by_node,
        depth_by_node=depth_by_node,
        terminal_nodes=set(terminals),
    )


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

