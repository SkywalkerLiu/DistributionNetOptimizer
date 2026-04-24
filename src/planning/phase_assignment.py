from __future__ import annotations

from typing import Any

import numpy as np

from src.planning.models import PHASE_INDEX, PHASES, AttachmentOption, RadialTreeResult
from src.planning.radial_tree_milp import path_to_root


def optimize_phase_assignment(
    *,
    tree: RadialTreeResult,
    attachment_choices: dict[int, AttachmentOption],
    users: Any,
    planning_cfg: dict[str, Any],
) -> tuple[dict[int, str], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """Assign single-phase users to A/B/C while reducing root and segment imbalance."""

    user_by_id = {int(row.user_id): row for row in users.itertuples()}
    path_by_user = {
        user_id: path_to_root(tree, option.attach_node_id)
        for user_id, option in attachment_choices.items()
    }
    assignments: dict[int, str] = {}
    edge_phase_loads: dict[tuple[str, str], np.ndarray] = {}
    edge_phase_kw: dict[tuple[str, str], np.ndarray] = {}
    transformer_load = np.zeros(3, dtype=float)
    transformer_kw = np.zeros(3, dtype=float)

    def ensure_edge(edge: tuple[str, str]) -> None:
        edge_phase_loads.setdefault(edge, np.zeros(3, dtype=float))
        edge_phase_kw.setdefault(edge, np.zeros(3, dtype=float))

    for user_id, row in user_by_id.items():
        if str(row.phase_type) != "three_phase":
            continue
        assignments[user_id] = "ABC"
        share_kva = float(row.apparent_kva) / 3.0
        share_kw = float(row.load_kw) / 3.0
        transformer_load += share_kva
        transformer_kw += share_kw
        for edge in path_by_user[user_id]:
            ensure_edge(edge)
            edge_phase_loads[edge] += share_kva
            edge_phase_kw[edge] += share_kw

    single_phase_users = [
        user_by_id[user_id]
        for user_id in sorted(attachment_choices)
        if str(user_by_id[user_id].phase_type) != "three_phase"
    ]
    single_phase_users.sort(key=lambda row: float(row.apparent_kva), reverse=True)

    tx_weight = float(planning_cfg.get("tx_unbalance_weight", 2.0))
    seg_weight = float(planning_cfg.get("segment_unbalance_weight", 1.0))
    for row in single_phase_users:
        user_id = int(row.user_id)
        best_phase = "A"
        best_score = float("inf")
        for phase in PHASES:
            phase_index = PHASE_INDEX[phase]
            trial_tx = transformer_load.copy()
            trial_tx_kw = transformer_kw.copy()
            trial_tx[phase_index] += float(row.apparent_kva)
            trial_tx_kw[phase_index] += float(row.load_kw)
            score = tx_weight * _unbalance_ratio(trial_tx)
            for edge in path_by_user[user_id]:
                ensure_edge(edge)
                trial_edge = edge_phase_loads[edge].copy()
                trial_edge[phase_index] += float(row.apparent_kva)
                score += seg_weight * _unbalance_ratio(trial_edge)
            if score < best_score:
                best_score = score
                best_phase = phase
        assignments[user_id] = best_phase
        phase_index = PHASE_INDEX[best_phase]
        transformer_load[phase_index] += float(row.apparent_kva)
        transformer_kw[phase_index] += float(row.load_kw)
        for edge in path_by_user[user_id]:
            ensure_edge(edge)
            edge_phase_loads[edge][phase_index] += float(row.apparent_kva)
            edge_phase_kw[edge][phase_index] += float(row.load_kw)

    for _ in range(2):
        improved = False
        for row in single_phase_users:
            user_id = int(row.user_id)
            current_phase = assignments[user_id]
            current_score = _assignment_score(
                assignments=assignments,
                attachment_choices=attachment_choices,
                users=users,
                planning_cfg=planning_cfg,
                tree=tree,
            )
            for phase in PHASES:
                if phase == current_phase:
                    continue
                assignments[user_id] = phase
                trial_score = _assignment_score(
                    assignments=assignments,
                    attachment_choices=attachment_choices,
                    users=users,
                    planning_cfg=planning_cfg,
                    tree=tree,
                )
                if trial_score + 1e-9 < current_score:
                    improved = True
                    current_score = trial_score
                    current_phase = phase
                else:
                    assignments[user_id] = current_phase
        if not improved:
            break

    return assignments, *_aggregate_phase_loads(assignments=assignments, attachment_choices=attachment_choices, users=users, tree=tree)


def _aggregate_phase_loads(
    *,
    assignments: dict[int, str],
    attachment_choices: dict[int, AttachmentOption],
    users: Any,
    tree: RadialTreeResult,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """Aggregate per-phase apparent and active power on every tree edge."""

    user_by_id = {int(row.user_id): row for row in users.itertuples()}
    edge_phase_loads: dict[tuple[str, str], np.ndarray] = {}
    edge_phase_kw: dict[tuple[str, str], np.ndarray] = {}

    for user_id, option in attachment_choices.items():
        row = user_by_id[user_id]
        phase = assignments[user_id]
        path_edges = path_to_root(tree, option.attach_node_id)
        if phase == "ABC":
            load = np.full(3, float(row.apparent_kva) / 3.0, dtype=float)
            kw = np.full(3, float(row.load_kw) / 3.0, dtype=float)
        else:
            load = np.zeros(3, dtype=float)
            kw = np.zeros(3, dtype=float)
            phase_index = PHASE_INDEX[phase]
            load[phase_index] = float(row.apparent_kva)
            kw[phase_index] = float(row.load_kw)
        for edge in path_edges:
            edge_phase_loads.setdefault(edge, np.zeros(3, dtype=float))
            edge_phase_kw.setdefault(edge, np.zeros(3, dtype=float))
            edge_phase_loads[edge] += load
            edge_phase_kw[edge] += kw
    return edge_phase_loads, edge_phase_kw


def _assignment_score(
    *,
    assignments: dict[int, str],
    attachment_choices: dict[int, AttachmentOption],
    users: Any,
    planning_cfg: dict[str, Any],
    tree: RadialTreeResult,
) -> float:
    """Score one full phase assignment."""

    edge_phase_loads, _ = _aggregate_phase_loads(
        assignments=assignments,
        attachment_choices=attachment_choices,
        users=users,
        tree=tree,
    )
    tx_load = np.zeros(3, dtype=float)
    for load in edge_phase_loads.values():
        tx_load = np.maximum(tx_load, load) if False else tx_load
    for row in users.itertuples():
        phase = assignments.get(int(row.user_id), "")
        if phase == "ABC":
            tx_load += float(row.apparent_kva) / 3.0
        elif phase in PHASE_INDEX:
            tx_load[PHASE_INDEX[phase]] += float(row.apparent_kva)

    score = float(planning_cfg.get("tx_unbalance_weight", 2.0)) * _unbalance_ratio(tx_load)
    score += float(planning_cfg.get("segment_unbalance_weight", 1.0)) * sum(
        _unbalance_ratio(load) for load in edge_phase_loads.values() if float(load.sum()) > 0.0
    )
    return score


def _unbalance_ratio(load: np.ndarray) -> float:
    """Return the phase imbalance ratio against the phase average."""

    total = float(load.sum())
    if total <= 0.0:
        return 0.0
    average = total / 3.0
    return float(np.max(np.abs(load - average)) / max(average, 1e-9))

