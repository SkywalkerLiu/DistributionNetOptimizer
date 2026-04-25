from __future__ import annotations

from typing import Callable

from src.planning.models import AttachmentOption, EvaluatedSolution


def improve_attachment_choices(
    *,
    options_by_user: dict[int, list[AttachmentOption]],
    initial_solution: EvaluatedSolution,
    evaluate: Callable[[dict[int, AttachmentOption]], EvaluatedSolution],
    max_iter: int,
    destroy_ratio: float,
    patience: int = 1,
    top_options: int = 5,
    max_full_evals: int = 0,
    performance_counters: dict[str, int] | None = None,
    progress_callback: Callable[[int, int, bool, EvaluatedSolution], None] | None = None,
) -> EvaluatedSolution:
    """Improve the solution through local user reattachment moves."""

    best = initial_solution
    choices = dict(initial_solution.attachment_choices)
    hot_count = max(1, int(round(len(choices) * max(destroy_ratio, 0.05))))
    bounded_patience = max(1, int(patience))
    bounded_top_options = max(1, int(top_options))
    bounded_max_full_evals = max(0, int(max_full_evals))
    no_improve_count = 0
    full_eval_count = 0

    for iteration in range(1, max_iter + 1):
        ranked_users = sorted(
            choices,
            key=lambda user_id: (
                best.power_flow.user_voltage_drop_pct.get(user_id, 0.0),
                choices[user_id].cost,
            ),
            reverse=True,
        )
        improved = False
        users_to_try = _rotating_user_window(
            ranked_users=ranked_users,
            hot_count=hot_count,
            iteration=iteration,
        )
        for user_id in users_to_try:
            current = choices[user_id]
            alternative_options = [
                option
                for option in options_by_user.get(user_id, [])
                if option.attach_node_id != current.attach_node_id
            ]
            _increment_counter(performance_counters, "local_search_trial_count", len(alternative_options))
            candidate_options = sorted(
                alternative_options,
                key=lambda option: _proxy_move_score(
                    current=current,
                    option=option,
                    voltage_drop_pct=best.power_flow.user_voltage_drop_pct.get(user_id, 0.0),
                ),
            )[:bounded_top_options]
            for option in candidate_options:
                if bounded_max_full_evals and full_eval_count >= bounded_max_full_evals:
                    break
                trial_choices = dict(choices)
                trial_choices[user_id] = option
                _increment_counter(performance_counters, "local_search_full_eval_count", 1)
                full_eval_count += 1
                trial_solution = evaluate(trial_choices)
                if trial_solution.objective + 1e-9 < best.objective:
                    best = trial_solution
                    choices = dict(trial_solution.attachment_choices)
                    improved = True
                    break
            if improved:
                break
            if bounded_max_full_evals and full_eval_count >= bounded_max_full_evals:
                break
        if progress_callback is not None:
            progress_callback(iteration, max_iter, improved, best)
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
        if bounded_max_full_evals and full_eval_count >= bounded_max_full_evals:
            break
        if no_improve_count >= bounded_patience:
            break
    return best


def _rotating_user_window(*, ranked_users: list[int], hot_count: int, iteration: int) -> list[int]:
    """Return a deterministic rolling window of high-risk users to inspect."""

    if not ranked_users:
        return []
    bounded_hot_count = min(max(1, int(hot_count)), len(ranked_users))
    start = ((max(1, int(iteration)) - 1) * bounded_hot_count) % len(ranked_users)
    rotated = ranked_users[start:] + ranked_users[:start]
    return rotated[:bounded_hot_count]


def _proxy_move_score(
    *,
    current: AttachmentOption,
    option: AttachmentOption,
    voltage_drop_pct: float,
) -> float:
    """Cheap proxy used before calling the full radial-tree and power-flow evaluator."""

    service_cost_delta = float(option.cost) - float(current.cost)
    length_delta = float(option.horizontal_length_m) - float(current.horizontal_length_m)
    voltage_risk = max(float(voltage_drop_pct), 0.0)
    return float(service_cost_delta + length_delta * (1.0 + 0.05 * voltage_risk))


def _increment_counter(counters: dict[str, int] | None, key: str, amount: int) -> None:
    if counters is None or amount <= 0:
        return
    counters[key] = int(counters.get(key, 0)) + int(amount)
