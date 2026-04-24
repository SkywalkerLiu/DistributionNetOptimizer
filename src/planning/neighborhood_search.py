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
    progress_callback: Callable[[int, int, bool, EvaluatedSolution], None] | None = None,
) -> EvaluatedSolution:
    """Improve the solution through local user reattachment moves."""

    best = initial_solution
    choices = dict(initial_solution.attachment_choices)
    hot_count = max(1, int(round(len(choices) * max(destroy_ratio, 0.05))))

    for iteration in range(1, max_iter + 1):
        ranked_users = sorted(
            choices,
            key=lambda user_id: (
                initial_solution.power_flow.user_voltage_drop_pct.get(user_id, 0.0),
                choices[user_id].cost,
            ),
            reverse=True,
        )
        improved = False
        for user_id in ranked_users[:hot_count]:
            current = choices[user_id]
            for option in options_by_user.get(user_id, []):
                if option.attach_node_id == current.attach_node_id:
                    continue
                trial_choices = dict(choices)
                trial_choices[user_id] = option
                trial_solution = evaluate(trial_choices)
                if trial_solution.objective + 1e-9 < best.objective:
                    best = trial_solution
                    choices = dict(trial_solution.attachment_choices)
                    improved = True
                    break
            if improved:
                break
        if progress_callback is not None:
            progress_callback(iteration, max_iter, improved, best)
        if not improved:
            break
    return best
