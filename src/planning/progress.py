from __future__ import annotations

import sys
import time
from typing import TextIO


class OptimizationProgress:
    """Lightweight terminal progress bar for the V2 optimizer."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_candidates: int,
        max_search_iter: int,
        bar_width: int = 32,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.total_candidates = max(1, int(total_candidates))
        self.max_search_iter = max(0, int(max_search_iter))
        self.bar_width = max(10, int(bar_width))
        self.stream = stream or sys.stdout
        self._start_time = time.perf_counter()
        self._last_width = 0
        self._best_objective: float | None = None
        self._closed = False

    def stage(self, *, progress: float, label: str, detail: str = "") -> None:
        """Render a stage-level progress update."""

        self._render(progress=progress, label=label, detail=detail)

    def candidate_started(self, *, index: int, total: int, detail: str = "") -> None:
        """Render the start of one candidate evaluation."""

        candidate_progress = (max(index, 1) - 1) / max(total, 1)
        overall = 0.25 + 0.65 * candidate_progress
        self._render(
            progress=overall,
            label="候选评估",
            detail=f"候选 {index}/{total} 准备评估{self._join_detail(detail)}",
        )

    def candidate_initial(
        self,
        *,
        index: int,
        total: int,
        objective: float,
        feasible: bool,
    ) -> None:
        """Render the initial solution evaluation for one candidate."""

        self._best_objective = objective if self._best_objective is None else min(self._best_objective, objective)
        progress_within_candidate = 1.0 / max(self.max_search_iter + 1, 1)
        candidate_progress = ((max(index, 1) - 1) + progress_within_candidate) / max(total, 1)
        overall = 0.25 + 0.65 * candidate_progress
        self._render(
            progress=overall,
            label="候选评估",
            detail=(
                f"候选 {index}/{total} 初始解 | obj={objective:.1f} | "
                f"{'feasible' if feasible else 'infeasible'}{self._global_best_suffix()}"
            ),
        )

    def candidate_iteration(
        self,
        *,
        index: int,
        total: int,
        iteration: int,
        max_iter: int,
        objective: float,
        improved: bool,
    ) -> None:
        """Render one local-search iteration."""

        self._best_objective = objective if self._best_objective is None else min(self._best_objective, objective)
        bounded_max_iter = max(1, max_iter)
        bounded_iteration = min(max(iteration, 0), bounded_max_iter)
        progress_within_candidate = (1.0 + bounded_iteration) / (bounded_max_iter + 1.0)
        candidate_progress = ((max(index, 1) - 1) + progress_within_candidate) / max(total, 1)
        overall = 0.25 + 0.65 * candidate_progress
        self._render(
            progress=overall,
            label="候选评估",
            detail=(
                f"候选 {index}/{total} 局部搜索 {bounded_iteration}/{bounded_max_iter} | "
                f"{'improved' if improved else 'no-improve'} | obj={objective:.1f}{self._global_best_suffix()}"
            ),
        )

    def candidate_finished(
        self,
        *,
        index: int,
        total: int,
        objective: float,
        feasible: bool,
    ) -> None:
        """Render the completion of one candidate evaluation."""

        self._best_objective = objective if self._best_objective is None else min(self._best_objective, objective)
        candidate_progress = max(index, 1) / max(total, 1)
        overall = 0.25 + 0.65 * candidate_progress
        self._render(
            progress=overall,
            label="候选评估",
            detail=(
                f"候选 {index}/{total} 完成 | obj={objective:.1f} | "
                f"{'feasible' if feasible else 'infeasible'}{self._global_best_suffix()}"
            ),
        )

    def finish(self, *, detail: str = "") -> None:
        """Render the final completion state and terminate the line."""

        self._render(progress=1.0, label="完成", detail=detail)
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _global_best_suffix(self) -> str:
        if self._best_objective is None:
            return ""
        return f" | best={self._best_objective:.1f}"

    def _join_detail(self, detail: str) -> str:
        if not detail:
            return ""
        return f" | {detail}"

    def _render(self, *, progress: float, label: str, detail: str) -> None:
        if not self.enabled or self._closed:
            return
        bounded = min(max(float(progress), 0.0), 1.0)
        filled = int(round(bounded * self.bar_width))
        filled = min(filled, self.bar_width)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        elapsed_s = time.perf_counter() - self._start_time
        text = f"\r优化进度 [{bar}] {bounded * 100:6.2f}% | {label}"
        if detail:
            text += f" | {detail}"
        text += f" | {elapsed_s:6.1f}s"
        padding = max(0, self._last_width - len(text))
        self.stream.write(text + (" " * padding))
        self.stream.flush()
        self._last_width = len(text)
