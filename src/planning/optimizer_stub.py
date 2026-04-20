from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OptimizationInputs:
    """Container for future optimization inputs."""

    dtm_path: Path
    slope_path: Path
    roughness_path: Path
    forbidden_mask_path: Path
    features_path: Path


@dataclass(slots=True)
class OptimizationResult:
    """Placeholder result for the future optimizer interface."""

    candidate_transformer_layer: str = "candidate_transformer"
    candidate_poles_layer: str = "candidate_poles"
    planned_lines_layer: str = "planned_lines"
    message: str = "Optimizer is not implemented yet."


def run_optimizer_stub(inputs: OptimizationInputs) -> OptimizationResult:
    """Return a typed placeholder response for the optimizer integration."""

    required_paths = [
        inputs.dtm_path,
        inputs.slope_path,
        inputs.roughness_path,
        inputs.forbidden_mask_path,
        inputs.features_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Optimizer inputs are missing: {joined}")
    return OptimizationResult()

