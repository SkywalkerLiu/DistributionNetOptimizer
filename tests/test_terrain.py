from __future__ import annotations

import copy

import numpy as np

from src.terrain.terrain_generator import generate_terrain
from src.terrain.terrain_validator import terrain_statistics, validate_terrain_array


def _config(seed: int = 11) -> dict:
    return {
        "scene": {
            "width_m": 128,
            "height_m": 96,
            "resolution_m": 4,
            "seed": seed,
        },
        "terrain": {
            "base_type": "saddle",
            "add_perlin_noise": True,
            "noise_scale": 0.1,
            "noise_amplitude": 8.0,
            "noise_octaves": 3,
            "add_gaussian_hills": True,
            "hill_count": 4,
            "valley_ratio": 0.25,
            "smooth_sigma": 1.0,
            "clip_min": 0,
            "clip_max": 120,
        },
    }


def test_generate_terrain_shape_and_range() -> None:
    config = _config()
    dtm = generate_terrain(config)

    assert dtm.shape == (24, 32)
    validate_terrain_array(dtm, clip_min=0, clip_max=120)
    stats = terrain_statistics(dtm)
    assert 0.0 <= stats["min_elevation_m"] <= stats["max_elevation_m"] <= 120.0
    assert stats["std_elevation_m"] > 0.0


def test_generate_terrain_reproducible() -> None:
    config = _config(seed=17)
    first = generate_terrain(config)
    second = generate_terrain(copy.deepcopy(config))
    different = generate_terrain(_config(seed=18))

    assert np.allclose(first, second)
    assert not np.allclose(first, different)

