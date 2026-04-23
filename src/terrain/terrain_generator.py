from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def generate_terrain(config: dict[str, Any]) -> np.ndarray:
    """Generate a synthetic terrain surface from the provided config."""

    scene_cfg = config["scene"]
    terrain_cfg = config["terrain"]
    width = int(round(scene_cfg["width_m"] / scene_cfg["resolution_m"]))
    height = int(round(scene_cfg["height_m"] / scene_cfg["resolution_m"]))

    rng = np.random.default_rng(int(scene_cfg["seed"]))
    terrain = _base_surface(height, width, terrain_cfg.get("base_type", "saddle"))

    if terrain_cfg.get("add_perlin_noise", True):
        terrain += _multiscale_noise(
            shape=(height, width),
            rng=rng,
            base_scale=float(terrain_cfg.get("noise_scale", 0.01)),
            amplitude=float(terrain_cfg.get("noise_amplitude", 25.0)),
            octaves=int(terrain_cfg.get("noise_octaves", 4)),
        )

    if terrain_cfg.get("add_gaussian_hills", True):
        terrain = _add_landforms(
            terrain=terrain,
            rng=rng,
            count=int(terrain_cfg.get("hill_count", 12)),
            valley_ratio=float(terrain_cfg.get("valley_ratio", 0.25)),
            sigma_min=float(terrain_cfg.get("hill_sigma_min", 0.08)),
            sigma_max=float(terrain_cfg.get("hill_sigma_max", 0.25)),
        )

    smooth_sigma = float(terrain_cfg.get("smooth_sigma", 0.0))
    if smooth_sigma > 0:
        terrain = gaussian_filter(terrain, sigma=smooth_sigma)

    return _scale_and_clip(
        terrain,
        clip_min=float(terrain_cfg["clip_min"]),
        clip_max=float(terrain_cfg["clip_max"]),
    )


def _base_surface(height: int, width: int, base_type: str) -> np.ndarray:
    """Build a smooth low-frequency terrain trend."""

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]

    if base_type == "ridge":
        surface = 0.8 * (1.0 - x**2) + 0.25 * y + 0.15 * np.sin(3.0 * np.pi * x)
    elif base_type == "basin":
        surface = -0.7 * (x**2 + y**2) + 0.2 * y - 0.15 * x
    else:
        surface = 0.5 * y - 0.35 * x + 0.45 * (y**2 - 0.6 * x**2) + 0.25 * x * y

    return surface.astype(np.float32)


def _multiscale_noise(
    *,
    shape: tuple[int, int],
    rng: np.random.Generator,
    base_scale: float,
    amplitude: float,
    octaves: int,
) -> np.ndarray:
    """Generate smooth fractal-like noise by upsampling coarse grids."""

    height, width = shape
    noise = np.zeros(shape, dtype=np.float32)
    total_weight = 0.0
    scale = max(base_scale, 1.0 / max(height, width))

    for octave in range(octaves):
        weight = 1.0 / (2**octave)
        coarse_h = max(4, int(round(height * scale * (2**octave))))
        coarse_w = max(4, int(round(width * scale * (2**octave))))
        coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
        upsampled = zoom(
            coarse,
            (height / coarse_h, width / coarse_w),
            order=3,
            mode="reflect",
        )
        noise += upsampled[:height, :width] * weight
        total_weight += weight

    noise /= max(total_weight, 1e-6)
    noise -= float(noise.mean())
    std = float(noise.std()) or 1.0
    noise /= std
    return noise * amplitude


def _add_landforms(
    *,
    terrain: np.ndarray,
    rng: np.random.Generator,
    count: int,
    valley_ratio: float,
    sigma_min: float = 0.08,
    sigma_max: float = 0.25,
) -> np.ndarray:
    """Add Gaussian hills and shallow valleys to the surface."""

    output = terrain.astype(np.float32, copy=True)
    height, width = output.shape
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)

    for _ in range(count):
        center_x = rng.uniform(0.05, 0.95)
        center_y = rng.uniform(0.05, 0.95)
        sigma_x = rng.uniform(sigma_min, sigma_max)
        sigma_y = rng.uniform(sigma_min, sigma_max)
        amplitude = rng.uniform(20.0, 90.0)
        if rng.random() < valley_ratio:
            amplitude *= -0.8

        dx = ((xs - center_x) ** 2) / (2.0 * sigma_x**2)
        dy = ((ys - center_y) ** 2) / (2.0 * sigma_y**2)
        output += amplitude * np.exp(-(dy[:, None] + dx[None, :])).astype(np.float32)

    return output


def _scale_and_clip(
    terrain: np.ndarray,
    *,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Normalize terrain into the configured elevation range."""

    normalized = terrain.astype(np.float32, copy=True)
    normalized -= float(normalized.min())
    peak = float(normalized.max()) or 1.0
    normalized /= peak
    normalized = normalized * (clip_max - clip_min) + clip_min
    normalized = np.clip(normalized, clip_min, clip_max)
    return normalized.astype(np.float32)

