"""
generator_rollout.py — Autoregressive synthetic path generation.

Algorithm for one path of length T:
  1. Seed window = last WINDOW_SIZE days of real data, each transformed via
     preprocessor.transform_single() (NOT transform() — avoids re-ranking).
  2. For each step t in [0, T):
     a. Sample z ~ N(0, I) of shape (1, noise_dim).
     b. r_scaled = G(z, window)  — (D,) in [-1, 1].
     c. Update window: roll left by 1, append r_scaled at the end.
  3. Inverse-transform the full (T, D) scaled array back to price-difference space.

Performance note:
  - Window update uses np.roll + in-place assignment to avoid per-step allocation.
  - generator.eval() is called before generation and restored afterwards.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from .preprocessor import ReturnPreprocessor
    from .wgan_model import ConditionalGenerator
except ImportError:
    from preprocessor import ReturnPreprocessor  # type: ignore[no-redef]
    from wgan_model import ConditionalGenerator  # type: ignore[no-redef]


def generate_single_path(
    generator: ConditionalGenerator,
    preprocessor: ReturnPreprocessor,
    seed_window_scaled: np.ndarray,   # (W, D) already scaled via transform_single
    path_length: int,
    noise_dim: int,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one synthetic path of shape (path_length, D) in original price-diff space.

    Args:
        generator          : trained ConditionalGenerator (eval mode expected)
        preprocessor       : fitted ReturnPreprocessor (for inverse transform)
        seed_window_scaled : (W, D) float32 — last W real days in scaled space
        path_length        : number of steps to generate
        noise_dim          : dimensionality of z
        device             : torch.device
        rng                : NumPy Generator for reproducible noise

    Returns:
        np.ndarray of shape (path_length, D) in original price-difference scale
    """
    W, D = seed_window_scaled.shape
    window = seed_window_scaled.astype(np.float32).copy()  # (W, D)

    generated_scaled = np.empty((path_length, D), dtype=np.float32)

    for t in range(path_length):
        z_np = rng.standard_normal((1, noise_dim)).astype(np.float32)
        z_t  = torch.tensor(z_np, device=device)
        w_t  = torch.tensor(window[np.newaxis, ...], device=device)  # (1, W, D)

        with torch.no_grad():
            r_scaled = generator(z_t, w_t).cpu().numpy()[0]  # (D,)

        generated_scaled[t] = r_scaled

        # Roll window: discard oldest, append new step (in-place, no allocation)
        window = np.roll(window, shift=-1, axis=0)
        window[-1] = r_scaled

    return preprocessor.inverse_transform(generated_scaled)  # (path_length, D)


def generate_all_paths(
    generator: ConditionalGenerator,
    preprocessor: ReturnPreprocessor,
    real_returns: np.ndarray,    # (T, D) original-scale price differences (training data)
    config: object,              # config module
    device: torch.device,
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Generate N_PATHS synthetic paths, each of length PATH_LENGTH.

    Seed window is built from the last WINDOW_SIZE rows of real_returns,
    transformed via transform_single() for correct inference-time scaling.

    Each path uses a deterministic RNG seed (GEN_SEED_BASE + path_index)
    for reproducibility.

    Returns:
        list of N_PATHS arrays, each of shape (PATH_LENGTH, D), in original
        price-difference scale.
    """
    W  = config.WINDOW_SIZE
    D  = real_returns.shape[1]

    # Build seed window (last W real observations, scaled one-by-one)
    seed_raw = real_returns[-W:, :]   # (W, D)
    seed_window_scaled = np.empty((W, D), dtype=np.float32)
    for i in range(W):
        seed_window_scaled[i] = preprocessor.transform_single(seed_raw[i]).astype(np.float32)

    generator.eval()

    paths: list[np.ndarray] = []
    for path_idx in range(config.N_PATHS):
        rng = np.random.default_rng(config.GEN_SEED_BASE + path_idx)
        path = generate_single_path(
            generator=generator,
            preprocessor=preprocessor,
            seed_window_scaled=seed_window_scaled,
            path_length=config.PATH_LENGTH,
            noise_dim=config.NOISE_DIM,
            device=device,
            rng=rng,
        )
        paths.append(path)

        if verbose and (path_idx + 1) % 100 == 0:
            print(f"  [rollout] Generated {path_idx + 1}/{config.N_PATHS} paths")

    return paths
