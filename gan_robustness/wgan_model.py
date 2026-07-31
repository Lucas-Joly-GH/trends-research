"""
wgan_model.py — Conditional WGAN-GP Generator and Critic (dense MLP, Approach A).

Architecture follows Amundi (2020) Section 3.3.4:
  - Both networks use fully-connected (Dense) layers with LeakyReLU.
  - Generator output uses Tanh -> maps to [-1, 1] (matching MinMax-scaled inputs).
  - Critic has no final activation (unbounded output required for Wasserstein distance).
  - Optional spectral normalisation on all critic linear layers (stabilises high-D training).

Input conventions:
  - z      : (batch, noise_dim)      -- noise vector for generator
  - window : (batch, window_size, D) -- conditioning context (last W scaled returns)
  - r      : (batch, D)              -- real or fake returns for critic
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _linear(in_dim: int, out_dim: int, use_spectral_norm: bool = False) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    return spectral_norm(layer) if use_spectral_norm else layer


class ConditionalGenerator(nn.Module):
    """
    Dense MLP Generator (Approach A).

    Input:  concat(z, window.flatten())
    Output: (batch, D) in [-1, 1] via Tanh.

    hidden_dims: list of hidden layer sizes; final output layer (size D) is
    appended automatically.
    """

    def __init__(
        self,
        noise_dim: int,
        window_size: int,
        n_instruments: int,
        hidden_dims: list[int],
        leaky_slope: float = 0.5,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.noise_dim      = noise_dim
        self.window_size    = window_size
        self.n_instruments  = n_instruments

        input_dim = noise_dim + window_size * n_instruments
        dims = [input_dim] + hidden_dims

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(_linear(dims[i], dims[i + 1], use_spectral_norm))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        layers.append(_linear(dims[-1], n_instruments, use_spectral_norm))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,       # (batch, noise_dim)
        window: torch.Tensor,  # (batch, window_size, D)
    ) -> torch.Tensor:         # (batch, D)
        w_flat = window.reshape(window.shape[0], -1)
        return self.net(torch.cat([z, w_flat], dim=1))


class Critic(nn.Module):
    """
    Dense MLP Critic (Approach A).

    Input:  concat(r, window.flatten())
    Output: (batch, 1) -- unbounded Wasserstein score.

    Spectral normalisation on all linear layers is recommended for D=62
    to prevent gradient explosion during high-dimensional training.
    """

    def __init__(
        self,
        n_instruments: int,
        window_size: int,
        hidden_dims: list[int],
        leaky_slope: float = 0.5,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_instruments  = n_instruments
        self.window_size    = window_size

        input_dim = n_instruments + window_size * n_instruments
        dims = [input_dim] + hidden_dims

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(_linear(dims[i], dims[i + 1], use_spectral_norm))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        layers.append(_linear(dims[-1], 1, use_spectral_norm))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        r: torch.Tensor,       # (batch, D)
        window: torch.Tensor,  # (batch, window_size, D)
    ) -> torch.Tensor:         # (batch, 1)
        w_flat = window.reshape(window.shape[0], -1)
        return self.net(torch.cat([r, w_flat], dim=1))


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
