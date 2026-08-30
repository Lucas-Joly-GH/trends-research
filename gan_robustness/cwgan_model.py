"""
cwgan_model.py -- Convolutional WGAN-GP Generator and Critic (Approach B).

Why convolutions for multi-asset time series?
  - Conv1d layers over the W=20-day window capture local temporal patterns
    (autocorrelation, volatility clustering) that dense layers miss.
  - Receptive field of 4 layers x kernel_size=3 covers ~9 time steps
    (enough to capture weekly return patterns).
  - Spectral normalisation on all layers prevents mode collapse for D=62.
  - InstanceNorm in the generator preserves inter-sample diversity (BatchNorm
    was found to cause mode collapse by averaging batch statistics).
  - Per Amundi (2020) Section 3.2.2, convolutional WGAN outperformed dense
    WGAN on multi-asset time series benchmarks.

Architecture:
  Generator:
    noise (B, noise_dim)   -> Linear -> (B, D*W) -> reshape -> (B, D, W)
    window (B, W, D)       -> permute -> (B, D, W)
    concat along channels  -> (B, 2D, W)
    Conv1d blocks          -> (B, channels[-1], W)
    Global average pool    -> (B, channels[-1])
    Linear -> Tanh         -> (B, D)

  Critic (all Conv/Linear layers wrapped with spectral_norm):
    r (B, D)               -> unsqueeze -> (B, D, 1)
    window (B, W, D)       -> permute   -> (B, D, W)
    concat along time      -> (B, D, W+1)
    Conv1d blocks          -> (B, channels[-1], W+1)
    Global average pool    -> (B, channels[-1])
    Linear                 -> (B, 1)   [unbounded]

Input conventions match wgan_model.py exactly, so trainer.py is architecture-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv1d(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    padding: int = 1,
    use_spectral_norm: bool = False,
) -> nn.Conv1d:
    layer = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
    return spectral_norm(layer) if use_spectral_norm else layer


def _linear(in_dim: int, out_dim: int, use_spectral_norm: bool = False) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    return spectral_norm(layer) if use_spectral_norm else layer


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convolutional Generator
# ---------------------------------------------------------------------------

class ConvGenerator(nn.Module):
    """
    1D convolutional conditional generator.

    Args:
        noise_dim       : dimension of input noise vector z
        window_size     : W, length of conditioning window (days)
        n_instruments   : D, number of output instruments
        channels        : list of channel sizes for Conv1d blocks
                          (e.g. [256, 256, 128, 64]); first block input is 2*D
        kernel_size     : Conv1d kernel size (3 recommended)
        leaky_slope     : LeakyReLU negative slope
    """

    def __init__(
        self,
        noise_dim: int,
        window_size: int,
        n_instruments: int,
        channels: list[int],
        kernel_size: int = 3,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.noise_dim     = noise_dim
        self.window_size   = window_size
        self.n_instruments = n_instruments
        D, W = n_instruments, window_size

        # Project noise to (D, W) feature map to concatenate with memory
        self.noise_proj = nn.Linear(noise_dim, D * W)

        # Conv blocks: input channels = 2*D (noise + memory)
        in_ch = 2 * D
        conv_blocks: list[nn.Module] = []
        for out_ch in channels:
            conv_blocks += [
                _conv1d(in_ch, out_ch, kernel_size=kernel_size,
                        padding=kernel_size // 2),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.LeakyReLU(leaky_slope, inplace=True),
            ]
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_blocks)

        # Global average pool then linear output
        self.out_proj = nn.Sequential(
            nn.Linear(channels[-1], D),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,       # (batch, noise_dim)
        window: torch.Tensor,  # (batch, window_size, D)
    ) -> torch.Tensor:         # (batch, D)
        B = z.shape[0]
        D, W = self.n_instruments, self.window_size

        # Project noise -> (B, D, W) feature map
        noise_feat = self.noise_proj(z).reshape(B, D, W)    # (B, D, W)

        # Memory -> (B, D, W)
        mem = window.permute(0, 2, 1)                        # (B, D, W)

        # Concatenate along channel dim -> (B, 2D, W)
        x = torch.cat([noise_feat, mem], dim=1)              # (B, 2D, W)

        # Conv blocks
        x = self.conv_net(x)                                 # (B, channels[-1], W)

        # Global average pool over time dimension
        x = x.mean(dim=2)                                    # (B, channels[-1])

        return self.out_proj(x)                              # (B, D)


# ---------------------------------------------------------------------------
# Convolutional Critic
# ---------------------------------------------------------------------------

class ConvCritic(nn.Module):
    """
    1D convolutional conditional critic with spectral normalisation.

    Concatenates the current return r (as a single time step) with the
    conditioning window along the time dimension, then processes with
    Conv1d blocks treating instruments as channels.

    Args:
        n_instruments   : D
        window_size     : W
        channels        : channel progression for Conv1d blocks
                          (e.g. [128, 256, 128, 64]); first input is D channels
        kernel_size     : Conv1d kernel size
        leaky_slope     : LeakyReLU negative slope
        use_spectral_norm: apply spectral norm to all Conv1d and Linear layers
    """

    def __init__(
        self,
        n_instruments: int,
        window_size: int,
        channels: list[int],
        kernel_size: int = 3,
        leaky_slope: float = 0.2,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_instruments    = n_instruments
        self.window_size      = window_size
        self.use_spectral_norm = use_spectral_norm
        D = n_instruments

        # Sequence length after concatenation: W + 1 (r appended as a step)
        in_ch = D
        conv_blocks: list[nn.Module] = []
        for out_ch in channels:
            conv_blocks += [
                _conv1d(in_ch, out_ch, kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        use_spectral_norm=use_spectral_norm),
                nn.LeakyReLU(leaky_slope, inplace=True),
            ]
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_blocks)

        # Cross-instrument self-attention (operates over time steps after conv)
        # Lets the critic detect joint structure (correlations, factor loadings)
        # that Conv1d + global avg pool alone cannot see.
        C_last = channels[-1]
        attn = nn.MultiheadAttention(
            embed_dim=C_last, num_heads=4, batch_first=True,
        )
        # Apply spectral norm to attention's internal projection weights
        if use_spectral_norm:
            for attr in ("in_proj_weight",):
                if hasattr(attn, attr):
                    nn.utils.parametrizations.spectral_norm(attn, attr)
        self.cross_attn = attn
        self.attn_norm = nn.LayerNorm(C_last)

        self.out_proj = _linear(channels[-1], 1, use_spectral_norm)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        r: torch.Tensor,       # (batch, D)
        window: torch.Tensor,  # (batch, window_size, D)
    ) -> torch.Tensor:         # (batch, 1)
        # r as a time step: (B, D) -> (B, D, 1)
        r_t = r.unsqueeze(2)                     # (B, D, 1)

        # Memory: (B, W, D) -> (B, D, W)
        mem = window.permute(0, 2, 1)            # (B, D, W)

        # Concatenate along time: (B, D, W+1)
        x = torch.cat([mem, r_t], dim=2)         # (B, D, W+1)

        # Conv blocks
        x = self.conv_net(x)                     # (B, channels[-1], W+1)

        # Self-attention over time steps (lets critic see cross-instrument
        # joint structure that conv + global avg pool alone misses)
        x = x.permute(0, 2, 1)                  # (B, W+1, channels[-1])
        attn_out, _ = self.cross_attn(x, x, x)  # (B, W+1, channels[-1])
        x = self.attn_norm(x + attn_out)         # residual + LayerNorm

        # Pool over sequence
        x = x.mean(dim=1)                        # (B, channels[-1])

        return self.out_proj(x)                  # (B, 1)
