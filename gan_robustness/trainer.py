"""
trainer.py -- WGAN-GP training loop (architecture-agnostic).

Supports both dense MLP (Approach A) and convolutional (Approach B) models.
The generator and critic are passed in already instantiated; trainer only
calls forward() and backward() through them.

Key features:
  - Cosine annealing LR schedule (config.LR_SCHEDULE = "cosine").
  - Warmup phase: more critic steps per generator step for the first
    CRITIC_WARMUP_EPOCHS epochs (helps critic lead the generator early).
  - Fresh mini-batch sampled for each critic step (no gradient bias).
  - Gaussian jitter on real samples (config.TRAIN_JITTER_STD > 0).
  - Moment matching regularization on generator (anti-mode-collapse).
  - Checkpoints saved every 100 epochs to out_dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def build_sliding_window_dataset(
    X_scaled: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window dataset from scaled returns.

    Returns:
        windows : (N, W, D) where N = T - W
        targets : (N, D)    next-step return for each window
    """
    T, D = X_scaled.shape
    N = T - window_size
    windows = np.empty((N, window_size, D), dtype=np.float32)
    targets = np.empty((N, D), dtype=np.float32)
    for i in range(N):
        windows[i] = X_scaled[i : i + window_size]
        targets[i] = X_scaled[i + window_size]
    return windows, targets


# ---------------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------------

def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,   # (batch, D)
    fake_samples: torch.Tensor,   # (batch, D)
    window: torch.Tensor,         # (batch, W, D)
    lambda_gp: float,
    device: torch.device,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al., 2017).

    GP = lambda * E[(||grad_{x_hat} D(x_hat, window)||_2 - 1)^2]

    Works for both dense and convolutional critics since both accept
    (r: (B,D), window: (B,W,D)) and return (B,1).
    """
    batch_size = real_samples.shape[0]
    epsilon = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
    x_hat = (epsilon * real_samples + (1.0 - epsilon) * fake_samples).requires_grad_(True)

    d_x_hat = critic(x_hat, window)

    gradients = torch.autograd.grad(
        outputs=d_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_x_hat),
        create_graph=True,
        retain_graph=True,
    )[0]  # (batch, D)

    gradient_norm = gradients.reshape(batch_size, -1).norm(2, dim=1)
    return lambda_gp * ((gradient_norm - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_wgan(
    generator: nn.Module,
    critic: nn.Module,
    X_scaled: np.ndarray,
    config: object,
    device: torch.device,
    out_dir: Path,
    verbose: bool = True,
    resume_from: str | Path | None = None,
) -> dict:
    """
    WGAN-GP training loop -- architecture-agnostic.

    Args:
        generator : any Module with forward(z, window) -> (B, D)
        critic    : any Module with forward(r, window) -> (B, 1)
        X_scaled  : (T, D) float64 preprocessed returns in [-1, 1]
        config    : config module
        device    : torch.device
        out_dir   : directory for intermediate checkpoints
        verbose   : print progress every 100 epochs
        resume_from : path to a wgan_checkpoint_epXXXXX.pt file to resume from

    Returns:
        history dict:
          wasserstein_estimate : list[float] -- per epoch
          loss_g               : list[float] -- per epoch
          lr_g                 : list[float] -- per epoch (learning rate)
    """
    X_f32 = X_scaled.astype(np.float32)
    windows_np, targets_np = build_sliding_window_dataset(X_f32, config.WINDOW_SIZE)
    N = windows_np.shape[0]
    n_params_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    n_params_c = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    n_batches  = max(1, N // config.BATCH_SIZE)

    if verbose:
        D = X_scaled.shape[1]
        print(
            f"[trainer] Dataset: {N} sequences | D={D} | "
            f"batches/epoch={n_batches} | batch_size={config.BATCH_SIZE}"
        )
        print(f"[trainer] Generator: {n_params_g:,} params | Critic: {n_params_c:,} params")

    opt_G = optim.Adam(generator.parameters(), lr=config.LR,
                       betas=(config.BETA1, config.BETA2))
    opt_C = optim.Adam(critic.parameters(),    lr=config.LR,
                       betas=(config.BETA1, config.BETA2))

    # LR schedulers
    lr_schedule = getattr(config, "LR_SCHEDULE", None)
    lr_min      = getattr(config, "LR_MIN", 1e-6)
    if lr_schedule == "cosine":
        sched_G = CosineAnnealingLR(opt_G, T_max=config.N_EPOCHS, eta_min=lr_min)
        sched_C = CosineAnnealingLR(opt_C, T_max=config.N_EPOCHS, eta_min=lr_min)
        if verbose:
            print(f"[trainer] LR schedule: cosine annealing "
                  f"{config.LR:.0e} -> {lr_min:.0e} over {config.N_EPOCHS} epochs")
    else:
        sched_G = sched_C = None

    # Warmup critic iterations
    critic_warmup_iters  = getattr(config, "CRITIC_ITERS_WARMUP",  config.N_CRITIC_STEPS)
    critic_warmup_epochs = getattr(config, "CRITIC_WARMUP_EPOCHS", 0)
    if verbose and critic_warmup_epochs > 0:
        print(f"[trainer] Critic warmup: {critic_warmup_iters} steps/epoch "
              f"for first {critic_warmup_epochs} epochs, "
              f"then {config.N_CRITIC_STEPS}")

    rng = np.random.default_rng(config.TRAIN_SEED)

    moment_lambda = getattr(config, "MOMENT_PENALTY_LAMBDA", 10.0)
    cov_lambda    = getattr(config, "COV_PENALTY_LAMBDA", 5.0)
    kurt_lambda   = getattr(config, "KURT_PENALTY_LAMBDA", 2.0)
    sort_lambda   = getattr(config, "SORT_MATCH_LAMBDA", 0.0)
    ckpt_every    = getattr(config, "CHECKPOINT_EVERY", 500)
    if verbose:
        if moment_lambda > 0:
            print(f"[trainer] Moment matching penalty: lambda={moment_lambda:.1f}")
        if cov_lambda > 0:
            print(f"[trainer] Covariance matching penalty: lambda={cov_lambda:.1f}")
        if kurt_lambda > 0:
            print(f"[trainer] Kurtosis matching penalty: lambda={kurt_lambda:.1f}")
        if sort_lambda > 0:
            print(f"[trainer] Sorted-samples (1D W2) matching penalty: lambda={sort_lambda:.1f}")
        print(f"[trainer] Checkpoint every {ckpt_every} epochs")

    history: dict[str, list[float]] = {
        "wasserstein_estimate": [],
        "loss_g": [],
        "moment_penalty": [],
        "cov_penalty": [],
        "kurt_penalty": [],
        "sort_penalty": [],
        "lr_g": [],
    }

    # Resume from checkpoint if provided
    start_epoch = 1
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            generator.load_state_dict(ckpt["generator_state"])
            critic.load_state_dict(ckpt["critic_state"])
            opt_G.load_state_dict(ckpt["opt_G_state"])
            opt_C.load_state_dict(ckpt["opt_C_state"])
            start_epoch = ckpt["epoch"] + 1
            if "history" in ckpt:
                history = ckpt["history"]
            # Advance LR schedulers to the correct position
            if sched_G is not None:
                for _ in range(start_epoch - 1):
                    sched_G.step()
                    sched_C.step()
            if verbose:
                print(f"[trainer] Resumed from epoch {ckpt['epoch']} -> starting at epoch {start_epoch}")
        else:
            if verbose:
                print(f"[trainer] Resume checkpoint not found: {resume_path}, starting from scratch")

    generator.train()
    critic.train()

    pbar_epoch = tqdm(range(start_epoch, config.N_EPOCHS + 1),
                      desc="Epochs", unit="ep",
                      initial=start_epoch - 1, total=config.N_EPOCHS,
                      position=0)

    for epoch in pbar_epoch:
        n_critic = (critic_warmup_iters
                    if epoch <= critic_warmup_epochs
                    else config.N_CRITIC_STEPS)

        idx_pool = rng.permutation(N)
        ptr = 0

        epoch_w_est: list[float] = []
        epoch_loss_g: list[float] = []
        epoch_moment_pen: list[float] = []
        epoch_cov_pen: list[float] = []
        epoch_kurt_pen: list[float] = []
        epoch_sort_pen: list[float] = []

        pbar_batch = tqdm(range(n_batches),
                          desc=f"  Ep {epoch:4d}",
                          unit="batch", leave=False,
                          position=1)

        for _batch_idx in pbar_batch:
            # ----------------------------------------------------------------
            # Critic updates
            # ----------------------------------------------------------------
            d_real_last = d_fake_last = torch.tensor(0.0, device=device)
            for _ in range(n_critic):
                if ptr + config.BATCH_SIZE > N:
                    idx_pool = rng.permutation(N)
                    ptr = 0
                bidx = idx_pool[ptr : ptr + config.BATCH_SIZE]
                ptr += config.BATCH_SIZE

                real_w_np = windows_np[bidx]
                real_r_np = targets_np[bidx]

                if config.TRAIN_JITTER_STD > 0:
                    real_r_np = real_r_np + rng.normal(
                        0, config.TRAIN_JITTER_STD, real_r_np.shape
                    ).astype(np.float32)
                    real_r_np = np.clip(real_r_np, -1.0, 1.0)

                real_w = torch.tensor(real_w_np, device=device)
                real_r = torch.tensor(real_r_np, device=device)
                B = real_r.shape[0]
                z = torch.randn(B, config.NOISE_DIM, device=device)

                with torch.no_grad():
                    fake_r = generator(z, real_w)

                opt_C.zero_grad()
                d_real = critic(real_r, real_w).mean()
                d_fake = critic(fake_r.detach(), real_w).mean()
                gp = compute_gradient_penalty(
                    critic, real_r, fake_r.detach(), real_w,
                    config.GRAD_PENALTY_LAMBDA, device,
                )
                loss_c = d_fake - d_real + gp
                loss_c.backward()
                opt_C.step()
                d_real_last, d_fake_last = d_real, d_fake

            epoch_w_est.append((d_real_last - d_fake_last).item())

            # ----------------------------------------------------------------
            # Generator update
            # ----------------------------------------------------------------
            if ptr + config.BATCH_SIZE > N:
                idx_pool = rng.permutation(N)
                ptr = 0
            bidx   = idx_pool[ptr : ptr + config.BATCH_SIZE]
            ptr   += config.BATCH_SIZE

            real_w = torch.tensor(windows_np[bidx], device=device)
            B = real_w.shape[0]
            z = torch.randn(B, config.NOISE_DIM, device=device)

            opt_G.zero_grad()
            fake_r = generator(z, real_w)
            loss_g_adv = -critic(fake_r, real_w).mean()

            # Moment matching regularization (anti-mode-collapse)
            mp = torch.tensor(0.0, device=device)
            cp = torch.tensor(0.0, device=device)
            kp = torch.tensor(0.0, device=device)
            sp = torch.tensor(0.0, device=device)

            if (moment_lambda > 0 or cov_lambda > 0 or kurt_lambda > 0
                    or sort_lambda > 0):
                # Sample a fresh real batch for moment comparison
                if ptr + config.BATCH_SIZE > N:
                    idx_pool = rng.permutation(N)
                    ptr = 0
                bidx_mm = idx_pool[ptr : ptr + config.BATCH_SIZE]
                ptr += config.BATCH_SIZE
                real_r_mm = torch.tensor(targets_np[bidx_mm], device=device)

                if moment_lambda > 0:
                    # Clamp std to prevent NaN gradients when any feature has
                    # near-zero variance in this batch.
                    mean_pen = ((fake_r.mean(0) - real_r_mm.mean(0)) ** 2).mean()
                    std_pen = ((fake_r.std(0).clamp(min=1e-6)
                                - real_r_mm.std(0).clamp(min=1e-6)) ** 2).mean()
                    mp = mean_pen + std_pen

                # Covariance matching: raw sample covariance (NOT corrcoef).
                # corrcoef divides by per-feature std, which produces NaN
                # gradients when any feature has near-zero variance.
                if cov_lambda > 0 and fake_r.shape[0] >= 4:
                    B_mm = real_r_mm.shape[0]
                    fake_centered = fake_r - fake_r.mean(0)
                    real_centered = real_r_mm - real_r_mm.mean(0)
                    cov_fake = (fake_centered.T @ fake_centered) / (B_mm - 1)
                    cov_real = (real_centered.T @ real_centered) / (B_mm - 1)
                    cp = ((cov_fake - cov_real) ** 2).mean()

                # Kurtosis matching: log-scaled per-instrument excess kurtosis.
                #
                # Real-world kurtosis is extremely heavy-tailed across instruments
                # (e.g. 6M=110, 6S=99, BTC=31; most others ~2-10). A raw squared-
                # difference MSE is dominated by the top 3-5 instruments, which
                # starves the generator of signal for the other 55+. Log-scaling
                # via sign(x)*log1p(|x|) dampens extreme values while preserving
                # ordering and sign, making the penalty contribute roughly equally
                # across all instruments.
                if kurt_lambda > 0 and fake_r.shape[0] >= 4:
                    def _excess_kurtosis(x: torch.Tensor) -> torch.Tensor:
                        m = x.mean(0)
                        s = x.std(0).clamp(min=1e-6)
                        return ((x - m) / s).pow(4).mean(0) - 3.0
                    fk = _excess_kurtosis(fake_r)
                    rk = _excess_kurtosis(real_r_mm)
                    fk_log = torch.sign(fk) * torch.log1p(fk.abs())
                    rk_log = torch.sign(rk) * torch.log1p(rk.abs())
                    kp = ((fk_log - rk_log) ** 2).mean()

                # Sorted-samples matching: 1D Wasserstein-2 distance per
                # instrument. Sort each column of fake and real, then penalise
                # squared rank-matched differences. This directly targets the
                # marginal distribution (QQ) mismatch that moment/kurt miss.
                # torch.sort is differentiable via gather, so gradients flow
                # back to every fake sample.
                if sort_lambda > 0 and fake_r.shape[0] >= 4:
                    fake_sorted = torch.sort(fake_r, dim=0)[0]
                    real_sorted = torch.sort(real_r_mm, dim=0)[0]
                    sp = ((fake_sorted - real_sorted) ** 2).mean()

            loss_g = (loss_g_adv
                      + moment_lambda * mp
                      + cov_lambda * cp
                      + kurt_lambda * kp
                      + sort_lambda * sp)
            loss_g.backward()
            opt_G.step()
            epoch_loss_g.append(loss_g.item())
            epoch_moment_pen.append(mp.item())
            epoch_cov_pen.append(cp.item())
            epoch_kurt_pen.append(kp.item())
            epoch_sort_pen.append(sp.item())

        # ----------------------------------------------------------------
        # End-of-epoch bookkeeping
        # ----------------------------------------------------------------
        if sched_G is not None:
            sched_G.step()
            sched_C.step()

        mean_w  = float(np.mean(epoch_w_est))
        mean_lg = float(np.mean(epoch_loss_g))
        mean_mp = float(np.mean(epoch_moment_pen)) if epoch_moment_pen else 0.0
        mean_cp = float(np.mean(epoch_cov_pen)) if epoch_cov_pen else 0.0
        mean_kp = float(np.mean(epoch_kurt_pen)) if epoch_kurt_pen else 0.0
        mean_sp = float(np.mean(epoch_sort_pen)) if epoch_sort_pen else 0.0
        cur_lr  = opt_G.param_groups[0]["lr"]
        history["wasserstein_estimate"].append(mean_w)
        history["loss_g"].append(mean_lg)
        history["moment_penalty"].append(mean_mp)
        history["cov_penalty"].append(mean_cp)
        history["kurt_penalty"].append(mean_kp)
        history["sort_penalty"].append(mean_sp)
        history["lr_g"].append(cur_lr)

        pbar_epoch.set_postfix(
            W=f"{mean_w:+.3f}",
            G=f"{mean_lg:+.1f}",
            mom=f"{mean_mp:.3f}",
            cov=f"{mean_cp:.3f}",
            krt=f"{mean_kp:.3f}",
            srt=f"{mean_sp:.3f}",
            lr=f"{cur_lr:.1e}",
            cs=n_critic,
        )

        # Save checkpoint every ckpt_every epochs + always on final epoch
        is_final = (epoch == config.N_EPOCHS)
        if epoch % ckpt_every == 0 or is_final:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"wgan_checkpoint_ep{epoch:05d}.pt"
            torch.save(
                {
                    "epoch":           epoch,
                    "generator_state": generator.state_dict(),
                    "critic_state":    critic.state_dict(),
                    "opt_G_state":     opt_G.state_dict(),
                    "opt_C_state":     opt_C.state_dict(),
                    "history":         history,
                },
                ckpt_path,
            )

    generator.eval()
    critic.eval()
    return history
