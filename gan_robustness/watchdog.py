"""
watchdog.py -- Monitor CWGAN training for convergence and kill it early
if it's diverging, so we don't waste 15+ hours on a bad run.

USAGE (run in a separate terminal alongside run_pipeline.py):

    # Report-only mode (default). Prints diagnostics, never kills.
    python watchdog.py

    # Auto-kill mode. Kills the run_pipeline.py process if criteria fail.
    python watchdog.py --kill

    # Custom poll interval (default 300s = 5 min):
    python watchdog.py --poll-seconds 120

HOW IT WORKS:
1. Polls the `checkpoints/` directory every N seconds.
2. When a new wgan_checkpoint_epXXXXX.pt file appears, loads it and
   inspects the `history` dict to check for divergence/collapse.
3. Logs verdict to stdout and `watchdog.log`.
4. In --kill mode, terminates the training process if a criterion fails.

CONVERGENCE CRITERIA (at each new checkpoint, based on last-10-epoch means):
  - Wasserstein collapse:  W < 0.2   (critic gave up)
  - Wasserstein explosion: W > 8.0   (critic overpowering)
  - Kurtosis blowup:       krt > 1.5x value at first checkpoint
  - NaN/Inf in any metric: meltdown
  - Generator runaway:     loss_g > 50

The watchdog does NOT fire until at least one checkpoint exists, so
training always runs for at least 250 epochs before any early-stop.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

_HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = _HERE / "checkpoints"
LOG_FILE = _HERE / "watchdog.log"

# Thresholds -- tune these based on observed training dynamics
THRESH_W_COLLAPSE   = 0.2    # W below this = critic gave up
THRESH_W_EXPLOSION  = 8.0    # W above this = critic overpowering
THRESH_KRT_BLOWUP   = 1.5    # factor vs first checkpoint
THRESH_LOSS_G_MAX   = 50.0   # loss_g upper bound
LAST_N              = 10     # mean over last-N epochs


def _log(msg: str, also_file: bool = True) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if also_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _last_n_mean(values: list[float], n: int = LAST_N) -> float:
    if not values:
        return float("nan")
    tail = values[-n:]
    return sum(tail) / len(tail)


def _any_bad(vals: Iterable[float]) -> bool:
    return any((not isinstance(v, (int, float))) or math.isnan(v) or math.isinf(v)
               for v in vals)


def _load_history(ckpt_path: Path) -> dict | None:
    try:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        _log(f"  [WARN] Could not load {ckpt_path.name}: {exc}")
        return None
    return ck.get("history")


def _evaluate(history: dict, first_krt_mean: float | None) -> tuple[bool, str, dict]:
    """
    Returns (is_healthy, reason, metrics_snapshot).
    """
    W_series    = history.get("wasserstein_estimate", [])
    G_series    = history.get("loss_g", [])
    mom_series  = history.get("moment_penalty", [])
    cov_series  = history.get("cov_penalty", [])
    krt_series  = history.get("kurt_penalty", [])

    W_mean   = _last_n_mean(W_series)
    G_mean   = _last_n_mean(G_series)
    mom_mean = _last_n_mean(mom_series)
    cov_mean = _last_n_mean(cov_series)
    krt_mean = _last_n_mean(krt_series)

    snap = dict(W=W_mean, G=G_mean, mom=mom_mean, cov=cov_mean, krt=krt_mean,
                epoch=len(W_series))

    # Check 1: NaN/Inf anywhere
    if _any_bad([W_mean, G_mean, mom_mean, cov_mean, krt_mean]):
        return False, "numerical meltdown (NaN/Inf in metrics)", snap

    # Check 2: Wasserstein collapse
    if W_mean < THRESH_W_COLLAPSE:
        return False, (f"W collapsed: last-{LAST_N} mean = {W_mean:.3f} "
                       f"< {THRESH_W_COLLAPSE} (critic gave up)"), snap

    # Check 3: Wasserstein explosion
    if W_mean > THRESH_W_EXPLOSION:
        return False, (f"W exploded: last-{LAST_N} mean = {W_mean:.3f} "
                       f"> {THRESH_W_EXPLOSION} (critic overpowering)"), snap

    # Check 4: Generator runaway
    if abs(G_mean) > THRESH_LOSS_G_MAX:
        return False, (f"loss_g runaway: last-{LAST_N} mean = {G_mean:+.3f} "
                       f"(|.| > {THRESH_LOSS_G_MAX})"), snap

    # Check 5: Kurtosis blowup vs first checkpoint
    if first_krt_mean is not None and first_krt_mean > 1e-6:
        ratio = krt_mean / first_krt_mean
        if ratio > THRESH_KRT_BLOWUP:
            return False, (f"krt blowup: last-{LAST_N} mean = {krt_mean:.3f} "
                           f"= {ratio:.2f}x the first-ckpt value "
                           f"{first_krt_mean:.3f} (> {THRESH_KRT_BLOWUP}x)"), snap

    return True, "all checks passed", snap


def _find_training_process() -> "psutil.Process | None":
    try:
        import psutil
    except ImportError:
        _log("  [WARN] psutil not installed; cannot auto-kill. "
             "Install with: pip install psutil")
        return None

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmd)
            if "run_pipeline.py" in cmd_str and "python" in (proc.info.get("name") or "").lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def _kill_training_process() -> bool:
    proc = _find_training_process()
    if proc is None:
        _log("  [WARN] Could not find training process to kill.")
        return False
    try:
        _log(f"  [KILL] Terminating training process PID={proc.pid}")
        proc.terminate()
        try:
            proc.wait(timeout=10)
            _log(f"  [KILL] Process {proc.pid} terminated cleanly.")
            return True
        except Exception:
            _log(f"  [KILL] Timeout -- force-killing PID={proc.pid}")
            proc.kill()
            return True
    except Exception as exc:
        _log(f"  [WARN] Failed to kill process: {exc}")
        return False


def _list_checkpoints() -> list[Path]:
    if not CHECKPOINT_DIR.exists():
        return []
    ckpts = sorted(CHECKPOINT_DIR.glob("wgan_checkpoint_ep*.pt"))
    return ckpts


def _epoch_of(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.name)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Watchdog for CWGAN training convergence.")
    ap.add_argument("--kill", action="store_true",
                    help="Kill run_pipeline.py if convergence fails (default: report only)")
    ap.add_argument("--poll-seconds", type=int, default=300,
                    help="Polling interval in seconds (default: 300 = 5 min)")
    args = ap.parse_args()

    mode = "KILL" if args.kill else "REPORT-ONLY"
    _log(f"=== Watchdog starting ({mode} mode, poll every {args.poll_seconds}s) ===")
    _log(f"Watching: {CHECKPOINT_DIR}")
    _log(f"Thresholds: W in [{THRESH_W_COLLAPSE}, {THRESH_W_EXPLOSION}], "
         f"|loss_g| <= {THRESH_LOSS_G_MAX}, krt <= {THRESH_KRT_BLOWUP}x first-ckpt")

    seen_paths: set[Path] = set()
    first_krt_mean: float | None = None

    try:
        while True:
            ckpts = _list_checkpoints()
            new_ckpts = [c for c in ckpts if c not in seen_paths]

            for ck in new_ckpts:
                epoch = _epoch_of(ck)
                _log(f"New checkpoint: ep{epoch:05d} ({ck.name})")
                history = _load_history(ck)
                seen_paths.add(ck)

                if history is None:
                    continue

                healthy, reason, snap = _evaluate(history, first_krt_mean)

                # Record first-checkpoint krt as baseline for blowup check
                if first_krt_mean is None:
                    first_krt_mean = snap["krt"]
                    _log(f"  [BASELINE] first-ckpt krt = {first_krt_mean:.3f}")

                _log(f"  metrics @ ep{snap['epoch']}: "
                     f"W={snap['W']:+.3f}  G={snap['G']:+.3f}  "
                     f"mom={snap['mom']:.3f}  cov={snap['cov']:.3f}  "
                     f"krt={snap['krt']:.3f}")

                if healthy:
                    _log(f"  [OK] {reason}")
                else:
                    _log(f"  [FAIL] {reason}")
                    if args.kill:
                        _log("  [ACTION] --kill flag set; terminating training.")
                        _kill_training_process()
                        _log("=== Watchdog exiting after kill. ===")
                        return 0
                    else:
                        _log("  [ACTION] --kill not set; reporting only. "
                             "You should manually stop training if this "
                             "failure is genuine.")

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        _log("=== Watchdog stopped by user (Ctrl+C). ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
