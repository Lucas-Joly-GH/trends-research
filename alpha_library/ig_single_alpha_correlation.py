"""
Single Alpha Correlation Analysis
==================================
Loads all Single_Alpha_*_portfolio_returns.csv files, aligns daily returns,
computes pairwise Pearson correlation, prints summary, saves CSV + heatmap.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent


def main():
    # ── 1. Discover all Single_Alpha portfolio return files ──
    csv_files = sorted(_HERE.glob("Single_Alpha_*_portfolio_returns.csv"))
    if not csv_files:
        print("[ERROR] No Single_Alpha_*_portfolio_returns.csv files found.")
        return

    print(f"Found {len(csv_files)} single-alpha return files:\n")

    # ── 2. Load and align daily returns ──
    returns_dict = {}
    for f in csv_files:
        name = f.stem.replace("_portfolio_returns", "").replace("Single_Alpha_", "")
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df.set_index("Date")
        returns_dict[name] = df["Daily_Return"]

    returns_df = pd.DataFrame(returns_dict)
    # Only keep dates where at least 2 alphas have non-zero returns
    mask = (returns_df != 0.0).sum(axis=1) >= 2
    returns_df = returns_df.loc[mask]
    print(f"Common date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"  {len(returns_df)} trading days, {len(returns_df.columns)} alphas\n")

    # ── 3. Pairwise Pearson correlation ──
    corr_matrix = returns_df.corr()

    print("=" * 80)
    print("  PAIRWISE CORRELATION MATRIX")
    print("=" * 80)
    # Print with nice formatting
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(corr_matrix.to_string())
    print()

    # ── 4. Save correlation matrix CSV ──
    corr_path = _HERE / "single_alpha_correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)
    print(f"[SAVED] {corr_path}")

    # ── 5. Heatmap ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Single Alpha Pairwise Correlation Matrix", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_path = _HERE / "single_alpha_correlation_heatmap.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {heatmap_path}")
    except ImportError:
        print("[WARN] matplotlib/seaborn not available; skipping heatmap.")

    # ── 6. Average correlation per alpha (sorted by most orthogonal) ──
    print("\n" + "=" * 80)
    print("  AVERAGE PAIRWISE CORRELATION (sorted: most orthogonal first)")
    print("=" * 80)
    avg_corr = {}
    for col in corr_matrix.columns:
        others = corr_matrix[col].drop(col)
        avg_corr[col] = others.mean()

    avg_corr_sorted = sorted(avg_corr.items(), key=lambda x: abs(x[1]))
    print(f"  {'Alpha':<25s} {'Avg Corr':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for name, ac in avg_corr_sorted:
        print(f"  {name:<25s} {ac:>10.4f}")

    # ── 7. Summary table with performance metrics ──
    print("\n" + "=" * 100)
    print("  SINGLE ALPHA PERFORMANCE SUMMARY")
    print("=" * 100)

    pkl_files = sorted(_HERE.glob("Single_Alpha_*_checkpoint.pkl"))
    summary_rows = []
    for f in pkl_files:
        name = f.stem.replace("_checkpoint", "").replace("Single_Alpha_", "")
        with open(f, "rb") as fh:
            ckpt = pickle.load(fh)

        # Post-2010 SR
        dates = ckpt["dates"]
        daily_ret = ckpt["daily_ret"]
        post2010_mask = dates >= pd.Timestamp("2010-01-01")
        if post2010_mask.sum() > 256:
            post_ret = daily_ret[post2010_mask]
            sr_post = (post_ret.mean() * 256) / (post_ret.std() * np.sqrt(256)) if post_ret.std() > 0 else 0.0
        else:
            sr_post = np.nan

        ac = avg_corr.get(name, np.nan)
        summary_rows.append({
            "Alpha": name,
            "SR_Full": ckpt["sr"],
            "SR_Post2010": sr_post,
            "CAGR": ckpt["cagr"],
            "Max_DD": ckpt["max_dd"],
            "Avg_Corr": ac,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("SR_Full", ascending=False)
    print(f"  {'Alpha':<25s} {'SR(Full)':>10s} {'SR(>2010)':>10s} {'CAGR':>10s} {'Max DD':>10s} {'Avg Corr':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Alpha']:<25s} {row['SR_Full']:>10.3f} {row['SR_Post2010']:>10.3f} "
              f"{row['CAGR']:>9.2%} {row['Max_DD']:>9.2%} {row['Avg_Corr']:>10.4f}")

    summary_df.to_csv(_HERE / "single_alpha_summary.csv", index=False)
    print(f"\n[SAVED] {_HERE / 'single_alpha_summary.csv'}")


if __name__ == "__main__":
    main()
