"""
Backtest: replay L2 order-book data through the trained model and
simulate execution with realistic transaction costs.

Usage
-----
    python research/30_backtest.py
    python research/30_backtest.py --config configs/default.yaml \\
                                   --data   data/processed/lob.parquet \\
                                   --fee_bps 10

Outputs  →  outputs/backtest/
    results.csv          tick-level simulation log
    metrics.json         summary performance metrics
    backtest_report.png  equity curve, drawdown, positions chart
"""

import argparse, json, sys, yaml
import numpy as np, pandas as pd
from pathlib import Path
from joblib import load

# Ensure sibling modules are importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils_data import make_features
from utils_backtest import ExecutionSimulator, compute_metrics


def main():
    ap = argparse.ArgumentParser(description="LOB strategy backtest")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/lob.parquet")
    ap.add_argument("--model", default=None, help="Override model path (.pkl)")
    ap.add_argument("--upper_thresh", type=float, default=0.55)
    ap.add_argument("--lower_thresh", type=float, default=0.45)
    ap.add_argument("--fee_bps", type=float, default=10.0,
                    help="One-way transaction cost in basis points (default 10 = 0.1%%)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_path = args.model or cfg["model_pkl"]

    # ── 1.  Load data & compute features ──────────────────────────
    print(f"[1/5] Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    print(f"       {len(df):,} ticks, {cfg['levels']} levels")

    X, y = make_features(df, cfg["levels"], cfg["rolling_windows_ms"], cfg["horizon_ms"])

    # Out-of-sample split (same as training)
    n = len(X)
    cut = int(n * (1 - cfg["val_split_time"]))
    X_val, y_val = X[cut:], y[cut:]

    # Matching mid-prices for PnL calculation
    mid_all = (
        df["ask_px_1"].values + df["bid_px_1"].values
    ) * 0.5
    # Trim to same length as X (make_features trims `step` rows from end)
    step = len(df) - len(X)           # guaranteed to match make_features
    if step > 0:
        mid_all = mid_all[:-step]
    mid_val = mid_all[cut:]

    print(f"       train {cut:,}  |  val {len(X_val):,} ticks")

    # ── 2.  Load model ────────────────────────────────────────────
    print(f"[2/5] Loading model from {model_path}")
    clf = load(model_path)

    # ── 3.  Generate predictions ──────────────────────────────────
    print(f"[3/5] Scoring {len(X_val):,} validation ticks")
    proba = clf.predict_proba(X_val)

    # Probability of the +1 (up) class
    classes = list(clf.classes_)
    up_idx = classes.index(1) if 1 in classes else len(classes) - 1
    signals = proba[:, up_idx]

    print(f"       signal range [{signals.min():.3f}, {signals.max():.3f}]  "
          f"mean {signals.mean():.3f}")

    # ── 4.  Simulate execution ────────────────────────────────────
    print(f"[4/5] Running execution sim  "
          f"(fee={args.fee_bps} bps, thresholds={args.lower_thresh}/{args.upper_thresh})")

    sim = ExecutionSimulator(maker_fee_bps=args.fee_bps, latency_ticks=1)
    results = sim.run(
        mid_val, signals,
        upper_thresh=args.upper_thresh,
        lower_thresh=args.lower_thresh,
    )

    # ── 5.  Metrics ───────────────────────────────────────────────
    ts = df["ts"].values
    ts_range_ns = float(ts[-1] - ts[0])
    ts_range_days = ts_range_ns / (86_400 * 1e9) if ts_range_ns > 0 else 1.0
    ticks_per_day = len(df) / max(ts_range_days, 1e-9)

    metrics = compute_metrics(results, ticks_per_day=ticks_per_day)

    # ── Save ──────────────────────────────────────────────────────
    out_dir = Path("outputs/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("   BACKTEST RESULTS  (out-of-sample)")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"   {k:25s} : {v}")
    print("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────
    print(f"\n[5/5] Saving plots to {out_dir}/")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        eq = results["equity"].values
        axes[0].plot(eq, linewidth=0.7, color="steelblue")
        axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].set_ylabel("Cumulative PnL")
        axes[0].set_title("Equity Curve  (out-of-sample)")
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        running_max = np.maximum.accumulate(eq)
        dd = eq - running_max
        axes[1].fill_between(range(len(dd)), dd, 0, color="salmon", alpha=0.6)
        axes[1].set_ylabel("Drawdown")
        axes[1].set_title("Drawdown")
        axes[1].grid(True, alpha=0.3)

        # Position
        axes[2].step(range(len(results)), results["position"].values,
                     linewidth=0.5, color="dimgray", where="post")
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Tick")
        axes[2].set_title("Position Over Time")
        axes[2].set_yticks([-1, 0, 1])
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "backtest_report.png", dpi=150)
        plt.close()
        print(f"   Saved {out_dir / 'backtest_report.png'}")

    except ImportError:
        print("   matplotlib not installed — skipping plots  (pip install matplotlib)")

    print(f"\nDone.  Full results in {out_dir}/")


if __name__ == "__main__":
    main()
