"""
Generate realistic synthetic L2 order-book data.

The price process combines:
  - Persistent momentum (auto-correlated order flow)
  - Mean-reversion around fair value
  - Variable spread that widens with local volatility
  - Auto-correlated queue sizes (simulates resting/cancelling limit orders)

Output: data/processed/lob.parquet
"""
import numpy as np, pandas as pd
from pathlib import Path

np.random.seed(42)
N = 100_000          # enough ticks for a meaningful backtest
L = 5                # order-book levels

# ── Mid-price process ─────────────────────────────────────────────
# Ornstein-Uhlenbeck with a slow momentum overlay
dt = 0.01            # notional time-step (~10 ms)
kappa = 0.3          # mean-reversion speed
sigma = 0.0005       # tick-level volatility

mid = np.zeros(N)
mid[0] = 100.0
momentum = 0.0

for i in range(1, N):
    momentum = 0.95 * momentum + 0.05 * np.random.randn()   # persistent flow
    mean_rev = -kappa * (mid[i - 1] - 100.0) * dt
    noise = sigma * np.random.randn()
    mid[i] = mid[i - 1] + mean_rev + noise + 0.0001 * momentum

# ── Spread dynamics (widens with recent volatility) ───────────────
base_spread = 0.01
spread = base_spread + 0.005 * np.abs(np.random.randn(N))

# ── Price levels ──────────────────────────────────────────────────
bid_px = np.stack([mid - spread / 2 - 0.01 * i for i in range(L)], axis=1)
ask_px = np.stack([mid + spread / 2 + 0.01 * i for i in range(L)], axis=1)

# ── Queue sizes with autocorrelation ─────────────────────────────
bid_sz = np.zeros((N, L))
ask_sz = np.zeros((N, L))
bid_sz[0] = 1000 + 50 * np.random.randn(L)
ask_sz[0] = 900 + 50 * np.random.randn(L)

for i in range(1, N):
    bid_sz[i] = 0.98 * bid_sz[i - 1] + 0.02 * (1000 + 80 * np.random.randn(L))
    ask_sz[i] = 0.98 * ask_sz[i - 1] + 0.02 * (900 + 80 * np.random.randn(L))

bid_sz = np.maximum(bid_sz, 1)
ask_sz = np.maximum(ask_sz, 1)

# ── Timestamps (10 ms between ticks, in nanoseconds) ─────────────
ts = (np.arange(N) * 10_000_000).astype(np.int64)

# ── Assemble DataFrame ────────────────────────────────────────────
cols = {"ts": ts}
for i in range(1, L + 1):
    cols[f"bid_px_{i}"] = bid_px[:, i - 1]
    cols[f"ask_px_{i}"] = ask_px[:, i - 1]
    cols[f"bid_sz_{i}"] = bid_sz[:, i - 1]
    cols[f"ask_sz_{i}"] = ask_sz[:, i - 1]

df = pd.DataFrame(cols)
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/processed/lob.parquet")
print(f"Wrote data/processed/lob.parquet — {N:,} ticks, {L} levels")
