import numpy as np, pandas as pd
from pathlib import Path

np.random.seed(0)
N = 20000
L = 5

# make a synthetic mid-price random walk + microstructure
mid = 100 + np.cumsum(np.random.randn(N)*0.001)
spread = np.full(N, 0.01)

bid_px = np.stack([mid - spread/2 - 0.01*i for i in range(L)], axis=1)
ask_px = np.stack([mid + spread/2 + 0.01*i for i in range(L)], axis=1)
bid_sz = np.maximum(1000 + 50*np.random.randn(N, L), 1)
ask_sz = np.maximum( 900 + 50*np.random.randn(N, L), 1)

ts = (np.arange(N) * 10_000_000).astype(np.int64)  # 10ms in ns

cols = {"ts": ts}
for i in range(1, L+1):
    cols[f"bid_px_{i}"] = bid_px[:, i-1]
    cols[f"ask_px_{i}"] = ask_px[:, i-1]
    cols[f"bid_sz_{i}"] = bid_sz[:, i-1]
    cols[f"ask_sz_{i}"] = ask_sz[:, i-1]

df = pd.DataFrame(cols)
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/processed/lob.parquet")
print("Wrote data/processed/lob.parquet with synthetic L2 data")
