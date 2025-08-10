import numpy as np, pandas as pd

def make_features(df: pd.DataFrame, levels: int, rolling_ms, horizon_ms: int):
    # Expect columns: ts, bid_px_i, ask_px_i, bid_sz_i, ask_sz_i for i=1..levels
    bid_px = np.vstack([df[f"bid_px_{i}"].to_numpy() for i in range(1, levels+1)]).T
    ask_px = np.vstack([df[f"ask_px_{i}"].to_numpy() for i in range(1, levels+1)]).T
    bid_sz = np.vstack([df[f"bid_sz_{i}"].to_numpy() for i in range(1, levels+1)]).T
    ask_sz = np.vstack([df[f"ask_sz_{i}"].to_numpy() for i in range(1, levels+1)]).T

    mid = (ask_px[:,0] + bid_px[:,0]) * 0.5
    spread = ask_px[:,0] - bid_px[:,0]

    qi = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)  # shape (N, L)

    # Top-level OFI (cheap proxy)
    bs0 = bid_sz[:,0]
    as0 = ask_sz[:,0]
    ofi = np.diff(np.concatenate([[bs0[0]], bs0])) - np.diff(np.concatenate([[as0[0]], as0]))

    X = np.column_stack([mid, spread, ofi, qi.reshape(len(df), -1)])

    # Label: sign of future mid change over rough horizon (index shift approximation)
    step = max(1, int(horizon_ms / 10))  # synthetic data has ~10ms resolution
    y = np.sign(np.roll(mid, -step) - mid)
    y[y == 0] = 0
    y = y[:-step]
    X = X[:-step]
    return X.astype("float32"), y.astype("int8")
