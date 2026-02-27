import numpy as np, pandas as pd


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using cumsum (no pandas dependency in hot path)."""
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cs[window:] - cs[:-window]) / window
    # Pad front with first valid value
    return np.concatenate([np.full(window - 1, out[0]), out])


def make_features(df: pd.DataFrame, levels: int, rolling_ms, horizon_ms: int):
    """
    Compute feature matrix from L2 order-book DataFrame.

    Canonical feature order (must match C++ FeatureEngine):
        [mid, spread, ofi, microprice,
         qi_0 .. qi_{L-1},
         rolling_ofi_{w0}, rolling_ofi_{w1}, ...,
         rolling_spread_{w0}, rolling_spread_{w1}, ...,
         rolling_imbalance_{w0}, rolling_imbalance_{w1}, ...]

    Returns
    -------
    X : ndarray, float32, shape (N-step, n_features)
    y : ndarray, int8,    shape (N-step,)  values in {-1, 0, 1}
    """
    bid_px = np.vstack([df[f"bid_px_{i}"].to_numpy() for i in range(1, levels+1)]).T
    ask_px = np.vstack([df[f"ask_px_{i}"].to_numpy() for i in range(1, levels+1)]).T
    bid_sz = np.vstack([df[f"bid_sz_{i}"].to_numpy() for i in range(1, levels+1)]).T
    ask_sz = np.vstack([df[f"ask_sz_{i}"].to_numpy() for i in range(1, levels+1)]).T

    mid = (ask_px[:, 0] + bid_px[:, 0]) * 0.5
    spread = ask_px[:, 0] - bid_px[:, 0]

    # Queue imbalance per level  — shape (N, L)
    qi = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)

    # Order-flow imbalance (top level, single-tick diff)
    bs0 = bid_sz[:, 0]
    as0 = ask_sz[:, 0]
    ofi = np.diff(np.concatenate([[bs0[0]], bs0])) - np.diff(np.concatenate([[as0[0]], as0]))

    # Microprice (size-weighted mid)
    microprice = (bid_px[:, 0] * ask_sz[:, 0] + ask_px[:, 0] * bid_sz[:, 0]) / \
                 (bid_sz[:, 0] + ask_sz[:, 0] + 1e-9)

    # ── Base features ─────────────────────────────────────────────
    base = [mid, spread, ofi, microprice, qi.reshape(len(df), -1)]

    # ── Rolling window features ───────────────────────────────────
    # Estimate tick duration from timestamps (ns)
    ts = df["ts"].values
    if len(ts) > 1:
        median_tick_ns = float(np.median(np.diff(ts)))
        tick_ms = max(median_tick_ns / 1e6, 0.001)  # at least 1 µs
    else:
        tick_ms = 10.0  # default for synthetic data

    rolling_cols: list[np.ndarray] = []
    windows = rolling_ms if rolling_ms else []
    for w_ms in windows:
        w_ticks = max(2, int(w_ms / tick_ms))
        rolling_cols.append(_rolling_mean(ofi, w_ticks))           # rolling OFI
        rolling_cols.append(_rolling_mean(spread, w_ticks))        # rolling spread
        rolling_cols.append(_rolling_mean(qi[:, 0], w_ticks))      # rolling top-level imbalance

    X_parts = [mid, spread, ofi, microprice]
    X_parts.append(qi.reshape(len(df), -1))
    for rc in rolling_cols:
        X_parts.append(rc)

    X = np.column_stack(X_parts)

    # Label: sign of future mid-price change
    step = max(1, int(horizon_ms / tick_ms))
    # Guard against step >= N
    step = min(step, len(mid) - 1)
    y = np.sign(np.roll(mid, -step) - mid)
    y[y == 0] = 0
    y = y[:-step]
    X = X[:-step]
    return X.astype("float32"), y.astype("int8")
