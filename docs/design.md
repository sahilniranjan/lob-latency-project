# Design Document

## 1. Pipeline Overview

```
Exchange API  ──▶  Python Collector  ──▶  Parquet file
                       │
                       ▼
              Feature Engineering  ──▶  Train (sklearn Pipeline)
                       │                       │
                       ▼                       ▼
              Backtest Engine           Export weights (.txt)
              (sim execution)                  │
                                               ▼
                                    C++ Inference Engine
                                    ┌─────────────────┐
                                    │ Producer thread  │
                                    │ (feed L2 ticks)  │
                                    │       │          │
                                    │  SPSC Ring Buf   │
                                    │       │          │
                                    │ Worker thread    │
                                    │ (feat → score)   │
                                    └─────────────────┘
```

## 2. Feature Order Contract

Python and C++ MUST produce features in identical order:

```
Index   Feature
──────  ────────────────────
  0     mid
  1     spread
  2     ofi
  3     microprice
 4–8    qi[0] … qi[4]
9–11    rolling_ofi     (10/20/50 ms)
12–14   rolling_spread  (10/20/50 ms)
15–17   rolling_qi0     (10/20/50 ms)
──────  ────────────────────
Total:  18 features
```

## 3. Thread Model (C++ Engine)

| Thread | Core Pin | Role |
|--------|----------|------|
| Producer | 2 | Parse feed → push `L2Tick` into SPSC ring buffer |
| Worker | 4 | Pop tick → compute features → dot-product score → emit signal |

- Lock-free SPSC with power-of-2 capacity (131072)
- `memory_order_acquire` on consumer, `memory_order_release` on producer
- No mutexes, no syscalls in hot path

## 4. Memory Layout

- `L2Tick`: 4 × `std::array<float,5>` + `uint64_t` = 88 bytes
- `FeatureVec`: `std::vector<float>` with `reserve(64)` — single heap allocation, reused
- `RollingMean`: fixed-size circular buffer per window — O(1) update, no allocation
- `SpscRingBuffer<TickMsg>`: contiguous `std::vector<TickMsg>` — pre-allocated, cache-friendly

## 5. Scaling Strategy

Training uses `StandardScaler` (z-score normalisation). Rather than scaling at
inference time, `export_linear_weights.py` bakes the scaler into the linear
weights:

```
w_combined[i] = w[i] / scale[i]
b_combined    = b − Σ(w[i] × mean[i] / scale[i])
```

The C++ engine does a single raw dot product — zero runtime overhead.

## 6. Data Collection

Multi-exchange REST poller (`scripts/collect_binance_l2.py`):
- KuCoin (default, global), Kraken, Bybit, Binance
- 3 Hz polling, ~180 snapshots/min
- No API key required
- Auto-saves raw + processed Parquet
