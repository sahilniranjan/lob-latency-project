# LOB Predictor — Latency‑Optimized Inference Engine

Predict short‑horizon mid‑price moves from L2 order‑book data and run inference end‑to‑end in **< 0.7 ms p99** on commodity CPU.

🌐 **Live Dashboard:** [lob-latency.streamlit.app](https://lob-latency.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![C++](https://img.shields.io/badge/C++-20-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────────┐
│  Data Source  │────▶│ Python Train  │────▶│  C++ Inference Eng.  │
│ (KuCoin API) │     │ + Backtest    │     │  (< 0.7 ms p99)     │
└──────────────┘     └───────────────┘     └──────────────────────┘
   REST/3 Hz           scikit-learn          lock-free SPSC ring
   L2 snapshots        StandardScaler        pre-alloc features
   BTC, ETH, …         + LogReg → .pkl       rolling windows
```

**Two‑phase pipeline:**
- **Research (Python)** — collect data → compute features → train model → backtest with execution sim
- **Engine (C++)** — lock‑free ring buffer → feature calc → dot‑product scoring → signal

## Features Computed

| Feature | Count | Description |
|---------|-------|-------------|
| Mid‑price | 1 | `(ask₁ + bid₁) / 2` |
| Spread | 1 | `ask₁ − bid₁` |
| Order‑flow imbalance | 1 | `Δbid_sz₁ − Δask_sz₁` |
| Microprice | 1 | Size‑weighted mid |
| Queue imbalance | 5 | Per‑level `(bid_sz − ask_sz) / (bid_sz + ask_sz)` |
| Rolling OFI | 3 | Mean OFI over 10/20/50 ms windows |
| Rolling spread | 3 | Mean spread over 10/20/50 ms windows |
| Rolling imbalance | 3 | Mean top‑level QI over 10/20/50 ms windows |
| **Total** | **18** | |

## Quickstart

```bash
# 1) Setup
pip install -r requirements.txt

# 2) Collect real data (free, no API key — uses KuCoin)
python scripts/collect_binance_l2.py --exchange kucoin --symbol BTCUSDT --duration_min 30

# 3) Train
python research/10_train_logreg.py

# 4) Backtest
python research/30_backtest.py

# 5) Export weights for C++ engine
python scripts/export_linear_weights.py

# 6) Build & benchmark C++ engine
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
./engine/build/bench
./engine/build/replay
```

## Project Structure

```
├── configs/
│   └── default.yaml           # Feature / training / backtest config
├── data/
│   ├── raw/                   # Raw exchange snapshots (auto-generated)
│   └── processed/             # lob.parquet ready for training
├── docs/
│   ├── design.md              # Architecture & thread model
│   ├── latency-report.md      # Benchmark results
│   └── results.md             # Backtest metrics
├── engine/                    # C++ inference engine
│   ├── include/
│   │   ├── features.hpp       # Feature engine + rolling buffers
│   │   ├── model_scorer.hpp   # Linear scorer interface
│   │   ├── ring_buffer.hpp    # Lock-free SPSC ring buffer
│   │   └── timing.hpp         # Microsecond timer
│   └── src/
│       ├── features.cpp       # Feature computation + CSV reader
│       ├── model_scorer.cpp   # Dot-product + sigmoid scorer
│       ├── bench_latency.cpp  # Single-threaded latency benchmark
│       └── main_replay.cpp    # Two-thread producer/consumer replay
├── research/                  # Python research pipeline
│   ├── 00_make_synth_data.py  # Synthetic data generator (100k ticks)
│   ├── 10_train_logreg.py     # Train StandardScaler + LogReg pipeline
│   ├── 20_export_onnx.py      # Export to ONNX (optional)
│   ├── 30_backtest.py         # Full backtest with execution sim
│   ├── utils_data.py          # Feature engineering
│   └── utils_backtest.py      # Execution simulator + metrics
├── scripts/
│   ├── collect_binance_l2.py  # Multi-exchange L2 data collector
│   ├── load_fi2010.py         # FI-2010 benchmark dataset loader
│   └── export_linear_weights.py  # Export pipeline weights for C++
├── outputs/                   # Model artifacts + backtest reports
└── requirements.txt
```

## Data Sources (All Free)

| Source | Type | How |
|--------|------|-----|
| **KuCoin REST API** | Live crypto L2 snapshots | `python scripts/collect_binance_l2.py` |
| **Kraken REST API** | Live crypto L2 snapshots | `--exchange kraken` |
| **FI-2010 benchmark** | Academic LOB dataset (5 Finnish stocks) | `python scripts/load_fi2010.py` |
| **Synthetic** | Ornstein-Uhlenbeck + microstructure | `python research/00_make_synth_data.py` |

## Latency Targets

| Stage | Target | Typical |
|-------|--------|---------|
| Feature calculation | < 200 µs | ~100 µs |
| Model scoring | < 50 µs | ~15 µs |
| End‑to‑end p99 | **< 700 µs** | ~200 µs |

## Backtest Metrics

The backtest engine ([research/30_backtest.py](research/30_backtest.py)) reports:

- **Sharpe ratio** (annualised from daily PnL)
- **Max drawdown**
- **Win rate** (per round-trip)
- **Cost analysis** (gross PnL vs transaction costs)
- **Equity curve + drawdown chart** (PNG)

## Key Design Decisions

1. **Logistic regression** — single dot product, deterministic latency, no branching
2. **Lock‑free SPSC ring buffer** — power‑of‑2 masking, `acquire/release` memory ordering
3. **Pre‑allocated feature vector** — `reserve(64)` avoids heap allocation in hot path
4. **StandardScaler baked into weights** — C++ engine does raw dot product, no runtime scaling
5. **Rolling windows in ring buffers** — O(1) per update, fixed memory, no allocations
6. **No ONNX Runtime dependency** — lightweight text weight format for the C++ engine

## License

MIT
