# LOB Predictor with Latency‑Optimized Inference

**Goal:** Predict short‑horizon mid‑price move (Δ in 100 ms) from L2 order book, and run inference end‑to‑end in **< 0.7 ms p99** on commodity CPU.

**Stack:** Python (training) → ONNX → C++ (inference), lock‑free ring buffer, pre‑allocated features, single‑producer/single‑consumer.

## Quickstart

```bash
# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Generate synthetic data (creates data/processed/lob.parquet)
python research/00_make_synth_data.py

# 3) Train + export weights
python research/10_train_logreg.py --config configs/default.yaml --data data/processed/lob.parquet

# Export ONNX (optional, if using onnxruntime)
python research/20_export_onnx.py --model_path outputs/logreg.pkl --onnx engine/src/onnx/model.onnx

# Or export lightweight linear weights (no ORT needed)
python scripts/export_linear_weights.py

# 4) Build C++ engine
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j

# 5) Benchmark & run
./engine/build/bench
./engine/build/replay
```

## Targets
- Feature calc: 100–200 µs
- Model score: 10–50 µs
- E2E p99: < 0.7 ms

## Data
This repo includes a synthetic data generator so you can run end‑to‑end without external data. Replace `data/processed/lob.parquet` with real L2 data when available (schema: `ts`, `bid_px_1..L`, `ask_px_1..L`, `bid_sz_1..L`, `ask_sz_1..L`).

## Notes
- If ONNX Runtime is unavailable, use the provided fallback `LinearLogitScorer` via `scripts/export_linear_weights.py`.
- Thread pinning and kernel tuning are omitted for portability.
