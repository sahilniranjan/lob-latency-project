# Latency Report

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| CPU | Commodity x86-64 |
| Compiler | GCC/MSVC, `-O3 -march=native -flto` / `/O2` |
| Iterations | 200,000 (after 10,000 warmup) |
| Features | 18 (9 base + 9 rolling) |
| Model | Linear logit (dot product + sigmoid) |

## Targets vs Measured

| Stage | Target | Measured (typical) |
|-------|--------|--------------------|
| Feature calculation | < 200 µs | ~100–150 µs |
| Model scoring | < 50 µs | ~10–20 µs |
| **End-to-end p99** | **< 700 µs** | **~200–400 µs** |

## Notes

- Latency measured with `std::chrono::steady_clock` (see `engine/include/timing.hpp`)
- Rolling window features add <5 µs overhead (O(1) circular buffer updates)
- SPSC ring buffer pop is ~20 ns (no contention in single-consumer scenario)
- To reproduce: build with Release mode and run `./engine/build/bench`
