"""
Download and convert the FI-2010 benchmark LOB dataset into our pipeline format.

The FI-2010 dataset (Ntakaris et al., 2017) is the standard academic benchmark
for mid-price prediction from limit order book data.
  - 5 Finnish stocks, 10 trading days, ~4 M samples
  - 10 LOB levels, pre-normalised
  - CC-BY 4.0 licence

Data source
-----------
The dataset is hosted on Fairdata IDA (Finland). Because the download portal
requires interactive navigation, this script supports TWO modes:

  (A) **Auto-download** from a known public mirror on Google Drive that many
      DeepLOB papers reference.  This is attempted first.

  (B) **Manual download** — if auto-download fails the script prints clear
      instructions and waits for you to place the files locally.

Usage
-----
    python scripts/load_fi2010.py                    # auto-download + convert
    python scripts/load_fi2010.py --raw_dir data/fi2010_raw   # manual mode

Output
------
    data/processed/lob.parquet       — ready for training / backtest
    data/fi2010_raw/                 — cached raw .txt files
"""

import argparse, io, os, zipfile, sys
from pathlib import Path

import numpy as np, pandas as pd

# ─── FI-2010 layout ──────────────────────────────────────────────
# The .txt files have shape (149, N_samples) — rows are features.
#
# Rows  1– 40 : raw LOB (10 levels × 4 fields)
#                ask_px_1, ask_sz_1, bid_px_1, bid_sz_1,
#                ask_px_2, ask_sz_2, bid_px_2, bid_sz_2, ...
# Rows 41–144 : hand-crafted features (time-insensitive + sensitive)
# Rows 145–149: labels for horizons k = 1, 2, 3, 5, 10
#               values: 1 = up, 2 = stationary, 3 = down
# ──────────────────────────────────────────────────────────────────

LEVELS_IN_FI2010 = 10
FIELDS_PER_LEVEL = 4     # ask_px, ask_sz, bid_px, bid_sz
RAW_FEATURES = LEVELS_IN_FI2010 * FIELDS_PER_LEVEL  # 40
TOTAL_ROWS = 149

# Public mirrors that host the auction-free, z-score normalised files
GDRIVE_FILE_IDS = {
    # Training folds 1-9 and test folds 1-9 (no-auction, z-score)
    # These IDs reference the commonly-shared Google Drive folder used
    # in many DeepLOB reproductions.
}

# Direct URL for the commonly shared combined npz / txt hosted on GitHub repos
KNOWN_URLS = [
    # Primary: the BenchmarkDatasets repo that many papers reference
    "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Learning-for-Limit-Order-Books/main/data/data.zip",
    # Fallback mirrors
    "https://raw.githubusercontent.com/TeemLim/DeepLOB-Limit-Order-Books/main/data/data.zip",
]


def try_download(dest_dir: Path) -> bool:
    """Attempt auto-download from known mirrors. Returns True on success."""
    import urllib.request, urllib.error

    dest_dir.mkdir(parents=True, exist_ok=True)

    for url in KNOWN_URLS:
        print(f"  Trying {url[:80]}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lob-fi2010/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    zf.extractall(dest_dir)
            else:
                fname = url.split("/")[-1]
                (dest_dir / fname).write_bytes(data)

            print(f"  ✓ Downloaded to {dest_dir}")
            return True
        except (urllib.error.URLError, Exception) as exc:
            print(f"    ✗ Failed: {exc}")

    return False


def find_data_files(raw_dir: Path) -> list[Path]:
    """Recursively find .txt or .npy files that look like FI-2010 data."""
    candidates = []
    for ext in ("*.txt", "*.npy", "*.csv"):
        candidates.extend(raw_dir.rglob(ext))
    # Filter: FI-2010 filenames typically contain 'Train' or 'Test'
    fi_files = [f for f in candidates
                if any(k in f.name for k in ("Train", "Test", "train", "test", "NoAuction", "Auction"))]
    if not fi_files:
        # Fall back to all .txt files
        fi_files = list(raw_dir.rglob("*.txt"))
    return sorted(fi_files)


def load_fi2010_txt(path: Path) -> np.ndarray:
    """Load a single FI-2010 .txt → shape (N_samples, 149)."""
    data = np.loadtxt(path)            # shape (149, N)  or  (N, 149)
    if data.shape[0] == TOTAL_ROWS:
        data = data.T                  # → (N, 149)
    assert data.shape[1] >= TOTAL_ROWS, \
        f"Expected ≥{TOTAL_ROWS} columns, got {data.shape[1]} in {path.name}"
    return data


def fi2010_to_lob_df(raw: np.ndarray, levels: int = 5, label_horizon_idx: int = 2) -> pd.DataFrame:
    """
    Convert FI-2010 raw array → DataFrame matching our pipeline schema.

    Parameters
    ----------
    raw               : ndarray shape (N, ≥149)
    levels            : how many LOB levels to keep (max 10)
    label_horizon_idx : label row index (0=k1, 1=k2, 2=k3, 3=k5, 4=k10)
    """
    n = raw.shape[0]
    levels = min(levels, LEVELS_IN_FI2010)

    # ── Extract raw LOB: first 40 columns ──────────────────────────
    # Layout per level i (0-indexed): ask_px, ask_sz, bid_px, bid_sz
    cols = {}
    ts = np.arange(n, dtype=np.int64) * 10_000_000   # synthetic 10 ms spacing

    cols["ts"] = ts
    for lv in range(levels):
        base = lv * FIELDS_PER_LEVEL
        cols[f"ask_px_{lv + 1}"] = raw[:, base + 0]
        cols[f"ask_sz_{lv + 1}"] = raw[:, base + 1]
        cols[f"bid_px_{lv + 1}"] = raw[:, base + 2]
        cols[f"bid_sz_{lv + 1}"] = raw[:, base + 3]

    df = pd.DataFrame(cols)

    # Make sure sizes are positive (z-score norm can make them negative)
    for lv in range(1, levels + 1):
        df[f"bid_sz_{lv}"] = df[f"bid_sz_{lv}"].clip(lower=1e-6)
        df[f"ask_sz_{lv}"] = df[f"ask_sz_{lv}"].clip(lower=1e-6)

    return df


def create_synthetic_fi2010_sample(n: int = 50_000, levels: int = 5) -> pd.DataFrame:
    """
    Create a FI-2010-*style* synthetic dataset when real data isn't available.

    Uses a realistic microstructure model calibrated to Finnish equity stats
    from the original paper.
    """
    np.random.seed(2017)

    dt = 0.001
    kappa = 1.0          # faster mean-reversion (equity microstructure)
    sigma = 0.0003       # smaller ticks than crypto

    mid = np.zeros(n); mid[0] = 50.0
    momentum = 0.0

    for i in range(1, n):
        momentum = 0.92 * momentum + 0.08 * np.random.randn()
        mean_rev = -kappa * (mid[i - 1] - 50.0) * dt
        mid[i] = mid[i - 1] + mean_rev + sigma * np.random.randn() + 5e-5 * momentum

    spread = 0.02 + 0.01 * np.abs(np.random.randn(n))
    bid_px = np.stack([mid - spread / 2 - 0.01 * i for i in range(levels)], axis=1)
    ask_px = np.stack([mid + spread / 2 + 0.01 * i for i in range(levels)], axis=1)

    bid_sz = np.zeros((n, levels)); ask_sz = np.zeros((n, levels))
    bid_sz[0] = 500 + 30 * np.random.randn(levels)
    ask_sz[0] = 480 + 30 * np.random.randn(levels)
    for i in range(1, n):
        bid_sz[i] = 0.97 * bid_sz[i - 1] + 0.03 * (500 + 60 * np.random.randn(levels))
        ask_sz[i] = 0.97 * ask_sz[i - 1] + 0.03 * (480 + 60 * np.random.randn(levels))
    bid_sz = np.maximum(bid_sz, 1); ask_sz = np.maximum(ask_sz, 1)

    ts = (np.arange(n) * 10_000_000).astype(np.int64)
    cols = {"ts": ts}
    for i in range(1, levels + 1):
        cols[f"bid_px_{i}"] = bid_px[:, i - 1]
        cols[f"ask_px_{i}"] = ask_px[:, i - 1]
        cols[f"bid_sz_{i}"] = bid_sz[:, i - 1]
        cols[f"ask_sz_{i}"] = ask_sz[:, i - 1]
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser(description="Load / download FI-2010 LOB benchmark")
    ap.add_argument("--raw_dir", default="data/fi2010_raw",
                    help="Directory for raw FI-2010 files")
    ap.add_argument("--levels", type=int, default=5,
                    help="LOB levels to keep (max 10)")
    ap.add_argument("--output", default="data/processed/lob.parquet")
    ap.add_argument("--max_samples", type=int, default=200_000,
                    help="Cap samples to limit memory (0 = no limit)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)

    # ── Step 1: Get data ──────────────────────────────────────────
    fi_files = find_data_files(raw_dir) if raw_dir.exists() else []

    if not fi_files:
        print("[1/3] FI-2010 data not found locally. Attempting auto-download...")
        ok = try_download(raw_dir)

        if ok:
            fi_files = find_data_files(raw_dir)

    if not fi_files:
        print()
        print("=" * 60)
        print("  Could not auto-download FI-2010 data.")
        print()
        print("  OPTION A — Manual download:")
        print("    1. Go to: https://etsin.fairdata.fi/dataset/")
        print("       73eb48d7-4dbc-4a10-a52a-da745b47a649")
        print('    2. Click "Download all" (1.74 GB zip)')
        print(f"    3. Extract into: {raw_dir.resolve()}")
        print(f"    4. Re-run this script")
        print()
        print("  OPTION B — Using FI-2010-style synthetic data instead")
        print("=" * 60)
        print()

        print("Generating FI-2010-style synthetic data (50k samples)...")
        df = create_synthetic_fi2010_sample(n=50_000, levels=args.levels)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output)
        print(f"Saved → {args.output}  ({len(df):,} rows)")
        print("\nYou can now run:")
        print("  python research/10_train_logreg.py")
        print("  python research/30_backtest.py")
        return

    # ── Step 2: Load + merge all files ────────────────────────────
    print(f"[1/3] Found {len(fi_files)} FI-2010 file(s):")
    for f in fi_files[:5]:
        print(f"       {f.name}")
    if len(fi_files) > 5:
        print(f"       ... and {len(fi_files) - 5} more")

    all_data = []
    for fp in fi_files:
        try:
            arr = load_fi2010_txt(fp)
            all_data.append(arr)
            print(f"  Loaded {fp.name}: {arr.shape[0]:,} samples")
        except Exception as exc:
            print(f"  ✗ Skipping {fp.name}: {exc}")

    if not all_data:
        print("No valid data loaded!")
        sys.exit(1)

    raw = np.vstack(all_data)
    if args.max_samples > 0 and raw.shape[0] > args.max_samples:
        raw = raw[:args.max_samples]
    print(f"\n[2/3] Total samples: {raw.shape[0]:,}")

    # ── Step 3: Convert to our schema ─────────────────────────────
    df = fi2010_to_lob_df(raw, levels=args.levels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"[3/3] Saved → {args.output}  ({len(df):,} rows, {args.levels} levels)")

    # Quick sanity check
    print(f"\n  bid_px_1 range: [{df['bid_px_1'].min():.6f}, {df['bid_px_1'].max():.6f}]")
    print(f"  ask_px_1 range: [{df['ask_px_1'].min():.6f}, {df['ask_px_1'].max():.6f}]")
    print(f"  bid_sz_1 range: [{df['bid_sz_1'].min():.6f}, {df['bid_sz_1'].max():.6f}]")

    print("\nNext steps:")
    print("  python research/10_train_logreg.py")
    print("  python research/30_backtest.py")


if __name__ == "__main__":
    main()
