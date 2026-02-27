"""
Collect live L2 order-book snapshots from public crypto exchange REST APIs.

Supported exchanges (all free, no API key required):
  - Bybit    (default, works globally)
  - Binance  (blocked in some regions)
  - KuCoin   (backup)

Usage
-----
    python scripts/collect_binance_l2.py                         # 30 min BTCUSDT via Bybit
    python scripts/collect_binance_l2.py --symbol BTCUSDT --duration_min 60
    python scripts/collect_binance_l2.py --exchange binance       # use Binance

Output
------
    data/raw/{exchange}_{symbol}_{datetime}.parquet   — raw snapshots
    data/processed/lob.parquet                        — ready for training / backtest
"""

import argparse, json, time, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np, pandas as pd

# ─── Exchange endpoints ───────────────────────────────────────────
EXCHANGES = {
    "kucoin": {
        "url": "https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={symbol}",
        "parse": lambda data, L: _parse_kucoin(data, L),
    },
    "kraken": {
        "url": "https://api.kraken.com/0/public/Depth?pair={symbol}&count={limit}",
        "parse": lambda data, L: _parse_kraken(data, L),
    },
    "bybit": {
        "url": "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={limit}",
        "parse": lambda data, L: _parse_bybit(data, L),
    },
    "binance": {
        "url": "https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}",
        "parse": lambda data, L: _parse_binance(data, L),
    },
}

# Symbol mapping per exchange
def _kucoin_sym(s: str) -> str:
    if "-" in s:
        return s
    # BTCUSDT → BTC-USDT
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote):
            return s[:-len(quote)] + "-" + quote
    return s

SYMBOL_MAP = {
    "kucoin":  _kucoin_sym,
    "kraken":  lambda s: s.replace("BTC", "XBT") if s.startswith("BTC") else s,
    "bybit":   lambda s: s,
    "binance": lambda s: s,
}


def _parse_kucoin(data: dict, L: int) -> dict:
    """Parse KuCoin level2_20 response."""
    book = data["data"]
    row = {}
    for i, (bid, ask) in enumerate(zip(book["bids"][:L], book["asks"][:L])):
        row[f"bid_px_{i + 1}"] = float(bid[0])
        row[f"bid_sz_{i + 1}"] = float(bid[1])
        row[f"ask_px_{i + 1}"] = float(ask[0])
        row[f"ask_sz_{i + 1}"] = float(ask[1])
    return row


def _parse_kraken(data: dict, L: int) -> dict:
    """Parse Kraken Depth response."""
    pair = list(data["result"].keys())[0]
    book = data["result"][pair]
    row = {}
    for i, (bid, ask) in enumerate(zip(book["bids"][:L], book["asks"][:L])):
        row[f"bid_px_{i + 1}"] = float(bid[0])
        row[f"bid_sz_{i + 1}"] = float(bid[1])
        row[f"ask_px_{i + 1}"] = float(ask[0])
        row[f"ask_sz_{i + 1}"] = float(ask[1])
    return row


def _parse_bybit(data: dict, L: int) -> dict:
    """Parse Bybit v5 orderbook response."""
    book = data["result"]
    bids = book["b"][:L]  # [[price, size], ...]
    asks = book["a"][:L]
    row = {}
    for i, (bid, ask) in enumerate(zip(bids, asks)):
        row[f"bid_px_{i + 1}"] = float(bid[0])
        row[f"bid_sz_{i + 1}"] = float(bid[1])
        row[f"ask_px_{i + 1}"] = float(ask[0])
        row[f"ask_sz_{i + 1}"] = float(ask[1])
    return row


def _parse_binance(data: dict, L: int) -> dict:
    """Parse Binance depth response."""
    row = {}
    for i, (bid, ask) in enumerate(zip(data["bids"][:L], data["asks"][:L])):
        row[f"bid_px_{i + 1}"] = float(bid[0])
        row[f"bid_sz_{i + 1}"] = float(bid[1])
        row[f"ask_px_{i + 1}"] = float(ask[0])
        row[f"ask_sz_{i + 1}"] = float(ask[1])
    return row


def _parse_kucoin(data: dict, L: int) -> dict:
    """Parse KuCoin level2_20 response."""
    book = data["data"]
    row = {}
    for i, (bid, ask) in enumerate(zip(book["bids"][:L], book["asks"][:L])):
        row[f"bid_px_{i + 1}"] = float(bid[0])
        row[f"bid_sz_{i + 1}"] = float(bid[1])
        row[f"ask_px_{i + 1}"] = float(ask[0])
        row[f"ask_sz_{i + 1}"] = float(ask[1])
    return row


def fetch_snapshot(exchange: str, symbol: str, levels: int) -> dict:
    """Single REST call — returns parsed row dict."""
    cfg = EXCHANGES[exchange]
    sym = SYMBOL_MAP[exchange](symbol)
    url = cfg["url"].format(symbol=sym, limit=levels)
    req = urllib.request.Request(url, headers={"User-Agent": "lob-collector/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    return cfg["parse"](data, levels)


def collect(exchange: str, symbol: str, levels: int, duration_min: float, poll_hz: float = 3.0):
    """
    Poll Binance depth endpoint and return a DataFrame.

    Parameters
    ----------
    exchange     : "bybit", "binance", or "kucoin"
    symbol       : e.g. "BTCUSDT"
    levels       : book depth (max 20 without extra weight)
    duration_min : collection window in minutes
    poll_hz      : requests per second (keep ≤ 4 to stay under weight limits)
    """
    records: list[dict] = []
    end_time = time.time() + duration_min * 60
    interval = 1.0 / poll_hz
    errs = 0

    print(f"[{exchange.upper()}] Collecting {symbol} L2 depth (top {levels} levels) "
          f"for {duration_min} min at {poll_hz} Hz …")

    while time.time() < end_time:
        t0 = time.time()
        try:
            row = fetch_snapshot(exchange, symbol, levels)
            row["ts"] = int(t0 * 1e9)
            records.append(row)
        except Exception as exc:
            errs += 1
            if errs <= 5:
                print(f"  ⚠ fetch error ({errs}): {exc}")

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

        if len(records) % 200 == 0 and len(records) > 0:
            elapsed_min = (time.time() - (end_time - duration_min * 60)) / 60
            print(f"  {len(records):>6,} snapshots  ({elapsed_min:.1f} min elapsed)")

    print(f"Collection done — {len(records):,} snapshots, {errs} errors.")
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description="Collect crypto L2 order-book data (free, no API key)")
    ap.add_argument("--exchange", default="kucoin",
                    choices=list(EXCHANGES.keys()),
                    help="Exchange to use (default: kucoin — works globally)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--duration_min", type=float, default=30,
                    help="How long to collect in minutes (default 30)")
    ap.add_argument("--poll_hz", type=float, default=3.0,
                    help="Requests per second (default 3, keep ≤ 4)")
    args = ap.parse_args()

    df = collect(args.exchange, args.symbol, args.levels, args.duration_min, args.poll_hz)

    if len(df) == 0:
        print("No data collected!")
        return

    # ── Save raw snapshot file ────────────────────────────────────
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    raw_path = f"data/raw/{args.exchange}_{args.symbol}_{date_str}.parquet"
    df.to_parquet(raw_path)
    print(f"Saved raw data  → {raw_path}  ({len(df):,} rows)")

    # ── Also write the standard processed path ────────────────────
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/lob.parquet")
    print(f"Saved processed → data/processed/lob.parquet")

    print("\nNext steps:")
    print("  1) python research/10_train_logreg.py   # train on this data")
    print("  2) python research/30_backtest.py        # run backtest")


if __name__ == "__main__":
    main()
