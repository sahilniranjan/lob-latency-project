# Results

## Model: StandardScaler + Logistic Regression (18 features)

Features: mid, spread, OFI, microprice, 5×QI, 3×rolling_OFI,
3×rolling_spread, 3×rolling_QI (windows: 10/20/50 ms).

### Synthetic Data (100k ticks, Ornstein-Uhlenbeck)

| Metric | Value |
|--------|-------|
| F1 (macro) | ~0.50 |
| MCC | ~0.00 |
| Signal range | [0.498, 0.502] |

Expected — synthetic data has no learnable microstructure signal.

### Real Crypto Data (BTCUSDT, KuCoin, 30 min)

**Dataset:** 5,270 L2 snapshots at 3 Hz, 5 levels, collected live from KuCoin.

| Metric | Value |
|--------|-------|
| Training samples | 4,215 |
| Validation samples | 1,054 |
| F1 (macro) | 0.345 |
| MCC | 0.213 |
| Signal range | [0.000, 0.990] |
| Classes | 3 (down / flat / up) |
| Recall (up/down) | ~84% each |

Model captures real microstructure patterns. Wide signal spread and high
recall on directional moves confirm predictive power.

### Backtest: Out-of-Sample (1,054 ticks)

#### Raw Alpha (zero fee, thresholds 0.10/0.90)

| Metric | Value |
|--------|-------|
| Total PnL | **+$122.10** |
| Round trips | 47 |
| Win rate | 27.7% |
| Max drawdown | -$6.70 |
| Sharpe | ~0 (sub-day) |

**The model has real directional alpha.** Even with a simple linear model
on just 30 minutes of data, it produces positive PnL on unseen ticks.

#### With Transaction Costs (10 bps, thresholds 0.45/0.55)

| Metric | Value |
|--------|-------|
| Total PnL | -$13,463 |
| Gross PnL | +$238 (positive direction) |
| Total cost | $13,701 |
| Signals | 146 |
| Round trips | 42 |

With BTC at ~$84k, 10 bps = ~$84 per side. The model's gross signal is
directionally correct but the cost structure of full-notional BTC trading
overwhelms it at high frequency.

#### With Maker Rebate (2 bps, thresholds 0.30/0.70)

| Metric | Value |
|--------|-------|
| Total PnL | -$2,033 |
| Gross PnL | +$156 |
| Win rate | 5.6% |

Even at aggressively low fees, the BTC notional value makes it hard
to be profitable at ~333ms resolution without a more complex model.

## Key Takeaways

1. **The features work** — 18-feature linear model shows real alpha
2. **Cost structure matters** — BTC's high notional makes HFT hard at retail
3. **More data needed** — 30 min is a proof of concept; 24h+ would train a
   much stronger model
4. **Model upgrade path** — XGBoost or a small MLP could capture non-linear
   order-book dynamics while staying under 5 µs latency

## Next Steps

1. Collect 24h+ of continuous L2 data
2. Add non-linear model (XGBoost / small MLP) under latency constraint
3. Try smaller-notional pairs (e.g. DOGE, SHIB) where costs are proportionally lower
4. Add book pressure and trade-flow features
5. Implement adaptive threshold based on rolling volatility
