"""
Backtest utilities — execution simulator and performance metrics.

All dollar amounts are in the same unit as mid-price (e.g. USD for BTC/USD).
Position is normalised: {-1, 0, +1} representing 1 unit of the base asset.
"""

import numpy as np, pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  Execution Simulator
# ═══════════════════════════════════════════════════════════════════
class ExecutionSimulator:
    """
    Tick-level execution simulator for an L2 order-book strategy.

    Assumptions
    -----------
    * Signal generated at tick *T* is acted on at tick *T + latency_ticks*.
    * Transaction cost = maker_fee_bps / 10 000 × |Δposition| × mid_price.
    * Position ∈ {-1, 0, +1}  (unit-normalised).
    """

    def __init__(self, maker_fee_bps: float = 10.0, latency_ticks: int = 1):
        self.maker_fee_bps = maker_fee_bps
        self.latency_ticks = latency_ticks

    def run(
        self,
        mid: np.ndarray,
        signals: np.ndarray,
        upper_thresh: float = 0.55,
        lower_thresh: float = 0.45,
    ) -> pd.DataFrame:
        """
        Run the simulation.

        Parameters
        ----------
        mid           : mid-price array of length N
        signals       : model probability-of-up-move, length N
        upper_thresh  : go long  when signal > upper_thresh
        lower_thresh  : go short when signal < lower_thresh

        Returns
        -------
        DataFrame with columns:
            mid, signal, position, pnl_gross, cost, pnl_net, equity
        """
        n = len(mid)
        position = np.zeros(n)

        # Target position from raw signal
        target = np.zeros(n)
        target[signals > upper_thresh] = 1.0
        target[signals < lower_thresh] = -1.0

        # Apply execution latency
        for t in range(self.latency_ticks, n):
            position[t] = target[t - self.latency_ticks]

        # Mark-to-market PnL: position[t-1] × (mid[t] − mid[t-1])
        mid_diff = np.diff(mid, prepend=mid[0])
        pnl_gross = np.roll(position, 1) * mid_diff
        pnl_gross[0] = 0.0

        # Costs when position changes
        pos_change = np.abs(np.diff(position, prepend=0))
        cost = pos_change * mid * (self.maker_fee_bps / 1e4)

        pnl_net = pnl_gross - cost
        equity = np.cumsum(pnl_net)

        return pd.DataFrame(
            {
                "mid": mid,
                "signal": signals,
                "position": position,
                "pnl_gross": pnl_gross,
                "cost": cost,
                "pnl_net": pnl_net,
                "equity": equity,
            }
        )


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(results: pd.DataFrame, ticks_per_day: float = 8_640_000.0) -> dict:
    """
    Compute strategy performance metrics from simulation results.

    Parameters
    ----------
    results        : DataFrame returned by ExecutionSimulator.run()
    ticks_per_day  : estimated ticks in one trading day (for annualisation)
    """
    net = results["pnl_net"].values
    equity = results["equity"].values
    position = results["position"].values

    total_pnl = equity[-1] if len(equity) else 0.0

    # ── Trade segmentation ────────────────────────────────────────
    pos_changes = np.diff(position, prepend=0)
    n_signals = int(np.sum(pos_changes != 0))

    trade_pnls: list[float] = []
    current_pnl = 0.0
    in_trade = False
    for t in range(len(position)):
        if position[t] != 0:
            current_pnl += net[t]
            in_trade = True
        elif in_trade:
            trade_pnls.append(current_pnl)
            current_pnl = 0.0
            in_trade = False
    if in_trade:
        trade_pnls.append(current_pnl)

    trade_pnls_arr = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    wins = int(np.sum(trade_pnls_arr > 0))
    win_rate = wins / max(len(trade_pnls_arr), 1)

    # ── Sharpe (annualised from daily buckets) ────────────────────
    daily_ticks = int(ticks_per_day)
    n_days = max(1, len(net) // daily_ticks)
    daily_pnl = np.array(
        [net[i * daily_ticks : (i + 1) * daily_ticks].sum() for i in range(n_days)]
    )
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = float((np.mean(daily_pnl) / np.std(daily_pnl)) * np.sqrt(252))
    else:
        sharpe = 0.0

    # ── Max drawdown ──────────────────────────────────────────────
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0

    total_cost = float(results["cost"].sum())
    gross_pnl = float(results["pnl_gross"].sum())

    return {
        "total_pnl": round(total_pnl, 6),
        "gross_pnl": round(gross_pnl, 6),
        "total_cost": round(total_cost, 6),
        "cost_pnl_ratio": round(total_cost / max(abs(total_pnl), 1e-9), 4),
        "total_signals": n_signals,
        "n_round_trips": len(trade_pnls_arr),
        "win_rate": round(win_rate, 4),
        "avg_pnl_per_trade": round(float(np.mean(trade_pnls_arr)), 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 6),
    }
