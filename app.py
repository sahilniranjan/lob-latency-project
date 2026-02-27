"""
LOB Latency Project — Streamlit Dashboard
==========================================
Interactive web dashboard for the Limit Order Book latency & prediction project.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo → done
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, sys, yaml
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Page config (must be first st call) ──────────────────────────
st.set_page_config(
    page_title="LOB Latency Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "outputs"
BACKTEST_DIR = OUT_DIR / "backtest"
CONFIG_PATH = ROOT / "configs" / "default.yaml"

# Add research/ to path for imports
sys.path.insert(0, str(ROOT / "research"))


# ── Helpers ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


@st.cache_data(ttl=60)
def load_lob_data():
    path = PROC_DIR / "lob.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=60)
def load_backtest_results():
    csv_path = BACKTEST_DIR / "results.csv"
    json_path = BACKTEST_DIR / "metrics.json"
    results, metrics = None, None
    if csv_path.exists():
        results = pd.read_csv(csv_path)
    if json_path.exists():
        metrics = json.loads(json_path.read_text())
    return results, metrics


@st.cache_data(ttl=60)
def list_raw_files():
    if RAW_DIR.exists():
        return sorted(RAW_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return []


# ── Sidebar navigation ───────────────────────────────────────────
st.sidebar.title("🏗️ LOB Latency Project")
st.sidebar.markdown("*Low-latency order book prediction*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Order Book Explorer", "🤖 Model & Features",
     "💰 Backtest Results", "🔴 Live Collector", "📐 Architecture"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Built by Sahil Niranjan • Free & Open Source")
st.sidebar.caption("Data: KuCoin REST API (free, no key)")


# ══════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 LOB Latency Project")
    st.markdown(
        "A **low-latency order-book prediction system** that combines "
        "machine learning with a sub-microsecond C++ inference engine. "
        "This dashboard lets you explore the data, model, and backtest results."
    )

    col1, col2, col3, col4 = st.columns(4)

    # Load data stats
    df = load_lob_data()
    _, metrics = load_backtest_results()
    cfg = load_config()

    with col1:
        n_ticks = f"{len(df):,}" if df is not None else "—"
        st.metric("L2 Snapshots", n_ticks)
    with col2:
        n_features = str(4 + cfg.get("levels", 5) + 3 * len(cfg.get("rolling_windows_ms", [])))
        st.metric("Features", n_features)
    with col3:
        pnl = f"${metrics['total_pnl']:,.2f}" if metrics else "—"
        st.metric("Backtest PnL", pnl)
    with col4:
        wr = f"{metrics['win_rate']:.1%}" if metrics else "—"
        st.metric("Win Rate", wr)

    st.divider()

    # Pipeline overview
    st.subheader("End-to-End Pipeline")
    st.markdown("""
    ```
    KuCoin REST API  →  L2 Snapshots  →  Feature Engine  →  Model  →  Backtest
         (free)          (parquet)        (18 features)    (logreg)    (PnL/Sharpe)
                                              ↓
                                     C++ Engine (<1 µs)
    ```
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔬 Research Stack (Python)")
        st.markdown("""
        - **Data collection**: KuCoin/Kraken REST API (free, no key)
        - **Feature engineering**: 18 microstructure features
        - **Model**: StandardScaler + Logistic Regression
        - **Backtest**: Tick-level execution simulator with costs
        """)
    with c2:
        st.subheader("⚡ Production Stack (C++20)")
        st.markdown("""
        - **Ring buffer**: Lock-free SPSC (128K depth)
        - **Feature engine**: O(1) rolling buffers
        - **Inference**: Baked scaler+weights, single dot product
        - **Target latency**: < 1 µs end-to-end
        """)

    st.divider()
    st.subheader("18 Features (canonical order)")
    feat_data = {
        "Feature": [
            "mid", "spread", "OFI", "microprice",
            *[f"QI level {i}" for i in range(1, 6)],
            *[f"rolling OFI ({w}ms)" for w in [10, 20, 50]],
            *[f"rolling spread ({w}ms)" for w in [10, 20, 50]],
            *[f"rolling QI ({w}ms)" for w in [10, 20, 50]],
        ],
        "Type": [
            "price", "price", "flow", "price",
            *["imbalance"] * 5,
            *["rolling"] * 9,
        ],
        "Description": [
            "(ask₁ + bid₁) / 2",
            "ask₁ − bid₁",
            "Δbid_sz₁ − Δask_sz₁",
            "size-weighted mid price",
            *[f"(bid_sz_{i} − ask_sz_{i}) / (bid_sz_{i} + ask_sz_{i})" for i in range(1, 6)],
            *[f"rolling mean of OFI over {w}ms" for w in [10, 20, 50]],
            *[f"rolling mean of spread over {w}ms" for w in [10, 20, 50]],
            *[f"rolling mean of top-level QI over {w}ms" for w in [10, 20, 50]],
        ],
    }
    st.dataframe(pd.DataFrame(feat_data), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: Order Book Explorer
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Order Book Explorer":
    st.title("📈 Order Book Explorer")

    df = load_lob_data()
    if df is None:
        st.warning("No data found. Collect data first (🔴 Live Collector page).")
        st.stop()

    st.success(f"Loaded **{len(df):,}** L2 snapshots")
    cfg = load_config()
    levels = cfg.get("levels", 5)

    # ── Mid price chart ───────────────────────────────────────────
    st.subheader("Mid Price Over Time")
    mid = (df["ask_px_1"].values + df["bid_px_1"].values) * 0.5
    spread = df["ask_px_1"].values - df["bid_px_1"].values

    chart_df = pd.DataFrame({
        "tick": range(len(mid)),
        "Mid Price ($)": mid,
    })
    st.line_chart(chart_df, x="tick", y="Mid Price ($)", width='stretch')

    # ── Spread chart ──────────────────────────────────────────────
    st.subheader("Bid-Ask Spread")
    spread_df = pd.DataFrame({
        "tick": range(len(spread)),
        "Spread ($)": spread,
    })
    st.line_chart(spread_df, x="tick", y="Spread ($)", width='stretch')

    # ── Order book snapshot ───────────────────────────────────────
    st.subheader("Order Book Snapshot")
    tick_idx = st.slider("Select tick", 0, len(df) - 1, len(df) // 2)
    row = df.iloc[tick_idx]

    col_bid, col_ask = st.columns(2)
    with col_bid:
        st.markdown("**🟢 Bids**")
        bid_data = []
        for i in range(1, levels + 1):
            bid_data.append({
                "Level": i,
                "Price": f"${row[f'bid_px_{i}']:,.2f}",
                "Size": f"{row[f'bid_sz_{i}']:,.4f}",
            })
        st.dataframe(pd.DataFrame(bid_data), width='stretch', hide_index=True)

    with col_ask:
        st.markdown("**🔴 Asks**")
        ask_data = []
        for i in range(1, levels + 1):
            ask_data.append({
                "Level": i,
                "Price": f"${row[f'ask_px_{i}']:,.2f}",
                "Size": f"{row[f'ask_sz_{i}']:,.4f}",
            })
        st.dataframe(pd.DataFrame(ask_data), width='stretch', hide_index=True)

    # ── Depth visualization ────────────────────────────────────
    st.subheader("Depth Chart (at selected tick)")
    bid_prices = [row[f"bid_px_{i}"] for i in range(1, levels + 1)]
    ask_prices = [row[f"ask_px_{i}"] for i in range(1, levels + 1)]
    bid_sizes = [row[f"bid_sz_{i}"] for i in range(1, levels + 1)]
    ask_sizes = [row[f"ask_sz_{i}"] for i in range(1, levels + 1)]

    bid_cum = np.cumsum(bid_sizes)
    ask_cum = np.cumsum(ask_sizes)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bid_prices[::-1], y=bid_cum[::-1],
        fill='tozeroy', name='Bids', line=dict(color='green'),
        fillcolor='rgba(0,200,0,0.2)',
    ))
    fig.add_trace(go.Scatter(
        x=ask_prices, y=ask_cum,
        fill='tozeroy', name='Asks', line=dict(color='red'),
        fillcolor='rgba(200,0,0,0.2)',
    ))
    fig.update_layout(
        xaxis_title="Price ($)", yaxis_title="Cumulative Size",
        height=350, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Queue imbalance heatmap ────────────────────────────────
    st.subheader("Queue Imbalance Heatmap (top 500 ticks)")
    window = min(500, len(df))
    start = max(0, tick_idx - window // 2)
    end = min(len(df), start + window)

    qi_matrix = np.zeros((end - start, levels))
    for i in range(1, levels + 1):
        bs = df[f"bid_sz_{i}"].values[start:end]
        az = df[f"ask_sz_{i}"].values[start:end]
        qi_matrix[:, i - 1] = (bs - az) / (bs + az + 1e-9)

    fig_hm = go.Figure(data=go.Heatmap(
        z=qi_matrix.T,
        x=list(range(start, end)),
        y=[f"Level {i}" for i in range(1, levels + 1)],
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        colorbar_title="QI",
    ))
    fig_hm.update_layout(
        xaxis_title="Tick", yaxis_title="Book Level",
        height=300, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_hm, width='stretch')

    # ── Raw data preview ──────────────────────────────────────────
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.head(100), width='stretch')


# ══════════════════════════════════════════════════════════════════
#  PAGE: Model & Features
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model & Features":
    st.title("🤖 Model & Features")

    df = load_lob_data()
    cfg = load_config()

    if df is None:
        st.warning("No data found. Collect data first.")
        st.stop()

    # Compute features
    from utils_data import make_features

    levels = cfg.get("levels", 5)
    rolling_ms = cfg.get("rolling_windows_ms", [10, 20, 50])
    horizon_ms = cfg.get("horizon_ms", 100)

    with st.spinner("Computing features..."):
        X, y = make_features(df, levels, rolling_ms, horizon_ms)

    st.success(f"Computed {X.shape[1]} features × {X.shape[0]:,} samples")

    feature_names = (
        ["mid", "spread", "OFI", "microprice"]
        + [f"QI_L{i}" for i in range(1, levels + 1)]
        + [f"roll_OFI_{w}" for w in rolling_ms]
        + [f"roll_spread_{w}" for w in rolling_ms]
        + [f"roll_QI_{w}" for w in rolling_ms]
    )

    # ── Feature distributions ─────────────────────────────────────
    st.subheader("Feature Distributions")
    feat_choice = st.selectbox("Select feature", feature_names)
    feat_idx = feature_names.index(feat_choice)

    fig_hist = px.histogram(
        x=X[:, feat_idx], nbins=80,
        labels={"x": feat_choice}, title=f"Distribution of {feat_choice}",
        color_discrete_sequence=["steelblue"],
    )
    fig_hist.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_hist, width='stretch')

    # ── Feature correlation matrix ────────────────────────────────
    st.subheader("Feature Correlation Matrix")
    corr = np.corrcoef(X.T)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr,
        x=feature_names,
        y=feature_names,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        colorbar_title="Correlation",
    ))
    fig_corr.update_layout(
        height=500, margin=dict(l=100, r=20, t=20, b=100),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_corr, width='stretch')

    # ── Label distribution ────────────────────────────────────────
    st.subheader("Label Distribution")
    unique, counts = np.unique(y, return_counts=True)
    label_map = {-1: "Down ↓", 0: "Flat →", 1: "Up ↑"}
    label_df = pd.DataFrame({
        "Label": [label_map.get(int(u), str(u)) for u in unique],
        "Count": counts,
        "Pct": [f"{c / len(y):.1%}" for c in counts],
    })
    st.dataframe(label_df, width='stretch', hide_index=True)

    # ── Model weights ─────────────────────────────────────────────
    st.subheader("Model Weights")
    model_path = ROOT / cfg.get("model_pkl", "outputs/logreg.pkl")
    if model_path.exists():
        from joblib import load as jl_load
        from sklearn.pipeline import Pipeline

        clf = jl_load(model_path)
        if isinstance(clf, Pipeline):
            logreg = clf.named_steps.get("logreg", clf[-1])
        else:
            logreg = clf

        classes = list(logreg.classes_)
        up_idx = classes.index(1) if 1 in classes else len(classes) - 1
        weights = logreg.coef_[up_idx] if logreg.coef_.ndim == 2 else logreg.coef_.ravel()

        w_df = pd.DataFrame({
            "Feature": feature_names[:len(weights)],
            "Weight (class +1)": weights,
            "Abs Weight": np.abs(weights),
        }).sort_values("Abs Weight", ascending=False)

        fig_w = px.bar(
            w_df, x="Feature", y="Weight (class +1)",
            color="Weight (class +1)",
            color_continuous_scale="RdYlGn",
            title="Logistic Regression Weights (class UP)",
        )
        fig_w.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=80), xaxis_tickangle=-45)
        st.plotly_chart(fig_w, width='stretch')
    else:
        st.info("No trained model found. Run the training pipeline first.")


# ══════════════════════════════════════════════════════════════════
#  PAGE: Backtest Results
# ══════════════════════════════════════════════════════════════════
elif page == "💰 Backtest Results":
    st.title("💰 Backtest Results")

    results, metrics = load_backtest_results()

    if results is None or metrics is None:
        st.warning("No backtest results found. Run the backtest first.")
        st.stop()

    # ── Key metrics ───────────────────────────────────────────────
    st.subheader("Performance Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PnL", f"${metrics['total_pnl']:,.2f}")
    c2.metric("Gross PnL", f"${metrics['gross_pnl']:,.2f}")
    c3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    c4.metric("Round Trips", metrics['n_round_trips'])
    c5.metric("Max Drawdown", f"${metrics['max_drawdown']:,.2f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Total Costs", f"${metrics['total_cost']:,.2f}")
    c7.metric("Avg PnL/Trade", f"${metrics['avg_pnl_per_trade']:,.4f}")
    c8.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────
    st.subheader("Equity Curve (Out-of-Sample)")

    eq = results["equity"].values
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        y=eq, mode='lines', name='Equity',
        line=dict(color='steelblue', width=1.5),
        fill='tozeroy', fillcolor='rgba(70,130,180,0.15)',
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig_eq.update_layout(
        yaxis_title="Cumulative PnL ($)", xaxis_title="Tick",
        height=400, margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_eq, width='stretch')

    # ── Drawdown ──────────────────────────────────────────────────
    st.subheader("Drawdown")
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        y=dd, mode='lines', name='Drawdown',
        fill='tozeroy', fillcolor='rgba(220,80,80,0.3)',
        line=dict(color='salmon', width=1),
    ))
    fig_dd.update_layout(
        yaxis_title="Drawdown ($)", xaxis_title="Tick",
        height=300, margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_dd, width='stretch')

    # ── Position over time ────────────────────────────────────────
    st.subheader("Position Over Time")
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        y=results["position"].values, mode='lines',
        line=dict(color='dimgray', width=0.8),
        name='Position',
    ))
    fig_pos.update_layout(
        yaxis_title="Position", xaxis_title="Tick",
        yaxis=dict(tickvals=[-1, 0, 1]),
        height=250, margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_pos, width='stretch')

    # ── Signal distribution ───────────────────────────────────────
    st.subheader("Signal Distribution")

    fig_sig = px.histogram(
        x=results["signal"].values, nbins=80,
        labels={"x": "P(up)"},
        color_discrete_sequence=["mediumorchid"],
    )
    cfg = load_config()
    upper = cfg.get("backtest_upper_thresh", 0.55)
    lower = cfg.get("backtest_lower_thresh", 0.45)
    fig_sig.add_vline(x=upper, line_dash="dash", line_color="green",
                      annotation_text="Long threshold")
    fig_sig.add_vline(x=lower, line_dash="dash", line_color="red",
                      annotation_text="Short threshold")
    fig_sig.update_layout(height=350, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_sig, width='stretch')

    # ── Interactive backtest ──────────────────────────────────────
    st.divider()
    st.subheader("🔧 Interactive Backtest")
    st.markdown("Adjust parameters and re-run the backtest in real time.")

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        fee_bps = st.slider("Fee (bps)", 0.0, 20.0, 10.0, 0.5)
    with ic2:
        upper_t = st.slider("Long threshold", 0.5, 0.99, 0.55, 0.01)
    with ic3:
        lower_t = st.slider("Short threshold", 0.01, 0.5, 0.45, 0.01)

    if st.button("▶ Run Backtest", type="primary"):
        from utils_backtest import ExecutionSimulator, compute_metrics

        df_data = load_lob_data()
        if df_data is not None:
            cfg = load_config()
            from utils_data import make_features
            X, y = make_features(
                df_data, cfg["levels"], cfg["rolling_windows_ms"], cfg["horizon_ms"]
            )
            n = len(X)
            cut = int(n * (1 - cfg["val_split_time"]))

            mid_all = (df_data["ask_px_1"].values + df_data["bid_px_1"].values) * 0.5
            step = len(df_data) - len(X)
            if step > 0:
                mid_all = mid_all[:-step]
            mid_val = mid_all[cut:]
            X_val = X[cut:]

            model_path = ROOT / cfg.get("model_pkl", "outputs/logreg.pkl")
            if model_path.exists():
                from joblib import load as jl_load
                clf = jl_load(model_path)
                proba = clf.predict_proba(X_val)
                classes = list(clf.classes_)
                up_idx = classes.index(1) if 1 in classes else len(classes) - 1
                signals = proba[:, up_idx]

                sim = ExecutionSimulator(maker_fee_bps=fee_bps, latency_ticks=1)
                res = sim.run(mid_val, signals, upper_thresh=upper_t, lower_thresh=lower_t)

                ts_data = df_data["ts"].values
                ts_range_ns = float(ts_data[-1] - ts_data[0])
                ts_range_days = ts_range_ns / (86_400 * 1e9) if ts_range_ns > 0 else 1.0
                tpd = len(df_data) / max(ts_range_days, 1e-9)

                m = compute_metrics(res, ticks_per_day=tpd)

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Total PnL", f"${m['total_pnl']:,.2f}")
                rc2.metric("Win Rate", f"{m['win_rate']:.1%}")
                rc3.metric("Round Trips", m['n_round_trips'])
                rc4.metric("Max Drawdown", f"${m['max_drawdown']:,.2f}")

                fig_eq2 = go.Figure()
                fig_eq2.add_trace(go.Scatter(
                    y=res["equity"].values, mode='lines',
                    line=dict(color='steelblue', width=1.5),
                    fill='tozeroy', fillcolor='rgba(70,130,180,0.15)',
                ))
                fig_eq2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_eq2.update_layout(
                    yaxis_title="Cumulative PnL ($)", xaxis_title="Tick",
                    height=350,
                )
                st.plotly_chart(fig_eq2, width='stretch')
            else:
                st.error("No trained model found.")
        else:
            st.error("No data loaded.")


# ══════════════════════════════════════════════════════════════════
#  PAGE: Live Collector
# ══════════════════════════════════════════════════════════════════
elif page == "🔴 Live Collector":
    st.title("🔴 Live Data Collector")
    st.markdown(
        "Collect real-time L2 order-book snapshots from **KuCoin** (free, no API key). "
        "Data is saved to `data/processed/lob.parquet` for training and backtesting."
    )

    # Show existing data files
    st.subheader("📁 Available Datasets")
    raw_files = list_raw_files()
    if raw_files:
        file_data = []
        for f in raw_files:
            size_kb = f.stat().st_size / 1024
            df_tmp = pd.read_parquet(f)
            file_data.append({
                "File": f.name,
                "Rows": f"{len(df_tmp):,}",
                "Size": f"{size_kb:.1f} KB",
                "Modified": f.stat().st_mtime,
            })
        st.dataframe(pd.DataFrame(file_data).drop(columns=["Modified"]),
                      width='stretch', hide_index=True)
    else:
        st.info("No data files yet. Collect some data below!")

    st.divider()

    # ── Collector controls ────────────────────────────────────────
    st.subheader("🎛️ Collect New Data")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])
    with cc2:
        duration = st.selectbox("Duration", [1, 2, 5, 10, 30], index=2)
        duration_label = f"{duration} min"
    with cc3:
        poll_hz = st.selectbox("Poll rate (Hz)", [1, 2, 3], index=2)

    if st.button("🔴 Start Collection", type="primary"):
        import urllib.request

        # Import collection function
        sys.path.insert(0, str(ROOT / "scripts"))
        from collect_binance_l2 import collect

        progress_bar = st.progress(0, text="Collecting...")
        status_text = st.empty()

        # Run collection (blocking but with progress)
        import time as _time

        records = []
        end_time = _time.time() + duration * 60
        interval = 1.0 / poll_hz
        total_expected = int(duration * 60 * poll_hz)
        errs = 0

        # Import fetch function
        from collect_binance_l2 import fetch_snapshot

        while _time.time() < end_time:
            t0 = _time.time()
            try:
                row = fetch_snapshot("kucoin", symbol, 5)
                row["ts"] = int(t0 * 1e9)
                records.append(row)
            except Exception as exc:
                errs += 1

            elapsed = _time.time() - t0
            _time.sleep(max(0, interval - elapsed))

            pct = min(len(records) / max(total_expected, 1), 1.0)
            progress_bar.progress(pct, text=f"Collected {len(records):,} snapshots...")

        progress_bar.progress(1.0, text="Done!")

        if records:
            df_new = pd.DataFrame(records)
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            Path("data/processed").mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d_%H%M")
            raw_path = f"data/raw/kucoin_{symbol}_{date_str}.parquet"
            df_new.to_parquet(raw_path)
            df_new.to_parquet("data/processed/lob.parquet")

            st.success(
                f"✅ Collected **{len(df_new):,}** snapshots ({errs} errors). "
                f"Saved to `{raw_path}`"
            )
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("No data collected!")

    # ── Quick train + backtest ────────────────────────────────────
    st.divider()
    st.subheader("⚡ Quick Train & Backtest")
    st.markdown("One-click: train model on current data, then run backtest.")

    if st.button("🚀 Train + Backtest", type="secondary"):
        df = load_lob_data()
        if df is None or len(df) < 100:
            st.error("Need at least 100 data points. Collect data first!")
        else:
            cfg = load_config()
            from utils_data import make_features
            from utils_backtest import ExecutionSimulator, compute_metrics
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline as SkPipeline
            from sklearn.preprocessing import StandardScaler
            from joblib import dump as jl_dump

            with st.spinner("Training model..."):
                X, y = make_features(df, cfg["levels"], cfg["rolling_windows_ms"], cfg["horizon_ms"])
                n = len(X)
                cut = int(n * (1 - cfg["val_split_time"]))
                Xtr, Xva = X[:cut], X[cut:]
                ytr, yva = y[:cut], y[cut:]

                model = SkPipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(C=cfg['C'], class_weight=cfg['class_weight'], max_iter=500)),
                ])
                model.fit(Xtr, ytr)
                Path("outputs").mkdir(exist_ok=True)
                jl_dump(model, cfg["model_pkl"])

            with st.spinner("Running backtest..."):
                proba = model.predict_proba(Xva)
                classes = list(model.classes_)
                up_idx = classes.index(1) if 1 in classes else len(classes) - 1
                signals = proba[:, up_idx]

                mid_all = (df["ask_px_1"].values + df["bid_px_1"].values) * 0.5
                step_trim = len(df) - len(X)
                if step_trim > 0:
                    mid_all = mid_all[:-step_trim]
                mid_val = mid_all[cut:]

                sim = ExecutionSimulator(maker_fee_bps=0, latency_ticks=1)
                res = sim.run(mid_val, signals, upper_thresh=0.90, lower_thresh=0.10)

                ts = df["ts"].values
                ts_range_ns = float(ts[-1] - ts[0])
                ts_range_days = ts_range_ns / (86_400 * 1e9) if ts_range_ns > 0 else 1.0
                tpd = len(df) / max(ts_range_days, 1e-9)

                m = compute_metrics(res, ticks_per_day=tpd)

                # Save results
                bt_dir = Path("outputs/backtest")
                bt_dir.mkdir(parents=True, exist_ok=True)
                res.to_csv(bt_dir / "results.csv", index=False)
                with open(bt_dir / "metrics.json", "w") as fh:
                    json.dump(m, fh, indent=2)

            st.success("✅ Training + backtest complete!")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Total PnL", f"${m['total_pnl']:,.2f}")
            rc2.metric("Win Rate", f"{m['win_rate']:.1%}")
            rc3.metric("Signals", m['total_signals'])
            rc4.metric("Max DD", f"${m['max_drawdown']:,.2f}")

            st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════
#  PAGE: Architecture
# ══════════════════════════════════════════════════════════════════
elif page == "📐 Architecture":
    st.title("📐 System Architecture")

    st.subheader("High-Level Pipeline")
    st.code("""
    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌─────────┐
    │  Exchange    │───▶│  L2 Snapshot │───▶│ Feature Engine │───▶│  Model  │
    │  (KuCoin)   │    │  Collector   │    │  (18 features) │    │ (LogReg)│
    └─────────────┘    └──────────────┘    └────────────────┘    └────┬────┘
                                                                      │
         ┌────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────┐    ┌──────────────────┐    ┌─────────────┐
    │  Signal │───▶│ Execution Sim    │───▶│  Performance│
    │ P(up)   │    │ (latency, costs) │    │  Metrics    │
    └─────────┘    └──────────────────┘    └─────────────┘
    """, language=None)

    st.subheader("C++ Engine (Production)")
    st.code("""
    ┌──────────────────────────────────────────────────────────────┐
    │                      C++ Engine (< 1 µs)                    │
    │                                                              │
    │  Producer Thread         SPSC Ring Buffer        Worker      │
    │  ┌───────────┐          ┌──────────────┐     ┌──────────┐  │
    │  │ Parse L2  │─push────▶│  Lock-free   │────▶│ Features │  │
    │  │ tick      │          │  128K slots   │     │ + Score  │  │
    │  └───────────┘          └──────────────┘     └──────────┘  │
    │                                                              │
    │  Memory: acquire/release ordering, no mutexes               │
    │  Rolling: O(1) circular buffers for 9 rolling features      │
    └──────────────────────────────────────────────────────────────┘
    """, language=None)

    st.subheader("Feature Computation")
    st.markdown("""
    | # | Feature | Computation | Complexity |
    |---|---------|-------------|------------|
    | 1 | Mid price | (ask₁ + bid₁) / 2 | O(1) |
    | 2 | Spread | ask₁ − bid₁ | O(1) |
    | 3 | OFI | Δbid_sz₁ − Δask_sz₁ | O(1) |
    | 4 | Microprice | size-weighted mid | O(1) |
    | 5-9 | QI levels 1-5 | (bid−ask)/(bid+ask) per level | O(L) |
    | 10-18 | Rolling features | Circular buffer mean | O(1) each |

    **Total: 18 features, all O(1) amortised**
    """)

    st.subheader("Ring Buffer Design")
    st.markdown("""
    - **Type**: Single-Producer Single-Consumer (SPSC)
    - **Capacity**: 131,072 slots (power of 2 for bitwise mask)
    - **Memory ordering**: `acquire` / `release` (no `seq_cst` overhead)
    - **Cache**: Head and tail on separate cache lines (no false sharing)
    - **Overflow policy**: Spin-wait (producer) / yield (consumer)
    """)

    st.subheader("Technology Stack")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Research (Python)**
        - Python 3.13
        - scikit-learn 1.8
        - pandas + pyarrow
        - numpy
        - Streamlit (this dashboard)
        - plotly (interactive charts)
        """)
    with c2:
        st.markdown("""
        **Production (C++)**
        - C++20, CMake 3.16+
        - MSVC `/O2` or GCC `-O3 -march=native -flto`
        - Lock-free SPSC ring buffer
        - `std::chrono::steady_clock` timing
        - No external dependencies
        """)

    st.subheader("Data Flow")
    st.markdown("""
    ```
    1. scripts/collect_binance_l2.py   →  data/raw/*.parquet       (L2 snapshots)
    2. research/10_train_logreg.py     →  outputs/logreg.pkl       (trained model)
    3. scripts/export_linear_weights.py →  engine/src/onnx/linear.txt  (C++ weights)
    4. research/30_backtest.py         →  outputs/backtest/*        (PnL, charts)
    5. app.py (this dashboard)         →  Web visualization
    ```
    """)
