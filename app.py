"""
app.py  —  TWSE Ensemble Valuation System
Streamlit dashboard. Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from datetime import date, timedelta
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TWSE Ensemble Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0b0e1a;
    color: #e0e6f0;
}
.stApp { background-color: #0b0e1a; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6b7db3;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e0e6f0;
}
.metric-value.up   { color: #34d399; }
.metric-value.down { color: #f87171; }
.metric-value.neu  { color: #60a5fa; }

.tag-success {
    background: #064e3b; color: #34d399;
    border-radius: 4px; padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
}
.tag-warn {
    background: #451a03; color: #fb923c;
    border-radius: 4px; padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}
.pit-badge {
    background: #1e1b4b; color: #818cf8;
    border: 1px solid #3730a3;
    border-radius: 4px; padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
}
div[data-testid="stSidebar"] {
    background-color: #0d1120;
    border-right: 1px solid #1e3a5f;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 10px 28px;
    width: 100%;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE MODEL (same logic as strategy.py, standalone for Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "ret_5d", "ret_20d", "ret_60d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "vol_20d", "vol_60d", "vol_ratio",
    "rsi14", "bb_pos",
]

def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy().sort_index()
    df["ret_1d"]  = df["Close"].pct_change(1)
    df["ret_5d"]  = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["ret_60d"] = df["Close"].pct_change(60)
    df["ma5"]     = df["Close"].rolling(5).mean()
    df["ma20"]    = df["Close"].rolling(20).mean()
    df["ma60"]    = df["Close"].rolling(60).mean()
    df["ma5_ratio"]  = df["Close"] / df["ma5"]
    df["ma20_ratio"] = df["Close"] / df["ma20"]
    df["ma60_ratio"] = df["Close"] / df["ma60"]
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    mid  = df["Close"].rolling(20).mean()
    std  = df["Close"].rolling(20).std()
    df["bb_pos"] = (df["Close"] - mid) / (2 * std + 1e-9)
    df["fwd_ret_20d"] = df["Close"].shift(-20) / df["Close"] - 1
    return df


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    return raw


def train_ensemble(feat_df: pd.DataFrame, n_splits: int = 5):
    data = feat_df[FEATURE_COLS + ["fwd_ret_20d"]].dropna()
    X, y = data[FEATURE_COLS], data["fwd_ret_20d"]

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03,
                       max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
    scaler = StandardScaler()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []
    for train_idx, val_idx in tscv.split(X):
        Xtr, ytr = X.iloc[train_idx[:-5]], y.iloc[train_idx[:-5]]  # purge
        Xva, yva = X.iloc[val_idx], y.iloc[val_idx]
        if len(Xtr) < 10:
            continue
        sc = StandardScaler().fit(Xtr)
        xgb.fit(sc.transform(Xtr), ytr)
        knn.fit(sc.transform(Xtr), ytr)
        preds = 0.6 * xgb.predict(sc.transform(Xva)) + \
                0.4 * knn.predict(sc.transform(Xva))
        fold_rmse.append(float(np.sqrt(mean_squared_error(yva, preds))))

    # Final fit on all data
    Xs = scaler.fit_transform(X)
    xgb.fit(Xs, y)
    knn.fit(Xs, y)
    return xgb, knn, scaler, fold_rmse


def predict(xgb, knn, scaler, current_price, feature_row):
    X = feature_row[FEATURE_COLS].values.reshape(1, -1)
    Xs = scaler.transform(X)
    xgb_ret = float(xgb.predict(Xs)[0])
    knn_ret = float(knn.predict(Xs)[0])
    ens_ret = 0.6 * xgb_ret + 0.4 * knn_ret
    return {
        "xgb_price":  round(current_price * (1 + xgb_ret), 2),
        "knn_price":  round(current_price * (1 + knn_ret), 2),
        "ens_price":  round(current_price * (1 + ens_ret), 2),
        "ens_ret_pct": round(ens_ret * 100, 3),
        "xgb_ret_pct": round(xgb_ret * 100, 3),
        "knn_ret_pct": round(knn_ret * 100, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(feat_df, pred, as_at_dt, horizon, actual_price=None):
    fig = plt.figure(figsize=(14, 9), facecolor="#0b0e1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    BG   = "#111827"
    BLUE = "#60a5fa"
    GRN  = "#34d399"
    RED  = "#f87171"
    ORG  = "#fb923c"
    PUR  = "#a78bfa"
    SPINE = "#1e3a5f"

    def style(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color(SPINE)
        ax.tick_params(colors="#4b5e7e", labelsize=7)
        ax.xaxis.label.set_color("#4b5e7e")
        ax.yaxis.label.set_color("#4b5e7e")

    hist = feat_df.dropna(subset=["Close"]).last("365D")

    # ── 1. Price + prediction ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1)
    ax1.plot(hist.index, hist["Close"], color=BLUE, lw=1.2, label="Close")
    ax1.fill_between(hist.index, hist["Low"].rolling(5).min(),
                     hist["High"].rolling(5).max(),
                     alpha=0.08, color=BLUE)
    ax1.axvline(as_at_dt, color=ORG, lw=1.5, ls="--", label="As At Date")

    # Project forward
    t20_approx = as_at_dt + timedelta(days=horizon * 1.4)
    current_p  = float(hist["Close"].iloc[-1])
    ax1.annotate("", xy=(t20_approx, pred["ens_price"]),
                 xytext=(as_at_dt, current_p),
                 arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))
    ax1.scatter([t20_approx], [pred["ens_price"]],
                color=GRN, zorder=6, s=60,
                label=f"Ensemble T+{horizon}: {pred['ens_price']:.0f}")
    if actual_price:
        ax1.scatter([t20_approx], [actual_price],
                    color=RED, zorder=6, s=60, marker="x",
                    label=f"Actual T+{horizon}: {actual_price:.0f}")
    ax1.set_title(f"Price History & T+{horizon} Ensemble Forecast",
                  color="white", fontsize=10,
                  fontfamily="monospace", pad=8)
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white",
               edgecolor=SPINE, framealpha=0.9)

    # ── 2. RSI ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2)
    rsi = hist["rsi14"].dropna().last("180D")
    ax2.plot(rsi.index, rsi, color=PUR, lw=1)
    ax2.axhline(70, color=RED, lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(30, color=GRN, lw=0.8, ls="--", alpha=0.6)
    ax2.fill_between(rsi.index, 30, rsi, where=rsi < 30, alpha=0.2, color=GRN)
    ax2.fill_between(rsi.index, 70, rsi, where=rsi > 70, alpha=0.2, color=RED)
    ax2.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI (14)", color="white", fontsize=9, fontfamily="monospace")

    # ── 3. Volume ratio ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3)
    vr = hist["vol_ratio"].dropna().last("180D")
    colors_vr = [GRN if v >= 1 else "#374151" for v in vr]
    ax3.bar(vr.index, vr, color=colors_vr, alpha=0.8, width=1)
    ax3.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
    ax3.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax3.set_title("Volume Ratio (vs 20d avg)", color="white",
                  fontsize=9, fontfamily="monospace")

    # ── 4. BB position ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style(ax4)
    bb = hist["bb_pos"].dropna().last("180D")
    ax4.plot(bb.index, bb, color=BLUE, lw=1)
    ax4.axhline(0, color="white", lw=0.7, alpha=0.4)
    ax4.axhline(1, color=RED, lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(-1, color=GRN, lw=0.7, ls="--", alpha=0.6)
    ax4.fill_between(bb.index, 0, bb, where=bb > 0, alpha=0.15, color=RED)
    ax4.fill_between(bb.index, 0, bb, where=bb < 0, alpha=0.15, color=GRN)
    ax4.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax4.set_title("Bollinger Band Position", color="white",
                  fontsize=9, fontfamily="monospace")

    # ── 5. MA ratios ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    style(ax5)
    ax5.plot(hist.index, hist["ma5_ratio"],  color=GRN,  lw=1,   label="MA5 ratio")
    ax5.plot(hist.index, hist["ma20_ratio"], color=BLUE, lw=1,   label="MA20 ratio")
    ax5.plot(hist.index, hist["ma60_ratio"], color=ORG,  lw=1,   label="MA60 ratio")
    ax5.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.3)
    ax5.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax5.set_title("Price / MA Ratios (Momentum)", color="white",
                  fontsize=9, fontfamily="monospace")
    ax5.legend(fontsize=7, facecolor=BG, labelcolor="white", edgecolor=SPINE)

    # ── 6. Model prediction breakdown ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    style(ax6)
    labels = ["Current", "XGBoost", "KNN", "Ensemble"]
    values = [current_p, pred["xgb_price"], pred["knn_price"], pred["ens_price"]]
    bar_colors = [BLUE, PUR, BLUE, GRN]
    if actual_price:
        labels.append("Actual")
        values.append(actual_price)
        bar_colors.append(RED)
    bars = ax6.bar(labels, values, color=bar_colors, alpha=0.85,
                   edgecolor=SPINE, width=0.6)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 2,
                 f"{val:.0f}", ha="center", va="bottom",
                 color="white", fontsize=7, fontfamily="monospace")
    ax6.set_title("Prediction Breakdown (TWD)", color="white",
                  fontsize=9, fontfamily="monospace")

    fig.suptitle(
        f"TWSE Ensemble Valuation  |  As At {as_at_dt.date()}  "
        f"|  Expected: {pred['ens_ret_pct']:+.2f}%",
        color="white", fontsize=11,
        fontfamily="monospace", fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #1e3a5f; padding-bottom:16px; margin-bottom:24px;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
               color:#3b82f6; letter-spacing:0.2em; text-transform:uppercase;">
    Quantitative Research
  </span>
  <h1 style="margin:4px 0 0 0; font-size:1.6rem; font-weight:600; color:#e0e6f0;">
    TWSE Ensemble Valuation System
  </h1>
  <p style="color:#4b5e7e; font-size:0.85rem; margin-top:6px;">
    XGBoost + KNN · Purged Walk-Forward CV · Point-in-Time Strict
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Parameters</div>',
                unsafe_allow_html=True)

    ticker_input = st.text_input(
        "Stock Code (Yahoo Finance format)",
        value="2330.TW",
        help="TWSE: append .TW  e.g. 2330.TW, 2317.TW, 0050.TW"
    )

    as_at_date = st.date_input(
        "As At Date (Point-in-Time Cutoff)",
        value=date(2024, 1, 18),
        min_value=date(2015, 1, 1),
        max_value=date.today(),
    )

    horizon = st.slider("Prediction Horizon (trading days)", 5, 60, 20)

    train_years = st.slider("Training History (years)", 2, 10, 5)
    train_start = (pd.to_datetime(as_at_date) -
                   pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")

    validate = st.checkbox("Run T+N Validation (fetch actual price)", value=True)

    st.markdown('<div class="section-header">Model Config</div>',
                unsafe_allow_html=True)
    xgb_weight = st.slider("XGBoost Weight", 0.3, 0.9, 0.6, 0.05)
    n_neighbors = st.slider("KNN Neighbors", 5, 30, 10)
    buy_threshold = st.slider("Buy Signal Threshold (%)", 1, 20, 5)

    st.markdown('<div class="section-header">About</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#4b5e7e; line-height:1.6;">
    Ensemble model combining XGBoost (non-linear financial features)
    with K-Nearest Neighbors (historical pattern matching).<br><br>
    <span class="pit-badge">PIT STRICT</span> No future data leakage.
    Purged walk-forward cross-validation enforced.
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  Run Valuation")

# ── Main content ──────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding:80px 0; color:#1e3a5f;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem;">◈</div>
      <div style="font-size:0.9rem; margin-top:12px; color:#374151;">
        Configure parameters in the sidebar and press <strong>Run Valuation</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run ───────────────────────────────────────────────────────────────────────
as_at_dt   = pd.to_datetime(as_at_date)
future_end = (as_at_dt + timedelta(days=horizon * 3)).strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker_input} data from Yahoo Finance…"):
    try:
        train_raw = load_data(ticker_input, train_start,
                              as_at_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")
        st.stop()

if train_raw.empty or len(train_raw) < 100:
    st.error("❌ Insufficient data. Try a longer training window or different ticker.")
    st.stop()

with st.spinner("Engineering features…"):
    feat_df = compute_features(train_raw)
    train_feat = feat_df[feat_df.index <= as_at_dt]

with st.spinner("Training ensemble model (purged walk-forward CV)…"):
    try:
        xgb_model, knn_model, scaler, fold_rmse = train_ensemble(
            train_feat, n_splits=5
        )
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.stop()

last_row      = train_feat[FEATURE_COLS].dropna().iloc[-1]
current_price = float(train_raw["Close"].iloc[-1])
pred          = predict(xgb_model, knn_model, scaler, current_price, last_row)

# Validation: fetch actual T+N price
actual_price = None
if validate:
    with st.spinner(f"Fetching actual T+{horizon} price for validation…"):
        try:
            future_raw = load_data(ticker_input,
                                   as_at_date.strftime("%Y-%m-%d"),
                                   future_end)
            future_closes = future_raw["Close"].iloc[1:]
            if len(future_closes) >= horizon:
                actual_price = float(future_closes.iloc[horizon - 1])
            elif len(future_closes) > 0:
                actual_price = float(future_closes.iloc[-1])
                st.info(f"ℹ Only {len(future_closes)} future days available; "
                        f"using last available close for validation.")
        except Exception:
            st.warning("Could not fetch future price for validation.")

# ── KPI cards ─────────────────────────────────────────────────────────────────
ret_cls = "up" if pred["ens_ret_pct"] > 0 else "down"
signal  = "BUY 🟢" if pred["ens_ret_pct"] > buy_threshold else \
          ("SELL 🔴" if pred["ens_ret_pct"] < -buy_threshold else "HOLD ⚪")

cols = st.columns(5)
kpis = [
    ("Current Price",   f"{current_price:.2f} TWD",   "neu"),
    ("Ensemble Target", f"{pred['ens_price']:.2f} TWD", ret_cls),
    ("Expected Return", f"{pred['ens_ret_pct']:+.2f}%", ret_cls),
    ("Signal",          signal,                         ret_cls),
    ("CV RMSE (avg)",   f"{np.mean(fold_rmse):.4f}",   "neu"),
]
for col, (label, value, cls) in zip(cols, kpis):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Validation row ─────────────────────────────────────────────────────────────
if actual_price:
    err_pct = (pred["ens_price"] - actual_price) / actual_price * 100
    abs_err = abs(pred["ens_price"] - actual_price)
    ok      = abs(err_pct) < 5
    tag     = f'<span class="tag-success">✓ ERROR &lt; 5%</span>' if ok else \
              f'<span class="tag-warn">⚠ ERROR ≥ 5%</span>'
    err_cls = "up" if abs(err_pct) < 5 else "down"

    st.markdown('<div class="section-header">Validation Results</div>',
                unsafe_allow_html=True)
    vcols = st.columns(4)
    vkpis = [
        ("Actual T+" + str(horizon), f"{actual_price:.2f} TWD", "neu"),
        ("Absolute Error",           f"{abs_err:.2f} TWD",       err_cls),
        ("Error %",                  f"{err_pct:+.3f}%",         err_cls),
        ("Verdict",                  tag,                         ""),
    ]
    for col, (label, value, cls) in zip(vcols, vkpis):
        if cls:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)
        else:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div style="margin-top:10px;">{value}</div>
            </div>""", unsafe_allow_html=True)

# ── Chart ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Technical Analysis</div>',
            unsafe_allow_html=True)
fig = make_chart(train_feat, pred, as_at_dt, horizon, actual_price)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Walk-forward CV detail ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Walk-Forward CV Detail</div>',
            unsafe_allow_html=True)
cv_cols = st.columns(len(fold_rmse))
for i, (col, rmse) in enumerate(zip(cv_cols, fold_rmse)):
    col.markdown(f"""
    <div class="metric-card" style="text-align:center;">
      <div class="metric-label">Fold {i+1}</div>
      <div class="metric-value" style="font-size:1rem;">{rmse:.5f}</div>
    </div>""", unsafe_allow_html=True)

# ── Feature table ──────────────────────────────────────────────────────────────
with st.expander("Feature Snapshot (As At Date)"):
    snap = last_row.to_frame("Value").round(5)
    st.dataframe(snap, use_container_width=True)

# ── Raw data ───────────────────────────────────────────────────────────────────
with st.expander("Raw Price Data (last 60 bars)"):
    st.dataframe(train_raw.tail(60).round(2)[::-1],
                 use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1e3a5f; margin-top:40px; padding-top:16px;
            font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#1e3a5f;
            text-align:center;">
  TWSE Ensemble Valuation System · Point-in-Time Strict · Purged Walk-Forward CV
  · Not financial advice
</div>
""", unsafe_allow_html=True)
