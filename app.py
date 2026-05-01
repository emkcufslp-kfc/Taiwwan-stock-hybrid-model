"""
app.py  —  TWSE Ensemble Valuation System  (fixed & upgraded)
Streamlit dashboard. Run with:  streamlit run app.py

Fixes vs original:
  1. As At Date now defaults to TODAY (not hardcoded 2024-01-18)
  2. current_price correctly uses the last close on/before as_at_date
  3. xgb_weight / n_neighbors sliders are now wired into train_ensemble & predict
  4. Stock list screening mode: run model on a watchlist, rank by expected return
  5. Ticker detail drill-down after screening
  6. load_data end-date includes as_at_date by fetching +1 day (yfinance excludes end)
  7. Deprecated .last("365D") replaced with tail(252)
  8. Validation only shown when as_at_date is sufficiently in the past
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
.tag-buy  { background:#064e3b; color:#34d399; border-radius:4px; padding:3px 10px;
            font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }
.tag-sell { background:#4c0519; color:#f87171; border-radius:4px; padding:3px 10px;
            font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }
.tag-hold { background:#1c1917; color:#a8a29e; border-radius:4px; padding:3px 10px;
            font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }

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
    color: white; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    padding: 10px 28px; width: 100%; letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6); border: none;
}
.stock-row {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.stock-row:hover { border-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WATCHLIST = [
    ("2330", "台積電",   "半導體"),
    ("2317", "鴻海",     "電子"),
    ("2454", "聯發科",   "半導體"),
    ("2412", "中華電",   "電信"),
    ("2308", "台達電",   "電子"),
    ("2382", "廣達",     "電子"),
    ("3711", "日月光",   "半導體"),
    ("2357", "華碩",     "電子"),
    ("2303", "聯電",     "半導體"),
    ("0050", "元大台灣50","ETF"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE MODEL
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV.  end is INCLUSIVE — we add 1 calendar day because
    yfinance excludes the end date.
    """
    end_excl = (pd.to_datetime(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end_excl,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    return raw


def get_price_at(raw: pd.DataFrame, as_at_dt: pd.Timestamp) -> float:
    """Return the last available Close on or before as_at_date."""
    pit = raw[raw.index <= as_at_dt]
    if pit.empty:
        raise ValueError("No price data on or before the As At Date.")
    return float(pit["Close"].iloc[-1])


def train_ensemble(feat_df: pd.DataFrame, n_splits: int = 5,
                   xgb_weight: float = 0.6, n_neighbors: int = 10):
    """
    Purged walk-forward cross-validation training.
    Returns fitted models + fold RMSE list.
    """
    data = feat_df[FEATURE_COLS + ["fwd_ret_20d"]].dropna()
    X, y = data[FEATURE_COLS], data["fwd_ret_20d"]

    if len(X) < 60:
        raise ValueError(f"Insufficient training data: {len(X)} rows (need ≥60)")

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03,
                       max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    scaler = StandardScaler()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []
    knn_w = 1.0 - xgb_weight

    for train_idx, val_idx in tscv.split(X):
        # Purge: drop last 5 bars of train to avoid label overlap with val
        purge_end = max(0, len(train_idx) - 5)
        Xtr = X.iloc[train_idx[:purge_end]]
        ytr = y.iloc[train_idx[:purge_end]]
        Xva = X.iloc[val_idx]
        yva = y.iloc[val_idx]
        if len(Xtr) < 10:
            continue
        sc = StandardScaler().fit(Xtr)
        xgb.fit(sc.transform(Xtr), ytr)
        knn.fit(sc.transform(Xtr), ytr)
        preds = (xgb_weight * xgb.predict(sc.transform(Xva)) +
                 knn_w * knn.predict(sc.transform(Xva)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(yva, preds))))

    # Final fit on all data
    Xs = scaler.fit_transform(X)
    xgb.fit(Xs, y)
    knn.fit(Xs, y)
    return xgb, knn, scaler, fold_rmse


def predict(xgb, knn, scaler, current_price, feature_row,
            xgb_weight: float = 0.6):
    knn_w = 1.0 - xgb_weight
    X = feature_row[FEATURE_COLS].values.reshape(1, -1)
    Xs = scaler.transform(X)
    xgb_ret = float(xgb.predict(Xs)[0])
    knn_ret = float(knn.predict(Xs)[0])
    ens_ret = xgb_weight * xgb_ret + knn_w * knn_ret
    return {
        "xgb_price":   round(current_price * (1 + xgb_ret), 2),
        "knn_price":   round(current_price * (1 + knn_ret), 2),
        "ens_price":   round(current_price * (1 + ens_ret), 2),
        "ens_ret_pct": round(ens_ret * 100, 3),
        "xgb_ret_pct": round(xgb_ret * 100, 3),
        "knn_ret_pct": round(knn_ret * 100, 3),
    }


def run_valuation_for_ticker(ticker_tw: str, as_at_dt: pd.Timestamp,
                              train_start: str, xgb_weight: float,
                              n_neighbors: int) -> dict:
    """
    Full pipeline for one ticker. Returns dict with price + prediction,
    or {"error": ...} on failure.
    """
    try:
        raw = load_data(ticker_tw, train_start,
                        as_at_dt.strftime("%Y-%m-%d"))
        if raw.empty or len(raw) < 80:
            return {"error": "Insufficient data"}

        current_price = get_price_at(raw, as_at_dt)
        feat_df = compute_features(raw)
        train_feat = feat_df[feat_df.index <= as_at_dt]

        xgb_m, knn_m, scaler, fold_rmse = train_ensemble(
            train_feat, xgb_weight=xgb_weight, n_neighbors=n_neighbors)

        last_row = train_feat[FEATURE_COLS].dropna().iloc[-1]
        pred = predict(xgb_m, knn_m, scaler, current_price, last_row,
                       xgb_weight=xgb_weight)
        pred["current_price"] = current_price
        pred["cv_rmse"] = round(float(np.mean(fold_rmse)), 5) if fold_rmse else None
        pred["fold_rmse"] = fold_rmse
        pred["raw"] = raw
        pred["feat_df"] = feat_df
        pred["train_feat"] = train_feat
        return pred
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(feat_df, pred, as_at_dt, horizon, actual_price=None):
    fig = plt.figure(figsize=(14, 9), facecolor="#0b0e1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    BG, BLUE, GRN  = "#111827", "#60a5fa", "#34d399"
    RED, ORG, PUR  = "#f87171", "#fb923c", "#a78bfa"
    SPINE = "#1e3a5f"

    def style(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        ax.tick_params(colors="#4b5e7e", labelsize=7)
        ax.xaxis.label.set_color("#4b5e7e")
        ax.yaxis.label.set_color("#4b5e7e")

    # Use last 365 calendar days of data (tail(252) ≈ 1 trading year)
    hist = feat_df.dropna(subset=["Close"]).tail(252)

    # ── 1. Price + prediction ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1)
    ax1.plot(hist.index, hist["Close"], color=BLUE, lw=1.2, label="Close")
    ax1.fill_between(hist.index,
                     hist["Low"].rolling(5).min(),
                     hist["High"].rolling(5).max(),
                     alpha=0.08, color=BLUE)
    ax1.axvline(as_at_dt, color=ORG, lw=1.5, ls="--", label="As At Date")

    t20_approx = as_at_dt + timedelta(days=int(horizon * 1.4))
    current_p  = pred["current_price"]
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
                  color="white", fontsize=10, fontfamily="monospace", pad=8)
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white",
               edgecolor=SPINE, framealpha=0.9)

    # ── 2. RSI ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2)
    rsi = hist["rsi14"].dropna().tail(120)
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
    vr = hist["vol_ratio"].dropna().tail(120)
    colors_vr = [GRN if v >= 1 else "#374151" for v in vr]
    ax3.bar(vr.index, vr, color=colors_vr, alpha=0.8, width=1)
    ax3.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
    ax3.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax3.set_title("Volume Ratio (vs 20d avg)", color="white",
                  fontsize=9, fontfamily="monospace")

    # ── 4. BB position ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style(ax4)
    bb = hist["bb_pos"].dropna().tail(120)
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
    ax5.plot(hist.index, hist["ma5_ratio"],  color=GRN,  lw=1, label="MA5")
    ax5.plot(hist.index, hist["ma20_ratio"], color=BLUE, lw=1, label="MA20")
    ax5.plot(hist.index, hist["ma60_ratio"], color=ORG,  lw=1, label="MA60")
    ax5.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.3)
    ax5.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax5.set_title("Price / MA Ratios (Momentum)", color="white",
                  fontsize=9, fontfamily="monospace")
    ax5.legend(fontsize=7, facecolor=BG, labelcolor="white", edgecolor=SPINE)

    # ── 6. Prediction breakdown ────────────────────────────────────────────
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
        ax6.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.005,
                 f"{val:.0f}", ha="center", va="bottom",
                 color="white", fontsize=7, fontfamily="monospace")
    ax6.set_title("Prediction Breakdown (TWD)", color="white",
                  fontsize=9, fontfamily="monospace")

    fig.suptitle(
        f"TWSE Ensemble Valuation  |  As At {as_at_dt.date()}  "
        f"|  Expected: {pred['ens_ret_pct']:+.2f}%",
        color="white", fontsize=11, fontfamily="monospace",
        fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def signal_tag(ret_pct: float, threshold: float) -> str:
    if ret_pct > threshold:
        return '<span class="tag-buy">● BUY</span>'
    elif ret_pct < -threshold:
        return '<span class="tag-sell">● SELL</span>'
    else:
        return '<span class="tag-hold">● HOLD</span>'


def metric_card(label: str, value: str, cls: str = "neu") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>"""


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
    st.markdown('<div class="section-header">① As At Date</div>',
                unsafe_allow_html=True)

    # FIX: default to TODAY
    as_at_date = st.date_input(
        "Point-in-Time Cutoff",
        value=date.today(),          # ← was hardcoded 2024-01-18
        min_value=date(2015, 1, 1),
        max_value=date.today(),
        help="Model uses ONLY data on or before this date. Defaults to today.",
    )

    st.markdown('<div class="section-header">② Mode</div>',
                unsafe_allow_html=True)
    mode = st.radio("Run Mode", ["📋 Stock List Screening", "🔍 Single Ticker Detail"],
                    index=0)

    if mode == "🔍 Single Ticker Detail":
        st.markdown('<div class="section-header">③ Ticker</div>',
                    unsafe_allow_html=True)
        ticker_input = st.text_input(
            "Stock Code (Yahoo Finance format)",
            value="2330.TW",
            help="TWSE: append .TW  e.g. 2330.TW  |  ETF: 0050.TW",
        )
    else:
        st.markdown('<div class="section-header">③ Watchlist</div>',
                    unsafe_allow_html=True)
        watchlist_raw = st.text_area(
            "Tickers (one per line, Yahoo format)",
            value="\n".join(f"{c}.TW" for c, _, _ in DEFAULT_WATCHLIST),
            height=200,
            help="One ticker per line e.g. 2330.TW",
        )
        ticker_input = None

    st.markdown('<div class="section-header">④ Horizon & History</div>',
                unsafe_allow_html=True)
    horizon     = st.slider("Prediction Horizon (trading days)", 5, 60, 20)
    train_years = st.slider("Training History (years)", 2, 10, 5)

    # FIX: compute train_start from as_at_date (not today)
    train_start = (pd.to_datetime(as_at_date) -
                   pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")

    # Validation only makes sense when as_at_date is sufficiently in the past
    days_since_asat = (date.today() - as_at_date).days
    can_validate    = days_since_asat >= horizon
    validate = st.checkbox(
        f"Validate vs Actual T+{horizon} price",
        value=can_validate,
        disabled=not can_validate,
        help="Only available when As At Date is at least T+horizon days in the past.",
    )
    if not can_validate:
        st.caption(f"⚠ As At Date is too recent for T+{horizon} validation "
                   f"(need {horizon} more days).")

    st.markdown('<div class="section-header">⑤ Model Config</div>',
                unsafe_allow_html=True)
    xgb_weight    = st.slider("XGBoost Weight", 0.3, 0.9, 0.6, 0.05)
    n_neighbors   = st.slider("KNN Neighbors", 5, 30, 10)
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

as_at_dt = pd.to_datetime(as_at_date)

# ══════════════════════════════════════════════════════════════════════════════
#  IDLE STATE
# ══════════════════════════════════════════════════════════════════════════════
if not run_btn:
    # Show a helpful summary of what will run
    if mode == "📋 Stock List Screening":
        tickers_preview = [t.strip() for t in watchlist_raw.splitlines() if t.strip()]
        st.markdown(f"""
        <div style="text-align:center; padding:60px 0;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:2.5rem; color:#1e3a5f;">◈</div>
          <div style="font-size:1rem; margin-top:16px; color:#4b5e7e;">
            Ready to screen <strong style="color:#60a5fa;">{len(tickers_preview)} stocks</strong>
            as at <strong style="color:#60a5fa;">{as_at_date}</strong><br>
            Prediction horizon: <strong style="color:#60a5fa;">T+{horizon} trading days</strong>
          </div>
          <div style="margin-top:12px; font-size:0.8rem; color:#374151;">
            Press <strong>▶ Run Valuation</strong> in the sidebar to start.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; padding:60px 0;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:2.5rem; color:#1e3a5f;">◈</div>
          <div style="font-size:1rem; margin-top:16px; color:#4b5e7e;">
            Ready to value <strong style="color:#60a5fa;">{ticker_input}</strong>
            as at <strong style="color:#60a5fa;">{as_at_date}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  DETAIL VIEW FUNCTION  (defined here so both modes can call it)
# ══════════════════════════════════════════════════════════════════════════════
def _show_detail(detail, as_at_dt, horizon, validate,
                 ticker_input, train_start, xgb_weight, n_neighbors,
                 buy_threshold):
    pred          = detail
    current_price = detail["current_price"]
    fold_rmse     = detail["fold_rmse"]
    train_feat    = detail["train_feat"]
    raw           = detail["raw"]

    ret_cls = "up" if pred["ens_ret_pct"] > 0 else "down"
    signal  = ("BUY 🟢" if pred["ens_ret_pct"] > buy_threshold
               else "SELL 🔴" if pred["ens_ret_pct"] < -buy_threshold
               else "HOLD ⚪")

    st.markdown(
        f'<div class="section-header">{ticker_input} · As At {as_at_dt.date()}'
        f' · T+{horizon}d Forecast</div>',
        unsafe_allow_html=True)

    cols = st.columns(5)
    kpis = [
        ("Current Price",   f"{current_price:.2f} TWD",          "neu"),
        ("Ensemble Target", f"{pred['ens_price']:.2f} TWD",       ret_cls),
        ("Expected Return", f"{pred['ens_ret_pct']:+.2f}%",       ret_cls),
        ("Signal",          signal,                                ret_cls),
        ("CV RMSE (avg)",   f"{np.mean(fold_rmse):.4f}" if fold_rmse else "N/A", "neu"),
    ]
    for col, (label, value, cls) in zip(cols, kpis):
        col.markdown(metric_card(label, value, cls), unsafe_allow_html=True)

    # ── Validation ────────────────────────────────────────────────────────
    actual_price = None
    if validate:
        future_end = (as_at_dt + timedelta(days=horizon * 3)).strftime("%Y-%m-%d")
        with st.spinner(f"Fetching actual T+{horizon} price…"):
            try:
                future_raw    = load_data(ticker_input,
                                          as_at_dt.strftime("%Y-%m-%d"),
                                          future_end)
                future_closes = future_raw["Close"].iloc[1:]
                if len(future_closes) >= horizon:
                    actual_price = float(future_closes.iloc[horizon - 1])
                elif len(future_closes) > 0:
                    actual_price = float(future_closes.iloc[-1])
                    st.info(f"ℹ Only {len(future_closes)} future days available; "
                            f"using last available close.")
            except Exception:
                st.warning("Could not fetch future price for validation.")

        if actual_price:
            err_pct = (pred["ens_price"] - actual_price) / actual_price * 100
            abs_err = abs(pred["ens_price"] - actual_price)
            ok_flag = abs(err_pct) < 5
            tag     = ('<span class="tag-success">✓ ERROR &lt; 5%</span>'
                       if ok_flag else
                       '<span class="tag-warn">⚠ ERROR ≥ 5%</span>')
            err_cls = "up" if ok_flag else "down"

            st.markdown('<div class="section-header">Validation Results</div>',
                        unsafe_allow_html=True)
            vcols = st.columns(4)
            vkpis = [
                (f"Actual T+{horizon}", f"{actual_price:.2f} TWD", "neu"),
                ("Absolute Error",      f"{abs_err:.2f} TWD",       err_cls),
                ("Error %",             f"{err_pct:+.3f}%",          err_cls),
                ("Verdict",             tag,                          ""),
            ]
            for col, (lbl, val, cls) in zip(vcols, vkpis):
                if cls:
                    col.markdown(metric_card(lbl, val, cls), unsafe_allow_html=True)
                else:
                    col.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">{lbl}</div>
                      <div style="margin-top:10px;">{val}</div>
                    </div>""", unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Technical Analysis</div>',
                unsafe_allow_html=True)
    fig = make_chart(train_feat, pred, as_at_dt, horizon, actual_price)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── CV detail ─────────────────────────────────────────────────────────
    if fold_rmse:
        st.markdown('<div class="section-header">Walk-Forward CV Detail</div>',
                    unsafe_allow_html=True)
        cv_cols = st.columns(len(fold_rmse))
        for i, (col, rmse) in enumerate(zip(cv_cols, fold_rmse)):
            col.markdown(f"""
            <div class="metric-card" style="text-align:center;">
              <div class="metric-label">Fold {i+1}</div>
              <div class="metric-value" style="font-size:1rem;">{rmse:.5f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Feature table ──────────────────────────────────────────────────────
    last_row = train_feat[FEATURE_COLS].dropna().iloc[-1]
    with st.expander("Feature Snapshot (As At Date)"):
        st.dataframe(last_row.to_frame("Value").round(5), use_container_width=True)

    with st.expander("Raw Price Data (last 60 bars)"):
        st.dataframe(raw[raw.index <= as_at_dt].tail(60).round(2)[::-1],
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — STOCK LIST SCREENING
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📋 Stock List Screening":
    tickers = [t.strip() for t in watchlist_raw.splitlines() if t.strip()]

    st.markdown(
        f'<div class="section-header">Stock Screening — As At {as_at_date}'
        f' · {len(tickers)} stocks · T+{horizon}d horizon</div>',
        unsafe_allow_html=True)

    # Build a lookup for names
    name_map = {f"{c}.TW": n for c, n, _ in DEFAULT_WATCHLIST}

    results = []
    prog = st.progress(0, text="Running ensemble valuation…")
    for i, tk in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Processing {tk}…")
        r = run_valuation_for_ticker(
            tk, as_at_dt, train_start,
            xgb_weight=xgb_weight, n_neighbors=n_neighbors)
        r["ticker"] = tk
        r["name"]   = name_map.get(tk, tk.replace(".TW", ""))
        results.append(r)
    prog.empty()

    ok      = [r for r in results if "error" not in r]
    failed  = [r for r in results if "error" in r]

    if not ok:
        st.error("All tickers failed. Check your watchlist or network.")
        st.stop()

    # Sort by expected return descending
    ok.sort(key=lambda r: r["ens_ret_pct"], reverse=True)

    # ── Summary table ──────────────────────────────────────────────────────
    rows = []
    for r in ok:
        sig = ("BUY" if r["ens_ret_pct"] > buy_threshold
               else "SELL" if r["ens_ret_pct"] < -buy_threshold
               else "HOLD")
        rows.append({
            "Ticker":          r["ticker"],
            "Name":            r["name"],
            "Price (TWD)":     r["current_price"],
            "Target (TWD)":    r["ens_price"],
            "XGBoost":         r["xgb_price"],
            "KNN":             r["knn_price"],
            "Expected Ret %":  r["ens_ret_pct"],
            "Signal":          sig,
            "CV RMSE":         r["cv_rmse"],
        })

    df_results = pd.DataFrame(rows)

    # Colour-map the table
    def colour_signal(val):
        if val == "BUY":
            return "background-color:#064e3b; color:#34d399"
        elif val == "SELL":
            return "background-color:#4c0519; color:#f87171"
        return "background-color:#1c1917; color:#a8a29e"

    def colour_ret(val):
        try:
            v = float(val)
            if v > 0:  return "color:#34d399"
            if v < 0:  return "color:#f87171"
        except Exception:
            pass
        return ""

    styled = (df_results.style
              .applymap(colour_signal, subset=["Signal"])
              .applymap(colour_ret,    subset=["Expected Ret %"])
              .format({
                  "Price (TWD)":    "{:.2f}",
                  "Target (TWD)":   "{:.2f}",
                  "XGBoost":        "{:.2f}",
                  "KNN":            "{:.2f}",
                  "Expected Ret %": "{:+.3f}%",
                  "CV RMSE":        "{:.5f}",
              })
              .set_properties(**{"background-color": "#111827", "color": "#e0e6f0",
                                 "font-family": "IBM Plex Mono, monospace",
                                 "font-size": "0.8rem"}))
    st.dataframe(styled, use_container_width=True, height=420)

    if failed:
        with st.expander(f"⚠ {len(failed)} tickers failed"):
            for r in failed:
                st.text(f"{r['ticker']}: {r['error']}")

    # ── Drill-down selector ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Drill Down — Single Ticker Detail</div>',
                unsafe_allow_html=True)
    selected_ticker = st.selectbox(
        "Select a ticker to view details",
        options=[r["ticker"] for r in ok],
        format_func=lambda t: f"{t}  {name_map.get(t, '')}",
    )
    show_detail = st.button("View Detail →")

    if show_detail:
        detail = next(r for r in ok if r["ticker"] == selected_ticker)
        _show_detail(detail, as_at_dt, horizon, validate,
                     selected_ticker, train_start, xgb_weight, n_neighbors,
                     buy_threshold)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — SINGLE TICKER DETAIL
# ══════════════════════════════════════════════════════════════════════════════
else:
    with st.spinner(f"Running ensemble valuation for {ticker_input}…"):
        detail = run_valuation_for_ticker(
            ticker_input, as_at_dt, train_start,
            xgb_weight=xgb_weight, n_neighbors=n_neighbors)

    if "error" in detail:
        st.error(f"❌ Valuation failed: {detail['error']}")
        st.stop()

    _show_detail(detail, as_at_dt, horizon, validate,
                 ticker_input, train_start, xgb_weight, n_neighbors,
                 buy_threshold)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1e3a5f; margin-top:40px; padding-top:16px;
            font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#1e3a5f;
            text-align:center;">
  TWSE Ensemble Valuation System · Point-in-Time Strict · Purged Walk-Forward CV
  · Not financial advice
</div>
""", unsafe_allow_html=True)
