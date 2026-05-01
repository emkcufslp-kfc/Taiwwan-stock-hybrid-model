"""
app.py  —  TWSE Ensemble Valuation System
Streamlit dashboard — redesigned per DESIGN-claude.md
  · Cream canvas / coral CTA / dark navy surfaces
  · Screener table with click-to-analyse panel (Step 3)
  · CSV batch upload for ticker lists
  · Individual ticker search
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN.md tokens ───────────────────────────────────────────────────────────
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & canvas ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #faf9f5 !important;
    color: #141413;
}
.stApp { background-color: #faf9f5 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
div[data-testid="stSidebar"] { background: #f5f0e8 !important; border-right: 1px solid #e6dfd8; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'EB Garamond', serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.5px;
    color: #141413;
}

/* ── Streamlit button override ── */
.stButton > button {
    background: #cc785c !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    letter-spacing: 0 !important;
    width: auto !important;
}
.stButton > button:hover {
    background: #a9583e !important;
    border: none !important;
}
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(204,120,92,0.25) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stDateInput > div > div > input {
    background: #faf9f5 !important;
    border: 1px solid #e6dfd8 !important;
    border-radius: 8px !important;
    color: #141413 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus,
.stDateInput > div > div > input:focus {
    border-color: #cc785c !important;
    box-shadow: 0 0 0 3px rgba(204,120,92,0.15) !important;
}
.stSlider > div { color: #141413 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #cc785c !important;
    border-color: #cc785c !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f5f0e8 !important;
    border: 1.5px dashed #e8e0d2 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #cc785c !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e6dfd8 !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: #6c6a64 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #efe9de !important;
    color: #141413 !important;
    border-color: #e6dfd8 !important;
    border-bottom-color: #efe9de !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #f5f0e8 !important;
    border: 1px solid #e6dfd8 !important;
    border-radius: 8px !important;
    color: #3d3d3a !important;
    font-size: 13px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #cc785c !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #faf9f5 !important;
    border: 1px solid #e6dfd8 !important;
    border-radius: 8px !important;
    color: #141413 !important;
    font-size: 13px !important;
}

/* ── Custom component classes ── */
.d-nav {
    background: #faf9f5;
    border-bottom: 1px solid #e6dfd8;
    padding: 0 32px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.d-wordmark {
    font-family: 'EB Garamond', serif;
    font-size: 18px;
    color: #141413;
    display: flex;
    align-items: center;
    gap: 8px;
}
.d-eyebrow {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6c6a64;
}
.d-summary-bar {
    background: #181715;
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
}
.d-sum-item { display: flex; flex-direction: column; }
.d-sum-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #a09d96;
    margin-bottom: 4px;
}
.d-sum-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    color: #faf9f5;
}
.d-sum-value.buy { color: #5db872; }
.d-sum-value.hold { color: #e8a55a; }
.d-sum-value.sell { color: #c64545; }
.d-sum-value.ret { color: #5db872; }
.d-sum-div {
    width: 1px;
    height: 36px;
    background: #2d2c28;
    margin: 0 24px;
    flex-shrink: 0;
}
.d-section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6c6a64;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e6dfd8;
}
.d-kpi-card {
    background: #efe9de;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 8px;
}
.d-kpi-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6c6a64;
    margin-bottom: 6px;
}
.d-kpi-value {
    font-family: 'EB Garamond', serif;
    font-size: 28px;
    color: #141413;
    letter-spacing: -0.3px;
    line-height: 1;
}
.d-kpi-value.up { color: #4a9e5c; }
.d-kpi-value.down { color: #c64545; }
.d-kpi-value.coral { color: #cc785c; }
.d-kpi-sub { font-size: 11px; color: #6c6a64; margin-top: 4px; }

.d-dark-card {
    background: #181715;
    border-radius: 12px;
    padding: 20px;
    color: #faf9f5;
    margin-bottom: 12px;
}
.d-dark-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #a09d96;
    margin-bottom: 12px;
}
.d-model-row {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 12px;
}
.d-model-item { display: flex; flex-direction: column; gap: 4px; }
.d-model-meta { display: flex; justify-content: space-between; align-items: baseline; }
.d-model-name { font-size: 12px; font-weight: 500; color: #a09d96; }
.d-model-price { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #faf9f5; }
.d-bar-track {
    height: 5px;
    background: #252320;
    border-radius: 9999px;
    overflow: hidden;
}
.d-bar-fill { height: 100%; border-radius: 9999px; }

.d-val-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.d-val-card {
    background: #faf9f5;
    border: 1px solid #e6dfd8;
    border-radius: 10px;
    padding: 12px 16px;
    flex: 1;
    min-width: 120px;
}
.d-val-label { font-size: 10px; color: #6c6a64; margin-bottom: 4px; }
.d-val-value { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 500; color: #141413; }
.d-val-value.success { color: #4a9e5c; }
.d-val-value.error { color: #c64545; }

.d-fold-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-bottom: 12px;
}
.d-fold-card { background: #efe9de; border-radius: 8px; padding: 10px; text-align: center; }
.d-fold-label { font-size: 10px; color: #6c6a64; margin-bottom: 3px; }
.d-fold-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #141413; }

.d-table-row-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: #cc785c;
}
.d-badge-semi {
    display: inline-block;
    background: #252320;
    color: #5db8a6;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 9999px;
}
.d-badge-elec {
    display: inline-block;
    background: #1a2a1a;
    color: #7bc98a;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 9999px;
}
.d-signal-buy {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #eaf3de;
    color: #4a9e5c;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 9999px;
}
.d-signal-hold {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #fef3e2;
    color: #a06b00;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 9999px;
}
.d-signal-sell {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #fdecea;
    color: #c64545;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 9999px;
}
.d-coral-cta {
    background: #cc785c;
    border-radius: 12px;
    padding: 32px 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 24px 0 0;
}
.d-coral-cta h2 {
    font-family: 'EB Garamond', serif !important;
    font-size: 26px !important;
    font-weight: 400 !important;
    color: #ffffff !important;
    letter-spacing: -0.4px !important;
    margin: 0 0 4px !important;
}
.d-coral-cta p { font-size: 13px; color: rgba(255,255,255,0.75); margin: 0; }
.d-footer {
    background: #181715;
    padding: 32px;
    margin-top: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.d-footer-brand {
    font-family: 'EB Garamond', serif;
    font-size: 16px;
    color: #faf9f5;
}
.d-footer-text { font-size: 12px; color: #a09d96; margin-top: 4px; }
.pit-pill {
    background: #efe9de;
    color: #6c6a64;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 9999px;
    border: 1px solid #e6dfd8;
    display: inline-block;
}
</style>
"""
st.markdown(DESIGN_CSS, unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not raw.empty:
            return raw
    except Exception:
        pass
    raise ValueError(f"No data returned for {ticker}")


def train_ensemble(feat_df: pd.DataFrame, n_splits: int = 5,
                   xgb_w: float = 0.6, n_neighbors: int = 10):
    data = feat_df[FEATURE_COLS + ["fwd_ret_20d"]].dropna()
    if len(data) < 60:
        raise ValueError(f"Insufficient data: {len(data)} rows (need ≥60)")
    X, y = data[FEATURE_COLS], data["fwd_ret_20d"]

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03,
                       max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    scaler = StandardScaler()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmse = []
    for train_idx, val_idx in tscv.split(X):
        Xtr = X.iloc[train_idx[:-5]] if len(train_idx) > 5 else X.iloc[train_idx]
        ytr = y.iloc[train_idx[:-5]] if len(train_idx) > 5 else y.iloc[train_idx]
        Xva, yva = X.iloc[val_idx], y.iloc[val_idx]
        if len(Xtr) < 10:
            continue
        sc = StandardScaler().fit(Xtr)
        xgb.fit(sc.transform(Xtr), ytr)
        knn.fit(sc.transform(Xtr), ytr)
        preds = xgb_w * xgb.predict(sc.transform(Xva)) + \
                (1 - xgb_w) * knn.predict(sc.transform(Xva))
        fold_rmse.append(float(np.sqrt(mean_squared_error(yva, preds))))

    Xs = scaler.fit_transform(X)
    xgb.fit(Xs, y)
    knn.fit(Xs, y)
    return xgb, knn, scaler, fold_rmse


def predict(xgb, knn, scaler, current_price, feature_row, xgb_w=0.6):
    X = feature_row[FEATURE_COLS].values.reshape(1, -1)
    Xs = scaler.transform(X)
    xgb_ret = float(xgb.predict(Xs)[0])
    knn_ret  = float(knn.predict(Xs)[0])
    ens_ret  = xgb_w * xgb_ret + (1 - xgb_w) * knn_ret
    return {
        "xgb_price":   round(current_price * (1 + xgb_ret), 2),
        "knn_price":   round(current_price * (1 + knn_ret), 2),
        "ens_price":   round(current_price * (1 + ens_ret), 2),
        "ens_ret_pct": round(ens_ret * 100, 3),
        "xgb_ret_pct": round(xgb_ret * 100, 3),
        "knn_ret_pct": round(knn_ret * 100, 3),
    }


def run_single_analysis(ticker: str, as_at_date, horizon: int,
                        train_years: int, xgb_w: float, n_neighbors: int,
                        buy_threshold: float, validate: bool):
    """Run full pipeline for one ticker. Returns dict with all results."""
    as_at_dt  = pd.to_datetime(as_at_date)
    train_start = (as_at_dt - pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
    as_at_str   = as_at_dt.strftime("%Y-%m-%d")
    future_end  = (as_at_dt + timedelta(days=horizon * 3)).strftime("%Y-%m-%d")

    # Normalise ticker
    ticker_yf = ticker if ticker.endswith(".TW") else f"{ticker}.TW"

    train_raw = load_data(ticker_yf, train_start, as_at_str)
    if train_raw.empty or len(train_raw) < 100:
        raise ValueError(f"Insufficient data for {ticker_yf}")

    feat_df    = compute_features(train_raw)
    train_feat = feat_df[feat_df.index <= as_at_dt]

    xgb_m, knn_m, scaler, fold_rmse = train_ensemble(
        train_feat, xgb_w=xgb_w, n_neighbors=n_neighbors
    )

    last_row      = train_feat[FEATURE_COLS].dropna().iloc[-1]
    current_price = float(train_raw["Close"].iloc[-1])
    pred          = predict(xgb_m, knn_m, scaler, current_price, last_row, xgb_w)

    # Determine signal
    ret = pred["ens_ret_pct"]
    if ret > buy_threshold:
        signal = "買入"
    elif ret < -buy_threshold:
        signal = "賣出"
    else:
        signal = "持有"

    actual_price = None
    if validate:
        try:
            future_raw   = load_data(ticker_yf, as_at_str, future_end)
            future_cl    = future_raw["Close"].iloc[1:]
            if len(future_cl) >= horizon:
                actual_price = float(future_cl.iloc[horizon - 1])
            elif len(future_cl) > 0:
                actual_price = float(future_cl.iloc[-1])
        except Exception:
            actual_price = None

    return {
        "ticker":        ticker_yf,
        "current_price": current_price,
        "pred":          pred,
        "fold_rmse":     fold_rmse,
        "cv_rmse_avg":   float(np.mean(fold_rmse)) if fold_rmse else None,
        "signal":        signal,
        "actual_price":  actual_price,
        "train_feat":    train_feat,
        "train_raw":     train_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHART (DESIGN.md palette: cream bg / coral / teal / amber)
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(feat_df, pred, as_at_dt, horizon, actual_price=None):
    CANVAS = "#faf9f5"
    CARD   = "#efe9de"
    DARK   = "#181715"
    CORAL  = "#cc785c"
    TEAL   = "#5db8a6"
    AMBER  = "#e8a55a"
    RED    = "#c64545"
    INK    = "#3d3d3a"
    MUTED  = "#8e8b82"
    HAIR   = "#e6dfd8"

    fig = plt.figure(figsize=(14, 9), facecolor=CANVAS)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    def style(ax, dark=False):
        bg = DARK if dark else CARD
        ax.set_facecolor(bg)
        spine_c = "#2d2c28" if dark else HAIR
        for sp in ax.spines.values():
            sp.set_color(spine_c)
        tick_c = "#a09d96" if dark else MUTED
        ax.tick_params(colors=tick_c, labelsize=7)
        ax.xaxis.label.set_color(tick_c)
        ax.yaxis.label.set_color(tick_c)

    hist = feat_df.dropna(subset=["Close"]).last("365D")

    # 1. Price + prediction
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, dark=True)
    ax1.plot(hist.index, hist["Close"], color=TEAL, lw=1.3, label="Close")
    ax1.fill_between(hist.index, hist["Low"].rolling(5).min(),
                     hist["High"].rolling(5).max(),
                     alpha=0.06, color=TEAL)
    ax1.axvline(as_at_dt, color=CORAL, lw=1.5, ls="--", label="As At Date")

    t20_approx = as_at_dt + timedelta(days=int(horizon * 1.4))
    current_p  = float(hist["Close"].iloc[-1])
    ax1.annotate("", xy=(t20_approx, pred["ens_price"]),
                 xytext=(as_at_dt, current_p),
                 arrowprops=dict(arrowstyle="->", color="#4a9e5c", lw=1.5))
    ax1.scatter([t20_approx], [pred["ens_price"]],
                color="#4a9e5c", zorder=6, s=60,
                label=f"Ensemble T+{horizon}: {pred['ens_price']:,.0f}")
    if actual_price:
        ax1.scatter([t20_approx], [actual_price],
                    color=RED, zorder=6, s=60, marker="x",
                    label=f"Actual T+{horizon}: {actual_price:,.0f}")
    ax1.set_title(f"Price History & T+{horizon} Ensemble Forecast",
                  color="#faf9f5", fontsize=10, pad=8)
    ax1.legend(fontsize=7, facecolor=DARK, labelcolor="#faf9f5",
               edgecolor="#2d2c28", framealpha=0.95)

    # 2. RSI
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, dark=True)
    rsi = hist["rsi14"].dropna().last("180D")
    ax2.plot(rsi.index, rsi, color="#a78bfa", lw=1)
    ax2.axhline(70, color=RED, lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(30, color="#4a9e5c", lw=0.8, ls="--", alpha=0.6)
    ax2.fill_between(rsi.index, 30, rsi, where=rsi < 30, alpha=0.2, color="#4a9e5c")
    ax2.fill_between(rsi.index, 70, rsi, where=rsi > 70, alpha=0.2, color=RED)
    ax2.axvline(as_at_dt, color=CORAL, lw=1, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI (14)", color="#faf9f5", fontsize=9)

    # 3. Volume ratio
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, dark=True)
    vr = hist["vol_ratio"].dropna().last("180D")
    colors_vr = [TEAL if v >= 1 else "#2d2c28" for v in vr]
    ax3.bar(vr.index, vr, color=colors_vr, alpha=0.85, width=1)
    ax3.axhline(1.0, color="#faf9f5", lw=0.7, ls="--", alpha=0.3)
    ax3.axvline(as_at_dt, color=CORAL, lw=1, ls="--")
    ax3.set_title("Volume Ratio (vs 20d avg)", color="#faf9f5", fontsize=9)

    # 4. Bollinger Band position
    ax4 = fig.add_subplot(gs[1, 2])
    style(ax4, dark=True)
    bb = hist["bb_pos"].dropna().last("180D")
    ax4.plot(bb.index, bb, color=TEAL, lw=1)
    ax4.axhline(0, color="#faf9f5", lw=0.7, alpha=0.3)
    ax4.axhline(1,  color=RED, lw=0.7, ls="--", alpha=0.5)
    ax4.axhline(-1, color="#4a9e5c", lw=0.7, ls="--", alpha=0.5)
    ax4.fill_between(bb.index, 0, bb, where=bb > 0, alpha=0.15, color=RED)
    ax4.fill_between(bb.index, 0, bb, where=bb < 0, alpha=0.15, color="#4a9e5c")
    ax4.axvline(as_at_dt, color=CORAL, lw=1, ls="--")
    ax4.set_title("Bollinger Band Position", color="#faf9f5", fontsize=9)

    # 5. MA ratios
    ax5 = fig.add_subplot(gs[2, 0:2])
    style(ax5)
    ax5.plot(hist.index, hist["ma5_ratio"],  color="#4a9e5c", lw=1, label="MA5")
    ax5.plot(hist.index, hist["ma20_ratio"], color=TEAL,      lw=1, label="MA20")
    ax5.plot(hist.index, hist["ma60_ratio"], color=AMBER,     lw=1, label="MA60")
    ax5.axhline(1.0, color=INK, lw=0.7, ls="--", alpha=0.3)
    ax5.axvline(as_at_dt, color=CORAL, lw=1, ls="--")
    ax5.set_title("Price / MA Ratios (Momentum)", color=INK, fontsize=9)
    ax5.legend(fontsize=7, facecolor=CARD, labelcolor=INK, edgecolor=HAIR)

    # 6. Prediction breakdown
    ax6 = fig.add_subplot(gs[2, 2])
    style(ax6)
    labels = ["Current", "XGBoost", "KNN", "Ensemble"]
    values = [current_p, pred["xgb_price"], pred["knn_price"], pred["ens_price"]]
    bar_colors = [TEAL, "#a78bfa", AMBER, CORAL]
    if actual_price:
        labels.append("Actual")
        values.append(actual_price)
        bar_colors.append(RED)
    bars = ax6.bar(labels, values, color=bar_colors, alpha=0.85,
                   edgecolor=HAIR, width=0.6)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.005,
                 f"{val:,.0f}", ha="center", va="bottom", color=INK, fontsize=7)
    ax6.set_title("Prediction Breakdown (TWD)", color=INK, fontsize=9)

    fig.suptitle(
        f"TWSE Ensemble Valuation  ·  As At {as_at_dt.date()}  "
        f"·  Expected: {pred['ens_ret_pct']:+.2f}%",
        color=INK, fontsize=11, fontweight="normal", y=1.01
    )
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — CSV PARSING
# ══════════════════════════════════════════════════════════════════════════════

TICKER_COL_NAMES = {"code", "ticker", "股票代碼", "symbol", "代碼"}

def parse_csv_tickers(file) -> list[str]:
    """Extract ticker codes from uploaded CSV."""
    try:
        df = pd.read_csv(file, dtype=str)
    except Exception as e:
        st.error(f"CSV 讀取失敗：{e}")
        return []

    # Try to find a matching column
    matched_col = None
    for col in df.columns:
        if col.strip().lower() in TICKER_COL_NAMES:
            matched_col = col
            break
    if matched_col is None:
        matched_col = df.columns[0]  # fall back to first column

    tickers = df[matched_col].dropna().astype(str).str.strip()
    tickers = tickers[tickers != ""].tolist()
    return tickers


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "screener_results": [],          # list of result dicts from batch run
        "selected_ticker":  None,        # ticker currently shown in panel
        "analysis_cache":   {},          # ticker -> result dict cache
        "batch_tickers":    [],          # list of tickers from CSV or text
        "batch_ran":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="d-nav">
  <div class="d-wordmark">
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="2.5" fill="#141413"/>
      <line x1="10" y1="10" x2="10" y2="1"  stroke="#141413" stroke-width="1.2" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="10" y2="19" stroke="#141413" stroke-width="1.2" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="1"  y2="10" stroke="#141413" stroke-width="1.2" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="19" y2="10" stroke="#141413" stroke-width="1.2" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="3.93" y2="3.93"   stroke="#141413" stroke-width="1" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="16.07" y2="16.07" stroke="#141413" stroke-width="1" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="16.07" y2="3.93"  stroke="#141413" stroke-width="1" stroke-linecap="round"/>
      <line x1="10" y1="10" x2="3.93" y2="16.07"  stroke="#141413" stroke-width="1" stroke-linecap="round"/>
    </svg>
    TWSE Ensemble Valuation
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <span class="pit-pill">PIT Strict</span>
    <span style="font-size:12px;color:#6c6a64;">XGBoost + KNN · Purged Walk-Forward CV</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — global parameters
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ 模型參數")
    as_at_date  = st.date_input("基準日 (As At Date)",
                                value=date.today(),
                                min_value=date(2015, 1, 1),
                                max_value=date.today())
    horizon     = st.slider("預測時間區間 (交易日)", 5, 60, 20)
    train_years = st.slider("訓練歷史 (年)", 2, 10, 5)
    validate    = st.checkbox("執行 T+N 驗證（獲取實際價格）", value=False)

    st.markdown("---")
    st.markdown("### 🤖 模型設定")
    xgb_weight    = st.slider("XGBoost 權重", 0.3, 0.9, 0.6, 0.05)
    n_neighbors   = st.slider("KNN 近鄰數", 5, 30, 10)
    buy_threshold = st.slider("買入門檻 (%)", 1, 20, 5)

    st.markdown("---")
    st.caption("不構成投資建議。僅供研究與教育用途。")


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT SECTION — ticker text OR CSV upload
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div style="padding: 20px 32px 0;">', unsafe_allow_html=True)

col_search, col_csv, col_date_disp = st.columns([3, 2, 1])

with col_search:
    ticker_text = st.text_input(
        "代碼（如 2330）或留空顯示全部 · 多個代碼用逗號分隔",
        placeholder="2330, 2317, 0050 …",
        label_visibility="visible",
    )

with col_csv:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    csv_file = st.file_uploader(
        "上傳 CSV 清單",
        type=["csv"],
        help="CSV 第一欄（或欄名為 code/ticker/股票代碼）作為代碼清單",
        label_visibility="visible",
    )

with col_date_disp:
    st.markdown(
        f"<div style='padding-top:28px;font-size:12px;color:#6c6a64;'>"
        f"基準日<br><span style='font-family:JetBrains Mono,monospace;"
        f"font-size:14px;color:#141413;font-weight:500;'>"
        f"{as_at_date.strftime('%Y/%m/%d')}</span></div>",
        unsafe_allow_html=True,
    )

# Resolve ticker list
if csv_file is not None:
    csv_tickers = parse_csv_tickers(csv_file)
    if csv_tickers:
        st.success(f"✓ 從 CSV 讀取 {len(csv_tickers)} 個代碼：{', '.join(csv_tickers[:8])}{'…' if len(csv_tickers) > 8 else ''}")
        st.session_state["batch_tickers"] = csv_tickers
elif ticker_text.strip():
    manual = [t.strip().upper() for t in ticker_text.split(",") if t.strip()]
    st.session_state["batch_tickers"] = manual
else:
    # Default watchlist when nothing provided
    if not st.session_state["batch_tickers"]:
        st.session_state["batch_tickers"] = [
            "2330", "2317", "2308", "2360", "2408",
            "2345", "3037", "3533", "4938", "2382",
        ]

col_run, col_reset = st.columns([1, 5])
with col_run:
    run_batch = st.button("▶ 執行批次估值")
with col_reset:
    if st.button("重置", key="reset_btn"):
        st.session_state["screener_results"] = []
        st.session_state["selected_ticker"]  = None
        st.session_state["analysis_cache"]   = {}
        st.session_state["batch_ran"]        = False
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH RUN
# ══════════════════════════════════════════════════════════════════════════════

if run_batch:
    tickers = st.session_state["batch_tickers"]
    results = []
    prog    = st.progress(0, text="批次估值中…")
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"分析 {t} … ({i+1}/{len(tickers)})")
        try:
            res = run_single_analysis(
                t, as_at_date, horizon, train_years,
                xgb_weight, n_neighbors, buy_threshold, validate
            )
            results.append(res)
            st.session_state["analysis_cache"][res["ticker"]] = res
        except Exception as e:
            results.append({"ticker": t, "error": str(e)})
    prog.empty()
    st.session_state["screener_results"] = results
    st.session_state["batch_ran"]        = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY BAR (only when results exist)
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state["screener_results"]
good    = [r for r in results if "error" not in r]

if good:
    n_buy  = sum(1 for r in good if r["signal"] == "買入")
    n_hold = sum(1 for r in good if r["signal"] == "持有")
    n_sell = sum(1 for r in good if r["signal"] == "賣出")
    avg_ret = np.mean([r["pred"]["ens_ret_pct"] for r in good])
    upd_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
    <div class="d-summary-bar">
      <div class="d-sum-item">
        <div class="d-sum-label">覆蓋標的</div>
        <div class="d-sum-value">{len(good)}</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item">
        <div class="d-sum-label">買入</div>
        <div class="d-sum-value buy">{n_buy}</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item">
        <div class="d-sum-label">持有</div>
        <div class="d-sum-value hold">{n_hold}</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item">
        <div class="d-sum-label">賣出</div>
        <div class="d-sum-value sell">{n_sell}</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item">
        <div class="d-sum-label">平均報酬</div>
        <div class="d-sum-value ret">{avg_ret:+.2f}%</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item" style="margin-left:auto;">
        <div class="d-sum-label">批次基準日</div>
        <div class="d-sum-value" style="font-size:14px;">{as_at_date.strftime('%Y-%m-%d')}</div>
      </div>
      <div class="d-sum-div"></div>
      <div class="d-sum-item">
        <div class="d-sum-label">最後更新</div>
        <div class="d-sum-value" style="font-size:14px;">{upd_time}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER TABLE + ANALYSIS PANEL
# ══════════════════════════════════════════════════════════════════════════════

if not good and not st.session_state["batch_ran"]:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:80px 32px;color:#6c6a64;">
      <div style="font-family:'EB Garamond',serif;font-size:48px;color:#e6dfd8;">◈</div>
      <div style="font-family:'EB Garamond',serif;font-size:28px;color:#141413;
                  margin:16px 0 8px;letter-spacing:-0.3px;">
        開始您的估值分析
      </div>
      <div style="font-size:14px;line-height:1.7;max-width:420px;margin:0 auto;">
        輸入代碼或上傳 CSV 清單，按下<strong>執行批次估值</strong>。<br>
        點擊結果列表中的任一標的，即可查看完整 Step 3 估值報告。
      </div>
    </div>
    """, unsafe_allow_html=True)

elif good:
    # Signal filter tabs
    st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
    tab_all, tab_buy, tab_hold, tab_sell = st.tabs([
        f"全部 {len(good)}",
        f"買入 {n_buy}",
        f"持有 {n_hold}",
        f"賣出 {n_sell}",
    ])

    def render_screener(filtered_results):
        selected = st.session_state.get("selected_ticker")
        table_col, panel_col = st.columns([3, 1.1])

        with table_col:
            # Table header
            hcols = st.columns([0.7, 2.5, 0.8, 1, 1.2, 1, 1, 0.8])
            for label, col in zip(
                ["代碼", "名稱 & 業務", "產業", "現價", "目標價", "預期報酬", "訊號", "RMSE"],
                hcols
            ):
                col.markdown(
                    f"<div style='font-size:10px;font-weight:500;letter-spacing:0.8px;"
                    f"text-transform:uppercase;color:#6c6a64;padding:6px 0;"
                    f"border-bottom:1px solid #e6dfd8;'>{label}</div>",
                    unsafe_allow_html=True
                )

            for r in filtered_results:
                if "error" in r:
                    st.markdown(
                        f"<div style='padding:8px 0;font-size:12px;color:#c64545;'>"
                        f"❌ {r['ticker']}: {r['error']}</div>",
                        unsafe_allow_html=True
                    )
                    continue

                ticker   = r["ticker"]
                code     = ticker.replace(".TW", "")
                pred     = r["pred"]
                signal   = r["signal"]
                rmse_avg = r["cv_rmse_avg"] or 0
                cur_p    = r["current_price"]
                ens_p    = pred["ens_price"]
                ret_pct  = pred["ens_ret_pct"]

                # Signal badge
                if signal == "買入":
                    sig_html = '<span class="d-signal-buy">● 買入</span>'
                elif signal == "賣出":
                    sig_html = '<span class="d-signal-sell">● 賣出</span>'
                else:
                    sig_html = '<span class="d-signal-hold">● 持有</span>'

                ret_color = "#4a9e5c" if ret_pct > 0 else "#c64545"
                rmse_pct  = min(int(rmse_avg / 0.25 * 100), 100)
                rmse_bar_col = "#5db8a6" if rmse_avg < 0.12 else ("#e8a55a" if rmse_avg < 0.18 else "#c64545")

                # Active row highlight
                is_active = (selected == ticker)
                row_bg    = "#efe9de" if is_active else "transparent"
                row_border = "border-left:3px solid #cc785c;" if is_active else "border-left:3px solid transparent;"

                row_cols = st.columns([0.7, 2.5, 0.8, 1, 1.2, 1, 1, 0.8])
                row_cols[0].markdown(
                    f"<div style='padding:10px 0 10px;{row_border}background:{row_bg};"
                    f"font-family:JetBrains Mono,monospace;font-size:13px;"
                    f"font-weight:500;color:#cc785c;'>{code}</div>",
                    unsafe_allow_html=True
                )
                row_cols[1].markdown(
                    f"<div style='padding:10px 4px;background:{row_bg};'>"
                    f"<div style='font-size:13px;font-weight:500;color:#141413;'>{code}</div>"
                    f"<div style='font-size:10px;color:#6c6a64;'>TWSE · {ticker}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                row_cols[2].markdown(
                    '<div style="padding:10px 0;">'
                    '<span class="d-badge-semi">半導體</span></div>',
                    unsafe_allow_html=True
                )
                row_cols[3].markdown(
                    f"<div style='padding:10px 0;font-family:JetBrains Mono,monospace;"
                    f"font-size:13px;color:#3d3d3a;text-align:right;'>{cur_p:,.2f}</div>",
                    unsafe_allow_html=True
                )
                row_cols[4].markdown(
                    f"<div style='padding:10px 0;text-align:right;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:13px;"
                    f"font-weight:500;color:#cc785c;'>{ens_p:,.2f}</div>"
                    f"<div style='font-size:10px;color:#6c6a64;'>"
                    f"xgb {pred['xgb_price']:,.0f} · knn {pred['knn_price']:,.0f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                row_cols[5].markdown(
                    f"<div style='padding:10px 0;font-family:JetBrains Mono,monospace;"
                    f"font-size:13px;font-weight:500;color:{ret_color};"
                    f"text-align:right;'>{ret_pct:+.2f}%</div>",
                    unsafe_allow_html=True
                )
                row_cols[6].markdown(
                    f"<div style='padding:10px 0;text-align:right;'>{sig_html}</div>",
                    unsafe_allow_html=True
                )
                row_cols[7].markdown(
                    f"<div style='padding:10px 0;text-align:right;'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                    f"color:#6c6a64;'>{rmse_avg:.3f}</div>"
                    f"<div style='height:3px;background:#e6dfd8;border-radius:9999px;"
                    f"overflow:hidden;margin-top:4px;'>"
                    f"<div style='height:100%;width:{rmse_pct}%;background:{rmse_bar_col};"
                    f"border-radius:9999px;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Click button (invisible label, styled as row)
                if row_cols[0].button("▶", key=f"sel_{ticker}", help=f"分析 {ticker}"):
                    st.session_state["selected_ticker"] = ticker
                    # Run analysis if not cached
                    if ticker not in st.session_state["analysis_cache"]:
                        with st.spinner(f"分析 {ticker}…"):
                            try:
                                res2 = run_single_analysis(
                                    ticker, as_at_date, horizon, train_years,
                                    xgb_weight, n_neighbors, buy_threshold, validate
                                )
                                st.session_state["analysis_cache"][ticker] = res2
                            except Exception as e:
                                st.error(f"分析失敗：{e}")
                    st.rerun()

        # ── Analysis panel ─────────────────────────────────────────────────
        with panel_col:
            sel = st.session_state.get("selected_ticker")
            cache = st.session_state.get("analysis_cache", {})

            if sel and sel in cache:
                r = cache[sel]
                pred    = r["pred"]
                rmse_l  = r["fold_rmse"]
                rmse_a  = r["cv_rmse_avg"] or 0
                cur_p   = r["current_price"]
                ens_p   = pred["ens_price"]
                ret_pct = pred["ens_ret_pct"]
                xgb_p   = pred["xgb_price"]
                knn_p   = pred["knn_price"]
                sig     = r["signal"]
                act_p   = r.get("actual_price")

                ret_color = "#4a9e5c" if ret_pct > 0 else "#c64545"
                code = sel.replace(".TW", "")

                st.markdown(
                    f"<div style='border-bottom:1px solid #e6dfd8;padding:14px 0 12px;"
                    f"display:flex;align-items:flex-start;justify-content:space-between;'>"
                    f"<div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:16px;"
                    f"font-weight:500;color:#cc785c;'>{sel}</div>"
                    f"<div style='font-family:EB Garamond,serif;font-size:18px;"
                    f"color:#141413;margin-top:2px;'>{code}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

                if st.button("✕ 關閉", key="close_panel"):
                    st.session_state["selected_ticker"] = None
                    st.rerun()

                # KPI mini cards
                st.markdown('<div class="d-section-label" style="margin-top:14px;">估值結果</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.markdown(f"""
                <div class="d-kpi-card">
                  <div class="d-kpi-label">現價</div>
                  <div class="d-kpi-value" style="font-size:20px;">{cur_p:,.2f}</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""
                <div class="d-kpi-card">
                  <div class="d-kpi-label">Ensemble 目標價</div>
                  <div class="d-kpi-value coral" style="font-size:20px;">{ens_p:,.2f}</div>
                </div>""", unsafe_allow_html=True)
                c1.markdown(f"""
                <div class="d-kpi-card">
                  <div class="d-kpi-label">預期報酬</div>
                  <div class="d-kpi-value {'up' if ret_pct>0 else 'down'}" style="font-size:20px;">{ret_pct:+.2f}%</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""
                <div class="d-kpi-card">
                  <div class="d-kpi-label">CV RMSE 平均</div>
                  <div class="d-kpi-value" style="font-size:20px;">{rmse_a:.4f}</div>
                </div>""", unsafe_allow_html=True)

                # Model breakdown
                st.markdown('<div class="d-section-label" style="margin-top:4px;">模型拆解</div>', unsafe_allow_html=True)
                max_p = max(cur_p, xgb_p, knn_p, ens_p) * 1.05
                for name, price, color in [
                    ("XGBoost (60%)", xgb_p, "#5db8a6"),
                    ("KNN (40%)",     knn_p, "#e8a55a"),
                    ("Ensemble",      ens_p, "#cc785c"),
                ]:
                    pct = int(price / max_p * 100)
                    st.markdown(
                        f"<div class='d-model-item' style='margin-bottom:8px;'>"
                        f"<div class='d-model-meta'>"
                        f"<span style='font-size:11px;font-weight:500;color:#6c6a64;'>{name}</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;color:#141413;'>{price:,.2f}</span>"
                        f"</div>"
                        f"<div class='d-bar-track' style='background:#e6dfd8;'>"
                        f"<div class='d-bar-fill' style='width:{pct}%;background:{color};'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )

                # Validation
                if act_p:
                    err_pct = (ens_p - act_p) / act_p * 100
                    abs_err = abs(ens_p - act_p)
                    ok = abs(err_pct) < 5
                    st.markdown('<div class="d-section-label" style="margin-top:4px;">T+N 驗證</div>', unsafe_allow_html=True)
                    v1, v2, v3 = st.columns(3)
                    v1.markdown(f'<div class="d-val-card"><div class="d-val-label">實際價格</div><div class="d-val-value">{act_p:,.2f}</div></div>', unsafe_allow_html=True)
                    v2.markdown(f'<div class="d-val-card"><div class="d-val-label">絕對誤差</div><div class="d-val-value">{abs_err:,.2f}</div></div>', unsafe_allow_html=True)
                    cls = "success" if ok else "error"
                    v3.markdown(f'<div class="d-val-card"><div class="d-val-label">誤差率</div><div class="d-val-value {cls}">{err_pct:+.2f}%</div></div>', unsafe_allow_html=True)

                # CV folds
                st.markdown('<div class="d-section-label" style="margin-top:8px;">Walk-Forward CV · 5折</div>', unsafe_allow_html=True)
                fold_cols = st.columns(len(rmse_l) if rmse_l else 5)
                for i, (fc, fv) in enumerate(zip(fold_cols, rmse_l or [])):
                    fc.markdown(
                        f"<div class='d-fold-card'><div class='d-fold-label'>F{i+1}</div>"
                        f"<div class='d-fold-val'>{fv:.3f}</div></div>",
                        unsafe_allow_html=True
                    )

                # Chart
                st.markdown('<div class="d-section-label" style="margin-top:8px;">技術分析圖表</div>', unsafe_allow_html=True)
                with st.spinner("繪製圖表…"):
                    fig = make_chart(r["train_feat"], pred,
                                     pd.to_datetime(as_at_date), horizon, act_p)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # Export CSV
                st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
                export_data = {
                    "ticker":        [sel],
                    "as_at_date":    [str(as_at_date)],
                    "current_price": [cur_p],
                    "xgb_target":    [xgb_p],
                    "knn_target":    [knn_p],
                    "ensemble_target": [ens_p],
                    "expected_ret_pct": [ret_pct],
                    "signal":        [sig],
                    "cv_rmse_avg":   [rmse_a],
                    "actual_price":  [act_p],
                }
                export_df = pd.DataFrame(export_data)
                st.download_button(
                    label="⬇ 匯出此標的 CSV",
                    data=export_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{code}_{as_at_date}_valuation.csv",
                    mime="text/csv",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                # Empty panel state
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;
                            justify-content:center;min-height:300px;text-align:center;
                            padding:24px;">
                  <div style="width:48px;height:48px;background:#efe9de;border-radius:12px;
                              display:flex;align-items:center;justify-content:center;
                              margin-bottom:14px;font-size:20px;color:#6c6a64;">◈</div>
                  <div style="font-family:'EB Garamond',serif;font-size:20px;
                              color:#141413;margin-bottom:6px;">點擊列表中的標的</div>
                  <div style="font-size:12px;color:#6c6a64;line-height:1.7;">
                    按左側 ▶ 按鈕選取任一標的<br>系統將執行完整 Step 3 估值分析
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_all:
        render_screener(good)
    with tab_buy:
        render_screener([r for r in good if r["signal"] == "買入"])
    with tab_hold:
        render_screener([r for r in good if r["signal"] == "持有"])
    with tab_sell:
        render_screener([r for r in good if r["signal"] == "賣出"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Bulk export
    if good:
        st.markdown('<div style="padding:0 32px;">', unsafe_allow_html=True)
        bulk_rows = []
        for r in good:
            if "error" in r:
                continue
            bulk_rows.append({
                "ticker":            r["ticker"],
                "as_at_date":        str(as_at_date),
                "current_price":     r["current_price"],
                "xgb_target":        r["pred"]["xgb_price"],
                "knn_target":        r["pred"]["knn_price"],
                "ensemble_target":   r["pred"]["ens_price"],
                "expected_ret_pct":  r["pred"]["ens_ret_pct"],
                "signal":            r["signal"],
                "cv_rmse_avg":       r["cv_rmse_avg"],
                "actual_price":      r.get("actual_price"),
            })
        bulk_df = pd.DataFrame(bulk_rows)
        st.markdown("""
        <div class="d-coral-cta">
          <div>
            <h2>匯出或部署此篩選器</h2>
            <p>下載批次 CSV · 部署至 Streamlit · 連接 FinMind API</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇ 下載全部結果 CSV",
            data=bulk_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"twse_valuation_{as_at_date}.csv",
            mime="text/csv",
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="d-footer">
  <div>
    <div class="d-footer-brand">TWSE Ensemble Valuation System</div>
    <div class="d-footer-text">XGBoost + KNN · Purged Walk-Forward CV · Point-in-Time Strict · 不構成投資建議</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center;">
    <span class="pit-pill">PIT Strict</span>
    <span style="font-size:12px;color:#a09d96;">僅供研究與教育用途</span>
  </div>
</div>
""", unsafe_allow_html=True)
