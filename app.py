"""
app.py  —  台股法說會估值系統 (TWSE Ensemble Valuation System)
Streamlit 儀表板。執行方式：  streamlit run app.py
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

# ── 頁面設定 ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="台股法說會估值",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自訂 CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Noto Sans TC', sans-serif;
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
#  核心模型
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "ret_5d", "ret_20d", "ret_60d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "vol_20d", "vol_60d", "vol_ratio",
    "rsi14", "bb_pos",
]


def last_n_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """傳回最後 n 個日曆天的資料列（相容新版 pandas）。"""
    if df.empty:
        return df
    cutoff = df.index[-1] - pd.Timedelta(days=n)
    return df[df.index >= cutoff]


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
        raise ValueError(f"查無 {ticker} 的資料")
    return raw


def train_ensemble(feat_df: pd.DataFrame, n_splits: int = 5,
                   xgb_w: float = 0.6, n_neighbors: int = 10):
    data = feat_df[FEATURE_COLS + ["fwd_ret_20d"]].dropna()
    X, y = data[FEATURE_COLS], data["fwd_ret_20d"]

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.03,
                       max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    scaler = StandardScaler()
    knn_w = 1.0 - xgb_w

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
        preds = xgb_w * xgb.predict(sc.transform(Xva)) + \
                knn_w * knn.predict(sc.transform(Xva))
        fold_rmse.append(float(np.sqrt(mean_squared_error(yva, preds))))

    # 最終以全資料訓練
    Xs = scaler.fit_transform(X)
    xgb.fit(Xs, y)
    knn.fit(Xs, y)
    return xgb, knn, scaler, fold_rmse, xgb_w, knn_w


def predict(xgb, knn, scaler, current_price, feature_row, xgb_w=0.6):
    knn_w = 1.0 - xgb_w
    X = feature_row[FEATURE_COLS].values.reshape(1, -1)
    Xs = scaler.transform(X)
    xgb_ret = float(xgb.predict(Xs)[0])
    knn_ret = float(knn.predict(Xs)[0])
    ens_ret = xgb_w * xgb_ret + knn_w * knn_ret
    return {
        "xgb_price":   round(current_price * (1 + xgb_ret), 2),
        "knn_price":   round(current_price * (1 + knn_ret), 2),
        "ens_price":   round(current_price * (1 + ens_ret), 2),
        "ens_ret_pct": round(ens_ret * 100, 3),
        "xgb_ret_pct": round(xgb_ret * 100, 3),
        "knn_ret_pct": round(knn_ret * 100, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  圖表
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(feat_df, pred, as_at_dt, horizon, actual_price=None):
    fig = plt.figure(figsize=(14, 9), facecolor="#0b0e1a")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    BG    = "#111827"
    BLUE  = "#60a5fa"
    GRN   = "#34d399"
    RED   = "#f87171"
    ORG   = "#fb923c"
    PUR   = "#a78bfa"
    SPINE = "#1e3a5f"

    def style(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color(SPINE)
        ax.tick_params(colors="#4b5e7e", labelsize=7)
        ax.xaxis.label.set_color("#4b5e7e")
        ax.yaxis.label.set_color("#4b5e7e")

    hist = last_n_days(feat_df.dropna(subset=["Close"]), 365)

    # ── 1. 股價 + 預測 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1)
    ax1.plot(hist.index, hist["Close"], color=BLUE, lw=1.2, label="收盤價")
    if "Low" in hist.columns and "High" in hist.columns:
        ax1.fill_between(hist.index, hist["Low"].rolling(5).min(),
                         hist["High"].rolling(5).max(),
                         alpha=0.08, color=BLUE)
    ax1.axvline(as_at_dt, color=ORG, lw=1.5, ls="--", label="評估基準日")

    t20_approx = as_at_dt + timedelta(days=horizon * 1.4)
    current_p  = float(hist["Close"].iloc[-1])
    ax1.annotate("", xy=(t20_approx, pred["ens_price"]),
                 xytext=(as_at_dt, current_p),
                 arrowprops=dict(arrowstyle="->", color=GRN, lw=1.5))
    ax1.scatter([t20_approx], [pred["ens_price"]],
                color=GRN, zorder=6, s=60,
                label=f"集成 T+{horizon}: {pred['ens_price']:.0f}")
    if actual_price:
        ax1.scatter([t20_approx], [actual_price],
                    color=RED, zorder=6, s=60, marker="x",
                    label=f"實際 T+{horizon}: {actual_price:.0f}")
    ax1.set_title(f"股價走勢與 T+{horizon} 集成預測",
                  color="white", fontsize=10, pad=8)
    ax1.legend(fontsize=7, facecolor=BG, labelcolor="white",
               edgecolor=SPINE, framealpha=0.9)

    # ── 2. RSI ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2)
    rsi = last_n_days(hist["rsi14"].dropna().to_frame(), 180)["rsi14"]
    ax2.plot(rsi.index, rsi, color=PUR, lw=1)
    ax2.axhline(70, color=RED, lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(30, color=GRN, lw=0.8, ls="--", alpha=0.6)
    ax2.fill_between(rsi.index, 30, rsi, where=rsi < 30, alpha=0.2, color=GRN)
    ax2.fill_between(rsi.index, 70, rsi, where=rsi > 70, alpha=0.2, color=RED)
    ax2.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI (14)", color="white", fontsize=9)

    # ── 3. 成交量比率 ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3)
    vr = last_n_days(hist["vol_ratio"].dropna().to_frame(), 180)["vol_ratio"]
    colors_vr = [GRN if v >= 1 else "#374151" for v in vr]
    ax3.bar(vr.index, vr, color=colors_vr, alpha=0.8, width=1)
    ax3.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
    ax3.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax3.set_title("成交量比率 (vs 20日均量)", color="white", fontsize=9)

    # ── 4. 布林通道位置 ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style(ax4)
    bb = last_n_days(hist["bb_pos"].dropna().to_frame(), 180)["bb_pos"]
    ax4.plot(bb.index, bb, color=BLUE, lw=1)
    ax4.axhline(0, color="white", lw=0.7, alpha=0.4)
    ax4.axhline(1, color=RED, lw=0.7, ls="--", alpha=0.6)
    ax4.axhline(-1, color=GRN, lw=0.7, ls="--", alpha=0.6)
    ax4.fill_between(bb.index, 0, bb, where=bb > 0, alpha=0.15, color=RED)
    ax4.fill_between(bb.index, 0, bb, where=bb < 0, alpha=0.15, color=GRN)
    ax4.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax4.set_title("布林通道位置", color="white", fontsize=9)

    # ── 5. 均線比率 ───────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    style(ax5)
    ax5.plot(hist.index, hist["ma5_ratio"],  color=GRN,  lw=1, label="MA5 比率")
    ax5.plot(hist.index, hist["ma20_ratio"], color=BLUE, lw=1, label="MA20 比率")
    ax5.plot(hist.index, hist["ma60_ratio"], color=ORG,  lw=1, label="MA60 比率")
    ax5.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.3)
    ax5.axvline(as_at_dt, color=ORG, lw=1, ls="--")
    ax5.set_title("股價/均線比率（動能指標）", color="white", fontsize=9)
    ax5.legend(fontsize=7, facecolor=BG, labelcolor="white", edgecolor=SPINE)

    # ── 6. 模型預測分解 ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    style(ax6)
    labels = ["當前", "XGBoost", "KNN", "集成"]
    values = [current_p, pred["xgb_price"], pred["knn_price"], pred["ens_price"]]
    bar_colors = [BLUE, PUR, BLUE, GRN]
    if actual_price:
        labels.append("實際")
        values.append(actual_price)
        bar_colors.append(RED)
    bars = ax6.bar(labels, values, color=bar_colors, alpha=0.85,
                   edgecolor=SPINE, width=0.6)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 2,
                 f"{val:.0f}", ha="center", va="bottom",
                 color="white", fontsize=7)
    ax6.set_title("預測分解 (TWD)", color="white", fontsize=9)

    fig.suptitle(
        f"台股法說會估值  |  評估日 {as_at_dt.date()}  "
        f"|  預期報酬: {pred['ens_ret_pct']:+.2f}%",
        color="white", fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── 標題列 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #1e3a5f; padding-bottom:16px; margin-bottom:24px;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
               color:#3b82f6; letter-spacing:0.2em; text-transform:uppercase;">
    量化研究
  </span>
  <h1 style="margin:4px 0 0 0; font-size:1.6rem; font-weight:600; color:#e0e6f0;">
    台股法說會估值
  </h1>
  <p style="color:#4b5e7e; font-size:0.85rem; margin-top:6px;">
    XGBoost + KNN · 淨化推進式交叉驗證 · 時間點嚴格限制
  </p>
</div>
""", unsafe_allow_html=True)

# ── 側邊欄 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">參數設定</div>',
                unsafe_allow_html=True)

    ticker_input = st.text_input(
        "股票代碼 (Yahoo Finance 格式)",
        value="2330.TW",
        help="台股請加 .TW，例如 2330.TW、2317.TW、0050.TW"
    )

    as_at_date = st.date_input(
        "評估基準日 (時間點截止)",
        value=date(2024, 1, 18),
        min_value=date(2015, 1, 1),
        max_value=date.today(),
    )

    horizon = st.slider("預測天數 (交易日)", 5, 60, 20)

    train_years = st.slider("訓練歷史 (年)", 2, 10, 5)
    train_start = (pd.to_datetime(as_at_date) -
                   pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")

    validate = st.checkbox("執行 T+N 驗證 (取得實際股價)", value=True)

    st.markdown('<div class="section-header">模型設定</div>',
                unsafe_allow_html=True)
    xgb_weight   = st.slider("XGBoost 權重", 0.3, 0.9, 0.6, 0.05)
    n_neighbors  = st.slider("KNN 鄰居數", 5, 30, 10)
    buy_threshold = st.slider("買入訊號門檻 (%)", 1, 20, 5)

    st.markdown('<div class="section-header">關於</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#4b5e7e; line-height:1.6;">
    集成模型結合 XGBoost（財務特徵非線性關係）
    與 K 最近鄰（歷史型態比對）。<br><br>
    <span class="pit-badge">時間點嚴格</span> 無未來資料外洩。
    採用淨化推進式交叉驗證。
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  執行估值")

# ── 主內容 ─────────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding:80px 0; color:#1e3a5f;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem;">◈</div>
      <div style="font-size:0.9rem; margin-top:12px; color:#374151;">
        請在側邊欄設定參數後，按下 <strong>執行估值</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── 執行 ─────────────────────────────────────────────────────────────────────
as_at_dt   = pd.to_datetime(as_at_date)
future_end = (as_at_dt + timedelta(days=horizon * 3)).strftime("%Y-%m-%d")

with st.spinner(f"正在從 Yahoo Finance 下載 {ticker_input} 數據…"):
    try:
        train_raw = load_data(ticker_input, train_start,
                              as_at_date.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"❌ 資料下載失敗：{e}")
        st.stop()

if train_raw.empty or len(train_raw) < 100:
    st.error("❌ 資料不足，請嘗試延長訓練年限或更換股票代碼。")
    st.stop()

with st.spinner("計算技術指標特徵…"):
    feat_df    = compute_features(train_raw)
    train_feat = feat_df[feat_df.index <= as_at_dt]

with st.spinner("訓練集成模型（淨化推進式交叉驗證）…"):
    try:
        xgb_model, knn_model, scaler, fold_rmse, xgb_w, knn_w = train_ensemble(
            train_feat, n_splits=5, xgb_w=xgb_weight, n_neighbors=n_neighbors
        )
    except Exception as e:
        st.error(f"❌ 模型訓練失敗：{e}")
        st.stop()

last_row      = train_feat[FEATURE_COLS].dropna().iloc[-1]
current_price = float(train_raw["Close"].iloc[-1])
pred          = predict(xgb_model, knn_model, scaler,
                        current_price, last_row, xgb_w=xgb_w)

# 驗證：取得 T+N 實際股價
actual_price = None
if validate:
    with st.spinner(f"正在取得 T+{horizon} 實際股價以進行驗證…"):
        try:
            future_raw    = load_data(ticker_input,
                                      as_at_date.strftime("%Y-%m-%d"),
                                      future_end)
            future_closes = future_raw["Close"].iloc[1:]
            if len(future_closes) >= horizon:
                actual_price = float(future_closes.iloc[horizon - 1])
            elif len(future_closes) > 0:
                actual_price = float(future_closes.iloc[-1])
                st.info(f"ℹ 僅取得 {len(future_closes)} 個未來交易日資料，"
                        f"以最後收盤價作為驗證基準。")
        except Exception:
            st.warning("⚠ 無法取得未來股價，略過驗證。")

# ── KPI 卡片 ──────────────────────────────────────────────────────────────────
ret_cls = "up" if pred["ens_ret_pct"] > 0 else "down"
signal  = "買入 🟢" if pred["ens_ret_pct"] > buy_threshold else \
          ("賣出 🔴" if pred["ens_ret_pct"] < -buy_threshold else "持有 ⚪")

cols = st.columns(5)
kpis = [
    ("當前股價",        f"{current_price:.2f} TWD",    "neu"),
    ("集成目標價",      f"{pred['ens_price']:.2f} TWD", ret_cls),
    ("預期報酬",        f"{pred['ens_ret_pct']:+.2f}%", ret_cls),
    ("交易訊號",        signal,                          ret_cls),
    ("交叉驗證 RMSE",   f"{np.mean(fold_rmse):.4f}",    "neu"),
]
for col, (label, value, cls) in zip(cols, kpis):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ── 驗證結果 ──────────────────────────────────────────────────────────────────
if actual_price:
    err_pct = (pred["ens_price"] - actual_price) / actual_price * 100
    abs_err = abs(pred["ens_price"] - actual_price)
    ok      = abs(err_pct) < 5
    tag     = f'<span class="tag-success">✓ 誤差 &lt; 5%</span>' if ok else \
              f'<span class="tag-warn">⚠ 誤差 ≥ 5%</span>'
    err_cls = "up" if abs(err_pct) < 5 else "down"

    st.markdown('<div class="section-header">驗證結果</div>',
                unsafe_allow_html=True)
    vcols = st.columns(4)
    vkpis = [
        ("實際 T+" + str(horizon), f"{actual_price:.2f} TWD", "neu"),
        ("絕對誤差",               f"{abs_err:.2f} TWD",       err_cls),
        ("誤差率",                 f"{err_pct:+.3f}%",         err_cls),
        ("驗證結論",               tag,                         ""),
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

# ── 技術分析圖表 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">技術分析</div>',
            unsafe_allow_html=True)
fig = make_chart(train_feat, pred, as_at_dt, horizon, actual_price)
st.pyplot(fig, use_container_width=True)
plt.close()

# ── 交叉驗證詳情 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">推進式交叉驗證詳情</div>',
            unsafe_allow_html=True)
cv_cols = st.columns(len(fold_rmse))
for i, (col, rmse) in enumerate(zip(cv_cols, fold_rmse)):
    col.markdown(f"""
    <div class="metric-card" style="text-align:center;">
      <div class="metric-label">第 {i+1} 折</div>
      <div class="metric-value" style="font-size:1rem;">{rmse:.5f}</div>
    </div>""", unsafe_allow_html=True)

# ── 特徵快照 ───────────────────────────────────────────────────────────────────
with st.expander("特徵快照（評估基準日）"):
    snap = last_row.to_frame("數值").round(5)
    st.dataframe(snap, use_container_width=True)

# ── 原始數據 ───────────────────────────────────────────────────────────────────
with st.expander("原始價格數據（最後 60 筆）"):
    st.dataframe(train_raw.tail(60).round(2)[::-1],
                 use_container_width=True)

# ── 頁尾 ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1e3a5f; margin-top:40px; padding-top:16px;
            font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#1e3a5f;
            text-align:center;">
  台股法說會估值系統 · 時間點嚴格限制 · 淨化推進式交叉驗證 · 本系統僅供研究參考，非投資建議
</div>
""", unsafe_allow_html=True)
