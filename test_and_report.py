"""
test_and_report.py
Runs the TWSE Ensemble Valuation System end-to-end:
  1. Validation test   (2330.TW, As At 2024-01-18)
  2. Backtrader run
  3. Generates validation + equity curve charts
  4. Prints a full markdown-style report
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, "/home/claude/twse_ensemble")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from strategy import run_validation, run_backtest, compute_technical_features

# ─── CONFIG ──────────────────────────────────
TICKER       = "2330.TW"   # TSMC on Yahoo Finance
AS_AT_DATE   = "2024-01-18"
TRAIN_START  = "2019-01-01"
HORIZON_DAYS = 20
# ─────────────────────────────────────────────


def print_report(val_result: dict, bt_result: dict):
    r  = val_result
    br = bt_result
    vr = br.get("valuation_result", {})

    print("\n" + "="*60)
    print("  📊 TWSE ENSEMBLE VALUATION SYSTEM — FULL REPORT")
    print("="*60)
    print(f"\n  ▶ 標的 (Ticker)        : {TICKER}")
    print(f"  ▶ 評估基準日 (As At)   : {AS_AT_DATE}")
    print(f"  ▶ 預測時間區間         : T+{HORIZON_DAYS} 交易日")
    print(f"  ▶ 訓練資料行數         : {r['training_rows']} bars")

    print(f"\n{'─'*60}")
    print("  ⚙️  特徵工程 & 估值結果")
    print(f"{'─'*60}")
    print(f"  當前股價 (As At Date) : {r['current_price']:.2f} TWD")
    print(f"  XGBoost 目標價        : {r['xgb_target']:.2f} TWD")
    print(f"  KNN 近鄰調整目標價    : {r['knn_target']:.2f} TWD")
    print(f"  💎 混合模型目標價     : {r['ensemble_target']:.2f} TWD")
    print(f"  預期報酬率            : {r['expected_ret_pct']:+.3f}%")

    print(f"\n{'─'*60}")
    print("  🔍 歷史回測驗證 (Validation)")
    print(f"{'─'*60}")
    print(f"  T+{HORIZON_DAYS} 實際收盤價          : {r['actual_price']:.2f} TWD")
    print(f"  絕對誤差 (Abs Error)  : {r['abs_error']:.2f} TWD")
    print(f"  預測誤差率 (%)        : {r['error_pct']:+.3f}%")
    print(f"  Walk-Forward CV RMSE  : {r['cv_rmse_avg']}")
    print(f"  各折 RMSE             : {r['fold_rmse_list']}")

    accuracy_ok = abs(r["error_pct"]) < 5.0
    print(f"\n  結論: {'✅ 預測成功 — 誤差率在 5% 以內' if accuracy_ok else '⚠️  誤差率超過 5%，模型需調整'}")

    print(f"\n{'─'*60}")
    print("  🏦 Backtrader 回測結果")
    print(f"{'─'*60}")
    print(f"  起始資金              : 1,000,000 TWD")
    print(f"  最終組合價值          : {br['final_portfolio_value']:,.2f} TWD")
    print(f"  總報酬率              : {br['total_return_pct']:+.3f}%")
    print(f"  夏普比率 (Sharpe)     : {br['sharpe_ratio']}")
    print(f"  最大回撤 (Max DD)     : {br['max_drawdown']}")

    print("\n" + "="*60 + "\n")


def plot_report(val_result, train_feat, future_raw):
    fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    title_kw  = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    label_kw  = dict(color="#aaaaaa", fontsize=8)
    tick_kw   = dict(colors="#888888", labelsize=7)
    spine_col = "#333333"

    def style_ax(ax):
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_color(spine_col)
        ax.tick_params(axis="both", **tick_kw)
        ax.yaxis.label.set_color("#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")

    # ── 1. Price history + prediction band ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)

    hist = train_feat.dropna(subset=["Close"]).tail(252)
    as_at_dt = pd.to_datetime(AS_AT_DATE)

    ax1.plot(hist.index, hist["Close"], color="#4fc3f7", lw=1.2, label="Close Price")

    # Mark as_at_date
    ax1.axvline(as_at_dt, color="#ff9800", lw=1.5, ls="--", label="As At Date")

    # Prediction marker
    pred_price  = val_result["ensemble_target"]
    actual_price = val_result["actual_price"]
    current_p   = val_result["current_price"]

    # Approx T+20 date from future_raw
    if len(future_raw) >= HORIZON_DAYS:
        t20_date = future_raw.index[HORIZON_DAYS]
    else:
        t20_date = future_raw.index[-1]

    # Connect current → predicted
    ax1.plot([as_at_dt, t20_date], [current_p, pred_price],
             color="#69f0ae", lw=1.5, ls=":", label=f"Predicted T+{HORIZON_DAYS}: {pred_price:.0f}")
    ax1.plot([as_at_dt, t20_date], [current_p, actual_price],
             color="#ff5252", lw=1.5, ls=":", label=f"Actual T+{HORIZON_DAYS}: {actual_price:.0f}")

    ax1.scatter([t20_date], [pred_price],  color="#69f0ae", zorder=5, s=60)
    ax1.scatter([t20_date], [actual_price], color="#ff5252",  zorder=5, s=60)
    ax1.scatter([as_at_dt], [current_p],   color="#ff9800",  zorder=5, s=60)

    ax1.set_title("TSMC (2330.TW) — Price History & T+20 Prediction vs Actual",
                  **title_kw)
    ax1.set_ylabel("Price (TWD)", **label_kw)
    ax1.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
               edgecolor=spine_col)

    # ── 2. Walk-forward CV RMSE by fold ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    folds = val_result["fold_rmse_list"]
    bars  = ax2.bar(range(1, len(folds)+1), folds, color="#7c4dff", alpha=0.85,
                    edgecolor=spine_col)
    ax2.axhline(val_result["cv_rmse_avg"], color="#ff9800", lw=1.2, ls="--",
                label=f"Avg: {val_result['cv_rmse_avg']:.4f}")
    ax2.set_title("Walk-Forward CV RMSE by Fold", **title_kw)
    ax2.set_xlabel("Fold", **label_kw)
    ax2.set_ylabel("RMSE (return units)", **label_kw)
    ax2.legend(fontsize=7, facecolor="#1a1d27", labelcolor="white",
               edgecolor=spine_col)

    # ── 3. Prediction summary bar ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    labels = ["Current\nPrice", "XGBoost\nTarget", "KNN\nTarget",
              "Ensemble\nTarget", "Actual\nT+20"]
    values = [val_result["current_price"], val_result["xgb_target"],
              val_result["knn_target"],    val_result["ensemble_target"],
              val_result["actual_price"]]
    colors = ["#4fc3f7", "#69f0ae", "#b2ff59", "#ff9800", "#ff5252"]
    ax3.bar(labels, values, color=colors, alpha=0.85, edgecolor=spine_col)
    ax3.set_title("Price Prediction Breakdown (TWD)", **title_kw)
    ax3.set_ylabel("Price (TWD)", **label_kw)
    for i, (v, c) in enumerate(zip(values, colors)):
        ax3.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom",
                 color=c, fontsize=8, fontweight="bold")

    # ── 4. RSI ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4)
    rsi_data = hist["rsi14"].dropna().tail(120)
    ax4.plot(rsi_data.index, rsi_data, color="#ce93d8", lw=1)
    ax4.axhline(70, color="#ff5252", lw=0.8, ls="--", alpha=0.7)
    ax4.axhline(30, color="#69f0ae", lw=0.8, ls="--", alpha=0.7)
    ax4.axvline(as_at_dt, color="#ff9800", lw=1.2, ls="--")
    ax4.fill_between(rsi_data.index, 30, rsi_data,
                     where=rsi_data < 30, alpha=0.25, color="#69f0ae")
    ax4.fill_between(rsi_data.index, 70, rsi_data,
                     where=rsi_data > 70, alpha=0.25, color="#ff5252")
    ax4.set_title("RSI(14) — Momentum Indicator", **title_kw)
    ax4.set_ylabel("RSI", **label_kw)
    ax4.set_ylim(0, 100)

    # ── 5. Volume ratio ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5)
    vr_data = hist["vol_ratio"].dropna().tail(120)
    ax5.bar(vr_data.index, vr_data, color="#4fc3f7", alpha=0.6, width=1)
    ax5.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax5.axvline(as_at_dt, color="#ff9800", lw=1.2, ls="--")
    ax5.set_title("Volume Ratio (vs 20-day avg)", **title_kw)
    ax5.set_ylabel("Ratio", **label_kw)

    # ── Main title ───────────────────────────────────────────────────
    fig.suptitle(
        f"TWSE Ensemble Valuation System  |  {TICKER}  |  As At {AS_AT_DATE}  "
        f"|  Error: {val_result['error_pct']:+.3f}%",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig("/mnt/user-data/outputs/twse_valuation_report.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  📈 Chart saved → twse_valuation_report.png")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "━"*55)
    print("  STEP 1: Running Validation Module")
    print("━"*55)
    val_result, train_feat, future_raw, model = run_validation(
        ticker_tw=TICKER,
        as_at_date=AS_AT_DATE,
        horizon_days=HORIZON_DAYS,
    )

    print("\n" + "━"*55)
    print("  STEP 2: Running Backtrader Backtest")
    print("━"*55)
    bt_result, raw, strat = run_backtest(
        ticker_tw=TICKER,
        as_at_date=AS_AT_DATE,
        start_date=TRAIN_START,
    )

    print("\n" + "━"*55)
    print("  STEP 3: Generating Charts & Report")
    print("━"*55)
    plot_report(val_result, train_feat, future_raw)
    print_report(val_result, bt_result)
