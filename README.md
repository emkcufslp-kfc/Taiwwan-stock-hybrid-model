# TWSE Ensemble Valuation System

XGBoost + KNN ensemble model for Taiwan stock valuation with Purged Walk-Forward CV.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (main UI) |
| `strategy.py` | Core model + Backtrader strategy |
| `synthetic_data.py` | Offline fallback data (dev only) |
| `test_and_report.py` | CLI test runner |
| `config.yaml` | Backtest configuration |
| `requirements.txt` | Python dependencies |

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → set **Main file** to `app.py`
4. Click **Deploy** — done

Streamlit Cloud has full internet access so `yfinance` will fetch real
TWSE data automatically.

## Usage

| Parameter | Description |
|-----------|-------------|
| Stock Code | Yahoo Finance format: `2330.TW`, `2317.TW`, `0050.TW` |
| As At Date | Point-in-time cutoff — no data after this date used in training |
| Prediction Horizon | T+N trading days to forecast |
| Training History | How many years of history to train on |
| Validate | Fetch actual T+N price and compute error % |

## Model Architecture

```
Features (11 technical indicators)
        │
   ┌────┴────┐
   │         │
XGBoost    KNN
(weight    (weight
  0.6)       0.4)
   │         │
   └────┬────┘
        │
   Ensemble Price
```

**Features used:**
- `ret_5d`, `ret_20d`, `ret_60d` — momentum returns
- `ma5_ratio`, `ma20_ratio`, `ma60_ratio` — MA momentum
- `vol_20d`, `vol_60d` — realized volatility
- `vol_ratio` — volume anomaly
- `rsi14` — RSI momentum
- `bb_pos` — Bollinger Band position

**Point-in-Time enforcement:**
All training data is strictly filtered to `≤ As At Date`.
Purged Walk-Forward CV with 5 folds and 5-bar embargo prevents leakage.

## Extending with FinMind

To add institutional investor flow data (3大法人), replace the data
fetch in `strategy.py` with:

```python
import requests

def fetch_finmind(dataset, ticker, start, end, token):
    r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
        "dataset": dataset,
        "data_id": ticker,
        "start_date": start,
        "end_date": end,
        "token": token,
    })
    return pd.DataFrame(r.json()["data"])

# Usage:
# fetch_finmind("TaiwanStockMonthRevenue", "2330", "2023-01-01", "2024-01-18", TOKEN)
# fetch_finmind("TaiwanStockInstitutionalInvestorsBuySell", "2330", ..., TOKEN)
```

Register for a free token at https://finmindtrade.com

## Disclaimer

Not financial advice. For research and educational purposes only.
