# Working Handoff — Taiwan Stock Hybrid Valuation Model

> **Last updated:** 2026-05-02  
> **Repo:** https://github.com/emkcufslp-kfc/Taiwwan-stock-hybrid-model  
> **Cloudflare project:** `taiwwan-stock-hybrid-model` (auto-deploys on push to `main`)  
> **Live URL:** check Cloudflare dashboard — auto-deploys from the repo above  

---

## Current App State

The app is a **single-file static site** (`index.html`) deployed to Cloudflare Workers.  
It has two pages: a **Hero** (single-stock valuation via Groq AI) and a **Screener** (pre-computed batch of 100 TWSE stocks).

### What is working ✓

| Feature | Status |
|---|---|
| Screener loads 100 stocks from `data/manifest.json` | ✓ working |
| Tab badges (全部/買入/持有/賣出) update from manifest | ✓ working |
| Stats bar (批次基準日, avg return, counts) loads from manifest | ✓ working |
| `sc-date` screener date input syncs from `m.as_at_date` | ✓ fixed 2026-05-02 |
| Detail panel `dp-meta` uses `r.as_at` (not the editable input) | ✓ fixed 2026-05-02 |
| Hero page — single stock valuation via Groq AI + yfinance CORS proxies | ✓ working |
| Groq model fallback (`llama-3.3-70b-versatile` → `llama3-8b-8192`) | ✓ working |
| Cloudflare Workers static deploy via `wrangler.jsonc` | ✓ working |

### Data

- **`data/manifest.json`** — aggregate of all 100 stocks, pre-computed as at **2026-04-30**
  - `total=100`, `buy=8`, `hold=83`, `sell=9`, `avg_ret=1.74%`
  - Each stock entry has: `code`, `name`, `ind`, `cur`, `xgb`, `knn`, `ens`, `ret`, `sig`, `rmse`, `as_at`, `chart`
  - All 100 entries verified: no missing fields, signal counts match header, `as_at='2026-04-30'` consistent
- **`data/stock_detail/*.json`** — 100 individual stock files (e.g. `2330_TW.json`)
  - Raw field names: `ticker`, `current_price`, `ens_price`, `cv_rmse_mean`, `name`, `status`, `as_at_date`
  - These are used by `openDP()` indirectly through the manifest (manifest maps raw → display fields)

---

## Repo Structure

```
Taiwwan-stock-hybrid-model/
├── index.html          ← entire frontend (single file, ~1000 lines)
├── wrangler.jsonc      ← Cloudflare Workers config (static assets, SPA routing)
├── package.json        ← no-op build (prevents Cloudflare from running pip install)
├── _redirects          ← empty (SPA routing handled by wrangler.jsonc)
├── _headers            ← cache headers
├── data/
│   ├── manifest.json   ← 100-stock aggregate (what the screener reads)
│   └── stock_detail/   ← 100 individual *_TW.json files
├── app.py              ← Streamlit version (not used in production)
├── strategy.py         ← XGBoost+KNN model core
├── push.bat            ← git add/commit/push helper script
└── working.md          ← this file
```

---

## Cloudflare Deployment

**Deploy command (from repo root):**
```powershell
npx wrangler deploy
```
Or just push to `main` — Cloudflare auto-deploys.

**`wrangler.jsonc` key settings:**
```json
{
  "name": "taiwwan-stock-hybrid-model",
  "compatibility_date": "2026-05-01",
  "assets": {
    "directory": ".",
    "not_found_handling": "single-page-application",
    "exclude": [".git", ".gitignore", "__pycache__", "*.py", "*.yaml",
                "*.bat", "*.png", "*.log", "*.md", "requirements.txt", "data"]
  }
}
```

> **Note:** `data/` is excluded from Cloudflare assets. The screener data (`manifest.json`, `stock_detail/*.json`) is served from the repo directly via relative paths — this works locally but **will not work on Cloudflare** unless `data/` is removed from the exclude list.  
> **Action needed:** Remove `"data"` from the `exclude` list in `wrangler.jsonc` so `data/manifest.json` is served by Cloudflare.

---

## Running Locally

The app uses `fetch()` so it must be served over HTTP, not `file://`:

```powershell
cd "D:\Claude projects\TW hybrid model\台股法說會估值\Taiwwan-stock-hybrid-model"
python -m http.server 8080
```
Then open: `http://localhost:8080/index.html`

**Groq API key** is hardcoded in `index.html` (search for `gsk_`). For local use this is fine; for production consider moving to an environment variable if security is a concern.

---

## Git Workflow

**Known issue — git index.lock:**  
If a previous session left a stale lock, you will see `fatal: Unable to create '.git/index.lock'`. Fix:
```powershell
del "D:\Claude projects\TW hybrid model\台股法說會估值\Taiwwan-stock-hybrid-model\.git\index.lock"
```

**Commit and push:**
```powershell
cd "D:\Claude projects\TW hybrid model\台股法說會估值\Taiwwan-stock-hybrid-model"
git add -A
git commit -m "describe your change"
git push origin main
```

Or use the helper: `push.bat`

---

## Recent Commits (as of 2026-05-02)

| Hash | Message |
|---|---|
| `2f32116` | fix: update tab badge counts dynamically from manifest + add package.json |
| `6993d72` | fix: load screener from manifest.json (100 real stocks) + dynamic stats |
| `f046770` | fix: groq model fallback llama-3.3-70b → llama3-8b-8192 on rate limit |
| `ae846c2` | fix: add package.json to skip Python auto-detection on Cloudflare |
| `3c72a01` | fix: remove _redirects loop + exclude .git and src files from assets |
| `c688581` | fix: add wrangler.jsonc for Cloudflare static deploy + gitignore |

**Uncommitted local changes (needs push):**
- `index.html` — two fixes applied 2026-05-02:
  1. `loadManifest()` now sets `sc-date` input value from `m.as_at_date`
  2. `openDP()` now uses `r.as_at` (per-stock data) for `dp-meta` 基準日 label, not the editable input
- `working.md` — this file (new)

---

## Data Integrity Verification (2026-05-02)

Ran full check on manifest and stock_detail files:

```
manifest.json:
  total=100, buy=8, hold=83, sell=9 ✓ (counts match stock list)
  as_at_date = 2026-04-30 (all 100 stock entries consistent) ✓
  All required fields present in every stock entry ✓
  No null/missing values ✓

stock_detail/*.json:
  100 files ✓
  All have as_at_date=2026-04-30 ✓
  Field names: ticker, current_price, ens_price, cv_rmse_mean, name, status
  (these are raw names; manifest maps them to code, cur, ens, rmse etc.)
```

---

## Next Tasks

1. **[ ] Fix Cloudflare data serving** — remove `"data"` from `exclude` list in `wrangler.jsonc` so `data/manifest.json` and `data/stock_detail/*.json` are served on the live URL. Without this, the screener works locally but shows fallback data (1 stock) on Cloudflare.

2. **[ ] Commit and push today's fixes** — the two `index.html` fixes and this `working.md` are not yet committed:
   ```powershell
   git add index.html working.md
   git commit -m "fix: as_at_date consistency + add working.md handoff"
   git push origin main
   ```

3. **[ ] Refresh pre-computed data** — `data/` is as at 2026-04-30. When you want fresh data, re-run the batch valuation pipeline (Python, see `strategy.py`) and regenerate `manifest.json` + `stock_detail/*.json`, then commit the new `data/` folder.

4. **[ ] Screener date input** — currently the screener has a date picker (`sc-date`) that is now auto-set to `m.as_at_date`. If you want the screener to be fully dynamic (re-run valuation for any chosen date), the screener would need to call the Groq API per stock — this is architecturally complex. For now the pre-computed batch approach is the right choice.

5. **[ ] Hero page UX** — the Hero page is functional but the chart in the detail result panel uses simulated historical prices (random walk from `r.cur`). A future improvement would be to fetch real 6-month OHLCV data from yfinance for the chart.

---

## Key Code Sections in index.html

| Function | Line range | Purpose |
|---|---|---|
| `loadManifest()` | ~880–910 | Fetches `data/manifest.json`, populates stats bar, tab badges, sets `sc-date`, calls `renderTbl()` |
| `renderTbl()` | ~912–938 | Renders screener table rows from `DATA[]`, respects `curSig` filter |
| `openDP(code)` | ~941–968 | Opens stock detail panel from `DATA`, draws mini price chart |
| `showPage(p)` | ~970–977 | Switches between hero/screener pages, calls `loadManifest()` once on first screener visit |
| `runModel()` | ~695–860 | Hero page: fetches Yahoo Finance prices, calls Groq AI, renders result |
| `fetchYahooRaw()` | ~670–694 | Tries 3 CORS proxies in sequence for Yahoo Finance data |

---

## Architecture Notes

- **No build step.** `index.html` is self-contained — all CSS, JS, HTML in one file.
- **Groq API** called client-side from the browser. Model: `llama-3.3-70b-versatile` (fallback: `llama3-8b-8192`).
- **Yahoo Finance** accessed via CORS proxy (corsproxy.io → allorigins.win → cors-anywhere).
- **FinMind** is referenced in `README.md` as a future data source but is **not connected** — the pipeline uses yfinance only.
- **Screener data** is pre-computed (static JSON files), not live. This is intentional for performance and to avoid rate limits.
