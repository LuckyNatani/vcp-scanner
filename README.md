# 📈 VCP Scanner — NSE Swing Trade Dashboard

Auto-scans NSE stocks daily for **VCP + Ascending Triangle** setups.  
Results published to a free web dashboard accessible on any device.

**Stack:** GitHub Actions (scanner) → JSON → GitHub Pages (dashboard)  
**Cost:** $0 forever

---

## 🚀 Deploy in 5 Steps

### 1. Fork or create this repo
Click **Use this template** or fork this repo to your GitHub account.

### 2. Enable GitHub Pages
```
Settings → Pages → Source: Deploy from branch → Branch: main → Folder: /docs → Save
```
Your dashboard URL will be: `https://YOUR-USERNAME.github.io/vcp-scanner/`

### 3. Allow Actions to write to the repo
```
Settings → Actions → General → Workflow permissions → Read and write permissions ✓
```

### 4. Run the first scan manually
```
Actions tab → "VCP Daily Scanner" → Run workflow → Run workflow
```
Wait ~3–5 minutes. Refresh your GitHub Pages URL.

### 5. Done! 🎉
The scanner runs automatically at **6:30 AM IST every weekday**.  
You can also trigger it anytime from the Actions tab.

---

## 📁 Project Structure

```
vcp-scanner/
├── .github/
│   └── workflows/
│       └── scan.yml          ← GitHub Actions cron job (6:30 AM IST weekdays)
├── scanner/
│   ├── vcp_scanner.py        ← Pattern detection engine
│   └── requirements.txt      ← Python dependencies
├── docs/
│   ├── index.html            ← Dashboard (served by GitHub Pages)
│   └── data/
│       └── results.json      ← Updated daily by the scanner ← auto-committed
└── README.md
```

---

## 📊 What the Scanner Finds

The scanner looks for stocks forming a **VCP (Volatility Contraction Pattern)** combined with an **Ascending Triangle** — the exact pattern visible in TATACAP (see screenshot):

| Criterion | What it means |
|-----------|--------------|
| **Stage 2 Uptrend** | Price > 50MA > 150MA > 200MA |
| **Flat Resistance** | 2+ touches of the same ceiling (±1.8%) |
| **Rising Lows** | Each swing low is higher than the last |
| **Volume Dry-up** | Base volume < 70% of 50-day average |
| **Tight Base** | Price range < 16% during consolidation |
| **Near Pivot** | Price within 6% of the breakout level |
| **ATR Contraction** | Range shrinking vs prior period |

Each stock gets a **VCP Score (0–100)**. Top 10 are displayed on the dashboard.

---

## ⚙️ Customize the Scanner

Edit `scanner/vcp_scanner.py` → `CONFIG` section:

```python
CONFIG = {
    "top_n":                10,    # How many results to show
    "base_period_days":     40,    # Base length to analyze
    "volume_dryup_ratio":  0.70,   # Stricter = 0.60, looser = 0.80
    "near_pivot_pct":       6.0,   # Max % below pivot to qualify
    "min_avg_volume":   150_000,   # Liquidity filter
}
```

To add more stocks, edit the `NSE_STOCKS` list in the scanner.

---

## 📱 Dashboard Features

- **Candlestick chart** — 45-day base with pivot line
- **Entry / Stop / Target levels** auto-calculated
- **Risk:Reward ratio** displayed per setup
- **RS vs NIFTY** relative strength bar
- **Pattern indicators** — touches, rising lows, vol dry-up, ATR
- Mobile-responsive — works on phone and desktop

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only.  
It does not constitute financial advice. Always do your own research  
and use proper risk management before placing any trade.

---

## 🔧 Troubleshooting

**No results showing?**  
→ Run the workflow manually (Actions → Run workflow) first.

**Scanner finds 0 stocks?**  
→ Try relaxing CONFIG: increase `volume_dryup_ratio` to 0.80 or `near_pivot_pct` to 8.0.

**GitHub Actions failing?**  
→ Check Settings → Actions → General → Workflow permissions → must be Read and Write.

**yfinance rate limiting?**  
→ Increase `sleep_between` in CONFIG to 0.3.
