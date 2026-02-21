"""
VCP + Ascending Triangle Scanner — NSE Stocks
Runs daily via GitHub Actions → saves results to docs/data/results.json
Dashboard at GitHub Pages reads this JSON automatically.
"""

import json
import time
import warnings
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION — tune these to get more/fewer results
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "top_n":                  10,     # How many results to save
    "base_period_days":       40,     # Consolidation window
    "max_price_range_pct":    16.0,   # Max % range during base
    "resistance_band_pct":    1.8,    # Tolerance for flat resistance
    "resistance_touch_count": 2,      # Min touches at resistance
    "volume_dryup_ratio":     0.70,   # Base vol < X × 50-day avg
    "near_pivot_pct":         6.0,    # Within % of breakout pivot
    "min_stock_price":        30,     # Filter very cheap stocks
    "min_avg_volume":         150_000,# Min daily volume (liquidity)
    "uptrend_ma_check":       True,   # Require Stage 2 uptrend
    "data_period":            "7mo",  # Data window to fetch
    "candle_history":         45,     # Candles to save per stock
    "rs_days":                63,     # RS lookback (~3 months)
    "sleep_between":          0.15,   # Seconds between API calls
}

# ─────────────────────────────────────────────────────────────────
#  NSE STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────────
NSE_STOCKS = [
    # NIFTY 50
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","LT","KOTAKBANK","HCLTECH","BAJFINANCE","AXISBANK",
    "ASIANPAINT","MARUTI","SUNPHARMA","TITAN","WIPRO","ULTRACEMCO",
    "NESTLEIND","POWERGRID","NTPC","JSWSTEEL","TATASTEEL","TECHM","HINDALCO",
    "INDUSINDBK","BAJAJFINSV","ONGC","COALINDIA","CIPLA","DRREDDY","EICHERMOT",
    "APOLLOHOSP","DIVISLAB","HEROMOTOCO","GRASIM","TATACONSUM","BRITANNIA",
    "BPCL","TATAMOTORS","M&M","ADANIPORTS","SHRIRAMFIN","BAJAJ-AUTO",
    "HDFCLIFE","SBILIFE","TRENT","ADANIENT",
    # Midcap / NIFTY Next 50
    "TATACAP","MUTHOOTFIN","PERSISTENT","COFORGE","LTIM","MPHASIS",
    "PIIND","LAURUSLABS","ALKEM","TORNTPHARM","AUROPHARMA","IPCA","ZYDUSLIFE",
    "GLAND","METROPOLIS","DIXON","AMBER","VOLTAS","BLUESTARCO","KAJARIACER",
    "ASTRAL","SUPREMEIND","POLYCAB","HAVELLS","VGUARD","KEI","PAGEIND",
    "MANYAVAR","VEDL","NATIONALUM","HINDZINC","CONCOR","IRFC","RVNL","CAMS",
    "CDSL","ANGELONE","MOTILALOFS","ICICIGI","HDFCAMC","NAUKRI","ZOMATO",
    "IRCTC","INDIGO","FORTIS","KIMS","NH","SYNGENE","DIVI","SUDARSCHEM",
    "NAVINFLUOR","SRF","AARTI","DEEPAKFERT","CHAMBAL","COROMANDEL","GNFC",
    "OLECTRA","ABB","SIEMENS","CUMMINSIND","THERMAX","BHEL","BEL","HAL",
    "GRSE","MAZAGON","HUDCO","PNBHOUSING","LICHSGFIN","OBEROIRLTY","DLF",
    "GODREJPROP","PRESTIGE","BRIGADE","SOBHA","PHOENIXLTD","TATAPOWER",
    "TORNTPOWER","ADANIGREEN","JSL","RATNAMANI","MAHINDCIE","ENDURANCE",
    "MOTHERSON","BALKRISIND","APOLLOTYRE","CEAT","MRF","EXIDEIND","TIINDIA",
    "SCHAEFFLER","SKF","JUSTDIAL","MAPMYINDIA","FIVESTAR","UGROCAP",
    "MAXHEALTH","RAINBOW","MEDANTA","GREENPANEL","CENTURYPLY","POWERINDIA",
]

def fetch_benchmark():
    try:
        df = yf.download("^NSEI", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        return df["Close"].squeeze()
    except Exception:
        return None

def fetch_stock(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 70:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        return df.dropna()
    except Exception:
        return None

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def check_stage2(df):
    if len(df) < 200:
        return False, {}
    c = df["close"]
    m50, m150, m200 = (c.rolling(w).mean().iloc[-1] for w in (50, 150, 200))
    price = c.iloc[-1]
    return price > m50 > m150 > m200, {
        "ma50": round(float(m50), 2),
        "ma150": round(float(m150), 2),
        "ma200": round(float(m200), 2),
    }

def rising_lows_count(df_base):
    seg = len(df_base) // 3
    if seg < 3:
        return 0
    lows = [df_base["low"].iloc[i*seg:(i+1)*seg].min() for i in range(3)]
    return sum(lows[i] > lows[i-1] for i in range(1, 3))

def calc_rs(stock_close, bench_close, days=63):
    try:
        s = float(stock_close.iloc[-1]) / float(stock_close.iloc[-days]) - 1
        b = float(bench_close.iloc[-1]) / float(bench_close.iloc[-days]) - 1
        return round((s - b) * 100, 2)
    except Exception:
        return None

def scan_stock(df, symbol):
    base_days = CONFIG["base_period_days"]
    if len(df) < base_days + 50:
        return None

    base = df.iloc[-base_days:]
    prev = df.iloc[-(base_days + 50):-base_days]

    price      = float(df["close"].iloc[-1])
    avg_vol50  = float(df["volume"].iloc[-50:].mean())

    # Liquidity & price floor
    if price < CONFIG["min_stock_price"]:
        return None
    if avg_vol50 < CONFIG["min_avg_volume"]:
        return None

    # Volatility contraction
    atr_base = atr(base)
    atr_prev = atr(prev) if len(prev) > 14 else atr_base
    contraction = atr_base / atr_prev if atr_prev > 0 else 1.0

    # Base range
    base_range_pct = ((float(base["high"].max()) - float(base["low"].min()))
                      / float(base["low"].min())) * 100
    if base_range_pct > CONFIG["max_price_range_pct"]:
        return None

    # Flat resistance
    pivot = float(base["high"].max())
    band  = pivot * (CONFIG["resistance_band_pct"] / 100)
    touches = int((base["high"] >= pivot - band).sum())
    if touches < CONFIG["resistance_touch_count"]:
        return None

    # Rising lows
    rl = rising_lows_count(base)

    # Volume dry-up
    vol_ratio = float(base["volume"].mean()) / avg_vol50
    vol_dryup = vol_ratio < CONFIG["volume_dryup_ratio"]

    # Near pivot
    dist_pct = ((pivot - price) / price) * 100
    near = dist_pct <= CONFIG["near_pivot_pct"]

    if not (near and vol_dryup and rl >= 1 and contraction < 0.97):
        return None

    # Score
    score = 0
    score += min(30, int((1 - contraction) * 70))
    score += 25 if touches >= 3 else 15
    score += 20 if rl >= 2 else 10
    score += 15 if vol_dryup else 0
    score += 10 if near else 5

    # Segment lows for stop
    seg = base_days // 3
    seg_lows = [float(df["low"].iloc[-base_days + i*seg: -base_days + (i+1)*seg].min())
                for i in range(3)]
    stop = round(min(seg_lows), 2)

    return {
        "symbol":          symbol,
        "price":           round(price, 2),
        "pivot":           round(pivot, 2),
        "dist_pct":        round(dist_pct, 2),
        "stop_loss":       stop,
        "target1":         round(pivot * 1.08, 2),
        "target2":         round(pivot * 1.15, 2),
        "risk_reward":     round((pivot * 1.08 - price) / max(price - stop, 0.01), 2),
        "vcp_score":       score,
        "vol_ratio":       round(vol_ratio, 3),
        "contraction":     round(contraction, 3),
        "resistance_touches": touches,
        "rising_lows":     rl,
        "base_range_pct":  round(base_range_pct, 2),
    }

def extract_candles(df, n=45):
    """Extract last N candles as list of [o,h,l,c,v] for the dashboard chart."""
    recent = df.iloc[-n:].copy()
    candles = []
    for _, row in recent.iterrows():
        candles.append([
            round(float(row["open"]),  2),
            round(float(row["high"]),  2),
            round(float(row["low"]),   2),
            round(float(row["close"]), 2),
            int(row["volume"]),
        ])
    return candles

def run():
    print("=" * 60)
    print("  VCP SCANNER — NSE  |", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 60)

    print("  Fetching NIFTY benchmark…")
    benchmark = fetch_benchmark()

    results   = []
    skipped   = 0

    for sym in tqdm(NSE_STOCKS, desc="  Scanning", ncols=60):
        try:
            df = fetch_stock(sym)
            if df is None:
                skipped += 1
                continue

            if CONFIG["uptrend_ma_check"]:
                stage2, ma_data = check_stage2(df)
                if not stage2:
                    continue
            else:
                _, ma_data = check_stage2(df)

            hit = scan_stock(df, sym)
            if hit is None:
                continue

            # Add RS
            if benchmark is not None:
                hit["rs_vs_nifty"] = calc_rs(df["close"], benchmark,
                                              CONFIG["rs_days"])
            else:
                hit["rs_vs_nifty"] = None

            hit["ma50"]    = ma_data.get("ma50")
            hit["ma150"]   = ma_data.get("ma150")
            hit["ma200"]   = ma_data.get("ma200")
            hit["candles"] = extract_candles(df, CONFIG["candle_history"])

            results.append(hit)
            time.sleep(CONFIG["sleep_between"])

        except Exception as e:
            skipped += 1
            continue

    results.sort(key=lambda x: x["vcp_score"], reverse=True)
    top = results[:CONFIG["top_n"]]

    # ── Write JSON ──────────────────────────────────────────────
    out = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "market_date":   datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_scanned": len(NSE_STOCKS),
        "total_found":   len(results),
        "stocks":        top,
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "docs", "data", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  ✅  {len(results)} candidates found | Top {len(top)} saved")
    print(f"  📁  → docs/data/results.json")
    print()

    for i, s in enumerate(top, 1):
        rs = f"{s['rs_vs_nifty']:+.1f}%" if s['rs_vs_nifty'] else "N/A"
        print(f"  {i:>2}. {s['symbol']:<14} "
              f"₹{s['price']:>8.2f}  "
              f"Pivot ₹{s['pivot']:>8.2f}  "
              f"Score {s['vcp_score']:>3}  "
              f"RS {rs}")

    print("=" * 60)

if __name__ == "__main__":
    run()
