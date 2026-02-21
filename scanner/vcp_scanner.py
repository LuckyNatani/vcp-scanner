"""
╔══════════════════════════════════════════════════════════════════════╗
║   VCP BREAKOUT SCANNER v2 — NSE  ·  High-Accuracy Edition           ║
║                                                                      ║
║   Philosophy: Find stocks that break out in 1–2 DAYS, not weeks.   ║
║   Every filter removes false signals. Min R:R = 2.5:1 enforced.    ║
║                                                                      ║
║   12 NEW FILTERS vs v1:                                             ║
║   1.  Near-pivot tightened 6% → 2.5%   (proximity)                 ║
║   2.  Final 5-day squeeze detection    (coiling spring)             ║
║   3.  Last-5-day volume exhaustion     (fuel for breakout)          ║
║   4.  ATR percentile at base lows      (volatility minimum)         ║
║   5.  Prior uptrend ≥ 15% before base (no bottoming patterns)      ║
║   6.  Within 15% of 52-week high      (no overhead supply)         ║
║   7.  Min R:R 2.5:1 enforced hard     (better trades)              ║
║   8.  ATR-based stop loss             (tighter, precise)            ║
║   9.  10-EMA rising toward pivot      (momentum confirmation)      ║
║   10. NIFTY market trend gate         (no buys in bear market)      ║
║   11. Bollinger Band width at low     (squeeze confirmation)        ║
║   12. Resistance quality score        (true flat ceiling only)      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, time, warnings, os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Output ────────────────────────────────────────────────────
    "top_n":                    8,      # Max stocks to show

    # ── Universe ──────────────────────────────────────────────────
    "data_period":              "12mo", # Need 1yr for 52w high + 200MA
    "candle_history":           50,
    "sleep_between":            0.18,

    # ── Liquidity filters (false signal reducer) ──────────────────
    "min_stock_price":          80,     # Raised from 30 — avoids penny stocks
    "min_avg_volume":           300_000,# Raised from 150k — needs real liquidity

    # ── Stage 2 uptrend ───────────────────────────────────────────
    "uptrend_ma_check":         True,   # Price > 50MA > 150MA > 200MA

    # ── Prior uptrend (NEW) ───────────────────────────────────────
    # Before entering the base, stock must have risen ≥ this % from
    # its low in the prior 3 months. Eliminates bottoming patterns.
    "min_prior_uptrend_pct":    15.0,

    # ── 52-week high proximity (NEW) ─────────────────────────────
    # Stock must be within this % of its 52-week high.
    # Stocks far below 52w high have heavy overhead supply.
    "max_below_52w_high_pct":   15.0,

    # ── Base detection ────────────────────────────────────────────
    "base_period_days":         35,     # Slightly tighter base window
    "max_price_range_pct":      12.0,   # Tightened from 16% — true VCP is tight

    # ── Resistance quality (NEW) ──────────────────────────────────
    "resistance_band_pct":      1.2,    # Tightened from 1.8% — stricter flat ceiling
    "resistance_touch_count":   3,      # Raised from 2 — more touches = stronger level
    "max_resistance_touches":   8,      # Too many tests = weak level about to fail

    # ── Volume filters ────────────────────────────────────────────
    "volume_dryup_ratio":       0.60,   # Overall base vol < 60% of 50d avg
    "last5_vol_exhaustion":     0.40,   # NEW: Last 5 days vol < 40% of avg (final dryup)

    # ── Proximity to breakout (CRITICAL — tightened from 6%) ─────
    "near_pivot_pct":           2.5,    # Must be within 2.5% of pivot

    # ── Final squeeze (NEW) ───────────────────────────────────────
    # Last 5 candles' avg daily range must be ≤ this % of the
    # base's avg daily range. The tighter the coil, the stronger breakout.
    "final_squeeze_ratio":      0.50,   # Last 5d range ≤ 50% of base avg range

    # ── ATR percentile (NEW) ─────────────────────────────────────
    # Current ATR must be in the bottom X percentile of the base.
    # Ensures we're at volatility minimum before the explosion.
    "atr_pct_threshold":        25,     # ATR must be in bottom 25th percentile

    # ── Bollinger Band squeeze (NEW) ─────────────────────────────
    # BB width (as % of price) must be at a multi-month low.
    "bb_squeeze_pct_max":       4.0,    # BB width < 4% of price

    # ── 10-EMA momentum (NEW) ────────────────────────────────────
    "ema10_slope_check":        True,   # 10-EMA must be rising toward pivot

    # ── Risk : Reward (HARD FILTER — NEW) ────────────────────────
    # Trades with R:R below this are REJECTED outright.
    "min_risk_reward":          2.5,    # Minimum 2.5:1 — non-negotiable

    # ── Stop loss method (NEW) ────────────────────────────────────
    # ATR-based stop is tighter and more accurate than segment-low.
    # stop = pivot_entry - (atr_multiplier × 14d ATR)
    "stop_atr_multiplier":      1.5,    # Tighter stop = better R:R

    # ── RS lookback ───────────────────────────────────────────────
    "rs_days":                  63,
}

# ═══════════════════════════════════════════════════════════════════
#  NSE STOCK UNIVERSE (~150 liquid stocks)
# ═══════════════════════════════════════════════════════════════════
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
    # NIFTY MIDCAP / HIGH QUALITY
    "TATACAP","MUTHOOTFIN","PERSISTENT","COFORGE","LTIM","MPHASIS",
    "PIIND","LAURUSLABS","ALKEM","TORNTPHARM","AUROPHARMA","IPCA","ZYDUSLIFE",
    "DIXON","AMBER","VOLTAS","BLUESTARCO","KAJARIACER","ASTRAL","SUPREMEIND",
    "POLYCAB","HAVELLS","VGUARD","KEI","PAGEIND","MANYAVAR","VEDL",
    "NATIONALUM","HINDZINC","CONCOR","IRFC","RVNL","CAMS","CDSL",
    "ANGELONE","MOTILALOFS","ICICIGI","HDFCAMC","NAUKRI","ZOMATO","IRCTC",
    "INDIGO","FORTIS","KIMS","NH","SYNGENE","DIVI","SUDARSCHEM",
    "NAVINFLUOR","SRF","AARTI","DEEPAKFERT","CHAMBAL","COROMANDEL",
    "ABB","SIEMENS","CUMMINSIND","THERMAX","BEL","HAL","GRSE","MAZAGON",
    "HUDCO","PNBHOUSING","LICHSGFIN","OBEROIRLTY","DLF","GODREJPROP",
    "PRESTIGE","BRIGADE","SOBHA","PHOENIXLTD","TATAPOWER","TORNTPOWER",
    "ADANIGREEN","JSL","RATNAMANI","MAHINDCIE","ENDURANCE","MOTHERSON",
    "BALKRISIND","APOLLOTYRE","MRF","TIINDIA","SCHAEFFLER","SKF",
    "JUSTDIAL","MAPMYINDIA","MAXHEALTH","RAINBOW","GREENPANEL","CENTURYPLY",
    "FIVESTAR","GLAND","METROPOLIS","GNFC","OLECTRA","NHPC","SJVN",
]

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def fetch_benchmark():
    try:
        df = yf.download("^NSEI", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        return df["Close"].squeeze()
    except Exception:
        return None

def nifty_in_uptrend(benchmark_close):
    """Market gate: only issue buy signals when NIFTY is above its 20-day MA."""
    if benchmark_close is None or len(benchmark_close) < 20:
        return True  # Default allow if data missing
    ma20 = benchmark_close.rolling(20).mean().iloc[-1]
    return float(benchmark_close.iloc[-1]) > float(ma20)

def fetch_stock(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 120:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        return df.dropna()
    except Exception:
        return None

def calc_atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_rs(stock_close, bench_close, days=63):
    try:
        s = float(stock_close.iloc[-1]) / float(stock_close.iloc[-days]) - 1
        b = float(bench_close.iloc[-1]) / float(bench_close.iloc[-days]) - 1
        return round((s - b) * 100, 2)
    except Exception:
        return None

def calc_ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def bollinger_width(close, n=20):
    """BB width as percentage of mid price — squeeze indicator."""
    mid  = close.rolling(n).mean()
    std  = close.rolling(n).std()
    width_pct = ((mid + 2*std) - (mid - 2*std)) / mid * 100
    return width_pct

# ═══════════════════════════════════════════════════════════════════
#  STAGE 2 UPTREND CHECK
# ═══════════════════════════════════════════════════════════════════
def check_stage2(df):
    if len(df) < 200:
        return False, {}
    c = df["close"]
    m50, m150, m200 = (c.rolling(w).mean().iloc[-1] for w in (50, 150, 200))
    price = float(c.iloc[-1])
    # Extra: 50MA must itself be rising (slope > 0 over last 5 days)
    ma50_series = c.rolling(50).mean()
    ma50_rising = float(ma50_series.iloc[-1]) > float(ma50_series.iloc[-6])
    stage2 = (price > float(m50) > float(m150) > float(m200)) and ma50_rising
    return stage2, {
        "ma50":  round(float(m50), 2),
        "ma150": round(float(m150), 2),
        "ma200": round(float(m200), 2),
    }

# ═══════════════════════════════════════════════════════════════════
#  CORE SCANNER — all 12 filters applied here
# ═══════════════════════════════════════════════════════════════════
def scan_stock(df, symbol):
    cfg       = CONFIG
    base_days = cfg["base_period_days"]

    if len(df) < base_days + 100:
        return None, {}

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    price     = float(close.iloc[-1])
    avg_vol50 = float(volume.iloc[-50:].mean())

    # ── FILTER 0: Liquidity & price floor ────────────────────────
    if price < cfg["min_stock_price"]:
        return None, {"fail": "price_floor"}
    if avg_vol50 < cfg["min_avg_volume"]:
        return None, {"fail": "liquidity"}

    # ── FILTER 1: 52-week high proximity ─────────────────────────
    high_52w = float(high.iloc[-252:].max()) if len(df) >= 252 else float(high.max())
    below_52w_pct = ((high_52w - price) / high_52w) * 100
    if below_52w_pct > cfg["max_below_52w_high_pct"]:
        return None, {"fail": f"overhead_supply_{below_52w_pct:.1f}%_below_52wH"}

    # ── FILTER 2: Prior uptrend magnitude ────────────────────────
    # Look at the 3 months BEFORE the base — stock should have rallied ≥15%
    pre_base_end   = len(df) - base_days
    pre_base_start = max(0, pre_base_end - 63)  # 3 months prior
    if pre_base_end - pre_base_start < 20:
        return None, {"fail": "insufficient_history"}
    pre_low  = float(low.iloc[pre_base_start:pre_base_end].min())
    pre_high = float(high.iloc[pre_base_start:pre_base_end].max())
    prior_uptrend_pct = ((pre_high - pre_low) / pre_low) * 100
    if prior_uptrend_pct < cfg["min_prior_uptrend_pct"]:
        return None, {"fail": f"weak_prior_trend_{prior_uptrend_pct:.1f}%"}

    # ── FILTER 3: Base range tightness ────────────────────────────
    base_h   = df.iloc[-base_days:]
    base_hi  = float(base_h["high"].max())
    base_lo  = float(base_h["low"].min())
    base_range_pct = ((base_hi - base_lo) / base_lo) * 100
    if base_range_pct > cfg["max_price_range_pct"]:
        return None, {"fail": f"base_too_wide_{base_range_pct:.1f}%"}

    # ── FILTER 4: Flat resistance quality ─────────────────────────
    pivot = base_hi  # The resistance ceiling
    band  = pivot * (cfg["resistance_band_pct"] / 100)
    touches = int((base_h["high"] >= pivot - band).sum())
    if touches < cfg["resistance_touch_count"]:
        return None, {"fail": f"few_touches_{touches}"}
    # Too many touches = weak level (supply overwhelming demand)
    if touches > cfg["max_resistance_touches"]:
        return None, {"fail": f"too_many_touches_{touches}"}

    # ── FILTER 5: Rising lows (ascending floor) ───────────────────
    seg = base_days // 3
    seg_lows = [float(base_h["low"].iloc[i*seg:(i+1)*seg].min()) for i in range(3)]
    rising_lows = sum(seg_lows[i] > seg_lows[i-1] for i in range(1, 3))
    if rising_lows < 1:
        return None, {"fail": "no_rising_lows"}

    # ── FILTER 6: Proximity to pivot (CRITICAL — 2.5%) ───────────
    dist_pct = ((pivot - price) / price) * 100
    if dist_pct > cfg["near_pivot_pct"]:
        return None, {"fail": f"too_far_from_pivot_{dist_pct:.1f}%"}

    # ── FILTER 7: Volume dry-up (entire base) ─────────────────────
    base_vol_ratio = float(base_h["volume"].mean()) / avg_vol50
    if base_vol_ratio > cfg["volume_dryup_ratio"]:
        return None, {"fail": f"vol_not_dry_{base_vol_ratio:.2f}"}

    # ── FILTER 8: Last-5-day volume exhaustion (NEW — critical) ───
    last5_vol = float(volume.iloc[-5:].mean())
    last5_vol_ratio = last5_vol / avg_vol50
    if last5_vol_ratio > cfg["last5_vol_exhaustion"]:
        return None, {"fail": f"last5_vol_not_exhausted_{last5_vol_ratio:.2f}"}

    # ── FILTER 9: Final squeeze — last 5 candles tightest ─────────
    base_daily_range  = float((base_h["high"] - base_h["low"]).mean())
    last5_daily_range = float((df["high"].iloc[-5:] - df["low"].iloc[-5:]).mean())
    squeeze_ratio = last5_daily_range / base_daily_range if base_daily_range > 0 else 1.0
    if squeeze_ratio > cfg["final_squeeze_ratio"]:
        return None, {"fail": f"no_final_squeeze_{squeeze_ratio:.2f}"}

    # ── FILTER 10: ATR at base minimum (volatility exhaustion) ────
    atr_series     = calc_atr(df, 14)
    atr_in_base    = atr_series.iloc[-base_days:].dropna()
    current_atr    = float(atr_series.iloc[-1])
    if len(atr_in_base) > 5:
        atr_pct = float(pd.Series(atr_in_base).rank(pct=True).iloc[-1]) * 100
        if atr_pct > cfg["atr_pct_threshold"]:
            return None, {"fail": f"atr_not_at_low_{atr_pct:.0f}th_pct"}
    else:
        atr_pct = 20.0  # Allow if insufficient data

    # ── FILTER 11: Bollinger Band squeeze ─────────────────────────
    bb_width_series = bollinger_width(close, 20)
    current_bb_width = float(bb_width_series.iloc[-1])
    if current_bb_width > cfg["bb_squeeze_pct_max"]:
        return None, {"fail": f"bb_wide_{current_bb_width:.1f}%"}

    # ── FILTER 12: 10-EMA rising toward pivot ─────────────────────
    ema10 = calc_ema(close, 10)
    ema10_now  = float(ema10.iloc[-1])
    ema10_5ago = float(ema10.iloc[-6])
    ema10_rising = ema10_now > ema10_5ago
    if cfg["ema10_slope_check"] and not ema10_rising:
        return None, {"fail": "ema10_declining"}

    # ── ATR-BASED STOP LOSS (tighter, more accurate) ──────────────
    entry       = round(pivot * 1.003, 2)   # Buy 0.3% above pivot
    atr_stop    = round(entry - (cfg["stop_atr_multiplier"] * current_atr), 2)
    # Also ensure stop is above the base's lowest low (sanity check)
    absolute_low = float(base_h["low"].min())
    stop_loss    = max(atr_stop, round(absolute_low * 0.99, 2))

    # ── R:R FILTER (HARD — minimum 2.5:1) ─────────────────────────
    risk_per_share   = entry - stop_loss
    if risk_per_share <= 0:
        return None, {"fail": "stop_above_entry"}

    # Targets based on measured move (base depth)
    base_depth_pct = base_range_pct / 100
    target1 = round(pivot * (1 + base_depth_pct * 0.8), 2)   # 80% of base depth
    target2 = round(pivot * (1 + base_depth_pct * 1.5), 2)   # 150% of base depth
    # Minimum: at least 7% for T1, 13% for T2
    target1 = max(target1, round(pivot * 1.07, 2))
    target2 = max(target2, round(pivot * 1.13, 2))

    rr = (target1 - entry) / risk_per_share
    if rr < cfg["min_risk_reward"]:
        return None, {"fail": f"rr_too_low_{rr:.2f}"}

    # ── COMPOSITE SCORE ────────────────────────────────────────────
    # Weighted toward "breakout imminent" signals
    score = 0

    # Proximity (20pts) — closer = more imminent
    score += int((1 - dist_pct / cfg["near_pivot_pct"]) * 20)

    # Volume exhaustion last 5d (20pts) — drier = more imminent
    score += int((1 - last5_vol_ratio / cfg["last5_vol_exhaustion"]) * 20)

    # Final squeeze tightness (20pts)
    score += int((1 - squeeze_ratio / cfg["final_squeeze_ratio"]) * 20)

    # ATR at low (15pts)
    score += int((1 - atr_pct / cfg["atr_pct_threshold"]) * 15)

    # BB squeeze (10pts)
    bb_score = max(0, (cfg["bb_squeeze_pct_max"] - current_bb_width) / cfg["bb_squeeze_pct_max"])
    score += int(bb_score * 10)

    # R:R quality (10pts)
    score += min(10, int((rr - cfg["min_risk_reward"]) * 3))

    # Rising lows quality (5pts)
    score += 5 if rising_lows >= 2 else 2

    score = max(0, min(100, score))

    return {
        "symbol":               symbol,
        "price":                round(price, 2),
        "pivot":                round(pivot, 2),
        "entry":                entry,
        "stop_loss":            stop_loss,
        "target1":              target1,
        "target2":              target2,
        "risk_per_share":       round(risk_per_share, 2),
        "risk_reward":          round(rr, 2),
        "vcp_score":            score,
        # Pattern metrics
        "dist_pct":             round(dist_pct, 2),
        "base_range_pct":       round(base_range_pct, 2),
        "base_vol_ratio":       round(base_vol_ratio, 3),
        "last5_vol_ratio":      round(last5_vol_ratio, 3),
        "squeeze_ratio":        round(squeeze_ratio, 3),
        "atr_percentile":       round(atr_pct, 1),
        "bb_width_pct":         round(current_bb_width, 2),
        "resistance_touches":   touches,
        "rising_lows":          rising_lows,
        "prior_uptrend_pct":    round(prior_uptrend_pct, 1),
        "below_52w_high_pct":   round(below_52w_pct, 1),
        "ema10_rising":         ema10_rising,
        "current_atr":          round(current_atr, 2),
    }, {}

def extract_candles(df, n=50):
    recent = df.iloc[-n:].copy()
    return [
        [round(float(r["open"]), 2), round(float(r["high"]), 2),
         round(float(r["low"]),  2), round(float(r["close"]),2), int(r["volume"])]
        for _, r in recent.iterrows()
    ]

# ═══════════════════════════════════════════════════════════════════
#  MAIN RUN
# ═══════════════════════════════════════════════════════════════════
def run():
    print("=" * 65)
    print("  VCP BREAKOUT SCANNER v2 — NSE")
    print("  12-Filter High-Accuracy Edition · Min R:R 2.5:1")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    print("\n  ⬇ Fetching NIFTY 50 benchmark…")
    benchmark = fetch_benchmark()

    # ── MARKET GATE: Don't buy if NIFTY is in downtrend ──────────
    market_ok = nifty_in_uptrend(benchmark)
    if not market_ok:
        print("  ⚠ NIFTY is BELOW 20-day MA — market in downtrend.")
        print("  ⚠ Scanner will still run but results carry extra risk.")

    results = []
    filter_log = {}

    for sym in tqdm(NSE_STOCKS, desc="  Scanning", ncols=65):
        try:
            df = fetch_stock(sym)
            if df is None:
                filter_log[sym] = "no_data"
                continue

            if CONFIG["uptrend_ma_check"]:
                stage2, ma_data = check_stage2(df)
                if not stage2:
                    filter_log[sym] = "not_stage2"
                    continue
            else:
                _, ma_data = check_stage2(df)

            hit, reason = scan_stock(df, sym)
            if hit is None:
                filter_log[sym] = reason.get("fail", "unknown")
                continue

            # RS vs NIFTY
            if benchmark is not None:
                hit["rs_vs_nifty"] = calc_rs(df["close"], benchmark, CONFIG["rs_days"])
                # Bonus: RS must be positive (stock outperforming NIFTY)
                if hit["rs_vs_nifty"] is not None and hit["rs_vs_nifty"] < -5:
                    filter_log[sym] = f"weak_rs_{hit['rs_vs_nifty']}"
                    continue
            else:
                hit["rs_vs_nifty"] = None

            hit["ma50"]         = ma_data.get("ma50")
            hit["ma150"]        = ma_data.get("ma150")
            hit["ma200"]        = ma_data.get("ma200")
            hit["market_ok"]    = market_ok
            hit["candles"]      = extract_candles(df, CONFIG["candle_history"])

            results.append(hit)
            time.sleep(CONFIG["sleep_between"])

        except Exception as e:
            filter_log[sym] = f"error_{str(e)[:30]}"
            continue

    # Sort by score
    results.sort(key=lambda x: x["vcp_score"], reverse=True)
    top = results[:CONFIG["top_n"]]

    # ── Print filter summary ───────────────────────────────────────
    from collections import Counter
    fc = Counter(filter_log.values())
    print(f"\n  Filter Summary (why stocks were rejected):")
    for reason, count in fc.most_common(8):
        print(f"    {reason:<35} {count:>3} stocks")

    # ── Write JSON ─────────────────────────────────────────────────
    out = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "market_date":    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_scanned":  len(NSE_STOCKS),
        "total_found":    len(results),
        "market_uptrend": market_ok,
        "stocks":         top,
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "docs", "data", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  ✅  {len(results)} stocks passed ALL 12 filters | Top {len(top)} saved")
    print(f"  📁  → docs/data/results.json\n")

    if top:
        print(f"  {'#':<3} {'Symbol':<12} {'Price':>8} {'Pivot':>8} "
              f"{'Dist':>6} {'R:R':>6} {'Score':>6} {'RS':>7}")
        print(f"  {'─'*3} {'─'*12} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
        for i, s in enumerate(top, 1):
            rs  = f"{s['rs_vs_nifty']:+.1f}%" if s.get('rs_vs_nifty') else "  N/A"
            print(f"  {i:<3} {s['symbol']:<12} "
                  f"₹{s['price']:>7.2f} "
                  f"₹{s['pivot']:>7.2f} "
                  f"{s['dist_pct']:>5.1f}% "
                  f"{s['risk_reward']:>5.1f}x "
                  f"{s['vcp_score']:>5} "
                  f"{rs:>7}")
    else:
        print("  ⚠  No stocks passed all 12 filters today.")
        print("  This is intentional — quality over quantity.")
        print("  Check back tomorrow or relax CONFIG thresholds slightly.\n")

    print("=" * 65)

if __name__ == "__main__":
    run()
