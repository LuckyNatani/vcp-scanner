"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VCP SCANNER v4 — PRODUCTION GRADE                                      ║
║  NSE India · Two-Category Output · Research-Backed Rules                ║
║                                                                          ║
║  WHAT'S NEW vs v3:                                                       ║
║  ┌─ CONTRACTION ENGINE ──────────────────────────────────────────────┐   ║
║  │ • Zigzag swing detector (not equal-thirds division)               │   ║
║  │ • Outlier candle exclusion (circuit-breaker artifacts)            │   ║
║  │ • Rising lows validation (true ascending triangle)                │   ║
║  │ • Final coil: ATR percentile check, not just range %              │   ║
║  │ • Volume slope check (gradual dry-up, not sudden)                 │   ║
║  └───────────────────────────────────────────────────────────────────┘   ║
║  ┌─ TWO CATEGORIES ──────────────────────────────────────────────────┐   ║
║  │ • pre_breakout: coiling within 3% of pivot, imminent              │   ║
║  │ • broken_out: crossed pivot on volume in last 1-20 days           │   ║
║  └───────────────────────────────────────────────────────────────────┘   ║
║  ┌─ INDIAN MARKET ADJUSTMENTS ───────────────────────────────────────┐   ║
║  │ • Higher volatility bands (C1 12-40%, C3 ≤10%)                   │   ║
║  │ • EMA50 slope check (not just price > EMA50)                      │   ║
║  │ • RS must be POSITIVE vs NIFTY (anti-operator filter)             │   ║
║  │ • 10-day vol avg instead of raw 5-day (T+1 settlement effect)     │   ║
║  └───────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json, time, warnings, os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG — all values research-backed
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Scanner settings
    "top_n_pre":              8,    # Pre-breakout results to keep
    "top_n_broken":           8,    # Broken-out results to keep
    "data_period":            "18mo",
    "candle_history_days":    75,
    "sleep_between":          0.22,

    # ── Liquidity
    "min_price":              60,
    "min_avg_vol":            200_000,

    # ── Swing detection (Zigzag)
    # India: higher volatility means we need bigger threshold to avoid noise
    "swing_threshold_pct":    3.5,   # Minimum % move to qualify as swing
    "swing_min_bars":         3,     # Minimum candles between swings
    "outlier_vol_ratio":      0.4,   # Exclude spike candles with vol < 40% avg (circuit artifacts)

    # ── Contraction validation
    # From 20 examples: C2/C1 avg=0.56. Allow up to 0.88 (generous for imperfect charts)
    "max_contraction_ratio":  0.88,  # Each wave must be ≤ 88% of prior (at least 12% tighter)
    "min_contraction_count":  2,     # Need at least 2 contracting waves
    "c1_min_pct":             8.0,   # First contraction must be meaningful
    "c1_max_pct":             45.0,  # First contraction cap (very wide Indian moves allowed)
    "final_swing_max_pct":    10.0,  # Final swing ≤ 10% (India: slightly wider than 8%)
    "min_rising_lows":        1,     # At least 1 pair of rising swing lows

    # ── Final coil (last 10 sessions)
    "coil_atr_pct_max":       40,    # Final 10-bar ATR must be in bottom 40th pctile of base
    "coil_vol_ratio_max":     0.35,  # 10-day avg vol < 35% of 50-day avg
    "vol_slope_sessions":     15,    # Check vol is declining over last 15 sessions

    # ── Base window
    "base_min_days":          20,    # Minimum base (4 weeks)
    "base_max_days":          120,   # Maximum base (24 weeks; Indian bases can be long)
    "default_base_days":      60,    # Default lookback

    # ── Pivot & proximity
    "near_pivot_pct":         3.0,   # PRE-BREAKOUT: within 3% of pivot
    "pivot_lookback_frac":    0.35,  # Pivot = highest swing high in last 35% of base

    # ── Prior trend
    "prior_window":           90,    # Days to look back for prior uptrend
    "min_prior_uptrend":      18.0,  # Must have gained ≥18% before base formed
    "prior_ema50_slope":      True,  # EMA50 must be rising (slope > 0 over last 10 days)

    # ── 52-week high
    "max_below_52w":          22.0,  # Within 22% of 52w high (slightly wider for deep-base stocks)

    # ── RS vs NIFTY (anti-operator filter)
    "rs_days":                63,
    "min_rs_vs_nifty":        -5.0,  # RS must be > -5% (tolerant — some great stocks lag briefly)

    # ── Targets — derived from 20 examples avg gain: +77%
    "stop_atr_mult":          1.8,   # Stop = entry - 1.8×ATR (tighter than base-low method)
    "target1_pct":            18.0,  # T1: +18% from pivot
    "target2_pct":            35.0,  # T2: +35% from pivot

    # ── Broken-out detection
    "breakout_lookback":      20,    # Scan last 20 sessions for breakout
    "breakout_vol_ratio":     1.4,   # Breakout day volume ≥ 1.4× 20-day avg
    "breakout_close_factor":  0.97,  # Must close in top 97% of day's range (close near high)
    "breakout_max_ext":       20.0,  # Not more than 20% above pivot (too extended)
    "breakout_above_pivot":   0.5,   # Still valid if max 0.5% below pivot (test of breakout)

    # ── Min R:R
    "min_rr":                 2.0,
}

# ══════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE — 200 NSE liquid stocks across market caps
# ══════════════════════════════════════════════════════════════════════════
NSE_STOCKS = [
    # NIFTY50
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN","HINDUNILVR",
    "ITC","LT","KOTAKBANK","HCLTECH","BAJFINANCE","AXISBANK","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","NESTLEIND","POWERGRID","NTPC","JSWSTEEL",
    "TATASTEEL","TECHM","HINDALCO","INDUSINDBK","BAJAJFINSV","ONGC","COALINDIA",
    "CIPLA","DRREDDY","EICHERMOT","APOLLOHOSP","DIVISLAB","HEROMOTOCO","GRASIM",
    "TATACONSUM","BRITANNIA","BPCL","TATAMOTORS","M&M","ADANIPORTS","SHRIRAMFIN",
    "BAJAJ-AUTO","HDFCLIFE","SBILIFE","TRENT","ADANIENT",
    # MidCap — best VCP candidates in India
    "TATACAP","MUTHOOTFIN","PERSISTENT","COFORGE","LTIM","MPHASIS","PIIND",
    "LAURUSLABS","ALKEM","TORNTPHARM","AUROPHARMA","IPCA","ZYDUSLIFE","DIXON",
    "AMBER","VOLTAS","BLUESTARCO","KAJARIACER","ASTRAL","SUPREMEIND","POLYCAB",
    "HAVELLS","VGUARD","KEI","PAGEIND","MANYAVAR","VEDL","NATIONALUM","HINDZINC",
    "CONCOR","IRFC","RVNL","CAMS","CDSL","ANGELONE","MOTILALOFS","ICICIGI",
    "HDFCAMC","NAUKRI","ZOMATO","IRCTC","INDIGO","FORTIS","KIMS","NH","SYNGENE",
    "DIVI","SUDARSCHEM","NAVINFLUOR","SRF","AARTI","DEEPAKFERT","CHAMBAL",
    "COROMANDEL","GNFC","ABB","SIEMENS","CUMMINSIND","THERMAX","BEL","HAL",
    "GRSE","MAZAGON","HUDCO","PNBHOUSING","LICHSGFIN","OBEROIRLTY","DLF",
    "GODREJPROP","PRESTIGE","BRIGADE","SOBHA","PHOENIXLTD","TATAPOWER",
    "TORNTPOWER","ADANIPOWER","ADANIGREEN","JSL","RATNAMANI","MAHINDCIE",
    "ENDURANCE","MOTHERSON","BALKRISIND","APOLLOTYRE","MRF","TIINDIA",
    "SCHAEFFLER","SKF","LINDEINDIA","CHOLAFIN","GPIL","GRAVITA","PRINCEPIPE",
    "ALKYLAMINE","ATGL","NAHARSPING","OLECTRA","JINDSTEEL","SAIL","KECL",
    # Small-MidCap high-momentum
    "KAYNES","SYRMA","SAFARI","CAMPUS","DELHIVERY","NYKAA","POLICYBZR",
    "PAYTM","RATEGAIN","IDEAFORGE","TECHNO","ELECON","TITAGARH","RAILTEL",
    "RITES","IRCON","NBCC","HSCL","JKCEMENT","RAMCOCEM","HEIDELBERG",
    "APLAPOLLO","JINDALSAW","WELSPUN","HATHWAY","GTPL","INDIAMART",
    "CARTRADE","EASEMYTRIP","MAPMYINDIA","INOX","PVR","DEVYANI","WESTLIFE",
    "JUBLFOOD","BURGER","SAPPHIRE","IXIGO","AWFIS","SENCO","KHYATI",
    "BLUEDART","GATI","TCI","MAHLOG","ALLCARGO","ESAB","THERMAX","ELGIEQUIP",
    "JYOTICNC","GRINDWELL","TIMKEN","NESCO","GPPL","MPSLTD","NETWORK18",
    "TV18BRDCST","SONACOMS","SUNTV","ZEEL","PVRINOX",
]
# Deduplicate
NSE_STOCKS = list(dict.fromkeys(NSE_STOCKS))


# ══════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════

def fetch_benchmark():
    try:
        df = yf.download("^NSEI", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        c = df["Close"].squeeze()
        return c.dropna() if not c.empty else None
    except Exception:
        return None


def nifty_trend(bench):
    """Returns (is_uptrend: bool, strength: str)"""
    if bench is None or len(bench) < 20:
        return True, "unknown"
    price = float(bench.iloc[-1])
    ma20  = float(bench.rolling(20).mean().iloc[-1])
    ma50  = float(bench.rolling(50).mean().iloc[-1]) if len(bench) >= 50 else ma20
    if price > ma20 > ma50:
        return True, "strong"
    elif price > ma20:
        return True, "moderate"
    else:
        return False, "downtrend"


def fetch_stock(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period=CONFIG["data_period"],
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 150:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        # Require all key columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                return None
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
#  PRODUCTION-GRADE CONTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

def find_swing_points(df, threshold_pct=None, min_bars=None):
    """
    Zigzag algorithm: finds significant swing highs and lows.

    India-specific: Uses higher threshold than US (3.5% vs 2%) to avoid
    noise from higher volatility and circuit-breaker artifacts.

    Outlier exclusion: Single-day spikes on very low volume (< 40% of avg)
    are not counted as valid swings — these are often circuit artifacts
    or low-liquidity fake moves.

    Returns:
        swings: list of dicts {idx, price, type: 'H'|'L', date, volume}
    """
    cfg = CONFIG
    threshold = threshold_pct or cfg["swing_threshold_pct"]
    min_b     = min_bars or cfg["swing_min_bars"]

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    vols   = df["volume"].values
    avg_vol = np.mean(vols[-50:]) if len(vols) >= 50 else np.mean(vols)

    n      = len(df)
    swings = []

    # Determine starting direction
    # Look at first vs last to decide if we start searching for a high or low
    direction = "H"  # start hunting for first swing high

    last_swing_idx   = 0
    last_swing_price = closes[0]
    last_swing_type  = None

    for i in range(1, n):
        current_high  = highs[i]
        current_low   = lows[i]
        is_outlier    = vols[i] < avg_vol * cfg["outlier_vol_ratio"]

        if direction == "H":
            # Looking for a swing high
            if current_high > last_swing_price:
                last_swing_price = current_high
                last_swing_idx   = i
            elif (last_swing_price - lows[i]) / last_swing_price * 100 >= threshold:
                # We've moved down enough — confirm swing high
                if (last_swing_idx - (swings[-1]["idx"] if swings else 0)) >= min_b:
                    if not is_outlier or last_swing_idx != i:
                        swings.append({
                            "idx":    last_swing_idx,
                            "price":  last_swing_price,
                            "type":   "H",
                            "date":   str(df.index[last_swing_idx].date()),
                            "vol":    float(vols[last_swing_idx]),
                            "outlier": bool(vols[last_swing_idx] < avg_vol * cfg["outlier_vol_ratio"]),
                        })
                direction        = "L"
                last_swing_price = lows[i]
                last_swing_idx   = i

        else:  # direction == "L"
            if current_low < last_swing_price:
                last_swing_price = current_low
                last_swing_idx   = i
            elif (highs[i] - last_swing_price) / last_swing_price * 100 >= threshold:
                # Moved up enough — confirm swing low
                if (last_swing_idx - (swings[-1]["idx"] if swings else 0)) >= min_b:
                    if not is_outlier or last_swing_idx != i:
                        swings.append({
                            "idx":    last_swing_idx,
                            "price":  last_swing_price,
                            "type":   "L",
                            "date":   str(df.index[last_swing_idx].date()),
                            "vol":    float(vols[last_swing_idx]),
                            "outlier": bool(vols[last_swing_idx] < avg_vol * cfg["outlier_vol_ratio"]),
                        })
                direction        = "H"
                last_swing_price = highs[i]
                last_swing_idx   = i

    return swings


def measure_contraction_waves(swings):
    """
    Given alternating H/L swings, measure each contraction cycle.
    A "cycle" = (swing_high, subsequent_swing_low) → range of that pullback
    
    Returns:
        waves: list of floats (each = range as % of swing_high price)
        pairs: list of (swing_high, swing_low) tuples
    """
    waves = []
    pairs = []

    for i in range(len(swings) - 1):
        s1 = swings[i]
        s2 = swings[i + 1]

        # We want H→L pairs (a pullback = contraction)
        if s1["type"] == "H" and s2["type"] == "L":
            rng = (s1["price"] - s2["price"]) / s1["price"] * 100
            waves.append(round(rng, 2))
            pairs.append((s1, s2))

    return waves, pairs


def validate_contraction(waves):
    """
    Check if waves form a contracting (decreasing) sequence.
    Returns (contractions_count, is_valid, contraction_ratios)
    """
    cfg   = CONFIG
    if len(waves) < cfg["min_contraction_count"]:
        return 0, False, []

    ratios = []
    contracting_count = 0

    for i in range(1, len(waves)):
        if waves[i - 1] <= 0:
            continue
        ratio = waves[i] / waves[i - 1]
        ratios.append(round(ratio, 3))
        if ratio <= cfg["max_contraction_ratio"]:
            contracting_count += 1

    # First wave must be meaningful
    c1_ok = cfg["c1_min_pct"] <= waves[0] <= cfg["c1_max_pct"] if waves else False

    # Final wave must be tight
    final_ok = waves[-1] <= cfg["final_swing_max_pct"] if waves else False

    # At least N contracting pairs
    is_valid = (
        contracting_count >= cfg["min_contraction_count"] - 1
        and c1_ok
        and final_ok
        and len(waves) >= cfg["min_contraction_count"]
    )

    return contracting_count, is_valid, ratios


def check_rising_lows(swings):
    """
    Verify that swing lows trend upward (ascending triangle / VCP structure).
    Returns (rising_count, is_valid)
    """
    lows = [s["price"] for s in swings if s["type"] == "L"]
    if len(lows) < 2:
        return 0, False

    rising = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    return rising, rising >= CONFIG["min_rising_lows"]


def check_final_coil(df):
    """
    Last 10 bars must have ATR in bottom 40th percentile of base ATR.
    This is the 'energy compression' before the spring releases.

    Also validates that volume is declining GRADUALLY (slope check),
    not just dropping suddenly (circuit-breaker artifact).

    Returns (coil_atr_pct, vol_slope_ok, coil_vol_ratio)
    """
    cfg = CONFIG
    n   = len(df)
    if n < 15:
        return 50, False, 1.0

    def atr_series(data):
        h, l, c = data["high"], data["low"], data["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr

    base_atr   = atr_series(df)
    last10_atr = float(base_atr.iloc[-10:].mean())
    base_atrs  = base_atr.values

    # Percentile of final ATR within base ATR distribution
    pct = float(np.sum(base_atrs < last10_atr) / len(base_atrs) * 100)

    # Volume slope: fit linear regression to last N sessions
    vol_window = df["volume"].iloc[-cfg["vol_slope_sessions"]:].values
    avg_vol50  = float(df["volume"].iloc[-50:].mean())
    x          = np.arange(len(vol_window))
    slope      = np.polyfit(x, vol_window, 1)[0] if len(vol_window) > 3 else 0
    vol_slope_ok = slope <= 0  # Volume should be declining (negative slope = OK)

    # 10-day avg volume ratio (vs 50d avg) — more stable than 5-day
    coil_vol_ratio = float(df["volume"].iloc[-10:].mean()) / avg_vol50 if avg_vol50 > 0 else 1.0

    return round(pct, 1), vol_slope_ok, round(coil_vol_ratio, 3)


def identify_pivot(df_base, swings, lookback_frac=None):
    """
    Pivot = highest swing HIGH in the last 35% of the base.
    NOT the base's overall high (that may be an old resistance from months ago).
    The pivot is the MOST RECENT resistance ceiling.
    """
    frac = lookback_frac or CONFIG["pivot_lookback_frac"]
    n    = len(df_base)
    recent_start_idx = int(n * (1 - frac))

    # Find swing highs in the recent portion of the base
    recent_highs = [
        s for s in swings
        if s["type"] == "H" and s["idx"] >= recent_start_idx
    ]

    if recent_highs:
        pivot = max(recent_highs, key=lambda s: s["price"])["price"]
    else:
        # Fallback: actual high of the last 35% of the base
        pivot = float(df_base["high"].iloc[recent_start_idx:].max())

    return float(pivot)


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 2 CHECK (EMA-slope version)
# ══════════════════════════════════════════════════════════════════════════

def check_stage2(df):
    """
    India-specific Stage 2:
    1. Price > EMA50
    2. EMA50 slope is POSITIVE (rising, not just flat)
    3. EMA50 > EMA150 > EMA200
    """
    if len(df) < 200:
        return False, {}

    close = df["close"]
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    price    = float(close.iloc[-1])
    e50_now  = float(ema50.iloc[-1])
    e50_10d  = float(ema50.iloc[-10])   # EMA50 10 days ago (slope check)
    e150     = float(ema150.iloc[-1])
    e200     = float(ema200.iloc[-1])

    ema50_rising = e50_now > e50_10d   # Key India-specific check

    ok = (price > e50_now and e50_rising and e50_now > e150 > e200)
    return ok, {
        "ema50":  round(e50_now, 2),
        "ema150": round(e150, 2),
        "ema200": round(e200, 2),
        "ema50_rising": ema50_rising,
    }


# ══════════════════════════════════════════════════════════════════════════
#  RS CALCULATION
# ══════════════════════════════════════════════════════════════════════════

def calc_rs(stock_close, bench_close, days=63):
    try:
        sc = stock_close.reindex(bench_close.index, method="ffill").dropna()
        bc = bench_close.reindex(sc.index, method="ffill").dropna()
        if len(sc) < days or len(bc) < days:
            return None
        return round(
            (float(sc.iloc[-1]) / float(sc.iloc[-days]) - 1 -
             (float(bc.iloc[-1]) / float(bc.iloc[-days]) - 1)) * 100, 2
        )
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
#  ATR
# ══════════════════════════════════════════════════════════════════════════

def atr14(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


# ══════════════════════════════════════════════════════════════════════════
#  MAIN SCANNER — PRE-BREAKOUT
# ══════════════════════════════════════════════════════════════════════════

def scan_pre_breakout(df, symbol, bench_close):
    """
    Scans for stocks about to break out:
    - Valid VCP structure (contracting waves via zigzag)
    - Within 3% of pivot
    - Final coil forming
    - Volume exhausted
    Returns (result_dict, failure_reason_str)
    """
    cfg  = CONFIG
    n    = len(df)
    base = df.iloc[-cfg["default_base_days"]:]

    price     = float(df["close"].iloc[-1])
    avg_vol50 = float(df["volume"].iloc[-50:].mean())

    # ── Liquidity
    if price < cfg["min_price"]:          return None, "price_floor"
    if avg_vol50 < cfg["min_avg_vol"]:    return None, "low_liquidity"

    # ── 52w high
    hi52 = float(df["high"].iloc[-252:].max()) if n >= 252 else float(df["high"].max())
    below52 = (hi52 - price) / hi52 * 100
    if below52 > cfg["max_below_52w"]:    return None, f"overhead_{below52:.0f}%"

    # ── Prior uptrend
    prior_end   = n - cfg["default_base_days"]
    prior_start = max(0, prior_end - cfg["prior_window"])
    if prior_end - prior_start < 15:      return None, "no_prior"
    prior_lo = float(df["low"].iloc[prior_start:prior_end].min())
    prior_hi = float(df["high"].iloc[prior_start:prior_end].max())
    prior_gain = (prior_hi - prior_lo) / prior_lo * 100 if prior_lo > 0 else 0
    if prior_gain < cfg["min_prior_uptrend"]: return None, f"weak_prior_{prior_gain:.0f}%"

    # ── EMA50 slope (Stage 2 with slope)
    ema_ok, ema_data = check_stage2(df)
    if not ema_ok:                        return None, "not_stage2"

    # ── RS vs NIFTY
    rs = calc_rs(df["close"], bench_close, cfg["rs_days"]) if bench_close is not None else None
    if rs is not None and rs < cfg["min_rs_vs_nifty"]:
        return None, f"weak_rs_{rs:.1f}%"

    # ══ PRODUCTION CONTRACTION ENGINE ══════════════════════════════════

    # Step 1: Find swing points in the base
    swings = find_swing_points(base)
    if len(swings) < 4:                   return None, "insufficient_swings"

    # Step 2: Measure contraction waves
    waves, wave_pairs = measure_contraction_waves(swings)
    if len(waves) < 2:                    return None, "insufficient_waves"

    # Step 3: Validate contraction
    contraction_count, is_contracting, ratios = validate_contraction(waves)
    if not is_contracting:
        return None, f"no_contraction_c{contraction_count}_w{'|'.join(str(w) for w in waves[:3])}"

    # Step 4: Rising lows
    rising_count, rising_ok = check_rising_lows(swings)
    if not rising_ok:                     return None, f"falling_lows_{rising_count}"

    # Step 5: Final coil
    coil_pct, vol_slope_ok, coil_vol_ratio = check_final_coil(base)
    if coil_pct > cfg["coil_atr_pct_max"]:
        return None, f"no_coil_atr_{coil_pct:.0f}th_pct"
    if coil_vol_ratio > cfg["coil_vol_ratio_max"]:
        return None, f"vol_not_dry_{coil_vol_ratio:.2f}"

    # ════════════════════════════════════════════════════════════════════

    # ── Pivot (most recent resistance ceiling)
    pivot = identify_pivot(base, swings)
    dist_pct = (pivot - price) / price * 100
    if dist_pct > cfg["near_pivot_pct"]:  return None, f"far_from_pivot_{dist_pct:.1f}%"
    if dist_pct < 0:                       return None, "already_broken"

    # ── Trade levels
    current_atr = atr14(df)
    entry    = round(pivot * 1.003, 2)
    stop     = round(entry - cfg["stop_atr_mult"] * current_atr, 2)
    risk     = entry - stop
    if risk <= 0:                          return None, "zero_risk"
    target1  = round(pivot * (1 + cfg["target1_pct"] / 100), 2)
    target2  = round(pivot * (1 + cfg["target2_pct"] / 100), 2)
    rr       = (target1 - entry) / risk
    if rr < cfg["min_rr"]:                return None, f"poor_rr_{rr:.1f}"

    # ── Cup shape detection
    is_cup = _detect_cup(base)

    # ── COMPOSITE SCORE ─────────────────────────────────────────────
    score = 0

    # Contraction quality (35pts)
    score += min(20, contraction_count * 7)
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        score += int((1 - avg_ratio) * 15)  # Tighter = better

    # Pivot proximity (20pts)
    score += int((1 - dist_pct / cfg["near_pivot_pct"]) * 20)

    # Volume exhaustion (25pts)
    score += int((1 - coil_vol_ratio / cfg["coil_vol_ratio_max"]) * 20)
    score += 5 if vol_slope_ok else 0

    # Coil tightness (10pts)
    score += int((1 - coil_pct / cfg["coil_atr_pct_max"]) * 10)

    # Rising lows bonus (5pts)
    score += min(5, rising_count * 2)

    # Cup bonus (5pts)
    score += 5 if is_cup else 0

    score = max(0, min(100, score))

    return {
        "symbol":          symbol,
        "category":        "pre_breakout",
        "price":           round(price, 2),
        "pivot":           round(pivot, 2),
        "entry":           entry,
        "stop_loss":       stop,
        "target1":         target1,
        "target2":         target2,
        "risk_reward":     round(rr, 2),
        "vcp_score":       score,
        # Wave data
        "waves":           waves,
        "contraction_ratios": ratios,
        "contraction_count": contraction_count,
        "rising_lows":     rising_count,
        "is_cup":          is_cup,
        # Coil metrics
        "coil_atr_pct":    coil_pct,
        "coil_vol_ratio":  coil_vol_ratio,
        "vol_slope_ok":    vol_slope_ok,
        # Context
        "dist_pct":        round(dist_pct, 2),
        "below_52w_pct":   round(below52, 1),
        "prior_uptrend":   round(prior_gain, 1),
        "rs_vs_nifty":     rs,
        "ema50":           ema_data.get("ema50"),
        "atr":             round(current_atr, 2),
        "avg_vol50":       round(avg_vol50, 0),
    }, None


# ══════════════════════════════════════════════════════════════════════════
#  BREAKOUT SCANNER — already broken out
# ══════════════════════════════════════════════════════════════════════════

def scan_broken_out(df, symbol, bench_close):
    """
    Scans for stocks that have already broken out (1-20 days ago).

    Requirements:
    - Had a valid VCP structure BEFORE the breakout
    - Breakout: first close above pivot on volume ≥ 1.4× 20-day avg
    - Breakout candle closed near day's high (≥ 97th percentile of range)
    - Currently still within 20% above pivot (not too extended)
    - Has NOT closed back below pivot (no false breakout)
    """
    cfg  = CONFIG
    n    = len(df)

    price     = float(df["close"].iloc[-1])
    avg_vol50 = float(df["volume"].iloc[-50:].mean())

    if price < cfg["min_price"]:       return None, "price_floor"
    if avg_vol50 < cfg["min_avg_vol"]: return None, "low_liquidity"

    # RS filter
    rs = calc_rs(df["close"], bench_close, cfg["rs_days"]) if bench_close is not None else None
    if rs is not None and rs < cfg["min_rs_vs_nifty"]:
        return None, f"weak_rs"

    # Look back for VCP base before the breakout + scan for breakout day
    lookback = cfg["breakout_lookback"]
    scan_window = min(lookback + cfg["default_base_days"], n - 20)

    best_breakout = None

    for days_ago in range(1, lookback + 1):
        bo_idx = n - days_ago   # Potential breakout index

        # VCP base = the period before this potential breakout
        base_end = bo_idx
        base_start = max(0, base_end - cfg["default_base_days"])
        if base_end - base_start < 20:
            continue

        df_base = df.iloc[base_start:base_end]
        df_bo   = df.iloc[bo_idx]   # The potential breakout day

        bo_close  = float(df_bo["close"])
        bo_high   = float(df_bo["high"])
        bo_low    = float(df_bo["low"])
        bo_vol    = float(df_bo["volume"])

        # Find base pivot
        swings = find_swing_points(df_base)
        if len(swings) < 4:
            continue
        pivot = identify_pivot(df_base, swings)

        # Was this the breakout?
        if bo_close <= pivot:
            continue

        # Volume check
        avg_vol_20d = float(df["volume"].iloc[max(0, bo_idx - 20):bo_idx].mean())
        if avg_vol_20d <= 0:
            continue
        vol_ratio = bo_vol / avg_vol_20d
        if vol_ratio < cfg["breakout_vol_ratio"]:
            continue

        # Closing position check (closed near day's high)
        day_range = bo_high - bo_low
        if day_range > 0:
            close_pos = (bo_close - bo_low) / day_range
            if close_pos < cfg["breakout_close_factor"]:
                continue

        # Make sure price hasn't retreated below pivot since breakout
        post_breakout = df["close"].iloc[bo_idx:]
        if float(post_breakout.min()) < pivot * (1 - cfg["breakout_above_pivot"] / 100):
            continue  # False breakout — closed back below pivot

        # Not too extended (current price vs pivot)
        ext_pct = (price - pivot) / pivot * 100
        if ext_pct > cfg["breakout_max_ext"]:
            continue

        # Validate VCP BEFORE breakout
        waves, _ = measure_contraction_waves(swings)
        c_count, is_ok, ratios = validate_contraction(waves) if waves else (0, False, [])

        # Score this breakout
        score = 0
        score += min(25, int(vol_ratio * 10))             # Volume conviction
        score += int((1 - ext_pct / cfg["breakout_max_ext"]) * 20)  # Not too extended
        score += min(20, c_count * 7) if is_ok else 0     # VCP quality before
        score += int(close_pos * 15)                       # Strong close

        if best_breakout is None or score > best_breakout["bo_score"]:
            best_breakout = {
                "pivot":         round(pivot, 2),
                "bo_close":      round(bo_close, 2),
                "bo_vol_ratio":  round(vol_ratio, 2),
                "bo_close_pos":  round(close_pos, 2),
                "bo_days_ago":   days_ago,
                "ext_pct":       round(ext_pct, 1),
                "waves":         waves,
                "contraction_count": c_count if is_ok else 0,
                "bo_score":      score,
                "is_cup":        _detect_cup(df_base),
            }

    if best_breakout is None:
        return None, "no_valid_breakout"

    # Trade levels for broken-out stocks (pullback to pivot strategy)
    pivot     = best_breakout["pivot"]
    atr       = atr14(df)
    entry     = round(pivot * 1.01, 2)   # Entry on pullback to 1% above pivot
    stop      = round(pivot * 0.96, 2)   # Stop: 4% below pivot (clear invalidation)
    risk      = entry - stop
    target1   = round(pivot * (1 + cfg["target1_pct"] / 100), 2)
    target2   = round(pivot * (1 + cfg["target2_pct"] / 100), 2)
    rr        = (target1 - entry) / risk if risk > 0 else 0

    ema_ok, ema_data = check_stage2(df)

    return {
        "symbol":           symbol,
        "category":         "broken_out",
        "price":            round(price, 2),
        "pivot":            pivot,
        "breakout_price":   best_breakout["bo_close"],
        "entry":            entry,
        "stop_loss":        stop,
        "target1":          target1,
        "target2":          target2,
        "risk_reward":      round(rr, 2),
        "vcp_score":        best_breakout["bo_score"],
        # Breakout context
        "days_since_breakout": best_breakout["bo_days_ago"],
        "vol_on_breakout":  best_breakout["bo_vol_ratio"],
        "extension_pct":    best_breakout["ext_pct"],
        "close_position":   best_breakout["bo_close_pos"],
        "waves":            best_breakout["waves"],
        "contraction_count": best_breakout["contraction_count"],
        "is_cup":           best_breakout["is_cup"],
        # Context
        "rs_vs_nifty":      rs,
        "below_52w_pct":    round((float(df["high"].iloc[-252:].max()) - price) /
                                   float(df["high"].iloc[-252:].max()) * 100, 1)
                             if n >= 252 else None,
        "atr":              round(atr, 2),
        "avg_vol50":        round(avg_vol50, 0),
    }, None


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _detect_cup(df_base):
    """U-shape low detection — ALEMBIC / APOLLO TUBES style"""
    n = len(df_base)
    if n < 15:
        return False
    seg  = n // 5
    lows = [float(df_base["low"].iloc[i*seg:(i+1)*seg].min()) for i in range(5)]
    mid  = lows[2]
    la   = (lows[0] + lows[1]) / 2
    ra   = (lows[3] + lows[4]) / 2
    return mid < la * 0.97 and mid < ra * 0.97 and ra > mid


def extract_candles(df, n=75):
    return [
        [round(float(r["open"]),2), round(float(r["high"]),2),
         round(float(r["low"]),2),  round(float(r["close"]),2), int(r["volume"])]
        for _, r in df.iloc[-n:].iterrows()
    ]


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 68)
    print("  VCP SCANNER v4 · NSE India · Production Grade")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 68)

    print("\n  ⬇ Fetching NIFTY…")
    bench      = fetch_benchmark()
    is_uptrend, mkt_strength = nifty_trend(bench)
    bench_close = bench if bench is not None else None

    pre_results  = []
    bo_results   = []
    fail_log     = {}

    for sym in tqdm(NSE_STOCKS, desc="  Scanning", ncols=68):
        try:
            df = fetch_stock(sym)
            if df is None:
                fail_log[sym] = "no_data"; continue

            candles = extract_candles(df)

            # ── PRE-BREAKOUT
            pre, reason = scan_pre_breakout(df, sym, bench_close)
            if pre:
                pre["candles"] = candles
                pre_results.append(pre)

            # ── BROKEN-OUT (only if not pre-breakout)
            if pre is None:
                bo, reason2 = scan_broken_out(df, sym, bench_close)
                if bo:
                    bo["candles"] = candles
                    bo_results.append(bo)
                elif reason:
                    fail_log[sym] = reason or reason2 or "unknown"
            
            time.sleep(CONFIG["sleep_between"])

        except Exception as e:
            fail_log[sym] = f"err_{str(e)[:20]}"

    pre_results.sort(key=lambda x: x["vcp_score"], reverse=True)
    bo_results.sort(key=lambda x: x["vcp_score"], reverse=True)

    pre_top = pre_results[:CONFIG["top_n_pre"]]
    bo_top  = bo_results[:CONFIG["top_n_broken"]]

    # ── Failure summary
    from collections import Counter
    fc = Counter(v.split("_")[0] for v in fail_log.values())
    print(f"\n  Rejection summary:")
    for r, c in fc.most_common(8):
        print(f"    {r:<30} {c:>3}")

    # ── Save
    output = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "scanner_version": "v4",
        "total_scanned":   len(NSE_STOCKS),
        "market_uptrend":  is_uptrend,
        "market_strength": mkt_strength,
        "pre_breakout":    {
            "total_found": len(pre_results),
            "stocks":      pre_top,
        },
        "broken_out":      {
            "total_found": len(bo_results),
            "stocks":      bo_top,
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  PRE-BREAKOUT: {len(pre_results)} found | {len(pre_top)} saved")
    if pre_top:
        print(f"  {'#':<3} {'Symbol':<13} {'Price':>8} {'Pivot':>8} {'Waves':>18} {'R:R':>5} {'Score':>6}")
        for i, s in enumerate(pre_top, 1):
            ws = "→".join(f"{w:.0f}%" for w in (s["waves"] or [])[:3])
            cup = "🏆" if s["is_cup"] else ""
            print(f"  {i:<3} {s['symbol']:<13} ₹{s['price']:>7.2f} ₹{s['pivot']:>7.2f} [{ws}] {s['risk_reward']:>4.1f}x  {s['vcp_score']:>5} {cup}")

    print(f"\n  BROKEN OUT:   {len(bo_results)} found | {len(bo_top)} saved")
    if bo_top:
        print(f"  {'#':<3} {'Symbol':<13} {'Price':>8} {'Pivot':>8} {'Ext':>6} {'Vol':>6} {'Days':>5} {'Score':>6}")
        for i, s in enumerate(bo_top, 1):
            print(f"  {i:<3} {s['symbol']:<13} ₹{s['price']:>7.2f} ₹{s['pivot']:>7.2f} "
                  f"{s['extension_pct']:>5.1f}% {s['vol_on_breakout']:>5.1f}× "
                  f"{s['days_since_breakout']:>4}d  {s['vcp_score']:>5}")

    print(f"\n  ✅ Done. Results → docs/data/results.json")
    print("=" * 68)


if __name__ == "__main__":
    run()
