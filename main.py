import os
import math
import threading
from typing import Optional, Callable, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import ollama
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import customtkinter as ctk

# Optional .env support (recommended)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass


# ============================================================
# CONFIG (Env-driven defaults)
# ============================================================
AI_PROVIDER_DEFAULT = os.getenv("AI_PROVIDER", "ollama").strip().lower()

# DeepSeek via Hugging Face Router (OpenAI-compatible)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1").strip()
DEEPSEEK_MODEL_DEFAULT = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1").strip()
DEEPSEEK_MAX_TOKENS_DEFAULT = int(os.getenv("DEEPSEEK_MAX_TOKENS", "700"))
DEEPSEEK_TEMPERATURE_DEFAULT = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

# Ollama
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "llama3:latest").strip()

# UI constants (fixed sidebar)
SIDEBAR_WIDTH = 280
SIDEBAR_PAD_X = 20
SIDEBAR_INNER_WIDTH = SIDEBAR_WIDTH - (SIDEBAR_PAD_X * 2)

# Projection config
PROJ_HORIZON = 5
PROJ_MIN_SAMPLES = 30


# ============================================================
# HELPERS
# ============================================================
def _sf(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x, nd=2):
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "N/A"
        return f"{v:,.{nd}f}"
    except Exception:
        return "N/A"


def fmt_pct(x, nd=2):
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "N/A"
        return f"{v * 100:.{nd}f}%"
    except Exception:
        return "N/A"


def fmt_money(x, nd=2):
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "N/A"
        return f"${v:,.{nd}f}"
    except Exception:
        return "N/A"


def fmt_big_money(x):
    if x is None:
        return "N/A"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "N/A"
        av = abs(v)
        if av >= 1e12:
            return f"${v/1e12:,.2f}T"
        if av >= 1e9:
            return f"${v/1e9:,.2f}B"
        if av >= 1e6:
            return f"${v/1e6:,.2f}M"
        if av >= 1e3:
            return f"${v/1e3:,.2f}K"
        return f"${v:,.0f}"
    except Exception:
        return "N/A"


def first_col_like(df: pd.DataFrame, prefix: str) -> Optional[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return cols[0] if cols else None


def last_valid_value(df: pd.DataFrame, col: Optional[str]) -> Optional[float]:
    if not col or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return None
    v = float(s.iloc[-1])
    return v if math.isfinite(v) else None


def latest_statement_value(stmt: pd.DataFrame, row_name: str) -> Optional[float]:
    try:
        if stmt is None or stmt.empty:
            return None
        if row_name not in stmt.index:
            return None
        latest_col = stmt.columns[0]
        return _sf(stmt.loc[row_name, latest_col], None)
    except Exception:
        return None


def is_finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _pick_first_finite(*vals):
    for v in vals:
        if is_finite(v):
            return float(v)
    return None


def _round_like_price(px: float) -> float:
    if px >= 1:
        return round(px, 4)
    return round(px, 6)


def _period_for(interval: str) -> str:
    interval = (interval or "").strip().lower()
    if interval == "15m":
        return "60d"
    if interval == "1h":
        return "180d"
    if interval == "1d":
        return "2y"
    return "60d"


def _normalize_index_to_naive_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        idx = pd.to_datetime(out.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        out.index = idx
    except Exception:
        pass
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _get_dates_index(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.to_datetime(idx).date


# ============================================================
# SIMPLE SHORT-HORIZON PROJECTION (NEXT N CANDLES)
# ============================================================
def forecast_next_closes(
    df: pd.DataFrame,
    interval: str,
    horizon: int = 5,
    lookback: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Produces a short-horizon probabilistic path for the next `horizon` candles using
    a lightweight AR(1) model on log returns.

    Returns:
      {
        "base": [p1..pH],
        "upper_1s": [...],
        "lower_1s": [...],
        "mu": float, "phi": float, "sigma": float,
        "method": "AR1-logret",
      }
    """
    out: Dict[str, Any] = {
        "base": [],
        "upper_1s": [],
        "lower_1s": [],
        "mu": None,
        "phi": None,
        "sigma": None,
        "method": "AR1-logret",
    }

    if df is None or df.empty or "Close" not in df.columns:
        return out

    interval = (interval or "15m").strip().lower()
    if lookback is None:
        lookback = 300 if interval in ("15m", "1h") else 252

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < PROJ_MIN_SAMPLES:
        return out

    close = close.iloc[-lookback:].copy()
    logp = np.log(close)
    r = logp.diff().dropna()  # log returns

    if len(r) < PROJ_MIN_SAMPLES:
        return out

    r_np = r.to_numpy(dtype=float)

    # Fit AR(1): r_t = a + phi * r_{t-1} + e_t
    # If too little variation, fallback to drift-only.
    x = r_np[:-1]
    y = r_np[1:]
    if np.std(x) < 1e-12:
        a = float(np.mean(r_np))
        phi = 0.0
        resid = y - a
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(r_np, ddof=1))
    else:
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        phi = float(beta[1])
        resid = y - (a + phi * x)
        sigma = float(np.std(resid, ddof=2)) if len(resid) > 3 else float(np.std(resid, ddof=1))

    last_r = float(r_np[-1])
    rhat: List[float] = []
    for _ in range(int(horizon)):
        last_r = a + phi * last_r
        rhat.append(float(last_r))

    last_close = float(close.iloc[-1])
    csum = np.cumsum(np.array(rhat, dtype=float))
    base = last_close * np.exp(csum)

    # 1-sigma band in log space: variance grows ~ h * sigma^2
    steps = np.arange(1, horizon + 1, dtype=float)
    sig_h = sigma * np.sqrt(steps)
    upper = last_close * np.exp(csum + sig_h)
    lower = last_close * np.exp(csum - sig_h)

    out["base"] = [float(v) for v in base]
    out["upper_1s"] = [float(v) for v in upper]
    out["lower_1s"] = [float(v) for v in lower]
    out["mu"] = a
    out["phi"] = phi
    out["sigma"] = sigma
    return out


def projection_text(proj: Dict[str, Any], interval: str, nd: int = 4) -> str:
    if not proj or not proj.get("base"):
        return "Projection unavailable (insufficient data)."

    base = proj["base"]
    lo = proj.get("lower_1s", [])
    hi = proj.get("upper_1s", [])
    mu = proj.get("mu")
    phi = proj.get("phi")
    sigma = proj.get("sigma")

    lines = []
    lines.append(f"Method: {proj.get('method','N/A')}  |  Interval: {interval.upper()}  |  Horizon: next {len(base)} candles")
    lines.append(f"Model params (log-returns): mu={fmt_num(mu, 8)}  phi={fmt_num(phi, 4)}  sigma={fmt_num(sigma, 8)}")
    lines.append("")
    lines.append("Next-candle projected closes (base) and 68% range (±1σ):")

    for i in range(len(base)):
        b = base[i]
        l = lo[i] if i < len(lo) else None
        h = hi[i] if i < len(hi) else None
        if l is not None and h is not None:
            lines.append(f"  +{i+1}:  {fmt_money(b, nd)}   (range: {fmt_money(l, nd)}  –  {fmt_money(h, nd)})")
        else:
            lines.append(f"  +{i+1}:  {fmt_money(b, nd)}")

    return "\n".join(lines)


def future_index(last_idx: pd.Timestamp, interval: str, horizon: int) -> pd.DatetimeIndex:
    interval = (interval or "15m").strip().lower()
    if interval == "15m":
        freq = "15min"
    elif interval == "1h":
        freq = "1H"
    else:
        freq = "1D"
    # Generate horizon future points (excluding last)
    return pd.date_range(start=last_idx, periods=horizon + 1, freq=freq)[1:]


# ============================================================
# MARKET STRUCTURE HELPERS
# ============================================================
def add_intraday_vwap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out = _normalize_index_to_naive_datetime(out)
        dates = _get_dates_index(out)
        tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
        pv = tp * out["Volume"]
        out["_date"] = dates
        out["VWAP"] = pv.groupby(out["_date"]).cumsum() / out["Volume"].groupby(out["_date"]).cumsum()
        out.drop(columns=["_date"], inplace=True, errors="ignore")
    except Exception:
        pass
    return out


def add_opening_range(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    out = df.copy()
    try:
        out = _normalize_index_to_naive_datetime(out)
        interval = (interval or "").lower()
        if interval == "1d":
            out["ORB_H"] = np.nan
            out["ORB_L"] = np.nan
            return out

        bars = 2 if interval == "15m" else 1
        out["_date"] = _get_dates_index(out)

        def _orb_hi(x: pd.DataFrame) -> float:
            return float(x["High"].iloc[:bars].max()) if len(x) else np.nan

        def _orb_lo(x: pd.DataFrame) -> float:
            return float(x["Low"].iloc[:bars].min()) if len(x) else np.nan

        or_hi = out.groupby("_date", sort=False).apply(_orb_hi)
        or_lo = out.groupby("_date", sort=False).apply(_orb_lo)

        out["ORB_H"] = out["_date"].map(or_hi)
        out["ORB_L"] = out["_date"].map(or_lo)
        out.drop(columns=["_date"], inplace=True, errors="ignore")
    except Exception:
        pass
    return out


def add_session_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out = _normalize_index_to_naive_datetime(out)
        out["_date"] = _get_dates_index(out)

        out["SES_OPEN"] = out.groupby("_date", sort=False)["Open"].transform("first")
        out["SES_RET"] = (out["Close"] / out["SES_OPEN"]) - 1.0

        out["SES_HI"] = out.groupby("_date", sort=False)["High"].cummax()
        out["SES_LO"] = out.groupby("_date", sort=False)["Low"].cummin()

        out.drop(columns=["_date"], inplace=True, errors="ignore")
    except Exception:
        pass
    return out


def add_pivot_levels(df: pd.DataFrame, left: int = 3, right: int = 3, price_round: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    try:
        out = _normalize_index_to_naive_datetime(out)
        win = left + right + 1
        ph = out["High"].rolling(win, center=True).max()
        pl = out["Low"].rolling(win, center=True).min()

        out["PIVOT_H"] = np.where(out["High"] == ph, out["High"], np.nan)
        out["PIVOT_L"] = np.where(out["Low"] == pl, out["Low"], np.nan)

        curr = float(out["Close"].iloc[-1])
        if price_round is None:
            price_round = max(0.05, 0.0015 * curr)

        pivots = pd.concat(
            [
                out[["PIVOT_H"]].rename(columns={"PIVOT_H": "PX"}),
                out[["PIVOT_L"]].rename(columns={"PIVOT_L": "PX"}),
            ],
            axis=0,
        ).dropna()

        if pivots.empty:
            out["SUP_MAIN"] = np.nan
            out["RES_MAIN"] = np.nan
            out["SUP_2"] = np.nan
            out["RES_2"] = np.nan
            return out

        pivots["LVL"] = (pivots["PX"] / price_round).round() * price_round
        lvls = np.array(sorted(pivots["LVL"].unique()), dtype=float)

        below = lvls[lvls < curr]
        above = lvls[lvls > curr]

        sup1 = float(below.max()) if below.size else np.nan
        res1 = float(above.min()) if above.size else np.nan

        sup2 = float(np.sort(below)[-2]) if below.size >= 2 else np.nan
        res2 = float(np.sort(above)[1]) if above.size >= 2 else np.nan

        out["SUP_MAIN"] = sup1
        out["RES_MAIN"] = res1
        out["SUP_2"] = sup2
        out["RES_2"] = res2
        return out
    except Exception:
        return out


def build_intraday_call_and_plan(
    curr: float,
    interval: str,
    atr: Optional[float],
    vwap_like: Optional[float],
    orb_h: Optional[float],
    orb_l: Optional[float],
    sup1: Optional[float],
    res1: Optional[float],
    sup2: Optional[float],
    res2: Optional[float],
    session_ret: Optional[float],
    ema20_slope: Optional[float],
    ema50_slope: Optional[float],
    adx: Optional[float],
):
    """
    Deterministic decision engine:
    - Returns CALL/STRATEGY/PLAN with directionally consistent levels.
    """
    interval = (interval or "").lower()
    atr_eff = _pick_first_finite(atr, max(0.002 * curr, 0.05))
    vwap_eff = _pick_first_finite(vwap_like)
    orb_h_eff = _pick_first_finite(orb_h)
    orb_l_eff = _pick_first_finite(orb_l)
    sup1_eff = _pick_first_finite(sup1)
    res1_eff = _pick_first_finite(res1)
    sup2_eff = _pick_first_finite(sup2)
    res2_eff = _pick_first_finite(res2)

    micro = 0
    if is_finite(vwap_eff):
        micro += 2 if curr >= vwap_eff else -2
    if is_finite(session_ret):
        micro += 1 if session_ret >= 0 else -1
    if is_finite(ema20_slope):
        micro += 1 if ema20_slope >= 0 else -1
    if is_finite(ema50_slope):
        micro += 1 if ema50_slope >= 0 else -1

    if interval in ("15m", "1h"):
        if is_finite(orb_h_eff) and curr > orb_h_eff:
            micro += 2
        if is_finite(orb_l_eff) and curr < orb_l_eff:
            micro -= 2

    strength = "WEAK"
    if is_finite(adx):
        if adx >= 25:
            strength = "STRONG"
            micro += 1
        elif adx >= 18:
            strength = "OK"
        else:
            strength = "WEAK"

    if micro >= 3:
        call = "Bullish"
    elif micro <= -3:
        call = "Bearish"
    else:
        call = "Neutral"

    if interval == "1d":
        if call == "Bullish":
            strategy = "Buy Pullbacks"
        else:
            strategy = "Wait for Confirmation"
    else:
        if call == "Bullish":
            strategy = "Breakout Entry" if (is_finite(orb_h_eff) and curr > orb_h_eff) else "Buy Pullbacks"
        elif call == "Bearish":
            strategy = "Breakout Entry" if (is_finite(orb_l_eff) and curr < orb_l_eff) else "Wait for Confirmation"
        else:
            strategy = "Wait for Confirmation"

    if call == "Bullish":
        pullback_anchor = None
        if is_finite(vwap_eff) and vwap_eff <= curr:
            pullback_anchor = vwap_eff
        if is_finite(sup1_eff) and sup1_eff <= curr:
            pullback_anchor = max(pullback_anchor, sup1_eff) if pullback_anchor is not None else sup1_eff

        entry = pullback_anchor if pullback_anchor is not None else curr

        stop_anchor = _pick_first_finite(sup1_eff, (orb_l_eff if interval in ("15m", "1h") else None), sup2_eff)
        if stop_anchor is None:
            stop = entry - 1.2 * atr_eff
        else:
            stop = min(stop_anchor - 0.35 * atr_eff, entry - 0.8 * atr_eff)

        t1 = _pick_first_finite(res1_eff, entry + 1.6 * atr_eff)
        t2 = _pick_first_finite(res2_eff, entry + 2.6 * atr_eff)

        stop = min(stop, entry - 0.3 * atr_eff)
        t1 = max(t1, entry + 0.6 * atr_eff)
        t2 = max(t2, t1 + 0.6 * atr_eff)

    elif call == "Bearish":
        pullback_anchor = None
        if is_finite(vwap_eff) and vwap_eff >= curr:
            pullback_anchor = vwap_eff
        if is_finite(res1_eff) and res1_eff >= curr:
            pullback_anchor = min(pullback_anchor, res1_eff) if pullback_anchor is not None else res1_eff

        entry = pullback_anchor if pullback_anchor is not None else curr

        stop_anchor = _pick_first_finite(res1_eff, (orb_h_eff if interval in ("15m", "1h") else None), res2_eff)
        if stop_anchor is None:
            stop = entry + 1.2 * atr_eff
        else:
            stop = max(stop_anchor + 0.35 * atr_eff, entry + 0.8 * atr_eff)

        t1 = _pick_first_finite(sup1_eff, entry - 1.6 * atr_eff)
        t2 = _pick_first_finite(sup2_eff, entry - 2.6 * atr_eff)

        stop = max(stop, entry + 0.3 * atr_eff)
        t1 = min(t1, entry - 0.6 * atr_eff)
        t2 = min(t2, t1 - 0.6 * atr_eff)

    else:
        entry = curr
        stop = curr
        t1 = curr
        t2 = curr

    if interval == "15m":
        time_horizon = "same-day (next 2–6 hours)"
    elif interval == "1h":
        time_horizon = "same-day (next 4–8 hours)"
    else:
        time_horizon = "swing (next 3–15 sessions)"

    plan = {
        "call": call,
        "strategy": strategy,
        "entry": _round_like_price(entry),
        "stop": _round_like_price(stop),
        "t1": _round_like_price(t1),
        "t2": _round_like_price(t2),
        "time_horizon": time_horizon,
        "micro_score": micro,
        "strength": strength,
    }
    return call, strategy, plan


# ============================================================
# LLM ROUTER (Ollama vs DeepSeek via HF Router)
# ============================================================
def run_llm(
    prompt: str,
    provider: str,
    ollama_model: str,
    hf_token: str,
    hf_base_url: str,
    deepseek_model: str,
    deepseek_max_tokens: int,
    deepseek_temperature: float,
) -> str:
    provider = (provider or "").strip().lower()

    if provider == "ollama":
        try:
            return ollama.generate(model=ollama_model, prompt=prompt)["response"]
        except Exception as e:
            return f"AI (Ollama) error: {e}"

    if provider == "deepseek":
        if not hf_token:
            return "AI (DeepSeek) error: HF_TOKEN is missing. Set it in your environment or .env."
        try:
            from openai import OpenAI  # pip install openai
        except Exception:
            return "AI (DeepSeek) error: openai package not installed. Run: pip install openai"

        try:
            client = OpenAI(api_key=hf_token, base_url=hf_base_url)
            resp = client.chat.completions.create(
                model=deepseek_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(deepseek_max_tokens),
                temperature=float(deepseek_temperature),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"AI (DeepSeek) error: {e}"

    return f"AI error: Unknown provider '{provider}'. Choose 'ollama' or 'deepseek'."


# ============================================================
# DATA ENGINE
# ============================================================
def get_pro_analysis(
    ticker_symbol: str,
    analysis_interval: str,
    progress_cb: Optional[Callable[[float, str], None]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    try:
        interval = (analysis_interval or "15m").strip().lower()
        if interval not in ("15m", "1h", "1d"):
            interval = "15m"

        period = _period_for(interval)

        if progress_cb:
            progress_cb(0.06, f"Fetching market data ({period} @ {interval})...")

        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            return None, None

        df = df.copy().dropna()
        df = _normalize_index_to_naive_datetime(df)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            return None, None

        if progress_cb:
            progress_cb(0.12, "Computing 52-week high/low context...")

        df_daily = df if interval == "1d" else ticker.history(period="2y", interval="1d", auto_adjust=True)
        hi_52w, lo_52w = np.nan, np.nan
        if df_daily is not None and not df_daily.empty and len(df_daily) >= 252:
            df_daily = df_daily.dropna()
            hi_52w = float(df_daily["High"].rolling(252).max().iloc[-1])
            lo_52w = float(df_daily["Low"].rolling(252).min().iloc[-1])

        if progress_cb:
            progress_cb(0.18, "Computing indicators (trend/momentum/vol/flow)...")

        if interval in ("15m", "1h"):
            df = add_intraday_vwap(df)
            df = add_opening_range(df, interval=interval)
            df = add_session_stats(df)
        else:
            df.ta.vwma(length=20, append=True)
            vwma_col = first_col_like(df, "VWMA_")
            df["VWAP"] = df[vwma_col] if vwma_col and vwma_col in df.columns else np.nan
            df["ORB_H"] = np.nan
            df["ORB_L"] = np.nan
            df["SES_OPEN"] = df["Open"]
            df["SES_RET"] = (df["Close"] / df["Open"]) - 1.0

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.sma(length=200, append=True)

        df.ta.rsi(length=14, append=True)
        df.ta.roc(length=10, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, append=True)

        df.ta.obv(append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.cmf(length=20, append=True)

        try:
            df.ta.supertrend(length=10, multiplier=3.0, append=True)
        except Exception:
            pass

        df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()
        df["RVOL"] = df["Volume"] / df["VOL_MA20"]

        bbu = first_col_like(df, "BBU_")
        bbl = first_col_like(df, "BBL_")
        bbm = first_col_like(df, "BBM_")
        if bbu and bbl and bbm and bbu in df.columns and bbl in df.columns and bbm in df.columns:
            df["BB_WIDTH"] = (df[bbu] - df[bbl]) / df[bbm]

        ema20_col = first_col_like(df, "EMA_20")
        ema50_col = first_col_like(df, "EMA_50")
        if ema20_col:
            df["EMA20_SLOPE_4"] = df[ema20_col] - df[ema20_col].shift(4)
        if ema50_col:
            df["EMA50_SLOPE_4"] = df[ema50_col] - df[ema50_col].shift(4)

        if progress_cb:
            progress_cb(0.28, "Deriving liquidity levels (pivot clustering)...")

        df = add_pivot_levels(df, left=3, right=3)
        df["RES_50"] = df["RES_MAIN"]
        df["SUP_50"] = df["SUP_MAIN"]

        df["HI_52W"] = hi_52w
        df["LO_52W"] = lo_52w

        if progress_cb:
            progress_cb(0.36, "Loading fundamentals & statements...")

        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        try:
            fin = ticker.financials
        except Exception:
            fin = pd.DataFrame()

        try:
            cf = ticker.cashflow
        except Exception:
            cf = pd.DataFrame()

        market_cap = _sf(info.get("marketCap"), None)
        enterprise_value = _sf(info.get("enterpriseValue"), None)

        trailing_pe = _sf(info.get("trailingPE"), None)
        forward_pe = _sf(info.get("forwardPE"), None)
        peg = _sf(info.get("pegRatio"), None)
        ps = _sf(info.get("priceToSalesTrailing12Months"), None)
        pb = _sf(info.get("priceToBook"), None)

        beta = _sf(info.get("beta"), None)
        div_yield = _sf(info.get("dividendYield"), None)
        payout = _sf(info.get("payoutRatio"), None)

        roe = _sf(info.get("returnOnEquity"), None)
        roa = _sf(info.get("returnOnAssets"), None)
        gross_m = _sf(info.get("grossMargins"), None)
        op_m = _sf(info.get("operatingMargins"), None)
        net_m = _sf(info.get("profitMargins"), None)

        rev_growth = _sf(info.get("revenueGrowth"), None)
        earn_growth = _sf(info.get("earningsGrowth"), None)

        current_ratio = _sf(info.get("currentRatio"), None)
        quick_ratio = _sf(info.get("quickRatio"), None)
        debt_to_equity = _sf(info.get("debtToEquity"), None)

        total_cash = _sf(info.get("totalCash"), None)
        total_debt = _sf(info.get("totalDebt"), None)

        ebitda = _sf(info.get("ebitda"), None)
        if ebitda is None:
            ebitda = latest_statement_value(fin, "EBITDA")

        fcf = _sf(info.get("freeCashflow"), None)
        if fcf is None:
            cfo = latest_statement_value(cf, "Total Cash From Operating Activities")
            capex = latest_statement_value(cf, "Capital Expenditures")
            if cfo is not None and capex is not None:
                fcf = cfo + capex

        op_cf = latest_statement_value(cf, "Total Cash From Operating Activities")

        net_debt = None
        if total_debt is not None and total_cash is not None:
            net_debt = total_debt - total_cash

        ev_to_ebitda = None
        if enterprise_value is not None and ebitda not in (None, 0):
            ev_to_ebitda = enterprise_value / ebitda

        fcf_yield = None
        if market_cap not in (None, 0) and fcf is not None:
            fcf_yield = fcf / market_cap

        net_debt_to_ebitda = None
        if net_debt is not None and ebitda not in (None, 0):
            net_debt_to_ebitda = net_debt / ebitda

        short_pct_float = _sf(info.get("shortPercentOfFloat"), None)
        short_ratio = _sf(info.get("shortRatio"), None)
        inst_own = _sf(info.get("heldPercentInstitutions"), None)
        insider_own = _sf(info.get("heldPercentInsiders"), None)

        target = _sf(info.get("targetMeanPrice"), None)
        name = info.get("shortName") or info.get("longName") or ticker_symbol

        fund_dashboard = (
            f"KEY FUNDAMENTAL METRICS DASHBOARD\n"
            f"------------------------------------------------------------\n"
            f"Size / Risk\n"
            f"• Market Cap:        {fmt_big_money(market_cap)}\n"
            f"• Enterprise Value:  {fmt_big_money(enterprise_value)}\n"
            f"• Beta:              {fmt_num(beta, 2)}\n\n"
            f"Valuation\n"
            f"• P/E (TTM):         {fmt_num(trailing_pe, 2)}\n"
            f"• P/E (Forward):     {fmt_num(forward_pe, 2)}\n"
            f"• PEG:               {fmt_num(peg, 2)}\n"
            f"• P/S (TTM):         {fmt_num(ps, 2)}\n"
            f"• P/B:               {fmt_num(pb, 2)}\n"
            f"• EV/EBITDA:         {fmt_num(ev_to_ebitda, 2)}\n"
            f"• FCF Yield:         {fmt_pct(fcf_yield, 2)}\n\n"
            f"Growth\n"
            f"• Revenue Growth:    {fmt_pct(rev_growth, 2)}\n"
            f"• Earnings Growth:   {fmt_pct(earn_growth, 2)}\n\n"
            f"Profitability\n"
            f"• Gross Margin:      {fmt_pct(gross_m, 2)}\n"
            f"• Operating Margin:  {fmt_pct(op_m, 2)}\n"
            f"• Net Margin:        {fmt_pct(net_m, 2)}\n"
            f"• ROE:               {fmt_pct(roe, 2)}\n"
            f"• ROA:               {fmt_pct(roa, 2)}\n\n"
            f"Balance Sheet / Liquidity\n"
            f"• Current Ratio:     {fmt_num(current_ratio, 2)}\n"
            f"• Quick Ratio:       {fmt_num(quick_ratio, 2)}\n"
            f"• Debt/Equity:       {fmt_num(debt_to_equity, 2)}\n"
            f"• Total Debt:        {fmt_big_money(total_debt)}\n"
            f"• Total Cash:        {fmt_big_money(total_cash)}\n"
            f"• Net Debt:          {fmt_big_money(net_debt)}\n"
            f"• Net Debt/EBITDA:   {fmt_num(net_debt_to_ebitda, 2)}\n\n"
            f"Cash Flow / Shareholder Return\n"
            f"• Operating Cashflow:{fmt_big_money(op_cf)}\n"
            f"• Free Cash Flow:    {fmt_big_money(fcf)}\n"
            f"• Dividend Yield:    {fmt_pct(div_yield, 2)}\n"
            f"• Payout Ratio:      {fmt_pct(payout, 2)}\n\n"
            f"Positioning (if available)\n"
            f"• Institutional Own: {fmt_pct(inst_own, 2)}\n"
            f"• Insider Own:       {fmt_pct(insider_own, 2)}\n"
            f"• Short % of Float:  {fmt_pct(short_pct_float, 2)}\n"
            f"• Short Ratio (DTC): {fmt_num(short_ratio, 2)}\n"
        )

        fundamentals = {
            "name": name,
            "target": target,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "peg": peg,
            "ps": ps,
            "pb": pb,
            "ev_to_ebitda": ev_to_ebitda,
            "fcf_yield": fcf_yield,
            "rev_growth": rev_growth,
            "earn_growth": earn_growth,
            "gross_m": gross_m,
            "op_m": op_m,
            "net_m": net_m,
            "roe": roe,
            "roa": roa,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "debt_to_equity": debt_to_equity,
            "total_debt": total_debt,
            "total_cash": total_cash,
            "net_debt": net_debt,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "op_cf": op_cf,
            "fcf": fcf,
            "div_yield": div_yield,
            "payout": payout,
            "beta": beta,
            "short_pct_float": short_pct_float,
            "short_ratio": short_ratio,
            "inst_own": inst_own,
            "insider_own": insider_own,
            "dashboard_text": fund_dashboard,
            "analysis_interval": interval,
            "period": period,
        }

        if progress_cb:
            progress_cb(0.44, "Data engine complete. Preparing analysis...")

        return df, fundamentals

    except Exception as e:
        print(f"Deep Analysis Error: {e}")
        return None, None


# ============================================================
# GUI APPLICATION
# ============================================================
class TradingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.fund: Optional[dict] = None
        self.proj: Optional[dict] = None
        self.interval_used: str = "15m"

        self.title("AI Quant Research Terminal v8.3 (15m / 1h / 1D + Next-5 Projection)")
        self.geometry("1200x1250")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----------------------------
        # SIDEBAR (fixed width)
        # ----------------------------
        self.sidebar = ctk.CTkFrame(self, width=SIDEBAR_WIDTH)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="DEEP SCANNER", font=("Arial", 22, "bold")).pack(pady=(20, 10))

        self.ticker_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Ticker (e.g. AAPL)")
        self.ticker_entry.pack(pady=8, padx=SIDEBAR_PAD_X, fill="x")

        ctk.CTkLabel(self.sidebar, text="Analysis Interval", font=("Arial", 13, "bold")).pack(pady=(10, 4))
        self.interval_var = ctk.StringVar(value="15m")
        self.interval_menu = ctk.CTkOptionMenu(self.sidebar, values=["15m", "1h", "1d"], variable=self.interval_var)
        self.interval_menu.pack(padx=SIDEBAR_PAD_X, fill="x")

        ctk.CTkLabel(self.sidebar, text="AI Provider", font=("Arial", 13, "bold")).pack(pady=(16, 4))
        self.provider_var = ctk.StringVar(
            value=AI_PROVIDER_DEFAULT if AI_PROVIDER_DEFAULT in ("ollama", "deepseek") else "ollama"
        )
        self.provider_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["ollama", "deepseek"], variable=self.provider_var, command=self.on_provider_change
        )
        self.provider_menu.pack(padx=SIDEBAR_PAD_X, fill="x")

        self.provider_settings = ctk.CTkFrame(self.sidebar, fg_color="#141414")
        self.provider_settings.pack(padx=SIDEBAR_PAD_X, pady=(10, 0), fill="x")

        self.ollama_frame = ctk.CTkFrame(self.provider_settings, fg_color="transparent")
        ctk.CTkLabel(self.ollama_frame, text="Ollama Model", font=("Arial", 12, "bold")).pack(pady=(8, 4))
        self.ollama_model_var = ctk.StringVar(value=OLLAMA_MODEL_DEFAULT)
        self.ollama_model_entry = ctk.CTkEntry(self.ollama_frame, textvariable=self.ollama_model_var)
        self.ollama_model_entry.pack(fill="x", padx=8, pady=(0, 8))

        self.deepseek_frame = ctk.CTkFrame(self.provider_settings, fg_color="transparent")
        ctk.CTkLabel(self.deepseek_frame, text="DeepSeek Model", font=("Arial", 12, "bold")).pack(pady=(8, 4))
        self.deepseek_model_var = ctk.StringVar(value=DEEPSEEK_MODEL_DEFAULT)
        self.deepseek_model_entry = ctk.CTkEntry(self.deepseek_frame, textvariable=self.deepseek_model_var)
        self.deepseek_model_entry.pack(fill="x", padx=8)

        ctk.CTkLabel(self.deepseek_frame, text="Max Tokens", font=("Arial", 12, "bold")).pack(pady=(10, 4))
        self.deepseek_tokens_var = ctk.StringVar(value=str(DEEPSEEK_MAX_TOKENS_DEFAULT))
        self.deepseek_tokens_entry = ctk.CTkEntry(self.deepseek_frame, textvariable=self.deepseek_tokens_var)
        self.deepseek_tokens_entry.pack(fill="x", padx=8)

        ctk.CTkLabel(self.deepseek_frame, text="Temperature", font=("Arial", 12, "bold")).pack(pady=(10, 4))
        self.deepseek_temp_var = ctk.StringVar(value=str(DEEPSEEK_TEMPERATURE_DEFAULT))
        self.deepseek_temp_entry = ctk.CTkEntry(self.deepseek_frame, textvariable=self.deepseek_temp_var)
        self.deepseek_temp_entry.pack(fill="x", padx=8, pady=(0, 8))

        self.btn_run = ctk.CTkButton(self.sidebar, text="GENERATE REPORT", command=self.start_thread, fg_color="#1e3799")
        self.btn_run.pack(pady=(16, 10), padx=SIDEBAR_PAD_X, fill="x")

        self.progress_text = ctk.CTkLabel(
            self.sidebar,
            text="Progress: 0%  |  Standby",
            text_color="gray",
            width=SIDEBAR_INNER_WIDTH,
            anchor="w",
            justify="left",
            wraplength=SIDEBAR_INNER_WIDTH,
        )
        self.progress_text.pack(padx=SIDEBAR_PAD_X, pady=(0, 6), fill="x")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=SIDEBAR_INNER_WIDTH)
        self.progress_bar.pack(padx=SIDEBAR_PAD_X, pady=(0, 10), fill="x")
        self.progress_bar.set(0.0)

        self.status = ctk.CTkLabel(
            self.sidebar,
            text="System Standby",
            text_color="gray",
            width=SIDEBAR_INNER_WIDTH,
            anchor="w",
            justify="left",
            wraplength=SIDEBAR_INNER_WIDTH,
        )
        self.status.pack(side="bottom", padx=SIDEBAR_PAD_X, pady=20, fill="x")

        # ----------------------------
        # MAIN AREA
        # ----------------------------
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.ticker_label = ctk.CTkLabel(self.main_frame, text="--", font=("Arial", 28, "bold"))
        self.ticker_label.pack(pady=(10, 0))
        self.price_label = ctk.CTkLabel(self.main_frame, text="$0.00", font=("Arial", 56, "bold"), text_color="#3498db")
        self.price_label.pack(pady=(0, 5))

        self.tag_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.tag_frame.pack(fill="x", pady=10)
        self.val_tag = ctk.CTkLabel(
            self.tag_frame, text="VALUATION: --", font=("Arial", 16, "bold"), height=50, corner_radius=8, fg_color="#333"
        )
        self.val_tag.pack(side="left", padx=5, expand=True, fill="x")
        self.tech_tag = ctk.CTkLabel(
            self.tag_frame, text="TECHNICAL: --", font=("Arial", 16, "bold"), height=50, corner_radius=8, fg_color="#333"
        )
        self.tech_tag.pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkLabel(self.main_frame, text="MARKET GEOMETRY & LIQUIDITY LEVELS", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        self.levels_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a")
        self.levels_frame.pack(fill="x", padx=10, pady=5)

        self.res_lbl = ctk.CTkLabel(self.levels_frame, text="RESISTANCE: --", font=("Consolas", 18, "bold"), text_color="#ff7675")
        self.res_lbl.grid(row=0, column=0, padx=40, pady=15)
        self.sup_lbl = ctk.CTkLabel(self.levels_frame, text="SUPPORT: --", font=("Consolas", 18, "bold"), text_color="#55efc4")
        self.sup_lbl.grid(row=0, column=1, padx=40, pady=15)
        self.hi52_lbl = ctk.CTkLabel(self.levels_frame, text="52W HIGH: --", font=("Consolas", 18, "bold"), text_color="#feca57")
        self.hi52_lbl.grid(row=1, column=0, padx=40, pady=(0, 15))
        self.lo52_lbl = ctk.CTkLabel(self.levels_frame, text="52W LOW: --", font=("Consolas", 18, "bold"), text_color="#54a0ff")
        self.lo52_lbl.grid(row=1, column=1, padx=40, pady=(0, 15))

        ctk.CTkLabel(self.main_frame, text="EXECUTION PLAN (DETERMINISTIC, ATR-AWARE)", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        self.exec_frame = ctk.CTkFrame(self.main_frame, fg_color="#0c2461", corner_radius=10)
        self.exec_frame.pack(fill="x", padx=10, pady=5)

        self.buy_lbl = ctk.CTkLabel(self.exec_frame, text="ENTRY: --", font=("Consolas", 18, "bold"))
        self.buy_lbl.grid(row=0, column=0, padx=30, pady=20)
        self.sl_lbl = ctk.CTkLabel(self.exec_frame, text="STOP: --", font=("Consolas", 18, "bold"), text_color="#ff7675")
        self.sl_lbl.grid(row=0, column=1, padx=30, pady=20)
        self.tp_lbl = ctk.CTkLabel(self.exec_frame, text="T1: --", font=("Consolas", 18, "bold"), text_color="#55efc4")
        self.tp_lbl.grid(row=0, column=2, padx=30, pady=20)

        ctk.CTkLabel(self.main_frame, text="NEXT 5 CANDLE PROJECTION (BASE + 68% RANGE)", font=("Arial", 13, "bold")).pack(pady=(15, 0))
        self.proj_box = ctk.CTkTextbox(self.main_frame, height=170, font=("Consolas", 13), fg_color="#1a1a1a")
        self.proj_box.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.main_frame, text="QUANTITATIVE RAW DATA (TECHNICAL SNAPSHOT)", font=("Arial", 13, "bold")).pack(pady=(15, 0))
        self.metrics_box = ctk.CTkTextbox(self.main_frame, height=250, font=("Consolas", 14), fg_color="#1a1a1a")
        self.metrics_box.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            self.main_frame,
            text="KEY FUNDAMENTAL METRICS (VALUATION • GROWTH • CASH FLOW • BALANCE SHEET)",
            font=("Arial", 13, "bold"),
            text_color="#82ccdd",
        ).pack(pady=(15, 0))
        self.fund_metrics_box = ctk.CTkTextbox(self.main_frame, height=300, font=("Consolas", 13), fg_color="#1a1a1a")
        self.fund_metrics_box.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            self.main_frame,
            text="DEEP FUNDAMENTAL ANALYSIS (MOAT • QUALITY • VALUATION • RISKS)",
            font=("Arial", 13, "bold"),
            text_color="#38ada9",
        ).pack(pady=(15, 0))
        self.fund_ai = ctk.CTkTextbox(self.main_frame, height=220, wrap="word")
        self.fund_ai.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            self.main_frame,
            text="DEEP TECHNICAL ANALYSIS (SESSION/VWAP • CONFLUENCE • LEVELS • PLAN • PROJECTION)",
            font=("Arial", 13, "bold"),
            text_color="#e58e26",
        ).pack(pady=(15, 0))
        self.tech_ai = ctk.CTkTextbox(self.main_frame, height=220, wrap="word")
        self.tech_ai.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            self.main_frame,
            text="CONCLUSION & STRATEGY (INTEGRATED INTERPRETATION)",
            font=("Arial", 13, "bold"),
            text_color="#b8e994",
        ).pack(pady=(15, 0))
        self.conclusion_ai = ctk.CTkTextbox(self.main_frame, height=240, wrap="word")
        self.conclusion_ai.pack(fill="x", padx=10, pady=5)

        self.btn_chart = ctk.CTkButton(
            self.main_frame,
            text="Open Plotly Terminal Chart (Multi-Pane + Projection)",
            command=self.show_chart,
            state="disabled",
        )
        self.btn_chart.pack(pady=20)

        self.on_provider_change(self.provider_var.get())

    def _set_progress(self, value: float, message: str, status_color: str = "orange"):
        value = max(0.0, min(1.0, float(value)))
        if message and len(message) > 70:
            message = message[:67] + "..."

        def _apply():
            self.progress_bar.set(value)
            self.progress_text.configure(text=f"Progress: {int(value * 100)}%  |  {message}")
            self.status.configure(text=message, text_color=status_color)

        self.after(0, _apply)

    def _progress_error(self, message: str = "Data Error"):
        def _apply():
            self.progress_bar.set(0.0)
            self.progress_text.configure(text=f"Progress: 0%  |  {message}")
            self.status.configure(text=message, text_color="red")
            self.btn_run.configure(state="normal")

        self.after(0, _apply)

    def on_provider_change(self, provider_value: str):
        provider_value = (provider_value or "").strip().lower()
        for child in self.provider_settings.winfo_children():
            child.pack_forget()

        if provider_value == "ollama":
            self.ollama_frame.pack(fill="x")
        else:
            self.deepseek_frame.pack(fill="x")

    def start_thread(self):
        ticker = self.ticker_entry.get().upper().strip()
        if not ticker:
            return

        interval = self.interval_var.get().strip().lower()
        self.interval_used = interval

        provider = self.provider_var.get().strip().lower()
        self.btn_run.configure(state="disabled")
        self._set_progress(0.02, f"Initializing ({ticker}, {provider}, {interval})...")

        threading.Thread(target=self.process, args=(ticker, interval), daemon=True).start()

    def process(self, ticker: str, interval: str):
        def progress_cb(p: float, msg: str):
            self._set_progress(p, msg, status_color="orange")

        self._set_progress(0.04, "Starting data engine...")

        df, f = get_pro_analysis(ticker, analysis_interval=interval, progress_cb=progress_cb)
        if df is None or f is None:
            self._progress_error("Data Error (ticker invalid or insufficient history)")
            return

        self.df = df
        self.fund = f
        curr = float(df["Close"].iloc[-1])

        self._set_progress(0.46, "Computing deterministic plan + projection...")

        rsi_col = first_col_like(df, "RSI")
        adx_col = first_col_like(df, "ADX")
        atr_col = first_col_like(df, "ATR")
        ema50_col = first_col_like(df, "EMA_50")
        sma200_col = first_col_like(df, "SMA_200")
        macd_line_col = first_col_like(df, "MACD_")
        macd_hist_col = first_col_like(df, "MACDh_")
        macd_sig_col = first_col_like(df, "MACDs_")

        roc_col = first_col_like(df, "ROC_")
        cmf_col = first_col_like(df, "CMF_")

        rsi = last_valid_value(df, rsi_col)
        adx = last_valid_value(df, adx_col)
        atr = last_valid_value(df, atr_col)
        ema50 = last_valid_value(df, ema50_col)
        sma200 = last_valid_value(df, sma200_col)

        macd_line = last_valid_value(df, macd_line_col)
        macd_hist = last_valid_value(df, macd_hist_col)
        macd_sig = last_valid_value(df, macd_sig_col)

        roc10 = last_valid_value(df, roc_col)
        cmf20 = last_valid_value(df, cmf_col)

        rvol = last_valid_value(df, "RVOL")
        bb_width = last_valid_value(df, "BB_WIDTH")

        vwap_like = last_valid_value(df, "VWAP")
        orb_h = last_valid_value(df, "ORB_H")
        orb_l = last_valid_value(df, "ORB_L")
        ses_ret = last_valid_value(df, "SES_RET")

        res_50 = last_valid_value(df, "RES_50")
        sup_50 = last_valid_value(df, "SUP_50")
        res_2 = last_valid_value(df, "RES_2")
        sup_2 = last_valid_value(df, "SUP_2")

        hi_52w = last_valid_value(df, "HI_52W")
        lo_52w = last_valid_value(df, "LO_52W")

        ema20_slope = last_valid_value(df, "EMA20_SLOPE_4")
        ema50_slope = last_valid_value(df, "EMA50_SLOPE_4")

        # --- Projection (next 5 candles)
        proj = forecast_next_closes(df, interval=interval, horizon=PROJ_HORIZON)
        self.proj = proj
        proj_txt = projection_text(proj, interval=interval, nd=4)
        proj_p5 = proj["base"][-1] if proj and proj.get("base") else None

        # Valuation overlay score (heuristic)
        score = 0
        if f.get("fcf_yield") is not None:
            score += 1 if f["fcf_yield"] >= 0.05 else (-1 if f["fcf_yield"] <= 0.02 else 0)
        if f.get("ev_to_ebitda") is not None:
            score += 1 if f["ev_to_ebitda"] <= 12 else (-1 if f["ev_to_ebitda"] >= 20 else 0)
        if f.get("forward_pe") is not None:
            score += 1 if f["forward_pe"] <= 18 else (-1 if f["forward_pe"] >= 30 else 0)
        if f.get("roe") is not None:
            score += 1 if f["roe"] >= 0.15 else (-1 if f["roe"] <= 0.07 else 0)
        if f.get("net_debt_to_ebitda") is not None:
            score += 1 if f["net_debt_to_ebitda"] <= 2 else (-1 if f["net_debt_to_ebitda"] >= 4 else 0)

        overlay = "ATTRACTIVE" if score >= 2 else "EXPENSIVE / RISK" if score <= -2 else "NEUTRAL"

        target = f.get("target")
        if target not in (None, 0):
            if target > curr * 1.10:
                v_s = "UNDERVALUED"
            elif target < curr * 0.90:
                v_s = "OVERVALUED"
            else:
                v_s = "FAIR VALUE"
        else:
            v_s = overlay

        v_c = "#2ecc71" if ("UNDER" in v_s or "ATTRACTIVE" in v_s) else "#e74c3c" if ("OVER" in v_s or "EXPENSIVE" in v_s) else "#3498db"

        call, strategy, plan = build_intraday_call_and_plan(
            curr=curr,
            interval=interval,
            atr=atr,
            vwap_like=vwap_like,
            orb_h=orb_h,
            orb_l=orb_l,
            sup1=sup_50,
            res1=res_50,
            sup2=sup_2,
            res2=res_2,
            session_ret=ses_ret,
            ema20_slope=ema20_slope,
            ema50_slope=ema50_slope,
            adx=adx,
        )

        # Enrich the "strategy" with a forward-looking projection hook (still deterministic)
        if proj_p5 is not None:
            strategy_with_proj = f"{strategy} | Probable price in next {PROJ_HORIZON} candles (base): {fmt_money(proj_p5,4)}"
        else:
            strategy_with_proj = strategy

        t_s = f"{call} | {plan['strength']} | micro={plan['micro_score']} [{interval.upper()}]"
        t_c = "#2ecc71" if call == "Bullish" else "#e74c3c" if call == "Bearish" else "#3498db"

        vwap_label = "VWAP" if interval in ("15m", "1h") else "VWMA(20)"
        orb_str = f"{fmt_money(orb_h,4)} / {fmt_money(orb_l,4)}" if interval in ("15m", "1h") else "N/A"

        metric_txt = (
            f"--- INTERVAL ({interval.upper()}) ---                          --- SESSION / FLOW ---\n"
            f"Price:         {fmt_money(curr,4):<15}  Session Ret:     {fmt_pct(ses_ret, 2)}\n"
            f"{vwap_label}:        {fmt_money(vwap_like,4):<15}  ORB High/Low:    {orb_str}\n"
            f"S1/R1:         {fmt_money(sup_50,4)} / {fmt_money(res_50,4)}   S2/R2: {fmt_money(sup_2,4)} / {fmt_money(res_2,4)}\n"
            f"RSI (14):      {fmt_num(rsi, 2):<15}  ADX (14):        {fmt_num(adx, 2)}\n"
            f"ATR (14):      {fmt_money(atr, 4):<15}  BB Width:       {fmt_num(bb_width, 4)}\n"
            f"RVOL (20):     {fmt_num(rvol, 2):<15}  CMF (20):        {fmt_num(cmf20, 3)}\n"
            f"ROC (10):      {fmt_num(roc10, 2):<15}  EMA20 slope:    {fmt_num(ema20_slope, 6)}\n"
            f"EMA50:         {fmt_money(ema50, 4):<15}  SMA200:         {fmt_money(sma200, 4)}\n"
            f"MACD:          {fmt_num(macd_line, 5):<15}  Signal:         {fmt_num(macd_sig, 5)}\n"
            f"MACD Hist:     {fmt_num(macd_hist, 5):<15}\n"
            f"52W High/Low:  {fmt_money(hi_52w,2)} / {fmt_money(lo_52w,2)}\n"
            f"\nDETERMINISTIC PLAN:\n"
            f"CALL={plan['call']} | STRATEGY={strategy_with_proj}\n"
            f"Entry={plan['entry']} Stop={plan['stop']} T1={plan['t1']} T2={plan['t2']}\n"
        )

        provider = self.provider_var.get().strip().lower()
        ollama_model = self.ollama_model_var.get().strip() or OLLAMA_MODEL_DEFAULT
        deepseek_model = self.deepseek_model_var.get().strip() or DEEPSEEK_MODEL_DEFAULT
        try:
            deepseek_max_tokens = int(self.deepseek_tokens_var.get().strip() or str(DEEPSEEK_MAX_TOKENS_DEFAULT))
        except Exception:
            deepseek_max_tokens = DEEPSEEK_MAX_TOKENS_DEFAULT
        try:
            deepseek_temperature = float(self.deepseek_temp_var.get().strip() or str(DEEPSEEK_TEMPERATURE_DEFAULT))
        except Exception:
            deepseek_temperature = DEEPSEEK_TEMPERATURE_DEFAULT

        # Prompts (now include the projection explicitly)
        f_p = f"""
You are a buy-side equity analyst. Write a deep but concise fundamental note for {ticker}.
Use bullet points and clear section headers. Keep it to ~12-16 bullets total.

Include:
1) Business/segment snapshot (1-2 bullets)
2) Quality & profitability (ROE, margins)
3) Growth profile (revenue/earnings growth if available)
4) Valuation (TTM/forward P/E, EV/EBITDA, P/S, FCF yield)
5) Balance sheet & solvency (current/quick, net debt, net debt/EBITDA)
6) Shareholder returns (dividend yield/payout if available)
7) 3 bull-case catalysts and 3 bear-case risks
8) What to monitor next quarter

Metrics:
- Market Cap {fmt_big_money(f.get("market_cap"))}, EV {fmt_big_money(f.get("enterprise_value"))}, Beta {fmt_num(f.get("beta"),2)}
- P/E TTM {fmt_num(f.get("trailing_pe"),2)}, P/E Fwd {fmt_num(f.get("forward_pe"),2)}, PEG {fmt_num(f.get("peg"),2)}
- EV/EBITDA {fmt_num(f.get("ev_to_ebitda"),2)}, P/S {fmt_num(f.get("ps"),2)}, P/B {fmt_num(f.get("pb"),2)}, FCF Yield {fmt_pct(f.get("fcf_yield"),2)}
- Rev Growth {fmt_pct(f.get("rev_growth"),2)}, Earn Growth {fmt_pct(f.get("earn_growth"),2)}
- Gross {fmt_pct(f.get("gross_m"),2)}, Oper {fmt_pct(f.get("op_m"),2)}, Net {fmt_pct(f.get("net_m"),2)}
- ROE {fmt_pct(f.get("roe"),2)}, ROA {fmt_pct(f.get("roa"),2)}
- Current {fmt_num(f.get("current_ratio"),2)}, Quick {fmt_num(f.get("quick_ratio"),2)}, D/E {fmt_num(f.get("debt_to_equity"),2)}
- Net Debt {fmt_big_money(f.get("net_debt"))}, Net Debt/EBITDA {fmt_num(f.get("net_debt_to_ebitda"),2)}
- Div Yield {fmt_pct(f.get("div_yield"),2)}, Payout {fmt_pct(f.get("payout"),2)}
- Street Target Mean {fmt_money(f.get("target"),2)}
""".strip()

        t_p = f"""
You are a technical strategist. This analysis interval is {interval.upper()}.

CRITICAL RULES:
- Your assessment MUST match the LOCKED CALL: {call}.
- Do not contradict the call.
- Do not invent prices or levels.

Write 10-14 bullets:
1) Bias drivers ({vwap_label} + session return + EMA slopes; ORB only if available)
2) Trend strength (ADX) and volatility (ATR, BB width)
3) Flow confirmation (RVOL, CMF)
4) Key levels (S1/R1/S2/R2 + {vwap_label})
5) 5-candle projection interpretation (base path + uncertainty); keep it probabilistic
6) How to execute the locked plan (entry trigger, invalidation)

Data:
- Price {fmt_money(curr,4)}
- {vwap_label} {fmt_money(vwap_like,4)} | ORB_H/L {orb_str}
- S1/R1 {fmt_money(sup_50,4)} / {fmt_money(res_50,4)} | S2/R2 {fmt_money(sup_2,4)} / {fmt_money(res_2,4)}
- RSI {fmt_num(rsi,1)} ADX {fmt_num(adx,1)} ATR {fmt_money(atr,4)}
- RVOL {fmt_num(rvol,2)} CMF {fmt_num(cmf20,3)} BB_Width {fmt_num(bb_width,4)}
- EMA20_slope {fmt_num(ema20_slope,6)} EMA50_slope {fmt_num(ema50_slope,6)}

PROJECTION (next {PROJ_HORIZON} candles):
{proj_txt}

LOCKED PLAN:
- Strategy {strategy} | Entry {plan['entry']} Stop {plan['stop']} T1 {plan['t1']} T2 {plan['t2']} | Horizon {plan['time_horizon']}
""".strip()

        c_p = f"""
You are a portfolio manager writing the final conclusion for a note on {ticker}.
This interval is {interval.upper()}.

CRITICAL RULES (must follow):
- You MUST use the OVERALL CALL exactly as given. Do not change it.
- You MUST use the STRATEGY exactly as given. Do not change it.
- You MUST reproduce the TRADE PLAN numbers EXACTLY as given. Do not change them.
- You MUST include the "probable price in the next {PROJ_HORIZON} candles (base)" if provided.
- Do not introduce new levels or prices. Do not invent numbers.

Output format (use these headers exactly):
1) OVERALL CALL (Bullish / Neutral / Bearish)
2) WHY (3-6 bullets; must be consistent with the call)
3) STRATEGY (choose one): Accumulate / Buy Pullbacks / Breakout Entry / Wait for Confirmation / Avoid
4) PROJECTION (next {PROJ_HORIZON} candles): Probable price (base) + brief probabilistic note
5) TRADE PLAN (numbers): Entry Zone, Invalidation/Stop, Target 1, Target 2, Time Horizon
6) RISK MANAGEMENT (3 bullets)
7) CHECKLIST (5 items)

LOCKED OUTPUT VALUES (use verbatim):
- OVERALL CALL: {call}
- STRATEGY: {strategy}
- Entry Zone: {plan['entry']}
- Invalidation/Stop: {plan['stop']}
- Target 1: {plan['t1']}
- Target 2: {plan['t2']}
- Time Horizon: {plan['time_horizon']}
- Probable price in next {PROJ_HORIZON} candles (base): {fmt_money(proj_p5,4)}

Context data (do not invent missing values):
- Price: {fmt_money(curr,4)}
- Session Ret: {fmt_pct(ses_ret,2)}
- {vwap_label}: {fmt_money(vwap_like,4)} | ORB_H/L: {orb_str}
- Levels: S1 {fmt_money(sup_50,4)} R1 {fmt_money(res_50,4)} S2 {fmt_money(sup_2,4)} R2 {fmt_money(res_2,4)}
- RSI: {fmt_num(rsi,1)} ADX: {fmt_num(adx,1)} ATR: {fmt_money(atr,4)}
- Projection details:
{proj_txt}
""".strip()

        self._set_progress(0.58, "Running AI: Fundamental analysis...")
        f_res = run_llm(
            f_p, provider, ollama_model, HF_TOKEN, HF_BASE_URL,
            deepseek_model, deepseek_max_tokens, deepseek_temperature
        )

        self._set_progress(0.74, "Running AI: Technical analysis...")
        t_res = run_llm(
            t_p, provider, ollama_model, HF_TOKEN, HF_BASE_URL,
            deepseek_model, deepseek_max_tokens, deepseek_temperature
        )

        self._set_progress(0.88, "Running AI: Conclusion & strategy...")
        c_res = run_llm(
            c_p, provider, ollama_model, HF_TOKEN, HF_BASE_URL,
            deepseek_model, deepseek_max_tokens, deepseek_temperature
        )

        self._set_progress(0.94, "Rendering UI components...")

        self.after(
            0,
            lambda: self.update_ui(
                f["name"], curr,
                v_s, v_c,
                t_s, t_c,
                plan["entry"], plan["stop"], plan["t1"], plan["t2"],
                proj_txt,
                metric_txt,
                f.get("dashboard_text", ""),
                f_res, t_res, c_res,
                provider
            ),
        )

    def update_ui(
        self, name, cp,
        v_s, v_c,
        t_s, t_c,
        entry, stop, t1, t2,
        proj_txt,
        m_txt, fund_metrics_txt, f_ai, t_ai, c_ai, provider
    ):
        self.ticker_label.configure(text=str(name).upper())
        self.price_label.configure(text=f"${cp:.4f}")

        self.val_tag.configure(text=v_s, fg_color=v_c)
        self.tech_tag.configure(text=t_s, fg_color=t_c)

        res = last_valid_value(self.df, "RES_50")
        sup = last_valid_value(self.df, "SUP_50")
        hi52 = last_valid_value(self.df, "HI_52W")
        lo52 = last_valid_value(self.df, "LO_52W")

        self.res_lbl.configure(text=f"RESISTANCE: {fmt_money(res,4)}")
        self.sup_lbl.configure(text=f"SUPPORT: {fmt_money(sup,4)}")
        self.hi52_lbl.configure(text=f"52W HIGH: {fmt_money(hi52,2)}")
        self.lo52_lbl.configure(text=f"52W LOW:  {fmt_money(lo52,2)}")

        self.buy_lbl.configure(text=f"ENTRY: ${float(entry):.4f}")
        self.sl_lbl.configure(text=f"STOP:  ${float(stop):.4f}")
        self.tp_lbl.configure(text=f"T1:    ${float(t1):.4f}")

        self.proj_box.delete("1.0", "end")
        self.proj_box.insert("1.0", proj_txt)

        self.metrics_box.delete("1.0", "end")
        self.metrics_box.insert("1.0", m_txt + f"\nNOTE: Target 2 (T2) = {t2}\n")

        self.fund_metrics_box.delete("1.0", "end")
        self.fund_metrics_box.insert("1.0", fund_metrics_txt)

        self.fund_ai.delete("1.0", "end")
        self.fund_ai.insert("1.0", f_ai)

        self.tech_ai.delete("1.0", "end")
        self.tech_ai.insert("1.0", t_ai)

        self.conclusion_ai.delete("1.0", "end")
        self.conclusion_ai.insert("1.0", c_ai)

        self.btn_chart.configure(state="normal")
        self.progress_bar.set(1.0)
        self.progress_text.configure(text="Progress: 100%  |  Complete")
        self.status.configure(text=f"Analysis Finalized ({provider})", text_color="green")
        self.btn_run.configure(state="normal")

    def show_chart(self):
        if self.df is None or self.df.empty:
            return

        df = _normalize_index_to_naive_datetime(self.df)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            return

        interval = self.interval_used or "15m"

        ema20_col = first_col_like(df, "EMA_20")
        ema50_col = first_col_like(df, "EMA_50")
        sma200_col = first_col_like(df, "SMA_200")

        rsi_col = first_col_like(df, "RSI")
        macd_line_col = first_col_like(df, "MACD_")
        macd_sig_col = first_col_like(df, "MACDs_")
        macd_hist_col = first_col_like(df, "MACDh_")

        bbl_col = first_col_like(df, "BBL_")
        bbm_col = first_col_like(df, "BBM_")
        bbu_col = first_col_like(df, "BBU_")

        vwap_col = "VWAP" if "VWAP" in df.columns else None
        rvol = df["RVOL"] if "RVOL" in df.columns else None

        res_50 = last_valid_value(df, "RES_50")
        sup_50 = last_valid_value(df, "SUP_50")
        hi_52w = last_valid_value(df, "HI_52W")
        lo_52w = last_valid_value(df, "LO_52W")

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.58, 0.16, 0.13, 0.13],
            vertical_spacing=0.03,
            subplot_titles=("Price + Overlays + Projection", "Volume + RVOL", "RSI", "MACD"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1, col=1
        )

        if ema20_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema20_col], name="EMA 20", mode="lines"), row=1, col=1)
        if ema50_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema50_col], name="EMA 50", mode="lines"), row=1, col=1)
        if sma200_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[sma200_col], name="SMA 200", mode="lines"), row=1, col=1)

        if bbu_col and bbm_col and bbl_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], name="BB Upper", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[bbm_col], name="BB Mid", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], name="BB Lower", mode="lines"), row=1, col=1)

        if vwap_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[vwap_col], name="VWAP/VWMA", mode="lines"), row=1, col=1)

        # Projection overlay
        if self.proj and self.proj.get("base"):
            base = np.array(self.proj["base"], dtype=float)
            up = np.array(self.proj.get("upper_1s", []), dtype=float) if self.proj.get("upper_1s") else None
            lo = np.array(self.proj.get("lower_1s", []), dtype=float) if self.proj.get("lower_1s") else None

            last_idx = pd.to_datetime(df.index[-1])
            x_future = future_index(last_idx, interval=interval, horizon=len(base))

            fig.add_trace(go.Scatter(x=x_future, y=base, name="Proj Base (Next 5)", mode="lines+markers"), row=1, col=1)
            if up is not None and lo is not None and len(up) == len(base) and len(lo) == len(base):
                fig.add_trace(go.Scatter(x=x_future, y=up, name="Proj +1σ", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_future, y=lo, name="Proj -1σ", mode="lines"), row=1, col=1)

        if res_50 is not None and math.isfinite(res_50):
            fig.add_hline(y=res_50, line_dash="dash", annotation_text="R1", row=1, col=1)
        if sup_50 is not None and math.isfinite(sup_50):
            fig.add_hline(y=sup_50, line_dash="dash", annotation_text="S1", row=1, col=1)
        if hi_52w is not None and math.isfinite(hi_52w):
            fig.add_hline(y=hi_52w, line_dash="dot", annotation_text="52W High", row=1, col=1)
        if lo_52w is not None and math.isfinite(lo_52w):
            fig.add_hline(y=lo_52w, line_dash="dot", annotation_text="52W Low", row=1, col=1)

        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
        if rvol is not None:
            fig.add_trace(go.Scatter(x=df.index, y=rvol, name="RVOL (20)", mode="lines"), row=2, col=1)

        if rsi_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[rsi_col], name="RSI(14)", mode="lines"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", annotation_text="70", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", annotation_text="30", row=3, col=1)

        if macd_hist_col:
            fig.add_trace(go.Bar(x=df.index, y=df[macd_hist_col], name="MACD Hist"), row=4, col=1)
        if macd_line_col and macd_sig_col:
            fig.add_trace(go.Scatter(x=df.index, y=df[macd_line_col], name="MACD", mode="lines"), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[macd_sig_col], name="Signal", mode="lines"), row=4, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=1150,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend_orientation="h",
            legend_y=1.02,
            legend_x=0,
        )
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)

        fig.show()


if __name__ == "__main__":
    app = TradingApp()
    app.mainloop()
