from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def _recent_cross_above(series_a: pd.Series, series_b: pd.Series, lookback: int = 3) -> bool:
    if len(series_a) < lookback + 2 or len(series_b) < lookback + 2:
        return False
    a = series_a.iloc[-(lookback + 1):]
    b = series_b.iloc[-(lookback + 1):]
    prev = (a.shift(1) <= b.shift(1))
    now = a > b
    return bool((prev & now).any())


def _htf_confirm(df_htf: pd.DataFrame | None) -> bool:
    if df_htf is None or len(df_htf) < 220:
        return True
    h = df_htf.iloc[-1]
    close = float(h.get("close", 0))
    ema20 = float(h.get("ema20", 0))
    ema50 = float(h.get("ema50", 0))
    ema200 = float(h.get("ema200", 0))
    adx = float(h.get("adx", 0))
    bb_rank = float(h.get("bb_width_rank", 0))
    return bool(close > ema200 and ema20 > ema50 > ema200 and adx >= 18.0 and bb_rank >= 0.30)


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    if df is None or len(df) < 260:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    if df_htf is not None and len(df_htf) > 0 and "atr" not in df_htf.columns:
        df_htf = compute_indicators(df_htf)

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(cur.get("close", 0))
    ema20 = float(cur.get("ema20", 0))
    ema50 = float(cur.get("ema50", 0))
    ema200 = float(cur.get("ema200", 0))
    adx = float(cur.get("adx", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    atr_rank = float(cur.get("atr_pct_rank", 0))
    rsi = float(cur.get("rsi", 50))
    vol = float(cur.get("volume", 0))
    vol_sma = float(cur.get("volume_sma20", 0))

    if close <= 0 or ema20 <= 0 or ema50 <= 0 or ema200 <= 0:
        return None

    if not (close > ema200 and ema20 > ema50 > ema200):
        return None

    if not _htf_confirm(df_htf):
        return None

    if adx < max(state.min_adx + 5.0, 23.0):
        return None

    if bb_rank < 0.38 or atr_rank < 0.32:
        return None

    if rsi < 52.0 or rsi > 70.0:
        return None

    had_pullback = bool(
        prev.get("close", 0) <= prev.get("ema20", 0)
        or prev.get("low", 0) <= prev.get("ema20", 0)
        or _recent_cross_above(df["close"], df["ema20"], lookback=3)
    )
    if not had_pullback:
        return None

    if vol_sma > 0 and vol < vol_sma * 1.08:
        return None

    entry = close
    atr = float(cur.get("atr", 0))
    if entry <= 0 or atr <= 0:
        return None

    stop = entry - (2.0 * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (2.3 * risk)
    tp2 = entry + (4.0 * risk)

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        "trend_following_v4",
        "trend",
        confidence=0.84,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp1_close_fraction=0.50,
        tp2_close_fraction=0.50,
        size_multiplier=0.80,
        cooldown_bars=30,
        max_bars_override=84,
    )
