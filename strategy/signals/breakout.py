from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    if df is None or len(df) < 220:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    cur = df.iloc[-1]

    close = float(cur.get("close", 0))
    high20 = float(cur.get("swing_high_20", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    atr = float(cur.get("atr", 0))
    vol = float(cur.get("volume", 0))
    vol_sma = float(cur.get("volume_sma20", 0))
    bb_streak = int(cur.get("bbwp_low_streak", 0))

    if close <= 0 or atr <= 0:
        return None

    # Require squeeze then expansion
    if bb_rank > 0.40:
        return None

    if bb_streak < 3:
        return None

    # Breakout above recent high
    if close <= high20:
        return None

    # Volume expansion
    if vol_sma > 0 and vol < vol_sma * 1.2:
        return None

    entry = close
    stop = entry - (1.8 * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (2.5 * risk)

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        "breakout_v3",
        "breakout",
        confidence=0.75,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        size_multiplier=0.7,
        cooldown_bars=24,
        max_bars_override=72,
    )
