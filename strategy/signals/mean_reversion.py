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
    bb_lower = float(cur.get("bb_lower", 0))
    bb_rank = float(cur.get("bb_width_rank", 0))
    rsi = float(cur.get("rsi", 50))
    atr = float(cur.get("atr", 0))

    if close <= 0 or atr <= 0:
        return None

    # Only in low-vol regimes
    if bb_rank > 0.30:
        return None

    # Deep oversold only
    if rsi > 32.0:
        return None

    # Must be near lower band
    if close > bb_lower * 1.01:
        return None

    entry = close
    stop = entry - (1.6 * atr)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (1.8 * risk)

    return Signal(
        "LONG",
        entry,
        stop,
        tp1,
        symbol,
        "mean_reversion_v3",
        "mean_reversion",
        confidence=0.65,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        size_multiplier=0.6,
        cooldown_bars=18,
        max_bars_override=36,
    )
