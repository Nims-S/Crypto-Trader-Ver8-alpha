from __future__ import annotations

import pandas as pd

from strategy.state import StrategyState, Signal
from strategy.indicators import compute_indicators


def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict | None = None):
    if df is None or len(df) < 180:
        return None

    df = compute_indicators(df) if "atr" not in df.columns else df
    cur = df.iloc[-1]

    if float(cur.get("rsi", 50.0)) > 45.0:
        return None

    entry = float(cur["close"])
    atr = float(cur["atr"])

    if entry <= 0 or atr <= 0:
        return None

    stop = entry - (1.2 * atr)

    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (1.2 * risk)

    return Signal("LONG", entry, stop, tp1, symbol, "mean_reversion_v2", "mean_reversion", confidence=0.6, stop_loss_pct=risk / entry, take_profit_pct=(tp1 - entry) / entry)
