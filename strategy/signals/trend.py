from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from strategy.state import StrategyState
from strategy.indicators import compute_indicators

@dataclass
class Signal:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    strategy: str
    regime: str
    confidence: float = 0.5
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    secondary_take_profit_pct: float = 0.0
    tp3_pct: float = 0.0
    tp3_close_fraction: float = 0.0
    trail_pct: float = 0.0
    trail_atr_mult: float = 0.0
    trail_ema20: bool = False
    tp1_close_fraction: float = 0.5
    tp2_close_fraction: float = 0.5
    be_trigger_rr: float = 0.0
    max_bars_override: int = 0
    cooldown_bars: int = 0
    size_multiplier: float = 1.0

def generate(df: pd.DataFrame, symbol: str, state: StrategyState, df_htf: pd.DataFrame | None = None, strategy_override: dict[str, Any] | None = None):
    if df is None or len(df) < 220:
        return None
    df = compute_indicators(df) if "atr" not in df.columns else df
    cur = df.iloc[-1]
    if float(cur.get("close", 0)) <= float(cur.get("ema200", 0)) * 0.975:
        return None
    entry = float(cur["close"]); atr = float(cur["atr"])
    if entry <= 0 or atr <= 0:
        return None
    stop = entry - (1.5 * atr)
    if stop >= entry:
        return None
    risk = entry - stop
    tp1 = entry + (2.0 * risk)
    return Signal("LONG", entry, stop, tp1, symbol, "trend_following_v1", "trend", confidence=0.7, stop_loss_pct=risk / entry, take_profit_pct=(tp1 - entry) / entry, secondary_take_profit_pct=4.0 * risk / entry, tp1_close_fraction=0.5, tp2_close_fraction=0.5, max_bars_override=60, trail_ema20=True, size_multiplier=1.0)
