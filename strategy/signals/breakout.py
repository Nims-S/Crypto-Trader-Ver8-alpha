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
    if df is None or len(df) < 180:
        return None
    df = compute_indicators(df) if "atr" not in df.columns else df
    cur = df.iloc[-1]
    if float(cur.get("close", 0)) < max(float(cur.get("bb_upper", 0)), float(cur.get("swing_high_20", 0))) * 0.998:
        return None
    entry = float(cur["close"]); atr = float(cur["atr"])
    if entry <= 0 or atr <= 0:
        return None
    stop = min(float(cur.get("swing_low_20", entry - 1.35 * atr)) * 0.995, entry - 1.35 * atr)
    if stop >= entry:
        return None
    risk = entry - stop
    tp1 = entry + (1.4 * risk); tp2 = entry + (3.1 * risk); tp3 = entry + (5.0 * risk)
    return Signal("LONG", entry, stop, tp1, symbol, "alt_breakout_v1", "breakout", confidence=0.72, stop_loss_pct=risk / entry, take_profit_pct=(tp1 - entry) / entry, secondary_take_profit_pct=(tp2 - entry) / entry, tp3_pct=(tp3 - entry) / entry, tp3_close_fraction=0.20, trail_atr_mult=1.7, tp1_close_fraction=0.20, tp2_close_fraction=0.50, be_trigger_rr=2.0, max_bars_override=24, size_multiplier=0.95)
