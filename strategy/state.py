from __future__ import annotations
from dataclasses import dataclass

@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_adx: float = 16.0
    min_atr_rank: float = 0.15
    min_bb_rank: float = 0.15
    rsi_long: float = 53.0
    rsi_short: float = 47.0
