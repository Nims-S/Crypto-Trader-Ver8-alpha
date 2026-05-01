from __future__ import annotations

from typing import Any

from strategy.state import StrategyState
from strategy.regime_classifier import classify_market
from strategy.signals.trend import generate as trend_generate
from strategy.signals.mean_reversion import generate as mr_generate
from strategy.signals.breakout import generate as breakout_generate


def generate_signal(
    df,
    state: StrategyState,
    symbol: str,
    df_htf=None,
    strategy_override: dict[str, Any] | None = None,
):
    if strategy_override:
        mode = strategy_override.get("entry_mode")
        if mode == "trend":
            return trend_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
        if mode == "breakout":
            return breakout_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
        if mode == "mean_reversion":
            return mr_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)

    regime = classify_market(df, df_htf)

    if regime == "trend":
        return trend_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
    if regime == "breakout":
        return breakout_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)

    return mr_generate(df, symbol, state, df_htf=df_htf, strategy_override=strategy_override)
