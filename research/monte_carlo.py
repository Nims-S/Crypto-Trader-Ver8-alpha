from __future__ import annotations

import random
import statistics
from typing import Any, List

from registry.store import export_trade_history


def _returns(trades: List[dict]) -> List[float]:
    return [float(t.get("pnl", 0.0)) for t in trades if t.get("pnl") is not None]


def run_monte_carlo(
    strategy_id: str | None = None,
    *,
    simulations: int = 1000,
    horizon: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)

    trades = export_trade_history(strategy_id=strategy_id)
    rets = _returns(trades)

    if not rets:
        return {"error": "no_trade_history"}

    horizon = horizon or len(rets)

    equity_paths = []

    for _ in range(simulations):
        equity = 0.0
        path = []
        for _ in range(horizon):
            r = random.choice(rets)
            equity += r
            path.append(equity)
        equity_paths.append(path)

    final_equities = [p[-1] for p in equity_paths if p]
    max_dds = []

    for path in equity_paths:
        peak = float("-inf")
        dd = 0.0
        for x in path:
            peak = max(peak, x)
            dd = min(dd, x - peak)
        max_dds.append(dd)

    return {
        "simulations": simulations,
        "mean_final": statistics.mean(final_equities),
        "median_final": statistics.median(final_equities),
        "p5": sorted(final_equities)[int(0.05 * len(final_equities))],
        "p95": sorted(final_equities)[int(0.95 * len(final_equities))],
        "avg_drawdown": statistics.mean(max_dds),
        "worst_drawdown": min(max_dds),
    }
