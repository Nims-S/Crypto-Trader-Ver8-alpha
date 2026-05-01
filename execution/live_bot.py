from __future__ import annotations

import time
from typing import Any

from config.defaults import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
from execution.router import route_strategies
from execution.allocator import allocate_capital
from execution.drift_monitor import compare_performance
from execution.portfolio_state import PortfolioState
from registry.store import rank_strategies


def run_live_cycle(total_capital: float = 1000.0) -> dict[str, Any]:
    portfolio = PortfolioState(total_capital=total_capital, cash=total_capital)

    routed = route_strategies(DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES)
    strategy_rows = [r["strategy"] for r in routed]

    allocations = allocate_capital(strategy_rows, total_capital)
    portfolio.apply_allocations(allocations)

    reports = []
    for alloc in allocations:
        sid = alloc["strategy_id"]
        row = next((r for r in strategy_rows if r.get("strategy_id") == sid), None)
        if not row:
            continue

        expected = (row.get("metrics") or {}).get("walk_forward") or {}
        live = portfolio.get_live_metrics(sid)

        drift = compare_performance(expected, live)

        reports.append(
            {
                "strategy_id": sid,
                "allocation": alloc,
                "drift": drift,
            }
        )

    return {
        "allocations": allocations,
        "reports": reports,
        "cash": portfolio.cash,
    }


def run_loop(interval_seconds: int = 60):
    while True:
        result = run_live_cycle()
        print(result)
        time.sleep(max(1, int(interval_seconds)))
