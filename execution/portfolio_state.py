from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class PortfolioState:
    total_capital: float
    cash: float
    allocations: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, dict] = field(default_factory=dict)
    trade_history: List[dict] = field(default_factory=list)
    live_metrics: Dict[str, dict] = field(default_factory=dict)

    def apply_allocations(self, allocs: list[dict]):
        # Allocations are advisory in the live loop; actual cash moves on open/close.
        self.allocations = {a["strategy_id"]: float(a.get("capital", 0.0)) for a in allocs}

    def open_position(self, position: dict):
        sid = position.get("strategy_id")
        if not sid:
            return
        capital = float(position.get("capital") or 0.0)
        if capital > self.cash:
            return
        self.cash -= capital
        self.positions[sid] = position

    def close_position(self, sid: str, exit_price: float, reason: str) -> dict | None:
        pos = self.positions.pop(sid, None)
        if not pos:
            return None
        entry = float(pos.get("entry_price") or 0.0)
        qty = float(pos.get("qty") or 0.0)
        side = pos.get("side", "LONG")
        pnl = 0.0
        if qty > 0 and entry > 0:
            if side == "LONG":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

        capital = float(pos.get("capital") or 0.0)
        self.cash += capital + pnl

        trade = {
            "strategy_id": sid,
            "symbol": pos.get("symbol"),
            "timeframe": pos.get("timeframe"),
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": pnl,
            "reason": reason,
        }
        self.trade_history.append(trade)
        return trade

    def get_position(self, sid: str) -> dict | None:
        return self.positions.get(sid)

    def update_live_metrics(self, strategy_id: str, metrics: dict[str, Any]):
        self.live_metrics[strategy_id] = metrics

    def get_live_metrics(self, strategy_id: str) -> dict[str, Any]:
        return self.live_metrics.get(strategy_id, {})
