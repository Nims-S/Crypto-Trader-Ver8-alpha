from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PortfolioState:
    total_capital: float
    cash: float
    allocations: Dict[str, float] = field(default_factory=dict)
    live_metrics: Dict[str, dict] = field(default_factory=dict)

    def apply_allocations(self, allocs: list[dict]):
        self.allocations = {a["strategy_id"]: a["capital"] for a in allocs}
        self.cash = self.total_capital - sum(self.allocations.values())

    def update_live_metrics(self, strategy_id: str, metrics: dict[str, Any]):
        self.live_metrics[strategy_id] = metrics

    def get_live_metrics(self, strategy_id: str) -> dict[str, Any]:
        return self.live_metrics.get(strategy_id, {})
