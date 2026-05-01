from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreDecision:
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
        }


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def score_metrics(m: dict[str, Any]) -> ScoreDecision:
    trades = int(m.get("trades", 0) or 0)
    pf = _safe(m.get("profit_factor", 0))
    wr = _safe(m.get("win_rate", 0))
    dd = _safe(m.get("max_drawdown_pct", 0))

    reasons: list[str] = []
    if trades < 20:
        reasons.append("trades<20")
    if pf < 1.1:
        reasons.append("pf<1.1")
    if wr < 0.45:
        reasons.append("wr<0.45")

    score = (
        0.4 * min(pf / 2.0, 1.0)
        + 0.3 * wr
        + 0.2 * max(0.0, 1.0 + dd / 20.0)
        + 0.1 * min(trades / 40.0, 1.0)
    )
    return ScoreDecision(score=score, passed=len(reasons) == 0 and score > 0.55, reasons=tuple(reasons))


def promotion_status(decision: ScoreDecision) -> str:
    return "validated" if decision.passed else "rejected"
