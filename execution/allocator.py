from __future__ import annotations

import math
from typing import Any, List, Dict


def _softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(1e-6, float(temperature or 1.0))
    scaled = [x / t for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _score_row(row: dict[str, Any]) -> float:
    wf = (row.get("metrics") or {}).get("walk_forward") or {}
    base = float(wf.get("score", 0.0) or 0.0)
    robustness = float(row.get("robustness_score", 0.0) or 0.0)
    return base * (1.0 + 0.3 * robustness)


def allocate_capital(
    strategies: List[dict[str, Any]],
    total_capital: float,
    *,
    temperature: float = 1.0,
    min_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    if not strategies:
        return []

    scores = [_score_row(r) for r in strategies]
    weights = _softmax(scores, temperature=temperature)

    if min_weight > 0:
        weights = [max(min_weight, w) for w in weights]
        s = sum(weights) or 1.0
        weights = [w / s for w in weights]

    allocations = []
    for row, w in zip(strategies, weights):
        allocations.append(
            {
                "strategy_id": row.get("strategy_id"),
                "weight": float(w),
                "capital": float(total_capital * w),
                "score": _score_row(row),
            }
        )
    return allocations
