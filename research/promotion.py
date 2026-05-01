from __future__ import annotations

import json
from datetime import datetime
from typing import List, Dict, Any

from registry.store import list_strategies


def _safe(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _latest_passed_experiment(row: dict) -> dict | None:
    exps = row.get("experiments") or []
    passed = [e for e in exps if e.get("passed")]
    if not passed:
        return None
    return sorted(passed, key=lambda x: x.get("timestamp", ""), reverse=True)[0]


def _score(row: dict) -> float:
    m = (row.get("metrics") or {}).get("walk_forward") or {}
    return _safe(m.get("score", 0.0))


def select_candidates(limit: int = 10) -> List[Dict[str, Any]]:
    rows = list_strategies(active_only=False)

    eligible = []
    for r in rows:
        exp = _latest_passed_experiment(r)
        if not exp:
            continue

        score = _score(r)
        eligible.append((score, r))

    eligible.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in eligible[:limit]]


def write_promotion_report(path="artifacts/promotion_report.json"):
    rows = select_candidates()
    out = {
        "generated_at": datetime.utcnow().isoformat(),
        "count": len(rows),
        "strategies": rows,
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Promotion report written: {path}")