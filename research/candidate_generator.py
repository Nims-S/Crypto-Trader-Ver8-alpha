from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from research.feedback import build_feedback_summary
from research.llm_batch import PromptJob, batch_prompts_sync
from research.prompt_templates import build_child_batch_prompts
from research.llm_client import get_default_llm_client


@dataclass
class StrategyCandidate:
    strategy_id: str
    base_strategy: str
    version: int
    parameters: Dict[str, Any]
    symbol: str
    timeframe: str
    tags: list
    source: str
    notes: str = ""


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _signature(params: dict) -> str:
    try:
        s = json.dumps(params, sort_keys=True)
    except Exception:
        s = str(params)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    total = 0.0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            total += abs(float(va) - float(vb)) / (abs(float(va)) + abs(float(vb)) + 1e-6)
        else:
            total += 0.0 if va == vb else 1.0
    return total / len(keys)


def _objective_from_feedback(feedback: dict) -> str:
    profile = (feedback or {}).get("failure_profile") or {}
    primary = str(profile.get("primary") or "")
    trade_mean = ((feedback or {}).get("trade_activity") or {}).get("mean", {}) or {}
    test_trades = _safe_float(trade_mean.get("test", 0.0), 0.0)
    quality_floor = bool((feedback or {}).get("quality_floor_passed", False))

    if not quality_floor:
        return "stability"

    if primary == "no_trades" or test_trades < 4:
        return "density"
    if primary in {"unstable", "overfit"}:
        return "stability"
    if primary in {"high_drawdown"}:
        return "drawdown"
    return "balanced"


def _candidate_priority(params: dict, objective: str) -> float:
    cooldown = _safe_float(params.get("cooldown_bars", 24), 24)
    max_bars = _safe_float(params.get("max_bars_override", 60), 60)
    filter_count = sum(
        1
        for k in (
            "use_htf_filter",
            "use_volume_filter",
            "use_structure_filter",
            "use_reclaim_filter",
            "use_trend_filter",
            "use_breakout_filter",
        )
        if params.get(k)
    )
    mode = str(params.get("entry_mode") or "")

    score = 0.0

    if objective == "density":
        score += (1.0 / (1.0 + cooldown)) * 2.0
        score += (1.0 / (1.0 + max_bars)) * 1.5
        score += (1.0 / (1.0 + filter_count)) * 1.0
        if mode in {"breakout", "mean_reversion"}:
            score += 0.5
    elif objective == "stability":
        score += (filter_count / 6.0) * 1.5
        score += _safe_float(params.get("min_adx", 10), 10) / 40.0
    elif objective == "drawdown":
        score += 1.0 / (1.0 + _safe_float(params.get("size_multiplier", 1.0), 1.0))
        score += _safe_float(params.get("stop_atr_mult", 1.5), 1.5) / 3.0
    else:
        score += 0.5 * (1.0 / (1.0 + cooldown))
        score += 0.5 * (1.0 / (1.0 + max_bars))

    return float(score)

# rest unchanged

