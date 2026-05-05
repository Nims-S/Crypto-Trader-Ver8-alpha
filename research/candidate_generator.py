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

    # If we have not cleared deployability, the main thing to optimize should be
    # the current bottleneck, not generic stability.
    if primary == "no_trades" or test_trades < 12:
        return "density"
    if primary == "low_profit_factor":
        return "profit_factor"
    if primary in {"unstable", "high_drawdown"}:
        return "stability"
    if not quality_floor:
        return "density"
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
        score += (1.0 / (1.0 + filter_count)) * 1.2
        if mode in {"breakout", "mean_reversion"}:
            score += 0.75
    elif objective == "stability":
        score += (filter_count / 6.0) * 1.5
        score += _safe_float(params.get("min_adx", 10), 10) / 40.0
    elif objective == "drawdown":
        score += 1.0 / (1.0 + _safe_float(params.get("size_multiplier", 1.0), 1.0))
        score += _safe_float(params.get("stop_atr_mult", 1.5), 1.5) / 3.0
    elif objective == "profit_factor":
        score += _safe_float(params.get("tp1_rr", 2.0), 2.0) / 5.0
        score += _safe_float(params.get("tp2_rr", 3.0), 3.0) / 8.0
        score += (1.0 / (1.0 + cooldown))
    else:
        score += 0.5 * (1.0 / (1.0 + cooldown))
        score += 0.5 * (1.0 / (1.0 + max_bars))
    return float(score)


def _apply_directives(params: dict, directives: dict, symbol: str) -> dict:
    if not directives:
        return params

    for key in (
        "use_htf_filter",
        "use_volume_filter",
        "use_structure_filter",
        "use_reclaim_filter",
        "use_trend_filter",
        "use_breakout_filter",
        "entry_mode",
    ):
        if key in directives:
            params[key] = directives[key]

    for key in (
        "min_adx",
        "min_bb_rank",
        "min_atr_rank",
        "htf_adx_min",
        "htf_bb_rank_min",
        "rsi_min",
        "rsi_max",
        "volume_multiplier",
        "pullback_lookback",
        "pullback_bars",
        "stop_atr_mult",
        "tp1_rr",
        "tp2_rr",
        "tp1_close_fraction",
        "tp2_close_fraction",
        "tp3_close_fraction",
        "trail_atr_mult",
        "trail_ema20",
        "cooldown_bars",
        "max_bars_override",
        "confidence",
        "size_multiplier",
        "be_trigger_rr",
    ):
        if key in directives:
            params[key] = directives[key]

    if symbol.startswith("BTC") and directives.get("prefer_trend_pullback"):
        params["entry_mode"] = "trend_pullback"
        params["use_htf_filter"] = True
        params["use_trend_filter"] = True

    return params


# existing mutation functions unchanged


def _llm_mutations(base_params: dict, feedback: dict, n: int, llm_client) -> List[dict]:
    context = {
        "symbol": feedback.get("symbol"),
        "timeframe": feedback.get("timeframe"),
        "mutation_directives": feedback.get("mutation_directives"),
    }
    prompts = build_child_batch_prompts(context, n=n)
    jobs = [PromptJob(name=p["goal"], prompt=p["prompt"]) for p in prompts]
    results = batch_prompts_sync(jobs, client=llm_client, max_concurrency=min(4, n))

    out = []
    for _, text in results.items():
        try:
            data = json.loads(text) if isinstance(text, str) else text
            updates = data.get("parameter_updates") if isinstance(data, dict) else {}
            params = dict(base_params)
            if isinstance(updates, dict):
                params.update(updates)
            out.append(params)
        except Exception:
            out.append(dict(base_params))
    return out


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None, llm_client=None, diversity_pool=None):
    rng = random.Random(seed)
    base_params = dict((parent or {}).get("parameters") or {})

    if feedback is None:
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)

    objective = _objective_from_feedback(feedback)

    if llm_client is None:
        llm_client = get_default_llm_client()

    directives = (feedback or {}).get("mutation_directives") or {}
    base_entry_mode = str(base_params.get("entry_mode") or directives.get("entry_mode") or "trend_pullback")

    oversample = max(n_children * 3, n_children + 2)
    raw_params: List[dict] = []

    if llm_client:
        llm_sets = _llm_mutations(base_params, feedback, oversample, llm_client)
        raw_params.extend(llm_sets)

    for _ in range(oversample):
        params = dict(base_params)
        params = _apply_directives(params, directives, symbol)

        mode = base_entry_mode
        if rng.random() < 0.60:
            mode = rng.choice(["trend_pullback", "breakout", "mean_reversion"])

        if mode == "trend_pullback":
            params = _mutate_trend_params(rng, params)
        elif mode == "breakout":
            params = _mutate_breakout_params(rng, params)
        else:
            params = _mutate_mean_reversion_params(rng, params)

        raw_params.append(params)

    scored = []
    seen = set()

    for params in raw_params:
        sig = _signature(params)
        if sig in seen:
            continue

        if diversity_pool:
            too_close = False
            for p in diversity_pool:
                if _distance(params, p.get("parameters", {})) < 0.10:
                    too_close = True
                    break
            if too_close:
                continue

        priority = _candidate_priority(params, objective)
        priority += rng.uniform(0.0, 0.05)

        scored.append((priority, params))
        seen.add(sig)

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[: max(1, n_children)]

    candidates = []
    for _, params in selected:
        sid = f"evo_{symbol.replace('/', '_').lower()}_{timeframe}_{rng.randint(1,999999)}"
        candidates.append(
            StrategyCandidate(
                sid,
                str((parent or {}).get("strategy_id") or "seed"),
                int((parent or {}).get("version", 0) or 0) + 1,
                params,
                symbol,
                timeframe,
                [symbol, timeframe, "evo", params.get("entry_mode", "mixed")],
                "evolution",
                notes=f"objective={objective}",
            )
        )

    return candidates


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(
        f"{family}_{symbol.replace('/', '_').lower()}_{timeframe}_{random.randint(1,999999)}",
        "seed",
        1,
        {},
        symbol,
        timeframe,
        [symbol, timeframe, family],
        "seed",
    )
