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
    d = 0.0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d += abs(float(va) - float(vb)) / (abs(float(va)) + abs(float(vb)) + 1e-6)
        else:
            d += 0.0 if va == vb else 1.0
    return d / len(keys)


def _apply_directives(params: dict, directives: dict, symbol: str) -> dict:
    if not directives:
        return params

    for key, val in directives.items():
        params[key] = val
    return params


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
    for name, text in results.items():
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

    directives = (feedback or {}).get("mutation_directives") or {}

    candidates = []
    seen = set()

    if llm_client:
        llm_sets = _llm_mutations(base_params, feedback, n_children, llm_client)
    else:
        llm_sets = []

    for i in range(max(1, n_children)):
        attempts = 0
        while attempts < 6:
            params = dict(base_params)
            params = _apply_directives(params, directives, symbol)

            if llm_client and i < len(llm_sets):
                params.update(llm_sets[i])
            else:
                params["random_noise"] = rng.random()

            sig = _signature(params)

            if sig in seen:
                attempts += 1
                continue

            if diversity_pool:
                too_close = False
                for p in diversity_pool:
                    if _distance(params, p.get("parameters", {})) < 0.15:
                        too_close = True
                        break
                if too_close:
                    attempts += 1
                    continue

            seen.add(sig)
            sid = f"evo_{symbol.replace('/', '_').lower()}_{timeframe}_{rng.randint(1,999999)}"
            candidates.append(
                StrategyCandidate(
                    sid,
                    str((parent or {}).get("strategy_id") or "seed"),
                    int((parent or {}).get("version", 0) or 0) + 1,
                    params,
                    symbol,
                    timeframe,
                    [symbol, timeframe, "evo"],
                    "evolution",
                )
            )
            break

    return candidates


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(f"{family}_{symbol.replace('/', '_').lower()}_{timeframe}_{random.randint(1,999999)}", "seed", 1, {}, symbol, timeframe, [symbol, timeframe, family], "seed")
