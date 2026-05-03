from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from execution.backtest.core import run_backtest
from registry.store import record_evolution_run, upsert_strategy, rank_strategies
from research.agent_scoring import AgentScore, score_candidate
from research.candidate_generator import StrategyCandidate, mutate_parent, seed_strategy
from research.feedback import build_feedback_summary
from research.validation import build_walk_forward_folds, summarize_walk_forward_reports


@dataclass(frozen=True)
class AgentConfig:
    symbol: str
    timeframe: str
    start: str
    end: str
    goal_return: float = 0.25
    max_dd: float = 15.0
    iterations: int = 100
    candidates: int = 5
    folds: int = 3
    workers: int = 4
    continuous: bool = False
    sleep_seconds: float = 1.0


@dataclass(frozen=True)
class CandidateResult:
    candidate: StrategyCandidate
    backtest: dict[str, Any]
    walk_forward: dict[str, Any]
    monte_carlo: dict[str, Any]
    score: AgentScore
    parent_id: str
    iteration: int


def _normalize_parent(row: Any, symbol: str, timeframe: str) -> dict[str, Any]:
    if row is None:
        seed = seed_strategy(symbol, timeframe)
        return asdict(seed)
    if isinstance(row, dict):
        return row
    try:
        return asdict(row)
    except Exception:
        return {
            "strategy_id": getattr(row, "strategy_id", "seed"),
            "base_strategy": getattr(row, "base_strategy", "seed"),
            "version": int(getattr(row, "version", 1) or 1),
            "parameters": getattr(row, "parameters", {}) or {},
            "symbol": symbol,
            "timeframe": timeframe,
            "tags": [symbol, timeframe, "seed"],
            "source": "seed",
        }


def _choose_parent(cfg: AgentConfig) -> dict[str, Any]:
    ranked = rank_strategies(symbol=cfg.symbol, timeframe=cfg.timeframe, limit=5)
    if not ranked:
        return _normalize_parent(None, cfg.symbol, cfg.timeframe)

    # diversify: pick random among top strategies with different signatures
    unique = {}
    for r in ranked:
        sig = str(r.get("logic_hash") or r.get("strategy_id"))
        if sig not in unique:
            unique[sig] = r

    pool = list(unique.values())
    return random.choice(pool)


def run_agent(cfg: AgentConfig) -> dict[str, Any]:
    parent = _choose_parent(cfg)
    best_overall: CandidateResult | None = None

    iteration = 0
    while True:
        iteration += 1
        if not cfg.continuous and iteration > cfg.iterations:
            break

        feedback = build_feedback_summary(symbol=cfg.symbol, timeframe=cfg.timeframe)

        diversity_pool = rank_strategies(symbol=cfg.symbol, timeframe=cfg.timeframe, limit=10)

        children = mutate_parent(
            parent,
            cfg.symbol,
            cfg.timeframe,
            n_children=max(1, int(cfg.candidates)),
            feedback=feedback,
            diversity_pool=diversity_pool,
        )

        results: list[CandidateResult] = []
        parent_id = str(parent.get("strategy_id") if isinstance(parent, dict) else getattr(parent, "strategy_id", "seed"))

        if cfg.workers > 1 and len(children) > 1:
            with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
                futures = [pool.submit(_evaluate_candidate, c, cfg, iteration, parent_id) for c in children]
                for fut in as_completed(futures):
                    results.append(fut.result())
        else:
            for c in children:
                results.append(_evaluate_candidate(c, cfg, iteration, parent_id))

        if not results:
            raise RuntimeError("agent produced no candidates")

        best = max(results, key=lambda r: r.score.score)
        _persist_candidate(best, cfg)

        if best_overall is None or best.score.score > best_overall.score.score:
            best_overall = best

        print(
            {
                "iteration": iteration,
                "best_strategy": best.candidate.strategy_id,
                "score": best.score.score,
                "passed": best.score.passed,
                "reasons": best.score.reasons,
                "return_pct": best.backtest.get("return_pct"),
                "max_dd": best.backtest.get("max_drawdown_pct"),
                "pf": best.backtest.get("profit_factor"),
                "wr": best.backtest.get("win_rate"),
                "wf_passed": best.walk_forward.get("passed"),
            }
        )

        if best.score.passed:
            return {
                "status": "target_achieved",
                "best_strategy": best.candidate.strategy_id,
                "score": best.score.as_dict(),
                "backtest": best.backtest,
                "walk_forward": best.walk_forward,
                "monte_carlo": best.monte_carlo,
            }

        parent = _normalize_parent(
            {
                "strategy_id": best.candidate.strategy_id,
                "base_strategy": best.candidate.base_strategy,
                "version": best.candidate.version,
                "parameters": best.candidate.parameters,
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "tags": best.candidate.tags,
                "source": best.candidate.source,
            },
            cfg.symbol,
            cfg.timeframe,
        )

        if cfg.continuous:
            time.sleep(max(0.0, float(cfg.sleep_seconds or 0.0)))

    if best_overall is None:
        raise RuntimeError("agent did not evaluate any candidates")

    return {
        "status": "stopped",
        "best_strategy": best_overall.candidate.strategy_id,
        "score": best_overall.score.as_dict(),
        "backtest": best_overall.backtest,
        "walk_forward": best_overall.walk_forward,
        "monte_carlo": best_overall.monte_carlo,
    }
