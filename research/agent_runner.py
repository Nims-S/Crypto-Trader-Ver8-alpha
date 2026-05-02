from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from execution.backtest.core import run_backtest
from research.candidate_generator import mutate_parent, seed_strategy
from research.monte_carlo import run_monte_carlo_from_summary
from research.validation import build_walk_forward_folds, summarize_walk_forward_reports
from registry.store import rank_strategies, record_evolution_run, upsert_strategy


@dataclass(frozen=True)
class AgentConfig:
    symbol: str
    timeframe: str
    start: str
    end: str
    goal_return: float = 30.0
    max_dd: float = 15.0
    iterations: int = 100
    candidates: int = 5
    folds: int = 3
    workers: int = 4
    continuous: bool = False
    sleep_seconds: float = 1.0


def _split_window(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    span = e - s
    train_end = s + span * 0.6
    val_end = train_end + span * 0.2
    return (
        (s.isoformat(), train_end.isoformat()),
        (train_end.isoformat(), val_end.isoformat()),
        (val_end.isoformat(), e.isoformat()),
    )


def _as_parent(row: Any):
    if row is None:
        seed = seed_strategy("BTC/USDT", "1h")
        return asdict(seed)
    if isinstance(row, dict):
        return row
    try:
        return asdict(row)
    except Exception:
        return {"strategy_id": getattr(row, "strategy_id", "seed"), "parameters": getattr(row, "parameters", {})}


def _eval_walk_forward(cfg: AgentConfig, params: dict[str, Any]) -> dict[str, Any]:
    folds = build_walk_forward_folds(cfg.start, cfg.end, folds=cfg.folds)
    reports: list[dict[str, Any]] = []

    for fold in folds:
        (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = _split_window(fold.start, fold.end)
        reports.append(
            {
                "label": fold.label,
                "train": run_backtest(cfg.symbol, cfg.timeframe, tr_s, tr_e, strategy_override={"parameters": params}),
                "val": run_backtest(cfg.symbol, cfg.timeframe, va_s, va_e, strategy_override={"parameters": params}),
                "test": run_backtest(cfg.symbol, cfg.timeframe, te_s, te_e, strategy_override={"parameters": params}),
            }
        )

    return summarize_walk_forward_reports(reports, timeframe=cfg.timeframe)


def _passes(bt: dict[str, Any], wf: dict[str, Any], mc: dict[str, Any], cfg: AgentConfig) -> bool:
    if "error" in bt or "error" in mc:
        return False
    return (
        float(bt.get("return_pct", 0.0) or 0.0) >= cfg.goal_return
        and abs(float(bt.get("max_drawdown_pct", 0.0) or 0.0)) <= cfg.max_dd
        and bool(wf.get("passed", False))
        and abs(float(mc.get("worst_drawdown", 0.0) or 0.0)) <= cfg.max_dd
    )


def _objective(bt: dict[str, Any], wf: dict[str, Any], mc: dict[str, Any]) -> float:
    if "error" in bt or "error" in mc:
        return -1e9
    return (
        0.5 * float(wf.get("score", 0.0) or 0.0)
        + 0.3 * float(bt.get("profit_factor", 0.0) or 0.0)
        - 0.2 * abs(float(mc.get("worst_drawdown", 0.0) or 0.0))
    )


def _evaluate_candidate(cfg: AgentConfig, candidate, parent_id: str | None, iteration: int) -> dict[str, Any]:
    params = dict(candidate.parameters or {})
    bt = run_backtest(cfg.symbol, cfg.timeframe, cfg.start, cfg.end, strategy_override={"parameters": params})
    if "error" in bt:
        return {
            "candidate": candidate,
            "score": -1e9,
            "passed": False,
            "backtest": bt,
            "walk_forward": {"passed": False, "score": 0.0, "reasons": [bt["error"]]},
            "monte_carlo": {"error": bt["error"]},
        }

    wf = _eval_walk_forward(cfg, params)
    mc = run_monte_carlo_from_summary(bt)
    score = _objective(bt, wf, mc)
    passed = _passes(bt, wf, mc, cfg)

    record_evolution_run(
        cycle_id=f"iter_{iteration}",
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        parent_strategy_id=parent_id,
        child_strategy_id=candidate.strategy_id,
        status="validated" if passed else "rejected",
        score=score,
        passed=passed,
        parameters=params,
        metrics={"backtest": bt, "walk_forward": wf, "monte_carlo": mc, "objective": score},
        notes="agent iteration",
    )

    upsert_strategy(
        candidate.strategy_id,
        base_strategy=candidate.base_strategy,
        version=int(candidate.version or 1),
        status="validated" if passed else "rejected",
        parameters=params,
        metrics={"backtest": bt, "walk_forward": wf, "monte_carlo": mc, "objective": score},
        tags=list(candidate.tags or []),
        source=candidate.source,
        notes=candidate.notes,
        active=passed,
        robustness_score=float(wf.get("score", 0.0) or 0.0),
        parent_strategy_id=parent_id,
    )

    return {
        "candidate": candidate,
        "score": score,
        "passed": passed,
        "backtest": bt,
        "walk_forward": wf,
        "monte_carlo": mc,
    }


def run_agent(cfg: AgentConfig):
    ranked = rank_strategies(symbol=cfg.symbol, timeframe=cfg.timeframe, limit=1)
    parent = _as_parent(ranked[0] if ranked else seed_strategy(cfg.symbol, cfg.timeframe))

    best_score = float("-inf")
    iteration = 0

    while True:
        iteration += 1
        children = mutate_parent(parent, cfg.symbol, cfg.timeframe, n_children=cfg.candidates)

        with cf.ThreadPoolExecutor(max_workers=max(1, int(cfg.workers))) as pool:
            results = list(pool.map(lambda cand: _evaluate_candidate(cfg, cand, parent.get("strategy_id"), iteration), children))

        best = max(results, key=lambda x: x["score"])
        parent = _as_parent(best["candidate"])

        if best["score"] > best_score:
            best_score = best["score"]

        print(
            {
                "iteration": iteration,
                "best_strategy": best["candidate"].strategy_id,
                "score": round(float(best["score"]), 6),
                "passed": bool(best["passed"]),
                "return_pct": best["backtest"].get("return_pct"),
                "max_dd": best["backtest"].get("max_drawdown_pct"),
                "pf": best["backtest"].get("profit_factor"),
                "wr": best["backtest"].get("win_rate"),
            }
        )

        if best["passed"]:
            print("TARGET ACHIEVED")
            return

        if not cfg.continuous and iteration >= cfg.iterations:
            break

        time.sleep(max(0.1, float(cfg.sleep_seconds)))
