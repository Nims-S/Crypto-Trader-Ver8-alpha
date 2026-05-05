from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / ".strategy_store.json"

EVALUATOR_MODULE_CANDIDATES = (
    "scripts.backtest",
    "scripts.evaluator",
    "backtest",
    "evaluator",
    "engine.backtest",
    "strategy_backtester",
    "strategy_engine",
)
EVALUATOR_FUNCTION_CANDIDATES = (
    "run_backtest",
    "backtest_strategy",
    "evaluate_strategy",
    "evaluate_candidate",
    "score_strategy",
    "run_evaluation",
)


@dataclass
class GateConfig:
    min_profit_factor: float = 0.95
    min_win_rate: float = 0.45
    min_return_pct: float = 0.0
    max_drawdown_pct: float = 15.0
    min_trades: int = 20
    max_mc_drawdown_pct: float = 15.0
    min_density: float = 0.75


@dataclass
class CandidateResult:
    strategy_id: str
    parent_strategy_id: Optional[str]
    child_strategy_id: str
    status: str
    passed: bool
    score: float
    wf_passed: bool
    reasons: Tuple[str, ...]
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    created_at: str
    cycle_id: str
    symbol: str
    timeframe: str


class StrategyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "counters": {"evolution_id": 0, "experiment_id": 0},
            "evolution_runs": [],
            "experiments": [],
        }

    def next_evolution_id(self) -> int:
        counters = self.data.setdefault("counters", {})
        counters["evolution_id"] = int(counters.get("evolution_id", 0)) + 1
        return counters["evolution_id"]

    def append_run(self, payload: Dict[str, Any]) -> None:
        self.data.setdefault("evolution_runs", []).append(payload)
        self._save()

    def best_parent(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        runs = [
            r
            for r in self.data.get("evolution_runs", [])
            if r.get("symbol") == symbol and r.get("timeframe") == timeframe
        ]
        if not runs:
            return None
        scored = sorted(runs, key=lambda r: (float(r.get("score", 0.0)), float(r.get("metrics", {}).get("backtest", {}).get("profit_factor", 0.0))), reverse=True)
        return scored[0]

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=False)
        tmp.replace(self.path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def infer_regime(parameters: Dict[str, Any]) -> str:
    entry_mode = str(parameters.get("entry_mode", "")).lower()
    if "trend" in entry_mode:
        return "trend"
    if "hybrid" in entry_mode:
        return "hybrid"
    return "mean_reversion"


def regime_weights(regime: str) -> Dict[str, float]:
    if regime == "trend":
        return {"trend": 0.65, "hybrid": 0.25, "mean_reversion": 0.10}
    if regime == "hybrid":
        return {"trend": 0.35, "hybrid": 0.40, "mean_reversion": 0.25}
    return {"trend": 0.20, "hybrid": 0.25, "mean_reversion": 0.55}


def choose_mutation_family(parent: Dict[str, Any], iteration: int, candidate_idx: int, rng: random.Random) -> str:
    parameters = parent.get("parameters", {}) if parent else {}
    parent_regime = infer_regime(parameters)
    weights = regime_weights(parent_regime)
    if parent.get("reasons"):
        reasons = " ".join(map(str, parent["reasons"]))
        if "val_weak" in reasons or "test_weak" in reasons:
            weights["trend"] += 0.10
            weights["hybrid"] += 0.10
        if "pf<" in reasons:
            weights["mean_reversion"] += 0.05
    if iteration > 5:
        weights["hybrid"] += 0.05
    families = list(weights.keys())
    probs = [max(0.01, weights[k]) for k in families]
    total = sum(probs)
    probs = [p / total for p in probs]
    return rng.choices(families, weights=probs, k=1)[0]


def mutate_parameters(base: Dict[str, Any], family: str, rng: random.Random) -> Dict[str, Any]:
    p = dict(base)

    def j(v: float, scale: float, lo: float, hi: float) -> float:
        return clamp(v + rng.gauss(0.0, scale), lo, hi)

    if family == "trend":
        p["entry_mode"] = "trend"
        p["use_trend_filter"] = True
        p["use_structure_filter"] = True
        p["use_htf_filter"] = True
        p["use_reclaim_filter"] = rng.random() < 0.35
        p["use_volume_filter"] = rng.random() < 0.50
        p["use_bb_filter"] = True
        p["min_adx"] = j(safe_float(p.get("min_adx", 12.0)), 3.0, 10.0, 35.0)
        p["min_bb_rank"] = j(safe_float(p.get("min_bb_rank", 0.20)), 0.08, 0.05, 0.65)
        p["min_atr_rank"] = j(safe_float(p.get("min_atr_rank", 0.18)), 0.07, 0.05, 0.60)
        p["rsi_max"] = j(safe_float(p.get("rsi_max", 40.0)), 5.0, 25.0, 60.0)
        p["stop_atr_mult"] = j(safe_float(p.get("stop_atr_mult", 2.8)), 0.5, 1.4, 5.0)
        p["tp1_rr"] = j(safe_float(p.get("tp1_rr", 2.2)), 0.4, 1.2, 5.0)
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 24)), 4.0, 8.0, 80.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 10)), 3.0, 1.0, 60.0))
    elif family == "hybrid":
        p["entry_mode"] = "hybrid"
        p["use_trend_filter"] = rng.random() < 0.75
        p["use_structure_filter"] = rng.random() < 0.80
        p["use_htf_filter"] = rng.random() < 0.85
        p["use_reclaim_filter"] = rng.random() < 0.55
        p["use_volume_filter"] = rng.random() < 0.55
        p["use_bb_filter"] = True
        p["min_adx"] = j(safe_float(p.get("min_adx", 8.0)), 2.0, 5.0, 28.0)
        p["min_bb_rank"] = j(safe_float(p.get("min_bb_rank", 0.12)), 0.05, 0.03, 0.55)
        p["min_atr_rank"] = j(safe_float(p.get("min_atr_rank", 0.12)), 0.05, 0.03, 0.55)
        p["rsi_max"] = j(safe_float(p.get("rsi_max", 34.0)), 4.0, 18.0, 55.0)
        p["stop_atr_mult"] = j(safe_float(p.get("stop_atr_mult", 2.0)), 0.35, 1.1, 4.5)
        p["tp1_rr"] = j(safe_float(p.get("tp1_rr", 2.0)), 0.3, 1.1, 4.5)
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 20)), 3.0, 6.0, 60.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 12)), 3.0, 1.0, 60.0))
    else:
        p["entry_mode"] = "mean_reversion"
        p["use_trend_filter"] = rng.random() < 0.20
        p["use_structure_filter"] = rng.random() < 0.35
        p["use_htf_filter"] = rng.random() < 0.30
        p["use_reclaim_filter"] = rng.random() < 0.70
        p["use_volume_filter"] = rng.random() < 0.45
        p["use_bb_filter"] = True
        p["min_adx"] = j(safe_float(p.get("min_adx", 5.0)), 1.5, 3.0, 18.0)
        p["min_bb_rank"] = j(safe_float(p.get("min_bb_rank", 0.08)), 0.04, 0.02, 0.35)
        p["min_atr_rank"] = j(safe_float(p.get("min_atr_rank", 0.08)), 0.04, 0.02, 0.35)
        p["rsi_max"] = j(safe_float(p.get("rsi_max", 30.0)), 3.5, 15.0, 42.0)
        p["stop_atr_mult"] = j(safe_float(p.get("stop_atr_mult", 1.6)), 0.20, 0.8, 3.0)
        p["tp1_rr"] = j(safe_float(p.get("tp1_rr", 1.8)), 0.25, 1.0, 3.5)
        p["max_bars_override"] = int(j(safe_float(p.get("max_bars_override", 18)), 2.0, 4.0, 40.0))
        p["cooldown_bars"] = int(j(safe_float(p.get("cooldown_bars", 16)), 3.0, 2.0, 80.0))

    p["confidence"] = clamp(j(safe_float(p.get("confidence", 0.6)), 0.10, 0.10, 0.95), 0.10, 0.95)
    p["size_multiplier"] = clamp(j(safe_float(p.get("size_multiplier", 0.85)), 0.08, 0.35, 1.35), 0.35, 1.35)
    p["tp1_close_fraction"] = clamp(j(safe_float(p.get("tp1_close_fraction", 0.2)), 0.08, 0.05, 0.70), 0.05, 0.70)
    p["tp2_close_fraction"] = clamp(j(safe_float(p.get("tp2_close_fraction", 0.3)), 0.08, 0.05, 0.85), 0.05, 0.85)

    return p


def build_candidate_id(symbol: str, timeframe: str, evolution_id: int, rng: random.Random) -> str:
    return f"evo_{symbol.lower().replace('/', '_')}_{timeframe}_{evolution_id}_{rng.randint(10000, 999999)}"


def metric_score_component(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def extract_split_stat(split: Dict[str, Any]) -> Dict[str, float]:
    return {
        "pf": safe_float(split.get("profit_factor"), 0.0),
        "wr": safe_float(split.get("win_rate"), 0.0),
        "return_pct": safe_float(split.get("return_pct"), 0.0),
        "dd": abs(safe_float(split.get("max_drawdown_pct"), 0.0)),
        "trades": safe_float(split.get("trades"), 0.0),
    }


def compute_ranking_score(metrics: Dict[str, Any]) -> float:
    backtest = metrics.get("backtest", {}) or {}
    walk_forward = metrics.get("walk_forward", {}) or {}
    monte_carlo = metrics.get("monte_carlo", {}) or {}

    pf = safe_float(backtest.get("profit_factor"), 0.0)
    wr = safe_float(backtest.get("win_rate"), 0.0)
    ret = safe_float(backtest.get("return_pct"), 0.0)
    dd = abs(safe_float(backtest.get("max_drawdown_pct"), 0.0))
    trades = safe_float(backtest.get("trades"), 0.0)

    wf_score = safe_float(walk_forward.get("score"), safe_float(walk_forward.get("composite"), 0.0))
    wf_spread = safe_float(walk_forward.get("score_spread"), 0.0)
    density = safe_float(walk_forward.get("density_mean"), 0.0)
    mc_dd = safe_float(monte_carlo.get("worst_drawdown_pct"), 0.0)

    pf_score = metric_score_component(pf, 0.85, 2.0)
    wr_score = metric_score_component(wr, 0.35, 0.70)
    ret_score = metric_score_component(ret, -25.0, 25.0)
    dd_score = 1.0 - metric_score_component(dd, 0.0, 30.0)
    trades_score = metric_score_component(trades, 10.0, 80.0)
    density_score = clamp(density, 0.0, 1.0)
    wf_score = clamp(wf_score, 0.0, 1.0)
    spread_penalty = metric_score_component(wf_spread, 0.0, 0.40)
    mc_penalty = metric_score_component(abs(mc_dd), 0.0, 20.0)

    score = (
        0.22 * pf_score
        + 0.18 * wr_score
        + 0.14 * ret_score
        + 0.16 * dd_score
        + 0.16 * wf_score
        + 0.08 * density_score
        + 0.06 * trades_score
        - 0.04 * spread_penalty
        - 0.06 * mc_penalty
    )
    return round(clamp(score, 0.0, 1.0), 6)


def evaluate_gates(metrics: Dict[str, Any], walk_forward: Dict[str, Any], cfg: GateConfig) -> Tuple[bool, Tuple[str, ...], Dict[str, bool]]:
    reasons: List[str] = []
    gate_state: Dict[str, bool] = {}

    backtest = metrics.get("backtest", {}) or {}
    monte_carlo = metrics.get("monte_carlo", {}) or {}

    pf = safe_float(backtest.get("profit_factor"), 0.0)
    wr = safe_float(backtest.get("win_rate"), 0.0)
    ret = safe_float(backtest.get("return_pct"), 0.0)
    dd = abs(safe_float(backtest.get("max_drawdown_pct"), 0.0))
    trades = int(safe_float(backtest.get("trades"), 0.0))
    mc_dd = abs(safe_float(monte_carlo.get("worst_drawdown_pct"), 0.0))

    gate_state["pf_gate"] = pf >= cfg.min_profit_factor
    gate_state["return_gate"] = ret >= cfg.min_return_pct
    gate_state["dd_gate"] = dd <= cfg.max_drawdown_pct
    gate_state["monte_carlo_gate"] = mc_dd <= cfg.max_mc_drawdown_pct
    gate_state["density_gate"] = trades >= cfg.min_trades
    gate_state["walk_forward_gate"] = bool(walk_forward.get("passed", False))

    if not gate_state["pf_gate"]:
        reasons.append(f"pf<{cfg.min_profit_factor}")
    if not gate_state["return_gate"]:
        reasons.append(f"return<{cfg.min_return_pct}")
    if not gate_state["dd_gate"]:
        reasons.append(f"dd>{cfg.max_drawdown_pct}")
    if not gate_state["walk_forward_gate"]:
        reasons.append("walk_forward_failed")
    if not gate_state["monte_carlo_gate"]:
        reasons.append(f"mc_dd>{cfg.max_mc_drawdown_pct}")
    if not gate_state["density_gate"]:
        reasons.append("density_gate")

    split_results = walk_forward.get("split_results", {}) or {}
    split_reasons: List[str] = []
    split_ok = True
    for split_name in ("train", "val", "test"):
        for split in split_results.get(split_name, []):
            stat = extract_split_stat(split)
            local_ok = True
            local_reasons: List[str] = []
            if stat["trades"] < cfg.min_trades:
                local_ok = False
                local_reasons.append("trades<20")
            if stat["pf"] < cfg.min_profit_factor:
                local_ok = False
                local_reasons.append(f"pf<{cfg.min_profit_factor}")
            if stat["wr"] < cfg.min_win_rate:
                local_ok = False
                local_reasons.append(f"wr<{cfg.min_win_rate}")
            if not local_ok:
                split_ok = False
                split_reasons.extend([f"{split_name}:{reason}" for reason in local_reasons])

    gate_state["split_gate"] = split_ok
    if not split_ok:
        reasons.extend(split_reasons)

    passed = all(gate_state.values())
    return passed, tuple(reasons), gate_state


def normalize_evaluation_output(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[1]
    raise TypeError(f"Unsupported evaluator return type: {type(result)!r}")


def resolve_evaluator() -> Callable[..., Any]:
    attempted: List[str] = []
    for module_name in EVALUATOR_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            attempted.append(module_name)
            continue
        for func_name in EVALUATOR_FUNCTION_CANDIDATES:
            fn = getattr(module, func_name, None)
            if callable(fn):
                return fn
    raise RuntimeError(
        "Could not locate an evaluator/backtest function. Tried: " + ", ".join(attempted) + ". "
        "Update EVALUATOR_MODULE_CANDIDATES / EVALUATOR_FUNCTION_CANDIDATES to match the repo."
    )


def call_evaluator(
    evaluator: Callable[..., Any],
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    sig = inspect.signature(evaluator)
    base_kwargs = {
        "symbol": symbol,
        "timeframe": timeframe,
        "ltf_timeframe": timeframe,
        "start": start,
        "end": end,
        "parameters": parameters,
        "params": parameters,
        "strategy_params": parameters,
        "config": parameters,
    }
    accepted = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
    try:
        result = evaluator(**accepted)
    except TypeError:
        # Fall back to several common calling conventions.
        for payload in (
            {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end, "parameters": parameters},
            {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end, "params": parameters},
            {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end},
            {"parameters": parameters},
            {"params": parameters},
        ):
            try:
                result = evaluator(**{k: v for k, v in payload.items() if k in sig.parameters})
                break
            except TypeError:
                continue
        else:
            raise
    return normalize_evaluation_output(result)


def strategy_family_name(parameters: Dict[str, Any]) -> str:
    return infer_regime(parameters)


def ensure_store_schema(store: StrategyRegistry) -> None:
    store.data.setdefault("counters", {}).setdefault("evolution_id", 0)
    store.data.setdefault("counters", {}).setdefault("experiment_id", 0)
    store.data.setdefault("evolution_runs", [])
    store.data.setdefault("experiments", [])


def select_parent(store: StrategyRegistry, symbol: str, timeframe: str) -> Dict[str, Any]:
    parent = store.best_parent(symbol, timeframe)
    if parent:
        return parent
    return {
        "strategy_id": "seed_mean_reversion",
        "parameters": {
            "entry_mode": "mean_reversion",
            "use_trend_filter": False,
            "use_structure_filter": False,
            "use_htf_filter": False,
            "use_reclaim_filter": True,
            "use_volume_filter": False,
            "min_adx": 5.0,
            "min_bb_rank": 0.08,
            "min_atr_rank": 0.08,
            "rsi_max": 30.0,
            "stop_atr_mult": 1.6,
            "tp1_rr": 1.8,
            "max_bars_override": 18,
            "cooldown_bars": 16,
            "confidence": 0.6,
            "size_multiplier": 0.85,
            "tp1_close_fraction": 0.2,
            "tp2_close_fraction": 0.3,
        },
        "score": 0.0,
        "reasons": (),
    }


def make_status(passed: bool, wf_passed: bool, hard_reasons: Sequence[str]) -> str:
    if passed and wf_passed:
        return "validated"
    if hard_reasons:
        return "rejected"
    return "candidate"


def evaluate_one(
    evaluator: Callable[..., Any],
    store: StrategyRegistry,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cfg: GateConfig,
    iteration: int,
    candidate_idx: int,
    rng_seed: int,
) -> CandidateResult:
    rng = random.Random(rng_seed)
    parent = select_parent(store, symbol, timeframe)
    family = choose_mutation_family(parent, iteration, candidate_idx, rng)
    parameters = mutate_parameters(parent.get("parameters", {}), family, rng)
    parameters["mutation_family"] = family
    parameters["regime_profile"] = strategy_family_name(parameters)

    child_id = build_candidate_id(symbol, timeframe, store.next_evolution_id(), rng)
    evaluation = call_evaluator(
        evaluator,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        parameters=parameters,
    )

    walk_forward = evaluation.get("walk_forward", {}) or {}
    passed, hard_reasons, gate_state = evaluate_gates(evaluation, walk_forward, cfg)
    score = compute_ranking_score(evaluation)
    status = make_status(passed, bool(walk_forward.get("passed", False)), hard_reasons)

    payload = {
        "id": store.next_evolution_id(),
        "cycle_id": f"iter_{iteration}",
        "child_strategy_id": child_id,
        "parent_strategy_id": parent.get("strategy_id"),
        "created_at": utc_now(),
        "symbol": symbol,
        "timeframe": timeframe,
        "status": status,
        "passed": passed,
        "score": score,
        "metrics": evaluation,
        "parameters": parameters,
        "notes": ", ".join(hard_reasons) if hard_reasons else "",
    }
    store.append_run(payload)

    return CandidateResult(
        strategy_id=child_id,
        parent_strategy_id=parent.get("strategy_id"),
        child_strategy_id=child_id,
        status=status,
        passed=passed,
        score=score,
        wf_passed=bool(walk_forward.get("passed", False)),
        reasons=hard_reasons,
        metrics={**evaluation, "gate_state": gate_state},
        parameters=parameters,
        created_at=utc_now(),
        cycle_id=f"iter_{iteration}",
        symbol=symbol,
        timeframe=timeframe,
    )


def print_candidate(result: CandidateResult) -> None:
    backtest = result.metrics.get("backtest", {}) or {}
    walk_forward = result.metrics.get("walk_forward", {}) or {}
    monte_carlo = result.metrics.get("monte_carlo", {}) or {}
    row = {
        "iteration": int(result.cycle_id.split("_")[-1]),
        "best_strategy": result.child_strategy_id,
        "score": result.score,
        "passed": result.passed,
        "status": result.status,
        "reasons": result.reasons,
        "return_pct": round(safe_float(backtest.get("return_pct"), 0.0), 4),
        "max_dd": round(safe_float(backtest.get("max_drawdown_pct"), 0.0), 4),
        "pf": round(safe_float(backtest.get("profit_factor"), 0.0), 4),
        "wr": round(safe_float(backtest.get("win_rate"), 0.0), 3),
        "wf_passed": result.wf_passed,
        "gate_state": result.metrics.get("gate_state", {}),
        "wf_score": round(safe_float(walk_forward.get("score"), safe_float(walk_forward.get("composite"), 0.0)), 6),
        "mc_dd": round(abs(safe_float(monte_carlo.get("worst_drawdown_pct"), 0.0)), 4),
    }
    print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolution agent with explicit gate logging")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pf", type=float, default=0.95)
    parser.add_argument("--min-wr", type=float, default=0.45)
    parser.add_argument("--min-return-pct", type=float, default=0.0)
    parser.add_argument("--max-dd-pct", type=float, default=15.0)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--max-mc-dd-pct", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = GateConfig(
        min_profit_factor=args.min_pf,
        min_win_rate=args.min_wr,
        min_return_pct=args.min_return_pct,
        max_drawdown_pct=args.max_dd_pct,
        min_trades=args.min_trades,
        max_mc_drawdown_pct=args.max_mc_dd_pct,
        min_density=0.75,
    )
    evaluator = resolve_evaluator()
    store = StrategyRegistry(STORE_PATH)
    ensure_store_schema(store)

    print({
        "git_revision": git_revision(),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "iterations": args.iterations,
        "candidates": args.candidates,
        "workers": args.workers,
        "gates": asdict(cfg),
    })

    rng = random.Random(args.seed)
    for iteration in range(1, args.iterations + 1):
        futures = []
        results: List[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for candidate_idx in range(max(1, args.candidates)):
                seed = rng.randint(0, 10**9)
                futures.append(
                    executor.submit(
                        evaluate_one,
                        evaluator,
                        store,
                        args.symbol,
                        args.timeframe,
                        args.start,
                        args.end,
                        cfg,
                        iteration,
                        candidate_idx,
                        seed,
                    )
                )
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: (r.score, float(r.metrics.get("backtest", {}).get("profit_factor", 0.0))), reverse=True)
        best = results[0]
        print_candidate(best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
