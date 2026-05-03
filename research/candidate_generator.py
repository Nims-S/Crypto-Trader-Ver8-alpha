from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict

from research.feedback import build_feedback_summary


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


def _apply_directives(params: dict, directives: dict, symbol: str) -> dict:
    if not directives:
        return params

    for key in (
        "use_htf_filter",
        "use_volume_filter",
        "use_structure_filter",
        "use_reclaim_filter",
        "use_trend_filter",
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
        "trail_atr_mult",
        "trail_ema20",
        "cooldown_bars",
        "max_bars_override",
        "confidence",
        "size_multiplier",
    ):
        if key in directives:
            params[key] = directives[key]

    if "min_adx_delta" in directives:
        params["min_adx"] = max(5.0, _safe_float(params.get("min_adx", 10), 10) + directives["min_adx_delta"])
    if "min_atr_rank_multiplier" in directives:
        params["min_atr_rank"] = max(0.01, _safe_float(params.get("min_atr_rank", 0.1), 0.1) * directives["min_atr_rank_multiplier"])
    if "min_bb_rank_multiplier" in directives:
        params["min_bb_rank"] = max(0.01, _safe_float(params.get("min_bb_rank", 0.1), 0.1) * directives["min_bb_rank_multiplier"])

    if symbol.startswith("BTC") and directives.get("prefer_trend_pullback"):
        params["entry_mode"] = "trend_pullback"
        params["use_htf_filter"] = True
        params["use_trend_filter"] = True

    return params


def _mutate_trend_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "trend_pullback"
    p["use_htf_filter"] = True
    p["use_trend_filter"] = True
    p["use_volume_filter"] = True
    p["use_structure_filter"] = True
    p.setdefault("min_adx", 22.0)
    p.setdefault("htf_adx_min", 18.0)
    p.setdefault("htf_bb_rank_min", 0.30)
    p.setdefault("min_bb_rank", 0.38)
    p.setdefault("min_atr_rank", 0.32)
    p.setdefault("rsi_min", 52.0)
    p.setdefault("rsi_max", 70.0)
    p.setdefault("volume_multiplier", 1.08)
    p.setdefault("pullback_lookback", 3)
    p.setdefault("pullback_bars", 1)
    p.setdefault("stop_atr_mult", 2.0)
    p.setdefault("tp1_rr", 2.3)
    p.setdefault("tp2_rr", 4.0)
    p.setdefault("tp1_close_fraction", 0.50)
    p.setdefault("tp2_close_fraction", 0.50)
    p.setdefault("trail_atr_mult", 1.5)
    p.setdefault("cooldown_bars", 30)
    p.setdefault("max_bars_override", 84)
    p.setdefault("confidence", 0.84)
    p.setdefault("size_multiplier", 0.80)

    p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 22.0), 22.0) + rng.uniform(-2.0, 3.0), 15.0, 35.0)
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.38), 0.38) + rng.uniform(-0.05, 0.05), 0.20, 0.60)
    p["min_atr_rank"] = _clamp(_safe_float(p.get("min_atr_rank", 0.32), 0.32) + rng.uniform(-0.05, 0.05), 0.15, 0.60)
    p["htf_adx_min"] = _clamp(_safe_float(p.get("htf_adx_min", 18.0), 18.0) + rng.uniform(-2.0, 2.0), 12.0, 30.0)
    p["htf_bb_rank_min"] = _clamp(_safe_float(p.get("htf_bb_rank_min", 0.30), 0.30) + rng.uniform(-0.04, 0.04), 0.10, 0.55)
    p["rsi_min"] = _clamp(_safe_float(p.get("rsi_min", 52.0), 52.0) + rng.uniform(-3.0, 2.0), 40.0, 60.0)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 70.0), 70.0) + rng.uniform(-2.0, 3.0), 60.0, 80.0)
    p["volume_multiplier"] = _clamp(_safe_float(p.get("volume_multiplier", 1.08), 1.08) + rng.uniform(-0.08, 0.12), 0.95, 1.35)
    p["pullback_lookback"] = int(_clamp(int(_safe_float(p.get("pullback_lookback", 3), 3)) + rng.choice([-1, 0, 1]), 2, 6))
    p["pullback_bars"] = int(_clamp(int(_safe_float(p.get("pullback_bars", 1), 1)) + rng.choice([0, 1]), 1, 4))
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 2.0), 2.0) + rng.uniform(-0.25, 0.35), 1.2, 3.0)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.3), 2.3) + rng.uniform(-0.35, 0.35), 1.4, 3.5)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 4.0), 4.0), p["tp1_rr"] + 0.8) + rng.uniform(-0.5, 0.6), 2.5, 6.5)
    p["tp1_close_fraction"] = _clamp(_safe_float(p.get("tp1_close_fraction", 0.50), 0.50) + rng.uniform(-0.10, 0.10), 0.20, 0.80)
    p["tp2_close_fraction"] = _clamp(1.0 - p["tp1_close_fraction"], 0.20, 0.80)
    p["trail_atr_mult"] = _clamp(_safe_float(p.get("trail_atr_mult", 1.5), 1.5) + rng.uniform(-0.20, 0.20), 1.0, 2.5)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 30), 30)) + rng.choice([-6, -3, 0, 3, 6]), 12, 48))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 84), 84)) + rng.choice([-18, -12, 0, 12, 18]), 36, 144))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.84), 0.84) + rng.uniform(-0.05, 0.05), 0.60, 0.95)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.80), 0.80) + rng.uniform(-0.15, 0.10), 0.40, 1.00)

    if rng.random() < 0.25:
        p["use_volume_filter"] = False
    if rng.random() < 0.20:
        p["use_structure_filter"] = False
    if rng.random() < 0.15:
        p["trail_ema20"] = True

    return p


def _mutate_breakout_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "breakout"
    p["use_htf_filter"] = True
    p["use_volume_filter"] = True
    p.setdefault("min_adx", 18.0)
    p.setdefault("min_bb_rank", 0.25)
    p.setdefault("stop_atr_mult", 1.7)
    p.setdefault("tp1_rr", 2.1)
    p.setdefault("tp2_rr", 3.5)
    p.setdefault("tp1_close_fraction", 0.40)
    p.setdefault("tp2_close_fraction", 0.60)
    p.setdefault("cooldown_bars", 24)
    p.setdefault("max_bars_override", 60)

    p["min_adx"] = _clamp(_safe_float(p.get("min_adx", 18.0), 18.0) + rng.uniform(-2.0, 4.0), 12.0, 32.0)
    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.25), 0.25) + rng.uniform(-0.05, 0.05), 0.10, 0.50)
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.7), 1.7) + rng.uniform(-0.20, 0.30), 1.0, 2.8)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 2.1), 2.1) + rng.uniform(-0.25, 0.25), 1.4, 3.2)
    p["tp2_rr"] = _clamp(max(_safe_float(p.get("tp2_rr", 3.5), 3.5), p["tp1_rr"] + 0.8) + rng.uniform(-0.4, 0.5), 2.2, 5.5)
    p["tp1_close_fraction"] = _clamp(_safe_float(p.get("tp1_close_fraction", 0.40), 0.40) + rng.uniform(-0.08, 0.08), 0.20, 0.60)
    p["tp2_close_fraction"] = _clamp(1.0 - p["tp1_close_fraction"], 0.30, 0.80)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 24), 24)) + rng.choice([-6, -3, 0, 3, 6]), 10, 40))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 60), 60)) + rng.choice([-12, -6, 0, 6, 12]), 24, 120))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.78), 0.78) + rng.uniform(-0.05, 0.05), 0.55, 0.92)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.65), 0.65) + rng.uniform(-0.12, 0.10), 0.30, 0.90)
    return p


def _mutate_mean_reversion_params(rng: random.Random, params: dict) -> dict:
    p = copy.deepcopy(params)
    p["entry_mode"] = "mean_reversion"
    p.setdefault("min_bb_rank", 0.30)
    p.setdefault("rsi_max", 32.0)
    p.setdefault("stop_atr_mult", 1.6)
    p.setdefault("tp1_rr", 1.8)
    p.setdefault("cooldown_bars", 18)
    p.setdefault("max_bars_override", 36)

    p["min_bb_rank"] = _clamp(_safe_float(p.get("min_bb_rank", 0.30), 0.30) + rng.uniform(-0.05, 0.04), 0.08, 0.45)
    p["rsi_max"] = _clamp(_safe_float(p.get("rsi_max", 32.0), 32.0) + rng.uniform(-3.0, 2.0), 20.0, 40.0)
    p["stop_atr_mult"] = _clamp(_safe_float(p.get("stop_atr_mult", 1.6), 1.6) + rng.uniform(-0.2, 0.3), 1.0, 2.5)
    p["tp1_rr"] = _clamp(_safe_float(p.get("tp1_rr", 1.8), 1.8) + rng.uniform(-0.2, 0.25), 1.2, 3.0)
    p["cooldown_bars"] = int(_clamp(int(_safe_float(p.get("cooldown_bars", 18), 18)) + rng.choice([-4, -2, 0, 2, 4]), 8, 30))
    p["max_bars_override"] = int(_clamp(int(_safe_float(p.get("max_bars_override", 36), 36)) + rng.choice([-6, 0, 6, 12]), 18, 72))
    p["confidence"] = _clamp(_safe_float(p.get("confidence", 0.65), 0.65) + rng.uniform(-0.05, 0.05), 0.50, 0.85)
    p["size_multiplier"] = _clamp(_safe_float(p.get("size_multiplier", 0.60), 0.60) + rng.uniform(-0.10, 0.08), 0.25, 0.85)
    return p


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None):
    rng = random.Random(seed)
    base_params = dict((parent or {}).get("parameters") or {})

    if feedback is None:
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)

    directives = (feedback or {}).get("mutation_directives") or {}
    base_entry_mode = str(base_params.get("entry_mode") or directives.get("entry_mode") or "trend_pullback")

    children = []
    for _ in range(max(1, n_children)):
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

        if rng.random() < 0.20:
            params["use_htf_filter"] = True
        if rng.random() < 0.15:
            params["use_volume_filter"] = False
        if rng.random() < 0.10:
            params["use_structure_filter"] = False

        sid = f"evo_{symbol.replace('/', '_').lower()}_{timeframe}_{rng.randint(1,999999)}"
        tags = [symbol, timeframe, "evo", mode]
        children.append(
            StrategyCandidate(
                sid,
                str((parent or {}).get("strategy_id") or "seed"),
                int((parent or {}).get("version", 0) or 0) + 1,
                params,
                symbol,
                timeframe,
                tags,
                "evolution",
            )
        )

    return children


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(f"{family}_{symbol.replace('/', '_').lower()}_{timeframe}_{random.randint(1,999999)}", "seed", 1, {}, symbol, timeframe, [symbol, timeframe, family], "seed")
