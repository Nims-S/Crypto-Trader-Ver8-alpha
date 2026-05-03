from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compact_list(values: Any, limit: int = 4) -> list[str]:
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return out
    for value in values[: max(0, int(limit))]:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _jsonish_lines(items: list[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for key, value in items:
        parts.append(f'"{key}": {value!r}')
    return "{ " + ", ".join(parts) + " }"


def build_hermes_prompt(context: dict[str, Any]) -> str:
    """Return a compact hypothesis-generation prompt."""
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    failure = context.get("failure_profile") or {}
    trade_activity = context.get("trade_activity") or {}
    directives = context.get("mutation_directives") or {}

    counts = failure.get("counts") or {}
    trade_mean = trade_activity.get("mean") or {}
    pf_mean = trade_activity.get("mean_pf") or {}
    wr_mean = trade_activity.get("mean_wr") or {}

    payload = _jsonish_lines(
        [
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("primary_failure", failure.get("primary", "other")),
            ("counts", counts),
            ("trade_mean", trade_mean),
            ("pf_mean", pf_mean),
            ("wr_mean", wr_mean),
            ("directives", directives),
        ]
    )

    return (
        "You are Hermes. Output 3 hypothesis packets as strict JSON only. "
        "No prose. Keep each packet distinct. Prefer speed over explanation.\n"
        f"Context: {payload}\n"
        "Rules: target one regime each; declare failure modes; declare the exact metric improved; "
        "include entry_ideas, exit_ideas, volatility_adaptation, trade_density_expectation, robustness_checks, gating_rules. "
        "At least one packet must favor density, one stability, one drawdown."
    )


def build_claude_prompt(context: dict[str, Any]) -> str:
    """Return a compact strategy-mutation prompt."""
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    parent_id = str(context.get("parent_strategy_id") or "seed")
    hypothesis = context.get("hypothesis_packet") or {}
    backtest = context.get("backtest") or {}
    walk_forward = context.get("walk_forward") or {}
    directives = context.get("mutation_directives") or {}

    payload = _jsonish_lines(
        [
            ("parent_strategy_id", parent_id),
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("hypothesis", hypothesis),
            ("backtest", {
                "return_pct": _safe_float(backtest.get("return_pct", 0.0)),
                "pf": _safe_float(backtest.get("profit_factor", 0.0)),
                "wr": _safe_float(backtest.get("win_rate", 0.0)),
                "dd": _safe_float(backtest.get("max_drawdown_pct", 0.0)),
                "trades": _safe_int(backtest.get("trades", 0)),
            }),
            ("wf", {
                "score": _safe_float(walk_forward.get("score", 0.0)),
                "passed": bool(walk_forward.get("passed", False)),
                "spread": _safe_float(walk_forward.get("score_spread", 0.0)),
            }),
            ("directives", directives),
        ]
    )

    return (
        "You are Claude Code. Mutate the parent strategy into 3 child variants. JSON only. "
        "Keep changes minimal, fast, and traceable.\n"
        f"Context: {payload}\n"
        "Hard rules: no new framework, no unrelated indicators, no broken risk control, no overfit hacks. "
        "Return 3 children; one density-focused, one stability-focused, one drawdown-focused."
    )


def build_validator_prompt(context: dict[str, Any]) -> str:
    """Return a compact deployability-check prompt."""
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    metrics = context.get("metrics") or {}
    payload = _jsonish_lines(
        [
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("metrics", metrics),
        ]
    )

    return (
        "You are a validator. Decide candidate|validated|paper|live|rejected. JSON only. "
        "Prioritize robustness, density, drawdown, then return.\n"
        f"Context: {payload}"
    )


def build_prompt_bundle(context: dict[str, Any]) -> dict[str, str]:
    """Return all prompt variants used by the research loop."""
    return {
        "hermes": build_hermes_prompt(context),
        "claude": build_claude_prompt(context),
        "validator": build_validator_prompt(context),
    }
