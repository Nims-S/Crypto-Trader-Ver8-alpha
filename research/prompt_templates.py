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
        "You are Hermes. Output 3 hypothesis packets as strict JSON only. No prose.\n"
        f"Context: {payload}"
    )


def build_claude_prompt(context: dict[str, Any]) -> str:
    symbol = str(context.get("symbol") or "BTC/USDT")
    timeframe = str(context.get("timeframe") or "1h")
    parent_id = str(context.get("parent_strategy_id") or "seed")

    payload = _jsonish_lines(
        [
            ("parent_strategy_id", parent_id),
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("directives", context.get("mutation_directives") or {}),
        ]
    )

    return (
        "Mutate strategy. Return JSON with parameter_updates only. No prose.\n"
        f"Context: {payload}"
    )


def build_child_batch_prompts(context: dict[str, Any], n: int = 3) -> list[dict[str, str]]:
    """Create diverse prompts for parallel mutation."""
    goals = ["density", "stability", "drawdown"]
    outputs = []
    for i in range(max(1, n)):
        goal = goals[i % len(goals)]
        prompt = (
            f"Mutate trading strategy focusing on {goal}. Return JSON: {{'parameter_updates':{{...}}}} only. "
            f"Context: {context}"
        )
        outputs.append({"goal": goal, "prompt": prompt})
    return outputs


def build_validator_prompt(context: dict[str, Any]) -> str:
    payload = _jsonish_lines(
        [
            ("symbol", context.get("symbol")),
            ("timeframe", context.get("timeframe")),
            ("metrics", context.get("metrics") or {}),
        ]
    )

    return f"Validate strategy JSON only. Context: {payload}"


def build_prompt_bundle(context: dict[str, Any]) -> dict[str, str]:
    return {
        "hermes": build_hermes_prompt(context),
        "claude": build_claude_prompt(context),
        "validator": build_validator_prompt(context),
    }
