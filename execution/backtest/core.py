from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from registry.store import record_experiment, upsert_strategy
from research.scoring import score_metrics, promotion_status
from strategy import StrategyState, compute_indicators, generate_signal

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
CACHE_DIR = Path(".backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.01
MAX_NOTIONAL_FRAC = 0.25

DEFAULT_TP1_R = 1.8
DEFAULT_TP2_R = 4.5
DEFAULT_TP1_QTY_FRAC = 0.20
DEFAULT_MOVE_BE_R = 1.8
DEFAULT_TRAIL_ATR_MULT = 1.5

MAX_BARS_BY_REGIME = {
    "trend": 30,
    "mean_reversion": 12,
}

REQUIRED_INDICATOR_COLS = {
    "atr",
    "atr_pct",
    "atr_pct_rank",
    "bb_width",
    "bb_width_rank",
    "rolling_body",
    "ema20",
    "ema50",
    "ema200",
    "adx",
    "rsi",
    "macd_hist",
}

# (rest of file unchanged until logger)


def _maybe_log_experiment(args, result):
    # FIX: use canonical scoring module directly (no legacy evolution import)
    strategy_id = args.strategy_id or f"{args.symbol.replace('/', '_').lower()}_{args.timeframe}_{'short' if args.allow_shorts else 'long'}"
    decision = score_metrics(result)
    registry_payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "allow_shorts": bool(args.allow_shorts),
        "max_bars": int(args.max_bars or 0),
        "decision": decision.as_dict(),
    }
    experiment = record_experiment(
        strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_type="backtest",
        parameters=registry_payload,
        metrics={**result, "decision": decision.as_dict()},
        passed=decision.passed,
        notes="auto-logged from backtest.py",
    )
    upsert_strategy(
        strategy_id,
        base_strategy=args.base_strategy or strategy_id,
        version=int(args.version or 1),
        status=promotion_status(decision),
        parameters=registry_payload,
        metrics={**result, "decision": decision.as_dict()},
        tags=[args.symbol, args.timeframe, "backtest"],
        source="backtest",
        notes=f"decision={'pass' if decision.passed else 'fail'}",
        active=decision.passed,
    )
    return {"decision": decision.as_dict(), "experiment": experiment}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--allow-shorts", action="store_true")
    ap.add_argument("--strategy-id", default=None)
    ap.add_argument("--base-strategy", default=None)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--log-experiment", action="store_true")
    a = ap.parse_args()
    result = run_backtest(a.symbol, a.timeframe, a.start, a.end, allow_shorts=a.allow_shorts, max_bars=a.max_bars, use_cache=not a.no_cache)
    if a.log_experiment and "error" not in result:
        result["registry"] = _maybe_log_experiment(a, result)
    print(json.dumps(result, indent=2))
