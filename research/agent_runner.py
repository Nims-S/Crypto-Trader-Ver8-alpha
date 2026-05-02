from __future__ import annotations

import concurrent.futures as cf
import random
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ccxt

from research.candidate_generator import mutate_parent, seed_strategy
from research.validation import build_walk_forward_folds, summarize_walk_forward_reports
from registry.store import record_experiment, upsert_strategy
from strategy import StrategyState, compute_indicators, generate_signal


# ---------------------------
# DATA
# ---------------------------
exchange = ccxt.binance({"enableRateLimit": True})


def fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=2000)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = compute_indicators(df)
    return df.set_index("ts")


# ---------------------------
# BACKTEST (SELF-CONTAINED)
# ---------------------------
def run_backtest(df: pd.DataFrame, symbol: str, params: Dict[str, Any]):
    state = StrategyState()
    cash = 10000
    trades = []
    position = None

    for i in range(250, len(df)-1):
        window = df.iloc[:i+1]
        bar = df.iloc[i+1]

        if position is None:
            sig = generate_signal(
                window,
                symbol=symbol,
                state=state,
                strategy_override={"parameters": params}
            )

            if sig:
                entry = float(bar["open"])
                stop = entry * (1 - sig.stop_loss_pct)
                tp = entry * (1 + sig.take_profit_pct)
                position = (entry, stop, tp)

        else:
            entry, stop, tp = position

            if bar["low"] <= stop:
                pnl = stop - entry
                trades.append({"pnl": pnl})
                cash += pnl
                position = None

            elif bar["high"] >= tp:
                pnl = tp - entry
                trades.append({"pnl": pnl})
                cash += pnl
                position = None

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    pf = sum(wins) / abs(sum(losses)) if losses else 1.0
    wr = len(wins) / max(len(trades), 1)

    return {
        "return_pct": (cash / 10000 - 1) * 100,
        "profit_factor": pf,
        "win_rate": wr,
        "trades": len(trades),
        "trades_detail": trades,
    }


# ---------------------------
# MONTE CARLO (FROM TRADES)
# ---------------------------
def monte_carlo(trades: List[dict], sims=200):
    if not trades:
        return {"worst_dd": 0}

    pnls = [t["pnl"] for t in trades]
    results = []

    for _ in range(sims):
        equity = 0
        peak = 0
        dd = 0

        for _ in range(len(pnls)):
            r = random.choice(pnls)
            equity += r
            peak = max(peak, equity)
            dd = min(dd, equity - peak)

        results.append(dd)

    return {"worst_dd": min(results)}


# ---------------------------
# EVALUATION
# ---------------------------
def evaluate(candidate, df, symbol):
    bt = run_backtest(df, symbol, candidate.parameters)

    folds = build_walk_forward_folds(df.reset_index(), n_splits=3)

    wf_reports = []
    for f in folds:
        sub = df[(df.index >= f.train_start) & (df.index <= f.test_end)]
        wf_reports.append(run_backtest(sub, symbol, candidate.parameters))

    wf = summarize_walk_forward_reports(wf_reports)

    mc = monte_carlo(bt["trades_detail"])

    score = bt["profit_factor"] + bt["win_rate"]

    return {
        "candidate": candidate,
        "backtest": bt,
        "walk_forward": wf,
        "monte_carlo": mc,
        "score": score,
    }


# ---------------------------
# AGENT LOOP
# ---------------------------
def run_agent(symbol="BTC/USDT", timeframe="1h", iterations=100, workers=4):
    df = fetch_data(symbol, timeframe)

    parent = seed_strategy(symbol, timeframe)

    for i in range(iterations):
        children = mutate_parent(parent, symbol, timeframe, n_children=workers)

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda c: evaluate(c, df, symbol), children))

        best = max(results, key=lambda x: x["score"])
        parent = best["candidate"]

        record_experiment(
            parent.strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            run_type="agent",
            parameters=asdict(parent),
            metrics=best,
            passed=True,
        )

        upsert_strategy(
            parent.strategy_id,
            base_strategy=parent.base_strategy,
            version=parent.version,
            parameters=asdict(parent),
            metrics=best,
        )

        print(
            f"[{i}] score={best['score']:.3f} "
            f"ret={best['backtest']['return_pct']:.2f}% "
            f"pf={best['backtest']['profit_factor']:.2f}"
        )