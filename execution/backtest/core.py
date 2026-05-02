from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from registry.store import record_experiment, upsert_strategy
from research.scoring import promotion_status, score_metrics
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


@dataclass
class Position:
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    qty: float
    open_index: int
    strategy_name: str
    bars_held: int = 0
    max_bars: int = 72
    trail_atr_mult: float = DEFAULT_TRAIL_ATR_MULT
    trail_ema20: bool = False
    be_moved: bool = False
    stop_to_be: float = 0.0
    tp_hit: bool = False


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _cache_path(sym: str, tf: str, since: int | None, until: int | None) -> Path:
    safe_sym = sym.replace("/", "_")
    since_s = str(since) if since is not None else "none"
    until_s = str(until) if until is not None else "none"
    return CACHE_DIR / f"{safe_sym}_{tf}_{since_s}_{until_s}.csv"


def _normalize_cached_frame(cached: pd.DataFrame) -> pd.DataFrame:
    if cached.empty or "timestamp" not in cached.columns:
        return pd.DataFrame()
    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
    cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if not REQUIRED_INDICATOR_COLS.issubset(cached.columns):
        cached = compute_indicators(cached.reset_index())
        if "timestamp" in cached.columns:
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
            cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return cached


def fetch_ohlcv_full(sym, tf, since=None, until=None, use_cache=True) -> pd.DataFrame:
    cache_file = _cache_path(sym, tf, since, until)
    if use_cache and cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            cached = _normalize_cached_frame(cached)
            if not cached.empty:
                return cached
        except Exception:
            pass

    rows = []
    cur = since
    while True:
        chunk = exchange.fetch_ohlcv(sym, timeframe=tf, since=cur, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        cur = chunk[-1][0] + 1
        if len(chunk) < 1000 or (until and cur >= until):
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = compute_indicators(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    if use_cache:
        df.reset_index().to_csv(cache_file, index=False)
    return df


def _htf_timeframe_for_symbol(symbol: str, ltf_timeframe: str) -> str:
    if symbol == "BTC/USDT":
        if ltf_timeframe == "1d":
            return "1w"
        return "1d" if ltf_timeframe in {"15m", "30m", "1h", "2h", "4h"} else "1h"
    return "4h" if ltf_timeframe in {"15m", "30m", "1h"} else "1d"


def _sig(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default) if obj is not None else default


def _slippage_price(price: float, atr: float, close: float, side: str) -> float:
    atr_pct = (atr / close) if close else 0.0
    slip = (SLIPPAGE_BPS / 10000.0) + (atr_pct * SLIPPAGE_ATR_MULT)
    return price * (1.0 + slip) if side == "LONG" else price * (1.0 - slip)


def _fees(price: float, qty: float, bps: float) -> float:
    return price * qty * (bps / 10000.0)


def _position_size(entry: float, stop: float, capital: float, size_multiplier: float = 1.0) -> float:
    if entry <= 0 or stop <= 0:
        return 0.0
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0
    risk_amount = capital * RISK_PER_TRADE
    qty_risk = risk_amount / stop_dist
    qty_notional = (capital * MAX_NOTIONAL_FRAC) / entry
    return max(0.0, min(qty_risk, qty_notional) * max(0.0, size_multiplier))


def _prepare_targets(signal, entry: float, stop: float):
    sl_dist = abs(entry - stop)
    tp1_pct = _sig(signal, "take_profit_pct", 0.0) or 0.0
    tp2_pct = _sig(signal, "secondary_take_profit_pct", 0.0) or 0.0
    tp3_pct = _sig(signal, "tp3_pct", 0.0) or 0.0

    if tp1_pct > 0:
        tp1 = entry * (1 + tp1_pct) if signal.side == "LONG" else entry * (1 - tp1_pct)
    else:
        tp1 = entry + sl_dist * DEFAULT_TP1_R if signal.side == "LONG" else entry - sl_dist * DEFAULT_TP1_R

    if tp2_pct > 0:
        tp2 = entry * (1 + tp2_pct) if signal.side == "LONG" else entry * (1 - tp2_pct)
    else:
        tp2 = entry + sl_dist * DEFAULT_TP2_R if signal.side == "LONG" else entry - sl_dist * DEFAULT_TP2_R

    if tp3_pct > 0:
        tp3 = entry * (1 + tp3_pct) if signal.side == "LONG" else entry * (1 - tp3_pct)
    else:
        tp3 = tp2

    be_trigger_rr = _sig(signal, "be_trigger_rr", DEFAULT_MOVE_BE_R) or DEFAULT_MOVE_BE_R
    be_trigger = entry + sl_dist * be_trigger_rr if signal.side == "LONG" else entry - sl_dist * be_trigger_rr

    return tp1, tp2, tp3, be_trigger


def _close_trade(position: Position, exit_price: float, reason: str, bar_close: float, bar_atr: float) -> dict[str, Any]:
    exit_price = _slippage_price(exit_price, bar_atr, bar_close, position.side)
    fee = _fees(exit_price, position.qty, MAKER_FEE_BPS)
    if position.side == "LONG":
        pnl = (exit_price - position.entry) * position.qty - fee
    else:
        pnl = (position.entry - exit_price) * position.qty - fee
    return {
        "side": position.side,
        "entry_price": position.entry,
        "exit_price": exit_price,
        "qty": position.qty,
        "pnl": pnl,
        "reason": reason,
        "bars_held": position.bars_held,
        "strategy": position.strategy_name,
    }


def run_backtest(
    sym: str,
    tf: str,
    start: str | None = None,
    end: str | None = None,
    allow_shorts: bool = False,
    max_bars: int = 0,
    use_cache: bool = True,
    strategy_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    since = _to_ms(start)
    until = _to_ms(end)

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if until:
        until = min(until, now_ms)

    df = fetch_ohlcv_full(sym, tf, since, until, use_cache=use_cache)
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    htf_tf = _htf_timeframe_for_symbol(sym, tf)
    df_htf = fetch_ohlcv_full(sym, htf_tf, since, until, use_cache=use_cache)
    if df_htf.empty:
        return {"error": f"no HTF data returned for {sym} on {htf_tf}"}

    if max_bars and max_bars > 0:
        warmup = min(400, len(df) - 1)
        df = df.iloc[-(max_bars + warmup):].copy()
        df_htf = df_htf[df_htf.index >= df.index.min()].copy()
        if df_htf.empty:
            return {"error": f"HTF data trimmed away for {sym} on {htf_tf}"}

    htf_pos = np.searchsorted(df_htf.index.values, df.index.values, side="right") - 1

    state = StrategyState(allow_shorts=allow_shorts)
    if strategy_override:
        params = strategy_override.get("parameters") or {}
        state = StrategyState(
            trades_this_week=state.trades_this_week,
            allow_shorts=bool(params.get("allow_shorts", state.allow_shorts)),
            min_adx=float(params.get("min_adx", state.min_adx)),
            min_atr_rank=float(params.get("min_atr_rank", state.min_atr_rank)),
            min_bb_rank=float(params.get("min_bb_rank", state.min_bb_rank)),
            rsi_long=float(params.get("rsi_long", state.rsi_long)),
            rsi_short=float(params.get("rsi_short", state.rsi_short)),
        )

    cash = 10_000.0
    equity_curve = [cash]
    trades: list[dict[str, Any]] = []
    position: Position | None = None
    cooldown_until = -1

    warmup_idx = max(260, 50)

    for i in range(warmup_idx, len(df) - 1):
        bar = df.iloc[i + 1]
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_open = float(bar["open"])
        bar_atr = float(bar["atr"])
        bar_ema20 = float(bar["ema20"]) if pd.notna(bar.get("ema20", np.nan)) else None
        current_htf = df_htf.iloc[: max(htf_pos[i + 1] + 1, 0)] if htf_pos[i + 1] >= 0 else df_htf.iloc[:0]

        if position is not None:
            position.bars_held += 1

            if position.bars_held >= position.max_bars:
                trade = _close_trade(position, bar_close, "MAX_BARS", bar_close, bar_atr)
                cash += position.entry * position.qty + trade["pnl"]
                trades.append(trade)
                position = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

            if position.side == "LONG":
                sl_hit = bar_low <= position.stop_loss
                tp_hit = bar_high >= position.take_profit
                be_hit = bar_high >= position.stop_to_be if position.stop_to_be else False
            else:
                sl_hit = bar_high >= position.stop_loss
                tp_hit = bar_low <= position.take_profit
                be_hit = bar_low <= position.stop_to_be if position.stop_to_be else False

            if sl_hit:
                trade = _close_trade(position, position.stop_loss, "SL", bar_close, bar_atr)
                cash += position.entry * position.qty + trade["pnl"]
                trades.append(trade)
                position = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

            if tp_hit:
                trade = _close_trade(position, position.take_profit, "TP", bar_close, bar_atr)
                cash += position.entry * position.qty + trade["pnl"]
                trades.append(trade)
                position = None
                cooldown_until = i + 1
                equity_curve.append(cash)
                continue

            if be_hit and not position.be_moved:
                position.stop_loss = position.entry
                position.be_moved = True

            if position.trail_ema20 and bar_ema20 is not None:
                if position.side == "LONG":
                    position.stop_loss = max(position.stop_loss, bar_ema20)
                else:
                    position.stop_loss = min(position.stop_loss, bar_ema20)
            else:
                trail = bar_close - (bar_atr * position.trail_atr_mult) if position.side == "LONG" else bar_close + (bar_atr * position.trail_atr_mult)
                if position.side == "LONG":
                    position.stop_loss = max(position.stop_loss, trail)
                else:
                    position.stop_loss = min(position.stop_loss, trail)

        if position is None and i + 1 > cooldown_until:
            window = df.iloc[: i + 1]
            signal = generate_signal(window, state=state, symbol=sym, df_htf=current_htf, strategy_override=strategy_override)

            if signal is None:
                equity_curve.append(cash)
                continue

            if signal.side not in {"LONG", "SHORT"}:
                equity_curve.append(cash)
                continue

            if signal.side == "SHORT" and not (allow_shorts or state.allow_shorts):
                equity_curve.append(cash)
                continue

            entry = _slippage_price(bar_open, bar_atr, bar_close, signal.side)
            sl_pct = _sig(signal, "stop_loss_pct", 0.0) or 0.0
            if sl_pct <= 0:
                equity_curve.append(cash)
                continue

            stop = entry * (1 - sl_pct) if signal.side == "LONG" else entry * (1 + sl_pct)
            tp1, tp2, tp3, be_trigger = _prepare_targets(signal, entry, stop)
            max_hold = int(_sig(signal, "max_bars_override", 0) or MAX_BARS_BY_REGIME.get(_sig(signal, "regime", "trend"), 30))
            size_multiplier = float(_sig(signal, "size_multiplier", 1.0) or 1.0)
            trail_ema20 = bool(_sig(signal, "trail_ema20", False))
            trail_atr_mult = float(_sig(signal, "trail_atr_mult", DEFAULT_TRAIL_ATR_MULT) or DEFAULT_TRAIL_ATR_MULT)

            qty = _position_size(entry, stop, cash, size_multiplier=size_multiplier)
            if qty <= 0:
                equity_curve.append(cash)
                continue

            fee = _fees(entry, qty, TAKER_FEE_BPS)
            cost = entry * qty + fee
            if cost > cash:
                equity_curve.append(cash)
                continue

            position = Position(
                side=signal.side,
                entry=entry,
                stop_loss=stop,
                take_profit=tp1,
                qty=qty,
                open_index=i + 1,
                strategy_name=str(_sig(signal, "strategy", "unknown")),
                bars_held=0,
                max_bars=max_hold,
                trail_atr_mult=trail_atr_mult,
                trail_ema20=trail_ema20,
                stop_to_be=be_trigger,
            )
            cash -= cost

        mark_to_market = cash + (position.qty * bar_close if position is not None else 0.0)
        equity_curve.append(mark_to_market)

    if position is not None:
        last = df.iloc[-1]
        trade = _close_trade(position, float(last["close"]), "EOD_CLOSE", float(last["close"]), float(last["atr"]))
        cash += position.entry * position.qty + trade["pnl"]
        trades.append(trade)

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses)) or 1e-9

    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak) * 100.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    result = {
        "symbol": sym,
        "ltf_timeframe": tf,
        "htf_timeframe": htf_tf,
        "trades": len(trades),
        "win_rate": round(len(wins) / max(len(trades), 1), 3),
        "profit_factor": round(gross_win / gross_loss, 4),
        "final_equity": round(cash, 2),
        "return_pct": round((cash / 10_000.0 - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "avg_trade_pnl": round(float(np.mean(pnls)) if pnls else 0.0, 4),
    }
    return result


def _maybe_log_experiment(args, result):
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
    ap.add_argument("--max-bars", type=int, default=0, help="Limit the tested window to the most recent N bars for faster iteration")
    ap.add_argument("--no-cache", action="store_true", help="Disable OHLCV caching")
    ap.add_argument("--allow-shorts", action="store_true", help="Enable short trades")
    ap.add_argument("--strategy-id", default=None, help="Registry id for the experiment")
    ap.add_argument("--base-strategy", default=None, help="Base strategy name for registry tracking")
    ap.add_argument("--version", type=int, default=1, help="Strategy registry version")
    ap.add_argument("--log-experiment", action="store_true", help="Write backtest result into strategy registry")
    a = ap.parse_args()
    result = run_backtest(
        a.symbol,
        a.timeframe,
        a.start,
        a.end,
        allow_shorts=a.allow_shorts,
        max_bars=a.max_bars,
        use_cache=not a.no_cache,
    )
    if a.log_experiment and "error" not in result:
        result["registry"] = _maybe_log_experiment(a, result)
    print(json.dumps(result, indent=2))
