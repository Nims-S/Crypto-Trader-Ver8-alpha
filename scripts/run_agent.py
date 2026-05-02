import argparse

from research.agent_runner import AgentConfig, run_agent


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run autonomous research agent")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--start", default="2022-01-02")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--goal-return", type=float, default=30.0)
    p.add_argument("--max-dd", type=float, default=15.0)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--continuous", action="store_true", help="Run indefinitely until target achieved")
    p.add_argument("--sleep-seconds", type=float, default=1.0)

    args = p.parse_args()

    cfg = AgentConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        goal_return=args.goal_return,
        max_dd=args.max_dd,
        iterations=args.iterations,
        candidates=args.candidates,
        folds=args.folds,
        workers=args.workers,
        continuous=args.continuous,
        sleep_seconds=args.sleep_seconds,
    )

    run_agent(cfg)
