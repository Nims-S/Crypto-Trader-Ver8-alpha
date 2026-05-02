from research.agent_runner import run_agent, AgentConfig

if __name__ == "__main__":
    cfg = AgentConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start="2022-01-02",
        end="2026-12-31",
        goal_return=30,
        max_dd=15,
        iterations=1000,
        candidates=5,
        continuous=True,
    )

    run_agent(cfg)