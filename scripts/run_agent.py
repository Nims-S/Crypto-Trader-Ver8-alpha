from research.agent_runner import run_agent

if __name__ == "__main__":
    run_agent(
        symbol="BTC/USDT",
        timeframe="1h",
        iterations=1000,
        workers=4
    )