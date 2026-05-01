from research.coordinator import evolve
from config.defaults import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
print(evolve(DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, max_cycles=1))
