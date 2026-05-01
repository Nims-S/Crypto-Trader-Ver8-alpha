from __future__ import annotations
import pandas as pd

def classify_market(df: pd.DataFrame, htf: pd.DataFrame | None = None) -> str:
    if df is None or df.empty:
        return "unknown"
    last = df.iloc[-1]
    trend = float(last.get("adx", 0.0) or 0.0)
    vol = float(last.get("bb_width_rank", 0.0) or 0.0)
    if trend >= 20 and vol >= 0.5:
        return "trend"
    if vol >= 0.65:
        return "breakout"
    return "mean_reversion"
