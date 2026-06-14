"""Simple conformal-style uncertainty intervals for breakout scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_conformal_intervals(
    frame: pd.DataFrame,
    score_col: str = "breakout_score",
    label_col: str = "breakout_label",
    alpha: float = 0.10,
) -> pd.DataFrame:
    out = frame.copy()
    if score_col not in out.columns:
        out[score_col] = 0.0
    if label_col in out.columns and out[label_col].nunique() > 1:
        residual = np.abs(out[label_col].astype(float) - out[score_col].astype(float))
        q = float(np.quantile(residual, 1.0 - alpha))
    else:
        q = 0.20
    out["breakout_lower"] = np.clip(out[score_col] - q, 0.0, 1.0)
    out["breakout_upper"] = np.clip(out[score_col] + q, 0.0, 1.0)
    out["breakout_interval_width"] = out["breakout_upper"] - out["breakout_lower"]
    return out
