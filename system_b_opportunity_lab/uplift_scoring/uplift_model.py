"""Causal uplift scoring for exploration exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def t_learner_uplift(
    frame: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate item-level uplift using separate treated/control regressors."""
    required = set(feature_cols + [treatment_col, outcome_col])
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = frame.copy()
    treated = out[treatment_col].astype(int) == 1
    X = out[feature_cols].fillna(0.0).astype(float)
    y = out[outcome_col].fillna(0.0).astype(float)

    if treated.sum() < 2 or (~treated).sum() < 2:
        treated_mean = float(y[treated].mean()) if treated.any() else 0.0
        control_mean = float(y[~treated].mean()) if (~treated).any() else 0.0
        out["uplift_score"] = treated_mean - control_mean
        out["treated_outcome_hat"] = treated_mean
        out["control_outcome_hat"] = control_mean
        return out

    treated_model = RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
    control_model = RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=random_state + 1, n_jobs=-1)
    treated_model.fit(X[treated], y[treated])
    control_model.fit(X[~treated], y[~treated])

    treated_hat = treated_model.predict(X)
    control_hat = control_model.predict(X)
    out["treated_outcome_hat"] = treated_hat
    out["control_outcome_hat"] = control_hat
    out["uplift_score"] = np.clip(treated_hat - control_hat, -1.0, 1.0)
    return out
