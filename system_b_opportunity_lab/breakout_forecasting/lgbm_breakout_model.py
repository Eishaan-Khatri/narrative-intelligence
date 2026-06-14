"""Breakout forecasting with LightGBM fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_breakout_model(
    features: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "breakout_label",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    X = features[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = features[label_col].fillna(0).astype(int)
    if y.nunique() < 2:
        out = features.copy()
        out["breakout_score"] = float(y.mean())
        return out, {"model": "constant", "roc_auc": 0.5, "average_precision": float(y.mean())}

    stratify = y if y.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=0.25, random_state=seed, stratify=stratify)
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            verbose=-1,
        )
        model_name = "lightgbm"
    except Exception:
        model = GradientBoostingClassifier(random_state=seed)
        model_name = "gradient_boosting"

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    out = features.copy()
    out["breakout_score"] = np.clip(scores, 0.0, 1.0)
    test_scores = out.iloc[test_idx]["breakout_score"]
    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y.iloc[test_idx], test_scores)) if y.iloc[test_idx].nunique() > 1 else 0.5,
        "average_precision": float(average_precision_score(y.iloc[test_idx], test_scores)),
        "positive_rate": float(y.mean()),
        "n_items": int(len(features)),
        "n_features": int(len(feature_cols)),
    }
    return out, metrics
