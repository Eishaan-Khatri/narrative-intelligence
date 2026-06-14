"""IPS stress tests for target-policy overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_b_opportunity_lab.offline_eval.ips_estimators import (
    clipped_ips,
    doubly_robust,
    inverse_propensity_score,
    self_normalized_ips,
)


def add_target_propensities(exposure_log: pd.DataFrame, item_scores: pd.DataFrame) -> pd.DataFrame:
    scores = item_scores[["item_id", "promotion_score"]].copy()
    scores["promotion_score"] = scores["promotion_score"].fillna(scores["promotion_score"].median()).clip(lower=0.001)
    score_sum = scores["promotion_score"].sum()
    scores["target_close"] = 0.75 * (1.0 / len(scores)) + 0.25 * (scores["promotion_score"] / score_sum)
    scores["target_moderate"] = 0.35 * (1.0 / len(scores)) + 0.65 * (scores["promotion_score"] / score_sum)
    scores["target_far"] = scores["promotion_score"] / score_sum
    return exposure_log.merge(scores[["item_id", "target_close", "target_moderate", "target_far"]], on="item_id", how="left")


def run_ips_stress_test(exposure_log: pd.DataFrame, item_scores: pd.DataFrame) -> pd.DataFrame:
    logged = add_target_propensities(exposure_log, item_scores)
    logged["q_hat_logged"] = logged.groupby("item_id")["reward"].transform("mean").fillna(logged["reward"].mean())
    global_q = float(logged["reward"].mean())
    rewards = logged["reward"].to_numpy(dtype=float)
    logging_p = logged["logging_propensity"].to_numpy(dtype=float)
    rows = []
    for name, col in [
        ("close_policy", "target_close"),
        ("moderate_policy", "target_moderate"),
        ("far_policy", "target_far"),
    ]:
        target_p = logged[col].fillna(1.0 / max(item_scores["item_id"].nunique(), 1)).to_numpy(dtype=float)
        q_logged = logged["q_hat_logged"].to_numpy(dtype=float)
        q_target = np.full_like(q_logged, global_q)
        weights = target_p / np.maximum(logging_p, 1e-8)
        rows.append(
            {
                "target_policy": name,
                "ips": inverse_propensity_score(rewards, logging_p, target_p),
                "snips": self_normalized_ips(rewards, logging_p, target_p),
                "clipped_ips_10": clipped_ips(rewards, logging_p, target_p, clip=10.0),
                "doubly_robust": doubly_robust(rewards, logging_p, target_p, q_logged, q_target),
                "mean_weight": float(np.mean(weights)),
                "p95_weight": float(np.quantile(weights, 0.95)),
                "max_weight": float(np.max(weights)),
                "effective_sample_size": float((weights.sum() ** 2) / np.maximum(np.sum(weights**2), 1e-8)),
            }
        )
    return pd.DataFrame(rows)
