"""Creator ecosystem simulations and fairness tradeoffs."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from system_b_opportunity_lab.fairness.gini_hhi import gini_coefficient, hhi, long_tail_viability


def exposure_fairness_by_policy(exposure_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_col = "policy" if "policy" in exposure_log.columns else "logging_policy"
    for (policy, day), group in exposure_log.groupby([policy_col, "day"]):
        creator_exposure = group.groupby("creator_id")["impression"].sum().to_numpy(dtype=float)
        rows.append(
            {
                "policy": policy,
                "day": int(day),
                "gini": gini_coefficient(creator_exposure),
                "hhi": hhi(creator_exposure),
                "long_tail_viability": long_tail_viability(creator_exposure, threshold=max(5, np.median(creator_exposure))),
                "total_impressions": float(creator_exposure.sum()),
                "active_creators": int((creator_exposure > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def multi_objective_score(
    relevance: np.ndarray,
    novelty: np.ndarray,
    creator_penalty: np.ndarray,
    lambda_novelty: float,
    lambda_fairness: float,
) -> np.ndarray:
    return relevance + lambda_novelty * novelty - lambda_fairness * creator_penalty


def pareto_frontier(
    item_scores: pd.DataFrame,
    lambda_values: list[float] | None = None,
    top_k: int = 200,
) -> pd.DataFrame:
    lambda_values = lambda_values or [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    items = item_scores.copy()
    relevance = items.get("promotion_score", items.get("shrunk_mean", pd.Series(0.5, index=items.index))).astype(float).to_numpy()
    popularity = items.get("popularity_percentile", pd.Series(0.5, index=items.index)).astype(float).to_numpy()
    novelty = 1.0 - popularity
    creator_pop = items.groupby("creator_id")["impressions"].transform("sum").fillna(0.0).astype(float)
    creator_penalty = creator_pop.rank(pct=True).to_numpy()

    rows = []
    for lambda_novelty, lambda_fairness in itertools.product(lambda_values, lambda_values):
        score = multi_objective_score(relevance, novelty, creator_penalty, lambda_novelty, lambda_fairness)
        selected = items.assign(_score=score).sort_values("_score", ascending=False).head(top_k)
        creator_exposure = selected.groupby("creator_id").size().to_numpy(dtype=float)
        rows.append(
            {
                "lambda_novelty": lambda_novelty,
                "lambda_fairness": lambda_fairness,
                "mean_relevance": float(selected.get("true_quality", selected["_score"]).mean()),
                "mean_novelty": float((1.0 - selected.get("popularity_percentile", 0.5)).mean()),
                "gini": gini_coefficient(creator_exposure),
                "hhi": hhi(creator_exposure),
                "long_tail_viability": long_tail_viability(creator_exposure, threshold=1),
            }
        )
    frontier = pd.DataFrame(rows)
    frontier["tradeoff_score"] = frontier["mean_relevance"] - 0.20 * frontier["gini"] + 0.10 * frontier["mean_novelty"]
    return frontier.sort_values("tradeoff_score", ascending=False).reset_index(drop=True)
