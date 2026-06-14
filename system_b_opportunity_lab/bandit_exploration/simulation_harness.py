"""Policy comparison harness for exploration strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_b_opportunity_lab.bandit_exploration.policies import (
    EpsilonGreedyPolicy,
    PopularityPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)


def _policy_factory(name: str, arms: list[str], popularity: dict[str, float], seed: int):
    rng = np.random.default_rng(seed)
    if name == "popularity":
        return PopularityPolicy(arms=arms, popularity=popularity, rng=rng)
    if name == "epsilon_greedy":
        return EpsilonGreedyPolicy(arms=arms, epsilon=0.10, rng=rng)
    if name == "ucb1":
        return UCB1Policy(arms=arms, c=1.4, rng=rng)
    if name == "thompson":
        return ThompsonSamplingPolicy(arms=arms, rng=rng)
    raise ValueError(f"Unknown policy: {name}")


def compare_bandit_policies(
    item_scores: pd.DataFrame,
    n_rounds: int = 12000,
    seed: int = 42,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    """Simulate reward/regret for several exploration policies."""
    policies = policies or ["popularity", "epsilon_greedy", "ucb1", "thompson"]
    items = item_scores.copy().reset_index(drop=True)
    if "promotion_score" not in items.columns:
        items["promotion_score"] = items.get("shrunk_mean", pd.Series(0.5, index=items.index))
    if "true_quality" not in items.columns:
        items["true_quality"] = items["promotion_score"]

    arms = items["item_id"].astype(str).tolist()
    true_quality = dict(zip(arms, items["true_quality"].astype(float)))
    popularity = dict(zip(arms, items.get("impressions", pd.Series(1.0, index=items.index)).astype(float)))
    oracle = max(true_quality.values()) if true_quality else 0.0
    rows = []

    for policy_name in policies:
        policy = _policy_factory(policy_name, arms, popularity, seed)
        rng = np.random.default_rng(seed + len(policy_name))
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        for t in range(1, n_rounds + 1):
            arm = policy.select_arm()
            p = float(np.clip(true_quality[arm], 0.0, 1.0))
            reward = float(rng.random() < p)
            policy.update(arm, reward)
            cumulative_reward += reward
            cumulative_regret += oracle - p
            if t % max(1, n_rounds // 60) == 0:
                rows.append(
                    {
                        "policy": policy_name,
                        "round": t,
                        "cumulative_reward": cumulative_reward,
                        "cumulative_regret": cumulative_regret,
                        "unique_items_exposed": sum(1 for arm_id in arms if policy.counts[arm_id] > 0),
                    }
                )
    return pd.DataFrame(rows)
