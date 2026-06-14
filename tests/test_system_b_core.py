import unittest

import numpy as np
import pandas as pd

from system_b_opportunity_lab.bandit_exploration.policies import (
    EpsilonGreedyPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)
from system_b_opportunity_lab.bayesian_shrinkage.beta_binomial_shrinkage import (
    beta_binomial_shrinkage,
)
from system_b_opportunity_lab.fairness.gini_hhi import gini_coefficient, hhi
from system_b_opportunity_lab.offline_eval.ips_estimators import (
    clipped_ips,
    doubly_robust,
    inverse_propensity_score,
    self_normalized_ips,
)
from system_b_opportunity_lab.uplift_scoring.uplift_model import t_learner_uplift
from system_b_opportunity_lab.uncertainty_promotion.promotion_policy import (
    uncertainty_aware_score,
)


class SystemBCoreTests(unittest.TestCase):
    def test_beta_binomial_shrinkage_pulls_small_samples_to_prior(self):
        df = pd.DataFrame(
            {
                "item_id": ["small", "large"],
                "successes": [2, 200],
                "trials": [2, 400],
            }
        )

        out = beta_binomial_shrinkage(df, success_col="successes", trial_col="trials")
        small = out.loc[out["item_id"].eq("small"), "shrunk_mean"].iloc[0]
        large = out.loc[out["item_id"].eq("large"), "shrunk_mean"].iloc[0]

        self.assertLess(small, 1.0)
        self.assertAlmostEqual(large, 0.5, delta=0.05)
        self.assertGreater(out.loc[out["item_id"].eq("small"), "posterior_uncertainty"].iloc[0], 0.0)

    def test_fairness_metrics_detect_concentration(self):
        equal = np.array([10, 10, 10, 10], dtype=float)
        concentrated = np.array([40, 0, 0, 0], dtype=float)

        self.assertAlmostEqual(gini_coefficient(equal), 0.0)
        self.assertGreater(gini_coefficient(concentrated), 0.7)
        self.assertAlmostEqual(hhi(equal), 0.25)
        self.assertAlmostEqual(hhi(concentrated), 1.0)

    def test_bandit_policies_return_valid_arms_and_update(self):
        arms = ["a", "b", "c"]
        rng = np.random.default_rng(4)

        eps = EpsilonGreedyPolicy(arms, epsilon=0.0, rng=rng)
        ucb = UCB1Policy(arms, c=1.0)
        ts = ThompsonSamplingPolicy(arms, rng=rng)

        for policy in [eps, ucb, ts]:
            selected = policy.select_arm()
            self.assertIn(selected, arms)
            policy.update(selected, reward=1.0)
            self.assertGreater(policy.counts[selected], 0)

    def test_ips_estimators_recover_logged_policy_value(self):
        rewards = np.array([1.0, 0.0, 1.0, 1.0])
        logging = np.array([0.5, 0.5, 0.5, 0.5])
        target = np.array([0.5, 0.5, 0.5, 0.5])
        q_hat = np.array([0.6, 0.4, 0.7, 0.8])
        target_q = np.array([0.6, 0.4, 0.7, 0.8])

        self.assertAlmostEqual(inverse_propensity_score(rewards, logging, target), 0.75)
        self.assertAlmostEqual(self_normalized_ips(rewards, logging, target), 0.75)
        self.assertAlmostEqual(clipped_ips(rewards, logging, target, clip=10), 0.75)
        self.assertAlmostEqual(doubly_robust(rewards, logging, target, q_hat, target_q), 0.75)

    def test_uncertainty_score_rewards_upside_but_respects_relevance_floor(self):
        score = uncertainty_aware_score(
            shrunk_quality=0.55,
            breakout_score=0.60,
            uplift_score=0.10,
            uncertainty=0.20,
            relevance=0.50,
            min_relevance=0.30,
        )
        blocked = uncertainty_aware_score(
            shrunk_quality=0.95,
            breakout_score=0.95,
            uplift_score=0.50,
            uncertainty=0.50,
            relevance=0.10,
            min_relevance=0.30,
        )

        self.assertGreater(score, 0.0)
        self.assertEqual(blocked, float("-inf"))

    def test_t_learner_uplift_returns_positive_effect_for_treated_better_group(self):
        df = pd.DataFrame(
            {
                "feature": [0, 0, 1, 1, 2, 2, 3, 3],
                "treated": [0, 1, 0, 1, 0, 1, 0, 1],
                "reward": [0.1, 0.4, 0.2, 0.5, 0.3, 0.7, 0.4, 0.8],
            }
        )

        out = t_learner_uplift(df, feature_cols=["feature"], treatment_col="treated", outcome_col="reward")

        self.assertIn("uplift_score", out.columns)
        self.assertGreater(out["uplift_score"].mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
