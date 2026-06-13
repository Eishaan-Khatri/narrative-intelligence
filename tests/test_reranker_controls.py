import unittest

import pandas as pd

from system_a_discovery_engine.layer3_ranking.lambdamart_ranker import (
    RANKING_FEATURES,
    apply_reranker_controls,
    build_ranking_features,
    generate_synthetic_ranking_data,
)


class RerankerControlTests(unittest.TestCase):
    def test_build_ranking_features_adds_missing_tail_boost(self):
        df = generate_synthetic_ranking_data(n_users=3, candidates_per_user=5, seed=11)
        df = df.drop(columns=["tail_boost_score"])

        prepared = build_ranking_features(df)

        self.assertIn("tail_boost_score", prepared.columns)
        self.assertIn("tail_boost_score", RANKING_FEATURES)
        self.assertTrue((prepared["tail_boost_score"] == 0.0).all())

    def test_apply_reranker_controls_bounds_hazard_and_scales_tail(self):
        df = pd.DataFrame(
            {
                "hazard_score": [0.1, 0.8],
                "novelty_score": [1.0, 6.0],
            }
        )

        controlled = apply_reranker_controls(
            df,
            tail_boost_weight=0.2,
            survival_penalty_weight=0.5,
            max_hazard_score=0.4,
        )

        self.assertEqual(controlled["hazard_score"].tolist(), [0.05, 0.2])
        self.assertEqual(controlled["tail_boost_score"].tolist(), [0.0, 0.2])


if __name__ == "__main__":
    unittest.main()
