import unittest

import pandas as pd

from system_b_opportunity_lab.breakout_forecasting.feature_builder import build_item_exposure_features
from system_b_opportunity_lab.exposure_simulation.simulation_harness import simulate_exposure_log


class SystemBPipelineSmokeTests(unittest.TestCase):
    def test_exposure_simulator_and_feature_builder_emit_required_columns(self):
        items = pd.DataFrame(
            {
                "item_id": ["i1", "i2", "i3"],
                "creator_id": ["c1", "c2", "c2"],
                "creator_name": ["C1", "C2", "C2"],
                "primary_genre": ["Fantasy", "Mystery", "Mystery"],
                "true_quality": [0.8, 0.4, 0.2],
                "base_popularity": [5.0, 2.0, 1.0],
                "popularity_percentile": [1.0, 0.5, 0.1],
            }
        )

        exposure, item_universe = simulate_exposure_log(
            item_universe=items,
            n_users=20,
            n_days=2,
            impressions_per_day=50,
            exploration_rate=0.2,
            seed=3,
        )
        features = build_item_exposure_features(exposure, item_universe)

        self.assertEqual(len(exposure), 100)
        self.assertIn("logging_propensity", exposure.columns)
        self.assertTrue((exposure["logging_propensity"] > 0).all())
        self.assertIn("reward_per_impression", features.columns)
        self.assertIn("breakout_label", features.columns)


if __name__ == "__main__":
    unittest.main()
