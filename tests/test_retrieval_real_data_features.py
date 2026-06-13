import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store.schema import (
    ITEM_TOWER_INPUT_DIM,
    USER_TOWER_INPUT_DIM,
)
from system_a_discovery_engine.layer2_retrieval.real_data_features import (
    build_real_training_arrays,
)
from system_a_discovery_engine.layer2_retrieval.train_loop import (
    build_item_popularity,
    oversample_tail_training_indices,
)


class RetrievalRealDataFeatureTests(unittest.TestCase):
    def test_build_real_training_arrays_from_pipeline_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_path = temp_path / "session_features.parquet"
            fingerprint_path = temp_path / "item_fingerprints.parquet"
            temporal_path = temp_path / "user_temporal_features.parquet"

            sessions = pd.DataFrame(
                [
                    {
                        "session_id": "s1",
                        "user_id": "u1",
                        "item_id": "i1",
                        "chapter_index": 0,
                        "session_duration_sec": 120.0,
                        "reading_velocity_wpm": 200.0,
                        "velocity_acceleration": 0.0,
                        "completion_curve": [0.2, 0.4, 0.6, 0.7, 0.8],
                        "completion_curve_shape": "steady",
                        "re_read_ratio": 0.0,
                        "final_completion_pct": 0.8,
                        "exit_reason": "mid_chapter",
                        "device_type": "mobile",
                        "timestamp_start": pd.Timestamp("2024-01-01T09:00:00"),
                        "timestamp_end": pd.Timestamp("2024-01-01T09:02:00"),
                    },
                    {
                        "session_id": "s2",
                        "user_id": "u1",
                        "item_id": "i2",
                        "chapter_index": 1,
                        "session_duration_sec": 180.0,
                        "reading_velocity_wpm": 210.0,
                        "velocity_acceleration": 0.0,
                        "completion_curve": [0.2, 0.4, 0.6, 0.8, 0.9],
                        "completion_curve_shape": "steady",
                        "re_read_ratio": 0.0,
                        "final_completion_pct": 0.9,
                        "exit_reason": "chapter_end",
                        "device_type": "desktop",
                        "timestamp_start": pd.Timestamp("2024-01-08T21:00:00"),
                        "timestamp_end": pd.Timestamp("2024-01-08T21:03:00"),
                    },
                ]
            )
            sessions.to_parquet(session_path, index=False)

            fingerprints = pd.DataFrame(
                {
                    "item_id": ["i1", "i2"],
                    **{f"tv_{i}": [0.1, 0.2] for i in range(40)},
                    **{f"ae_{i}": [0.01, 0.02] for i in range(32)},
                    "quality_score": [0.5, 0.8],
                    **{f"sf_{i}": [0.03, 0.04] for i in range(8)},
                }
            )
            fingerprints.to_parquet(fingerprint_path, index=False)

            temporal = pd.DataFrame(
                [
                    {
                        "user_id": "u1",
                        "time_bucket": "2024-W01",
                        "engagement_profile_vector": [0.1] * 8,
                        "inter_chapter_gap_trend": 0.0,
                        "genre_drift_flag": False,
                    },
                    {
                        "user_id": "u1",
                        "time_bucket": "2024-W02",
                        "engagement_profile_vector": [0.2] * 8,
                        "inter_chapter_gap_trend": 0.0,
                        "genre_drift_flag": True,
                    },
                ]
            )
            temporal.to_parquet(temporal_path, index=False)

            (
                user_features,
                item_features,
                log_popularity,
                item_catalog_features,
                item_ids,
            ) = build_real_training_arrays(
                session_path=session_path,
                fingerprint_path=fingerprint_path,
                temporal_path=temporal_path,
            )

            self.assertEqual(user_features.shape, (2, USER_TOWER_INPUT_DIM))
            self.assertEqual(item_features.shape, (2, ITEM_TOWER_INPUT_DIM))
            self.assertEqual(log_popularity.shape, (2,))
            self.assertEqual(item_catalog_features.shape, (2, ITEM_TOWER_INPUT_DIM))
            self.assertEqual(item_ids.tolist(), [0, 1])

    def test_tail_oversampling_repeats_low_popularity_positives(self):
        item_ids = np.array([0, 0, 0, 1, 2, 3], dtype=np.int64)
        train_indices = np.arange(len(item_ids), dtype=np.int64)
        item_popularity = build_item_popularity(item_ids, n_items=4)

        unchanged = oversample_tail_training_indices(
            train_indices=train_indices,
            item_ids=item_ids,
            item_popularity=item_popularity,
            factor=1,
        )
        oversampled = oversample_tail_training_indices(
            train_indices=train_indices,
            item_ids=item_ids,
            item_popularity=item_popularity,
            factor=3,
        )

        self.assertTrue(np.array_equal(unchanged, train_indices))
        self.assertGreater(len(oversampled), len(train_indices))
        repeated_item_ids = item_ids[oversampled[len(train_indices):]]
        self.assertTrue(set(repeated_item_ids).issubset({1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
