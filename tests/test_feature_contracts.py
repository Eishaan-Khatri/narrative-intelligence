import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_store.build_session_features import compute_session_features
from feature_store.build_temporal_features import build_temporal_features


class FeatureContractTests(unittest.TestCase):
    def test_session_completion_is_fraction_not_raw_percent(self):
        events = pd.DataFrame(
            [
                {
                    "reconstructed_session_id": "s1",
                    "session_id": "raw_s1",
                    "user_id": "u1",
                    "item_id": "i1",
                    "chapter_index": 0,
                    "timestamp": pd.Timestamp("2024-01-01T00:00:00"),
                    "event_type": "OPEN",
                    "scroll_depth_pct": 0.0,
                    "dwell_time_sec": 0.0,
                    "device_type": "mobile",
                    "metadata": {},
                },
                {
                    "reconstructed_session_id": "s1",
                    "session_id": "raw_s1",
                    "user_id": "u1",
                    "item_id": "i1",
                    "chapter_index": 0,
                    "timestamp": pd.Timestamp("2024-01-01T00:01:00"),
                    "event_type": "SCROLL_TICK",
                    "scroll_depth_pct": 50.0,
                    "dwell_time_sec": 60.0,
                    "device_type": "mobile",
                    "metadata": {},
                },
                {
                    "reconstructed_session_id": "s1",
                    "session_id": "raw_s1",
                    "user_id": "u1",
                    "item_id": "i1",
                    "chapter_index": 0,
                    "timestamp": pd.Timestamp("2024-01-01T00:02:00"),
                    "event_type": "SCROLL_TICK",
                    "scroll_depth_pct": 75.0,
                    "dwell_time_sec": 60.0,
                    "device_type": "mobile",
                    "metadata": {},
                },
            ]
        )
        catalog = pd.DataFrame(
            [{"item_id": "i1", "avg_chapter_word_count": 1000}]
        )

        features = compute_session_features(events, catalog)

        self.assertEqual(len(features), 1)
        self.assertEqual(features.loc[0, "final_completion_pct"], 0.75)

    def test_temporal_features_accept_nmf_tv_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_path = temp_path / "session_features.parquet"
            topic_path = temp_path / "topic_vectors.parquet"
            output_path = temp_path / "user_temporal_features.parquet"

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
                        "timestamp_start": pd.Timestamp("2024-01-01T00:00:00"),
                        "timestamp_end": pd.Timestamp("2024-01-01T00:02:00"),
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
                        "device_type": "mobile",
                        "timestamp_start": pd.Timestamp("2024-01-08T00:00:00"),
                        "timestamp_end": pd.Timestamp("2024-01-08T00:03:00"),
                    },
                ]
            )
            sessions.to_parquet(session_path, index=False)

            topics = pd.DataFrame(
                {
                    "item_id": ["i1", "i2"],
                    **{
                        f"tv_{idx}": [
                            1.0 if idx == 0 else 0.0,
                            1.0 if idx == 1 else 0.0,
                        ]
                        for idx in range(40)
                    },
                }
            )
            topics.to_parquet(topic_path, index=False)

            features = build_temporal_features(
                session_features_path=session_path,
                output_path=output_path,
                topic_vectors_path=topic_path,
            )

            self.assertEqual(len(features), 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(bool(features.iloc[1]["genre_drift_flag"]))


if __name__ == "__main__":
    unittest.main()
