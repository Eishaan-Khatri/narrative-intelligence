import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_store.simulator.markov_event_simulator import (
    _condition_emissions,
    _condition_transitions,
    _S,
)
from scripts.enrich_synthetic_catalog_text import enrich_catalog_text


class FinalSweepControlTests(unittest.TestCase):
    def test_simulator_calibration_knobs_reduce_exit_pressure(self):
        base_emissions = _condition_emissions(
            quality=0.5,
            affinity=0.3,
            speed_factor=1.0,
            patience_factor=1.0,
        )
        calibrated_emissions = _condition_emissions(
            quality=0.5,
            affinity=0.3,
            speed_factor=1.0,
            patience_factor=1.0,
            exit_probability_multiplier=0.45,
            speed_multiplier=1.25,
            patience_multiplier=1.35,
        )

        self.assertGreater(calibrated_emissions[_S["ENGAGED_SLOW"], 0], base_emissions[_S["ENGAGED_SLOW"], 0])
        self.assertLess(calibrated_emissions[_S["ENGAGED_SLOW"], 3], base_emissions[_S["ENGAGED_SLOW"], 3])

    def test_transition_knobs_reduce_exit_column(self):
        base_transitions = _condition_transitions(quality=0.4, affinity=0.2, chapter_index=4)
        calibrated_transitions = _condition_transitions(
            quality=0.4,
            affinity=0.2,
            chapter_index=4,
            transition_exit_multiplier=0.45,
            valley_multiplier=0.4,
            engaged_boost_multiplier=1.25,
        )

        self.assertLess(
            calibrated_transitions[_S["ENGAGED_SLOW"], _S["EXITING"]],
            base_transitions[_S["ENGAGED_SLOW"], _S["EXITING"]],
        )
        self.assertAlmostEqual(float(calibrated_transitions[_S["ENGAGED_SLOW"]].sum()), 1.0)

    def test_external_text_enrichment_preserves_item_ids_and_adds_descriptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            catalog_path = temp_path / "catalog.parquet"
            external_path = temp_path / "external.parquet"
            output_path = temp_path / "catalog_enriched.parquet"

            pd.DataFrame(
                [
                    {
                        "item_id": "item_00001",
                        "title": "Synthetic 1",
                        "author_id": "author_1",
                        "author_name": "Author 1",
                        "genres": "['Fantasy']",
                        "latent_quality": 0.7,
                        "topic_vector": str([0.2] * 5),
                    }
                ]
            ).to_parquet(catalog_path, index=False)

            pd.DataFrame(
                [
                    {
                        "title": "External Novel",
                        "description": "A city of memory, ambition, betrayal, loyalty, and political inheritance.",
                        "genres": ["Literary Fiction"],
                    }
                ]
            ).to_parquet(external_path, index=False)

            enriched = enrich_catalog_text(
                catalog_path=catalog_path,
                output_path=output_path,
                external_catalog_paths=[external_path],
                seed=7,
            )

            self.assertEqual(enriched["item_id"].tolist(), ["item_00001"])
            self.assertIn("description", enriched.columns)
            self.assertIn("political inheritance", enriched.loc[0, "description"])
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
