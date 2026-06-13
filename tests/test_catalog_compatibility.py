import unittest

import pandas as pd

from system_a_discovery_engine.layer1_content.catalog_utils import (
    ensure_layer1_catalog_columns,
)


class CatalogCompatibilityTests(unittest.TestCase):
    def test_simulator_catalog_gets_layer1_defaults(self):
        catalog = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "author_id": "a1",
                    "genres": "['Fantasy']",
                    "rating_count": 100,
                    "chapter_count": 10,
                    "avg_chapter_word_count": 2500,
                    "latent_quality": 0.75,
                }
            ]
        )

        normalized = ensure_layer1_catalog_columns(catalog)

        self.assertEqual(normalized.loc[0, "genres"], ["Fantasy"])
        self.assertEqual(normalized.loc[0, "review_count"], 8)
        self.assertEqual(normalized.loc[0, "page_count"], 100)
        self.assertIn("publish_date", normalized.columns)
        self.assertIn("avg_rating", normalized.columns)


if __name__ == "__main__":
    unittest.main()
