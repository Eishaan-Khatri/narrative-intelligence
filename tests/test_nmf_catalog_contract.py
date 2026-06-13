import unittest
from unittest.mock import patch

import pandas as pd

from system_a_discovery_engine.layer1_content import nmf_topics


class NmfCatalogContractTests(unittest.TestCase):
    def test_prepare_catalog_text_fields_accepts_simulator_schema(self):
        catalog = pd.DataFrame(
            [
                {
                    "item_id": "i1",
                    "title": "Book 1",
                    "author_name": "Author 1",
                    "genres": "['Fantasy', 'Mystery']",
                    "topic_vector": [0.9, 0.1] + [0.0] * 38,
                }
            ]
        )

        descriptions, genres = nmf_topics.prepare_catalog_text_fields(catalog)

        self.assertEqual(genres.iloc[0], ["Fantasy", "Mystery"])
        self.assertIn("Book 1", descriptions.iloc[0])
        self.assertIn("latent_topic_0", descriptions.iloc[0])

        processed = nmf_topics.preprocess_texts(descriptions, genres)
        self.assertIn("latent_topic_0", processed[0])

    def test_load_stopwords_falls_back_when_nltk_is_missing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "nltk":
                raise ImportError("missing nltk")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            stopwords = nmf_topics._load_stopwords()

        self.assertIn("the", stopwords)


if __name__ == "__main__":
    unittest.main()
