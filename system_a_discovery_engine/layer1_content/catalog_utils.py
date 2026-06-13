"""Catalog compatibility helpers for Layer 1 content modules."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def normalise_genres(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        return [value]
    return []


def ensure_layer1_catalog_columns(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a catalog with the metadata columns Layer 1 modules expect."""
    catalog = catalog.copy()

    if "genres" in catalog.columns:
        catalog["genres"] = catalog["genres"].apply(normalise_genres)
    else:
        catalog["genres"] = [[] for _ in range(len(catalog))]

    if "avg_rating" not in catalog.columns:
        if "latent_quality" in catalog.columns:
            catalog["avg_rating"] = 1.0 + 4.0 * catalog["latent_quality"].astype(float)
        else:
            catalog["avg_rating"] = 3.5

    if "rating_count" not in catalog.columns:
        catalog["rating_count"] = 0

    if "review_count" not in catalog.columns:
        catalog["review_count"] = np.maximum(
            0,
            np.rint(catalog["rating_count"].fillna(0).astype(float) * 0.08).astype(int),
        )

    if "chapter_count" not in catalog.columns:
        catalog["chapter_count"] = 1

    if "page_count" not in catalog.columns:
        if "avg_chapter_word_count" in catalog.columns:
            words = (
                catalog["avg_chapter_word_count"].fillna(2500).astype(float)
                * catalog["chapter_count"].fillna(1).astype(float)
            )
            catalog["page_count"] = np.maximum(1, np.rint(words / 250.0).astype(int))
        else:
            catalog["page_count"] = 200

    if "publish_date" not in catalog.columns:
        base = pd.Timestamp("2020-01-01")
        catalog["publish_date"] = [base + pd.Timedelta(days=int(i % 1460)) for i in range(len(catalog))]
    else:
        catalog["publish_date"] = pd.to_datetime(catalog["publish_date"])

    return catalog
