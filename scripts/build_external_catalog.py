"""Combine normalized Gutenberg/Amazon catalogs into one external catalog."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_EXTERNAL_DIR = PROJECT_ROOT / "data" / "raw" / "external"
SYNTHETIC_CATALOG = PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"

REQUIRED_COLUMNS = [
    "item_id",
    "title",
    "author_id",
    "author_name",
    "description",
    "genres",
    "avg_rating",
    "rating_count",
    "review_count",
    "page_count",
    "chapter_count",
    "publish_date",
]


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] Missing {path}")
        return None
    df = pd.read_parquet(path)
    print(f"[OK] Loaded {len(df)} rows from {path.name}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "author_id": "unknown_author",
        "author_name": "Unknown",
        "description": "",
        "genres": [[] for _ in range(len(out))],
        "avg_rating": 3.5,
        "rating_count": 0,
        "review_count": 0,
        "page_count": 200,
        "chapter_count": 1,
        "publish_date": pd.Timestamp("2020-01-01"),
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce").fillna(pd.Timestamp("2020-01-01"))
    return out[REQUIRED_COLUMNS + [c for c in out.columns if c not in REQUIRED_COLUMNS]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined external Gutenberg/Amazon catalog.")
    parser.add_argument("--activate", action="store_true", help="Copy combined catalog to data/synthetic/catalog.parquet after backing up current catalog.")
    args = parser.parse_args()

    RAW_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    frames = [
        load_if_exists(RAW_EXTERNAL_DIR / "gutenberg_catalog.parquet"),
        load_if_exists(RAW_EXTERNAL_DIR / "amazon_catalog.parquet"),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        print("[FAIL] No external catalogs found. Run download_gutenberg_sample.py and/or prepare_amazon_sample.py first.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_columns(combined)
    combined = combined.drop_duplicates("item_id").reset_index(drop=True)
    out_path = RAW_EXTERNAL_DIR / "external_catalog_combined.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[OK] Saved {len(combined)} combined external catalog rows to {out_path}")

    if args.activate:
        SYNTHETIC_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        if SYNTHETIC_CATALOG.exists():
            backup = SYNTHETIC_CATALOG.with_suffix(".backup.parquet")
            shutil.copy2(SYNTHETIC_CATALOG, backup)
            print(f"[OK] Backed up current catalog to {backup}")
        shutil.copy2(out_path, SYNTHETIC_CATALOG)
        print(f"[OK] Activated external catalog at {SYNTHETIC_CATALOG}")
        print("[WARN] Retrieval training still needs session/event data for these items.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
