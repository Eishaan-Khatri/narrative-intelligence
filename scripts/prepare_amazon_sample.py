"""Normalize a local Amazon reviews/metadata sample into project catalog shape.

Amazon dataset mirrors vary in schema and size. This script intentionally takes
a local file path instead of assuming one source. Supported inputs: JSONL, JSONL
gzip, CSV, and Parquet.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "external"


def read_json_lines(path: Path, limit: int | None) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def read_input(path: Path, limit: int | None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz") or suffixes.endswith(".json.gz"):
        return read_json_lines(path, limit)
    if suffixes.endswith(".csv"):
        return pd.read_csv(path, nrows=limit)
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.head(limit) if limit is not None else df
    raise ValueError(f"Unsupported Amazon input format: {path}")


def first_present(row: pd.Series, names: Iterable[str], default: object = None) -> object:
    for name in names:
        if name in row and pd.notna(row[name]):
            value = row[name]
            if isinstance(value, (list, dict)) or str(value).strip():
                return value
    return default


def normalize_genres(value: object) -> list[str]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [str(x) for group in value for x in group][:5]
        return [str(x) for x in value][:5]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.replace(">", ",").split(",") if part.strip()][:5]
    return ["Amazon"]


def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        asin = str(first_present(row, ["asin", "parent_asin", "item_id", "product_id"], f"amazon_{idx}"))
        title = str(first_present(row, ["title", "product_title", "name"], f"Amazon Item {asin}"))
        description_value = first_present(row, ["description", "feature", "reviewText", "text", "summary"], "")
        if isinstance(description_value, list):
            description = " ".join(str(x) for x in description_value)
        else:
            description = str(description_value)
        rating = first_present(row, ["average_rating", "avg_rating", "overall", "rating"], 3.5)
        rating_count = first_present(row, ["rating_number", "rating_count", "review_count", "helpful_vote"], 0)
        genres = normalize_genres(first_present(row, ["categories", "category", "main_category", "store"], "Amazon"))
        records.append(
            {
                "source": "amazon",
                "item_id": f"amazon_{asin}",
                "external_id": asin,
                "title": title,
                "author_id": f"amazon_author_{abs(hash(title)) % 100000:05d}",
                "author_name": str(first_present(row, ["author", "brand", "store"], "Amazon")),
                "description": description[:5000],
                "genres": genres,
                "avg_rating": float(pd.to_numeric(pd.Series([rating]), errors="coerce").fillna(3.5).iloc[0]),
                "rating_count": int(pd.to_numeric(pd.Series([rating_count]), errors="coerce").fillna(0).iloc[0]),
                "review_count": int(pd.to_numeric(pd.Series([rating_count]), errors="coerce").fillna(0).iloc[0]),
                "page_count": 200,
                "chapter_count": 1,
                "publish_date": pd.Timestamp("2020-01-01"),
            }
        )
    out = pd.DataFrame(records)
    out = out.drop_duplicates("item_id").reset_index(drop=True)
    return out


def write_demo(path: Path) -> None:
    rows = [
        {"asin": "demo_001", "title": "Demo Fantasy Novel", "description": ["Magic, exile, and an impossible return."], "average_rating": 4.2, "rating_number": 120, "categories": [["Books", "Fantasy"]]},
        {"asin": "demo_002", "title": "Demo Mystery Novel", "description": ["A detective follows a vanished manuscript."], "average_rating": 3.9, "rating_number": 85, "categories": [["Books", "Mystery"]]},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a local Amazon sample file.")
    parser.add_argument("--input", type=Path, help="Path to Amazon JSONL/JSONL.GZ/CSV/Parquet file.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--demo", action="store_true", help="Create and normalize a tiny demo file.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = args.input
    if args.demo:
        input_path = OUT_DIR / "amazon_demo.jsonl"
        write_demo(input_path)
    if input_path is None:
        print("[FAIL] Provide --input PATH, or use --demo for a tiny schema check.")
        return 2

    df = read_input(input_path, args.limit)
    catalog = normalize_catalog(df)
    out_path = OUT_DIR / "amazon_catalog.parquet"
    catalog.to_parquet(out_path, index=False)
    print(f"[OK] Saved {len(catalog)} Amazon catalog rows to {out_path}")
    return 0 if len(catalog) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
