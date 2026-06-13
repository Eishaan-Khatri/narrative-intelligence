"""Download and normalize a small Project Gutenberg sample catalog.

Outputs are written under ``data/raw/external/`` and do not replace the main
pipeline catalog automatically. This gives the project real public-domain text
material without making the synthetic training path brittle.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "external"

DEFAULT_IDS = [
    1342,  # Pride and Prejudice
    84,    # Frankenstein
    11,    # Alice's Adventures in Wonderland
    1661,  # Sherlock Holmes
    2701,  # Moby-Dick
    98,    # A Tale of Two Cities
    74,    # Tom Sawyer
    76,    # Huckleberry Finn
    1080,  # A Modest Proposal
    5200,  # Metamorphosis
]


def fetch_text(gutenberg_id: int, timeout: int) -> str:
    urls = [
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
    ]
    errors: list[str] = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def metadata_value(text: str, key: str, fallback: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text[:5000])
    return match.group(1).strip() if match else fallback


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def stylometric_features(text: str) -> dict[str, float | int]:
    words = re.findall(r"[A-Za-z']+", text)
    word_count = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    unique_words = len({w.lower() for w in words})
    content_words = [w for w in words if w.lower() not in ENGLISH_STOP_WORDS]
    quote_chars = text.count('"') + text.count("'")
    chapter_count = max(1, len(re.findall(r"\bchapter\b", text, flags=re.IGNORECASE)))
    return {
        "word_count": int(word_count),
        "sentence_count": int(len(sentences)),
        "avg_sentence_length": float(word_count / max(len(sentences), 1)),
        "lexical_density": float(len(content_words) / max(word_count, 1)),
        "type_token_ratio": float(unique_words / max(word_count, 1)),
        "dialogue_ratio": float(quote_chars / max(len(text), 1)),
        "chapter_count": int(chapter_count),
        "page_count": int(max(1, round(word_count / 250))),
    }


def build_record(gutenberg_id: int, text: str) -> dict:
    title = metadata_value(text, "Title", f"Gutenberg {gutenberg_id}")
    author = metadata_value(text, "Author", "Unknown")
    clean_text = strip_gutenberg_boilerplate(text)
    description = " ".join(clean_text.split()[:220])
    features = stylometric_features(clean_text)
    return {
        "source": "gutenberg",
        "item_id": f"gutenberg_{gutenberg_id}",
        "external_id": str(gutenberg_id),
        "title": title,
        "author_id": "gutenberg_" + re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")[:48],
        "author_name": author,
        "description": description,
        "genres": ["Public Domain", "Literature"],
        "avg_rating": 3.8,
        "rating_count": 0,
        "review_count": 0,
        "publish_date": pd.Timestamp("1900-01-01"),
        "full_text": clean_text,
        **features,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Project Gutenberg sample texts.")
    parser.add_argument("--ids", default=",".join(str(i) for i in DEFAULT_IDS), help="Comma-separated Gutenberg IDs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    if args.limit is not None:
        ids = ids[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for gutenberg_id in ids:
        print(f"[Gutenberg] Downloading {gutenberg_id}...")
        try:
            text = fetch_text(gutenberg_id, timeout=args.timeout)
            records.append(build_record(gutenberg_id, text))
            time.sleep(args.sleep)
        except Exception as exc:
            failures.append({"gutenberg_id": gutenberg_id, "error": str(exc)})
            print(f"  [WARN] failed: {exc}")

    catalog = pd.DataFrame(records)
    catalog_path = OUT_DIR / "gutenberg_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "gutenberg_failures.csv", index=False)

    print(f"[OK] Saved {len(catalog)} Gutenberg rows to {catalog_path}")
    if failures:
        print(f"[WARN] {len(failures)} downloads failed; see {OUT_DIR / 'gutenberg_failures.csv'}")
    return 0 if len(catalog) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
