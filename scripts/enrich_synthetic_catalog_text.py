"""Enrich the synthetic catalog with stronger text fields for Layer 1 content.

External catalogs do not provide compatible user-session labels for this
project. This script uses external text as content augmentation while
preserving the synthetic item_id space used by the simulator and retrieval
labels.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"
RAW_EXTERNAL_DIR = PROJECT_ROOT / "data" / "raw" / "external"
DEFAULT_EXTERNAL_PATHS = [
    RAW_EXTERNAL_DIR / "external_catalog_combined.parquet",
    RAW_EXTERNAL_DIR / "gutenberg_catalog.parquet",
    RAW_EXTERNAL_DIR / "amazon_catalog.parquet",
]

GENRE_VOCAB = {
    "Fantasy": "magic kingdom prophecy quest mythic creature ancient curse",
    "Science Fiction": "space station artificial intelligence colony quantum signal future machine",
    "Sci-Fi": "space station artificial intelligence colony quantum signal future machine",
    "Romance": "relationship longing trust separation reconciliation intimate choice",
    "Thriller": "conspiracy surveillance pursuit secret threat deadline escape",
    "Mystery": "detective clue witness alibi hidden motive investigation",
    "Horror": "haunting fear ritual shadow isolation dread nightmare",
    "Literary Fiction": "memory family identity regret ambition silence inheritance",
    "Historical Fiction": "empire war migration archive court rebellion tradition",
    "Young Adult": "friendship school identity courage first choice belonging",
    "Non-Fiction": "evidence history society analysis practical insight argument",
    "Biography": "life struggle career letters public legacy private conflict",
    "Self-Help": "habit discipline mindset routine growth resilience goal",
    "Adventure": "journey map wilderness storm survival discovery",
    "Dystopian": "regime scarcity surveillance resistance forbidden city",
    "Comedy": "misunderstanding satire absurd family social embarrassment",
    "Drama": "conflict betrayal grief loyalty ambition decision consequence",
    "Crime": "case suspect corruption evidence city confession",
    "Mythology": "god legend ritual fate hero sacred world",
    "Philosophy": "ethics meaning truth reason consciousness society",
    "Poetry": "image rhythm grief landscape voice fragment longing",
}


def _normalise_genres(value: object) -> list[str]:
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


def _topic_tokens(value: object, top_k: int = 6) -> str:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return ""
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return ""
    top_indices = np.argsort(arr)[-top_k:][::-1]
    tokens: list[str] = []
    for idx in top_indices:
        weight = max(1, int(round(float(arr[idx]) * 30)))
        tokens.extend([f"latent_topic_{idx}"] * weight)
    return " ".join(tokens)


def _quality_tokens(value: object) -> str:
    try:
        quality = float(value)
    except (TypeError, ValueError):
        quality = 0.5
    if quality >= 0.75:
        return "high_quality immersive coherent polished satisfying"
    if quality >= 0.45:
        return "medium_quality readable uneven promising accessible"
    return "low_quality fragmented slow inconsistent niche"


def _load_external_descriptions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "description" not in df.columns:
            df["description"] = ""
        if "genres" not in df.columns:
            df["genres"] = [[] for _ in range(len(df))]
        if "title" not in df.columns:
            df["title"] = ""
        source = path.stem.replace("_catalog", "")
        df = df.copy()
        df["external_source"] = source
        frames.append(df[["title", "description", "genres", "external_source"]])
    if not frames:
        return pd.DataFrame(columns=["title", "description", "genres", "external_source"])
    external = pd.concat(frames, ignore_index=True)
    external["description"] = external["description"].fillna("").astype(str)
    external = external[external["description"].str.len() >= 40].reset_index(drop=True)
    return external


def enrich_catalog_text(
    catalog_path: Path = DEFAULT_CATALOG,
    output_path: Path | None = None,
    external_catalog_paths: list[Path] | None = None,
    external_mix_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Write an enriched catalog while preserving existing item ids."""
    output_path = output_path or catalog_path
    external_catalog_paths = external_catalog_paths or DEFAULT_EXTERNAL_PATHS
    rng = np.random.default_rng(seed)

    catalog = pd.read_parquet(catalog_path).copy()
    external = _load_external_descriptions(external_catalog_paths)
    use_external = not external.empty and external_mix_ratio > 0

    descriptions: list[str] = []
    sources: list[str] = []
    for _, row in catalog.iterrows():
        genres = _normalise_genres(row.get("genres", []))
        genre_text = " ".join(GENRE_VOCAB.get(genre, genre.lower()) for genre in genres)
        base_parts = [
            str(row.get("title", "")),
            str(row.get("author_name", "")),
            " ".join(genres),
            genre_text,
            _quality_tokens(row.get("latent_quality")),
            _topic_tokens(row.get("topic_vector")),
        ]

        source = "synthetic_template"
        if use_external and rng.random() <= external_mix_ratio:
            ext_row = external.iloc[int(rng.integers(0, len(external)))]
            ext_genres = " ".join(_normalise_genres(ext_row.get("genres", [])))
            base_parts.extend(
                [
                    str(ext_row.get("title", "")),
                    ext_genres,
                    str(ext_row.get("description", ""))[:2500],
                ]
            )
            source = str(ext_row.get("external_source", "external"))

        descriptions.append(" ".join(part for part in base_parts if str(part).strip()))
        sources.append(source)

    catalog["description"] = descriptions
    catalog["text_enrichment_source"] = sources
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output_path, index=False)
    print(f"[OK] Enriched catalog text saved to {output_path}")
    if use_external:
        print(f"[OK] Used {len(external)} external text rows from available catalogs.")
    else:
        print("[INFO] No external text rows found; used synthetic genre/topic templates.")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Add stronger description text to the synthetic catalog.")
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--external-catalog", type=Path, action="append", default=[])
    parser.add_argument("--external-mix-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    external_paths = args.external_catalog or DEFAULT_EXTERNAL_PATHS
    enrich_catalog_text(
        catalog_path=args.catalog_path,
        output_path=args.output_path,
        external_catalog_paths=external_paths,
        external_mix_ratio=args.external_mix_ratio,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
