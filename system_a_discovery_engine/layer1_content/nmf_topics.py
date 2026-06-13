"""
NMF Topic Model — Layer 1 Content Understanding
=================================================

Algorithm
---------
1. Load the catalog (item_id, title, description, genres).
2. Combine ``description`` and ``genres`` into a single text field per item.
   Lowercase, tokenise (simple word-boundary split), remove English stopwords
   via NLTK.
3. Build a TF-IDF matrix (max_features=20 000, min_df=5, max_df=0.95).
4. Fit sklearn NMF with K=40 components (init='nndsvda', random_state=42).
5. **Stability check** — also fit K=35 and K=45.  For 200 random items compute
   cosine similarity between the K=40 topic vector and the nearest-matching
   vector from K=35/K=45 after aligning components via the Hungarian algorithm
   (``scipy.optimize.linear_sum_assignment``) on a topic-word cosine similarity
   matrix.  Report mean stability across the two alternative fits.
6. Persist:
   - ``data/processed/topic_vectors.parquet``  — (item_id, tv_0 … tv_39)
   - ``data/processed/tfidf_vectorizer.pkl``   — fitted TF-IDF vectorizer
   - ``data/processed/nmf_model.pkl``          — fitted NMF model

Inputs
------
- ``data/synthetic/catalog.parquet``  (columns: item_id, title, description, genres)

Outputs
-------
- ``data/processed/topic_vectors.parquet``
- ``data/processed/tfidf_vectorizer.pkl``
- ``data/processed/nmf_model.pkl``

Design decisions
----------------
* NNDSVDA initialisation yields deterministic, non-negative starting points —
  critical for reproducibility.
* We cap TF-IDF at 20 000 tokens because the synthetic catalog is modest; for
  production the cap would rise to ~100 k with sub-linear TF.
* Stability is measured by *aligned cosine similarity* rather than simple
  component-index matching, which is meaningless because NMF components are
  unordered.
"""

from __future__ import annotations

import ast
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root & schema imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_store.schema import NMF_TOPICS_DIM  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CATALOG_PATH = PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TOPIC_VECTORS_PATH = OUTPUT_DIR / "topic_vectors.parquet"
TFIDF_PKL_PATH = OUTPUT_DIR / "tfidf_vectorizer.pkl"
NMF_PKL_PATH = OUTPUT_DIR / "nmf_model.pkl"

# ---------------------------------------------------------------------------
# NMF hyper-parameters
# ---------------------------------------------------------------------------
K_MAIN: int = NMF_TOPICS_DIM  # 40
K_LOW: int = 35
K_HIGH: int = 45
TFIDF_MAX_FEATURES: int = 20_000
TFIDF_MIN_DF: int = 5
TFIDF_MAX_DF: float = 0.95
STABILITY_SAMPLE_SIZE: int = 200
RANDOM_STATE: int = 42


# ===================================================================
# Synthetic catalog generator (used when real data is absent)
# ===================================================================

_GENRES = [
    "Fantasy", "Sci-Fi", "Romance", "Thriller", "Mystery",
    "Horror", "Literary Fiction", "Historical Fiction",
    "Young Adult", "Non-Fiction", "Self-Help", "Biography",
]

_LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat Duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat "
    "non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
)


def _generate_synthetic_catalog(n_items: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal synthetic catalog for standalone testing."""
    rng = np.random.default_rng(seed)
    words = _LOREM.split()
    rows: list[dict] = []
    for i in range(n_items):
        n_genres = rng.integers(1, 4)
        genres = rng.choice(_GENRES, size=n_genres, replace=False).tolist()
        # Build a pseudo-description by sampling words + injecting genre tokens
        desc_len = rng.integers(30, 120)
        desc_words = rng.choice(words, size=desc_len, replace=True).tolist()
        # Splice genre tokens in so TF-IDF picks up meaningful signal
        for g in genres:
            pos = rng.integers(0, len(desc_words))
            desc_words.insert(pos, g.lower())
        rows.append({
            "item_id": f"item_{i:05d}",
            "title": f"Title {i}",
            "author_id": f"author_{rng.integers(0, n_items // 5):04d}",
            "author_name": f"Author {rng.integers(0, n_items // 5)}",
            "description": " ".join(desc_words),
            "genres": genres,
            "avg_rating": round(float(rng.uniform(1.0, 5.0)), 2),
            "rating_count": int(rng.integers(0, 500)),
            "review_count": int(rng.integers(0, 200)),
            "page_count": int(rng.integers(50, 800)),
            "chapter_count": int(rng.integers(1, 40)),
            "publish_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 1800))),
        })
    df = pd.DataFrame(rows)
    # Persist so downstream modules can reuse
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CATALOG_PATH, index=False)
    print(f"[nmf_topics] Generated synthetic catalog → {CATALOG_PATH}  ({len(df)} items)")
    return df


# ===================================================================
# Text preprocessing
# ===================================================================

def _load_stopwords() -> set[str]:
    """Return NLTK English stopwords, downloading if needed."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except ImportError:
        return set(ENGLISH_STOP_WORDS)


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


def _latent_topic_tokens(value: object, top_k: int = 5) -> str:
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
        weight = max(1, int(round(float(arr[idx]) * 20)))
        tokens.extend([f"latent_topic_{idx}"] * weight)
    return " ".join(tokens)


def prepare_catalog_text_fields(catalog: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return text descriptions and normalized genres for NMF preprocessing."""
    catalog = catalog.copy()
    if "genres" in catalog.columns:
        genres = catalog["genres"].apply(_normalise_genres)
    else:
        genres = pd.Series([[] for _ in range(len(catalog))], index=catalog.index)

    if "description" in catalog.columns:
        descriptions = catalog["description"].fillna("").astype(str)
    else:
        descriptions = pd.Series(["" for _ in range(len(catalog))], index=catalog.index)

    fallback_parts = []
    for _, row in catalog.iterrows():
        parts = [
            str(row.get("title", "")),
            str(row.get("author_name", "")),
            " ".join(_normalise_genres(row.get("genres", []))),
            _latent_topic_tokens(row.get("topic_vector", "")),
        ]
        fallback_parts.append(" ".join(part for part in parts if part.strip()))

    fallback = pd.Series(fallback_parts, index=catalog.index)
    descriptions = descriptions.where(descriptions.str.strip().ne(""), fallback)
    return descriptions, genres


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")


def preprocess_texts(descriptions: pd.Series, genres: pd.Series) -> list[str]:
    """Combine description + genres, lowercase, tokenise, remove stopwords.

    Parameters
    ----------
    descriptions : pd.Series[str]
    genres : pd.Series[list[str]]

    Returns
    -------
    list[str]  — cleaned text per item (space-joined tokens).
    """
    stopwords = _load_stopwords()
    out: list[str] = []
    for desc, genre_list in tqdm(
        zip(descriptions, genres), total=len(descriptions), desc="Preprocessing text"
    ):
        raw = str(desc).lower()
        if isinstance(genre_list, list):
            raw += " " + " ".join(g.lower() for g in genre_list)
        elif isinstance(genre_list, str):
            raw += " " + genre_list.lower()
        tokens = _TOKEN_RE.findall(raw)
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        out.append(" ".join(tokens))
    return out


# ===================================================================
# NMF pipeline
# ===================================================================

def fit_tfidf(texts: list[str]) -> tuple:
    """Fit TF-IDF vectorizer and return (vectorizer, tfidf_matrix)."""
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    print(f"[nmf_topics] TF-IDF matrix: {X.shape[0]} docs × {X.shape[1]} terms")
    return vec, X


def fit_nmf(tfidf_matrix, n_components: int = K_MAIN, random_state: int = RANDOM_STATE) -> NMF:
    """Fit NMF model on the TF-IDF matrix."""
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    model.fit(tfidf_matrix)
    print(f"[nmf_topics] NMF(K={n_components}) reconstruction error: {model.reconstruction_err_:.2f}")
    return model


def extract_topic_vectors(nmf_model: NMF, tfidf_matrix) -> np.ndarray:
    """Transform documents into topic-proportion space (num_items × K)."""
    W = nmf_model.transform(tfidf_matrix)
    # Row-normalise to proportions (L1)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero
    W = W / row_sums
    return W


# ===================================================================
# Stability analysis
# ===================================================================

def _align_components(H_ref: np.ndarray, H_alt: np.ndarray) -> np.ndarray:
    """Align columns of H_alt to H_ref using the Hungarian algorithm on
    pairwise cosine similarity between topic-word vectors.

    Parameters
    ----------
    H_ref : (K_ref, n_terms) — topic-word matrix for the reference model.
    H_alt : (K_alt, n_terms) — topic-word matrix for the alternative model.

    Returns
    -------
    np.ndarray of shape (min(K_ref, K_alt),) — column indices into H_alt
        that best match each row of H_ref (up to the smaller K).
    """
    sim = cosine_similarity(H_ref, H_alt)  # (K_ref, K_alt)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def stability_score(
    tfidf_matrix,
    nmf_main: NMF,
    k_alt: int,
    n_sample: int = STABILITY_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> float:
    """Compute mean cosine similarity between K=40 topic vectors and their
    Hungarian-aligned counterparts from K=k_alt, over a random sample of items.

    Returns
    -------
    float — mean cosine similarity ∈ [0, 1].
    """
    rng = np.random.default_rng(random_state)
    nmf_alt = fit_nmf(tfidf_matrix, n_components=k_alt, random_state=random_state)

    # Topic-word matrices
    H_main = nmf_main.components_  # (K_main, n_terms)
    H_alt = nmf_alt.components_  # (K_alt, n_terms)

    # Align alternative components to main
    alignment = _align_components(H_main, H_alt)  # indices into H_alt

    # Document-topic matrices
    W_main = nmf_main.transform(tfidf_matrix)  # (n_docs, K_main)
    W_alt = nmf_alt.transform(tfidf_matrix)  # (n_docs, K_alt)

    # Sub-select aligned columns from W_alt to match the ordering of W_main
    # Only use the first min(K_main, K_alt) components
    k_shared = min(K_MAIN, k_alt)
    W_main_aligned = W_main[:, :k_shared]
    W_alt_aligned = W_alt[:, alignment[:k_shared]]

    # Sample items
    n_docs = W_main.shape[0]
    sample_idx = rng.choice(n_docs, size=min(n_sample, n_docs), replace=False)

    sims: list[float] = []
    for idx in sample_idx:
        v_main = W_main_aligned[idx]
        v_alt = W_alt_aligned[idx]
        norm_main = np.linalg.norm(v_main)
        norm_alt = np.linalg.norm(v_alt)
        if norm_main < 1e-12 or norm_alt < 1e-12:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(v_main, v_alt) / (norm_main * norm_alt)))

    return float(np.mean(sims))


# ===================================================================
# Persistence
# ===================================================================

def save_topic_vectors(item_ids: pd.Series, topic_matrix: np.ndarray, path: Path = TOPIC_VECTORS_PATH) -> None:
    """Save topic vectors to parquet: item_id + tv_0 … tv_{K-1}."""
    cols = {f"tv_{i}": topic_matrix[:, i] for i in range(topic_matrix.shape[1])}
    df = pd.DataFrame({"item_id": item_ids.values, **cols})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[nmf_topics] Saved topic vectors → {path}  ({len(df)} rows × {topic_matrix.shape[1]} dims)")


def save_model_artifacts(vectorizer: TfidfVectorizer, nmf_model: NMF) -> None:
    """Pickle the fitted TF-IDF vectorizer and NMF model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_PKL_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(NMF_PKL_PATH, "wb") as f:
        pickle.dump(nmf_model, f)
    print(f"[nmf_topics] Saved TF-IDF vectorizer → {TFIDF_PKL_PATH}")
    print(f"[nmf_topics] Saved NMF model        → {NMF_PKL_PATH}")


# ===================================================================
# Top words per topic (display helper)
# ===================================================================

def top_words_per_topic(
    vectorizer: TfidfVectorizer,
    nmf_model: NMF,
    n_top: int = 10,
    n_topics: Optional[int] = None,
) -> dict[int, list[str]]:
    """Return top-n words for each (or the first n_topics) topic.

    Returns
    -------
    dict  mapping topic_index → list[str] of top words.
    """
    feature_names = vectorizer.get_feature_names_out()
    topics: dict[int, list[str]] = {}
    k = n_topics if n_topics is not None else nmf_model.n_components
    for topic_idx in range(k):
        word_indices = nmf_model.components_[topic_idx].argsort()[::-1][:n_top]
        topics[topic_idx] = [feature_names[i] for i in word_indices]
    return topics


# ===================================================================
# Full pipeline
# ===================================================================

def run_pipeline(catalog_path: Path = CATALOG_PATH) -> dict:
    """Execute the full NMF topic-modelling pipeline.

    Returns
    -------
    dict with keys: 'topic_matrix', 'vectorizer', 'nmf_model',
                    'stability_35', 'stability_45', 'catalog_df'
    """
    # 1. Load catalog --------------------------------------------------
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        print(f"[nmf_topics] Loaded catalog from {catalog_path}  ({len(catalog)} items)")
    else:
        print(f"[nmf_topics] Catalog not found at {catalog_path} — generating synthetic data")
        catalog = _generate_synthetic_catalog()

    # 2. Preprocess ----------------------------------------------------
    descriptions, genres = prepare_catalog_text_fields(catalog)
    texts = preprocess_texts(descriptions, genres)

    # 3. TF-IDF --------------------------------------------------------
    vectorizer, tfidf_matrix = fit_tfidf(texts)

    # 4. NMF (K=40) ----------------------------------------------------
    nmf_model = fit_nmf(tfidf_matrix, n_components=K_MAIN)
    topic_matrix = extract_topic_vectors(nmf_model, tfidf_matrix)

    # 5. Stability analysis --------------------------------------------
    print("\n--- Stability Analysis ---")
    stab_35 = stability_score(tfidf_matrix, nmf_model, k_alt=K_LOW)
    stab_45 = stability_score(tfidf_matrix, nmf_model, k_alt=K_HIGH)
    mean_stab = (stab_35 + stab_45) / 2.0
    print(f"  K=35 stability (mean cosine sim): {stab_35:.4f}")
    print(f"  K=45 stability (mean cosine sim): {stab_45:.4f}")
    print(f"  Mean stability:                   {mean_stab:.4f}")

    # 6. Save ----------------------------------------------------------
    save_topic_vectors(catalog["item_id"], topic_matrix)
    save_model_artifacts(vectorizer, nmf_model)

    return {
        "topic_matrix": topic_matrix,
        "vectorizer": vectorizer,
        "nmf_model": nmf_model,
        "stability_35": stab_35,
        "stability_45": stab_45,
        "catalog_df": catalog,
    }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    results = run_pipeline()

    # Print top 10 words for the first 10 topics
    tw = top_words_per_topic(results["vectorizer"], results["nmf_model"], n_top=10, n_topics=10)
    print("\n" + "=" * 72)
    print("TOP 10 WORDS PER TOPIC (first 10 topics)")
    print("=" * 72)
    for idx, words in tw.items():
        print(f"  Topic {idx:2d}: {', '.join(words)}")

    # Stability summary
    print("\n" + "=" * 72)
    print("STABILITY SUMMARY")
    print("=" * 72)
    print(f"  K=35 vs K=40: {results['stability_35']:.4f}")
    print(f"  K=45 vs K=40: {results['stability_45']:.4f}")
    mean_s = (results["stability_35"] + results["stability_45"]) / 2.0
    print(f"  Mean:          {mean_s:.4f}")
    if mean_s > 0.80:
        print("  ✅ Topics are STABLE across neighbouring K values.")
    else:
        print("  ⚠️  Topics show moderate instability — consider tuning K.")

    print("\n✅ NMF topic pipeline complete.")
