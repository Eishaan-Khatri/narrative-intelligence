"""
Proxy Quality Score & Item Fingerprint — Layer 1 Content Understanding
=======================================================================

Algorithm
---------
1. Compute 12 quality signals per item from session features and catalog
   metadata.  Some signals are directly observed (completion, return rate);
   others are simulated with realistic distributions when ground-truth
   data is unavailable (organic discovery rate, vocabulary richness, etc.).
2. Standardise all 12 signals (z-score per column).
3. Fit PCA, keep components covering ≥70 % of total variance.
4. Composite quality score = Σ(explained_variance_ratio_i × PC_i_score).
5. Validation: build an editorial proxy set (top 5 % by avg_rating × log(1 +
   rating_count)) and compute recall@top-decile of quality_score.
6. Save:
   - ``data/processed/quality_scores.parquet``   (item_id, quality_score, 12 raw signals)
   - ``data/processed/item_fingerprints.parquet`` (item_id, 40 topic + 32 author + 1 quality + 8 structural = 81-dim)

Inputs
------
- ``data/processed/session_features.parquet``
- ``data/synthetic/catalog.parquet``
- ``data/processed/topic_vectors.parquet``    (from nmf_topics.py)
- ``data/processed/author_embeddings.parquet`` (from author_embeddings.py)

Outputs
-------
- ``data/processed/quality_scores.parquet``
- ``data/processed/item_fingerprints.parquet``

Design decisions
----------------
* The PCA composite weights each principal component by its explained variance
  ratio, so the first PC (most informative quality axis) dominates while later
  PCs contribute diminishing refinements.
* Simulated signals (organic_discovery_rate, vocab richness, dialogue_ratio,
  etc.) use genre-stratified distributions to mimic real editorial metadata
  variance.
* The editorial proxy recall metric is a sanity check: a composite quality
  score should surface editorially-valued items (high rating + many ratings)
  into the top decile with reasonable recall (>0.3 is acceptable).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root & schema imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_store.schema import (  # noqa: E402
    AUTHOR_EMBEDDING_DIM,
    ITEM_FINGERPRINT_DIM,
    NMF_TOPICS_DIM,
    STRUCTURAL_FEATURES_DIM,
)
from system_a_discovery_engine.layer1_content.catalog_utils import (  # noqa: E402
    ensure_layer1_catalog_columns,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CATALOG_PATH = PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"
SESSION_PATH = PROJECT_ROOT / "data" / "processed" / "session_features.parquet"
TOPIC_VECTORS_PATH = PROJECT_ROOT / "data" / "processed" / "topic_vectors.parquet"
AUTHOR_EMBED_PATH = PROJECT_ROOT / "data" / "processed" / "author_embeddings.parquet"
QUALITY_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "quality_scores.parquet"
ITEM_FINGERPRINTS_PATH = PROJECT_ROOT / "data" / "processed" / "item_fingerprints.parquet"

# ---------------------------------------------------------------------------
# Signal names (the 12 raw quality proxies)
# ---------------------------------------------------------------------------
SIGNAL_NAMES: list[str] = [
    "chapter_completion_rate",
    "first_chapter_drop_rate",
    "return_rate",
    "re_read_depth",
    "comment_read_ratio",
    "sentiment_proxy",
    "session_depth",
    "organic_discovery_rate",
    "author_update_consistency",
    "new_reader_acquisition",
    "vocabulary_richness",
    "dialogue_ratio",
]

PCA_VARIANCE_THRESHOLD: float = 0.70


# ===================================================================
# Synthetic data helpers
# ===================================================================

def _ensure_catalog(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    from system_a_discovery_engine.layer1_content.nmf_topics import _generate_synthetic_catalog
    return _generate_synthetic_catalog()


def _ensure_sessions(path: Path, catalog: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    from system_a_discovery_engine.layer1_content.author_embeddings import (
        _generate_synthetic_session_features,
    )
    return _generate_synthetic_session_features(catalog)


# ===================================================================
# VADER sentiment helper
# ===================================================================

def _vader_compound(texts: list[str]) -> list[float]:
    """Return VADER compound sentiment score for each text."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        # Fallback: random sentiment in [-1, 1] if VADER unavailable
        rng = np.random.default_rng(42)
        return rng.uniform(-0.5, 0.9, size=len(texts)).tolist()
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def _generate_synthetic_reviews(
    item_ids: np.ndarray, avg_ratings: np.ndarray, seed: int = 42
) -> dict[str, str]:
    """Generate short synthetic review text per item, conditioned on rating."""
    rng = np.random.default_rng(seed)
    positive_phrases = [
        "Loved every page, couldn't put it down.",
        "Beautifully written and deeply moving.",
        "An absolute masterpiece of storytelling.",
        "Characters are vivid and the plot is gripping.",
        "A wonderful read that stays with you.",
    ]
    neutral_phrases = [
        "It was okay, had some good moments.",
        "Decent read but not unforgettable.",
        "Some parts were interesting, others dragged.",
        "Average book, nothing special.",
        "Readable but formulaic.",
    ]
    negative_phrases = [
        "Struggled to finish, very slow pacing.",
        "Disappointing and predictable.",
        "Not my cup of tea, poorly developed characters.",
        "Hard to follow and overly convoluted.",
        "Would not recommend to others.",
    ]
    reviews: dict[str, str] = {}
    for iid, rating in zip(item_ids, avg_ratings):
        if rating >= 3.5:
            reviews[iid] = rng.choice(positive_phrases)
        elif rating >= 2.5:
            reviews[iid] = rng.choice(neutral_phrases)
        else:
            reviews[iid] = rng.choice(negative_phrases)
    return reviews


# ===================================================================
# Per-item quality signal computation
# ===================================================================

def compute_quality_signals(
    catalog: pd.DataFrame, sessions: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Compute 12 quality proxy signals per item.

    Parameters
    ----------
    catalog  : catalog DataFrame (item_id, avg_rating, rating_count, review_count, genres, …)
    sessions : session features DataFrame (session_id, user_id, item_id, …)

    Returns
    -------
    DataFrame with columns [item_id] + SIGNAL_NAMES (12 signals).
    """
    rng = np.random.default_rng(seed)
    item_ids = catalog["item_id"].unique()
    n_items = len(item_ids)

    # Pre-aggregate session-level stats per item
    sess_item = sessions.groupby("item_id").agg(
        mean_completion=("final_completion_pct", "mean"),
        total_sessions=("session_id", "nunique"),
        mean_velocity=("reading_velocity_wpm", "mean"),
        mean_re_read=("re_read_ratio", "mean"),
        mean_duration=("session_duration_sec", "mean"),
        n_users=("user_id", "nunique"),
    ).reset_index()

    # --- Signal 1: chapter_completion_rate ---
    # mean final_completion_pct
    sig_completion = sess_item.set_index("item_id")["mean_completion"].reindex(item_ids, fill_value=0.5)

    # --- Signal 2: first_chapter_drop_rate ---
    first_chap = sessions[sessions["chapter_index"] == 0].copy()
    if "exit_reason" in first_chap.columns:
        first_chap["dropped"] = (
            (first_chap["exit_reason"] == "mid_chapter") &
            (first_chap["final_completion_pct"] < 0.3)
        )
        drop_rate = first_chap.groupby("item_id")["dropped"].mean()
    else:
        drop_rate = pd.Series(0.2, index=item_ids)
    sig_drop = drop_rate.reindex(item_ids, fill_value=0.2)

    # --- Signal 3: return_rate (% of users with >1 session, gap > 1 day) ---
    if "timestamp_start" in sessions.columns:
        sessions["timestamp_start"] = pd.to_datetime(sessions["timestamp_start"])
        user_item_sessions = (
            sessions.groupby(["item_id", "user_id"])
            .agg(n_sess=("session_id", "nunique"),
                 first_ts=("timestamp_start", "min"),
                 last_ts=("timestamp_start", "max"))
            .reset_index()
        )
        user_item_sessions["gap_days"] = (
            (user_item_sessions["last_ts"] - user_item_sessions["first_ts"]).dt.total_seconds() / 86400.0
        )
        user_item_sessions["returned"] = (user_item_sessions["n_sess"] > 1) & (user_item_sessions["gap_days"] > 1)
        return_rate = user_item_sessions.groupby("item_id")["returned"].mean()
    else:
        return_rate = pd.Series(0.3, index=item_ids)
    sig_return = return_rate.reindex(item_ids, fill_value=0.3)

    # --- Signal 4: re_read_depth (genre-baseline adjusted) ---
    # Merge genres and compute per-genre baseline
    sessions_with_genre = sessions.merge(
        catalog[["item_id", "genres"]], on="item_id", how="left"
    )
    sessions_with_genre["primary_genre"] = sessions_with_genre["genres"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    genre_baseline = sessions_with_genre.groupby("primary_genre")["re_read_ratio"].mean()
    item_genre = catalog.set_index("item_id")["genres"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
    )
    item_reread = sess_item.set_index("item_id")["mean_re_read"].reindex(item_ids, fill_value=0.0)
    item_genre_aligned = item_genre.reindex(item_ids, fill_value="Unknown")
    genre_base_aligned = item_genre_aligned.map(genre_baseline).fillna(genre_baseline.mean())
    sig_reread = (item_reread - genre_base_aligned).fillna(0.0)

    # --- Signal 5: comment_read_ratio ---
    review_counts = catalog.set_index("item_id")["review_count"].reindex(item_ids, fill_value=0)
    total_sess = sess_item.set_index("item_id")["total_sessions"].reindex(item_ids, fill_value=1)
    sig_comment = (review_counts / total_sess.clip(lower=1)).fillna(0.0)

    # --- Signal 6: sentiment_proxy (VADER on synthetic reviews) ---
    avg_ratings = catalog.set_index("item_id")["avg_rating"].reindex(item_ids, fill_value=3.0)
    reviews = _generate_synthetic_reviews(item_ids, avg_ratings.values, seed=seed)
    review_texts = [reviews.get(iid, "Average read.") for iid in item_ids]
    sentiments = _vader_compound(review_texts)
    sig_sentiment = pd.Series(sentiments, index=item_ids)

    # --- Signal 7: session_depth (mean chapters read per user) ---
    if "chapter_index" in sessions.columns:
        chapters_per_user = (
            sessions.groupby(["item_id", "user_id"])["chapter_index"]
            .nunique()
            .reset_index(name="chapters_read")
        )
        sess_depth = chapters_per_user.groupby("item_id")["chapters_read"].mean()
    else:
        sess_depth = pd.Series(3.0, index=item_ids)
    sig_depth = sess_depth.reindex(item_ids, fill_value=1.0)

    # --- Signal 8: organic_discovery_rate (simulated 30-50 %) ---
    sig_organic = pd.Series(rng.uniform(0.30, 0.50, size=n_items), index=item_ids)

    # --- Signal 9: author_update_consistency (std dev of chapter publish intervals) ---
    # Simulated: lower std → more consistent → higher quality signal
    sig_consistency = pd.Series(rng.exponential(5.0, size=n_items), index=item_ids)

    # --- Signal 10: new_reader_acquisition ---
    # % of readers who haven't read this author before
    # Simulate using session + catalog
    author_map = catalog.set_index("item_id")["author_id"]
    sessions_with_author = sessions.merge(
        catalog[["item_id", "author_id"]], on="item_id", how="left"
    )
    # Track first-seen author per user (chronological)
    if "timestamp_start" in sessions_with_author.columns:
        sessions_with_author = sessions_with_author.sort_values("timestamp_start")
    user_author_first = (
        sessions_with_author
        .drop_duplicates(subset=["user_id", "author_id"], keep="first")
        .rename(columns={"item_id": "first_item_for_author"})
    )
    # For each (item, user), check if this item is their first by this author
    sess_merged = sessions.merge(catalog[["item_id", "author_id"]], on="item_id", how="left")
    sess_merged = sess_merged.merge(
        user_author_first[["user_id", "author_id", "first_item_for_author"]],
        on=["user_id", "author_id"],
        how="left",
    )
    sess_merged["is_new_reader"] = sess_merged["item_id"] == sess_merged["first_item_for_author"]
    new_reader_rate = sess_merged.groupby("item_id")["is_new_reader"].mean()
    sig_new_reader = new_reader_rate.reindex(item_ids, fill_value=0.5)

    # --- Signal 11: vocabulary_richness (simulated TTR, genre-level median) ---
    genre_ttr_medians = {
        "Fantasy": 0.55, "Sci-Fi": 0.58, "Romance": 0.42, "Thriller": 0.45,
        "Mystery": 0.48, "Horror": 0.46, "Literary Fiction": 0.65,
        "Historical Fiction": 0.60, "Young Adult": 0.40, "Non-Fiction": 0.62,
        "Self-Help": 0.50, "Biography": 0.56,
    }
    ttr_values = []
    for iid in item_ids:
        genre = item_genre_aligned.get(iid, "Unknown")
        median_ttr = genre_ttr_medians.get(genre, 0.50)
        ttr_values.append(float(np.clip(rng.normal(median_ttr, 0.08), 0.3, 0.7)))
    sig_vocab = pd.Series(ttr_values, index=item_ids)

    # --- Signal 12: dialogue_ratio (simulated, genre-level median) ---
    genre_dialogue_medians = {
        "Fantasy": 0.30, "Sci-Fi": 0.25, "Romance": 0.40, "Thriller": 0.35,
        "Mystery": 0.38, "Horror": 0.28, "Literary Fiction": 0.32,
        "Historical Fiction": 0.27, "Young Adult": 0.42, "Non-Fiction": 0.12,
        "Self-Help": 0.10, "Biography": 0.18,
    }
    dial_values = []
    for iid in item_ids:
        genre = item_genre_aligned.get(iid, "Unknown")
        median_dial = genre_dialogue_medians.get(genre, 0.30)
        dial_values.append(float(np.clip(rng.normal(median_dial, 0.06), 0.1, 0.5)))
    sig_dialogue = pd.Series(dial_values, index=item_ids)

    # Assemble all 12 signals into a DataFrame
    quality_df = pd.DataFrame({
        "item_id": item_ids,
        "chapter_completion_rate": sig_completion.values,
        "first_chapter_drop_rate": sig_drop.values,
        "return_rate": sig_return.values,
        "re_read_depth": sig_reread.values,
        "comment_read_ratio": sig_comment.values,
        "sentiment_proxy": sig_sentiment.values,
        "session_depth": sig_depth.values,
        "organic_discovery_rate": sig_organic.values,
        "author_update_consistency": sig_consistency.values,
        "new_reader_acquisition": sig_new_reader.values,
        "vocabulary_richness": sig_vocab.values,
        "dialogue_ratio": sig_dialogue.values,
    })

    print(f"[quality_pca] Computed 12 quality signals for {len(quality_df)} items")
    return quality_df


# ===================================================================
# PCA composite quality score
# ===================================================================

def compute_composite_quality(
    quality_df: pd.DataFrame,
) -> tuple[pd.DataFrame, PCA, StandardScaler, float]:
    """Standardise 12 signals, PCA, and compute composite quality score.

    Returns
    -------
    quality_df   : input df augmented with 'quality_score'
    pca          : fitted PCA
    scaler       : fitted StandardScaler
    n_components : number of PCs retained
    """
    X = quality_df[SIGNAL_NAMES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA first to inspect variance
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1)
    n_components = min(n_components, X_scaled.shape[1])

    # Refit with selected n_components
    pca = PCA(n_components=n_components, random_state=42)
    PC_scores = pca.fit_transform(X_scaled)  # (n_items, n_components)

    # Composite = Σ(var_ratio_i × PC_i)
    composite = (PC_scores * pca.explained_variance_ratio_).sum(axis=1)

    quality_df = quality_df.copy()
    quality_df["quality_score"] = composite

    print(f"[quality_pca] PCA retained {n_components} components "
          f"(cumulative variance = {cumvar[n_components - 1]:.4f})")
    print(f"[quality_pca] Explained variance ratios: "
          f"{[f'{v:.4f}' for v in pca.explained_variance_ratio_]}")

    return quality_df, pca, scaler, n_components


# ===================================================================
# Editorial proxy validation
# ===================================================================

def validate_editorial_proxy(
    quality_df: pd.DataFrame, catalog: pd.DataFrame
) -> float:
    """Compute recall of editorial-proxy items in the top decile of quality_score.

    Editorial proxy = top 5 % by avg_rating × log(1 + rating_count).
    """
    cat = catalog[["item_id", "avg_rating", "rating_count"]].copy()
    cat["editorial_score"] = cat["avg_rating"] * np.log1p(cat["rating_count"])
    threshold = cat["editorial_score"].quantile(0.95)
    editorial_set = set(cat.loc[cat["editorial_score"] >= threshold, "item_id"])

    # Top decile by composite quality
    q90 = quality_df["quality_score"].quantile(0.90)
    top_decile = set(quality_df.loc[quality_df["quality_score"] >= q90, "item_id"])

    if len(editorial_set) == 0:
        return 0.0

    recall = len(editorial_set & top_decile) / len(editorial_set)
    print(f"[quality_pca] Editorial proxy recall@top-decile: {recall:.4f} "
          f"({len(editorial_set & top_decile)}/{len(editorial_set)} editorial items found)")
    return recall


# ===================================================================
# Structural features (8-dim)
# ===================================================================

def compute_structural_features(
    catalog: pd.DataFrame, sessions: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Compute 8 structural features per item.

    Features
    --------
    0. log_page_count
    1. log_chapter_count
    2. avg_chapter_length (pages / chapters)
    3. series_flag (1 if in a series, 0 otherwise)
    4. series_position_normalised
    5. mean_session_duration (from sessions)
    6. device_mobile_fraction
    7. recency (days since publish_date, log-transformed)
    """
    rng = np.random.default_rng(seed)
    item_ids = catalog["item_id"].values

    page_count = catalog.set_index("item_id")["page_count"].reindex(item_ids, fill_value=200).values.astype(float)
    chapter_count = catalog.set_index("item_id")["chapter_count"].reindex(item_ids, fill_value=10).values.astype(float)

    sf0 = np.log1p(page_count)
    sf1 = np.log1p(chapter_count)
    sf2 = page_count / np.clip(chapter_count, 1, None)

    series_id = catalog.set_index("item_id").get("series_id", pd.Series(index=item_ids))
    sf3 = (~series_id.reindex(item_ids).isna()).astype(float).values
    series_pos = catalog.set_index("item_id").get("series_position", pd.Series(index=item_ids))
    sf4 = series_pos.reindex(item_ids).fillna(0).astype(float).values / 10.0

    # Mean session duration per item from sessions
    sess_dur = sessions.groupby("item_id")["session_duration_sec"].mean()
    sf5 = sess_dur.reindex(item_ids, fill_value=300.0).values.astype(float)
    sf5 = np.log1p(sf5)

    # Device mobile fraction
    if "device_type" in sessions.columns:
        mobile_frac = sessions.groupby("item_id")["device_type"].apply(
            lambda x: (x == "mobile").mean()
        )
    else:
        mobile_frac = pd.Series(0.6, index=item_ids)
    sf6 = mobile_frac.reindex(item_ids, fill_value=0.6).values.astype(float)

    # Recency
    if "publish_date" in catalog.columns:
        pub_dates = pd.to_datetime(catalog.set_index("item_id")["publish_date"]).reindex(item_ids)
        ref = pub_dates.max() + pd.Timedelta(days=30)
        days_since = (ref - pub_dates).dt.total_seconds() / 86400.0
        sf7 = np.log1p(days_since.fillna(365.0).values)
    else:
        sf7 = np.log1p(rng.uniform(30, 1500, size=len(item_ids)))

    structural = np.column_stack([sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7])

    # Normalise each feature to [0, 1]
    for col_idx in range(structural.shape[1]):
        col_min = structural[:, col_idx].min()
        col_max = structural[:, col_idx].max()
        if col_max - col_min > 1e-12:
            structural[:, col_idx] = (structural[:, col_idx] - col_min) / (col_max - col_min)

    cols = {f"sf_{i}": structural[:, i] for i in range(STRUCTURAL_FEATURES_DIM)}
    df = pd.DataFrame({"item_id": item_ids, **cols})
    print(f"[quality_pca] Computed {STRUCTURAL_FEATURES_DIM} structural features for {len(df)} items")
    return df


# ===================================================================
# Item fingerprint assembly (81-dim)
# ===================================================================

def assemble_item_fingerprints(
    item_ids: np.ndarray,
    topic_vectors: Optional[pd.DataFrame],
    author_embeddings: Optional[pd.DataFrame],
    quality_scores: pd.DataFrame,
    structural_features: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full 81-dim item fingerprint table.

    Fingerprint = topic_vector(40) + author_embedding(32) + quality_score(1) + structural(8)
    """
    n = len(item_ids)
    tv_cols = [f"tv_{i}" for i in range(NMF_TOPICS_DIM)]
    ae_cols = [f"ae_{i}" for i in range(AUTHOR_EMBEDDING_DIM)]
    sf_cols = [f"sf_{i}" for i in range(STRUCTURAL_FEATURES_DIM)]

    # --- Topic vectors ---
    if topic_vectors is not None and len(topic_vectors) > 0:
        tv_source_cols = [c for c in topic_vectors.columns if c.startswith("tv_")]
        tv_indexed = topic_vectors.set_index("item_id")[tv_source_cols].reindex(item_ids, fill_value=0.0)
        tv_matrix = tv_indexed.values
    else:
        tv_matrix = np.zeros((n, NMF_TOPICS_DIM))
        print("[quality_pca] ⚠ Topic vectors not found — using zeros")

    # --- Author embeddings ---
    if author_embeddings is not None and len(author_embeddings) > 0:
        emb_source_cols = [c for c in author_embeddings.columns if c.startswith("emb_")]
        # Map item_id → author_id → embedding
        author_map = catalog.set_index("item_id")["author_id"].reindex(item_ids)
        ae_indexed = author_embeddings.set_index("author_id")[emb_source_cols]
        ae_matrix = ae_indexed.reindex(author_map.values, fill_value=0.0).values
    else:
        ae_matrix = np.zeros((n, AUTHOR_EMBEDDING_DIM))
        print("[quality_pca] ⚠ Author embeddings not found — using zeros")

    # Ensure correct column count
    if ae_matrix.shape[1] < AUTHOR_EMBEDDING_DIM:
        pad = np.zeros((ae_matrix.shape[0], AUTHOR_EMBEDDING_DIM - ae_matrix.shape[1]))
        ae_matrix = np.hstack([ae_matrix, pad])

    # --- Quality score ---
    qs_indexed = quality_scores.set_index("item_id")["quality_score"].reindex(item_ids, fill_value=0.0)
    qs_matrix = qs_indexed.values.reshape(-1, 1)

    # --- Structural features ---
    sf_indexed = structural_features.set_index("item_id")[sf_cols].reindex(item_ids, fill_value=0.0)
    sf_matrix = sf_indexed.values

    # Concatenate → 81 dims
    fingerprint = np.hstack([tv_matrix, ae_matrix, qs_matrix, sf_matrix])
    assert fingerprint.shape[1] == ITEM_FINGERPRINT_DIM, (
        f"Expected {ITEM_FINGERPRINT_DIM}-dim fingerprint, got {fingerprint.shape[1]}"
    )

    # Build DataFrame
    fp_cols = tv_cols + ae_cols + ["quality_score"] + sf_cols
    fp_df = pd.DataFrame(fingerprint, columns=fp_cols)
    fp_df.insert(0, "item_id", item_ids)

    print(f"[quality_pca] Assembled {ITEM_FINGERPRINT_DIM}-dim fingerprints for {len(fp_df)} items")
    return fp_df


# ===================================================================
# Full pipeline
# ===================================================================

def run_pipeline(
    catalog_path: Path = CATALOG_PATH,
    session_path: Path = SESSION_PATH,
) -> dict:
    """Execute quality score + item fingerprint pipeline end-to-end.

    Returns
    -------
    dict with keys: 'quality_df', 'fingerprints_df', 'pca', 'recall', 'n_components'
    """
    # 1. Load data
    catalog = _ensure_catalog(catalog_path)
    sessions = _ensure_sessions(session_path, catalog)

    catalog = ensure_layer1_catalog_columns(catalog)

    # 2. Compute 12 quality signals
    quality_df = compute_quality_signals(catalog, sessions)

    # 3. PCA composite score
    quality_df, pca, scaler, n_components = compute_composite_quality(quality_df)

    # 4. Editorial proxy validation
    recall = validate_editorial_proxy(quality_df, catalog)

    # 5. Save quality scores
    QUALITY_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_parquet(QUALITY_SCORES_PATH, index=False)
    print(f"[quality_pca] Saved quality scores → {QUALITY_SCORES_PATH}")

    # 6. Structural features
    structural_df = compute_structural_features(catalog, sessions)

    # 7. Load upstream artefacts for fingerprint assembly
    topic_vectors = None
    if TOPIC_VECTORS_PATH.exists():
        topic_vectors = pd.read_parquet(TOPIC_VECTORS_PATH)
        print(f"[quality_pca] Loaded topic vectors ({len(topic_vectors)} rows)")
    else:
        print("[quality_pca] ⚠ topic_vectors.parquet not found — running NMF pipeline first")
        try:
            from system_a_discovery_engine.layer1_content.nmf_topics import run_pipeline as run_nmf
            nmf_results = run_nmf()
            topic_vectors = pd.read_parquet(TOPIC_VECTORS_PATH) if TOPIC_VECTORS_PATH.exists() else None
        except Exception as e:
            print(f"[quality_pca] NMF pipeline failed: {e} — using zero topic vectors")

    author_embeddings = None
    if AUTHOR_EMBED_PATH.exists():
        author_embeddings = pd.read_parquet(AUTHOR_EMBED_PATH)
        print(f"[quality_pca] Loaded author embeddings ({len(author_embeddings)} rows)")
    else:
        print("[quality_pca] ⚠ author_embeddings.parquet not found — running author pipeline first")
        try:
            from system_a_discovery_engine.layer1_content.author_embeddings import (
                run_pipeline as run_author,
            )
            author_embeddings = run_author()
        except Exception as e:
            print(f"[quality_pca] Author pipeline failed: {e} — using zero author embeddings")

    # 8. Assemble item fingerprints
    item_ids = catalog["item_id"].values
    fp_df = assemble_item_fingerprints(
        item_ids, topic_vectors, author_embeddings, quality_df, structural_df, catalog
    )
    fp_df.to_parquet(ITEM_FINGERPRINTS_PATH, index=False)
    print(f"[quality_pca] Saved item fingerprints → {ITEM_FINGERPRINTS_PATH}")

    return {
        "quality_df": quality_df,
        "fingerprints_df": fp_df,
        "pca": pca,
        "recall": recall,
        "n_components": n_components,
    }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    results = run_pipeline()
    qdf = results["quality_df"]
    pca = results["pca"]

    print("\n" + "=" * 72)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 72)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        print(f"  PC{i + 1}: {var:.4f}  (cumulative: {cum:.4f})")

    print("\n" + "=" * 72)
    print("TOP 5 ITEMS BY QUALITY SCORE")
    print("=" * 72)
    top5 = qdf.nlargest(5, "quality_score")
    for _, row in top5.iterrows():
        print(f"  {row['item_id']}: quality_score = {row['quality_score']:.4f}")

    print("\n" + "=" * 72)
    print("BOTTOM 5 ITEMS BY QUALITY SCORE")
    print("=" * 72)
    bot5 = qdf.nsmallest(5, "quality_score")
    for _, row in bot5.iterrows():
        print(f"  {row['item_id']}: quality_score = {row['quality_score']:.4f}")

    print("\n" + "=" * 72)
    print("EDITORIAL PROXY RECALL")
    print("=" * 72)
    print(f"  Recall@top-decile: {results['recall']:.4f}")
    if results["recall"] >= 0.3:
        print("  ✅ Quality score shows meaningful editorial alignment.")
    else:
        print("  ⚠️  Quality score has weak editorial alignment — consider signal refinement.")

    # Fingerprint summary
    fp_df = results["fingerprints_df"]
    fp_numeric = fp_df.drop(columns=["item_id"]).values
    print(f"\n  Item fingerprint shape:  {fp_numeric.shape}")
    print(f"  Fingerprint L2 norms — mean: {np.linalg.norm(fp_numeric, axis=1).mean():.4f}, "
          f"std: {np.linalg.norm(fp_numeric, axis=1).std():.4f}")

    print("\n✅ Quality score + item fingerprint pipeline complete.")
