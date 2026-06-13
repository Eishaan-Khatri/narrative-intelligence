"""
Author Embeddings — Layer 1 Content Understanding
===================================================

Algorithm
---------
1. Load the catalog (item metadata, including author_id and publish_date) and
   session features (per-session reading telemetry).
2. For each author with ≥2 works, build an *interaction profile* per work:
   [avg_rating, completion_rate_proxy, avg_velocity_wpm, avg_re_read_ratio,
    return_rate, session_count, avg_session_duration, genre_diversity_of_readers].
3. Apply exponential time decay to each work's profile:
       weight = exp(-λ · (reference_date − publish_date))
       λ = ln(2) / 730   →  half-life ≈ 2 years
4. Compute the weighted average of a given author's per-work profiles.
5. Stack all multi-work author profiles and project to 32 dims via PCA.
6. For **first-time authors** (only 1 work), find the genre-cluster centroid
   among authors who share the same dominant NMF topic, and assign that
   centroid as the cold-start embedding.

Inputs
------
- ``data/synthetic/catalog.parquet``
- ``data/processed/session_features.parquet``
- (optional) ``data/processed/topic_vectors.parquet``  — for cold-start matching

Outputs
-------
- ``data/processed/author_embeddings.parquet``  (author_id, emb_0 … emb_31)

Design decisions
----------------
* Time-decay biases toward recent performance while keeping legacy signal.
* PCA is preferred over autoencoders at this scale (<10 k authors) because it
  is deterministic, invertible, and trivially debuggable.
* Cold-start via genre-cluster centroid leverages the community structure in
  the NMF topic space without requiring any interaction data for the new author.
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

from feature_store.schema import AUTHOR_EMBEDDING_DIM, NMF_TOPICS_DIM  # noqa: E402
from system_a_discovery_engine.layer1_content.catalog_utils import (  # noqa: E402
    ensure_layer1_catalog_columns,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CATALOG_PATH = PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"
SESSION_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "session_features.parquet"
TOPIC_VECTORS_PATH = PROJECT_ROOT / "data" / "processed" / "topic_vectors.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "author_embeddings.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTERACTION_PROFILE_DIM: int = 8
HALFLIFE_DAYS: float = 730.0  # 2 years
LAMBDA_DECAY: float = np.log(2) / HALFLIFE_DAYS


# ===================================================================
# Synthetic data generators (fallback when real data is absent)
# ===================================================================

def _generate_synthetic_session_features(
    catalog: pd.DataFrame, n_sessions: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """Create minimal synthetic session features for standalone testing."""
    rng = np.random.default_rng(seed)
    item_ids = catalog["item_id"].values
    rows: list[dict] = []
    user_pool = [f"user_{i:05d}" for i in range(300)]
    for i in range(n_sessions):
        item_id = rng.choice(item_ids)
        cat_row = catalog[catalog["item_id"] == item_id].iloc[0]
        chapter_idx = int(rng.integers(0, max(1, cat_row.get("chapter_count", 5))))
        dur = float(rng.exponential(600))
        vel = float(np.clip(rng.normal(250, 80), 50, 600))
        comp = float(np.clip(rng.beta(3, 2), 0, 1))
        reread = float(np.clip(rng.exponential(0.1), 0, 1))
        exit_reasons = ["chapter_end", "mid_chapter", "app_close"]
        ts_start = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=int(rng.integers(0, 15000)))
        rows.append({
            "session_id": f"sess_{i:06d}",
            "user_id": rng.choice(user_pool),
            "item_id": item_id,
            "chapter_index": chapter_idx,
            "session_duration_sec": dur,
            "reading_velocity_wpm": vel,
            "velocity_acceleration": float(rng.normal(0, 10)),
            "completion_curve": [float(np.clip(rng.beta(2 + j, 3), 0, 1)) for j in range(5)],
            "completion_curve_shape": rng.choice(["cliff", "decay", "steady", "abandon_early"]),
            "re_read_ratio": reread,
            "final_completion_pct": comp,
            "exit_reason": rng.choice(exit_reasons),
            "device_type": rng.choice(["mobile", "desktop", "tablet"]),
            "timestamp_start": ts_start,
            "timestamp_end": ts_start + pd.Timedelta(seconds=dur),
            "rating": float(rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.05, 0.1, 0.15, 0.2, 0.2])),
        })
    df = pd.DataFrame(rows)
    SESSION_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SESSION_FEATURES_PATH, index=False)
    print(f"[author_embed] Generated synthetic session features → {SESSION_FEATURES_PATH}  ({len(df)} rows)")
    return df


# ===================================================================
# Interaction profile computation
# ===================================================================

def compute_interaction_profiles(
    catalog: pd.DataFrame, sessions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute an 8-dim interaction profile per (author_id, item_id).

    Returns
    -------
    profiles_df : DataFrame  with columns [author_id, item_id, publish_date, ip_0..ip_7]
    author_work_counts : DataFrame  with columns [author_id, n_works]
    """
    # Count works per author
    author_works = catalog.groupby("author_id")["item_id"].nunique().reset_index(name="n_works")

    # Merge sessions with catalog to get author_id and genres
    merged = sessions.merge(catalog[["item_id", "author_id", "genres", "avg_rating", "publish_date"]],
                            on="item_id", how="left")
    merged = merged.dropna(subset=["author_id"])

    # -- Per-(item) aggregated signals --
    item_agg = merged.groupby("item_id").agg(
        author_id=("author_id", "first"),
        avg_rating=("avg_rating", "first"),
        completion_rate_proxy=("final_completion_pct", "mean"),
        avg_velocity_wpm=("reading_velocity_wpm", "mean"),
        avg_re_read_ratio=("re_read_ratio", "mean"),
        session_count=("session_id", "nunique"),
        avg_session_duration=("session_duration_sec", "mean"),
        n_unique_users=("user_id", "nunique"),
        publish_date=("publish_date", "first"),
    ).reset_index()

    # return_rate: fraction of users with >1 session on this item
    user_item_sess_counts = merged.groupby(["item_id", "user_id"])["session_id"].nunique().reset_index(name="sess_cnt")
    return_rates = (
        user_item_sess_counts.groupby("item_id")
        .apply(lambda g: (g["sess_cnt"] > 1).mean(), include_groups=False)
        .reset_index(name="return_rate")
    )
    item_agg = item_agg.merge(return_rates, on="item_id", how="left")
    item_agg["return_rate"] = item_agg["return_rate"].fillna(0.0)

    # genre_diversity_of_readers: use the number of unique genres read by
    # users who read this item as a proxy (normalised)
    def _reader_genre_diversity(item_id: str) -> float:
        users = merged.loc[merged["item_id"] == item_id, "user_id"].unique()
        if len(users) == 0:
            return 0.0
        genres_read = merged.loc[merged["user_id"].isin(users), "genres"]
        flat = []
        for g in genres_read:
            if isinstance(g, list):
                flat.extend(g)
            elif isinstance(g, str):
                flat.append(g)
        return min(len(set(flat)) / max(len(_ALL_GENRES_CACHE), 1), 1.0)

    # Build genre cache once
    global _ALL_GENRES_CACHE
    all_genres = []
    for g in catalog["genres"]:
        if isinstance(g, list):
            all_genres.extend(g)
        elif isinstance(g, str):
            all_genres.append(g)
    _ALL_GENRES_CACHE = list(set(all_genres))

    # Vectorised approximation for genre diversity: count unique genres
    # across all sessions of users who interacted with each item
    # (full per-item loop is acceptable for <50 k items)
    print("[author_embed] Computing genre diversity of readers per item …")
    genre_div = {}
    for iid in tqdm(item_agg["item_id"].unique(), desc="Genre diversity"):
        genre_div[iid] = _reader_genre_diversity(iid)
    item_agg["genre_diversity_of_readers"] = item_agg["item_id"].map(genre_div)

    # Build the 8-dim profile
    profile_cols = [
        "avg_rating", "completion_rate_proxy", "avg_velocity_wpm",
        "avg_re_read_ratio", "return_rate", "session_count",
        "avg_session_duration", "genre_diversity_of_readers",
    ]
    for col in profile_cols:
        item_agg[col] = item_agg[col].fillna(0.0).astype(float)

    # Rename to ip_0 .. ip_7
    profiles_df = item_agg[["author_id", "item_id", "publish_date"] + profile_cols].copy()
    for i, col in enumerate(profile_cols):
        profiles_df = profiles_df.rename(columns={col: f"ip_{i}"})

    return profiles_df, author_works


_ALL_GENRES_CACHE: list[str] = []


# ===================================================================
# Time-decayed weighted average → PCA projection
# ===================================================================

def compute_author_embeddings(
    profiles_df: pd.DataFrame,
    author_works: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """Compute 32-dim author embeddings via time-decayed weighted PCA.

    Parameters
    ----------
    profiles_df : DataFrame with [author_id, item_id, publish_date, ip_0..ip_7]
    author_works : DataFrame with [author_id, n_works]
    reference_date : anchor for time decay (defaults to max publish_date + 30d)

    Returns
    -------
    embeddings_df : (author_id, emb_0 .. emb_31) for multi-work authors
    pca : fitted PCA object
    scaler : fitted StandardScaler
    """
    multi_work_authors = author_works.loc[author_works["n_works"] >= 2, "author_id"].values
    profiles_multi = profiles_df[profiles_df["author_id"].isin(multi_work_authors)].copy()

    if reference_date is None:
        max_date = pd.to_datetime(profiles_multi["publish_date"]).max()
        reference_date = max_date + pd.Timedelta(days=30) if pd.notna(max_date) else pd.Timestamp.now()

    ip_cols = [f"ip_{i}" for i in range(INTERACTION_PROFILE_DIM)]

    # Compute time-decay weight per work
    profiles_multi["publish_date"] = pd.to_datetime(profiles_multi["publish_date"])
    profiles_multi["days_ago"] = (reference_date - profiles_multi["publish_date"]).dt.total_seconds() / 86400.0
    profiles_multi["days_ago"] = profiles_multi["days_ago"].clip(lower=0)
    profiles_multi["weight"] = np.exp(-LAMBDA_DECAY * profiles_multi["days_ago"].values)

    # Weighted average per author
    def _weighted_avg(group: pd.DataFrame) -> pd.Series:
        weights = group["weight"].values
        if weights.sum() < 1e-12:
            weights = np.ones_like(weights)
        w_norm = weights / weights.sum()
        vals = group[ip_cols].values  # (n_works, 8)
        avg = (vals * w_norm[:, None]).sum(axis=0)
        return pd.Series(avg, index=ip_cols)

    print("[author_embed] Computing weighted author profiles …")
    author_profiles = (
        profiles_multi
        .groupby("author_id")
        .apply(_weighted_avg, include_groups=False)
        .reset_index()
    )

    # Standardise + PCA to 32 dims
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(author_profiles[ip_cols].values)

    # PCA dimension: min(AUTHOR_EMBEDDING_DIM, n_features, n_samples)
    n_components = min(AUTHOR_EMBEDDING_DIM, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # If PCA produced fewer than 32 dims, zero-pad
    if X_pca.shape[1] < AUTHOR_EMBEDDING_DIM:
        pad = np.zeros((X_pca.shape[0], AUTHOR_EMBEDDING_DIM - X_pca.shape[1]))
        X_pca = np.hstack([X_pca, pad])

    emb_cols = [f"emb_{i}" for i in range(AUTHOR_EMBEDDING_DIM)]
    embeddings_df = pd.DataFrame(X_pca, columns=emb_cols)
    embeddings_df.insert(0, "author_id", author_profiles["author_id"].values)

    print(f"[author_embed] Computed embeddings for {len(embeddings_df)} multi-work authors "
          f"(PCA explained variance ratio sum = {pca.explained_variance_ratio_.sum():.4f})")

    return embeddings_df, pca, scaler


# ===================================================================
# Cold-start: first-time authors
# ===================================================================

def cold_start_embeddings(
    catalog: pd.DataFrame,
    author_works: pd.DataFrame,
    multi_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Assign cold-start embeddings for single-work authors using
    genre-cluster centroids from the NMF topic space.

    Falls back to the global mean if topic vectors are unavailable.
    """
    single_authors = author_works.loc[author_works["n_works"] < 2, "author_id"].values
    if len(single_authors) == 0:
        return pd.DataFrame(columns=multi_embeddings.columns)

    emb_cols = [c for c in multi_embeddings.columns if c.startswith("emb_")]

    # Try loading topic vectors for NMF-topic-based matching
    topic_available = False
    if TOPIC_VECTORS_PATH.exists():
        try:
            tv = pd.read_parquet(TOPIC_VECTORS_PATH)
            tv_cols = [c for c in tv.columns if c.startswith("tv_")]
            if len(tv_cols) > 0:
                topic_available = True
        except Exception:
            pass

    # Build catalogue lookup: author_id → dominant genre (primary genre of their work)
    author_genre_map: dict[str, str] = {}
    for _, row in catalog.iterrows():
        aid = row["author_id"]
        genres = row.get("genres", [])
        if isinstance(genres, list) and len(genres) > 0:
            author_genre_map.setdefault(aid, genres[0])
        elif isinstance(genres, str):
            author_genre_map.setdefault(aid, genres)

    if topic_available:
        # Assign each multi-work author a dominant topic
        author_items = catalog[["author_id", "item_id"]].merge(tv, on="item_id", how="inner")
        author_topic = (
            author_items
            .groupby("author_id")[tv_cols]
            .mean()
            .reset_index()
        )
        # Dominant topic index
        author_topic["dominant_topic"] = author_topic[tv_cols].values.argmax(axis=1)

        # Merge dominant topic onto embeddings
        multi_with_topic = multi_embeddings.merge(
            author_topic[["author_id", "dominant_topic"]], on="author_id", how="left"
        )
        multi_with_topic["dominant_topic"] = multi_with_topic["dominant_topic"].fillna(-1).astype(int)

        # Cluster centroids: mean embedding per dominant topic
        centroids = multi_with_topic.groupby("dominant_topic")[emb_cols].mean()

        # For each single-author, find their item's dominant topic and use that centroid
        cold_rows: list[dict] = []
        for aid in single_authors:
            items = catalog.loc[catalog["author_id"] == aid, "item_id"].values
            if len(items) == 0:
                vec = multi_embeddings[emb_cols].mean().values
            else:
                item_topics = tv[tv["item_id"].isin(items)]
                if len(item_topics) > 0:
                    dom = int(item_topics[tv_cols].values.mean(axis=0).argmax())
                    if dom in centroids.index:
                        vec = centroids.loc[dom].values
                    else:
                        vec = multi_embeddings[emb_cols].mean().values
                else:
                    vec = multi_embeddings[emb_cols].mean().values
            row_dict = {"author_id": aid}
            for j, col in enumerate(emb_cols):
                row_dict[col] = float(vec[j])
            cold_rows.append(row_dict)
        cold_df = pd.DataFrame(cold_rows)
    else:
        # Fallback: global mean embedding
        global_mean = multi_embeddings[emb_cols].mean().values
        cold_rows = []
        for aid in single_authors:
            row_dict = {"author_id": aid}
            for j, col in enumerate(emb_cols):
                row_dict[col] = float(global_mean[j])
            cold_rows.append(row_dict)
        cold_df = pd.DataFrame(cold_rows)

    print(f"[author_embed] Assigned cold-start embeddings for {len(cold_df)} single-work authors")
    return cold_df


# ===================================================================
# Full pipeline
# ===================================================================

def run_pipeline(
    catalog_path: Path = CATALOG_PATH,
    session_path: Path = SESSION_FEATURES_PATH,
) -> pd.DataFrame:
    """Execute the author-embedding pipeline end-to-end.

    Returns
    -------
    DataFrame of all author embeddings (multi-work + cold-start).
    """
    # 1. Load catalog
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        print(f"[author_embed] Loaded catalog ({len(catalog)} items)")
    else:
        # Import NMF module to generate catalog
        from system_a_discovery_engine.layer1_content.nmf_topics import _generate_synthetic_catalog
        catalog = _generate_synthetic_catalog()

    catalog = ensure_layer1_catalog_columns(catalog)

    # 2. Load session features
    if session_path.exists():
        sessions = pd.read_parquet(session_path)
        print(f"[author_embed] Loaded session features ({len(sessions)} rows)")
    else:
        print("[author_embed] Session features not found — generating synthetic data")
        sessions = _generate_synthetic_session_features(catalog)

    # Ensure a rating column exists for avg_rating computation
    if "rating" not in sessions.columns:
        rng = np.random.default_rng(42)
        sessions["rating"] = rng.choice([0, 1, 2, 3, 4, 5], size=len(sessions),
                                        p=[0.3, 0.05, 0.1, 0.15, 0.2, 0.2]).astype(float)

    # 3. Interaction profiles
    profiles_df, author_works = compute_interaction_profiles(catalog, sessions)

    # 4. Multi-work author embeddings
    multi_emb, pca, scaler = compute_author_embeddings(profiles_df, author_works)

    # 5. Cold-start embeddings for single-work authors
    cold_emb = cold_start_embeddings(catalog, author_works, multi_emb)

    # 6. Combine & save
    all_emb = pd.concat([multi_emb, cold_emb], ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_emb.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[author_embed] Saved all author embeddings → {OUTPUT_PATH}  ({len(all_emb)} authors)")

    return all_emb


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    embeddings = run_pipeline()

    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    print("\n" + "=" * 72)
    print("AUTHOR EMBEDDING STATS")
    print("=" * 72)
    print(f"  Total authors embedded:       {len(embeddings)}")
    print(f"  Embedding dimensionality:     {len(emb_cols)}")
    emb_mat = embeddings[emb_cols].values
    print(f"  Embedding L2-norm (mean):     {np.linalg.norm(emb_mat, axis=1).mean():.4f}")
    print(f"  Embedding L2-norm (std):      {np.linalg.norm(emb_mat, axis=1).std():.4f}")
    print(f"  Min value across all embeds:  {emb_mat.min():.4f}")
    print(f"  Max value across all embeds:  {emb_mat.max():.4f}")

    # Show first 5 authors
    print(f"\n  First 5 authors:")
    for _, row in embeddings.head(5).iterrows():
        vec = row[emb_cols].values[:5]
        print(f"    {row['author_id']}: [{', '.join(f'{v:.3f}' for v in vec)}, …]")

    print("\n✅ Author embedding pipeline complete.")
