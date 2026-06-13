"""
Narrative Intelligence Platform — User Temporal Features Builder
================================================================
Computes rolling user-level engagement profiles per weekly time bucket.

Input:  data/processed/session_features.parquet
Output: data/processed/user_temporal_features.parquet

Features computed per (user_id, time_bucket):
  - engagement_profile_vector (8-dim rolling average)
  - inter_chapter_gap_trend (slope of gap-in-days for active stories)
  - genre_drift_flag (boolean)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_store.schema import (
    ENGAGEMENT_PROFILE_DIM,
    NMF_TOPICS_DIM,
)


def compute_weekly_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ISO week bucket string to each session."""
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["timestamp_start"]).dt.isocalendar().apply(
        lambda row: f"{row.year}-W{row.week:02d}", axis=1
    )
    return df


def compute_engagement_profile(group: pd.DataFrame) -> np.ndarray:
    """
    Compute 8-dim engagement profile vector for a user in a time bucket.

    Dimensions:
      0: avg_velocity          — mean reading_velocity_wpm
      1: avg_completion_pct    — mean final_completion_pct
      2: avg_re_read_ratio     — mean re_read_ratio
      3: pct_cliff_sessions    — fraction of sessions with cliff shape
      4: pct_decay_sessions    — fraction of sessions with decay shape
      5: avg_session_duration   — mean session_duration_sec
      6: sessions_per_week     — total session count (normalized later)
      7: distinct_genres_touched — unique item genre clusters (proxy: unique items)
    """
    n = len(group)
    if n == 0:
        return np.zeros(ENGAGEMENT_PROFILE_DIM)

    avg_velocity = group["reading_velocity_wpm"].mean()
    avg_completion = group["final_completion_pct"].mean()
    avg_reread = group["re_read_ratio"].mean()

    shapes = group["completion_curve_shape"]
    pct_cliff = (shapes == "cliff").mean()
    pct_decay = (shapes == "decay").mean()

    avg_duration = group["session_duration_sec"].mean()
    sessions_count = float(n)
    distinct_items = float(group["item_id"].nunique())

    return np.array([
        avg_velocity,
        avg_completion,
        avg_reread,
        pct_cliff,
        pct_decay,
        avg_duration,
        sessions_count,
        distinct_items,
    ])


def compute_inter_chapter_gap_trend(user_sessions: pd.DataFrame) -> float:
    """
    For each (user, item) the user is actively reading, compute linear
    regression slope of gap-in-days between consecutive chapter reads
    over the last 5 chapters. Return mean across active stories.

    A positive slope means increasing gaps (losing interest).
    A negative slope means decreasing gaps (accelerating engagement).
    """
    slopes = []
    for item_id, item_group in user_sessions.groupby("item_id"):
        item_sorted = item_group.sort_values("chapter_index")
        if len(item_sorted) < 3:
            continue

        # Take last 5 chapter reads
        recent = item_sorted.tail(5)
        timestamps = pd.to_datetime(recent["timestamp_start"])
        gaps_days = timestamps.diff().dt.total_seconds().dropna() / 86400.0

        if len(gaps_days) < 2:
            continue

        # Linear regression: slope of gap vs chapter position
        x = np.arange(len(gaps_days))
        try:
            slope = np.polyfit(x, gaps_days.values, deg=1)[0]
            slopes.append(slope)
        except (np.linalg.LinAlgError, ValueError):
            continue

    return float(np.mean(slopes)) if slopes else 0.0


def compute_genre_drift(
    user_id: str,
    current_bucket: str,
    user_bucket_topics: dict[str, dict[str, np.ndarray]],
) -> bool:
    """
    Check if the user's dominant NMF topic cluster has changed between
    the current time bucket and the previous 3 buckets.
    """
    if user_id not in user_bucket_topics:
        return False

    buckets = sorted(user_bucket_topics[user_id].keys())
    if current_bucket not in buckets:
        return False

    idx = buckets.index(current_bucket)
    if idx < 1:
        return False

    # Get dominant topic for current and previous buckets
    current_dominant = np.argmax(user_bucket_topics[user_id][current_bucket])

    prev_dominants = []
    for i in range(max(0, idx - 3), idx):
        prev_topic = user_bucket_topics[user_id][buckets[i]]
        prev_dominants.append(np.argmax(prev_topic))

    if not prev_dominants:
        return False

    # Drift = current dominant topic differs from majority of previous
    from collections import Counter
    most_common_prev = Counter(prev_dominants).most_common(1)[0][0]
    return current_dominant != most_common_prev


def build_temporal_features(
    session_features_path: Path,
    output_path: Path,
    topic_vectors_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build user temporal features table from session features.

    Parameters
    ----------
    session_features_path : Path to session_features.parquet
    output_path : Path to save user_temporal_features.parquet
    topic_vectors_path : Optional path to topic_vectors.parquet for genre drift

    Returns
    -------
    pd.DataFrame with columns: user_id, time_bucket, engagement_profile_vector,
                                inter_chapter_gap_trend, genre_drift_flag
    """
    print("Loading session features...")
    sf = pd.read_parquet(session_features_path)
    sf["timestamp_start"] = pd.to_datetime(sf["timestamp_start"])

    # Assign time buckets
    print("Assigning weekly time buckets...")
    iso_cal = sf["timestamp_start"].dt.isocalendar()
    sf["time_bucket"] = iso_cal["year"].astype(str) + "-W" + iso_cal["week"].astype(str).str.zfill(2)

    # Load topic vectors for genre drift detection (optional)
    item_topics = None
    if topic_vectors_path and topic_vectors_path.exists():
        print("Loading topic vectors for genre drift detection...")
        tv = pd.read_parquet(topic_vectors_path)
        if "topic_vector" in tv.columns:
            item_topics = dict(zip(tv["item_id"], tv["topic_vector"].apply(np.array)))
        else:
            tv_cols = sorted(
                [c for c in tv.columns if c.startswith("tv_")],
                key=lambda col: int(col.split("_", 1)[1]),
            )
            if len(tv_cols) != NMF_TOPICS_DIM:
                raise ValueError(
                    f"{topic_vectors_path} must contain either topic_vector or "
                    f"{NMF_TOPICS_DIM} tv_* columns; found {len(tv_cols)}."
                )
            item_topics = dict(zip(tv["item_id"], tv[tv_cols].to_numpy(dtype=float)))

    # Build per-user-per-bucket aggregation of dominant topics (for drift)
    user_bucket_topics: dict[str, dict[str, np.ndarray]] = {}
    if item_topics:
        print("Computing per-user-per-bucket topic distributions...")
        for (user_id, bucket), group in tqdm(
            sf.groupby(["user_id", "time_bucket"]),
            desc="Topic distributions",
        ):
            item_vecs = [item_topics[iid] for iid in group["item_id"].unique() if iid in item_topics]
            if item_vecs:
                avg_topic = np.mean(item_vecs, axis=0)
                user_bucket_topics.setdefault(user_id, {})[bucket] = avg_topic

    # Compute features per (user, time_bucket)
    print("Computing engagement profiles...")
    records = []
    grouped = sf.groupby(["user_id", "time_bucket"])

    for (user_id, time_bucket), group in tqdm(grouped, desc="User temporal features"):
        # 1. Engagement profile vector (8-dim)
        profile = compute_engagement_profile(group)

        # 2. Inter-chapter gap trend
        gap_trend = compute_inter_chapter_gap_trend(group)

        # 3. Genre drift flag
        drift = compute_genre_drift(user_id, time_bucket, user_bucket_topics)

        records.append({
            "user_id": user_id,
            "time_bucket": time_bucket,
            "engagement_profile_vector": profile.tolist(),
            "inter_chapter_gap_trend": gap_trend,
            "genre_drift_flag": drift,
        })

    result = pd.DataFrame(records)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"[OK] Saved {len(result)} user-temporal-feature rows to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    """Build temporal features from the standard processed feature-store paths."""
    data_dir = PROJECT_ROOT / "data"
    session_features_path = data_dir / "processed" / "session_features.parquet"
    output_path = data_dir / "processed" / "user_temporal_features.parquet"
    topic_vectors_path = data_dir / "processed" / "topic_vectors.parquet"

    if not session_features_path.exists():
        raise FileNotFoundError(
            f"{session_features_path} does not exist. Run session_features first."
        )

    return build_temporal_features(
        session_features_path=session_features_path,
        output_path=output_path,
        topic_vectors_path=topic_vectors_path if topic_vectors_path.exists() else None,
    )


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    session_features_path = data_dir / "processed" / "session_features.parquet"
    output_path = data_dir / "processed" / "user_temporal_features.parquet"
    topic_vectors_path = data_dir / "processed" / "topic_vectors.parquet"

    if not session_features_path.exists():
        print("⚠ session_features.parquet not found. Generating synthetic data for demo...")

        # Generate minimal synthetic session features
        np.random.seed(42)
        n_sessions = 5000
        n_users = 200
        n_items = 100

        base_time = pd.Timestamp("2024-01-01")
        records = []
        for i in range(n_sessions):
            user_id = f"u_{np.random.randint(n_users):04d}"
            item_id = f"i_{np.random.randint(n_items):04d}"
            chapter = np.random.randint(0, 15)
            start = base_time + pd.Timedelta(days=np.random.randint(0, 90),
                                              hours=np.random.randint(0, 24))
            duration = np.random.exponential(300)
            velocity = np.clip(np.random.normal(200, 80), 50, 600)
            completion = np.clip(np.random.beta(2, 1.5), 0, 1)
            shapes = ["cliff", "decay", "steady", "abandon_early"]

            records.append({
                "session_id": f"sess_{i:06d}",
                "user_id": user_id,
                "item_id": item_id,
                "chapter_index": chapter,
                "session_duration_sec": duration,
                "reading_velocity_wpm": velocity,
                "velocity_acceleration": np.random.normal(0, 20),
                "completion_curve": np.sort(np.random.rand(5)).tolist(),
                "completion_curve_shape": np.random.choice(shapes, p=[0.15, 0.2, 0.5, 0.15]),
                "re_read_ratio": np.random.beta(1, 10),
                "final_completion_pct": completion,
                "exit_reason": np.random.choice(["chapter_end", "mid_chapter", "app_close"],
                                                 p=[0.4, 0.4, 0.2]),
                "device_type": np.random.choice(["mobile", "desktop", "tablet"],
                                                 p=[0.6, 0.3, 0.1]),
                "timestamp_start": start.isoformat(),
                "timestamp_end": (start + pd.Timedelta(seconds=duration)).isoformat(),
            })

        sf_df = pd.DataFrame(records)
        session_features_path.parent.mkdir(parents=True, exist_ok=True)
        sf_df.to_parquet(session_features_path, index=False)
        print(f"  Generated {n_sessions} synthetic sessions.")

    # Run pipeline
    result = build_temporal_features(
        session_features_path=session_features_path,
        output_path=output_path,
        topic_vectors_path=topic_vectors_path if topic_vectors_path.exists() else None,
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("USER TEMPORAL FEATURES — SUMMARY")
    print("=" * 60)
    print(f"Total rows:         {len(result)}")
    print(f"Unique users:       {result['user_id'].nunique()}")
    print(f"Unique time buckets: {result['time_bucket'].nunique()}")

    profiles = np.array(result["engagement_profile_vector"].tolist())
    dim_names = [
        "avg_velocity", "avg_completion", "avg_reread",
        "pct_cliff", "pct_decay", "avg_duration",
        "sessions_per_week", "distinct_items",
    ]
    print("\nEngagement Profile Dimensions (mean ± std):")
    for i, name in enumerate(dim_names):
        print(f"  {name:25s}: {profiles[:, i].mean():8.3f} ± {profiles[:, i].std():8.3f}")

    print(f"\nInter-chapter gap trend (mean): {result['inter_chapter_gap_trend'].mean():.4f}")
    print(f"Genre drift rate:               {result['genre_drift_flag'].mean():.2%}")
    print("\n✅ User temporal features built successfully.")
