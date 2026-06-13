"""
Parquet-to-array feature extraction for the two-tower retrieval trainer.

This module is intentionally pandas/numpy-only so the feature contract can be
tested without importing PyTorch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store.schema import (
    AUTHOR_EMBEDDING_DIM,
    ENGAGEMENT_PROFILE_DIM,
    ITEM_EMBEDDING_DIM,
    ITEM_FINGERPRINT_DIM,
    ITEM_TOWER_INPUT_DIM,
    NMF_TOPICS_DIM,
    STRUCTURAL_FEATURES_DIM,
    USER_TOWER_CONTEXT_DIM,
    USER_TOWER_INPUT_DIM,
)


def _fingerprint_columns() -> list[str]:
    return (
        [f"tv_{idx}" for idx in range(NMF_TOPICS_DIM)]
        + [f"ae_{idx}" for idx in range(AUTHOR_EMBEDDING_DIM)]
        + ["quality_score"]
        + [f"sf_{idx}" for idx in range(STRUCTURAL_FEATURES_DIM)]
    )


def _parse_vector(value: object, expected_dim: int) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected vector length {expected_dim}, got {arr.shape[0]}")
    return arr


def _time_bucket(series: pd.Series) -> pd.Series:
    iso = pd.to_datetime(series).dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def _normalise_engagement(profile: np.ndarray) -> np.ndarray:
    profile = profile.astype(np.float32, copy=True)
    if profile.shape[0] != ENGAGEMENT_PROFILE_DIM:
        raise ValueError(
            f"Expected engagement profile length {ENGAGEMENT_PROFILE_DIM}, got {profile.shape[0]}"
        )

    normalised = np.zeros(ENGAGEMENT_PROFILE_DIM, dtype=np.float32)
    normalised[0] = np.clip(profile[0] / 600.0, 0.0, 1.0)
    normalised[1] = np.clip(profile[1], 0.0, 1.0)
    normalised[2] = np.clip(profile[2], 0.0, 1.0)
    normalised[3] = np.clip(profile[3], 0.0, 1.0)
    normalised[4] = np.clip(profile[4], 0.0, 1.0)
    normalised[5] = np.clip(np.log1p(max(profile[5], 0.0)) / np.log1p(3600.0), 0.0, 1.0)
    normalised[6] = np.clip(np.log1p(max(profile[6], 0.0)) / np.log1p(20.0), 0.0, 1.0)
    normalised[7] = np.clip(np.log1p(max(profile[7], 0.0)) / np.log1p(20.0), 0.0, 1.0)
    return normalised


def _derive_engagement_from_sessions(user_sessions: pd.DataFrame) -> np.ndarray:
    n = max(len(user_sessions), 1)
    shapes = user_sessions["completion_curve_shape"].fillna("")
    profile = np.array(
        [
            user_sessions["reading_velocity_wpm"].mean(),
            user_sessions["final_completion_pct"].mean(),
            user_sessions["re_read_ratio"].mean(),
            (shapes == "cliff").mean(),
            (shapes == "decay").mean(),
            user_sessions["session_duration_sec"].mean(),
            float(n),
            float(user_sessions["item_id"].nunique()),
        ],
        dtype=np.float32,
    )
    return _normalise_engagement(profile)


def _context_features(row: pd.Series) -> np.ndarray:
    context = np.zeros(USER_TOWER_CONTEXT_DIM, dtype=np.float32)

    hour = pd.Timestamp(row["timestamp_start"]).hour
    if 5 <= hour < 11:
        context[0] = 1.0
    elif 11 <= hour < 17:
        context[1] = 1.0
    elif 17 <= hour < 22:
        context[2] = 1.0
    elif 22 <= hour or hour < 2:
        context[3] = 1.0
    else:
        context[4] = 1.0

    device = str(row.get("device_type", "mobile")).lower()
    if device == "mobile":
        context[5] = 1.0
    else:
        context[6] = 1.0

    completion = float(row.get("final_completion_pct", 0.0))
    exit_reason = str(row.get("exit_reason", "mid_chapter"))
    if exit_reason == "chapter_end" or completion >= 0.95:
        context[7] = 1.0
    elif completion >= 0.5:
        context[8] = 1.0
    else:
        context[9] = 1.0

    return context


def _load_temporal_profiles(temporal_path: Path | None) -> dict[tuple[str, str], np.ndarray]:
    if temporal_path is None or not temporal_path.exists():
        return {}

    temporal = pd.read_parquet(temporal_path)
    required = {"user_id", "time_bucket", "engagement_profile_vector"}
    missing = required - set(temporal.columns)
    if missing:
        raise ValueError(f"{temporal_path} missing required columns: {sorted(missing)}")

    profiles: dict[tuple[str, str], np.ndarray] = {}
    for row in temporal.itertuples(index=False):
        vector = _parse_vector(row.engagement_profile_vector, ENGAGEMENT_PROFILE_DIM)
        profiles[(row.user_id, row.time_bucket)] = _normalise_engagement(vector)
    return profiles


def build_real_training_arrays(
    session_path: Path,
    fingerprint_path: Path,
    temporal_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two-tower training arrays from processed pipeline artifacts."""
    sessions = pd.read_parquet(session_path).copy()
    fingerprints = pd.read_parquet(fingerprint_path).copy()

    required_session_cols = {
        "user_id",
        "item_id",
        "timestamp_start",
        "device_type",
        "exit_reason",
        "final_completion_pct",
        "reading_velocity_wpm",
        "re_read_ratio",
        "completion_curve_shape",
        "session_duration_sec",
    }
    missing_sessions = required_session_cols - set(sessions.columns)
    if missing_sessions:
        raise ValueError(f"{session_path} missing required columns: {sorted(missing_sessions)}")

    fp_cols = _fingerprint_columns()
    missing_fp = set(["item_id", *fp_cols]) - set(fingerprints.columns)
    if missing_fp:
        raise ValueError(f"{fingerprint_path} missing required columns: {sorted(missing_fp)}")

    sessions["timestamp_start"] = pd.to_datetime(sessions["timestamp_start"])
    sessions["time_bucket"] = _time_bucket(sessions["timestamp_start"])
    fingerprints = fingerprints.drop_duplicates("item_id").reset_index(drop=True)

    valid_items = set(fingerprints["item_id"])
    sessions = sessions[sessions["item_id"].isin(valid_items)].sort_values("timestamp_start")
    if sessions.empty:
        raise ValueError("No sessions reference items present in item_fingerprints.parquet")

    item_ids_ordered = fingerprints["item_id"].tolist()
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids_ordered)}
    fingerprint_matrix = fingerprints[fp_cols].to_numpy(dtype=np.float32)
    if fingerprint_matrix.shape[1] != ITEM_FINGERPRINT_DIM:
        raise ValueError(
            f"Expected {ITEM_FINGERPRINT_DIM} fingerprint columns, got {fingerprint_matrix.shape[1]}"
        )

    max_ts = sessions["timestamp_start"].max()
    counts_total = sessions.groupby("item_id").size().reindex(item_ids_ordered, fill_value=0)
    recent_7 = sessions[
        sessions["timestamp_start"] >= max_ts - pd.Timedelta(days=7)
    ].groupby("item_id").size().reindex(item_ids_ordered, fill_value=0)
    recent_30 = sessions[
        sessions["timestamp_start"] >= max_ts - pd.Timedelta(days=30)
    ].groupby("item_id").size().reindex(item_ids_ordered, fill_value=0)

    log_interactions = np.log1p(counts_total.to_numpy(dtype=np.float32)).reshape(-1, 1)
    velocity = (
        (recent_7.to_numpy(dtype=np.float32) / 7.0)
        / np.maximum(recent_30.to_numpy(dtype=np.float32) / 30.0, 1e-6)
    )
    velocity = np.clip(velocity, 0.0, 10.0).reshape(-1, 1)
    item_catalog_features = np.hstack([fingerprint_matrix, log_interactions, velocity]).astype(np.float32)
    if item_catalog_features.shape[1] != ITEM_TOWER_INPUT_DIM:
        raise ValueError(
            f"Expected {ITEM_TOWER_INPUT_DIM} item feature columns, got {item_catalog_features.shape[1]}"
        )

    fingerprint_by_item = {
        item_id: fingerprint_matrix[idx]
        for item_id, idx in item_index.items()
    }
    user_history_pool: dict[str, np.ndarray] = {}
    for user_id, user_group in sessions.groupby("user_id"):
        vectors = [fingerprint_by_item[item_id] for item_id in user_group["item_id"].unique()]
        mean_fp = np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(ITEM_FINGERPRINT_DIM)
        pooled = np.zeros(ITEM_EMBEDDING_DIM, dtype=np.float32)
        pooled[:ITEM_FINGERPRINT_DIM] = mean_fp
        user_history_pool[user_id] = pooled

    temporal_profiles = _load_temporal_profiles(temporal_path)
    derived_profiles = {
        user_id: _derive_engagement_from_sessions(user_group)
        for user_id, user_group in sessions.groupby("user_id")
    }

    user_features: list[np.ndarray] = []
    item_features: list[np.ndarray] = []
    sample_log_popularity: list[float] = []
    sample_item_ids: list[int] = []

    for _, row in sessions.iterrows():
        item_id = row["item_id"]
        item_idx = item_index[item_id]
        engagement = temporal_profiles.get(
            (row["user_id"], row["time_bucket"]),
            derived_profiles[row["user_id"]],
        )
        user_vec = np.concatenate(
            [
                user_history_pool[row["user_id"]],
                engagement,
                _context_features(row),
            ]
        ).astype(np.float32)
        if user_vec.shape[0] != USER_TOWER_INPUT_DIM:
            raise ValueError(
                f"Expected {USER_TOWER_INPUT_DIM} user feature columns, got {user_vec.shape[0]}"
            )

        user_features.append(user_vec)
        item_features.append(item_catalog_features[item_idx])
        sample_log_popularity.append(float(log_interactions[item_idx, 0]))
        sample_item_ids.append(item_idx)

    return (
        np.vstack(user_features).astype(np.float32),
        np.vstack(item_features).astype(np.float32),
        np.asarray(sample_log_popularity, dtype=np.float32),
        item_catalog_features,
        np.asarray(sample_item_ids, dtype=np.int64),
    )
