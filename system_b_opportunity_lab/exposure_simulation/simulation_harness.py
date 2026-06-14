"""Synthetic exposure-log generator for System B."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYSTEM_B_DIR = PROCESSED_DIR / "system_b"


def _minmax(series: pd.Series) -> pd.Series:
    s = series.fillna(series.median() if series.notna().any() else 0.0).astype(float)
    span = s.max() - s.min()
    if span <= 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / span


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


def load_item_universe(limit_items: int | None = None) -> pd.DataFrame:
    """Load System A artifacts into one item universe for System B."""
    fingerprints = pd.read_parquet(PROCESSED_DIR / "item_fingerprints.parquet")
    quality = pd.read_parquet(PROCESSED_DIR / "quality_scores.parquet")
    sessions = pd.read_parquet(PROCESSED_DIR / "session_features.parquet")

    catalog_path = SYNTHETIC_DIR / "catalog.parquet"
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
    else:
        catalog = pd.DataFrame({"item_id": fingerprints["item_id"]})

    session_agg = (
        sessions.groupby("item_id")
        .agg(
            sessions=("session_id", "nunique"),
            users=("user_id", "nunique"),
            mean_completion=("final_completion_pct", "mean"),
            completion_events=("exit_reason", lambda s: int((s == "chapter_end").sum())),
            reread_mean=("re_read_ratio", "mean"),
        )
        .reset_index()
    )

    item = fingerprints[["item_id", "quality_score"] + [c for c in fingerprints.columns if c.startswith("tv_")][:8]].copy()
    item = item.merge(quality, on="item_id", how="left", suffixes=("", "_quality"))
    item = item.merge(session_agg, on="item_id", how="left")
    item = item.merge(
        catalog[[c for c in ["item_id", "title", "author_id", "author_name", "genres", "latent_quality", "rating_count"] if c in catalog.columns]],
        on="item_id",
        how="left",
    )

    item["creator_id"] = item.get("author_id", pd.Series("unknown_creator", index=item.index)).fillna("unknown_creator")
    item["creator_name"] = item.get("author_name", pd.Series("Unknown", index=item.index)).fillna("Unknown")
    item["genres_norm"] = item.get("genres", pd.Series([[] for _ in range(len(item))], index=item.index)).apply(_normalise_genres)
    item["primary_genre"] = item["genres_norm"].apply(lambda x: x[0] if x else "Unknown")
    item["sessions"] = item["sessions"].fillna(0).astype(int)
    item["users"] = item["users"].fillna(0).astype(int)
    item["completion_events"] = item["completion_events"].fillna(0).astype(int)
    item["rating_count"] = item.get("rating_count", pd.Series(0, index=item.index)).fillna(0).astype(float)

    quality_signal = _minmax(item["quality_score"])
    completion_signal = _minmax(item["mean_completion"].fillna(0.0))
    latent_signal = _minmax(item.get("latent_quality", pd.Series(0.5, index=item.index)))
    item["true_quality"] = np.clip(0.45 * quality_signal + 0.35 * completion_signal + 0.20 * latent_signal, 0.02, 0.98)
    item["base_popularity"] = np.log1p(item["sessions"] + item["rating_count"])
    item["popularity_percentile"] = item["base_popularity"].rank(pct=True)
    item["item_index"] = np.arange(len(item))

    if limit_items is not None:
        item = item.sort_values(["sessions", "true_quality"], ascending=False).head(limit_items).reset_index(drop=True)
        item["item_index"] = np.arange(len(item))
    return item


def simulate_exposure_log(
    item_universe: pd.DataFrame | None = None,
    n_users: int = 8000,
    n_days: int = 45,
    impressions_per_day: int = 3000,
    exploration_rate: float = 0.12,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate logged feed exposure with known propensities."""
    rng = np.random.default_rng(seed)
    items = item_universe.copy() if item_universe is not None else load_item_universe()
    items = items.reset_index(drop=True)
    n_items = len(items)
    if n_items == 0:
        raise ValueError("No items available for exposure simulation.")

    popularity = np.maximum(items["base_popularity"].to_numpy(dtype=float), 0.01)
    popularity_probs = popularity / popularity.sum()
    tail_weight = 1.0 - items["popularity_percentile"].to_numpy(dtype=float)
    tail_probs = np.maximum(tail_weight, 0.001)
    tail_probs = tail_probs / tail_probs.sum()

    user_clusters = rng.integers(0, 20, size=n_users)
    cluster_bias = rng.normal(0.0, 0.08, size=(20, n_items))
    topic_cols = [c for c in items.columns if c.startswith("tv_")][:8]
    if topic_cols:
        topic_matrix = items[topic_cols].fillna(0.0).to_numpy(dtype=float)
        cluster_topic = rng.normal(0.0, 1.0, size=(20, len(topic_cols)))
        cluster_bias += 0.08 * (cluster_topic @ topic_matrix.T)

    records: list[dict] = []
    true_quality = items["true_quality"].to_numpy(dtype=float)
    creators = items["creator_id"].astype(str).to_numpy()
    item_ids = items["item_id"].astype(str).to_numpy()

    for day in range(n_days):
        for _ in range(impressions_per_day):
            user_idx = int(rng.integers(0, n_users))
            cluster = int(user_clusters[user_idx])
            explore = bool(rng.random() < exploration_rate)
            if explore:
                item_idx = int(rng.choice(n_items, p=tail_probs))
                logging_propensity = exploration_rate * tail_probs[item_idx] + (1.0 - exploration_rate) * popularity_probs[item_idx]
            else:
                item_idx = int(rng.choice(n_items, p=popularity_probs))
                logging_propensity = (1.0 - exploration_rate) * popularity_probs[item_idx] + exploration_rate * tail_probs[item_idx]

            affinity = float(np.clip(true_quality[item_idx] + cluster_bias[cluster, item_idx], 0.01, 0.99))
            click_p = float(np.clip(0.04 + 0.35 * affinity, 0.01, 0.70))
            complete_p = float(np.clip(0.02 + 0.45 * affinity, 0.005, 0.80))
            return_p = float(np.clip(0.01 + 0.28 * affinity, 0.002, 0.60))
            click = int(rng.random() < click_p)
            complete = int(click and rng.random() < complete_p)
            return_7d = int(complete and rng.random() < return_p)
            reward = 0.5 * click + 0.3 * complete + 0.2 * return_7d

            records.append(
                {
                    "day": day,
                    "time_bucket": f"day_{day:03d}",
                    "user_id": f"b_user_{user_idx:05d}",
                    "user_cluster": cluster,
                    "item_id": item_ids[item_idx],
                    "creator_id": creators[item_idx],
                    "impression": 1,
                    "click": click,
                    "chapter_complete": complete,
                    "return_7d": return_7d,
                    "reward": reward,
                    "treated": int(explore),
                    "logging_policy": "popularity_epsilon_explore",
                    "logging_propensity": float(max(logging_propensity, 1e-8)),
                    "true_quality": float(true_quality[item_idx]),
                }
            )

    exposure = pd.DataFrame(records)
    SYSTEM_B_DIR.mkdir(parents=True, exist_ok=True)
    items.to_parquet(SYSTEM_B_DIR / "item_universe.parquet", index=False)
    exposure.to_parquet(SYSTEM_B_DIR / "exposure_log.parquet", index=False)
    return exposure, items
