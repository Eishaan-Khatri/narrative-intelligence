"""Feature construction for System B breakout and opportunity models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_item_exposure_features(exposure_log: pd.DataFrame, item_universe: pd.DataFrame) -> pd.DataFrame:
    max_day = int(exposure_log["day"].max()) if "day" in exposure_log.columns and not exposure_log.empty else 0
    cutoff_day = max(0, int(max_day * 0.40))
    early_log = exposure_log[exposure_log["day"] <= cutoff_day].copy()
    future_log = exposure_log[exposure_log["day"] > cutoff_day].copy()
    if early_log.empty:
        early_log = exposure_log.copy()
    if future_log.empty:
        future_log = exposure_log.copy()

    agg = (
        early_log.groupby("item_id")
        .agg(
            impressions=("impression", "sum"),
            clicks=("click", "sum"),
            completions=("chapter_complete", "sum"),
            returns_7d=("return_7d", "sum"),
            reward_sum=("reward", "sum"),
            explored_impressions=("treated", "sum"),
            avg_logging_propensity=("logging_propensity", "mean"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        )
        .reset_index()
    )
    agg["ctr"] = agg["clicks"] / agg["impressions"].clip(lower=1)
    agg["completion_rate"] = agg["completions"] / agg["clicks"].clip(lower=1)
    agg["return_rate"] = agg["returns_7d"] / agg["completions"].clip(lower=1)
    agg["reward_per_impression"] = agg["reward_sum"] / agg["impressions"].clip(lower=1)
    agg["exposure_span_days"] = (agg["last_day"] - agg["first_day"] + 1).clip(lower=1)
    agg["impression_velocity"] = agg["impressions"] / agg["exposure_span_days"]
    agg["exploration_share"] = agg["explored_impressions"] / agg["impressions"].clip(lower=1)

    future = (
        future_log.groupby("item_id")
        .agg(
            future_impressions=("impression", "sum"),
            future_clicks=("click", "sum"),
            future_completions=("chapter_complete", "sum"),
            future_returns_7d=("return_7d", "sum"),
            future_reward_sum=("reward", "sum"),
        )
        .reset_index()
    )
    future["future_reward_per_impression"] = future["future_reward_sum"] / future["future_impressions"].clip(lower=1)

    merged = item_universe.merge(agg, on="item_id", how="left")
    merged = merged.merge(future, on="item_id", how="left")
    for col in [
        "impressions",
        "clicks",
        "completions",
        "returns_7d",
        "reward_sum",
        "explored_impressions",
        "ctr",
        "completion_rate",
        "return_rate",
        "reward_per_impression",
        "impression_velocity",
        "exploration_share",
        "future_impressions",
        "future_clicks",
        "future_completions",
        "future_returns_7d",
        "future_reward_sum",
        "future_reward_per_impression",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    future_threshold = merged["future_reward_per_impression"].quantile(0.85)
    merged["breakout_label"] = (
        (merged["future_reward_per_impression"] >= future_threshold)
        & (merged["future_impressions"] >= merged["future_impressions"].quantile(0.20))
    ).astype(int)
    merged["underexposed"] = (merged["impressions"].rank(pct=True) <= 0.50).astype(int)
    merged["opportunity_label"] = ((merged["breakout_label"] == 1) & (merged["underexposed"] == 1)).astype(int)
    merged["feature_window_max_day"] = cutoff_day
    return merged


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        "breakout_label",
        "opportunity_label",
        "item_index",
        "posterior_alpha",
        "posterior_beta",
    }
    cols = []
    for col in frame.columns:
        if col in exclude:
            continue
        if col.startswith("future_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def add_embedding_summary(features: pd.DataFrame, item_embeddings: pd.DataFrame | None) -> pd.DataFrame:
    if item_embeddings is None or item_embeddings.empty:
        features["embedding_norm"] = 0.0
        return features

    emb = item_embeddings.copy()
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    if not emb_cols:
        features["embedding_norm"] = 0.0
        return features
    emb["embedding_norm"] = np.linalg.norm(emb[emb_cols].fillna(0.0).to_numpy(dtype=float), axis=1)
    if not set(emb["item_id"]).intersection(set(features["item_id"])):
        emb["item_id"] = [f"item_{i:05d}" for i in range(len(emb))]
    return features.merge(emb[["item_id", "embedding_norm"]], on="item_id", how="left").assign(
        embedding_norm=lambda df: df["embedding_norm"].fillna(0.0)
    )
