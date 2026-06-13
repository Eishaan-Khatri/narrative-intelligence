"""
Narrative Intelligence Platform — System A, Layer 4
====================================================
Retrieval Oracle Experiment
----------------------------

**Purpose**:
Measure the *ceiling* performance of the Two-Tower retrieval stage.
For every held-out high-quality positive interaction (Grade ≥ 3), we check
whether that item appears in the retrieval model's top-500 candidates.
This **Recall@500** is the maximum that *any* downstream re-ranker can
achieve — it quantifies what the retrieval layer leaves on the table.

**Analysis dimensions**:
  1. **Overall** oracle Recall@500
  2. **By popularity quartile** — does retrieval systematically miss
     long-tail / niche items?
  3. **Gap analysis** — what characteristics do the "missed" high-quality
     items share?  (genre profile, quality score, age, author niche-ness)

**Outputs**:
  - ``data/processed/oracle_analysis.parquet``
  - ``data/processed/oracle_recall.png`` (bar chart by quartile)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from system_a_discovery_engine.layer4_evaluation.completion_ndcg import (  # noqa: E402
    assign_relevance_grades_vec,
)

OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"
ORACLE_PARQUET = OUTPUT_DIR / "oracle_analysis.parquet"
ORACLE_PNG = OUTPUT_DIR / "oracle_recall.png"


# ===================================================================
# Synthetic data generation
# ===================================================================

def generate_oracle_data(
    n_users: int = 400,
    n_items: int = 1000,
    top_k: int = 500,
    seed: int = 42,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Produce synthetic data simulating a Two-Tower retrieval stage.

    Creates:
    - Item metadata (popularity percentile, quality score, genre vector,
      author niche-ness, age days)
    - Per-user ground-truth interactions with completion percentages
    - Per-user top-k retrieval lists (with a realistic popularity bias
      so that long-tail items are harder to retrieve)

    Parameters
    ----------
    n_users : int
    n_items : int
    top_k : int
    seed : int

    Returns
    -------
    dict with keys:
      - ``item_meta``       : pd.DataFrame (n_items rows)
      - ``user_positives``  : dict mapping user_id → list of positive item_ids
      - ``retrieval_top_k`` : dict mapping user_id → np.ndarray of top-k item ids
      - ``positive_details``: pd.DataFrame (one row per positive interaction)
    """
    rng = np.random.default_rng(seed)

    # --- Item metadata ---
    popularity_raw = rng.pareto(a=1.5, size=n_items)
    popularity_pctl = np.argsort(np.argsort(popularity_raw)) / n_items  # percentile
    quality_scores = rng.beta(3, 2, size=n_items)
    age_days = rng.exponential(90, size=n_items)
    author_niche = rng.beta(2, 5, size=n_items)  # low = mainstream, high = niche
    genre_vector = rng.dirichlet(np.ones(5), size=n_items)

    item_meta = pd.DataFrame({
        "item_id": [f"i_{j:05d}" for j in range(n_items)],
        "popularity_pctl": popularity_pctl,
        "quality_score": quality_scores,
        "age_days": age_days,
        "author_niche": author_niche,
    })
    for g in range(5):
        item_meta[f"genre_{g}"] = genre_vector[:, g]

    # Assign popularity quartile
    item_meta["pop_quartile"] = pd.qcut(
        item_meta["popularity_pctl"], q=4,
        labels=["Q1 (tail)", "Q2", "Q3", "Q4 (popular)"]
    )

    # --- Per-user positives ---
    user_positives: Dict[str, List[int]] = {}
    positive_rows: list[dict] = []

    for u in range(n_users):
        uid = f"u_{u:04d}"
        # Each user has 3–15 positive interactions
        n_pos = rng.integers(3, 16)
        pos_items = rng.choice(n_items, size=n_pos, replace=False)
        completion = rng.beta(4, 2, size=n_pos)  # skewed high
        returned = rng.random(n_pos) < 0.3
        grades = assign_relevance_grades_vec(completion, returned)

        # Keep only Grade ≥ 3 as "held-out positives"
        high_quality_mask = grades >= 3
        hq_items = pos_items[high_quality_mask].tolist()
        user_positives[uid] = hq_items

        for idx in range(n_pos):
            if grades[idx] >= 3:
                positive_rows.append({
                    "user_id": uid,
                    "item_id": f"i_{pos_items[idx]:05d}",
                    "item_idx": int(pos_items[idx]),
                    "completion_pct": float(completion[idx]),
                    "returned_7d": bool(returned[idx]),
                    "grade": int(grades[idx]),
                })

    positive_details = pd.DataFrame(positive_rows)

    # --- Retrieval top-k (popularity-biased) ---
    retrieval_top_k: Dict[str, np.ndarray] = {}
    retrieval_probs = popularity_pctl ** 0.8  # popularity bias
    retrieval_probs /= retrieval_probs.sum()

    for u in range(n_users):
        uid = f"u_{u:04d}"
        # Sample top-k with popularity bias (popular items more likely retrieved)
        retrieved = rng.choice(n_items, size=top_k, replace=False, p=retrieval_probs)
        retrieval_top_k[uid] = retrieved

    return {
        "item_meta": item_meta,
        "user_positives": user_positives,
        "retrieval_top_k": retrieval_top_k,
        "positive_details": positive_details,
    }


# ===================================================================
# Oracle Recall computation
# ===================================================================

def compute_oracle_recall(
    user_positives: Dict[str, List[int]],
    retrieval_top_k: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, float], Dict[str, List[int]], Dict[str, List[int]]]:
    """Compute oracle Recall@500 for held-out positives.

    Parameters
    ----------
    user_positives : dict
        user_id → list of positive item indices (Grade ≥ 3).
    retrieval_top_k : dict
        user_id → np.ndarray of retrieved item indices.

    Returns
    -------
    overall_recall : float
    per_user_recall : dict mapping user_id → recall value
    hits : dict mapping user_id → list of item indices that were found
    misses : dict mapping user_id → list of item indices that were missed
    """
    total_positives = 0
    total_hits = 0
    per_user_recall: Dict[str, float] = {}
    all_hits: Dict[str, List[int]] = {}
    all_misses: Dict[str, List[int]] = {}

    for uid, pos_items in user_positives.items():
        if not pos_items:
            continue
        retrieved_set = set(retrieval_top_k.get(uid, np.array([])).tolist())
        hits = [item for item in pos_items if item in retrieved_set]
        misses = [item for item in pos_items if item not in retrieved_set]
        recall = len(hits) / len(pos_items) if pos_items else 0.0

        per_user_recall[uid] = recall
        all_hits[uid] = hits
        all_misses[uid] = misses
        total_positives += len(pos_items)
        total_hits += len(hits)

    overall = total_hits / total_positives if total_positives > 0 else 0.0
    return overall, per_user_recall, all_hits, all_misses


def recall_by_popularity_quartile(
    positive_details: pd.DataFrame,
    item_meta: pd.DataFrame,
    user_positives: Dict[str, List[int]],
    retrieval_top_k: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Break down oracle recall by item popularity quartile.

    Parameters
    ----------
    positive_details : pd.DataFrame
        One row per Grade ≥ 3 interaction, with ``item_idx`` column.
    item_meta : pd.DataFrame
        Item metadata with ``pop_quartile`` column.
    user_positives, retrieval_top_k : dicts from oracle data.

    Returns
    -------
    pd.DataFrame with columns ``quartile``, ``n_positives``, ``n_hits``,
    ``recall``.
    """
    # Build a set of (user, item) hits
    hit_set: set = set()
    for uid, pos_items in user_positives.items():
        retrieved_set = set(retrieval_top_k.get(uid, np.array([])).tolist())
        for item in pos_items:
            if item in retrieved_set:
                hit_set.add((uid, item))

    # Map each positive to its quartile and check hit
    records: list[dict] = []
    for _, row in positive_details.iterrows():
        item_idx = int(row["item_idx"])
        uid = row["user_id"]
        quartile = item_meta.loc[item_idx, "pop_quartile"]
        is_hit = (uid, item_idx) in hit_set
        records.append({"quartile": quartile, "is_hit": is_hit})

    qdf = pd.DataFrame(records)
    summary = (
        qdf
        .groupby("quartile", observed=True)
        .agg(n_positives=("is_hit", "count"), n_hits=("is_hit", "sum"))
        .reset_index()
    )
    summary["recall"] = summary["n_hits"] / summary["n_positives"]
    return summary


# ===================================================================
# Gap analysis
# ===================================================================

def gap_analysis(
    positive_details: pd.DataFrame,
    item_meta: pd.DataFrame,
    user_positives: Dict[str, List[int]],
    retrieval_top_k: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Characterise the items that retrieval *misses*.

    Compares average features of hit vs. missed items to identify
    systematic blind spots (e.g., the retrieval model under-surfaces
    niche or older content).

    Returns
    -------
    pd.DataFrame with columns ``feature``, ``hit_mean``, ``miss_mean``,
    ``delta``, ``interpretation``.
    """
    hit_set: set = set()
    for uid, pos_items in user_positives.items():
        retrieved_set = set(retrieval_top_k.get(uid, np.array([])).tolist())
        for item in pos_items:
            if item in retrieved_set:
                hit_set.add((uid, item))

    positive_details = positive_details.copy()
    positive_details["is_hit"] = [
        (row["user_id"], int(row["item_idx"])) in hit_set
        for _, row in positive_details.iterrows()
    ]

    # Merge item metadata
    enriched = positive_details.merge(
        item_meta[["item_id", "popularity_pctl", "quality_score",
                   "age_days", "author_niche"]],
        on="item_id",
        how="left",
    )

    features = ["popularity_pctl", "quality_score", "age_days", "author_niche"]
    rows: list[dict] = []
    for feat in features:
        hit_mean = enriched.loc[enriched["is_hit"], feat].mean()
        miss_mean = enriched.loc[~enriched["is_hit"], feat].mean()
        delta = miss_mean - hit_mean

        # Interpretation
        if feat == "popularity_pctl":
            interp = "Missed items are less popular" if delta < 0 else "Missed items are more popular"
        elif feat == "quality_score":
            interp = "Missed items have lower quality" if delta < 0 else "Missed items have higher quality"
        elif feat == "age_days":
            interp = "Missed items are older" if delta > 0 else "Missed items are newer"
        elif feat == "author_niche":
            interp = "Missed items are more niche" if delta > 0 else "Missed items are more mainstream"
        else:
            interp = ""

        rows.append({
            "feature": feat,
            "hit_mean": round(hit_mean, 4),
            "miss_mean": round(miss_mean, 4),
            "delta": round(delta, 4),
            "interpretation": interp,
        })

    return pd.DataFrame(rows)


# ===================================================================
# Plotting
# ===================================================================

def plot_recall_by_quartile(
    quartile_df: pd.DataFrame,
    overall_recall: float,
    path: Path | None = None,
) -> Path:
    """Bar chart of oracle recall broken down by popularity quartile.

    Parameters
    ----------
    quartile_df : pd.DataFrame
        Output of ``recall_by_popularity_quartile``.
    overall_recall : float
        The global oracle recall (plotted as a horizontal line).
    path : Path, optional

    Returns
    -------
    Path — saved PNG location.
    """
    out = path or ORACLE_PNG
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(quartile_df))
    colours = ["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"]

    bars = ax.bar(x, quartile_df["recall"].values, color=colours[:len(x)],
                  edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, quartile_df["recall"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # Overall recall line
    ax.axhline(overall_recall, color="#333333", linestyle="--", linewidth=1.5,
               label=f"Overall Recall@500 = {overall_recall:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(quartile_df["quartile"].values, fontsize=11)
    ax.set_xlabel("Popularity Quartile", fontsize=12)
    ax.set_ylabel("Oracle Recall@500", fontsize=12)
    ax.set_title("Retrieval Oracle — Recall by Popularity Quartile",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return out


# ===================================================================
# Persistence
# ===================================================================

def save_oracle_results(
    quartile_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    overall_recall: float,
    path: Path | None = None,
) -> Path:
    """Save combined oracle analysis to Parquet."""
    out = path or ORACLE_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)

    # Store both tables + overall metric in a single Parquet with a section marker
    quartile_df = quartile_df.copy()
    quartile_df["section"] = "recall_by_quartile"
    gap_df_save = gap_df.copy()
    gap_df_save["section"] = "gap_analysis"

    # For compatibility, align columns
    combined = pd.concat([
        quartile_df.assign(**{c: None for c in gap_df_save.columns if c not in quartile_df.columns}),
        gap_df_save.assign(**{c: None for c in quartile_df.columns if c not in gap_df_save.columns}),
    ], ignore_index=True)
    combined["overall_recall_500"] = overall_recall
    combined.to_parquet(out, index=False)
    return out


# ===================================================================
# Main pipeline
# ===================================================================

def run_oracle_experiment(
    data: Dict | None = None,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Execute the full retrieval oracle experiment.

    Parameters
    ----------
    data : dict, optional
        Output of ``generate_oracle_data``.
    verbose : bool

    Returns
    -------
    (overall_recall, quartile_df, gap_df)
    """
    def _format_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False, floatfmt=".4f")
        except ImportError:
            return df.to_string(index=False)

    if data is None:
        data = generate_oracle_data()

    item_meta = data["item_meta"]
    user_pos = data["user_positives"]
    retrieval = data["retrieval_top_k"]
    pos_details = data["positive_details"]

    # 1. Overall recall
    overall, per_user, hits, misses = compute_oracle_recall(user_pos, retrieval)
    if verbose:
        print(f"[Oracle] Overall Recall@500: {overall:.4f}")
        print(f"[Oracle] Users evaluated: {len(per_user)}")
        total_pos = sum(len(v) for v in user_pos.values())
        total_hits = sum(len(v) for v in hits.values())
        print(f"[Oracle] Total positives: {total_pos} | "
              f"Hits: {total_hits} | "
              f"Misses: {total_pos - total_hits}")

    # 2. By popularity quartile
    quartile_df = recall_by_popularity_quartile(pos_details, item_meta, user_pos, retrieval)
    if verbose:
        print(f"\n{'=' * 60}")
        print("  Recall@500 by Popularity Quartile")
        print(f"{'=' * 60}")
        print(_format_table(quartile_df))

    # 3. Gap analysis
    gap_df = gap_analysis(pos_details, item_meta, user_pos, retrieval)
    if verbose:
        print(f"\n{'=' * 60}")
        print("  Gap Analysis: Hit vs. Missed Items")
        print(f"{'=' * 60}")
        print(_format_table(gap_df))

    # 4. Save
    pq_path = save_oracle_results(quartile_df, gap_df, overall)
    png_path = plot_recall_by_quartile(quartile_df, overall)
    if verbose:
        print(f"\n[OK] Oracle analysis -> {pq_path}")
        print(f"[OK] Recall chart    -> {png_path}")

    return overall, quartile_df, gap_df


# ===================================================================
# Standalone demo
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Narrative Intelligence Platform — Retrieval Oracle Experiment")
    print("=" * 70, "\n")

    overall, quartile_df, gap_df = run_oracle_experiment(verbose=True)

    # Per-user recall distribution
    data = generate_oracle_data()
    _, per_user, _, _ = compute_oracle_recall(
        data["user_positives"], data["retrieval_top_k"]
    )
    recalls = np.array(list(per_user.values()))
    print(f"\n{'=' * 70}")
    print("  Per-User Recall@500 Distribution")
    print("=" * 70)
    print(f"  Mean:   {recalls.mean():.4f}")
    print(f"  Median: {np.median(recalls):.4f}")
    print(f"  Std:    {recalls.std():.4f}")
    print(f"  Min:    {recalls.min():.4f}")
    print(f"  Max:    {recalls.max():.4f}")

    # Users with 0 recall
    zero_recall = (recalls == 0).sum()
    print(f"  Users with 0 recall: {zero_recall} / {len(recalls)} "
          f"({zero_recall / len(recalls) * 100:.1f}%)")

    print("\n✅ Retrieval oracle experiment complete.")
