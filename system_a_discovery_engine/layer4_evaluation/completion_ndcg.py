"""
Narrative Intelligence Platform — System A, Layer 4
====================================================
Completion-Weighted NDCG
-------------------------

**Motivation**:
Standard binary NDCG treats every positive interaction equally — an item
a user opened for 30 seconds scores the same as one they devoured cover
to cover.  Completion-Weighted NDCG (CW-NDCG) replaces binary relevance
with a graded label that captures *depth* of engagement.

**Graded relevance scheme**:

  ======  ===========================================================
  Grade   Criteria
  ======  ===========================================================
  0       Opened, but final_completion_pct < 10 %
  1       10 % ≤ completion < 50 %
  2       50 % ≤ completion < 90 %
  3       completion ≥ 90 %, single session (fully read)
  4       completion ≥ 90 % AND user returned within 7 days (love it)
  ======  ===========================================================

**Functions**:
  - ``assign_relevance_grade`` — vectorised label assignment
  - ``dcg_at_k``              — Discounted Cumulative Gain
  - ``ndcg_at_k``             — Normalised DCG for one ranking
  - ``compute_completion_weighted_ndcg`` — aggregate over all users

**Design decision**:
Both CW-NDCG and binary NDCG are computed side-by-side.  Binary NDCG
uses a threshold of Grade ≥ 3 (i.e., effectively treating 90 %+
completions as positives).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Relevance grading
# ===================================================================

def assign_relevance_grade(
    final_completion_pct: float,
    returned_within_7d: bool = False,
) -> int:
    """Map a single reading outcome to a graded relevance label.

    Parameters
    ----------
    final_completion_pct : float
        Fraction of the story read (0 – 1).
    returned_within_7d : bool
        Whether the user returned within 7 days.

    Returns
    -------
    int — relevance grade ∈ {0, 1, 2, 3, 4}.
    """
    if final_completion_pct < 0.10:
        return 0
    elif final_completion_pct < 0.50:
        return 1
    elif final_completion_pct < 0.90:
        return 2
    elif returned_within_7d:
        return 4
    else:
        return 3


def assign_relevance_grades_vec(
    completion_pct: np.ndarray,
    returned: np.ndarray,
) -> np.ndarray:
    """Vectorised version of ``assign_relevance_grade``.

    Parameters
    ----------
    completion_pct : np.ndarray, shape (N,)
        Completion percentages in [0, 1].
    returned : np.ndarray, shape (N,)
        Boolean array — True if user returned within 7 days.

    Returns
    -------
    np.ndarray of ints, shape (N,).
    """
    grades = np.zeros(len(completion_pct), dtype=np.int32)
    grades[completion_pct >= 0.10] = 1
    grades[completion_pct >= 0.50] = 2
    grades[completion_pct >= 0.90] = 3
    grades[(completion_pct >= 0.90) & returned] = 4
    return grades


# ===================================================================
# DCG / NDCG core
# ===================================================================

def dcg_at_k(relevance_scores: np.ndarray, k: int = 10) -> float:
    """Compute Discounted Cumulative Gain @ k.

    Uses the exponential gain formulation:
        DCG@k = Σ_{i=1}^{k}  (2^{rel_i} − 1) / log₂(i + 1)

    Parameters
    ----------
    relevance_scores : np.ndarray
        Relevance labels in ranked order (position 0 = rank 1).
    k : int
        Cutoff rank.

    Returns
    -------
    float — DCG value.
    """
    rel = relevance_scores[:k].astype(np.float64)
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2.0 ** rel - 1.0) / discounts))


def ndcg_at_k(
    predicted_ranking: np.ndarray,
    true_relevances: np.ndarray,
    k: int = 10,
) -> float:
    """Compute NDCG@k for a single query / user.

    Parameters
    ----------
    predicted_ranking : np.ndarray
        Predicted scores — higher ⇒ ranked earlier.
    true_relevances : np.ndarray
        Ground-truth graded relevance labels (aligned with
        ``predicted_ranking``).
    k : int
        Cutoff rank.

    Returns
    -------
    float — NDCG ∈ [0, 1].  Returns 0 if IDCG is zero (no relevant
            items in the ranking).
    """
    # Sort items by predicted score (descending)
    order = np.argsort(-predicted_ranking)
    sorted_rel = true_relevances[order]
    dcg = dcg_at_k(sorted_rel, k)

    # Ideal: sort by true relevance descending
    ideal_rel = np.sort(true_relevances)[::-1]
    idcg = dcg_at_k(ideal_rel, k)

    return float(dcg / idcg) if idcg > 0 else 0.0


# ===================================================================
# Aggregate metrics
# ===================================================================

def compute_completion_weighted_ndcg(
    user_rankings: Dict[str, Dict[str, np.ndarray]],
    k: int = 10,
    binary_threshold: int = 3,
) -> Tuple[float, float]:
    """Compute mean CW-NDCG and mean binary NDCG across all users.

    Parameters
    ----------
    user_rankings : dict
        Mapping of ``user_id`` → dict with keys:
          - ``"predicted_scores"`` : np.ndarray of predicted ranking scores
          - ``"true_relevances"``  : np.ndarray of graded relevance labels
    k : int
        Cutoff rank.
    binary_threshold : int
        Grade threshold for converting graded → binary relevance
        (default 3, i.e., Grade ≥ 3 is "positive").

    Returns
    -------
    (mean_cw_ndcg, mean_binary_ndcg) : Tuple[float, float]
    """
    cw_ndcgs: List[float] = []
    bin_ndcgs: List[float] = []

    for uid, data in user_rankings.items():
        preds = data["predicted_scores"]
        rels = data["true_relevances"]

        # CW-NDCG: use graded relevance as-is
        cw = ndcg_at_k(preds, rels, k)
        cw_ndcgs.append(cw)

        # Binary NDCG: binarise at threshold
        binary_rels = (rels >= binary_threshold).astype(np.float64)
        bn = ndcg_at_k(preds, binary_rels, k)
        bin_ndcgs.append(bn)

    mean_cw = float(np.mean(cw_ndcgs)) if cw_ndcgs else 0.0
    mean_bn = float(np.mean(bin_ndcgs)) if bin_ndcgs else 0.0
    return mean_cw, mean_bn


def compute_ndcg_from_dataframe(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "predicted_score",
    relevance_col: str = "relevance",
    k: int = 10,
    binary_threshold: int = 3,
) -> Tuple[float, float]:
    """Convenience wrapper: compute CW-NDCG and binary NDCG from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``user_col``, ``score_col``, ``relevance_col``.
    k : int
    binary_threshold : int

    Returns
    -------
    (mean_cw_ndcg, mean_binary_ndcg)
    """
    rankings: Dict[str, Dict[str, np.ndarray]] = {}
    for uid, grp in df.groupby(user_col):
        rankings[str(uid)] = {
            "predicted_scores": grp[score_col].values,
            "true_relevances": grp[relevance_col].values,
        }
    return compute_completion_weighted_ndcg(rankings, k, binary_threshold)


# ===================================================================
# Synthetic data generation
# ===================================================================

def generate_synthetic_rankings(
    n_users: int = 200,
    items_per_user: int = 25,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create synthetic user rankings for demo / testing.

    A latent quality signal controls both the predicted score and the
    true relevance, with noise to create realistic imperfection.

    Returns
    -------
    dict  — format expected by ``compute_completion_weighted_ndcg``.
    """
    rng = np.random.default_rng(seed)
    rankings: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(n_users):
        latent = rng.uniform(0, 1, items_per_user)
        # Predicted score = latent + noise
        preds = latent + rng.normal(0, 0.2, items_per_user)
        # True relevance = graded from latent (mapped to 0–4)
        completion = latent + rng.normal(0, 0.1, items_per_user)
        completion = np.clip(completion, 0, 1)
        returned = rng.random(items_per_user) < 0.3
        rels = assign_relevance_grades_vec(completion, returned)

        rankings[f"u_{i:04d}"] = {
            "predicted_scores": preds,
            "true_relevances": rels,
        }

    return rankings


# ===================================================================
# Standalone demo
# ===================================================================
def main() -> pd.DataFrame:
    """Run completion-weighted NDCG evaluation and save a metric table."""
    rankings = generate_synthetic_rankings()
    k_values = [5, 10, 20]
    results: list[dict] = []

    for k in k_values:
        cw, bn = compute_completion_weighted_ndcg(rankings, k=k)
        results.append({"k": k, "CW_NDCG": cw, "Binary_NDCG": bn, "delta": cw - bn})

    results_df = pd.DataFrame(results)
    output_path = _PROJECT_ROOT / "data" / "processed" / "completion_ndcg_metrics.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    print(f"[OK] Completion-weighted NDCG metrics saved to {output_path}")
    return results_df


if __name__ == "__main__":
    print("=" * 70)
    print("  Narrative Intelligence Platform — Completion-Weighted NDCG")
    print("=" * 70, "\n")

    rankings = generate_synthetic_rankings()

    # Compute at multiple k values
    k_values = [5, 10, 20]
    results: list[dict] = []

    for k in k_values:
        cw, bn = compute_completion_weighted_ndcg(rankings, k=k)
        results.append({"k": k, "CW-NDCG": cw, "Binary-NDCG": bn, "Δ": cw - bn})

    results_df = pd.DataFrame(results)
    print("Metric comparison table:")
    print("-" * 50)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

    # Grade distribution
    all_rels = np.concatenate([r["true_relevances"] for r in rankings.values()])
    print(f"\nRelevance grade distribution (N={len(all_rels)}):")
    for g in range(5):
        count = (all_rels == g).sum()
        pct = count / len(all_rels) * 100
        print(f"  Grade {g}: {count:5d} ({pct:5.1f}%)")

    # Per-user NDCG distribution at k=10
    per_user_cw: list[float] = []
    per_user_bn: list[float] = []
    for uid, data in rankings.items():
        per_user_cw.append(ndcg_at_k(data["predicted_scores"],
                                     data["true_relevances"], k=10))
        binary_rels = (data["true_relevances"] >= 3).astype(float)
        per_user_bn.append(ndcg_at_k(data["predicted_scores"],
                                     binary_rels, k=10))

    print(f"\nPer-user NDCG@10 statistics:")
    print(f"  CW-NDCG  — mean={np.mean(per_user_cw):.4f}, "
          f"std={np.std(per_user_cw):.4f}, "
          f"median={np.median(per_user_cw):.4f}")
    print(f"  Bin-NDCG — mean={np.mean(per_user_bn):.4f}, "
          f"std={np.std(per_user_bn):.4f}, "
          f"median={np.median(per_user_bn):.4f}")

    print("\n✅ Completion-Weighted NDCG evaluation complete.")
