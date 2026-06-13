"""
Narrative Intelligence Platform — System A, Layer 3
====================================================
LambdaMART Re-Ranker
---------------------

**Algorithm Overview**:
A Learning-to-Rank (LTR) model that re-orders candidate items retrieved
by the Two-Tower retrieval layer.  We use LightGBM's ``LGBMRanker`` with
the *LambdaMART* objective, which directly optimises NDCG via pairwise
lambda gradients.

**Feature set** (8 features per user × candidate_item pair):

  ===  ===========================  =====================================
  #    Feature                      Intuition
  ===  ===========================  =====================================
  1    retrieval_score              Two-Tower cosine dot product
  2    hazard_score                 Dropout hazard (0 for new items)
  3    quality_score                Quality from item fingerprint
  4    engagement_fit_score         cos-sim(user engagement, item avg)
  5    author_affinity              cos-sim(user author history, item)
  6    recency_decay                exp(-age_days / 30)
  7    genre_match_score            cos-sim(user topic, item topic)
  8    novelty_score                -log2(item_popularity_pctl + 0.01)
  9    tail_boost_score             controlled boost for under-popular items
  ===  ===========================  =====================================

**Training**:
  - Group-by ``user_id`` (query-id)
  - Graded relevance labels 0–4 from a completion-weighted scheme
  - 80 / 20 train / test split by user
  - Evaluation via NDCG@10

**Outputs**:
  - ``data/processed/lambdamart_model.txt``  (LightGBM text dump)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"
MODEL_PATH = OUTPUT_DIR / "lambdamart_model.txt"

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
RANKING_FEATURES: list[str] = [
    "retrieval_score",
    "hazard_score",
    "quality_score",
    "engagement_fit_score",
    "author_affinity",
    "recency_decay",
    "genre_match_score",
    "novelty_score",
    "tail_boost_score",
]


# ===================================================================
# Relevance labelling  (completion-weighted, grades 0–4)
# ===================================================================

def assign_relevance_grade(
    final_completion_pct: float,
    returned_within_7d: bool = False,
) -> int:
    """Map a reading-completion outcome to a graded relevance label.

    Parameters
    ----------
    final_completion_pct : float
        Fraction of the story the user actually read (0-1).
    returned_within_7d : bool
        Whether the user returned to read more within 7 days.

    Returns
    -------
    int  — relevance grade in {0, 1, 2, 3, 4}.

    Grading scheme
    --------------
    - 0 : opened but < 10 % read
    - 1 : 10 – 50 % read
    - 2 : 50 – 90 % read
    - 3 : ≥ 90 % read in a single session
    - 4 : ≥ 90 % read AND returned within 7 days (strongest signal)
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


# ===================================================================
# Data preparation
# ===================================================================

def build_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the ranking feature columns are present and numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``RANKING_FEATURES`` + ``user_id`` + ``relevance``.

    Returns
    -------
    pd.DataFrame  — sorted by user_id for proper group construction.
    """
    df = df.copy()
    if "tail_boost_score" not in df.columns:
        df["tail_boost_score"] = 0.0
    required = set(RANKING_FEATURES) | {"user_id", "relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for feature in RANKING_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    df = df.sort_values("user_id").reset_index(drop=True)
    return df


def apply_reranker_controls(
    df: pd.DataFrame,
    tail_boost_weight: float = 0.0,
    survival_penalty_weight: float = 1.0,
    max_hazard_score: float = 0.5,
) -> pd.DataFrame:
    """Apply conservative reranker controls before LambdaMART training.

    ``survival_penalty_weight`` lets us reduce hazard influence when survival
    risk overpowers relevance. ``tail_boost_weight`` adds a bounded signal for
    under-popular candidates, using ``tail_boost_score`` when present or deriving
    it from ``novelty_score`` as a fallback.
    """
    out = df.copy()
    if "hazard_score" in out.columns:
        out["hazard_score"] = (
            pd.to_numeric(out["hazard_score"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0, upper=max_hazard_score)
            * survival_penalty_weight
        )
    if "tail_boost_score" not in out.columns:
        if "novelty_score" in out.columns:
            novelty = pd.to_numeric(out["novelty_score"], errors="coerce").fillna(0.0)
            threshold = novelty.quantile(0.75)
            out["tail_boost_score"] = (novelty >= threshold).astype(float)
        else:
            out["tail_boost_score"] = 0.0
    out["tail_boost_score"] = (
        pd.to_numeric(out["tail_boost_score"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        * tail_boost_weight
    )
    return out


def make_group_array(df: pd.DataFrame, qid_col: str = "user_id") -> np.ndarray:
    """Create the *group* array expected by LGBMRanker.

    Each element is the count of candidates for a given query (user).

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by ``qid_col``.
    qid_col : str
        Query-id column name.

    Returns
    -------
    np.ndarray of ints, shape ``(n_queries,)``
    """
    return df.groupby(qid_col).size().values


def split_by_user(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 train/test split, stratified by user.

    Ensures that all candidates for a user land in the same fold
    (no information leakage across queries).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                           random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["user_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ===================================================================
# Model training & evaluation
# ===================================================================

def train_lambdamart(
    train_df: pd.DataFrame,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    verbose: int = -1,
) -> lgb.LGBMRanker:
    """Train a LambdaMART ranker with LightGBM.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with ``RANKING_FEATURES``, ``relevance``, ``user_id``.
    n_estimators, learning_rate, num_leaves : hyper-parameters.
    verbose : int
        LightGBM verbosity.

    Returns
    -------
    lgb.LGBMRanker  — fitted model.
    """
    train_df = train_df.sort_values("user_id").reset_index(drop=True)
    X_train = train_df[RANKING_FEATURES].values
    y_train = train_df["relevance"].values.astype(np.float64)
    groups_train = make_group_array(train_df)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        importance_type="gain",
        random_state=42,
        verbose=verbose,
    )
    ranker.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_train, y_train)],
        eval_group=[groups_train],
        eval_at=[10],
    )
    return ranker


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """Compute NDCG@k for a single query.

    Parameters
    ----------
    y_true : np.ndarray  — ground-truth relevance labels.
    y_score : np.ndarray — predicted scores (higher = better rank).
    k : int

    Returns
    -------
    float — NDCG value in [0, 1].
    """
    order = np.argsort(-y_score)[:k]
    dcg = np.sum((2.0 ** y_true[order] - 1) / np.log2(np.arange(2, k + 2)[:len(order)]))
    ideal_order = np.argsort(-y_true)[:k]
    idcg = np.sum((2.0 ** y_true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)[:len(ideal_order)]))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_ranker(
    ranker: lgb.LGBMRanker,
    test_df: pd.DataFrame,
    k: int = 10,
) -> float:
    """Evaluate the ranker on held-out test data and return mean NDCG@k.

    Parameters
    ----------
    ranker : lgb.LGBMRanker
    test_df : pd.DataFrame
    k : int

    Returns
    -------
    float — mean NDCG@k across all test queries.
    """
    test_df = test_df.sort_values("user_id").reset_index(drop=True)
    X_test = test_df[RANKING_FEATURES].values
    y_test = test_df["relevance"].values.astype(np.float64)
    preds = ranker.predict(X_test)

    ndcgs: list[float] = []
    start = 0
    for _, grp in test_df.groupby("user_id", sort=True):
        n = len(grp)
        ndcgs.append(ndcg_at_k(y_test[start:start + n], preds[start:start + n], k))
        start += n

    return float(np.mean(ndcgs))


# ===================================================================
# Model persistence
# ===================================================================

def save_ranker(ranker: lgb.LGBMRanker, path: Path | None = None) -> Path:
    """Save the LGBMRanker model as a LightGBM text file."""
    out = path or MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(out))
    return out


def load_ranker(path: Path | None = None) -> lgb.Booster:
    """Load a saved LightGBM model from text file."""
    return lgb.Booster(model_file=str(path or MODEL_PATH))


# ===================================================================
# Feature importance
# ===================================================================

def feature_importance_table(ranker: lgb.LGBMRanker) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances (gain-based)."""
    imp = ranker.feature_importances_
    return (
        pd.DataFrame({"feature": RANKING_FEATURES, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ===================================================================
# Synthetic data generation
# ===================================================================

def generate_synthetic_ranking_data(
    n_users: int = 300,
    candidates_per_user: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Produce a synthetic ranking dataset for demo / testing.

    Each user gets ``candidates_per_user`` items with randomly generated
    features.  Relevance labels are stochastically correlated with
    quality, genre-match, and author-affinity to simulate a realistic
    signal.

    Parameters
    ----------
    n_users : int
    candidates_per_user : int
    seed : int

    Returns
    -------
    pd.DataFrame — columns: ``user_id``, ``item_id``, ``relevance``,
                   + ``RANKING_FEATURES``.
    """
    rng = np.random.default_rng(seed)
    n_total = n_users * candidates_per_user

    retrieval_score = rng.uniform(0.0, 1.0, n_total)
    hazard_score = rng.uniform(0.0, 0.5, n_total)
    quality_score = rng.uniform(0.0, 1.0, n_total)
    engagement_fit = rng.uniform(-0.3, 1.0, n_total).clip(0, 1)
    author_affinity = rng.uniform(-0.2, 1.0, n_total).clip(0, 1)
    age_days = rng.exponential(60, n_total)
    recency_decay = np.exp(-age_days / 30.0)
    genre_match = rng.uniform(0.0, 1.0, n_total)
    popularity_pctl = rng.uniform(0.0, 1.0, n_total)
    novelty_score = -np.log2(popularity_pctl + 0.01)
    tail_boost_score = (popularity_pctl <= 0.25).astype(float)

    # Synthetic relevance — probabilistically correlated with features
    latent_quality = (
        0.3 * quality_score
        + 0.25 * genre_match
        + 0.2 * author_affinity
        + 0.1 * engagement_fit
        + 0.1 * retrieval_score
        - 0.3 * hazard_score
        + rng.normal(0, 0.15, n_total)
    )
    # Map to grades 0–4
    thresholds = np.percentile(latent_quality, [40, 60, 80, 95])
    relevance = np.digitize(latent_quality, thresholds).astype(int)

    user_ids = np.repeat([f"u_{i:04d}" for i in range(n_users)], candidates_per_user)
    item_ids = np.array([f"i_{j:05d}" for j in range(n_total)])

    return pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "retrieval_score": retrieval_score,
        "hazard_score": hazard_score,
        "quality_score": quality_score,
        "engagement_fit_score": engagement_fit,
        "author_affinity": author_affinity,
        "recency_decay": recency_decay,
        "genre_match_score": genre_match,
        "novelty_score": novelty_score,
        "tail_boost_score": tail_boost_score,
        "relevance": relevance,
    })


# ===================================================================
# Main pipeline
# ===================================================================

def run_ranking_pipeline(
    df: pd.DataFrame | None = None,
    tail_boost_weight: float = 0.10,
    survival_penalty_weight: float = 0.75,
    max_hazard_score: float = 0.35,
    verbose: bool = True,
) -> Tuple[lgb.LGBMRanker, float]:
    """End-to-end LambdaMART training and evaluation.

    1. Build / validate features → 2. Split 80/20 by user
    → 3. Train LGBMRanker → 4. Evaluate NDCG@10 → 5. Save model.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Ranking data.  If *None*, synthetic data is generated.
    verbose : bool

    Returns
    -------
    (ranker, ndcg10) : Tuple[lgb.LGBMRanker, float]
    """
    if df is None:
        df = generate_synthetic_ranking_data()
    df = apply_reranker_controls(
        df,
        tail_boost_weight=tail_boost_weight,
        survival_penalty_weight=survival_penalty_weight,
        max_hazard_score=max_hazard_score,
    )
    df = build_ranking_features(df)

    train_df, test_df = split_by_user(df)
    if verbose:
        print(f"[LambdaMART] Train users: "
              f"{train_df['user_id'].nunique()}, "
              f"rows: {len(train_df)}")
        print(f"[LambdaMART] Test  users: "
              f"{test_df['user_id'].nunique()}, "
              f"rows: {len(test_df)}")

    ranker = train_lambdamart(train_df)
    ndcg10 = evaluate_ranker(ranker, test_df, k=10)
    if verbose:
        print(f"\n[LambdaMART] NDCG@10 on test set: {ndcg10:.4f}")
        print(f"\nFeature importances (gain):")
        print(feature_importance_table(ranker).to_string(index=False))

    model_path = save_ranker(ranker)
    if verbose:
        print(f"\n✓ Model saved → {model_path}")

    return ranker, ndcg10


# ===================================================================
# Standalone demo
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Narrative Intelligence Platform — LambdaMART Re-Ranker")
    print("=" * 70, "\n")

    ranker, ndcg10 = run_ranking_pipeline(verbose=True)

    # Quick sanity: re-rank a small batch
    test_batch = generate_synthetic_ranking_data(n_users=5, candidates_per_user=20, seed=99)
    scores = ranker.predict(test_batch[RANKING_FEATURES].values)
    test_batch["predicted_rank_score"] = scores
    top5 = (
        test_batch
        .sort_values(["user_id", "predicted_rank_score"], ascending=[True, False])
        .groupby("user_id")
        .head(5)
    )
    print(f"\n{'=' * 70}")
    print("  Top-5 predictions for 5 sample users")
    print("=" * 70)
    print(top5[["user_id", "item_id", "relevance", "predicted_rank_score"]]
          .to_string(index=False))

    print("\n✅ LambdaMART re-ranker pipeline complete.")
