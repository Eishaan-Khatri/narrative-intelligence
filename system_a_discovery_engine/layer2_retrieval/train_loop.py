"""
Narrative Intelligence Platform — System A, Layer 2
====================================================
Training Loop with Hard-Negative Mining
----------------------------------------

**Two-phase training schedule**

Phase 1 — In-batch negatives (epochs 1–5)
    For every mini-batch of B positive (user, item) pairs, every *other*
    item in the batch serves as a negative for each user, yielding B×(B−1)
    implicit negatives.  This is cheap but produces only "easy" negatives
    on average.

Phase 2 — Hard-negative mining (epochs 6–15)
    Every 2 epochs we refresh a hard-negative pool: for each training user
    we score 5 000 random catalog items through the current item tower,
    take the top-200 *non-interacted* items as the hard pool, and sample 5
    per positive for the next training interval.  These semi-hard negatives
    force the towers to learn finer-grained distinctions.

**Metrics tracked per epoch**
    * Training BPR loss (with popularity correction).
    * Recall@50 and Recall@500 on a held-out validation split.

**Checkpointing**
    The model with the best validation Recall@50 is saved to
    ``data/processed/two_tower_model.pt``.

**Outputs**
    * ``data/processed/two_tower_model.pt``    — best model checkpoint.
    * ``data/processed/item_embeddings.parquet`` — (N_items, 128) embeddings.
    * ``data/processed/user_embeddings.parquet`` — (N_users, 128) embeddings.
    * ``data/processed/training_curves.png``     — loss + recall plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the import path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_store.schema import (
    ENGAGEMENT_PROFILE_DIM,
    ITEM_EMBEDDING_DIM,
    ITEM_FINGERPRINT_DIM,
    ITEM_TOWER_INPUT_DIM,
    USER_EMBEDDING_DIM,
    USER_TOWER_CONTEXT_DIM,
    USER_TOWER_INPUT_DIM,
)
from system_a_discovery_engine.layer2_retrieval.two_tower_model import (
    RecommendationDataset,
    TwoTowerModel,
    bpr_loss,
    device,
)
from system_a_discovery_engine.layer2_retrieval.real_data_features import (
    build_real_training_arrays,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-5
PHASE1_EPOCHS = 5
TOTAL_EPOCHS = 15
HARD_NEG_REFRESH_INTERVAL = 2  # refresh every N epochs during Phase 2
HARD_NEG_CATALOG_SAMPLE = 5_000
HARD_NEG_POOL_SIZE = 200
HARD_NEG_SAMPLES_PER_POS = 5
RETRIEVAL_METRICS_PATH = DATA_DIR / "retrieval_metrics.parquet"


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_users: int = 2_000,
    n_items: int = 500,
    interactions_per_user: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data with latent preference structure.

    We plant a low-rank structure: users and items each have a latent vector
    in R^8, and user–item affinity is their dot product.  Positive items are
    sampled proportionally to this affinity so the model has something
    learnable, while popularity follows a power-law.

    Returns:
        user_features:      (N_interactions, 146) float32
        item_features:      (N_interactions, 83)  float32
        log_pop:            (N_interactions,)      float32
        item_catalog_feats: (N_items, 83)          float32  — full catalog
        item_ids:           (N_interactions,)      int      — item index per row
    """
    rng = np.random.RandomState(seed)
    latent_dim = 8

    # Latent factors
    user_latent = rng.randn(n_users, latent_dim).astype(np.float32)
    item_latent = rng.randn(n_items, latent_dim).astype(np.float32)
    affinity = user_latent @ item_latent.T  # (n_users, n_items)

    # Power-law popularity counts
    popularity = (rng.pareto(a=1.5, size=n_items) * 10 + 1).astype(np.float32)
    log_popularity = np.log1p(popularity).astype(np.float32)

    # Item catalog features (83-dim: fingerprint + log_interactions + velocity)
    item_fingerprints = rng.randn(n_items, ITEM_FINGERPRINT_DIM).astype(np.float32)
    item_log_interact = np.log1p(popularity).reshape(-1, 1)
    item_velocity = rng.rand(n_items, 1).astype(np.float32)
    item_catalog_feats = np.hstack([item_fingerprints, item_log_interact, item_velocity])

    # Sample interactions
    all_user_feats: list[np.ndarray] = []
    all_item_feats: list[np.ndarray] = []
    all_log_pop: list[float] = []
    all_item_ids: list[int] = []

    for u in range(n_users):
        # Softmax over affinity → sampling distribution
        probs = np.exp(affinity[u] - affinity[u].max())
        probs /= probs.sum()
        chosen = rng.choice(n_items, size=interactions_per_user, replace=False, p=probs)

        # User feature: mean-pool chosen item embeddings → pad to 128, plus
        # engagement (8) + context (10)
        mean_pool = item_fingerprints[chosen].mean(axis=0)  # 81-dim
        # Pad/project to 128 for the mean-pooled item embedding slot
        mean_pool_128 = np.zeros(128, dtype=np.float32)
        mean_pool_128[: len(mean_pool)] = mean_pool

        engagement = rng.rand(ENGAGEMENT_PROFILE_DIM).astype(np.float32)
        context = rng.rand(USER_TOWER_CONTEXT_DIM).astype(np.float32)
        user_vec = np.concatenate([mean_pool_128, engagement, context])

        for item_idx in chosen:
            all_user_feats.append(user_vec)
            all_item_feats.append(item_catalog_feats[item_idx])
            all_log_pop.append(log_popularity[item_idx])
            all_item_ids.append(item_idx)

    user_features = np.stack(all_user_feats)
    item_features = np.stack(all_item_feats)
    log_pop_arr = np.array(all_log_pop, dtype=np.float32)
    item_ids_arr = np.array(all_item_ids, dtype=np.int64)

    return user_features, item_features, log_pop_arr, item_catalog_feats, item_ids_arr


def try_load_real_data() -> Optional[Tuple[np.ndarray, ...]]:
    """Attempt to load real session + fingerprint data from Parquet files.

    Returns None if the files don't exist or are incomplete.
    """
    session_path = DATA_DIR / "session_features.parquet"
    fp_path = DATA_DIR / "item_fingerprints.parquet"
    temporal_path = DATA_DIR / "user_temporal_features.parquet"
    if not session_path.exists() or not fp_path.exists():
        return None

    try:
        real = build_real_training_arrays(
            session_path=session_path,
            fingerprint_path=fp_path,
            temporal_path=temporal_path if temporal_path.exists() else None,
        )
        user_features, _item_features, _log_pop, item_catalog, _item_ids = real
        print(
            f"[INFO] Loaded real retrieval data: "
            f"{user_features.shape[0]} interactions, {item_catalog.shape[0]} items."
        )
        return real
    except Exception as exc:
        warnings.warn(f"Could not load real data: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# In-batch negative BPR step
# ═══════════════════════════════════════════════════════════════════════════

def inbatch_bpr_step(
    model: TwoTowerModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One training step using in-batch negatives.

    For a batch of B positives, each user is paired with every *other*
    positive item as a negative.  We compute the BPR loss over the full
    B × (B−1) matrix and take the mean.

    Args:
        model:     The TwoTowerModel (already on ``device``).
        batch:     Dict from the DataLoader.
        optimizer: The Adam optimiser.

    Returns:
        Scalar loss value (float).
    """
    user_feat = batch["user_features"].to(device)
    pos_feat = batch["pos_item_features"].to(device)
    pos_pop = batch["pos_log_pop"].to(device)

    user_emb, pos_emb = model(user_feat, pos_feat)
    B = user_emb.shape[0]
    if B < 2:
        return 0.0

    # All-pairs score matrix: (B, B) — score[i, j] = user_i · item_j
    score_matrix = torch.mm(user_emb, pos_emb.t())  # (B, B)

    # Positive scores: diagonal
    pos_scores = score_matrix.diag() - pos_pop  # (B,)

    # For each user i, negatives are all items j ≠ i
    # Expand pos_scores to (B, 1) and subtract from full matrix
    neg_scores = score_matrix - pos_pop.unsqueeze(0)  # broadcast pop per item column

    # Mask diagonal (positive pairs)
    mask = ~torch.eye(B, dtype=torch.bool, device=device)
    neg_scores_flat = neg_scores[mask].view(B, B - 1)  # (B, B-1)

    # BPR: for each user, loss over all negatives
    pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores_flat)  # (B, B-1)
    diff = pos_expanded - neg_scores_flat
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# Hard-negative BPR step
# ═══════════════════════════════════════════════════════════════════════════

def hardneg_bpr_step(
    model: TwoTowerModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_weight: float = 1.0,
) -> float:
    """One training step using pre-mined hard negatives.

    Args:
        model:     The TwoTowerModel.
        batch:     Dict with ``neg_item_features`` and ``neg_log_pop`` present.
        optimizer: The Adam optimiser.

    Returns:
        Scalar loss value.
    """
    user_feat = batch["user_features"].to(device)
    pos_feat = batch["pos_item_features"].to(device)
    neg_feat = batch["neg_item_features"].to(device)
    pos_pop = batch["pos_log_pop"].to(device)
    neg_pop = batch["neg_log_pop"].to(device)

    user_emb, pos_emb = model(user_feat, pos_feat)
    neg_emb = model.item_tower(neg_feat)

    loss = bpr_loss(user_emb, pos_emb, neg_emb, pos_pop, neg_pop) * loss_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# Hard-negative mining
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mine_hard_negatives(
    model: TwoTowerModel,
    dataset: RecommendationDataset,
    item_catalog_feats: np.ndarray,
    item_ids_per_sample: np.ndarray,
    catalog_sample_size: int = HARD_NEG_CATALOG_SAMPLE,
    pool_size: int = HARD_NEG_POOL_SIZE,
    samples_per_pos: int = HARD_NEG_SAMPLES_PER_POS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mine hard negatives for each training sample.

    For each unique user in the dataset, we:
    1. Sample ``catalog_sample_size`` random items from the catalog.
    2. Score them with the current item tower + the user's embedding.
    3. Remove items the user has already interacted with.
    4. Keep the top-``pool_size`` highest-scoring non-interacted items.
    5. Randomly sample ``samples_per_pos`` per positive interaction.

    Since each user may have multiple positive interactions we replicate the
    hard-negative for each one (randomly choosing from the pool).

    Args:
        model:              Current TwoTowerModel.
        dataset:            The training dataset.
        item_catalog_feats: (N_items, 83) full catalog item features.
        item_ids_per_sample: (N_samples,) item index for each training sample.
        catalog_sample_size: Number of random catalog items to score.
        pool_size:          Top-K non-interacted items to keep.
        samples_per_pos:    How many hard negatives to draw per positive.
        seed:               Random seed for reproducibility.

    Returns:
        neg_item_features: (N_samples, 83)
        neg_log_pop:       (N_samples,)
    """
    rng = np.random.RandomState(seed)
    model.eval()
    n_catalog = item_catalog_feats.shape[0]
    n_samples = len(dataset)

    # Pre-compute all item embeddings for the catalog subset
    catalog_tensor = torch.as_tensor(item_catalog_feats, dtype=torch.float32, device=device)

    neg_feats = np.zeros((n_samples, ITEM_TOWER_INPUT_DIM), dtype=np.float32)
    neg_pops = np.zeros(n_samples, dtype=np.float32)

    # Group samples by user (indices into dataset)
    # For synthetic data, user features are identical within a user
    # We use a simple hash to group — but in practice user_id would be stored
    sample_user_feats = dataset.user_features.numpy()

    # Build user → sample_indices mapping via hashing
    user_hashes: Dict[int, List[int]] = {}
    for idx in range(n_samples):
        h = hash(sample_user_feats[idx].data.tobytes())
        user_hashes.setdefault(h, []).append(idx)

    for _h, sample_indices in tqdm(
        user_hashes.items(), desc="Mining hard negatives", leave=False,
    ):
        # Get user embedding (same for all samples of this user)
        u_feat = dataset.user_features[sample_indices[0]].unsqueeze(0).to(device)
        u_emb = model.user_tower(u_feat)  # (1, 128)

        # Interacted item set for this user
        interacted = set(item_ids_per_sample[sample_indices].tolist())

        # Sample catalog items
        sample_size = min(catalog_sample_size, n_catalog)
        cand_indices = rng.choice(n_catalog, size=sample_size, replace=False)

        # Score candidates
        cand_feats = catalog_tensor[cand_indices]  # (sample_size, 83)
        cand_emb = model.item_tower(cand_feats)  # (sample_size, 128)
        scores = torch.mm(u_emb, cand_emb.t()).squeeze(0)  # (sample_size,)

        # Remove interacted
        mask = torch.tensor(
            [cand_indices[i] not in interacted for i in range(sample_size)],
            dtype=torch.bool, device=device,
        )
        scores_masked = scores.clone()
        scores_masked[~mask] = -float("inf")

        # Top-K
        k = min(pool_size, mask.sum().item())
        if k == 0:
            # Fallback: random negatives
            for si in sample_indices:
                rand_idx = rng.randint(0, n_catalog)
                neg_feats[si] = item_catalog_feats[rand_idx]
                neg_pops[si] = np.log1p(1.0)
            continue

        topk_vals, topk_local = torch.topk(scores_masked, k)
        topk_catalog_idx = cand_indices[topk_local.cpu().numpy()]

        # For each sample of this user, randomly pick from the pool
        for si in sample_indices:
            pick = rng.choice(len(topk_catalog_idx))
            chosen_item = topk_catalog_idx[pick]
            neg_feats[si] = item_catalog_feats[chosen_item]
            neg_pops[si] = np.log1p(float(chosen_item + 1))  # proxy popularity

    model.train()
    return neg_feats, neg_pops


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation — Recall@K
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_recall_at_k(
    model: TwoTowerModel,
    val_user_feats: np.ndarray,
    val_item_ids: np.ndarray,
    item_catalog_feats: np.ndarray,
    k_values: Tuple[int, ...] = (50, 500),
) -> Dict[int, float]:
    """Compute Recall@K on a validation set.

    For each validation user, score *all* catalog items and check whether
    the true positive item appears in the top-K.

    Args:
        model:              TwoTowerModel (will be set to eval mode).
        val_user_feats:     (N_val, 146) user feature vectors.
        val_item_ids:       (N_val,) ground-truth item indices.
        item_catalog_feats: (N_items, 83) full catalog features.
        k_values:           Tuple of K thresholds.

    Returns:
        Dict mapping K → recall (fraction of users whose positive is in top-K).
    """
    model.eval()

    # Pre-compute all item embeddings
    cat_tensor = torch.as_tensor(item_catalog_feats, dtype=torch.float32, device=device)
    # Process in chunks to avoid OOM
    chunk_size = 2048
    all_item_embs = []
    for start in range(0, cat_tensor.shape[0], chunk_size):
        chunk = cat_tensor[start : start + chunk_size]
        all_item_embs.append(model.item_tower(chunk).cpu())
    item_emb_matrix = torch.cat(all_item_embs, dim=0)  # (N_items, 128)

    max_k = min(max(k_values), item_emb_matrix.shape[0])
    hits: Dict[int, int] = {k: 0 for k in k_values}
    n_val = val_user_feats.shape[0]
    if n_val == 0:
        return {k: 0.0 for k in k_values}

    # Process users in chunks
    for start in range(0, n_val, chunk_size):
        end = min(start + chunk_size, n_val)
        u_tensor = torch.as_tensor(
            val_user_feats[start:end], dtype=torch.float32, device=device,
        )
        u_emb = model.user_tower(u_tensor).cpu()  # (chunk, 128)
        scores = torch.mm(u_emb, item_emb_matrix.t())  # (chunk, N_items)

        _, topk_indices = torch.topk(scores, max_k, dim=-1)  # (chunk, max_k)
        topk_np = topk_indices.numpy()
        true_items = val_item_ids[start:end]

        for i in range(end - start):
            for k in k_values:
                if true_items[i] in topk_np[i, : min(k, topk_np.shape[1])]:
                    hits[k] += 1

    model.train()
    return {k: v / n_val for k, v in hits.items()}


@torch.no_grad()
def compute_retrieval_metrics(
    model: TwoTowerModel,
    val_user_feats: np.ndarray,
    val_item_ids: np.ndarray,
    item_catalog_feats: np.ndarray,
    item_popularity: np.ndarray,
    k_values: Tuple[int, ...] = (10, 20, 50, 500),
) -> pd.DataFrame:
    """Compute retrieval metrics and popularity-split recall on validation data."""
    model.eval()
    n_val = val_user_feats.shape[0]
    if n_val == 0:
        return pd.DataFrame()

    cat_tensor = torch.as_tensor(item_catalog_feats, dtype=torch.float32, device=device)
    chunk_size = 2048
    all_item_embs = []
    for start in range(0, cat_tensor.shape[0], chunk_size):
        chunk = cat_tensor[start:start + chunk_size]
        all_item_embs.append(model.item_tower(chunk).cpu())
    item_emb_matrix = torch.cat(all_item_embs, dim=0)

    max_k = min(max(k_values), item_emb_matrix.shape[0])
    ranks: list[int] = []
    for start in range(0, n_val, chunk_size):
        end = min(start + chunk_size, n_val)
        u_tensor = torch.as_tensor(val_user_feats[start:end], dtype=torch.float32, device=device)
        u_emb = model.user_tower(u_tensor).cpu()
        scores = torch.mm(u_emb, item_emb_matrix.t())
        _, topk_indices = torch.topk(scores, max_k, dim=-1)
        topk_np = topk_indices.numpy()
        true_items = val_item_ids[start:end]
        for i, true_item in enumerate(true_items):
            matches = np.where(topk_np[i] == true_item)[0]
            ranks.append(int(matches[0]) + 1 if len(matches) else max_k + 1)

    ranks_arr = np.asarray(ranks, dtype=np.int32)
    true_pop = item_popularity[val_item_ids]
    q1, q3 = np.quantile(true_pop, [0.25, 0.75])
    buckets = np.full(n_val, "mid", dtype=object)
    buckets[true_pop <= q1] = "tail"
    buckets[true_pop >= q3] = "popular"

    rows: list[dict] = []
    for k in k_values:
        k_eff = min(k, item_emb_matrix.shape[0])
        hit_mask = ranks_arr <= k_eff
        reciprocal = np.where(hit_mask, 1.0 / ranks_arr, 0.0)
        ndcg = np.where(hit_mask, 1.0 / np.log2(ranks_arr + 1), 0.0)
        rows.append({
            "segment": "all",
            "k": k,
            "Recall": float(hit_mask.mean()),
            "MRR": float(reciprocal.mean()) if k <= 10 else np.nan,
            "NDCG": float(ndcg.mean()) if k <= 10 else np.nan,
            "n": int(n_val),
        })
        for segment in ("tail", "mid", "popular"):
            seg_mask = buckets == segment
            if not seg_mask.any():
                continue
            seg_hits = hit_mask[seg_mask]
            seg_ranks = ranks_arr[seg_mask]
            seg_recip = np.where(seg_hits, 1.0 / seg_ranks, 0.0)
            seg_ndcg = np.where(seg_hits, 1.0 / np.log2(seg_ranks + 1), 0.0)
            rows.append({
                "segment": segment,
                "k": k,
                "Recall": float(seg_hits.mean()),
                "MRR": float(seg_recip.mean()) if k <= 10 else np.nan,
                "NDCG": float(seg_ndcg.mean()) if k <= 10 else np.nan,
                "n": int(seg_mask.sum()),
            })

    model.train()
    return pd.DataFrame(rows)


def build_item_popularity(item_ids: np.ndarray, n_items: int) -> np.ndarray:
    """Return interaction-count popularity per item, with nonzero smoothing."""
    counts = np.bincount(item_ids.astype(np.int64), minlength=n_items).astype(np.float32)
    return counts + 1.0


def oversample_tail_training_indices(
    train_indices: np.ndarray,
    item_ids: np.ndarray,
    item_popularity: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Repeat tail-item positive samples to make tail positives less rare."""
    if factor <= 1:
        return train_indices
    train_popularity = item_popularity[item_ids[train_indices]]
    q1 = np.quantile(train_popularity, 0.25)
    tail_mask = train_popularity <= q1
    tail_indices = train_indices[tail_mask]
    if len(tail_indices) == 0:
        return train_indices
    repeated_tail = np.repeat(tail_indices, factor - 1)
    return np.concatenate([train_indices, repeated_tail])


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    losses: List[float],
    recall50: List[float],
    recall500: List[float],
    save_path: Path,
) -> None:
    """Plot loss and recall curves side-by-side and save to disk.

    Args:
        losses:    Per-epoch training loss.
        recall50:  Per-epoch Recall@50.
        recall500: Per-epoch Recall@500.
        save_path: Output file path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(1, len(losses) + 1))

    # ---- Loss ----
    ax1.plot(epochs, losses, "b-o", linewidth=1.5, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BPR Loss")
    ax1.set_title("Training Loss")
    ax1.axvline(x=PHASE1_EPOCHS + 0.5, color="grey", linestyle="--", alpha=0.5, label="Phase 1→2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Recall ----
    ax2.plot(epochs, recall50, "g-s", linewidth=1.5, markersize=4, label="Recall@50")
    ax2.plot(epochs, recall500, "r-^", linewidth=1.5, markersize=4, label="Recall@500")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall")
    ax2.set_title("Validation Recall")
    ax2.axvline(x=PHASE1_EPOCHS + 0.5, color="grey", linestyle="--", alpha=0.5, label="Phase 1→2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Embedding export
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_embeddings(
    model: TwoTowerModel,
    item_catalog_feats: np.ndarray,
    train_user_feats: np.ndarray,
    item_save_path: Path,
    user_save_path: Path,
) -> None:
    """Compute and save item + user embeddings as Parquet.

    Args:
        model:              Trained TwoTowerModel.
        item_catalog_feats: (N_items, 83) item features.
        train_user_feats:   (N_unique_users, 146) unique user features.
        item_save_path:     Output path for item embeddings Parquet.
        user_save_path:     Output path for user embeddings Parquet.
    """
    model.eval()

    # ---- Item embeddings ----
    cat_tensor = torch.as_tensor(item_catalog_feats, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, cat_tensor.shape[0], 2048):
        chunks.append(model.item_tower(cat_tensor[start : start + 2048]).cpu().numpy())
    item_embs = np.concatenate(chunks, axis=0)

    item_df = pd.DataFrame(
        item_embs,
        columns=[f"emb_{i}" for i in range(ITEM_EMBEDDING_DIM)],
    )
    item_df.insert(0, "item_id", [f"item_{i:04d}" for i in range(len(item_embs))])
    item_save_path.parent.mkdir(parents=True, exist_ok=True)
    item_df.to_parquet(item_save_path, index=False)
    print(f"[INFO] Item embeddings ({item_embs.shape}) saved to {item_save_path}")

    # ---- User embeddings (deduplicated) ----
    u_tensor = torch.as_tensor(train_user_feats, dtype=torch.float32, device=device)
    u_chunks = []
    for start in range(0, u_tensor.shape[0], 2048):
        u_chunks.append(model.user_tower(u_tensor[start : start + 2048]).cpu().numpy())
    user_embs = np.concatenate(u_chunks, axis=0)

    user_df = pd.DataFrame(
        user_embs,
        columns=[f"emb_{i}" for i in range(USER_EMBEDDING_DIM)],
    )
    user_df.insert(0, "user_id", [f"user_{i:04d}" for i in range(len(user_embs))])
    user_save_path.parent.mkdir(parents=True, exist_ok=True)
    user_df.to_parquet(user_save_path, index=False)
    print(f"[INFO] User embeddings ({user_embs.shape}) saved to {user_save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train(
    n_users: int = 2_000,
    n_items: int = 500,
    interactions_per_user: int = 10,
    val_fraction: float = 0.15,
    total_epochs: int = TOTAL_EPOCHS,
    phase1_epochs: int = PHASE1_EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    phase1_only: bool = False,
    hard_negative_weight: float = 1.0,
    tail_oversample_factor: int = 1,
) -> TwoTowerModel:
    """Run the full two-phase training pipeline.

    Args:
        n_users:               Number of synthetic users.
        n_items:               Number of synthetic catalog items.
        interactions_per_user: Positive interactions per user.
        val_fraction:          Fraction of data held out for validation.
        total_epochs:          Total training epochs (Phase 1 + Phase 2).
        phase1_epochs:         Epochs in Phase 1 (in-batch negatives).
        batch_size:            Mini-batch size.
        seed:                  Random seed.

    Returns:
        The trained ``TwoTowerModel``.
    """
    print("=" * 70)
    print("Two-Tower Training Pipeline")
    print("=" * 70)
    print(f"  Users: {n_users}  |  Items: {n_items}  |  Interactions/user: {interactions_per_user}")
    if phase1_only:
        phase1_epochs = total_epochs

    print(f"  Epochs: {total_epochs} (Phase 1: {phase1_epochs}, Phase 2: {total_epochs - phase1_epochs})")
    print(f"  Batch size: {batch_size}  |  Device: {device}")
    print(f"  Phase 1 only: {phase1_only}  |  Hard-neg weight: {hard_negative_weight:.2f}  |  Tail oversample: {tail_oversample_factor}x")
    print()

    # ---- Data ----
    real = try_load_real_data()
    if real is not None:
        user_feats, item_feats, log_pop, item_catalog, item_ids = real
    else:
        print("[INFO] Generating synthetic training data...")
        user_feats, item_feats, log_pop, item_catalog, item_ids = generate_synthetic_data(
            n_users=n_users, n_items=n_items,
            interactions_per_user=interactions_per_user, seed=seed,
        )
    print(f"[INFO] Dataset: {user_feats.shape[0]} interactions, {item_catalog.shape[0]} catalog items")

    # ---- Train / Val split ----
    rng = np.random.RandomState(seed)
    n_total = user_feats.shape[0]
    if n_total < 2:
        raise ValueError("Need at least two interactions to train and validate the two-tower model.")
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_fraction))
    n_val = min(n_val, n_total - 1)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    item_popularity = build_item_popularity(item_ids, item_catalog.shape[0])
    train_idx = oversample_tail_training_indices(
        train_indices=train_idx,
        item_ids=item_ids,
        item_popularity=item_popularity,
        factor=tail_oversample_factor,
    )

    train_ds = RecommendationDataset(
        user_features=user_feats[train_idx],
        item_features=item_feats[train_idx],
        log_pop=log_pop[train_idx],
    )
    train_item_ids = item_ids[train_idx]
    val_user_feats = user_feats[val_idx]
    val_item_ids = item_ids[val_idx]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    print(f"[INFO] Train: {len(train_ds)} samples, Val: {n_val} samples")

    # ---- Model + Optimizer ----
    model = TwoTowerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- Metric tracking ----
    history_loss: List[float] = []
    history_r50: List[float] = []
    history_r500: List[float] = []
    metrics_history: list[pd.DataFrame] = []
    best_r50 = -1.0
    best_state = None

    # ---- Training loop ----
    for epoch in range(1, total_epochs + 1):
        model.train()
        phase = 1 if epoch <= phase1_epochs else 2
        is_hard_neg_epoch = (
            phase == 2
            and (epoch - phase1_epochs - 1) % HARD_NEG_REFRESH_INTERVAL == 0
        )

        # -- Hard-negative refresh --
        if is_hard_neg_epoch:
            print(f"  [Epoch {epoch}] Mining hard negatives...")
            neg_feats, neg_pops = mine_hard_negatives(
                model, train_ds, item_catalog, train_item_ids, seed=epoch,
            )
            train_ds.set_hard_negatives(neg_feats, neg_pops)
            # Recreate DataLoader to pick up new data
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=0, drop_last=False,
            )

        # -- Epoch training --
        epoch_losses: list[float] = []
        step_fn = hardneg_bpr_step if (phase == 2 and train_ds.has_hard_negatives) else inbatch_bpr_step
        desc = f"Epoch {epoch:2d}/{total_epochs} [Phase {phase}]"
        for batch in tqdm(train_loader, desc=desc, leave=False):
            if step_fn is hardneg_bpr_step:
                loss_val = step_fn(model, batch, optimizer, hard_negative_weight)
            else:
                loss_val = step_fn(model, batch, optimizer)
            epoch_losses.append(loss_val)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # -- Validation --
        recalls = compute_recall_at_k(
            model, val_user_feats, val_item_ids, item_catalog, k_values=(50, 500),
        )
        r50 = recalls[50]
        r500 = recalls[500]
        epoch_metrics = compute_retrieval_metrics(
            model,
            val_user_feats,
            val_item_ids,
            item_catalog,
            item_popularity,
            k_values=(10, 20, 50, 500),
        )
        epoch_metrics.insert(0, "epoch", epoch)
        epoch_metrics["phase"] = phase
        metrics_history.append(epoch_metrics)

        history_loss.append(avg_loss)
        history_r50.append(r50)
        history_r500.append(r500)

        marker = ""
        if r50 > best_r50:
            best_r50 = r50
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " * best"

        print(
            f"  Epoch {epoch:2d}  |  Loss: {avg_loss:.4f}  |  "
            f"R@50: {r50:.4f}  |  R@500: {r500:.4f}{marker}"
        )

    # ---- Save best model ----
    model_path = DATA_DIR / "two_tower_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, str(model_path))
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), str(model_path))
    print(f"\n[INFO] Best model (R@50={best_r50:.4f}) saved to {model_path}")

    if metrics_history:
        metrics_df = pd.concat(metrics_history, ignore_index=True)
        metrics_df["is_best_r50_epoch"] = metrics_df["epoch"] == int(np.argmax(history_r50) + 1)
        RETRIEVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(RETRIEVAL_METRICS_PATH, index=False)
        best_rows = metrics_df[metrics_df["is_best_r50_epoch"] & metrics_df["segment"].eq("all")]
        print(f"[INFO] Retrieval metrics saved to {RETRIEVAL_METRICS_PATH}")
        if not best_rows.empty:
            summary = best_rows[["k", "Recall", "MRR", "NDCG"]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            print("[INFO] Best-epoch retrieval metrics:")
            print(summary)

    # ---- Export embeddings ----
    # Deduplicate user features for export
    unique_user_feats = np.unique(user_feats, axis=0)
    export_embeddings(
        model, item_catalog, unique_user_feats,
        item_save_path=DATA_DIR / "item_embeddings.parquet",
        user_save_path=DATA_DIR / "user_embeddings.parquet",
    )

    # ---- Plot ----
    plot_training_curves(
        history_loss, history_r50, history_r500,
        save_path=DATA_DIR / "training_curves.png",
    )

    print("\n[OK] Training pipeline complete.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(n_users=2_000, n_items=500, interactions_per_user=10)
