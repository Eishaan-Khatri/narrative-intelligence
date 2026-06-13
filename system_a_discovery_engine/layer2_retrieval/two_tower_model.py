"""
Narrative Intelligence Platform — System A, Layer 2
====================================================
Two-Tower Retrieval Model (PyTorch)
------------------------------------

**Algorithm overview**

The two-tower architecture produces *independently-computed* embeddings for
users and items so that relevance can be approximated by inner product in a
shared 128-dimensional space.  This enables sub-linear retrieval via FAISS
(see ``faiss_index.py``).

**User tower**
    Input (146-dim) = mean-pooled item embeddings the user has interacted with
    (128) ‖ engagement profile vector (8) ‖ context features (10).
    → Linear(146, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → L2-norm.

**Item tower**
    Input (83-dim) = item fingerprint (81) ‖ log(1 + total_interactions) (1)
    ‖ interaction_velocity_7d_30d ratio (1).
    → Linear(83, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → L2-norm.

**Training objective**
    Bayesian Personalised Ranking (BPR) loss with a log-popularity correction
    that down-weights "easy" positives from head items and prevents the negative
    sampler from being dominated by long-tail items.

**Design decisions**
    * L2-normalised output → inner product = cosine similarity, which lets us
      use ``IndexFlatIP`` / ``IndexIVFPQ`` with inner-product metric.
    * Dropout only in the hidden layer (not on the embedding) so that the
      final representations are deterministic at inference time.
    * ``RecommendationDataset`` implements *in-batch* negative sampling:
      every other positive item in the batch serves as a negative, giving
      O(B²) training signal per forward pass without extra index lookups.
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

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from feature_store.schema import (
    ENGAGEMENT_PROFILE_DIM,
    ITEM_EMBEDDING_DIM,
    ITEM_FINGERPRINT_DIM,
    ITEM_TOWER_INPUT_DIM,
    USER_EMBEDDING_DIM,
    USER_TOWER_CONTEXT_DIM,
    USER_TOWER_INPUT_DIM,
)

# ---------------------------------------------------------------------------
# Global device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Tower Modules
# ═══════════════════════════════════════════════════════════════════════════

class UserTower(nn.Module):
    """Map a 146-dim user feature vector to a unit-normalised 128-dim embedding.

    Input layout (concatenated before being fed in):
        [mean_pool_item_embeddings (128)
         | engagement_profile_vector (8)
         | context_features (10)]             → 146-dim

    Architecture:
        Linear(146, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → L2-norm
    """

    def __init__(
        self,
        input_dim: int = USER_TOWER_INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = USER_EMBEDDING_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 146) user feature tensor.

        Returns:
            (batch, 128) L2-normalised user embeddings.
        """
        return F.normalize(self.net(x), p=2, dim=-1)


class ItemTower(nn.Module):
    """Map an 83-dim item feature vector to a unit-normalised 128-dim embedding.

    Input layout (concatenated before being fed in):
        [item_fingerprint (81)
         | log(1 + total_interactions) (1)
         | interaction_velocity_7d_30d (1)]   → 83-dim

    Architecture:
        Linear(83, 256) → ReLU → Dropout(0.2) → Linear(256, 128) → L2-norm
    """

    def __init__(
        self,
        input_dim: int = ITEM_TOWER_INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = ITEM_EMBEDDING_DIM,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 83) item feature tensor.

        Returns:
            (batch, 128) L2-normalised item embeddings.
        """
        return F.normalize(self.net(x), p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Combined Two-Tower Model
# ═══════════════════════════════════════════════════════════════════════════

class TwoTowerModel(nn.Module):
    """Two-tower model encapsulating both user and item towers.

    The towers are intentionally *separate* submodules so that at serving
    time item embeddings can be pre-computed offline while user embeddings
    are computed on the fly from the latest context.
    """

    def __init__(self) -> None:
        super().__init__()
        self.user_tower = UserTower()
        self.item_tower = ItemTower()

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both embeddings in a single forward call.

        Args:
            user_features: (batch, 146)
            item_features: (batch, 83)

        Returns:
            Tuple of (user_emb, item_emb) each (batch, 128) and L2-normalised.
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return user_emb, item_emb

    @staticmethod
    def score(
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Relevance score = dot-product (equivalent to cosine for unit vectors).

        Args:
            user_emb: (batch, 128) or (1, 128)
            item_emb: (batch, 128) or (N, 128)

        Returns:
            (batch,) or (batch, N) similarity scores.
        """
        # If both are 2-D and second dim matches, element-wise dot product
        if user_emb.dim() == 2 and item_emb.dim() == 2:
            if user_emb.shape[0] == item_emb.shape[0]:
                return (user_emb * item_emb).sum(dim=-1)
            # Broadcasting: user (B, D) x items (N, D)^T → (B, N)
            return torch.mm(user_emb, item_emb.t())
        return (user_emb * item_emb).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# BPR Loss with Popularity Correction
# ═══════════════════════════════════════════════════════════════════════════

def bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    pos_log_pop: torch.Tensor,
    neg_log_pop: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalised Ranking loss with log-popularity correction.

    The popularity bias term ``log_pop`` is subtracted from the raw dot-
    product score so that head items do not receive an inflated positive
    signal merely because they appear frequently in the training data.

    .. math::

        \\mathcal{L} = -\\frac{1}{B} \\sum_{i=1}^{B}
            \\log \\sigma\\bigl(
                (u_i \\cdot p_i - \\mathrm{pop}_p)
              - (u_i \\cdot n_i - \\mathrm{pop}_n)
            \\bigr)

    Args:
        user_emb:      (B, D) user embeddings.
        pos_item_emb:  (B, D) positive item embeddings.
        neg_item_emb:  (B, D) negative item embeddings.
        pos_log_pop:   (B,) log(1 + popularity_count) for positive items.
        neg_log_pop:   (B,) log(1 + popularity_count) for negative items.

    Returns:
        Scalar loss (mean over batch).
    """
    pos_score = (user_emb * pos_item_emb).sum(dim=-1) - pos_log_pop
    neg_score = (user_emb * neg_item_emb).sum(dim=-1) - neg_log_pop
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset with In-Batch Negative Sampling
# ═══════════════════════════════════════════════════════════════════════════

class RecommendationDataset(Dataset):
    """Holds user features, positive item features, and item popularity.

    **In-batch negative sampling** is performed at the *training-loop* level
    (not inside ``__getitem__``) so that each mini-batch of B positives yields
    B×(B−1) implicit negatives for free.  This class simply returns the
    positive (user, item, log_pop) triple for a given index.

    For *hard-negative mining* (Phase 2 of training), callers can supply an
    explicit ``neg_item_features`` array and ``neg_log_pop`` via
    ``set_hard_negatives()``, after which ``__getitem__`` also returns the
    pre-mined negative.

    Attributes:
        user_features:  (N, 146) np.ndarray or torch.Tensor.
        item_features:  (N, 83)  np.ndarray or torch.Tensor  — the *positive*
                        item for each user interaction.
        log_pop:        (N,) log(1 + interaction_count) for each positive item.
    """

    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        log_pop: np.ndarray,
        neg_item_features: Optional[np.ndarray] = None,
        neg_log_pop: Optional[np.ndarray] = None,
    ) -> None:
        assert user_features.shape[0] == item_features.shape[0] == log_pop.shape[0]
        self.user_features = torch.as_tensor(user_features, dtype=torch.float32)
        self.item_features = torch.as_tensor(item_features, dtype=torch.float32)
        self.log_pop = torch.as_tensor(log_pop, dtype=torch.float32)

        # Optional hard-negative arrays (set during Phase 2)
        self._neg_item_features: Optional[torch.Tensor] = None
        self._neg_log_pop: Optional[torch.Tensor] = None
        if neg_item_features is not None and neg_log_pop is not None:
            self.set_hard_negatives(neg_item_features, neg_log_pop)

    # ----- public API -----

    def set_hard_negatives(
        self,
        neg_item_features: np.ndarray,
        neg_log_pop: np.ndarray,
    ) -> None:
        """Attach pre-mined hard negatives for Phase 2 training.

        Args:
            neg_item_features: (N, 83) negative item features aligned with
                               each positive sample.
            neg_log_pop:       (N,) log-popularity for each negative item.
        """
        assert neg_item_features.shape[0] == len(self)
        self._neg_item_features = torch.as_tensor(
            neg_item_features, dtype=torch.float32,
        )
        self._neg_log_pop = torch.as_tensor(neg_log_pop, dtype=torch.float32)

    def clear_hard_negatives(self) -> None:
        """Remove hard negatives (revert to in-batch sampling mode)."""
        self._neg_item_features = None
        self._neg_log_pop = None

    @property
    def has_hard_negatives(self) -> bool:
        return self._neg_item_features is not None

    # ----- Dataset interface -----

    def __len__(self) -> int:
        return self.user_features.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Returns a dict with keys:
            * ``user_features``  — (146,)
            * ``pos_item_features`` — (83,)
            * ``pos_log_pop`` — scalar
            * ``neg_item_features`` — (83,) *only if hard negatives are set*
            * ``neg_log_pop`` — scalar  *only if hard negatives are set*
        """
        sample = {
            "user_features": self.user_features[idx],
            "pos_item_features": self.item_features[idx],
            "pos_log_pop": self.log_pop[idx],
        }
        if self._neg_item_features is not None:
            sample["neg_item_features"] = self._neg_item_features[idx]
            sample["neg_log_pop"] = self._neg_log_pop[idx]
        return sample


# ═══════════════════════════════════════════════════════════════════════════
# Standalone Demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {device}")
    print("=" * 60)
    print("Two-Tower Model — shape verification")
    print("=" * 60)

    batch_size = 32

    # ---- Random inputs ----
    user_feat = torch.randn(batch_size, USER_TOWER_INPUT_DIM, device=device)
    item_feat = torch.randn(batch_size, ITEM_TOWER_INPUT_DIM, device=device)

    model = TwoTowerModel().to(device)
    user_emb, item_emb = model(user_feat, item_feat)

    print(f"\nUserTower  input  shape: {user_feat.shape}  (expected [{batch_size}, {USER_TOWER_INPUT_DIM}])")
    print(f"UserTower  output shape: {user_emb.shape}  (expected [{batch_size}, {USER_EMBEDDING_DIM}])")
    print(f"ItemTower  input  shape: {item_feat.shape}  (expected [{batch_size}, {ITEM_TOWER_INPUT_DIM}])")
    print(f"ItemTower  output shape: {item_emb.shape}  (expected [{batch_size}, {ITEM_EMBEDDING_DIM}])")

    # ---- Verify L2 normalisation ----
    user_norms = torch.norm(user_emb, dim=-1)
    item_norms = torch.norm(item_emb, dim=-1)
    print(f"\nUser embedding L2 norms — min: {user_norms.min():.6f}, max: {user_norms.max():.6f}")
    print(f"Item embedding L2 norms — min: {item_norms.min():.6f}, max: {item_norms.max():.6f}")
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-5), "User embeddings not unit-normalised!"
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-5), "Item embeddings not unit-normalised!"
    print("✓ Both towers produce unit-normalised embeddings.")

    # ---- Score function ----
    scores = TwoTowerModel.score(user_emb, item_emb)
    print(f"\nScore shape (pair-wise): {scores.shape}  (expected [{batch_size}])")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ---- BPR loss ----
    neg_item_feat = torch.randn(batch_size, ITEM_TOWER_INPUT_DIM, device=device)
    neg_item_emb = model.item_tower(neg_item_feat)
    pos_pop = torch.rand(batch_size, device=device)
    neg_pop = torch.rand(batch_size, device=device)
    loss = bpr_loss(user_emb, item_emb, neg_item_emb, pos_pop, neg_pop)
    print(f"\nBPR loss (random): {loss.item():.4f}")

    # ---- Dataset ----
    ds = RecommendationDataset(
        user_features=np.random.randn(100, USER_TOWER_INPUT_DIM).astype(np.float32),
        item_features=np.random.randn(100, ITEM_TOWER_INPUT_DIM).astype(np.float32),
        log_pop=np.random.rand(100).astype(np.float32),
    )
    sample = ds[0]
    print(f"\nRecommendationDataset sample keys: {list(sample.keys())}")
    print(f"  user_features shape: {sample['user_features'].shape}")
    print(f"  pos_item_features shape: {sample['pos_item_features'].shape}")
    print(f"  has_hard_negatives: {ds.has_hard_negatives}")

    # ---- Model parameter count ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    print("\n✅ All Two-Tower model checks passed.")
