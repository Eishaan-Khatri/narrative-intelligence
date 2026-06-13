"""Portable CLI entrypoint for two-tower retrieval training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system_a_discovery_engine.layer2_retrieval.train_loop import train  # noqa: E402
from system_a_discovery_engine.layer2_retrieval.two_tower_model import device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the two-tower retrieval model.")
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs.")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="In-batch-negative epochs.")
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Disable hard-negative Phase 2 and train only with in-batch negatives.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=1.0,
        help="Multiplier for Phase 2 hard-negative BPR loss.",
    )
    parser.add_argument(
        "--tail-oversample-factor",
        type=int,
        default=1,
        help="Repeat tail-item positive samples this many times during training.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--hard-negative-popularity-alpha",
        type=float,
        default=0.0,
        help="Inverse-popularity weighting strength when sampling from mined hard-negative pools.",
    )
    parser.add_argument(
        "--synthetic-users",
        type=int,
        default=2000,
        help="Fallback synthetic user count if processed artifacts are missing.",
    )
    parser.add_argument(
        "--synthetic-items",
        type=int,
        default=500,
        help="Fallback synthetic item count if processed artifacts are missing.",
    )
    parser.add_argument(
        "--synthetic-interactions-per-user",
        type=int,
        default=10,
        help="Fallback synthetic interactions per user if processed artifacts are missing.",
    )
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Torch device: {device}")

    train(
        n_users=args.synthetic_users,
        n_items=args.synthetic_items,
        interactions_per_user=args.synthetic_interactions_per_user,
        total_epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        phase1_only=args.phase1_only,
        hard_negative_weight=args.hard_negative_weight,
        tail_oversample_factor=args.tail_oversample_factor,
        hard_negative_popularity_alpha=args.hard_negative_popularity_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
