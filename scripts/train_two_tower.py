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
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
