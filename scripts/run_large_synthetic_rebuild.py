"""Run a larger synthetic data rebuild without manual step sequencing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FEATURE_STEPS = [
    "session_features",
    "temporal_features",
    "nmf_topics",
    "author_embeddings",
    "quality_scores",
]


def run_command(cmd: list[str]) -> None:
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_pipeline_step(step: str) -> None:
    run_command([sys.executable, "run_pipeline.py", "--step", step])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate larger synthetic data and rebuild feature artifacts.")
    parser.add_argument("--num-users", type=int, default=3000)
    parser.add_argument("--num-items", type=int, default=5000)
    parser.add_argument("--sessions-per-user", type=int, default=20)
    parser.add_argument("--avg-chapters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-retrieval-pipeline", action="store_true", help="After feature rebuild, run pipeline from retrieval.")
    args = parser.parse_args()

    run_command(
        [
            sys.executable,
            "feature_store/simulator/markov_event_simulator.py",
            "--num-users",
            str(args.num_users),
            "--num-items",
            str(args.num_items),
            "--sessions-per-user",
            str(args.sessions_per_user),
            "--avg-chapters",
            str(args.avg_chapters),
            "--seed",
            str(args.seed),
        ]
    )

    for step in FEATURE_STEPS:
        run_pipeline_step(step)

    if args.include_retrieval_pipeline:
        run_command([sys.executable, "run_pipeline.py", "--from", "retrieval"])

    print("[OK] Large synthetic feature rebuild complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
