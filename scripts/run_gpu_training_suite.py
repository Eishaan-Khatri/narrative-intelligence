"""Run System A retrieval experiments in one GPU session.

This script is meant for machines that are hard to access repeatedly. It runs a
small set of two-tower configurations, saves every run under
``data/processed/experiments/``, selects the best run using strict retrieval
metrics, restores that run to the canonical artifact paths, and optionally runs
the downstream FAISS/ranking/evaluation steps.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system_a_discovery_engine.layer2_retrieval.train_loop import train  # noqa: E402
from system_a_discovery_engine.layer2_retrieval.two_tower_model import device  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENT_DIR = DATA_DIR / "experiments"
ARTIFACTS = [
    "two_tower_model.pt",
    "item_embeddings.parquet",
    "user_embeddings.parquet",
    "training_curves.png",
    "retrieval_metrics.parquet",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    kwargs: dict[str, Any]


EXPERIMENTS = [
    Experiment(
        "phase1_only_lr1e3",
        {
            "total_epochs": 5,
            "phase1_only": True,
            "learning_rate": 1e-3,
            "tail_oversample_factor": 1,
        },
    ),
    Experiment(
        "phase1_tail_lr5e4",
        {
            "total_epochs": 8,
            "phase1_only": True,
            "learning_rate": 5e-4,
            "tail_oversample_factor": 3,
        },
    ),
    Experiment(
        "tuned_hardneg_tail",
        {
            "total_epochs": 15,
            "phase1_epochs": 5,
            "learning_rate": 5e-4,
            "hard_negative_weight": 0.25,
            "tail_oversample_factor": 3,
            "hard_negative_popularity_alpha": 0.75,
        },
    ),
]


def copy_current_artifacts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        src = DATA_DIR / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


def restore_artifacts(source_dir: Path) -> None:
    for name in ARTIFACTS:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, DATA_DIR / name)


def metric_value(metrics: pd.DataFrame, segment: str, k: int, column: str) -> float:
    view = metrics[(metrics["segment"] == segment) & (metrics["k"] == k)]
    if "is_best_r50_epoch" in view.columns and view["is_best_r50_epoch"].fillna(False).any():
        view = view[view["is_best_r50_epoch"].fillna(False)]
    elif "epoch" in view.columns and not view.empty:
        view = view[view["epoch"] == view["epoch"].max()]
    if view.empty or column not in view.columns:
        return 0.0
    values = view[column].dropna()
    return float(values.iloc[-1]) if not values.empty else 0.0


def summarize_experiment(name: str, experiment_dir: Path) -> dict[str, float | str]:
    metrics_path = experiment_dir / "retrieval_metrics.parquet"
    metrics = pd.read_parquet(metrics_path)
    recall_10 = metric_value(metrics, "all", 10, "Recall")
    recall_50 = metric_value(metrics, "all", 50, "Recall")
    tail_recall_50 = metric_value(metrics, "tail", 50, "Recall")
    mrr_10 = metric_value(metrics, "all", 10, "MRR")
    ndcg_10 = metric_value(metrics, "all", 10, "NDCG")
    selection_score = recall_50 + 0.50 * tail_recall_50 + 0.25 * mrr_10
    return {
        "experiment": name,
        "Recall@10": recall_10,
        "Recall@50": recall_50,
        "Tail_Recall@50": tail_recall_50,
        "MRR@10": mrr_10,
        "NDCG@10": ndcg_10,
        "selection_score": selection_score,
    }


def run_downstream_pipeline() -> bool:
    try:
        subprocess.run(
            [sys.executable, "run_pipeline.py", "--from", "faiss_index"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Downstream pipeline failed after retrieval artifacts were saved: {exc}")
        print("[WARN] You can still push the selected two-tower artifacts and rerun downstream later.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU two-tower experiment suite.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-users", type=int, default=2000)
    parser.add_argument("--synthetic-items", type=int, default=5000)
    parser.add_argument("--synthetic-interactions-per-user", type=int, default=10)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Torch device: {device}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | str]] = []
    for experiment in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"Running experiment: {experiment.name}")
        print("=" * 70)
        kwargs = {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_users": args.synthetic_users,
            "n_items": args.synthetic_items,
            "interactions_per_user": args.synthetic_interactions_per_user,
            **experiment.kwargs,
        }
        train(**kwargs)
        out_dir = EXPERIMENT_DIR / experiment.name
        copy_current_artifacts(out_dir)
        summaries.append(summarize_experiment(experiment.name, out_dir))

    summary_df = pd.DataFrame(summaries).sort_values("selection_score", ascending=False)
    summary_path = DATA_DIR / "gpu_training_suite_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_name = str(summary_df.iloc[0]["experiment"])
    restore_artifacts(EXPERIMENT_DIR / best_name)
    print("\n" + "=" * 70)
    print(f"[OK] Best experiment restored to canonical artifacts: {best_name}")
    print(f"[OK] Summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    if not args.skip_downstream:
        print("\n[INFO] Running downstream pipeline from FAISS index...")
        run_downstream_pipeline()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
