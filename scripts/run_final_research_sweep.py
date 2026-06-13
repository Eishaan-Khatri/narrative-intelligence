"""Run the final System A research sweep in one long GPU session.

The sweep tests simulator calibration variants and retrieval-training variants
in a single run, then restores the best model artifacts to canonical paths.
It is designed for limited access to the GPU machine.
"""

from __future__ import annotations

import argparse
import gc
import json
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

from feature_store.simulator.markov_event_simulator import generate_synthetic_dataset  # noqa: E402
from scripts.enrich_synthetic_catalog_text import enrich_catalog_text  # noqa: E402
from system_a_discovery_engine.layer2_retrieval.train_loop import train  # noqa: E402
from system_a_discovery_engine.layer2_retrieval.two_tower_model import device  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
REPORT_DIR = PROJECT_ROOT / "reports"
SWEEP_DIR = DATA_DIR / "final_sweep"

FEATURE_STEPS = [
    "session_features",
    "temporal_features",
    "nmf_topics",
    "author_embeddings",
    "quality_scores",
]

ARTIFACTS = [
    "two_tower_model.pt",
    "item_embeddings.parquet",
    "user_embeddings.parquet",
    "training_curves.png",
    "retrieval_metrics.parquet",
]


@dataclass(frozen=True)
class DatasetVariant:
    name: str
    rationale: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    rationale: str
    kwargs: dict[str, Any]


DATASET_VARIANTS = [
    DatasetVariant(
        name="calibrated_balanced",
        rationale="Lower transition exits and stronger engagement to fix the prior near-total abandonment problem.",
        kwargs={
            "exit_probability_multiplier": 0.45,
            "transition_exit_multiplier": 0.35,
            "speed_multiplier": 1.35,
            "patience_multiplier": 1.40,
            "valley_multiplier": 0.45,
            "engaged_boost_multiplier": 1.20,
            "quality_alpha": 2.8,
            "quality_beta": 4.5,
            "topic_concentration": 0.45,
            "taste_concentration": 0.45,
        },
    ),
    DatasetVariant(
        name="high_completion_stress",
        rationale="Tests whether retrieval improves when labels contain many clear positive completions.",
        kwargs={
            "exit_probability_multiplier": 0.25,
            "transition_exit_multiplier": 0.20,
            "speed_multiplier": 1.55,
            "patience_multiplier": 1.80,
            "valley_multiplier": 0.25,
            "engaged_boost_multiplier": 1.40,
            "quality_alpha": 3.2,
            "quality_beta": 3.8,
            "topic_concentration": 0.35,
            "taste_concentration": 0.35,
        },
    ),
    DatasetVariant(
        name="long_tail_harder",
        rationale="Keeps the task harder with sharper user/item taste clusters and stronger long-tail separation.",
        kwargs={
            "exit_probability_multiplier": 0.55,
            "transition_exit_multiplier": 0.45,
            "speed_multiplier": 1.30,
            "patience_multiplier": 1.35,
            "valley_multiplier": 0.50,
            "engaged_boost_multiplier": 1.25,
            "quality_alpha": 2.2,
            "quality_beta": 5.0,
            "topic_concentration": 0.22,
            "taste_concentration": 0.25,
        },
    ),
]

TRAINING_VARIANTS = [
    TrainingVariant(
        name="phase1_lr1e3",
        rationale="Original phase-1-only baseline; checks if hard negatives are still unnecessary.",
        kwargs={"total_epochs": 6, "phase1_only": True, "learning_rate": 1e-3, "tail_oversample_factor": 1},
    ),
    TrainingVariant(
        name="phase1_tail_lr5e4",
        rationale="Previous best family: lower LR with tail-positive oversampling.",
        kwargs={"total_epochs": 10, "phase1_only": True, "learning_rate": 5e-4, "tail_oversample_factor": 3},
    ),
    TrainingVariant(
        name="phase1_tail_lr3e4",
        rationale="More conservative LR to test if ranking stability improves.",
        kwargs={"total_epochs": 14, "phase1_only": True, "learning_rate": 3e-4, "tail_oversample_factor": 4},
    ),
    TrainingVariant(
        name="phase1_tail_strong_lr3e4",
        rationale="Stronger tail-positive oversampling for weak tail recall.",
        kwargs={"total_epochs": 14, "phase1_only": True, "learning_rate": 3e-4, "tail_oversample_factor": 6},
    ),
    TrainingVariant(
        name="mild_hardneg_tail",
        rationale="Tests whether hard negatives work only when their loss is heavily downweighted.",
        kwargs={
            "total_epochs": 14,
            "phase1_epochs": 6,
            "learning_rate": 3e-4,
            "hard_negative_weight": 0.10,
            "tail_oversample_factor": 4,
            "hard_negative_popularity_alpha": 0.75,
        },
    ),
    TrainingVariant(
        name="pop_balanced_hardneg",
        rationale="Tests popularity-balanced hard-negative mining against phase-1-only training.",
        kwargs={
            "total_epochs": 16,
            "phase1_epochs": 8,
            "learning_rate": 2e-4,
            "hard_negative_weight": 0.05,
            "tail_oversample_factor": 5,
            "hard_negative_popularity_alpha": 1.00,
        },
    ),
]


def run_command(cmd: list[str], required: bool = True) -> int:
    print("\n[RUN] " + " ".join(str(part) for part in cmd))
    result = subprocess.run([str(part) for part in cmd], cwd=PROJECT_ROOT)
    if required and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def selected_dataset_variants(profile: str, names: list[str] | None) -> list[DatasetVariant]:
    variants = DATASET_VARIANTS
    if profile == "standard":
        variants = DATASET_VARIANTS[:2]
    if names:
        wanted = set(names)
        variants = [variant for variant in DATASET_VARIANTS if variant.name in wanted]
    if not variants:
        raise SystemExit("[FAIL] No dataset variants selected.")
    return variants


def selected_training_variants(profile: str, names: list[str] | None) -> list[TrainingVariant]:
    variants = TRAINING_VARIANTS
    if profile == "standard":
        variants = TRAINING_VARIANTS[:4]
    if names:
        wanted = set(names)
        variants = [variant for variant in TRAINING_VARIANTS if variant.name in wanted]
    if not variants:
        raise SystemExit("[FAIL] No training variants selected.")
    return variants


def run_feature_rebuild() -> None:
    for step in FEATURE_STEPS:
        run_command([sys.executable, "run_pipeline.py", "--step", step])


def summarize_sessions() -> dict[str, Any]:
    path = DATA_DIR / "session_features.parquet"
    df = pd.read_parquet(path)
    exit_counts = df["exit_reason"].value_counts(normalize=True)
    shape_counts = df["completion_curve_shape"].value_counts(normalize=True)
    return {
        "sessions": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "mean_completion_pct": float(df["final_completion_pct"].mean()),
        "median_completion_pct": float(df["final_completion_pct"].median()),
        "chapter_end_rate": float(exit_counts.get("chapter_end", 0.0)),
        "mid_chapter_rate": float(exit_counts.get("mid_chapter", 0.0)),
        "abandon_early_rate": float(shape_counts.get("abandon_early", 0.0)),
        "steady_rate": float(shape_counts.get("steady", 0.0)),
    }


def copy_current_artifacts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        src = DATA_DIR / artifact
        if src.exists():
            shutil.copy2(src, target_dir / artifact)


def restore_artifacts(source_dir: Path) -> None:
    for artifact in ARTIFACTS:
        src = source_dir / artifact
        if src.exists():
            shutil.copy2(src, DATA_DIR / artifact)


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


def summarize_training_result(dataset_name: str, training_name: str, artifact_dir: Path) -> dict[str, Any]:
    metrics = pd.read_parquet(artifact_dir / "retrieval_metrics.parquet")
    recall_10 = metric_value(metrics, "all", 10, "Recall")
    recall_20 = metric_value(metrics, "all", 20, "Recall")
    recall_50 = metric_value(metrics, "all", 50, "Recall")
    tail_recall_50 = metric_value(metrics, "tail", 50, "Recall")
    mrr_10 = metric_value(metrics, "all", 10, "MRR")
    ndcg_10 = metric_value(metrics, "all", 10, "NDCG")
    selection_score = recall_50 + 0.75 * tail_recall_50 + 0.25 * mrr_10 + 0.25 * ndcg_10
    return {
        "dataset_variant": dataset_name,
        "training_variant": training_name,
        "Recall@10": recall_10,
        "Recall@20": recall_20,
        "Recall@50": recall_50,
        "Tail_Recall@50": tail_recall_50,
        "MRR@10": mrr_10,
        "NDCG@10": ndcg_10,
        "selection_score": selection_score,
    }


def run_training_variant(
    dataset: DatasetVariant,
    training: TrainingVariant,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    kwargs = {
        "batch_size": batch_size,
        "seed": seed,
        **training.kwargs,
    }
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset.name}")
    print(f"TRAINING: {training.name}")
    print(training.rationale)
    print("=" * 80)
    train(**kwargs)
    artifact_dir = SWEEP_DIR / dataset.name / training.name
    copy_current_artifacts(artifact_dir)
    return summarize_training_result(dataset.name, training.name, artifact_dir)


def write_report(
    summary_df: pd.DataFrame,
    session_df: pd.DataFrame,
    dataset_variants: list[DatasetVariant],
    training_variants: list[TrainingVariant],
    best_row: pd.Series,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SWEEP_DIR / "final_research_sweep_summary.csv"
    sessions_path = SWEEP_DIR / "final_research_sweep_session_calibration.csv"
    json_path = REPORT_DIR / "system_a_final_research_sweep.json"
    md_path = REPORT_DIR / "system_a_final_research_sweep.md"
    summary_df.to_csv(summary_path, index=False)
    session_df.to_csv(sessions_path, index=False)

    payload = {
        "best": best_row.to_dict(),
        "dataset_variants": [variant.__dict__ for variant in dataset_variants],
        "training_variants": [variant.__dict__ for variant in training_variants],
        "summary_csv": str(summary_path.relative_to(PROJECT_ROOT)),
        "session_calibration_csv": str(sessions_path.relative_to(PROJECT_ROOT)),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# System A Final Research Sweep",
        "",
        "## Best Run",
        f"- Dataset variant: {best_row['dataset_variant']}",
        f"- Training variant: {best_row['training_variant']}",
        f"- Recall@10: {best_row['Recall@10']:.4f}",
        f"- Recall@20: {best_row['Recall@20']:.4f}",
        f"- Recall@50: {best_row['Recall@50']:.4f}",
        f"- Tail Recall@50: {best_row['Tail_Recall@50']:.4f}",
        f"- MRR@10: {best_row['MRR@10']:.4f}",
        f"- NDCG@10: {best_row['NDCG@10']:.4f}",
        f"- Selection score: {best_row['selection_score']:.6f}",
        "",
        "## Scientific Interventions Tested",
        "- Simulator calibration: lower exit pressure, lower valley-of-death churn, higher patience, faster chapter progress.",
        "- Catalog/content enrichment: external Gutenberg/Amazon text when available, otherwise richer genre/topic descriptions.",
        "- Retrieval variants: phase-1-only baselines, tail-positive oversampling, conservative learning rates, downweighted hard negatives, popularity-balanced hard negatives.",
        "",
        "## Interpretation Rule",
        "If hard-negative variants lose to phase-1-only variants, report that hard negatives introduced false negatives/noisy pressure and were downweighted or disabled.",
        "If high-completion datasets improve retrieval, report that the previous simulator produced too few positive labels for a stable recommender objective.",
        "",
        "## Output Files",
        f"- {summary_path.relative_to(PROJECT_ROOT)}",
        f"- {sessions_path.relative_to(PROJECT_ROOT)}",
        f"- {json_path.relative_to(PROJECT_ROOT)}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote {sessions_path}")
    print(f"[OK] Wrote {md_path}")
    print(f"[OK] Wrote {json_path}")


def run_external_preparation(args: argparse.Namespace) -> None:
    if args.install_main or args.install_optional:
        setup_cmd = [sys.executable, "scripts/setup_optional_deps.py"]
        if args.install_main:
            setup_cmd.append("--install-main")
        if args.install_optional:
            setup_cmd.append("--install-optional")
        run_command(setup_cmd, required=not args.install_optional)

    if args.download_gutenberg:
        gutenberg_cmd = [
            sys.executable,
            "scripts/download_gutenberg_sample.py",
            "--limit",
            str(args.gutenberg_limit),
        ]
        if args.gutenberg_ids:
            gutenberg_cmd.extend(["--ids", args.gutenberg_ids])
        run_command(gutenberg_cmd, required=False)

    if args.amazon_input:
        run_command(
            [
                sys.executable,
                "scripts/prepare_amazon_sample.py",
                "--input",
                str(args.amazon_input),
                "--limit",
                str(args.amazon_limit),
            ],
            required=False,
        )

    if args.download_gutenberg or args.amazon_input or args.build_external_catalog:
        run_command([sys.executable, "scripts/build_external_catalog.py"], required=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final System A research sweep.")
    parser.add_argument("--profile", choices=["standard", "exhaustive"], default="exhaustive")
    parser.add_argument("--dataset-variant", action="append")
    parser.add_argument("--training-variant", action="append")
    parser.add_argument("--install-main", action="store_true")
    parser.add_argument("--install-optional", action="store_true")
    parser.add_argument("--download-gutenberg", action="store_true")
    parser.add_argument("--gutenberg-limit", type=int, default=10)
    parser.add_argument("--gutenberg-ids", help="Comma-separated Project Gutenberg IDs for a larger curated text pull.")
    parser.add_argument("--amazon-input", type=Path)
    parser.add_argument("--amazon-limit", type=int, default=25000)
    parser.add_argument("--build-external-catalog", action="store_true")
    parser.add_argument("--num-users", type=int, default=3000)
    parser.add_argument("--num-items", type=int, default=5000)
    parser.add_argument("--sessions-per-user", type=int, default=25)
    parser.add_argument("--avg-chapters", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-mix-ratio", type=float, default=1.0)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Torch device: {device}")
    run_external_preparation(args)

    dataset_variants = selected_dataset_variants(args.profile, args.dataset_variant)
    training_variants = selected_training_variants(args.profile, args.training_variant)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    for dataset in dataset_variants:
        print("\n" + "#" * 80)
        print(f"BUILDING DATASET VARIANT: {dataset.name}")
        print(dataset.rationale)
        print("#" * 80)
        events_df, _catalog_df, _users_df = generate_synthetic_dataset(
            num_users=args.num_users,
            num_items=args.num_items,
            avg_chapters=args.avg_chapters,
            sessions_per_user=args.sessions_per_user,
            seed=args.seed,
            output_dir=SYNTHETIC_DIR,
            **dataset.kwargs,
        )
        print(f"[INFO] Generated {len(events_df):,} raw events for {dataset.name}")
        del events_df
        gc.collect()

        enrich_catalog_text(
            catalog_path=SYNTHETIC_DIR / "catalog.parquet",
            output_path=SYNTHETIC_DIR / "catalog.parquet",
            external_mix_ratio=args.external_mix_ratio,
            seed=args.seed,
        )

        run_feature_rebuild()
        session_summary = summarize_sessions()
        session_summary["dataset_variant"] = dataset.name
        session_summary["dataset_rationale"] = dataset.rationale
        session_rows.append(session_summary)

        dataset_dir = SWEEP_DIR / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SYNTHETIC_DIR / "catalog.parquet", dataset_dir / "catalog.parquet")
        shutil.copy2(DATA_DIR / "session_features.parquet", dataset_dir / "session_features.parquet")

        for training in training_variants:
            row = run_training_variant(dataset, training, args.batch_size, args.seed)
            row.update(session_summary)
            row["dataset_rationale"] = dataset.rationale
            row["training_rationale"] = training.rationale
            result_rows.append(row)

    summary_df = pd.DataFrame(result_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    session_df = pd.DataFrame(session_rows)
    best_row = summary_df.iloc[0]
    best_dir = SWEEP_DIR / str(best_row["dataset_variant"]) / str(best_row["training_variant"])
    restore_artifacts(best_dir)

    write_report(summary_df, session_df, dataset_variants, training_variants, best_row)

    print("\n" + "=" * 80)
    print("[OK] Best research-sweep artifacts restored to canonical processed paths")
    print(summary_df.head(12).to_string(index=False))
    print("=" * 80)

    if not args.skip_downstream:
        run_command([sys.executable, "run_pipeline.py", "--from", "faiss_index"], required=False)
        run_command([sys.executable, "scripts/final_artifact_report.py"], required=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
