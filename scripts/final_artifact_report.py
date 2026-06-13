"""Create a concise final artifact report for System A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
RAW_EXTERNAL_DIR = PROJECT_ROOT / "data" / "raw" / "external"
REPORT_DIR = PROJECT_ROOT / "reports"

ARTIFACTS = [
    "session_features.parquet",
    "user_temporal_features.parquet",
    "topic_vectors.parquet",
    "author_embeddings.parquet",
    "quality_scores.parquet",
    "item_fingerprints.parquet",
    "two_tower_model.pt",
    "item_embeddings.parquet",
    "user_embeddings.parquet",
    "retrieval_metrics.parquet",
    "faiss_index.bin",
    "lambdamart_model.txt",
    "ablation_results.parquet",
    "completion_ndcg_metrics.parquet",
    "oracle_analysis.parquet",
]


def parquet_shape(path: Path) -> tuple[int, int] | None:
    if not path.exists() or path.suffix != ".parquet":
        return None
    df = pd.read_parquet(path)
    return int(df.shape[0]), int(df.shape[1])


def artifact_status() -> list[dict[str, Any]]:
    rows = []
    for name in ARTIFACTS:
        path = DATA_DIR / name
        shape = parquet_shape(path)
        rows.append(
            {
                "name": name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "rows": shape[0] if shape else None,
                "columns": shape[1] if shape else None,
            }
        )
    return rows


def metric_value(metrics: pd.DataFrame, segment: str, k: int, column: str) -> float | None:
    view = metrics[(metrics["segment"] == segment) & (metrics["k"] == k)]
    if "is_best_r50_epoch" in view.columns and view["is_best_r50_epoch"].fillna(False).any():
        view = view[view["is_best_r50_epoch"].fillna(False)]
    elif "epoch" in view.columns and not view.empty:
        view = view[view["epoch"] == view["epoch"].max()]
    if view.empty or column not in view.columns:
        return None
    values = view[column].dropna()
    return float(values.iloc[-1]) if not values.empty else None


def retrieval_summary() -> dict[str, Any]:
    path = DATA_DIR / "retrieval_metrics.parquet"
    if not path.exists():
        return {"available": False}
    metrics = pd.read_parquet(path)
    return {
        "available": True,
        "Recall@10": metric_value(metrics, "all", 10, "Recall"),
        "Recall@20": metric_value(metrics, "all", 20, "Recall"),
        "Recall@50": metric_value(metrics, "all", 50, "Recall"),
        "Tail_Recall@50": metric_value(metrics, "tail", 50, "Recall"),
        "MRR@10": metric_value(metrics, "all", 10, "MRR"),
        "NDCG@10": metric_value(metrics, "all", 10, "NDCG"),
    }


def gpu_suite_summary() -> list[dict[str, Any]]:
    path = DATA_DIR / "gpu_training_suite_summary.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def dataset_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, path in {
        "synthetic_catalog": SYNTHETIC_DIR / "catalog.parquet",
        "synthetic_events": SYNTHETIC_DIR / "events.parquet",
        "gutenberg_catalog": RAW_EXTERNAL_DIR / "gutenberg_catalog.parquet",
        "amazon_catalog": RAW_EXTERNAL_DIR / "amazon_catalog.parquet",
        "external_catalog_combined": RAW_EXTERNAL_DIR / "external_catalog_combined.parquet",
    }.items():
        shape = parquet_shape(path)
        out[label] = {
            "exists": path.exists(),
            "rows": shape[0] if shape else None,
            "columns": shape[1] if shape else None,
        }
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# System A Final Artifact Report",
        "",
        "## Retrieval Metrics",
    ]
    retrieval = report["retrieval"]
    if retrieval.get("available"):
        for key in ["Recall@10", "Recall@20", "Recall@50", "Tail_Recall@50", "MRR@10", "NDCG@10"]:
            value = retrieval.get(key)
            lines.append(f"- {key}: {value:.4f}" if value is not None else f"- {key}: missing")
    else:
        lines.append("- retrieval_metrics.parquet is missing.")

    lines.extend(["", "## GPU Suite"])
    suite = report["gpu_suite"]
    if suite:
        best = suite[0]
        lines.append(f"- Best listed experiment: {best.get('experiment')}")
        lines.append(f"- Selection score: {best.get('selection_score')}")
    else:
        lines.append("- gpu_training_suite_summary.csv is missing.")

    lines.extend(["", "## Dataset Inputs"])
    for name, info in report["datasets"].items():
        lines.append(f"- {name}: exists={info['exists']}, rows={info['rows']}")

    lines.extend(["", "## Required Artifacts"])
    for item in report["artifacts"]:
        status = "OK" if item["exists"] else "MISSING"
        shape = f", rows={item['rows']}, columns={item['columns']}" if item["rows"] is not None else ""
        lines.append(f"- {status}: {item['name']} ({item['bytes']} bytes{shape})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "project_root": str(PROJECT_ROOT),
        "datasets": dataset_summary(),
        "artifacts": artifact_status(),
        "retrieval": retrieval_summary(),
        "gpu_suite": gpu_suite_summary(),
    }
    json_path = REPORT_DIR / "system_a_final_artifact_report.json"
    md_path = REPORT_DIR / "system_a_final_artifact_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"[OK] Wrote {md_path}")
    print(f"[OK] Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
