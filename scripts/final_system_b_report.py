"""Write a markdown summary report for System B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_B_DIR = PROJECT_ROOT / "data" / "processed" / "system_b"
REPORT_DIR = PROJECT_ROOT / "reports"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SYSTEM_B_DIR / "system_b_pipeline_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    predictions = _read(SYSTEM_B_DIR / "promotion_scores.parquet")
    bandit = _read(SYSTEM_B_DIR / "bandit_policy_metrics.parquet")
    fairness = _read(SYSTEM_B_DIR / "fairness_metrics.parquet")
    frontier = _read(SYSTEM_B_DIR / "pareto_frontier.parquet")
    ips = _read(SYSTEM_B_DIR / "ips_stress_test.parquet")

    top = predictions.sort_values("promotion_score", ascending=False).head(10) if not predictions.empty else pd.DataFrame()
    bandit_final = bandit.sort_values("round").groupby("policy").tail(1) if not bandit.empty else pd.DataFrame()
    fairness_last = fairness.sort_values("day").groupby("policy").tail(1) if not fairness.empty else pd.DataFrame()
    knee = frontier.head(1) if not frontier.empty else pd.DataFrame()

    lines = [
        "# System B Final Report",
        "",
        "## Scope",
        "System B is a simulation-backed content and creator opportunity lab. It uses System A artifacts as inputs, then evaluates exploration and promotion policies for underexposed content.",
        "",
        "## Data",
        f"- Items: {summary.get('n_items', 'n/a')}",
        f"- Logged exposures: {summary.get('n_exposures', 'n/a')}",
        "- Logging policy: popularity ranking with epsilon exploration and known propensities.",
        "",
        "## Breakout Forecasting",
    ]
    metrics = summary.get("breakout_metrics", {})
    lines.extend(
        [
            f"- Model: {metrics.get('model', 'n/a')}",
            f"- ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}" if isinstance(metrics.get("roc_auc"), (float, int)) else "- ROC-AUC: n/a",
            f"- Average precision: {metrics.get('average_precision', 0.0):.4f}" if isinstance(metrics.get("average_precision"), (float, int)) else "- Average precision: n/a",
            "",
            "## Top Opportunity Items",
        ]
    )
    if not top.empty:
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.item_id}: promotion={row.promotion_score:.4f}, shrinkage={row.shrunk_mean:.4f}, breakout={row.breakout_score:.4f}, uplift={row.uplift_score:.4f}"
            )
    else:
        lines.append("- No opportunity rows found.")

    lines.extend(["", "## Bandit Policy Comparison"])
    if not bandit_final.empty:
        for row in bandit_final.itertuples(index=False):
            lines.append(
                f"- {row.policy}: reward={row.cumulative_reward:.1f}, regret={row.cumulative_regret:.2f}, unique_items={row.unique_items_exposed}"
            )

    lines.extend(["", "## Fairness Snapshot"])
    if not fairness_last.empty:
        for row in fairness_last.itertuples(index=False):
            lines.append(f"- {row.policy}: Gini={row.gini:.4f}, HHI={row.hhi:.4f}, active_creators={row.active_creators}")

    lines.extend(["", "## Pareto Frontier Knee"])
    if not knee.empty:
        row = knee.iloc[0]
        lines.append(
            f"- lambda_novelty={row['lambda_novelty']:.2f}, lambda_fairness={row['lambda_fairness']:.2f}, relevance={row['mean_relevance']:.4f}, Gini={row['gini']:.4f}, novelty={row['mean_novelty']:.4f}"
        )

    lines.extend(["", "## IPS Stress Test"])
    if not ips.empty:
        for row in ips.itertuples(index=False):
            lines.append(
                f"- {row.target_policy}: IPS={row.ips:.4f}, SNIPS={row.snips:.4f}, DR={row.doubly_robust:.4f}, p95_weight={row.p95_weight:.2f}, ESS={row.effective_sample_size:.1f}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Bayesian shrinkage prevents tiny-sample items from dominating opportunity rankings.",
            "- Uplift scoring separates items that are merely good from items likely to benefit from extra exposure.",
            "- Uncertainty-aware promotion keeps exploration focused on high-upside candidates while enforcing a relevance floor.",
            "- IPS estimates are most reliable when target policies remain close to the logging policy; the stress test reports weight growth and effective sample size degradation.",
            "",
            "## Limitation",
            "This is a controlled simulation using synthetic exposure logs and System A artifacts. It demonstrates policy design and offline-evaluation mechanics, not live production impact.",
        ]
    )

    out_path = REPORT_DIR / "system_b_final_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
