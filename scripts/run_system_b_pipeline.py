"""Run System B opportunity intelligence pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system_b_opportunity_lab.bandit_exploration.simulation_harness import compare_bandit_policies
from system_b_opportunity_lab.bayesian_shrinkage.beta_binomial_shrinkage import beta_binomial_shrinkage
from system_b_opportunity_lab.breakout_forecasting.conformal_calibration import add_conformal_intervals
from system_b_opportunity_lab.breakout_forecasting.feature_builder import (
    add_embedding_summary,
    build_item_exposure_features,
    numeric_feature_columns,
)
from system_b_opportunity_lab.breakout_forecasting.lgbm_breakout_model import train_breakout_model
from system_b_opportunity_lab.exposure_simulation.simulation_harness import (
    PROCESSED_DIR,
    SYSTEM_B_DIR,
    load_item_universe,
    simulate_exposure_log,
)
from system_b_opportunity_lab.fairness.ecosystem_simulation import (
    exposure_fairness_by_policy,
    pareto_frontier,
)
from system_b_opportunity_lab.offline_eval.policy_stress_test import run_ips_stress_test
from system_b_opportunity_lab.uncertainty_promotion.promotion_policy import uncertainty_aware_score
from system_b_opportunity_lab.uplift_scoring.uplift_model import t_learner_uplift


def _load_item_embeddings() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "item_embeddings.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def add_promotion_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, fallback in [
        ("shrunk_mean", 0.5),
        ("breakout_score", 0.0),
        ("uplift_score", 0.0),
        ("posterior_uncertainty", 0.0),
        ("breakout_interval_width", 0.0),
        ("true_quality", 0.5),
    ]:
        if col not in out.columns:
            out[col] = fallback
        out[col] = out[col].fillna(fallback).astype(float)
    uncertainty = np.maximum(out["posterior_uncertainty"].to_numpy(), out["breakout_interval_width"].to_numpy())
    out["promotion_score"] = [
        uncertainty_aware_score(
            shrunk_quality=float(row["shrunk_mean"]),
            breakout_score=float(row["breakout_score"]),
            uplift_score=float(row["uplift_score"]),
            uncertainty=float(uncertainty[idx]),
            relevance=float(row["true_quality"]),
            min_relevance=0.20,
        )
        for idx, row in out.iterrows()
    ]
    finite = np.isfinite(out["promotion_score"])
    out.loc[~finite, "promotion_score"] = -1.0
    return out


def run_pipeline(args: argparse.Namespace) -> dict:
    SYSTEM_B_DIR.mkdir(parents=True, exist_ok=True)

    print("[System B] Loading item universe from System A artifacts...")
    item_universe = load_item_universe(limit_items=args.limit_items)
    print(f"[System B] Items: {len(item_universe)}")

    print("[System B] Simulating exposure log...")
    exposure_log, item_universe = simulate_exposure_log(
        item_universe=item_universe,
        n_users=args.n_users,
        n_days=args.n_days,
        impressions_per_day=args.impressions_per_day,
        exploration_rate=args.exploration_rate,
        seed=args.seed,
    )
    print(f"[System B] Exposure rows: {len(exposure_log)}")

    print("[System B] Building item features...")
    features = build_item_exposure_features(exposure_log, item_universe)
    features = add_embedding_summary(features, _load_item_embeddings())

    print("[System B] Bayesian shrinkage...")
    shrink_input = features.assign(successes=features["completions"], trials=features["clicks"].clip(lower=1))
    shrink = beta_binomial_shrinkage(
        shrink_input,
        success_col="successes",
        trial_col="trials",
        group_col="primary_genre",
        prior_strength=args.prior_strength,
    )

    print("[System B] Breakout model...")
    feature_cols = numeric_feature_columns(shrink)
    predictions, breakout_metrics = train_breakout_model(shrink, feature_cols=feature_cols, label_col="breakout_label")
    predictions = add_conformal_intervals(predictions)

    print("[System B] Uplift scoring...")
    uplift_features = exposure_log.merge(
        predictions[["item_id", "shrunk_mean", "breakout_score", "posterior_uncertainty"]],
        on="item_id",
        how="left",
    )
    if "true_quality" not in uplift_features.columns:
        uplift_features["true_quality"] = 0.5
    uplift_cols = ["true_quality", "shrunk_mean", "breakout_score", "posterior_uncertainty", "user_cluster"]
    uplift_rows = t_learner_uplift(
        uplift_features,
        feature_cols=uplift_cols,
        treatment_col="treated",
        outcome_col="reward",
    )
    uplift_item = uplift_rows.groupby("item_id")["uplift_score"].mean().reset_index()
    predictions = predictions.merge(uplift_item, on="item_id", how="left")
    predictions["uplift_score"] = predictions["uplift_score"].fillna(0.0)
    predictions = add_promotion_scores(predictions)

    print("[System B] Bandit policy comparison...")
    policy_metrics = compare_bandit_policies(
        predictions.sort_values("promotion_score", ascending=False).head(args.bandit_items),
        n_rounds=args.bandit_rounds,
        seed=args.seed,
    )

    print("[System B] Fairness and Pareto frontier...")
    logging_fairness = exposure_fairness_by_policy(exposure_log)
    frontier = pareto_frontier(predictions, top_k=min(args.pareto_top_k, len(predictions)))

    print("[System B] IPS stress test...")
    ips = run_ips_stress_test(exposure_log, predictions)

    outputs = {
        "item_universe": SYSTEM_B_DIR / "item_universe.parquet",
        "exposure_log": SYSTEM_B_DIR / "exposure_log.parquet",
        "item_features": SYSTEM_B_DIR / "item_features.parquet",
        "shrunk_quality": SYSTEM_B_DIR / "shrunk_quality.parquet",
        "breakout_predictions": SYSTEM_B_DIR / "breakout_predictions.parquet",
        "uplift_scores": SYSTEM_B_DIR / "uplift_scores.parquet",
        "promotion_scores": SYSTEM_B_DIR / "promotion_scores.parquet",
        "bandit_policy_metrics": SYSTEM_B_DIR / "bandit_policy_metrics.parquet",
        "fairness_metrics": SYSTEM_B_DIR / "fairness_metrics.parquet",
        "pareto_frontier": SYSTEM_B_DIR / "pareto_frontier.parquet",
        "ips_stress_test": SYSTEM_B_DIR / "ips_stress_test.parquet",
    }
    features.to_parquet(outputs["item_features"], index=False)
    shrink.to_parquet(outputs["shrunk_quality"], index=False)
    predictions[["item_id", "uplift_score"]].to_parquet(outputs["uplift_scores"], index=False)
    predictions.to_parquet(outputs["breakout_predictions"], index=False)
    predictions.to_parquet(outputs["promotion_scores"], index=False)
    policy_metrics.to_parquet(outputs["bandit_policy_metrics"], index=False)
    logging_fairness.to_parquet(outputs["fairness_metrics"], index=False)
    frontier.to_parquet(outputs["pareto_frontier"], index=False)
    ips.to_parquet(outputs["ips_stress_test"], index=False)

    summary = {
        "n_items": int(len(item_universe)),
        "n_exposures": int(len(exposure_log)),
        "breakout_metrics": breakout_metrics,
        "top_opportunity_items": predictions.sort_values("promotion_score", ascending=False).head(10)[
            ["item_id", "title", "creator_id", "primary_genre", "promotion_score", "shrunk_mean", "breakout_score", "uplift_score"]
        ].to_dict(orient="records"),
        "outputs": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in outputs.items()},
    }
    (SYSTEM_B_DIR / "system_b_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[System B] Complete.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run System B opportunity intelligence pipeline.")
    parser.add_argument("--limit-items", type=int, default=3000)
    parser.add_argument("--n-users", type=int, default=8000)
    parser.add_argument("--n-days", type=int, default=45)
    parser.add_argument("--impressions-per-day", type=int, default=3000)
    parser.add_argument("--exploration-rate", type=float, default=0.12)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--bandit-items", type=int, default=500)
    parser.add_argument("--bandit-rounds", type=int, default=12000)
    parser.add_argument("--pareto-top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
