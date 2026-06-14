# System B - Content & Creator Opportunity Intelligence Lab

System B is the platform-facing companion to System A. System A asks what a
specific reader should see. System B asks which underexposed items and creators
deserve exploration traffic, how much uncertainty remains, and whether a new
policy can be evaluated safely before deployment.

## Diagnosis

Pure engagement ranking creates exposure concentration. Popular creators keep
getting data, underexposed creators remain uncertain, and promising niche items
may never receive enough impressions to prove their quality. A naive "promote
the highest early CTR" approach is also unsafe because tiny-sample items can
look artificially strong.

System B addresses this with four controls:

- Bayesian shrinkage for low-data item quality.
- Causal uplift scoring for whether extra exposure is likely to help.
- Uncertainty-aware promotion with a relevance floor.
- Offline policy evaluation using logged propensities.

## Pipeline

Run:

```powershell
python scripts/run_system_b_pipeline.py
python scripts/final_system_b_report.py
streamlit run dashboards/system_b_demo/app.py
```

Outputs are written under:

```text
data/processed/system_b/
reports/system_b_final_report.md
```

## Components

### Exposure Simulation

`exposure_simulation/simulation_harness.py`

Builds a synthetic exposure log from System A artifacts. Every row has known
logging propensity, which is required for IPS and doubly robust evaluation.

### Bayesian Shrinkage

`bayesian_shrinkage/beta_binomial_shrinkage.py`

Uses Beta-Binomial shrinkage so small-sample items regress toward their
genre-level prior while mature items stay close to observed completion rate.

### Breakout Forecasting

`breakout_forecasting/`

Builds item-level features and trains a LightGBM classifier when available,
falling back to sklearn. Adds conformal-style uncertainty intervals around
breakout scores.

### Uplift Scoring

`uplift_scoring/uplift_model.py`

Uses a T-learner to estimate the incremental effect of exploration exposure on
reward. This separates "good item" from "item likely to benefit from extra
exposure."

### Uncertainty-Aware Promotion

`uncertainty_promotion/promotion_policy.py`

Combines shrunk quality, breakout score, uplift score, and uncertainty while
enforcing a minimum relevance floor.

### Bandit Exploration

`bandit_exploration/`

Compares popularity baseline, epsilon-greedy, UCB1, and Thompson Sampling using
cumulative reward, regret, and exploration breadth.

### Fairness Simulation

`fairness/`

Computes Gini, HHI, long-tail viability, and a relevance-vs-fairness Pareto
frontier.

### Offline Policy Evaluation

`offline_eval/`

Runs IPS, clipped IPS, self-normalized IPS, and doubly robust IPS. The stress
test shows how estimates degrade when the target policy moves too far away from
the logging policy.

## Current Limitations

- Exposure logs are simulated, not real production logs.
- Amazon/Gutenberg content can enrich item metadata, but does not provide true
  user exposure outcomes.
- IPS is only valid when logging propensities are known and target-policy
  overlap is sufficient.
- Uplift estimates are simulation-backed; they should be treated as policy
  design evidence, not live causal proof.
