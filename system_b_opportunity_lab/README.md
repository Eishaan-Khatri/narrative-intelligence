# System B: Opportunity Lab

System B is the platform-side layer of the project.

System A asks:

```text
What should this reader see next?
```

System B asks:

```text
Which underexposed items should receive controlled exploration traffic?
```

It is not a production fairness system. It is an offline lab for testing ranking policies, uncertainty, exposure concentration, and off-policy evaluation.

## Why It Exists

Popularity ranking gives more data to items that already have data. That makes the platform worse at finding good niche items and new creators.

A naive fix is also risky. If an item has three impressions and two completions, its raw rate looks excellent, but the sample is too small to trust.

System B handles that problem with:

- Beta-Binomial shrinkage for low-sample item quality,
- breakout forecasting from early signals,
- uplift scoring for extra exposure,
- promotion scoring with a relevance floor,
- bandit policy comparison,
- creator exposure concentration metrics,
- IPS/SNIPS/doubly robust stress tests.

## Run

```powershell
python scripts/run_system_b_pipeline.py
python scripts/final_system_b_report.py
streamlit run dashboards/system_b_demo/app.py
```

Outputs:

```text
data/processed/system_b/
reports/system_b_final_report.md
```

## Main Files

```text
exposure_simulation/
  Builds the logged exposure table with propensities.

bayesian_shrinkage/
  Pulls tiny-sample quality rates toward a prior.

breakout_forecasting/
  Predicts future upside from early-window item features.

uplift_scoring/
  Estimates whether extra exposure is expected to help.

uncertainty_promotion/
  Combines quality, breakout, uplift, and uncertainty.

bandit_exploration/
  Compares popularity, epsilon-greedy, UCB1, and Thompson sampling.

fairness/
  Measures exposure concentration and relevance/fairness tradeoffs.

offline_eval/
  Runs IPS, clipped IPS, SNIPS, and doubly robust estimates.
```

## Current Evidence

The current artifact run reports:

```text
Items: 3000
Logged exposures: 135000
Breakout ROC-AUC: 0.6814
Average precision: 0.2422
Best reward among tested policies: epsilon_greedy
```

Read these artifacts first:

```text
promotion_scores.parquet
bandit_policy_metrics.parquet
fairness_metrics.parquet
ips_stress_test.parquet
pareto_frontier.parquet
```

## Limitations

- Exposure data is simulated.
- Uplift scores are not causal proof.
- IPS requires known propensities and policy overlap.
- Creator spam, safety, fatigue, and gaming are not modeled.
- A real system would need logged traffic and randomized exploration buckets.

Safe framing:

> System B demonstrates how to evaluate exploration policy before deployment. It does not claim live platform impact.
