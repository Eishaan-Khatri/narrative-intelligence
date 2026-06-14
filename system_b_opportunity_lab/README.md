# System B - Opportunity Lab

This folder contains the opportunity-ranking layer used during the project
build. The standalone System B repository is the cleaner place to review this
work now.

System B answers a different question from System A:

```text
Which underexposed items should get measured exploration traffic?
```

## What It Does

- Builds simulated exposure logs with known propensities.
- Shrinks noisy item-quality rates.
- Predicts possible breakout items.
- Estimates uplift from extra exposure.
- Compares bandit policies.
- Measures creator exposure concentration.
- Runs IPS, SNIPS, clipped IPS, and doubly robust checks.

## Run

```powershell
python scripts/run_system_b_pipeline.py
python scripts/final_system_b_report.py
streamlit run dashboards/system_b_demo/app.py
```

## Main Folders

```text
exposure_simulation/
  Builds exposure logs.

bayesian_shrinkage/
  Reduces noise in small samples.

breakout_forecasting/
  Finds items with possible future upside.

uplift_scoring/
  Estimates benefit from extra exposure.

uncertainty_promotion/
  Builds the promotion score.

bandit_exploration/
  Compares exploration policies.

fairness/
  Measures exposure concentration.

offline_eval/
  Runs off-policy checks.
```

## Limits

- The exposure log is simulated.
- Uplift is not causal proof.
- IPS needs known propensities and enough policy overlap.
- A real rollout would need randomized exploration traffic.
