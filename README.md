# Narrative Intelligence Platform

Two linked recommendation projects:

- **System A**: a reading recommender that turns reading sessions into retrieval, ranking, and evaluation artifacts.
- **System B**: a platform-side opportunity lab for deciding which underexposed items should receive measured exploration traffic.

The repo is intentionally honest about its limits. The behavior data is synthetic. Gutenberg text is used for content enrichment. Amazon can be added if a local dataset file is available. The project should be read as an end-to-end research and evaluation pipeline, not as a production recommender benchmark.

## Current State

System A has a complete pipeline:

1. simulate reading events,
2. build session and temporal features,
3. build item topics, author features, quality scores, and item fingerprints,
4. train a two-tower retrieval model,
5. build a FAISS index,
6. run survival/reranking/evaluation,
7. show the results in a Streamlit dashboard.

System B has a separate policy layer:

1. build simulated exposure logs,
2. shrink noisy item quality estimates,
3. forecast breakout candidates,
4. estimate uplift from exploration,
5. compare bandit policies,
6. measure exposure concentration,
7. stress-test off-policy estimates.

## Latest System A Result

The final sweep ran 91 retrieval experiments over 8,000 items and about 239k sessions.

Best run:

```text
Dataset variant: app_like_balanced
Training variant: phase1_tail_strong_lr3e4
Recall@10: 0.0057
Recall@20: 0.0112
Recall@50: 0.0268
Tail Recall@50: 0.0118
MRR@10: 0.0019
NDCG@10: 0.0028
```

Interpretation:

- Simulator calibration helped: the old baseline had about 10% mean completion; the selected app-like dataset has about 37%.
- Tail-positive oversampling was more useful than aggressive hard-negative mining.
- Hard negatives often reduced loss while hurting recall, which points to false-negative pressure in sparse implicit feedback.
- Retrieval quality is still modest. Tail discovery remains the main weakness.

Use these files for the final System A story:

```text
reports/system_a_final_research_sweep.md
data/processed/final_sweep/final_research_sweep_summary.csv
data/processed/final_sweep/final_research_sweep_best_by_dataset.csv
data/processed/final_sweep/final_research_sweep_best_by_training.csv
```

## Latest System B Result

System B is stronger as a policy/evaluation artifact than as a live product claim.

Current generated report:

```text
reports/system_b_final_report.md
```

Headline results:

```text
Items: 3000
Logged exposures: 135000
Breakout ROC-AUC: 0.6814
Average precision: 0.2422
Best simulated bandit reward: epsilon_greedy
```

Interpretation:

- Bayesian shrinkage keeps low-sample items from dominating.
- Uplift separates "good item" from "item worth exploring."
- Bandit comparisons show the reward/discovery tradeoff.
- IPS diagnostics show when offline estimates stop being trustworthy.
- The exposure log is simulated, so none of this is live causal evidence.

## Run The Dashboards

```powershell
streamlit run dashboards/system_a_demo/app.py
streamlit run dashboards/system_b_demo/app.py
```

## Rebuild System A

For the final GPU sweep:

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --gutenberg-large-list --gutenberg-limit 80 --profile super_extensive --num-users 4000 --num-items 8000 --sessions-per-user 30 --batch-size 1024
python scripts/final_artifact_report.py
```

If CUDA memory is tight, lower the batch size:

```powershell
--batch-size 512
```

Do not commit `data/synthetic/events.parquet`. Full event logs can exceed GitHub limits and are intentionally ignored.

## Rebuild System B

```powershell
python scripts/run_system_b_pipeline.py
python scripts/final_system_b_report.py
```

## External Data

Gutenberg:

- downloaded by script,
- used as text enrichment,
- does not provide user-session labels.

Amazon:

- not downloaded automatically,
- requires a local JSONL/CSV/Parquet file,
- can enrich item metadata,
- does not replace real logged behavior.

Example:

```powershell
python scripts/run_final_research_sweep.py --amazon-input E:\path\to\amazon.jsonl.gz --amazon-limit 50000 --build-external-catalog
```

## Repository Map

```text
feature_store/
  simulator and session/temporal feature builders

system_a_discovery_engine/
  content features, two-tower retrieval, FAISS, survival, reranking, evaluation

system_b_opportunity_lab/
  shrinkage, breakout forecasting, uplift, bandits, fairness, IPS evaluation

dashboards/
  Streamlit dashboards for System A and System B

scripts/
  rebuild, training, reporting, and handoff scripts

reports/
  generated summaries from the latest artifacts
```

## What Not To Claim

Do not claim:

- production recommendation quality,
- real user lift,
- real causal uplift,
- mature long-tail retrieval performance.

Safe claim:

> This is a complete offline recommendation and policy-evaluation pipeline. It exposes where the recommender works, where it fails, and which corrections were tested.
