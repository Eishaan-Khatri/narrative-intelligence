# Final Research Sweep Guide

Use this when GPU access is limited and you want one long run that tests the
main scientific fixes.

## What This Run Tests

- Simulator calibration: lower exit pressure, higher patience, faster chapter
  progress, weaker valley-of-death churn.
- Dataset realism: 5k-item catalog, sharper user/item taste clusters, richer
  catalog text, optional Gutenberg/Amazon text enrichment.
- Retrieval training: phase-1-only training, tail-positive oversampling,
  lower learning rates, downweighted hard negatives, popularity-balanced hard
  negatives.
- Evaluation: Recall@10, Recall@20, Recall@50, MRR@10, NDCG@10, and tail split.

## Main Command

```powershell
git pull
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --profile exhaustive --num-users 3000 --num-items 5000 --sessions-per-user 25 --batch-size 1024
python scripts/final_artifact_report.py
```

## With Amazon Data

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --amazon-input E:\path\to\amazon.jsonl.gz --build-external-catalog --profile exhaustive --num-users 3000 --num-items 5000 --sessions-per-user 25 --batch-size 1024
python scripts/final_artifact_report.py
```

## Faster But Still Useful

```powershell
python scripts/run_final_research_sweep.py --profile standard --num-users 2500 --num-items 5000 --sessions-per-user 20 --batch-size 1024
```

## Output Files

- `data/processed/final_sweep/final_research_sweep_summary.csv`
- `data/processed/final_sweep/final_research_sweep_session_calibration.csv`
- `reports/system_a_final_research_sweep.md`
- `reports/system_a_final_research_sweep.json`
- `reports/system_a_final_artifact_report.md`

The best run is automatically copied back to:

- `data/processed/two_tower_model.pt`
- `data/processed/item_embeddings.parquet`
- `data/processed/user_embeddings.parquet`
- `data/processed/retrieval_metrics.parquet`

## What To Commit

```powershell
git add data/processed data/synthetic/catalog.parquet data/synthetic/users.parquet reports
git commit -m "Update final System A research sweep artifacts"
git push
```

Do not commit `data/synthetic/events.parquet`. It is ignored because full event
logs can exceed GitHub's file limit.

## How To Explain The Experiment

If phase-1-only wins, say hard-negative mining likely created false negatives in
an implicit-feedback reading setting, so the final system used in-batch
negatives plus tail-positive oversampling.

If calibrated simulator variants win, say the earlier simulator produced too
many abandonment-only labels, so you calibrated the event generator to produce a
more useful range of partial and completed reading sessions.

If external text helps content features but not retrieval, say external text
improved content representation but cannot replace real user-session labels.
