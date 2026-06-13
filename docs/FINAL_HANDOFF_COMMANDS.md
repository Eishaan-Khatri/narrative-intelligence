# Final Handoff Commands

Use these on the other system after cloning/pulling the repo.

## Recommended Final Research Sweep

```powershell
git pull
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --profile exhaustive --num-users 3000 --num-items 5000 --sessions-per-user 25 --batch-size 1024
python scripts/final_artifact_report.py
git add data/processed data/synthetic/catalog.parquet data/synthetic/users.parquet reports
git commit -m "Update final System A artifacts"
git push
```

Do not add `data/synthetic/events.parquet`; it is intentionally ignored because
full event logs can exceed GitHub's file size limit.

## With Amazon File

Use this if you have a local Amazon metadata/review file.

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --amazon-input E:\path\to\amazon.jsonl.gz --build-external-catalog --profile exhaustive --num-users 3000 --num-items 5000 --sessions-per-user 25 --batch-size 1024
python scripts/final_artifact_report.py
```

Amazon/Gutenberg text is used as content enrichment for the synthetic catalog.
It does not create real user-session labels by itself.

## Faster Fallback Run

Use this if the final sweep is too slow.

```powershell
python scripts/check_training_ready.py
python scripts/run_gpu_training_suite.py --batch-size 1024
python scripts/final_artifact_report.py
```

## Manual Remaining Steps

- Download/choose the Amazon file if you want Amazon included.
- Wait for simulator variants, feature rebuilds, and GPU training.
- Commit and push generated artifacts.
- Review `reports/system_a_final_artifact_report.md`.
- Review `reports/system_a_final_research_sweep.md`.
