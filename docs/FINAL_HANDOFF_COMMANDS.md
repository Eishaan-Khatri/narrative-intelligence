# Final Handoff Commands

Use these on the other system after cloning/pulling the repo.

## Recommended Run

```powershell
git pull
python scripts/run_full_gpu_handoff.py --install-main --download-gutenberg --large-synthetic --run-training --batch-size 1024
python scripts/final_artifact_report.py
git add data/processed data/synthetic reports
git commit -m "Update final System A artifacts"
git push
```

## With Amazon File

Use this if you have a local Amazon metadata/review file.

```powershell
python scripts/run_full_gpu_handoff.py --download-gutenberg --amazon-input E:\path\to\amazon.jsonl.gz --build-external-catalog --large-synthetic --run-training --batch-size 1024
python scripts/final_artifact_report.py
```

## Faster Training-Only Run

Use this if large synthetic regeneration is too slow.

```powershell
python scripts/check_training_ready.py
python scripts/run_gpu_training_suite.py --batch-size 1024
python scripts/final_artifact_report.py
```

## Manual Remaining Steps

- Download/choose the Amazon file if you want Amazon included.
- Wait for large synthetic rebuild and GPU training.
- Commit and push generated artifacts.
- Review `reports/system_a_final_artifact_report.md`.
