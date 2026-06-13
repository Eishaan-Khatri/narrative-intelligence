# Transfer and Two-Tower Training Guide

This folder is now runnable for the System A two-tower retrieval training path.
The verified training path is:

1. `data/processed/session_features.parquet`
2. `data/processed/topic_vectors.parquet`
3. `data/processed/author_embeddings.parquet`
4. `data/processed/quality_scores.parquet`
5. `data/processed/item_fingerprints.parquet`
6. `scripts/train_two_tower.py`

The two-tower trainer automatically uses real processed artifacts when
`session_features.parquet` and `item_fingerprints.parquet` exist. It falls back
to synthetic toy data only when those artifacts are missing or invalid.

## Transfer Checklist

Copy the full `narrative-intelligence-platform` folder, including:

- `feature_store/`
- `system_a_discovery_engine/`
- `dashboards/`
- `scripts/`
- `tests/`
- `data/synthetic/`
- `data/processed/`
- `requirements.txt`
- `run_pipeline.py`

Do not copy only the Python files. The current processed artifacts are needed if
you want the next machine to start training immediately.

## Environment Setup

Recommended Python: `3.12` or newer.

Create and activate an environment:

```powershell
cd D:\Projects\narrative-intelligence-platform
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional survival-model extra:

```powershell
python -m pip install -r requirements-optional.txt
```

If `requirements-optional.txt` fails, the project still runs the survival layer
with CoxPH only. Random Survival Forest comparison is skipped.

For GPU training, install the PyTorch build that matches the target machine's
CUDA version from the official PyTorch selector. Then verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Verify The Copied Folder

Run:

```powershell
python scripts/check_training_ready.py
```

Expected result:

```text
[OK] Folder is ready for two-tower training.
```

If `item_fingerprints.parquet` is missing, rebuild the required upstream stages:

```powershell
python run_pipeline.py --step nmf_topics
python run_pipeline.py --step author_embeddings
python run_pipeline.py --step quality_scores
```

## Train Two-Tower Model

Quick smoke test:

```powershell
python scripts/train_two_tower.py --epochs 1 --phase1-epochs 1 --batch-size 512
```

Recommended if GPU access is limited:

```powershell
python scripts/run_gpu_training_suite.py --batch-size 1024
```

This runs three retrieval experiments in one session:

- Phase 1 only, learning rate `1e-3`
- Phase 1 only with tail-positive oversampling, learning rate `5e-4`
- Tuned hard negatives with lower loss weight, tail-positive oversampling, and
  popularity-balanced hard-negative sampling

The suite saves each run under `data/processed/experiments/`, writes
`data/processed/gpu_training_suite_summary.csv`, restores the best run to the
normal artifact paths, and then runs downstream FAISS/ranking/evaluation steps.

Manual fallback commands:

```powershell
# 1. Baseline: no hard-negative phase, because the previous run degraded after epoch 5.
python scripts/train_two_tower.py --epochs 5 --phase1-only --batch-size 1024 --learning-rate 1e-3

# 2. Phase 1 with tail-positive oversampling.
python scripts/train_two_tower.py --epochs 8 --phase1-only --batch-size 1024 --learning-rate 5e-4 --tail-oversample-factor 3

# 3. Tuned Phase 2: lower hard-negative pressure, tail positives, and popularity-balanced hard negatives.
python scripts/train_two_tower.py --epochs 15 --phase1-epochs 5 --batch-size 1024 --learning-rate 5e-4 --hard-negative-weight 0.25 --tail-oversample-factor 3 --hard-negative-popularity-alpha 0.75
```

Compare the two runs using `data/processed/retrieval_metrics.parquet`. Treat
`Recall@10`, `Recall@20`, `Recall@50`, `MRR@10`, `NDCG@10`, and tail/mid/popular
split rows as the real retrieval report. `Recall@500` is only a ceiling
diagnostic because the current catalog can be small enough for top-500 to cover
most or all items.

Outputs are written to `data/processed/`:

- `two_tower_model.pt`
- `item_embeddings.parquet`
- `user_embeddings.parquet`
- `training_curves.png`
- `retrieval_metrics.parquet`
- `gpu_training_suite_summary.csv` when using the suite script

## Larger Catalog Regeneration

The simulator can now be run directly with larger catalog settings:

```powershell
python feature_store/simulator/markov_event_simulator.py --num-users 3000 --num-items 5000 --sessions-per-user 20
python run_pipeline.py --from session_features
```

This is slower than retrieval-only training. Use it only when you have enough
time on the target machine to regenerate the upstream feature store.

## Current Limitations

- The full Markov event file has about 1.4M events. Session feature extraction is
  correct but slow on CPU; expect roughly 10-12 minutes on this Windows machine
  for the current synthetic dataset.
- FAISS indexing, survival modeling, and the Streamlit dashboard require their
  own optional dependencies and should be validated separately on the target
  system.
- The current processed session file is a 5,000-row synthetic fallback generated
  by the author embedding stage. It is sufficient for verifying the training
  path, but a serious training run should regenerate `session_features.parquet`
  from the full event log first.
- The current retrieval catalog is still synthetic and small compared with the
  original blueprint. For a stronger submission, regenerate with a larger
  catalog and then rerun `python run_pipeline.py --from session_features`.
