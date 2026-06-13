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

Full default training:

```powershell
python scripts/train_two_tower.py --epochs 15 --phase1-epochs 5 --batch-size 1024
```

Outputs are written to `data/processed/`:

- `two_tower_model.pt`
- `item_embeddings.parquet`
- `user_embeddings.parquet`
- `training_curves.png`

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
