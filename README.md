# Narrative Intelligence Platform
## System A — Adaptive Discovery & Personalization Engine

> A reading platform's recommendation system that scores content based on how users actually consume it — not just whether they clicked — through a layered pipeline of behavioral signal engineering, content understanding, scalable retrieval, survival-aware re-ranking, and an evaluation methodology that penalizes the "opened but not read" failure mode.

---

## Architecture

```
                         RAW EVENT STREAM
                  (impressions, opens, scroll, pause,
                   exit, completions, ratings, follows)
                              │
                              ▼
                   SHARED BEHAVIORAL FEATURE STORE
                   ┌──────────┼──────────┐
                   │          │          │
                   ▼          ▼          ▼
            session_     user_       item_
            features   temporal   fingerprint
                   │    features     (81-dim)
                   │          │          │
                   └──────────┼──────────┘
                              │
                   ┌──────────┼──────────┐
                   │                     │
                   ▼                     ▼
            LAYER 2                LAYER 3
         Two-Tower               Survival Model
         Retrieval                (Cox PH / RSF)
         (PyTorch +                    │
          FAISS)                       ▼
             │                  dropout_hazard
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                    LAYER 3b
               LambdaMART Re-Ranker
               (LightGBM, 8 features)
                         │
                         ▼
                    LAYER 4
               Evaluation Framework
            (Completion-Weighted NDCG)
            (5-Model Ablation Study)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# 2. Run the full pipeline end-to-end
python run_pipeline.py

# 3. Launch the demo dashboard
streamlit run dashboards/system_a_demo/app.py
```

## Two-Tower Training Transfer

For moving this folder to another CPU/GPU machine and training the retrieval
model, use the dedicated guide:

```bash
python scripts/check_training_ready.py
python scripts/train_two_tower.py --epochs 1 --phase1-epochs 1 --batch-size 512
python scripts/train_two_tower.py --epochs 5 --phase1-only --batch-size 1024
python scripts/train_two_tower.py --epochs 15 --phase1-epochs 5 --batch-size 1024 --hard-negative-weight 0.25 --tail-oversample-factor 3
```

Full instructions: `docs/TRANSFER_TRAINING_GUIDE.md`.

The two-tower trainer uses real processed artifacts when
`data/processed/session_features.parquet` and
`data/processed/item_fingerprints.parquet` exist. It falls back to synthetic
toy data only when those artifacts are absent or invalid.

The trainer now writes `data/processed/retrieval_metrics.parquet` with
Recall@10, Recall@20, Recall@50, MRR@10, NDCG@10, and tail/mid/popular splits.
Use Recall@500 only as a ceiling diagnostic on the current small catalog.

## Repository Structure

```
narrative-intelligence-platform/
├── README.md
├── requirements.txt
├── run_pipeline.py                     # End-to-end orchestrator
├── data/
│   ├── raw/                            # Downloaded datasets
│   ├── synthetic/                      # Simulated event streams
│   └── processed/                      # Parquet feature store outputs
├── feature_store/
│   ├── schema.py                       # Pydantic schemas for all data
│   ├── build_session_features.py       # Session reconstruction + signals
│   ├── build_temporal_features.py      # User engagement profiles
│   ├── build_item_fingerprint.py       # 81-dim content fingerprint
│   └── simulator/
│       └── markov_event_simulator.py   # Calibrated telemetry simulator
├── system_a_discovery_engine/
│   ├── layer1_content/
│   │   ├── nmf_topics.py               # 40-component NMF topic model
│   │   ├── author_embeddings.py        # Time-decayed author representations
│   │   └── quality_score_pca.py        # 12-signal PCA quality composite
│   ├── layer2_retrieval/
│   │   ├── two_tower_model.py          # PyTorch user/item towers
│   │   ├── train_loop.py              # BPR + hard negative mining
│   │   └── faiss_index.py             # IVF-PQ + cold-start flat index
│   ├── layer3_ranking/
│   │   ├── survival_model.py          # Cox PH + Random Survival Forest
│   │   └── lambdamart_ranker.py       # LightGBM LambdaMART re-ranker
│   └── layer4_evaluation/
│       ├── completion_ndcg.py         # Completion-weighted NDCG
│       ├── ablation_runner.py         # 5-model ablation study
│       └── retrieval_oracle.py        # Retrieval ceiling analysis
└── dashboards/
    └── system_a_demo/
        └── app.py                     # Streamlit demo + feature attribution
```

## Cross-System Integration Points (System A → System B)

| Output | Dim | Consumed By |
|---|---|---|
| `item_fingerprint` | 81 | B: Breakout forecasting features |
| `item_embedding` | 128 | B: Audience clustering, breakout model |
| `user_embedding` | 128 | B: Bandit context vector |
| `dropout_hazard` | 1 | B: Intervention candidate flagging |

## Tech Stack

| Category | Libraries |
|---|---|
| Core | pandas, numpy, pyarrow, duckdb |
| ML | scikit-learn, lightgbm, torch, faiss-cpu, lifelines, scikit-survival, mapie |
| NLP | nltk, vaderSentiment |
| Stats | scipy, statsmodels |
| Viz | matplotlib, plotly, streamlit |
| Utils | tqdm, pydantic |
