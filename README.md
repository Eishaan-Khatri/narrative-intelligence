# Narrative Intelligence Platform - System A

System A is a reading recommender.

Think of a reading app. One story gets opened and closed after one chapter.
Another story gets read slowly, bookmarked, and finished. A normal recommender
may count both as a click. This project tries to tell the difference.

It turns reading events into model features, trains a two-tower retrieval model,
reranks candidates with reading-quality and dropout signals, and reports where
the system works and where it still misses.

Related project: System B is kept in a separate repository.

## What This Project Does

- Rebuilds reading sessions from raw events.
- Creates user, session, author, topic, and item features.
- Scores content quality from completion, return, structure, and engagement signals.
- Trains a two-tower retrieval model in PyTorch.
- Builds a FAISS vector index for fast candidate search.
- Adds dropout-risk and LambdaMART reranking signals.
- Measures ranking quality with top-k metrics and ablation tables.
- Shows the current artifacts in a Streamlit dashboard.

## The Main Idea

A click is weak evidence.

If a reader opens a story, reads 3%, and leaves, that should not count the same
as a story they finish. System A uses signals like reading speed, completion,
return behavior, chapter progress, author history, and item quality to build a
better recommendation pipeline.

The project has five layers:

```text
1. Event simulator
   Creates open, scroll, pause, completion, and exit events.

2. Feature store
   Turns raw events into reusable tables for users, sessions, and items.

3. Content features
   Builds topics, author embeddings, quality scores, and item fingerprints.

4. Retrieval
   Uses a two-tower model to find likely candidate items quickly.

5. Ranking and evaluation
   Reranks candidates and checks Recall@10, Recall@50, MRR@10, NDCG@10,
   tail recall, and ablation results.
```

## Current Saved Result

The current saved retrieval run uses a larger 8,000-item catalog. These are the
main dashboard metrics from the saved artifacts:

```text
Recall@10:       0.0057
Recall@20:       0.0112
Recall@50:       0.0268
Tail Recall@50:  0.0118
MRR@10:          0.0019
NDCG@10:         0.0028
```

How to read this:

- Recall@50 means: out of the items a user later liked or finished, how many
  showed up in the top 50 retrieved candidates.
- Tail Recall@50 checks the same thing only for less popular items.
- MRR@10 rewards the model when the first useful item appears near the top.
- NDCG@10 rewards useful items more when they appear higher in the list.

The result is not high yet. The useful part is that the project now measures the
right failure: tail items are still hard to retrieve.

## What Improved During The Final Runs

- The catalog moved from a tiny setup to a larger 8,000-item setup.
- Recall@500 stopped being the headline metric because it is too easy to inflate.
- The training script now reports Recall@10, Recall@20, Recall@50, MRR@10,
  NDCG@10, and tail/mid/popular splits.
- Tail-positive oversampling worked better than heavy hard-negative mining.
- Hard-negative mining often made the loss look better while recall got worse.

That last point matters. A lower loss is not always a better recommender.

## Run The Dashboard

```powershell
streamlit run dashboards/system_a_demo/app.py
```

The dashboard reads the current files under:

```text
data/processed/
data/synthetic/
```

## Rebuild Or Train

Basic readiness check:

```powershell
python scripts/check_training_ready.py
```

Train the two-tower model:

```powershell
python scripts/train_two_tower.py
```

Run the larger research sweep:

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --gutenberg-large-list --gutenberg-limit 80 --profile super_extensive --num-users 4000 --num-items 8000 --sessions-per-user 30 --batch-size 1024
python scripts/final_artifact_report.py
```

Use `--batch-size 512` if CUDA memory is tight.

## Important Files

```text
feature_store/
  schema.py
  build_session_features.py
  build_temporal_features.py
  simulator/markov_event_simulator.py

system_a_discovery_engine/layer1_content/
  nmf_topics.py
  author_embeddings.py
  quality_score_pca.py

system_a_discovery_engine/layer2_retrieval/
  two_tower_model.py
  train_loop.py
  faiss_index.py

system_a_discovery_engine/layer3_ranking/
  survival_model.py
  lambdamart_ranker.py

system_a_discovery_engine/layer4_evaluation/
  completion_ndcg.py
  ablation_runner.py
  retrieval_oracle.py

dashboards/system_a_demo/app.py
  Streamlit dashboard for the saved artifacts.
```

## Data Notes

The project mainly uses synthetic reading behavior. Gutenberg text is used for
content enrichment. The important point is that user-session behavior is not
real production traffic.

Do not commit the full synthetic event log:

```text
data/synthetic/events.parquet
```

It can exceed GitHub's file-size limit.

## Limits

- The behavior data is synthetic.
- The model is not trained on live user traffic.
- The current retrieval score is still modest.
- Tail discovery is still the main weak point.
- The dashboard is for project review, not production monitoring.

The strongest next step is to train on real logged reading sessions with known
impressions, completions, and returns.
