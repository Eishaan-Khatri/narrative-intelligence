# System A Decision Log

This log records the major approaches, corrections, and current final-run plan
for System A.

## Starting Point

The project began as System A of the Narrative Intelligence Platform:

- Feature store and Markov reading-event simulator
- Layer 1 content understanding: NMF topics, author embeddings, PCA quality score
- Layer 2 retrieval: two-tower model, hard-negative mining, FAISS index
- Layer 3 ranking: survival model and LambdaMART reranker
- Layer 4 evaluation: completion-weighted NDCG, ablation, oracle analysis
- Streamlit dashboard

Initial implementation was broad, but several parts were either unverified or
too synthetic for a strong submission.

## Main Problems Found

1. Catalog was too small.
   - Earlier catalog had only 500 items.
   - Recall@500 was inflated because top-500 covered almost the whole catalog.

2. Hard-negative mining hurt retrieval.
   - Phase 1 often gave the best result.
   - After hard negatives started, Recall@50 frequently collapsed.
   - Interpretation: mined negatives were often false negatives in sparse
     implicit-feedback data.

3. Tail discovery was weak.
   - Tail recall was much lower than mid/popular recall.
   - The model favored popular/easy items.

4. Reranker was not beating the baseline.
   - Ablation showed the full reranking stack below the baseline.
   - This made the dashboard/evaluation story mixed, not clean.

5. Evaluation metrics were initially too weak.
   - Needed Recall@10, Recall@20, Recall@50, MRR@10, NDCG@10, and tail/mid/popular split.

6. Real-data ingestion was incomplete.
   - Original plan mentioned Goodreads, Amazon Reviews, and Gutenberg.
   - Current project mainly uses synthetic user-session labels.
   - Gutenberg/Amazon can enrich content, but they do not create real reading behavior labels.

7. Simulator calibration was weak.
   - Previous full session artifact had very high abandonment.
   - Most sessions exited mid-chapter.
   - This made positive labels noisy and sparse.

8. Stylometric Gutenberg features were only partially handled.
   - Gutenberg downloader extracts basic text/stylometric fields.
   - Full production-grade text feature extraction is still limited.

## Corrections Already Made

### Dashboard

- Fixed dashboard metric key mismatch.
- Fixed Streamlit cache hashing issue by avoiding DataFrame hashing problem.
- Improved dashboard readability and reviewer-facing explanation.
- Added honest reporting around weak/mixed evaluation results.

### Git And Handoff

- Recreated/pushed clean repo history under:
  - `Eishaan Khatri <eishaankhatri@gmail.com>`
- Removed/ignored oversized synthetic event log:
  - `data/synthetic/events.parquet`
- Added clear GPU handoff scripts and docs.

### Retrieval Metrics

Added strict retrieval metrics:

- Recall@10
- Recall@20
- Recall@50
- MRR@10
- NDCG@10
- tail/mid/popular split

These are more meaningful than Recall@500 for small/medium catalogs.

### Two-Tower Training Controls

Added and tested controls for:

- phase-1-only training
- hard-negative loss weight
- hard-negative popularity balancing
- tail-positive oversampling
- learning-rate variants

Finding so far:

- Hard negatives often reduce loss but damage Recall@50.
- Phase-1-only plus tail oversampling is the safer family.

### Larger Synthetic Setup

Moved from small catalog to larger synthetic generation:

- 5k item run already completed earlier.
- Current final sweep supports 8k items and 4k users.

This makes top-k metrics harder and more realistic.

### External Data Scripts

Added/kept scripts for:

- Gutenberg download and normalization
- Amazon local-file normalization
- combined external catalog building
- synthetic catalog text enrichment using external text

Important constraint:

- Gutenberg can be downloaded by script.
- Amazon is not downloaded automatically; user must provide a local file.
- External text enriches item content but does not provide user-session labels.

### Simulator Calibration Controls

Added simulator parameters for:

- exit probability multiplier
- transition exit multiplier
- speed multiplier
- patience multiplier
- valley-of-death multiplier
- engaged-state boost multiplier
- quality distribution
- topic concentration
- taste concentration

Purpose:

- Test whether low retrieval scores were caused by overly harsh synthetic
  abandonment behavior.

### Catalog Text Enrichment

Added `scripts/enrich_synthetic_catalog_text.py`.

It preserves synthetic `item_id`s while adding richer descriptions from:

- genre templates
- latent topic tokens
- quality tokens
- Gutenberg/Amazon text when available

This makes NMF/topic/content features stronger without breaking synthetic
interaction labels.

### Final Research Sweep

Added `scripts/run_final_research_sweep.py`.

It runs multiple dataset variants and training variants in one GPU session,
then restores the best artifacts to canonical paths.

Current serious command:

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --gutenberg-large-list --gutenberg-limit 80 --profile super_extensive --num-users 4000 --num-items 8000 --sessions-per-user 30 --batch-size 1024
```

Outputs:

- `data/processed/final_sweep/final_research_sweep_summary.csv`
- `data/processed/final_sweep/final_research_sweep_session_calibration.csv`
- `data/processed/final_sweep/final_research_sweep_best_by_dataset.csv`
- `data/processed/final_sweep/final_research_sweep_best_by_training.csv`
- `reports/system_a_final_research_sweep.md`
- `reports/system_a_final_research_sweep.json`

## Final Sweep Variants

### Dataset/Simulator Variants

- `baseline_legacy_enriched`
  - Control: original simulator pressure with richer text.

- `calibrated_balanced`
  - Lower exit pressure and stronger engagement.

- `high_completion_stress`
  - Tests whether more clear positive completions improve retrieval.

- `app_like_balanced`
  - More app-like engagement distribution.

- `diverse_taste_clusters`
  - Sharper taste clusters and stronger topical separation.

- `tail_discovery_stress`
  - Harder tail-discovery setting.

- `long_tail_harder`
  - Stronger long-tail and niche structure.

### Training Variants

- `phase1_lr1e3`
  - Original phase-1-only baseline.

- `phase1_tail_lr5e4`
  - Previous best family: lower LR and tail oversampling.

- `phase1_tail_lr7e4`
  - Middle learning-rate point.

- `phase1_tail_lr3e4`
  - More conservative LR.

- `phase1_tail_lr2e4_long`
  - Longer conservative phase-1-only run.

- `phase1_tail_lr1e4_long`
  - Very conservative long run.

- `phase1_tail_strong_lr3e4`
  - Stronger tail-positive oversampling.

- `phase1_tail_extreme_lr2e4`
  - Extreme tail-positive oversampling.

- `phase1_tail_weightdecay_lr3e4`
  - Adds stronger regularization.

- `mild_hardneg_tail`
  - Low hard-negative pressure.

- `late_mild_hardneg_tail`
  - Delayed hard-negative mining.

- `pop_balanced_hardneg`
  - Popularity-balanced hard negatives.

- `ultra_low_hardneg_tail`
  - Hard negatives only as weak ordering regularizer.

## Current Hard-Negative Finding

Observed pattern:

```text
Before hard negatives:
R@50 around 0.02
R@500 around 0.16

After hard negatives:
loss drops sharply
R@50 collapses
R@500 collapses
```

Interpretation:

- Loss is decreasing because the objective changes.
- Retrieval quality collapses because the hard negatives are likely false negatives.
- In sparse implicit feedback, unseen does not mean disliked.

Scientific explanation:

> Hard-negative mining introduced false-negative pressure in a sparse
> implicit-feedback recommendation setting. Although the training loss dropped,
> validation Recall@50 collapsed, so final model selection favored phase-1
> in-batch negatives with tail-positive oversampling.

## What Is Still Not Solved Fully

1. True real-user behavior data is still missing.
   - Goodreads/Amazon/Gutenberg metadata does not equal real user reading sessions.

2. Amazon is not auto-downloaded.
   - It requires a local file because public sources differ and may require
     terms, login, or credentials.

3. Reranker quality is still uncertain.
   - LambdaMART labels need deeper validation.
   - Survival penalty may need tuning.

4. Tail recall may still be weak.
   - The final sweep tests tail methods, but results decide whether it improved.

5. Final score depends on GPU run output.
   - The code is ready, but final evidence comes from the generated reports.

## Final Submission Framing

Best honest framing:

- This is a working personalization research pipeline, not a production-scale recommender.
- It implements feature store, content understanding, retrieval, ranking, evaluation, and dashboard.
- It tests multiple scientifically motivated corrections.
- It reports failures honestly, especially hard-negative collapse.
- It uses external text enrichment when available, but does not falsely claim real behavioral training data.

## Final Commands

Without Amazon:

```powershell
git pull
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --gutenberg-large-list --gutenberg-limit 80 --profile super_extensive --num-users 4000 --num-items 8000 --sessions-per-user 30 --batch-size 1024
python scripts/final_artifact_report.py
git add data/processed data/synthetic/catalog.parquet data/synthetic/users.parquet reports
git commit -m "Update final System A research sweep artifacts"
git push
```

With Amazon:

```powershell
python scripts/run_final_research_sweep.py --install-main --download-gutenberg --gutenberg-large-list --gutenberg-limit 80 --amazon-input E:\path\to\amazon.jsonl.gz --amazon-limit 50000 --build-external-catalog --profile super_extensive --num-users 4000 --num-items 8000 --sessions-per-user 30 --batch-size 1024
```

Do not commit:

```text
data/synthetic/events.parquet
```
