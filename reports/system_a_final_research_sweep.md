# System A Final Research Sweep

## Best Run
- Dataset variant: app_like_balanced
- Training variant: phase1_tail_strong_lr3e4
- Recall@10: 0.0057
- Recall@20: 0.0112
- Recall@50: 0.0268
- Tail Recall@50: 0.0118
- MRR@10: 0.0019
- NDCG@10: 0.0028
- Selection score: 0.036894

## Scientific Interventions Tested
- Simulator calibration: lower exit pressure, lower valley-of-death churn, higher patience, faster chapter progress.
- Catalog/content enrichment: external Gutenberg/Amazon text when available, otherwise richer genre/topic descriptions.
- Retrieval variants: phase-1-only baselines, tail-positive oversampling, conservative learning rates, weight decay, downweighted hard negatives, late hard negatives, popularity-balanced hard negatives.

## Interpretation Rule
If hard-negative variants lose to phase-1-only variants, report that hard negatives introduced false negatives/noisy pressure and were downweighted or disabled.
If high-completion datasets improve retrieval, report that the previous simulator produced too few positive labels for a stable recommender objective.

## Output Files
- data\processed\final_sweep\final_research_sweep_summary.csv
- data\processed\final_sweep\final_research_sweep_session_calibration.csv
- data\processed\final_sweep\final_research_sweep_best_by_dataset.csv
- data\processed\final_sweep\final_research_sweep_best_by_training.csv
- reports\system_a_final_research_sweep.json
