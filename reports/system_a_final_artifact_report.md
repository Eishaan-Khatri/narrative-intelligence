# System A Final Artifact Report

## Retrieval Metrics
- Recall@10: 0.0104
- Recall@20: 0.0210
- Recall@50: 0.0447
- Tail_Recall@50: 0.0106
- MRR@10: 0.0032
- NDCG@10: 0.0048

## GPU Suite
- Best listed experiment: phase1_tail_lr5e4
- Selection score: 0.0508545723502606

## Dataset Inputs
- synthetic_catalog: exists=True, rows=5000
- synthetic_events: exists=True, rows=2815365
- gutenberg_catalog: exists=True, rows=10
- amazon_catalog: exists=False, rows=None
- external_catalog_combined: exists=False, rows=None

## Required Artifacts
- OK: session_features.parquet (6664030 bytes, rows=119949, columns=15)
- OK: user_temporal_features.parquet (167761 bytes, rows=3000, columns=5)
- OK: topic_vectors.parquet (950988 bytes, rows=5000, columns=41)
- OK: author_embeddings.parquet (97806 bytes, rows=995, columns=33)
- OK: quality_scores.parquet (393620 bytes, rows=5000, columns=14)
- OK: item_fingerprints.parquet (1225042 bytes, rows=5000, columns=82)
- OK: two_tower_model.pt (502944 bytes)
- OK: item_embeddings.parquet (3703565 bytes, rows=5000, columns=129)
- OK: user_embeddings.parquet (12929038 bytes, rows=16937, columns=129)
- OK: retrieval_metrics.parquet (7205 bytes, rows=128, columns=9)
- OK: faiss_index.bin (303252 bytes)
- OK: lambdamart_model.txt (693505 bytes)
- OK: ablation_results.parquet (5569 bytes, rows=5, columns=8)
- OK: completion_ndcg_metrics.parquet (2927 bytes, rows=3, columns=4)
- OK: oracle_analysis.parquet (7299 bytes, rows=8, columns=11)
