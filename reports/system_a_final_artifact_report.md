# System A Final Artifact Report

## Retrieval Metrics
- retrieval_metrics.parquet is missing.

## GPU Suite
- gpu_training_suite_summary.csv is missing.

## Dataset Inputs
- synthetic_catalog: exists=True, rows=500
- synthetic_events: exists=True, rows=1425071
- gutenberg_catalog: exists=False, rows=None
- amazon_catalog: exists=False, rows=None
- external_catalog_combined: exists=False, rows=None

## Required Artifacts
- OK: session_features.parquet (3333274 bytes, rows=60054, columns=15)
- OK: user_temporal_features.parquet (60995 bytes, rows=1000, columns=5)
- OK: topic_vectors.parquet (107039 bytes, rows=500, columns=41)
- OK: author_embeddings.parquet (26504 bytes, rows=98, columns=33)
- OK: quality_scores.parquet (55856 bytes, rows=500, columns=14)
- OK: item_fingerprints.parquet (163411 bytes, rows=500, columns=82)
- OK: two_tower_model.pt (502944 bytes)
- OK: item_embeddings.parquet (400122 bytes, rows=500, columns=129)
- OK: user_embeddings.parquet (4365665 bytes, rows=5910, columns=129)
- MISSING: retrieval_metrics.parquet (0 bytes)
- OK: faiss_index.bin (195252 bytes)
- OK: lambdamart_model.txt (697130 bytes)
- OK: ablation_results.parquet (3814 bytes, rows=5, columns=5)
- OK: completion_ndcg_metrics.parquet (2927 bytes, rows=3, columns=4)
- OK: oracle_analysis.parquet (7283 bytes, rows=8, columns=11)
