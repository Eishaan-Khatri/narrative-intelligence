# System A Final Artifact Report

## Retrieval Metrics
- Recall@10: 0.0057
- Recall@20: 0.0112
- Recall@50: 0.0268
- Tail_Recall@50: 0.0118
- MRR@10: 0.0019
- NDCG@10: 0.0028

## GPU Suite
- Best listed experiment: phase1_tail_lr5e4
- Selection score: 0.0508545723502606

## Dataset Inputs
- synthetic_catalog: exists=True, rows=8000
- synthetic_events: exists=True, rows=8516636
- gutenberg_catalog: exists=True, rows=76
- amazon_catalog: exists=False, rows=None
- external_catalog_combined: exists=True, rows=76

## Required Artifacts
- OK: session_features.parquet (13284581 bytes, rows=239362, columns=15)
- OK: user_temporal_features.parquet (243764 bytes, rows=4300, columns=5)
- OK: topic_vectors.parquet (1113272 bytes, rows=8000, columns=41)
- OK: author_embeddings.parquet (146528 bytes, rows=1588, columns=33)
- OK: quality_scores.parquet (632463 bytes, rows=8000, columns=14)
- OK: item_fingerprints.parquet (1532224 bytes, rows=8000, columns=82)
- OK: two_tower_model.pt (502944 bytes)
- OK: item_embeddings.parquet (5884354 bytes, rows=8000, columns=129)
- OK: user_embeddings.parquet (40184502 bytes, rows=52159, columns=129)
- OK: retrieval_metrics.parquet (8387 bytes, rows=224, columns=9)
- OK: faiss_index.bin (375252 bytes)
- OK: lambdamart_model.txt (693505 bytes)
- OK: ablation_results.parquet (5569 bytes, rows=5, columns=8)
- OK: completion_ndcg_metrics.parquet (2927 bytes, rows=3, columns=4)
- OK: oracle_analysis.parquet (7299 bytes, rows=8, columns=11)
