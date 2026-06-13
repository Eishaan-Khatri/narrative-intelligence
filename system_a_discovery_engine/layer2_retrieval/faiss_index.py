"""
Narrative Intelligence Platform — System A, Layer 2
====================================================
FAISS Indexing + Cold-Start Retrieval
--------------------------------------

**Index architecture**

Main index — ``IndexIVFPQ(quantizer, d=128, nlist=100, m=16, nbits=8)``
    * ``quantizer``: ``IndexFlatIP(128)`` — inner product for unit-normalised
      embeddings (equivalent to cosine similarity).
    * PQ sub-quantisation compresses each 128-dim vector into 16 bytes,
      enabling memory-efficient approximate nearest-neighbour search.

Cold-start index — ``IndexFlatIP(128)``
    * A brute-force flat index containing only *recently published* items
      (proxy: newest 5 % of the catalog).
    * This ensures newly ingested items are not buried by an IVF partition
      that was trained before they existed.

**Query function**
    ``query(user_embedding, k_main=450, k_cold=50)``
    retrieves ``k_main`` candidates from the IVF-PQ index and ``k_cold``
    from the cold-start index, de-duplicates, and returns the merged
    top-500 by inner-product score.

**Benchmarks**
    * *Recall-vs-latency* for ``nprobe ∈ {1, 4, 16, 64, 128}`` — plots the
      Pareto front and saves to ``data/processed/faiss_benchmark.png``.
    * *Cold-start discovery rate* — fraction of cold-start items appearing
      in any user's top-500, with and without the flat sidecar index.

**Design decisions**
    * ``nlist = 100`` keeps the number of Voronoi cells manageable for
      catalogs of 500 – 100 000 items.  For larger catalogs we would scale
      ``nlist ≈ 4 × sqrt(N)``.
    * ``m = 16`` (sub-quantisers) divides the 128-dim vector into 16
      sub-vectors of 8 dims each, a standard PQ configuration.
    * ``nbits = 8`` gives 256 centroids per sub-quantiser, the most common
      setting for IVFPQ.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the import path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu (or faiss-gpu) is required.  "
        "Install with: pip install faiss-cpu>=1.7.4"
    )

from feature_store.schema import ITEM_EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data" / "processed"

EMBEDDING_DIM: int = ITEM_EMBEDDING_DIM  # 128
NLIST: int = 100          # IVF Voronoi cells
PQ_M: int = 16            # PQ sub-quantisers
PQ_NBITS: int = 8         # bits per sub-quantiser
COLD_START_FRACTION: float = 0.05  # newest 5 % treated as cold-start
BENCHMARK_NPROBES: Tuple[int, ...] = (1, 4, 16, 64, 128)
BENCHMARK_N_QUERIES: int = 1_000
K_MAIN: int = 450
K_COLD: int = 50
K_TOTAL: int = K_MAIN + K_COLD  # 500


# ═══════════════════════════════════════════════════════════════════════════
# Index Builder
# ═══════════════════════════════════════════════════════════════════════════

class FaissRetriever:
    """Manages the main IVF-PQ index and a cold-start flat sidecar.

    Attributes:
        main_index:      faiss.IndexIVFPQ trained on all item embeddings.
        cold_index:      faiss.IndexFlatIP for cold-start items.
        cold_item_ids:   np.ndarray of original item indices in the cold index.
        all_embeddings:  (N, 128) reference copy for ground-truth recall.
        n_items:         Total catalog size.
    """

    def __init__(self) -> None:
        self.main_index: Optional[faiss.IndexIVFPQ] = None
        self.cold_index: Optional[faiss.IndexFlatIP] = None
        self.cold_item_ids: Optional[np.ndarray] = None
        self.all_embeddings: Optional[np.ndarray] = None
        self.n_items: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        cold_start_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Build both the main IVF-PQ and cold-start flat indices.

        Args:
            embeddings:      (N, 128) float32 item embeddings (unit-normalised).
            cold_start_mask: (N,) bool — True for cold-start items.  If None,
                             the newest ``COLD_START_FRACTION`` of items are
                             flagged as cold-start (by index order).
        """
        n, d = embeddings.shape
        assert d == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}-dim, got {d}"
        self.n_items = n
        self.all_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # ---- Cold-start mask ----
        if cold_start_mask is None:
            cold_start_mask = np.zeros(n, dtype=bool)
            n_cold = max(1, int(n * COLD_START_FRACTION))
            cold_start_mask[-n_cold:] = True
        self.cold_item_ids = np.where(cold_start_mask)[0].astype(np.int64)

        # ---- Main IVF-PQ index ----
        effective_nlist = min(NLIST, n // 2)  # safety for tiny catalogs
        quantizer = faiss.IndexFlatIP(d)
        self.main_index = faiss.IndexIVFPQ(quantizer, d, effective_nlist, PQ_M, PQ_NBITS)
        self.main_index.metric_type = faiss.METRIC_INNER_PRODUCT

        print(f"[FAISS] Training IVF-PQ index (nlist={effective_nlist}, m={PQ_M}, nbits={PQ_NBITS}) on {n} vectors...")
        self.main_index.train(self.all_embeddings)
        self.main_index.add(self.all_embeddings)
        print(f"[FAISS] Main index: {self.main_index.ntotal} vectors indexed.")

        # ---- Cold-start flat index ----
        cold_embeddings = self.all_embeddings[cold_start_mask]
        self.cold_index = faiss.IndexFlatIP(d)
        self.cold_index.add(cold_embeddings)
        print(f"[FAISS] Cold-start index: {self.cold_index.ntotal} vectors (newest {cold_start_mask.sum()}).")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        user_embedding: np.ndarray,
        k_main: int = K_MAIN,
        k_cold: int = K_COLD,
        nprobe: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top candidates by merging main + cold-start results.

        Args:
            user_embedding: (1, 128) or (128,) float32, unit-normalised.
            k_main:         Number of candidates from the IVF-PQ index.
            k_cold:         Number of candidates from the cold-start index.
            nprobe:         IVF search granularity.

        Returns:
            (ids, scores) — merged and de-duplicated, sorted by descending
            inner-product score, length ≤ k_main + k_cold.
        """
        assert self.main_index is not None, "Index not built. Call build() first."
        q = np.ascontiguousarray(
            user_embedding.reshape(1, -1), dtype=np.float32,
        )

        # Main index search
        self.main_index.nprobe = nprobe
        main_scores, main_ids = self.main_index.search(q, k_main)
        main_scores = main_scores.ravel()
        main_ids = main_ids.ravel()

        # Cold-start search
        cold_scores, cold_local_ids = self.cold_index.search(q, min(k_cold, self.cold_index.ntotal))
        cold_scores = cold_scores.ravel()
        cold_local_ids = cold_local_ids.ravel()
        # Map local cold IDs back to catalog IDs
        cold_ids = self.cold_item_ids[cold_local_ids]

        # Merge + deduplicate
        all_ids = np.concatenate([main_ids, cold_ids])
        all_scores = np.concatenate([main_scores, cold_scores])

        # Remove invalid IDs (FAISS returns -1 for unfilled slots)
        valid = all_ids >= 0
        all_ids = all_ids[valid]
        all_scores = all_scores[valid]

        # Deduplicate — keep highest score per ID
        unique_ids, inv = np.unique(all_ids, return_inverse=True)
        best_scores = np.full(len(unique_ids), -np.inf, dtype=np.float32)
        np.maximum.at(best_scores, inv, all_scores)

        # Sort descending
        order = np.argsort(-best_scores)
        k_total = k_main + k_cold
        return unique_ids[order[:k_total]], best_scores[order[:k_total]]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the main IVF-PQ index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.main_index, str(path))
        print(f"[FAISS] Index saved to {path}")

    def load(self, path: Path) -> None:
        """Load a pre-built IVF-PQ index from disk."""
        self.main_index = faiss.read_index(str(path))
        print(f"[FAISS] Index loaded from {path} ({self.main_index.ntotal} vectors)")


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_recall_latency(
    retriever: FaissRetriever,
    query_embeddings: np.ndarray,
    nprobes: Tuple[int, ...] = BENCHMARK_NPROBES,
    k: int = K_TOTAL,
    save_path: Optional[Path] = None,
) -> Dict[int, Dict[str, float]]:
    """Measure Recall@K and latency for varying nprobe values.

    Ground truth is computed via exact inner-product search
    (``IndexFlatIP``).

    Args:
        retriever:        Built FaissRetriever.
        query_embeddings: (N_queries, 128) user embeddings.
        nprobes:          Tuple of nprobe values to test.
        k:                Number of results to retrieve.
        save_path:        If provided, save recall-vs-latency plot here.

    Returns:
        Dict[nprobe → {"recall": float, "latency_ms": float}].
    """
    n_queries = query_embeddings.shape[0]
    queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    # ---- Ground truth (exact search) ----
    exact_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    exact_index.add(retriever.all_embeddings)
    _, gt_ids = exact_index.search(queries, k)

    results: Dict[int, Dict[str, float]] = {}

    for nprobe in nprobes:
        retriever.main_index.nprobe = nprobe

        t0 = time.perf_counter()
        _, approx_ids = retriever.main_index.search(queries, k)
        elapsed = time.perf_counter() - t0
        avg_latency_ms = (elapsed / n_queries) * 1000

        # Recall: fraction of ground-truth IDs found in approximate results
        total_hits = 0
        total_relevant = 0
        for i in range(n_queries):
            gt_set = set(gt_ids[i][gt_ids[i] >= 0].tolist())
            approx_set = set(approx_ids[i][approx_ids[i] >= 0].tolist())
            total_hits += len(gt_set & approx_set)
            total_relevant += len(gt_set)

        recall = total_hits / max(total_relevant, 1)
        results[nprobe] = {"recall": recall, "latency_ms": avg_latency_ms}

        print(
            f"  nprobe={nprobe:>3d}  |  Recall@{k}: {recall:.4f}  |  "
            f"Latency: {avg_latency_ms:.3f} ms/query"
        )

    # ---- Plot ----
    if save_path is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        recalls = [results[np_]["recall"] for np_ in nprobes]
        latencies = [results[np_]["latency_ms"] for np_ in nprobes]

        ax.plot(latencies, recalls, "b-o", linewidth=2, markersize=8)
        for np_, lat, rec in zip(nprobes, latencies, recalls):
            ax.annotate(
                f"nprobe={np_}", (lat, rec),
                textcoords="offset points", xytext=(8, -5), fontsize=9,
            )
        ax.set_xlabel("Average query latency (ms)")
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(f"FAISS IVF-PQ — Recall@{k} vs Latency")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"[INFO] Benchmark plot saved to {save_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Cold-Start Experiment
# ═══════════════════════════════════════════════════════════════════════════

def cold_start_experiment(
    retriever: FaissRetriever,
    query_embeddings: np.ndarray,
    k_main: int = K_MAIN,
    k_cold: int = K_COLD,
    nprobe: int = 16,
) -> Dict[str, float]:
    """Measure cold-start discovery rate with and without the flat sidecar.

    **Discovery rate** = fraction of cold-start items that appear in *any*
    user's top-K results.

    Args:
        retriever:        Built FaissRetriever with cold-start index.
        query_embeddings: (N_queries, 128) user embeddings.
        k_main:           Candidates from main index.
        k_cold:           Candidates from cold-start index.
        nprobe:           IVF search granularity.

    Returns:
        Dict with keys "discovery_with_cold", "discovery_without_cold".
    """
    n_queries = query_embeddings.shape[0]
    cold_set = set(retriever.cold_item_ids.tolist())
    k_total = k_main + k_cold

    # ---- WITH cold-start sidecar ----
    discovered_with: set = set()
    for i in range(n_queries):
        ids, _ = retriever.query(
            query_embeddings[i], k_main=k_main, k_cold=k_cold, nprobe=nprobe,
        )
        discovered_with.update(int(x) for x in ids if int(x) in cold_set)

    rate_with = len(discovered_with) / max(len(cold_set), 1)

    # ---- WITHOUT cold-start sidecar (main index only) ----
    discovered_without: set = set()
    retriever.main_index.nprobe = nprobe
    queries_cont = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    _, main_ids = retriever.main_index.search(queries_cont, k_total)
    for i in range(n_queries):
        discovered_without.update(
            int(x) for x in main_ids[i] if int(x) >= 0 and int(x) in cold_set
        )
    rate_without = len(discovered_without) / max(len(cold_set), 1)

    print(f"\n{'='*50}")
    print("Cold-Start Discovery Experiment")
    print(f"{'='*50}")
    print(f"  Cold-start items: {len(cold_set)}")
    print(f"  Queries:          {n_queries}")
    print(f"  Discovery WITH cold-start index:    {rate_with:.4f} ({len(discovered_with)}/{len(cold_set)})")
    print(f"  Discovery WITHOUT cold-start index: {rate_without:.4f} ({len(discovered_without)}/{len(cold_set)})")
    print(f"  Lift: {rate_with - rate_without:+.4f}")

    return {
        "discovery_with_cold": rate_with,
        "discovery_without_cold": rate_without,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_item_embeddings(path: Optional[Path] = None) -> np.ndarray:
    """Load item embeddings from Parquet or generate random fallback.

    Args:
        path: Path to item_embeddings.parquet.

    Returns:
        (N, 128) float32 array of L2-normalised item embeddings.
    """
    if path is None:
        path = DATA_DIR / "item_embeddings.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        embeddings = df[emb_cols].values.astype(np.float32)
        print(f"[INFO] Loaded {embeddings.shape[0]} item embeddings from {path}")
    else:
        print("[WARN] item_embeddings.parquet not found — generating random embeddings.")
        n_items = 500
        embeddings = np.random.randn(n_items, EMBEDDING_DIM).astype(np.float32)

    # Ensure L2-normalised
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms
    return embeddings


def load_user_embeddings(path: Optional[Path] = None, n_fallback: int = 1_000) -> np.ndarray:
    """Load user embeddings from Parquet or generate random fallback.

    Args:
        path:       Path to user_embeddings.parquet.
        n_fallback: Number of random user embeddings if file not found.

    Returns:
        (N, 128) float32 array of L2-normalised user embeddings.
    """
    if path is None:
        path = DATA_DIR / "user_embeddings.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        embeddings = df[emb_cols].values.astype(np.float32)
        print(f"[INFO] Loaded {embeddings.shape[0]} user embeddings from {path}")
    else:
        print(f"[WARN] user_embeddings.parquet not found — generating {n_fallback} random user embeddings.")
        embeddings = np.random.randn(n_fallback, EMBEDDING_DIM).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Full FAISS pipeline: build, benchmark, cold-start experiment, save."""
    print("=" * 70)
    print("FAISS Index — Build, Benchmark, Cold-Start")
    print("=" * 70)

    # ---- Load embeddings ----
    item_embs = load_item_embeddings()
    user_embs = load_user_embeddings(n_fallback=BENCHMARK_N_QUERIES)

    # Cap query count for benchmarking
    if user_embs.shape[0] > BENCHMARK_N_QUERIES:
        rng = np.random.RandomState(42)
        idx = rng.choice(user_embs.shape[0], size=BENCHMARK_N_QUERIES, replace=False)
        query_embs = user_embs[idx]
    else:
        query_embs = user_embs

    n_items = item_embs.shape[0]
    print(f"[INFO] Catalog: {n_items} items, Queries: {query_embs.shape[0]} users")

    # ---- Build index ----
    retriever = FaissRetriever()
    retriever.build(item_embs)

    # ---- Benchmark recall vs latency ----
    print(f"\n{'='*50}")
    print(f"Recall@{K_TOTAL} vs Latency Benchmark")
    print(f"{'='*50}")
    benchmark_recall_latency(
        retriever, query_embs,
        nprobes=BENCHMARK_NPROBES,
        k=K_TOTAL,
        save_path=DATA_DIR / "faiss_benchmark.png",
    )

    # ---- Cold-start experiment ----
    cold_start_experiment(retriever, query_embs)

    # ---- Sample query ----
    print(f"\n{'='*50}")
    print("Sample Query")
    print(f"{'='*50}")
    sample_q = query_embs[0]
    ids, scores = retriever.query(sample_q, k_main=K_MAIN, k_cold=K_COLD, nprobe=16)
    print(f"  Top-10 item IDs: {ids[:10].tolist()}")
    print(f"  Top-10 scores:   {[f'{s:.4f}' for s in scores[:10]]}")
    print(f"  Total candidates returned: {len(ids)}")

    # ---- Save index ----
    retriever.save(DATA_DIR / "faiss_index.bin")

    print("\n✅ FAISS pipeline complete.")


if __name__ == "__main__":
    main()
