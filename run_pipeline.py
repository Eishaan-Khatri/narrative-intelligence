"""
Narrative Intelligence Platform — Pipeline Orchestrator
========================================================
Runs the full System A pipeline end-to-end:

  1. Generate synthetic data (Markov event simulator)
  2. Build session features (Layer 0)
  3. Build temporal features (Layer 0)
  4. Build content features — NMF topics, author embeddings, quality scores (Layer 1)
  5. Build item fingerprints (Layer 1)
  6. Train Two-Tower model (Layer 2)
  7. Build FAISS index (Layer 2)
  8. Train survival model (Layer 3)
  9. Train LambdaMART re-ranker (Layer 3)
  10. Run evaluation — completion-weighted NDCG, ablation, oracle (Layer 4)

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --step simulate   # single step
    python run_pipeline.py --from retrieval  # start from a specific step
"""

from __future__ import annotations

import argparse
import io
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def configure_console_encoding() -> None:
    """Use UTF-8 console streams on Windows so module status prints do not fail."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        encoding = (getattr(stream, "encoding", None) or "").lower()
        if encoding not in ("utf-8", "utf8") and hasattr(stream, "buffer"):
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                ),
            )


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

STEPS = [
    {
        "name": "simulate",
        "description": "Generate synthetic event stream via Markov simulator",
        "module": "feature_store.simulator.markov_event_simulator",
        "function": "generate_synthetic_dataset",
    },
    {
        "name": "session_features",
        "description": "Build session features from event log",
        "module": "feature_store.build_session_features",
        "function": "run_pipeline",
    },
    {
        "name": "temporal_features",
        "description": "Build user temporal engagement profiles",
        "module": "feature_store.build_temporal_features",
        "function": "main",
    },
    {
        "name": "nmf_topics",
        "description": "Train NMF topic model on catalog text",
        "module": "system_a_discovery_engine.layer1_content.nmf_topics",
        "function": "run_pipeline",
    },
    {
        "name": "author_embeddings",
        "description": "Compute time-decayed author embeddings",
        "module": "system_a_discovery_engine.layer1_content.author_embeddings",
        "function": "run_pipeline",
    },
    {
        "name": "quality_scores",
        "description": "Compute 12-signal PCA quality score + item fingerprints",
        "module": "system_a_discovery_engine.layer1_content.quality_score_pca",
        "function": "run_pipeline",
    },
    {
        "name": "retrieval",
        "description": "Train Two-Tower model (Phase 1 + Phase 2 hard negatives)",
        "module": "system_a_discovery_engine.layer2_retrieval.train_loop",
        "function": "train",
    },
    {
        "name": "faiss_index",
        "description": "Build FAISS IVF-PQ index + cold-start flat index",
        "module": "system_a_discovery_engine.layer2_retrieval.faiss_index",
        "function": "main",
    },
    {
        "name": "survival",
        "description": "Fit Cox PH / RSF survival model for dropout hazard",
        "module": "system_a_discovery_engine.layer3_ranking.survival_model",
        "function": "run_survival_pipeline",
    },
    {
        "name": "ranker",
        "description": "Train LambdaMART re-ranker",
        "module": "system_a_discovery_engine.layer3_ranking.lambdamart_ranker",
        "function": "run_ranking_pipeline",
    },
    {
        "name": "evaluation",
        "description": "Run completion-weighted NDCG evaluation",
        "module": "system_a_discovery_engine.layer4_evaluation.completion_ndcg",
        "function": "main",
    },
    {
        "name": "ablation",
        "description": "Run 5-model ablation study",
        "module": "system_a_discovery_engine.layer4_evaluation.ablation_runner",
        "function": "main",
    },
    {
        "name": "oracle",
        "description": "Run retrieval oracle ceiling analysis",
        "module": "system_a_discovery_engine.layer4_evaluation.retrieval_oracle",
        "function": "run_oracle_experiment",
    },
]


def run_step(step: dict) -> bool:
    """Run a single pipeline step. Returns True on success."""
    configure_console_encoding()
    print(f"\n{'=' * 70}")
    print(f"  STEP: {step['name']}")
    print(f"  {step['description']}")
    print(f"{'=' * 70}")

    start = time.time()
    try:
        mod = importlib.import_module(step["module"])

        function_name = step.get("function")
        if function_name:
            if not hasattr(mod, function_name):
                raise AttributeError(
                    f"{step['module']} has no configured function {function_name!r}"
                )
            getattr(mod, function_name)()
        elif hasattr(mod, "main"):
            mod.main()
        else:
            raise AttributeError(f"{step['module']} has no callable pipeline entry point")

        elapsed = time.time() - start
        print(f"\n  [OK] {step['name']} completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] {step['name']} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Narrative Intelligence Platform — System A Pipeline Orchestrator",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Run a single step by name (e.g., 'simulate', 'retrieval')",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        type=str,
        default=None,
        help="Start from a specific step (e.g., 'retrieval' to skip data generation)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available pipeline steps",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable pipeline steps:")
        print("-" * 50)
        for i, step in enumerate(STEPS, 1):
            print(f"  {i:2d}. {step['name']:25s} -- {step['description']}")
        return

    # Ensure data directories exist
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "synthetic").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    if args.step:
        matching = [s for s in STEPS if s["name"] == args.step]
        if not matching:
            print(f"[FAIL] Unknown step: {args.step}")
            print("Run with --list to see available steps.")
            sys.exit(1)
        steps_to_run = matching
    elif args.from_step:
        names = [s["name"] for s in STEPS]
        if args.from_step not in names:
            print(f"[FAIL] Unknown step: {args.from_step}")
            sys.exit(1)
        idx = names.index(args.from_step)
        steps_to_run = STEPS[idx:]
    else:
        steps_to_run = STEPS

    # Run
    print("\n" + "=" * 70)
    print("  NARRATIVE INTELLIGENCE PLATFORM -- SYSTEM A PIPELINE")
    print(f"  Running {len(steps_to_run)} step(s)")
    print("=" * 70)

    total_start = time.time()
    results = []

    for step in steps_to_run:
        success = run_step(step)
        results.append((step["name"], success))
        if not success:
            print(f"\n[WARN] Pipeline stopped at step '{step['name']}'. Fix the error and re-run with:")
            print(f"   python run_pipeline.py --from {step['name']}")
            break

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    failed = [name for name, success in results if not success]
    if failed:
        print(f"\n  [FAIL] {len(failed)} step(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  [OK] All {len(results)} steps completed successfully!")
        print("\n  Next: Launch the dashboard with:")
        print("    streamlit run dashboards/system_a_demo/app.py")


if __name__ == "__main__":
    main()
