"""Check whether this folder is ready for two-tower training on another system."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "matplotlib",
    "tqdm",
    "pydantic",
    "torch",
]

TRAINING_ARTIFACTS = [
    "session_features.parquet",
    "topic_vectors.parquet",
    "author_embeddings.parquet",
    "quality_scores.parquet",
    "item_fingerprints.parquet",
]


def _check_imports() -> bool:
    ok = True
    print("Dependency check")
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"  [OK] {name}: {version}")
        except Exception as exc:
            ok = False
            print(f"  [FAIL] {name}: {exc}")
    return ok


def _check_torch() -> None:
    try:
        import torch
    except Exception:
        return

    print("\nTorch runtime")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        print("  Training will run on CPU unless a CUDA-enabled torch build is installed.")


def _check_artifacts() -> bool:
    ok = True
    print("\nArtifact check")
    for name in TRAINING_ARTIFACTS:
        path = PROCESSED / name
        if path.exists():
            print(f"  [OK] {name}: {path.stat().st_size:,} bytes")
        else:
            ok = False
            print(f"  [FAIL] {name}: missing")

    session_path = PROCESSED / "session_features.parquet"
    if session_path.exists():
        sf = pd.read_parquet(session_path, columns=["final_completion_pct"])
        max_completion = float(sf["final_completion_pct"].max())
        if max_completion <= 1.0:
            print(f"  [OK] final_completion_pct scale: max={max_completion:.4f}")
        else:
            ok = False
            print(
                "  [FAIL] final_completion_pct scale: "
                f"expected 0-1, found max={max_completion:.4f}"
            )

    fp_path = PROCESSED / "item_fingerprints.parquet"
    if fp_path.exists():
        fp = pd.read_parquet(fp_path)
        feature_cols = [c for c in fp.columns if c != "item_id"]
        if len(feature_cols) == 81:
            print(f"  [OK] item fingerprint width: {len(feature_cols)}")
        else:
            ok = False
            print(f"  [FAIL] item fingerprint width: expected 81, found {len(feature_cols)}")

    return ok


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    deps_ok = _check_imports()
    _check_torch()
    artifacts_ok = _check_artifacts()
    if deps_ok and artifacts_ok:
        print("\n[OK] Folder is ready for two-tower training.")
        return 0
    print("\n[FAIL] Resolve the failed checks before training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
