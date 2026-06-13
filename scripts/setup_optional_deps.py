"""Environment helper for System A handoff machines.

The script checks core and optional dependencies and can install the optional
requirements when explicitly requested. It does not force scikit-survival
because that package is platform-sensitive, especially on Windows.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
OPTIONAL_REQUIREMENTS = PROJECT_ROOT / "requirements-optional.txt"

CORE_IMPORTS = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "torch",
    "lightgbm",
    "streamlit",
    "plotly",
]

OPTIONAL_IMPORTS = [
    "sksurv",
]


def check_import(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "installed")
        print(f"  [OK] {module_name}: {version}")
        return True
    except Exception as exc:
        print(f"  [MISSING] {module_name}: {exc}")
        return False


def run_pip(requirements_path: Path) -> int:
    if not requirements_path.exists():
        print(f"[FAIL] Missing requirements file: {requirements_path}")
        return 1
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    print("[RUN] " + " ".join(cmd))
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def print_torch_status() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"\nTorch runtime unavailable: {exc}")
        return
    print("\nTorch runtime")
    print(f"  version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/install System A optional dependencies.")
    parser.add_argument("--install-main", action="store_true", help="Install requirements.txt with pip.")
    parser.add_argument("--install-optional", action="store_true", help="Try installing requirements-optional.txt with pip.")
    args = parser.parse_args()

    if args.install_main:
        code = run_pip(REQUIREMENTS)
        if code != 0:
            return code

    if args.install_optional:
        code = run_pip(OPTIONAL_REQUIREMENTS)
        if code != 0:
            print("[WARN] Optional dependency install failed. CoxPH fallback can still run.")

    print(f"Project root: {PROJECT_ROOT}")
    print("\nCore dependency check")
    core_ok = all(check_import(name) for name in CORE_IMPORTS)
    print("\nOptional dependency check")
    optional_ok = all(check_import(name) for name in OPTIONAL_IMPORTS)
    print_torch_status()

    if not optional_ok:
        print("\nOptional install command:")
        print(f"  {sys.executable} -m pip install -r {OPTIONAL_REQUIREMENTS}")
        print("If scikit-survival fails on Windows, continue without it; CoxPH fallback remains usable.")

    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
