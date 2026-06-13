"""One-command handoff runner for the GPU/long-runtime system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], required: bool = True) -> int:
    print("\n[RUN] " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if required and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run System A handoff tasks on the target machine.")
    parser.add_argument("--install-main", action="store_true", help="Install requirements.txt first.")
    parser.add_argument("--install-optional", action="store_true", help="Try requirements-optional.txt first.")
    parser.add_argument("--download-gutenberg", action="store_true")
    parser.add_argument("--gutenberg-limit", type=int, default=10)
    parser.add_argument("--amazon-input", type=Path)
    parser.add_argument("--build-external-catalog", action="store_true")
    parser.add_argument("--activate-external-catalog", action="store_true")
    parser.add_argument("--large-synthetic", action="store_true")
    parser.add_argument("--num-users", type=int, default=3000)
    parser.add_argument("--num-items", type=int, default=5000)
    parser.add_argument("--sessions-per-user", type=int, default=20)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()

    setup_cmd = [sys.executable, "scripts/setup_optional_deps.py"]
    if args.install_main:
        setup_cmd.append("--install-main")
    if args.install_optional:
        setup_cmd.append("--install-optional")
    run_command(setup_cmd, required=not args.install_optional)

    if args.download_gutenberg:
        run_command(
            [
                sys.executable,
                "scripts/download_gutenberg_sample.py",
                "--limit",
                str(args.gutenberg_limit),
            ],
            required=False,
        )

    if args.amazon_input is not None:
        run_command(
            [
                sys.executable,
                "scripts/prepare_amazon_sample.py",
                "--input",
                str(args.amazon_input),
            ],
            required=False,
        )

    if args.build_external_catalog or args.activate_external_catalog:
        external_cmd = [sys.executable, "scripts/build_external_catalog.py"]
        if args.activate_external_catalog:
            external_cmd.append("--activate")
        run_command(external_cmd, required=False)

    if args.large_synthetic:
        run_command(
            [
                sys.executable,
                "scripts/run_large_synthetic_rebuild.py",
                "--num-users",
                str(args.num_users),
                "--num-items",
                str(args.num_items),
                "--sessions-per-user",
                str(args.sessions_per_user),
            ]
        )

    run_command([sys.executable, "scripts/check_training_ready.py"])

    if args.run_training:
        training_cmd = [
            sys.executable,
            "scripts/run_gpu_training_suite.py",
            "--batch-size",
            str(args.batch_size),
        ]
        if args.skip_downstream:
            training_cmd.append("--skip-downstream")
        run_command(training_cmd)
    else:
        print("[INFO] Skipping training because --run-training was not provided.")

    run_command([sys.executable, "scripts/final_artifact_report.py"], required=False)
    print("[OK] Handoff runner finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
