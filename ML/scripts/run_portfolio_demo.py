#!/usr/bin/env python
"""Run a small, self-contained end-to-end portfolio demonstration.

This command validates the software pipeline without downloading or committing
the multi-gigabyte Simulink datasets. Its deliberately separable synthetic
data must never be reported as a research result.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
DEFAULT_OUTPUT = ML_ROOT / "results" / "portfolio_demo"


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data-to-metrics smoke pipeline without large datasets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated artifact directory (default: ML/results/portfolio_demo).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "synthetic_feature_dataset.csv"
    metrics_dir = output_dir / "benchmark"
    model_dir = output_dir / "models"

    _run(
        [
            sys.executable,
            str(ML_ROOT / "tests" / "make_synthetic_dataset.py"),
            str(data_path),
            "--groups",
            "15",
            "--windows-per-class",
            "4",
            "--seed",
            "240727",
        ]
    )
    _run(
        [
            sys.executable,
            str(ML_ROOT / "scripts" / "run_benchmark.py"),
            "--data",
            str(data_path),
            "--config",
            str(ML_ROOT / "configs" / "smoke.yaml"),
            "--output",
            str(metrics_dir),
            "--model-dir",
            str(model_dir),
            "--models",
            "random_forest",
            "--group-cv",
        ]
    )

    comparison_path = metrics_dir / "model_comparison.csv"
    group_cv_path = metrics_dir / "group_cv" / "summary.csv"
    if not comparison_path.is_file() or not group_cv_path.is_file():
        raise RuntimeError("Expected benchmark artifacts were not generated.")

    comparison = pd.read_csv(comparison_path)
    required_splits = {"validation", "test"}
    if set(comparison["Split"]) != required_splits:
        raise RuntimeError(
            f"Unexpected split set: {sorted(set(comparison['Split']))}"
        )
    if comparison["MacroF1"].isna().any():
        raise RuntimeError("MacroF1 contains missing values.")

    print("\nPortfolio demo completed successfully.")
    for row in comparison.itertuples(index=False):
        print(
            f"  {row.Split}: Macro-F1={row.MacroF1:.4f}, "
            f"healthy FAR={row.HealthyFalseAlarmRate:.4f}"
        )
    print(f"  Metrics: {comparison_path}")
    print("  NOTE: synthetic smoke metrics are pipeline checks, not research claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
