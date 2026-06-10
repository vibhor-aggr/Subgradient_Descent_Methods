#!/usr/bin/env python
"""Run the fresh subgradient method comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from subgrad_methods.experiments import ExperimentSettings, run_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use small dataset slices and short defaults.")
    parser.add_argument(
        "--dataset-mode",
        choices=["full", "quick"],
        default=None,
        help="Dataset size mode. --quick forces quick mode.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-classifier", type=int, default=None)
    parser.add_argument("--epochs-regression", type=int, default=None)
    parser.add_argument("--tune-epochs-classifier", type=int, default=None)
    parser.add_argument("--tune-epochs-regression", type=int, default=None)
    parser.add_argument("--batch-size-classifier", type=int, default=256)
    parser.add_argument("--batch-size-regression", type=int, default=128)
    parser.add_argument("--projection-kind", choices=["l1", "l2"], default="l1")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_mode = "quick" if args.quick else (args.dataset_mode or "full")
    quick = dataset_mode == "quick"
    settings = ExperimentSettings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_mode=dataset_mode,
        seed=args.seed,
        epochs_classifier=args.epochs_classifier or (2 if quick else 20),
        epochs_regression=args.epochs_regression or (5 if quick else 100),
        tune_epochs_classifier=args.tune_epochs_classifier or (1 if quick else 3),
        tune_epochs_regression=args.tune_epochs_regression or (2 if quick else 10),
        batch_size_classifier=args.batch_size_classifier,
        batch_size_regression=args.batch_size_regression,
        projection_kind=args.projection_kind,
        run_classifier=not args.skip_classifier,
        run_regression=not args.skip_regression,
    )
    summary = run_all(settings)
    print(f"Wrote results to {settings.output_dir.resolve()}")
    for row in summary:
        dataset = row["dataset"]
        method = row["method"]
        if dataset == "fashion_mnist":
            print(f"{dataset:18s} {method:38s} test_accuracy={float(row['final_test_accuracy']):.4f}")
        else:
            print(
                f"{dataset:18s} {method:38s} "
                f"rmse={float(row['final_test_rmse']):.4f} r2={float(row['final_test_r2']):.4f}"
            )


if __name__ == "__main__":
    main()
