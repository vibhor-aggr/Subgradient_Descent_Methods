"""Experiment orchestration for the subgradient method comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import ClassificationData, RegressionData, load_california_lasso, load_fashion_mnist
from .evaluation import accuracy, r2_score, rmse, write_csv
from .models import LassoRegression, ReLUMultiLayerNetwork
from .optimizers import METHOD_SPECS, OptimizerConfig, default_projection_radius, train_model
from .plotting import plot_metric


METHOD_ORDER = [
    "projected_subgradient",
    "proximal_subgradient",
    "stochastic_subgradient",
    "stochastic_proximal_subgradient",
    "stochastic_projected_subgradient",
    "stochastic_subgradient_momentum",
]


@dataclass(frozen=True)
class ExperimentSettings:
    data_dir: Path
    output_dir: Path
    dataset_mode: str
    seed: int
    epochs_classifier: int
    epochs_regression: int
    tune_epochs_classifier: int
    tune_epochs_regression: int
    batch_size_classifier: int
    batch_size_regression: int
    projection_kind: str
    run_classifier: bool = True
    run_regression: bool = True


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    l1_lambda: float
    projection_factor: float | None = None


@dataclass(frozen=True)
class SelectedConfig:
    candidate: Candidate
    config: OptimizerConfig
    validation_score: float


def classification_candidates(method: str, quick: bool) -> list[Candidate]:
    lrs = [0.05] if quick else [0.05, 0.02]
    l1_values = [1e-6] if quick else [1e-6, 1e-5]
    factors = [1.0] if quick else [0.8, 1.2]
    projected = METHOD_SPECS[method].projected
    candidates: list[Candidate] = []
    for lr in lrs:
        for l1_lambda in l1_values:
            if projected:
                for factor in factors:
                    candidates.append(Candidate(lr, l1_lambda, factor))
            else:
                candidates.append(Candidate(lr, l1_lambda))
    return candidates


def regression_candidates(method: str, quick: bool) -> list[Candidate]:
    lrs = [0.05] if quick else [0.1, 0.05]
    l1_values = [0.01] if quick else [0.001, 0.01]
    factors = [2.0] if quick else [2.0, 5.0]
    projected = METHOD_SPECS[method].projected
    candidates: list[Candidate] = []
    for lr in lrs:
        for l1_lambda in l1_values:
            if projected:
                for factor in factors:
                    candidates.append(Candidate(lr, l1_lambda, factor))
            else:
                candidates.append(Candidate(lr, l1_lambda))
    return candidates


def make_config(
    method: str,
    candidate: Candidate,
    model: object,
    batch_size: int,
    projection_kind: str,
) -> OptimizerConfig:
    projection_radius = None
    if METHOD_SPECS[method].projected:
        base_radius = default_projection_radius(model, projection_kind)
        projection_radius = base_radius * float(candidate.projection_factor or 1.0)
    return OptimizerConfig(
        method=method,
        learning_rate=candidate.learning_rate,
        l1_lambda=candidate.l1_lambda,
        batch_size=batch_size,
        projection_kind=projection_kind,
        projection_radius=projection_radius,
    )


def make_classifier(seed: int) -> ReLUMultiLayerNetwork:
    return ReLUMultiLayerNetwork(
        input_dim=784,
        hidden_dims=(256, 128),
        output_dim=10,
        seed=seed,
        dtype=np.float32,
    )


def make_regressor(seed: int, n_features: int) -> LassoRegression:
    return LassoRegression(n_features=n_features, seed=seed, dtype=np.float64)


def select_classifier_config(
    method: str,
    data: ClassificationData,
    settings: ExperimentSettings,
) -> SelectedConfig:
    quick = settings.dataset_mode == "quick"
    best: SelectedConfig | None = None
    for candidate in classification_candidates(method, quick):
        model = make_classifier(settings.seed)
        config = make_config(
            method, candidate, model, settings.batch_size_classifier, settings.projection_kind
        )

        def callback(current_model: ReLUMultiLayerNetwork, epoch: int):
            del epoch
            return {
                "validation_accuracy": accuracy(
                    data.y_val, current_model.predict(data.X_val)
                ),
            }

        history = train_model(
            model,
            data.X_train,
            data.y_train,
            config,
            epochs=settings.tune_epochs_classifier,
            metric_callback=callback,
            seed=settings.seed,
        )
        final = history[-1]
        score = float(final.get("validation_accuracy", -np.inf))
        if bool(final.get("stopped_early", False)):
            score = -np.inf
        if best is None or score > best.validation_score:
            best = SelectedConfig(candidate, config, score)
    if best is None:
        raise RuntimeError(f"no classifier config selected for {method}")
    return best


def select_regression_config(
    method: str,
    data: RegressionData,
    settings: ExperimentSettings,
) -> SelectedConfig:
    quick = settings.dataset_mode == "quick"
    best: SelectedConfig | None = None
    for candidate in regression_candidates(method, quick):
        model = make_regressor(settings.seed, data.X_train.shape[1])
        config = make_config(
            method, candidate, model, settings.batch_size_regression, settings.projection_kind
        )

        def callback(current_model: LassoRegression, epoch: int):
            del epoch
            pred = data.val_target_scaler.inverse(current_model.predict(data.X_val))
            return {"validation_rmse": rmse(data.y_val_original, pred)}

        history = train_model(
            model,
            data.X_train,
            data.y_train,
            config,
            epochs=settings.tune_epochs_regression,
            metric_callback=callback,
            seed=settings.seed,
        )
        final = history[-1]
        score = float(final.get("validation_rmse", np.inf))
        if bool(final.get("stopped_early", False)):
            score = np.inf
        if best is None or score < best.validation_score:
            best = SelectedConfig(candidate, config, score)
    if best is None:
        raise RuntimeError(f"no regression config selected for {method}")
    return best


def run_classification(settings: ExperimentSettings) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = load_fashion_mnist(
        data_dir=settings.data_dir,
        seed=settings.seed,
        dataset_mode=settings.dataset_mode,
    )
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for method in tqdm(METHOD_ORDER, desc="Fashion-MNIST"):
        selected = select_classifier_config(method, data, settings)
        model = make_classifier(settings.seed)
        final_config = make_config(
            method,
            selected.candidate,
            model,
            settings.batch_size_classifier,
            settings.projection_kind,
        )

        def callback(current_model: ReLUMultiLayerNetwork, epoch: int):
            del epoch
            return {
                "dataset": "fashion_mnist",
                "train_loss": current_model.objective(
                    data.X_final, data.y_final, final_config.l1_lambda
                ),
                "train_accuracy": accuracy(data.y_final, current_model.predict(data.X_final)),
                "test_accuracy": accuracy(data.y_test, current_model.predict(data.X_test)),
                "best_validation_accuracy": selected.validation_score,
                "l1_lambda": final_config.l1_lambda,
                "projection_radius": final_config.projection_radius or "",
            }

        history = train_model(
            model,
            data.X_final,
            data.y_final,
            final_config,
            epochs=settings.epochs_classifier,
            metric_callback=callback,
            seed=settings.seed,
        )
        rows.extend(history)
        final_row = history[-1]
        summary.append(
            {
                "dataset": "fashion_mnist",
                "method": method,
                "best_learning_rate": final_config.learning_rate,
                "best_l1_lambda": final_config.l1_lambda,
                "best_validation_metric": selected.validation_score,
                "projection_radius": final_config.projection_radius or "",
                "final_train_loss": final_row.get("train_loss", ""),
                "final_test_accuracy": final_row.get("test_accuracy", ""),
                "final_test_rmse": "",
                "final_test_r2": "",
                "projection_used": final_row.get("projection_used", ""),
                "stopped_early": final_row.get("stopped_early", ""),
            }
        )

    write_csv(settings.output_dir / "metrics_fashion_mnist.csv", rows)
    plot_metric(
        rows,
        "train_loss",
        "Fashion-MNIST Training Objective",
        "Cross-entropy + L1 penalty",
        settings.output_dir / "fashion_mnist_training_loss.png",
    )
    plot_metric(
        rows,
        "test_accuracy",
        "Fashion-MNIST Test Accuracy",
        "Accuracy",
        settings.output_dir / "fashion_mnist_test_accuracy.png",
    )
    return rows, summary


def run_regression(settings: ExperimentSettings) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = load_california_lasso(
        data_dir=settings.data_dir,
        seed=settings.seed,
        dataset_mode=settings.dataset_mode,
    )
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for method in tqdm(METHOD_ORDER, desc="California LASSO"):
        selected = select_regression_config(method, data, settings)
        model = make_regressor(settings.seed, data.X_final.shape[1])
        final_config = make_config(
            method,
            selected.candidate,
            model,
            settings.batch_size_regression,
            settings.projection_kind,
        )

        def callback(current_model: LassoRegression, epoch: int):
            del epoch
            pred = data.final_target_scaler.inverse(current_model.predict(data.X_test))
            return {
                "dataset": "california_lasso",
                "train_objective": current_model.objective(
                    data.X_final, data.y_final, final_config.l1_lambda
                ),
                "test_rmse": rmse(data.y_test_original, pred),
                "test_r2": r2_score(data.y_test_original, pred),
                "best_validation_rmse": selected.validation_score,
                "l1_lambda": final_config.l1_lambda,
                "projection_radius": final_config.projection_radius or "",
            }

        history = train_model(
            model,
            data.X_final,
            data.y_final,
            final_config,
            epochs=settings.epochs_regression,
            metric_callback=callback,
            seed=settings.seed,
        )
        rows.extend(history)
        final_row = history[-1]
        summary.append(
            {
                "dataset": "california_lasso",
                "method": method,
                "best_learning_rate": final_config.learning_rate,
                "best_l1_lambda": final_config.l1_lambda,
                "best_validation_metric": selected.validation_score,
                "projection_radius": final_config.projection_radius or "",
                "final_train_loss": final_row.get("train_objective", ""),
                "final_test_accuracy": "",
                "final_test_rmse": final_row.get("test_rmse", ""),
                "final_test_r2": final_row.get("test_r2", ""),
                "projection_used": final_row.get("projection_used", ""),
                "stopped_early": final_row.get("stopped_early", ""),
            }
        )

    write_csv(settings.output_dir / "metrics_california_lasso.csv", rows)
    plot_metric(
        rows,
        "train_objective",
        "California Housing LASSO Training Objective",
        "0.5 MSE + L1 penalty",
        settings.output_dir / "california_lasso_training_objective.png",
    )
    plot_metric(
        rows,
        "test_rmse",
        "California Housing LASSO Test RMSE",
        "RMSE",
        settings.output_dir / "california_lasso_test_rmse.png",
    )
    plot_metric(
        rows,
        "test_r2",
        "California Housing LASSO Test R2",
        "R2",
        settings.output_dir / "california_lasso_test_r2.png",
    )
    return rows, summary


def run_all(settings: ExperimentSettings) -> list[dict[str, object]]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    if settings.run_classifier:
        _, classifier_summary = run_classification(settings)
        summary_rows.extend(classifier_summary)
    if settings.run_regression:
        _, regression_summary = run_regression(settings)
        summary_rows.extend(regression_summary)
    write_csv(settings.output_dir / "summary.csv", summary_rows)
    return summary_rows
