"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import TargetScaler


@dataclass(frozen=True)
class ClassificationData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_final: np.ndarray
    y_final: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class RegressionData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_val_original: np.ndarray
    val_target_scaler: TargetScaler
    X_final: np.ndarray
    y_final: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_original: np.ndarray
    final_target_scaler: TargetScaler


def _fetch_openml_compat(name: str, version: int, data_home: str):
    try:
        return fetch_openml(
            name=name,
            version=version,
            as_frame=False,
            data_home=data_home,
            parser="auto",
        )
    except TypeError:
        return fetch_openml(name=name, version=version, as_frame=False, data_home=data_home)


def stratified_subset(
    X: np.ndarray, y: np.ndarray, n_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples is None or n_samples >= X.shape[0]:
        return X, y
    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        random_state=seed,
        stratify=y,
    )
    return X_subset, y_subset


def load_fashion_mnist(
    data_dir: Path,
    seed: int = 42,
    dataset_mode: str = "full",
    validation_fraction: float = 0.1,
) -> ClassificationData:
    dataset = _fetch_openml_compat("Fashion-MNIST", version=1, data_home=str(data_dir))
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(np.int64)

    X_train_full, y_train_full = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    if dataset_mode == "quick":
        X_train_full, y_train_full = stratified_subset(X_train_full, y_train_full, 2000, seed)
        X_test, y_test = stratified_subset(X_test, y_test, 500, seed)
    elif dataset_mode != "full":
        raise ValueError("dataset_mode must be 'full' or 'quick'")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_full,
    )
    return ClassificationData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_final=X_train_full,
        y_final=y_train_full,
        X_test=X_test,
        y_test=y_test,
    )


def _standardized_target(y_train: np.ndarray, *others: np.ndarray):
    mean = float(np.mean(y_train))
    scale = float(np.std(y_train))
    if scale == 0.0:
        scale = 1.0
    scaler = TargetScaler(mean=mean, scale=scale)
    transformed = [(arr - mean) / scale for arr in (y_train, *others)]
    return scaler, transformed


def load_california_lasso(
    data_dir: Path,
    seed: int = 42,
    dataset_mode: str = "full",
) -> RegressionData:
    dataset = fetch_california_housing(data_home=str(data_dir))
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64)

    if dataset_mode == "quick":
        rng = np.random.default_rng(seed)
        indices = rng.choice(X.shape[0], size=3000, replace=False)
        X = X[indices]
        y = y[indices]
    elif dataset_mode != "full":
        raise ValueError("dataset_mode must be 'full' or 'quick'")

    X_final_raw, X_test_raw, y_final_raw, y_test_original = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train_raw, X_val_raw, y_train_raw, y_val_original = train_test_split(
        X_final_raw, y_final_raw, test_size=0.2, random_state=seed
    )

    train_scaler = StandardScaler().fit(X_train_raw)
    X_train = train_scaler.transform(X_train_raw)
    X_val = train_scaler.transform(X_val_raw)
    val_target_scaler, (y_train, y_val) = _standardized_target(y_train_raw, y_val_original)

    final_scaler = StandardScaler().fit(X_final_raw)
    X_final = final_scaler.transform(X_final_raw)
    X_test = final_scaler.transform(X_test_raw)
    final_target_scaler, (y_final, y_test) = _standardized_target(
        y_final_raw, y_test_original
    )

    return RegressionData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        y_val_original=y_val_original,
        val_target_scaler=val_target_scaler,
        X_final=X_final,
        y_final=y_final,
        X_test=X_test,
        y_test=y_test,
        y_test_original=y_test_original,
        final_target_scaler=final_target_scaler,
    )
