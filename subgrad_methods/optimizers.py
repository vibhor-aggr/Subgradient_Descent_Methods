"""Subgradient optimizer variants implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .operators import (
    apply_soft_threshold_to_params,
    flatten_params,
    project_params,
    regularized_l1_norm,
    regularized_l2_norm,
)


MetricCallback = Callable[[object, int], dict[str, float | int | str | bool]]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    stochastic: bool
    proximal: bool
    projected: bool
    momentum: bool


METHOD_SPECS: dict[str, MethodSpec] = {
    "projected_subgradient": MethodSpec(
        "projected_subgradient", stochastic=False, proximal=False, projected=True, momentum=False
    ),
    "proximal_subgradient": MethodSpec(
        "proximal_subgradient", stochastic=False, proximal=True, projected=False, momentum=False
    ),
    "stochastic_subgradient": MethodSpec(
        "stochastic_subgradient", stochastic=True, proximal=False, projected=False, momentum=False
    ),
    "stochastic_proximal_subgradient": MethodSpec(
        "stochastic_proximal_subgradient", stochastic=True, proximal=True, projected=False, momentum=False
    ),
    "stochastic_projected_subgradient": MethodSpec(
        "stochastic_projected_subgradient", stochastic=True, proximal=False, projected=True, momentum=False
    ),
    "stochastic_subgradient_momentum": MethodSpec(
        "stochastic_subgradient_momentum", stochastic=True, proximal=False, projected=False, momentum=True
    ),
}


@dataclass(frozen=True)
class OptimizerConfig:
    method: str
    learning_rate: float
    l1_lambda: float
    batch_size: int = 256
    projection_kind: str = "l1"
    projection_radius: float | None = None
    momentum: float = 0.9


def add_l1_subgradient(model: object, grads: dict[str, np.ndarray], l1_lambda: float) -> None:
    if l1_lambda <= 0.0:
        return
    for key in model.regularized_keys:
        grads[key] = grads[key] + l1_lambda * np.sign(model.params[key])


def params_are_finite(model: object) -> bool:
    return all(np.all(np.isfinite(value)) for value in model.params.values())


def gradient_step(
    model: object,
    grads: dict[str, np.ndarray],
    learning_rate: float,
    velocity: dict[str, np.ndarray] | None,
    momentum: float,
) -> dict[str, np.ndarray] | None:
    if velocity is None:
        for key, grad in grads.items():
            model.params[key][...] -= learning_rate * grad
        return None

    for key, grad in grads.items():
        velocity[key][...] = momentum * velocity[key] - learning_rate * grad
        model.params[key][...] += velocity[key]
    return velocity


def default_projection_radius(model: object, projection_kind: str) -> float:
    keys = list(model.regularized_keys)
    flat_size = flatten_params(model.params, keys).size
    if projection_kind == "l1":
        radius = regularized_l1_norm(model.params, keys)
    elif projection_kind == "l2":
        radius = regularized_l2_norm(model.params, keys)
    else:
        raise ValueError(f"unknown projection kind: {projection_kind}")
    return float(radius if radius > 0.0 else max(1.0, np.sqrt(flat_size)))


def train_model(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: OptimizerConfig,
    epochs: int,
    metric_callback: MetricCallback | None = None,
    seed: int = 42,
) -> list[dict[str, float | int | str | bool]]:
    """Train a model and return one metrics row per epoch."""
    if config.method not in METHOD_SPECS:
        raise ValueError(f"unknown optimizer method: {config.method}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.l1_lambda < 0.0:
        raise ValueError("l1_lambda must be non-negative")

    spec = METHOD_SPECS[config.method]
    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]
    batch_size = min(max(1, config.batch_size), n_samples)
    velocity = None
    if spec.momentum:
        velocity = {key: np.zeros_like(value) for key, value in model.params.items()}

    projection_radius = config.projection_radius
    if spec.projected and projection_radius is None:
        projection_radius = default_projection_radius(model, config.projection_kind)

    history: list[dict[str, float | int | str | bool]] = []
    projection_used = config.projection_kind if spec.projected else "none"
    stopped_early = False

    for epoch in range(1, epochs + 1):
        epoch_lr = config.learning_rate / np.sqrt(epoch)
        if spec.stochastic:
            n_batches = int(np.ceil(n_samples / batch_size))
            batches = (
                indices
                for indices in np.array_split(rng.permutation(n_samples), n_batches)
                if indices.size > 0
            )
        else:
            batches = (np.arange(n_samples),)

        for batch_indices in batches:
            loss, grads = model.loss_and_grad(X_train[batch_indices], y_train[batch_indices])
            if not np.isfinite(loss):
                stopped_early = True
                break

            if not spec.proximal:
                add_l1_subgradient(model, grads, config.l1_lambda)

            velocity = gradient_step(
                model,
                grads,
                epoch_lr,
                velocity,
                config.momentum if spec.momentum else 0.0,
            )

            if spec.proximal and config.l1_lambda > 0.0:
                apply_soft_threshold_to_params(
                    model.params, model.regularized_keys, epoch_lr * config.l1_lambda
                )

            if spec.projected:
                assert projection_radius is not None
                projection_used = project_params(
                    model.params,
                    model.regularized_keys,
                    radius=projection_radius,
                    kind=config.projection_kind,
                )

            if not params_are_finite(model):
                stopped_early = True
                break

        row: dict[str, float | int | str | bool] = {
            "epoch": epoch,
            "learning_rate": float(epoch_lr),
            "method": config.method,
            "projection_used": projection_used,
            "stopped_early": stopped_early,
        }
        if metric_callback is not None:
            row.update(metric_callback(model, epoch))
        history.append(row)
        if stopped_early:
            break

    return history
