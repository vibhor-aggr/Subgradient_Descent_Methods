"""Projection and proximal operators used by the optimizers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the elementwise proximal operator for threshold * ||x||_1."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def project_l2_ball(values: np.ndarray, radius: float) -> np.ndarray:
    """Project a vector onto the closed L2 ball with the given radius."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    norm = float(np.linalg.norm(values.ravel(), ord=2))
    if norm <= radius or norm == 0.0:
        return values.copy()
    return values * (radius / norm)


def project_l1_ball(values: np.ndarray, radius: float) -> np.ndarray:
    """Project a vector onto the closed L1 ball using the Duchi et al. method."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    original_shape = values.shape
    flat = values.ravel()
    abs_flat = np.abs(flat)
    if float(abs_flat.sum()) <= radius:
        return values.copy()
    if radius == 0.0:
        return np.zeros_like(values)

    sorted_abs = np.sort(abs_flat)[::-1]
    cssv = np.cumsum(sorted_abs)
    indices = np.arange(1, sorted_abs.size + 1)
    valid = sorted_abs * indices > (cssv - radius)
    if not np.any(valid):
        return np.zeros_like(values)
    rho = int(np.nonzero(valid)[0][-1])
    theta = (cssv[rho] - radius) / float(rho + 1)
    projected = np.sign(flat) * np.maximum(abs_flat - theta, 0.0)
    return projected.reshape(original_shape)


def regularized_l1_norm(params: dict[str, np.ndarray], keys: Iterable[str]) -> float:
    return float(sum(np.abs(params[key]).sum() for key in keys))


def regularized_l2_norm(params: dict[str, np.ndarray], keys: Iterable[str]) -> float:
    total = sum(float(np.sum(params[key] ** 2)) for key in keys)
    return float(np.sqrt(total))


def flatten_params(params: dict[str, np.ndarray], keys: Iterable[str]) -> np.ndarray:
    arrays = [params[key].ravel() for key in keys]
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)


def assign_flat_params(
    params: dict[str, np.ndarray], keys: Iterable[str], flat_values: np.ndarray
) -> None:
    offset = 0
    for key in keys:
        size = params[key].size
        params[key][...] = flat_values[offset : offset + size].reshape(params[key].shape)
        offset += size
    if offset != flat_values.size:
        raise ValueError("flat_values size does not match selected parameters")


def apply_soft_threshold_to_params(
    params: dict[str, np.ndarray], keys: Iterable[str], threshold: float
) -> None:
    for key in keys:
        params[key][...] = soft_threshold(params[key], threshold)


def project_params(
    params: dict[str, np.ndarray],
    keys: Iterable[str],
    radius: float,
    kind: str = "l1",
) -> str:
    """Project selected parameters in place.

    Returns the projection kind actually used. L1 projection is attempted first
    when requested; if it yields non-finite values, L2 projection is used.
    """
    keys = list(keys)
    flat = flatten_params(params, keys)
    if kind == "l1":
        projected = project_l1_ball(flat, radius)
        if np.all(np.isfinite(projected)):
            assign_flat_params(params, keys, projected)
            return "l1"
        projected = project_l2_ball(flat, radius)
        assign_flat_params(params, keys, projected)
        return "l2_fallback"
    if kind == "l2":
        projected = project_l2_ball(flat, radius)
        assign_flat_params(params, keys, projected)
        return "l2"
    raise ValueError(f"unknown projection kind: {kind}")
