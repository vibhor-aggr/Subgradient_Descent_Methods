"""Models trained by the subgradient optimizers.

The model code is intentionally small and explicit. No automatic
differentiation library is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import regularized_l1_norm


ArrayDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    scale: float

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.scale + self.mean


class LassoRegression:
    """Linear regression model for the LASSO objective."""

    regularized_keys = ("w",)

    def __init__(self, n_features: int, seed: int = 42, dtype: np.dtype = np.float64):
        del seed
        self.params: ArrayDict = {
            "w": np.zeros(n_features, dtype=dtype),
            "b": np.zeros(1, dtype=dtype),
        }

    def copy(self) -> "LassoRegression":
        clone = LassoRegression(self.params["w"].size, dtype=self.params["w"].dtype)
        clone.set_params(self.params)
        return clone

    def set_params(self, params: ArrayDict) -> None:
        for key, value in params.items():
            self.params[key][...] = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.params["w"] + self.params["b"][0]

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, ArrayDict]:
        residual = self.predict(X) - y
        n_samples = X.shape[0]
        loss = 0.5 * float(np.mean(residual**2))
        grads = {
            "w": (X.T @ residual) / n_samples,
            "b": np.array([float(np.mean(residual))], dtype=self.params["b"].dtype),
        }
        return loss, grads

    def objective(self, X: np.ndarray, y: np.ndarray, l1_lambda: float) -> float:
        loss, _ = self.loss_and_grad(X, y)
        return loss + l1_lambda * regularized_l1_norm(self.params, self.regularized_keys)


class ReLUMultiLayerNetwork:
    """Fully connected ReLU classifier with manual softmax backprop."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        output_dim: int = 10,
        seed: int = 42,
        dtype: np.dtype = np.float64,
    ):
        rng = np.random.default_rng(seed)
        layer_dims = (input_dim, *hidden_dims, output_dim)
        self.params: ArrayDict = {}
        self.regularized_keys: tuple[str, ...] = tuple(
            f"W{i}" for i in range(1, len(layer_dims))
        )
        for i, (fan_in, fan_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:]), 1):
            scale = np.sqrt(2.0 / fan_in) if i < len(layer_dims) - 1 else np.sqrt(1.0 / fan_in)
            self.params[f"W{i}"] = (rng.normal(0.0, scale, size=(fan_in, fan_out))).astype(dtype)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=dtype)
        self.n_layers = len(layer_dims) - 1

    def copy(self) -> "ReLUMultiLayerNetwork":
        input_dim = self.params["W1"].shape[0]
        hidden_dims = tuple(self.params[f"W{i}"].shape[1] for i in range(1, self.n_layers))
        output_dim = self.params[f"W{self.n_layers}"].shape[1]
        clone = ReLUMultiLayerNetwork(input_dim, hidden_dims, output_dim, dtype=self.params["W1"].dtype)
        clone.set_params(self.params)
        return clone

    def set_params(self, params: ArrayDict) -> None:
        for key, value in params.items():
            self.params[key][...] = value

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations: list[np.ndarray] = []
        current = X
        for layer in range(1, self.n_layers):
            z = current @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            pre_activations.append(z)
            current = np.maximum(z, 0.0)
            activations.append(current)
        scores = current @ self.params[f"W{self.n_layers}"] + self.params[f"b{self.n_layers}"]
        return scores, activations, pre_activations

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        scores, _, _ = self._forward(X)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_logits(X), axis=1)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, ArrayDict]:
        scores, activations, pre_activations = self._forward(X)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        n_samples = X.shape[0]
        loss = -float(np.mean(np.log(probs[np.arange(n_samples), y] + 1e-12)))

        dscores = probs
        dscores[np.arange(n_samples), y] -= 1.0
        dscores /= n_samples

        grads: ArrayDict = {}
        layer = self.n_layers
        grads[f"W{layer}"] = activations[-1].T @ dscores
        grads[f"b{layer}"] = np.sum(dscores, axis=0)

        upstream = dscores @ self.params[f"W{layer}"].T
        for layer in range(self.n_layers - 1, 0, -1):
            upstream = upstream * (pre_activations[layer - 1] > 0.0)
            grads[f"W{layer}"] = activations[layer - 1].T @ upstream
            grads[f"b{layer}"] = np.sum(upstream, axis=0)
            if layer > 1:
                upstream = upstream @ self.params[f"W{layer}"].T

        return loss, grads

    def objective(self, X: np.ndarray, y: np.ndarray, l1_lambda: float) -> float:
        loss, _ = self.loss_and_grad(X, y)
        return loss + l1_lambda * regularized_l1_norm(self.params, self.regularized_keys)
