import numpy as np

from subgrad_methods.models import LassoRegression, ReLUMultiLayerNetwork


def finite_difference(model, X, y, key, index, epsilon=1e-6):
    original = float(model.params[key][index])
    model.params[key][index] = original + epsilon
    plus, _ = model.loss_and_grad(X, y)
    model.params[key][index] = original - epsilon
    minus, _ = model.loss_and_grad(X, y)
    model.params[key][index] = original
    return (plus - minus) / (2.0 * epsilon)


def test_lasso_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    model = LassoRegression(n_features=3, dtype=np.float64)
    model.params["w"][...] = rng.normal(size=3)
    model.params["b"][0] = 0.2
    _, grads = model.loss_and_grad(X, y)
    approx = finite_difference(model, X, y, "w", (1,))
    np.testing.assert_allclose(grads["w"][1], approx, rtol=1e-5, atol=1e-6)


def test_relu_network_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    y = np.array([0, 1, 2, 1, 0, 2])
    model = ReLUMultiLayerNetwork(
        input_dim=4,
        hidden_dims=(5, 4),
        output_dim=3,
        seed=3,
        dtype=np.float64,
    )
    model.params["b1"][...] = 0.1
    model.params["b2"][...] = 0.1
    _, grads = model.loss_and_grad(X, y)
    approx = finite_difference(model, X, y, "W2", (2, 1))
    np.testing.assert_allclose(grads["W2"][2, 1], approx, rtol=1e-4, atol=1e-5)
