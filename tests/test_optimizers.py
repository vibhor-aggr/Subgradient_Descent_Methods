import numpy as np

from subgrad_methods.experiments import METHOD_ORDER
from subgrad_methods.models import LassoRegression
from subgrad_methods.optimizers import OptimizerConfig, train_model


def test_all_optimizers_run_without_nonfinite_values():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 4))
    true_w = np.array([1.0, -2.0, 0.0, 0.5])
    y = X @ true_w + 0.1 * rng.normal(size=40)

    for method in METHOD_ORDER:
        model = LassoRegression(n_features=4, dtype=np.float64)
        config = OptimizerConfig(
            method=method,
            learning_rate=0.05,
            l1_lambda=0.01,
            batch_size=8,
            projection_radius=5.0,
        )

        def callback(current_model, epoch):
            del epoch
            return {"objective": current_model.objective(X, y, config.l1_lambda)}

        history = train_model(model, X, y, config, epochs=3, metric_callback=callback, seed=7)
        assert len(history) == 3
        assert all(np.isfinite(row["objective"]) for row in history)
        assert not history[-1]["stopped_early"]
