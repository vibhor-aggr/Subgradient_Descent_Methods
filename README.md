# Subgradient Descent Method Comparison

This repository contains implementation of six subgradient descent variants and compares them on two tasks:

- Fashion-MNIST classification with a multi-layer ReLU network.
- California Housing LASSO regression.

## Implemented Methods

- Projected subgradient descent.
- Proximal subgradient descent.
- Stochastic subgradient descent.
- Stochastic proximal subgradient descent.
- Stochastic projected subgradient descent.
- Stochastic subgradient descent with momentum.

The optimizers, model gradients, ReLU network, LASSO model, projection operators, and proximal operators are implemented from scratch in `subgrad_methods/`. Scikit-learn is used only for dataset fetching, splitting, and feature standardization.

## Models

- ReLU classifier: `784 -> 256 -> 128 -> 10`, softmax cross-entropy, manual backpropagation.
- LASSO regression: standardized linear regression with an unpenalized intercept and L1 penalty on weights.

Projected methods use L1-ball projection by default. The implementation also supports L2 projection with `--projection-kind l2`.

## Setup

Use a local Python 3.12 environment. The default Python in this workspace may not include the scientific stack.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Run a smoke test:

```bash
python run_experiments.py --quick --epochs-classifier 1 --epochs-regression 2
```

Run the full comparison:

```bash
python run_experiments.py --dataset-mode full
```

## Outputs

Generated files are written to `results/`:

- `metrics_fashion_mnist.csv`
- `metrics_california_lasso.csv`
- `summary.csv`
- `fashion_mnist_training_loss.png`
- `fashion_mnist_test_accuracy.png`
- `california_lasso_training_objective.png`
- `california_lasso_test_rmse.png`
- `california_lasso_test_r2.png`

`results/` is ignored by git because these artifacts are generated and can be reproduced.

## Tests

```bash
pytest
```

The tests cover L1/L2 projection, soft-thresholding, finite-difference checks for LASSO and the ReLU network, and basic stability for every optimizer variant.

## Acknowledgments

- Fashion-MNIST is provided by Zalando Research and is available under the MIT license: https://github.com/zalandoresearch/fashion-mnist
- Fashion-MNIST is fetched through OpenML where available: https://www.openml.org/d/40996
- California Housing is fetched through scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
