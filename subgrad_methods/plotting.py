"""Plotting utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_metric(
    rows: list[dict[str, object]],
    y_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    plt.figure(figsize=(10, 6))
    for method, method_rows in by_method.items():
        method_rows = sorted(method_rows, key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in method_rows]
        values = [float(row[y_key]) for row in method_rows]
        plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=3, label=method)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
