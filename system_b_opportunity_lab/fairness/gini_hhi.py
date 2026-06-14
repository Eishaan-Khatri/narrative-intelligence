"""Creator exposure concentration metrics."""

from __future__ import annotations

import numpy as np


def gini_coefficient(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - arr.min()
    total = arr.sum()
    if total <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * total)) - ((n + 1.0) / n))


def hhi(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    total = arr.sum()
    if arr.size == 0 or total <= 0:
        return 0.0
    shares = arr / total
    return float(np.sum(shares**2))


def long_tail_viability(values: np.ndarray | list[float], threshold: float = 100.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    cutoff = np.quantile(arr, 0.5)
    tail = arr[arr <= cutoff]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail >= threshold))
