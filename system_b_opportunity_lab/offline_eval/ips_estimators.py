"""Off-policy evaluation estimators for logged exploration data."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def _weights(logging_propensity, target_propensity) -> np.ndarray:
    logging = np.asarray(logging_propensity, dtype=float)
    target = np.asarray(target_propensity, dtype=float)
    return target / np.maximum(logging, EPS)


def inverse_propensity_score(rewards, logging_propensity, target_propensity) -> float:
    rewards = np.asarray(rewards, dtype=float)
    weights = _weights(logging_propensity, target_propensity)
    return float(np.mean(rewards * weights))


def clipped_ips(rewards, logging_propensity, target_propensity, clip: float = 10.0) -> float:
    rewards = np.asarray(rewards, dtype=float)
    weights = np.minimum(_weights(logging_propensity, target_propensity), clip)
    return float(np.mean(rewards * weights))


def self_normalized_ips(rewards, logging_propensity, target_propensity) -> float:
    rewards = np.asarray(rewards, dtype=float)
    weights = _weights(logging_propensity, target_propensity)
    denom = np.maximum(weights.sum(), EPS)
    return float(np.sum(rewards * weights) / denom)


def doubly_robust(rewards, logging_propensity, target_propensity, q_hat_logged, q_hat_target) -> float:
    rewards = np.asarray(rewards, dtype=float)
    weights = _weights(logging_propensity, target_propensity)
    q_logged = np.asarray(q_hat_logged, dtype=float)
    q_target = np.asarray(q_hat_target, dtype=float)
    return float(np.mean(q_target + weights * (rewards - q_logged)))


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 500, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        estimates.append(float(np.mean(sample)))
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )
