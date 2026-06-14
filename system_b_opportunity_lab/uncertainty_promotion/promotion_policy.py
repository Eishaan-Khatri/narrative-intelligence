"""Uncertainty-aware promotion scores for exploration candidates."""

from __future__ import annotations


def uncertainty_aware_score(
    shrunk_quality: float,
    breakout_score: float,
    uplift_score: float,
    uncertainty: float,
    relevance: float,
    min_relevance: float = 0.25,
    uncertainty_weight: float = 0.20,
    uplift_weight: float = 0.25,
    breakout_weight: float = 0.25,
    quality_weight: float = 0.30,
) -> float:
    """Score candidate exploration items with a relevance floor guardrail."""
    if relevance < min_relevance:
        return float("-inf")
    return float(
        quality_weight * shrunk_quality
        + breakout_weight * breakout_score
        + uplift_weight * uplift_score
        + uncertainty_weight * uncertainty
    )
