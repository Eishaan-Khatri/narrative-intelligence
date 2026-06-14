"""Beta-Binomial shrinkage for low-exposure item quality estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def beta_binomial_shrinkage(
    frame: pd.DataFrame,
    success_col: str,
    trial_col: str,
    group_col: str | None = None,
    prior_strength: float = 50.0,
) -> pd.DataFrame:
    """Return posterior means and uncertainty for binomial item outcomes.

    The empirical prior is computed globally or within ``group_col``. This makes
    a 2/2 item regress heavily toward its genre/topic prior while a 200/400 item
    stays close to its observed rate.
    """
    if success_col not in frame.columns or trial_col not in frame.columns:
        raise KeyError(f"Missing required columns: {success_col}, {trial_col}")

    out = frame.copy()
    successes = out[success_col].fillna(0).astype(float).clip(lower=0)
    trials = out[trial_col].fillna(0).astype(float).clip(lower=0)
    successes = np.minimum(successes, trials)

    if group_col and group_col in out.columns:
        prior_rates = (
            out.assign(_successes=successes, _trials=trials)
            .groupby(group_col)
            .apply(lambda g: (g["_successes"].sum() + 1.0) / (g["_trials"].sum() + 2.0), include_groups=False)
        )
        prior_rate = out[group_col].map(prior_rates).fillna((successes.sum() + 1.0) / (trials.sum() + 2.0)).astype(float)
    else:
        prior_rate = pd.Series((successes.sum() + 1.0) / (trials.sum() + 2.0), index=out.index)

    alpha = prior_rate * prior_strength
    beta = (1.0 - prior_rate) * prior_strength
    posterior_alpha = alpha + successes
    posterior_beta = beta + (trials - successes)
    total = posterior_alpha + posterior_beta

    out["observed_rate"] = np.divide(successes, trials.replace(0, np.nan)).fillna(prior_rate)
    out["prior_rate"] = prior_rate
    out["posterior_alpha"] = posterior_alpha
    out["posterior_beta"] = posterior_beta
    out["shrunk_mean"] = posterior_alpha / total
    out["posterior_uncertainty"] = np.sqrt((posterior_alpha * posterior_beta) / ((total**2) * (total + 1.0)))
    out["effective_sample_size"] = trials
    return out
