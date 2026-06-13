"""
Narrative Intelligence Platform — System A, Layer 3
====================================================
Survival Model for Dropout Hazard
----------------------------------

**Algorithm Overview**:
This module models reader dropout (abandonment) as a survival-analysis
problem.  Time is measured in *chapter indices*; the event of interest is
abandonment (the user stopped reading before the final chapter).  Readers
who are still actively reading at the last observed chapter are treated as
right-censored observations.

Two models are fitted and compared:
1. **Cox Proportional-Hazards (CoxPH)** via ``lifelines.CoxPHFitter`` —
   a semi-parametric model that is easy to interpret via hazard ratios.
2. **Random Survival Forest (RSF)** via ``sksurv.ensemble`` — a
   non-parametric ensemble that can capture non-linear covariate effects.

**Decision rule**: if the RSF C-index exceeds the CoxPH C-index by more
than 0.02, adopt RSF; otherwise keep CoxPH for interpretability.

**Covariates** (per user × story × chapter observation):
  1. inter_chapter_gap_trend  — slope of time-gaps between chapters
  2. completion_curve_shape   — one-hot encoded (4 categories)
  3. author_hiatus_days       — days since the author last published
  4. user_genre_hazard_baseline — user's historical genre-level dropout rate
  5. completion_proximity     — chapter_index / total_chapters
  6. velocity_acceleration    — acceleration of reading velocity

**Outputs**:
  - ``data/processed/dropout_hazard.parquet``
  - ``data/processed/survival_model.pkl``
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from feature_store.schema import CompletionCurveShape  # noqa: E402

# Suppress convergence warnings during demo
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPLETION_CURVE_CATEGORIES: list[str] = [e.value for e in CompletionCurveShape]
COVARIATE_COLUMNS: list[str] = [
    "inter_chapter_gap_trend",
    "author_hiatus_days",
    "user_genre_hazard_baseline",
    "completion_proximity",
    "velocity_acceleration",
] + [f"ccs_{cat}" for cat in COMPLETION_CURVE_CATEGORIES]

OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"
HAZARD_PARQUET = OUTPUT_DIR / "dropout_hazard.parquet"
MODEL_PKL = OUTPUT_DIR / "survival_model.pkl"

C_INDEX_THRESHOLD = 0.02  # minimum RSF advantage to justify complexity


# ===================================================================
# Data preparation helpers
# ===================================================================

def one_hot_completion_curve(
    series: pd.Series,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode the completion_curve_shape column.

    Parameters
    ----------
    series : pd.Series
        Column containing string labels from ``CompletionCurveShape``.
    categories : list[str], optional
        Explicit category order.  Defaults to ``COMPLETION_CURVE_CATEGORIES``.

    Returns
    -------
    pd.DataFrame
        One-hot encoded DataFrame with columns ``ccs_<category>``.
    """
    cats = categories or COMPLETION_CURVE_CATEGORIES
    dummies = pd.get_dummies(series, prefix="ccs").reindex(
        columns=[f"ccs_{c}" for c in cats], fill_value=0
    )
    return dummies.astype(np.float64)


def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and augment a raw survival DataFrame.

    Expects columns: user_id, item_id, chapter_index, total_chapters,
    inter_chapter_gap_trend, completion_curve_shape, author_hiatus_days,
    user_genre_hazard_baseline, velocity_acceleration, event_observed.

    Adds ``completion_proximity`` and one-hot encodes the curve shape.
    """
    df = df.copy()
    df["completion_proximity"] = df["chapter_index"] / df["total_chapters"].clip(lower=1)
    ohe = one_hot_completion_curve(df["completion_curve_shape"])
    df = pd.concat([df, ohe], axis=1)
    return df


# ===================================================================
# Model fitting
# ===================================================================

def fit_cox_ph(
    df: pd.DataFrame,
    duration_col: str = "chapter_index",
    event_col: str = "event_observed",
    covariate_cols: list[str] | None = None,
    penalizer: float = 0.01,
) -> Any:
    """Fit a Cox Proportional-Hazards model.

    Parameters
    ----------
    df : pd.DataFrame
        Survival DataFrame with covariates, duration, and event columns.
    duration_col : str
        Column representing time-to-event (chapter index).
    event_col : str
        Binary column (1 = abandoned, 0 = censored).
    covariate_cols : list[str]
        Feature columns to include.  Defaults to ``COVARIATE_COLUMNS``.
    penalizer : float
        L2-penalizer for regularisation (helps with multicollinearity
        in one-hot encoded features).

    Returns
    -------
    lifelines.CoxPHFitter
        Fitted CoxPH model.
    """
    from lifelines import CoxPHFitter  # lazy import

    cols = covariate_cols or COVARIATE_COLUMNS
    fit_df = df[[duration_col, event_col] + cols].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(fit_df, duration_col=duration_col, event_col=event_col)
    return cph


def fit_rsf(
    df: pd.DataFrame,
    duration_col: str = "chapter_index",
    event_col: str = "event_observed",
    covariate_cols: list[str] | None = None,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> Any:
    """Fit a Random Survival Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Survival DataFrame.
    duration_col, event_col : str
        Time and event indicator columns.
    covariate_cols : list[str]
        Feature columns.  Defaults to ``COVARIATE_COLUMNS``.
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth (kept shallow to avoid overfitting small data).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sksurv.ensemble.RandomSurvivalForest
        Fitted RSF model.
    """
    from sksurv.ensemble import RandomSurvivalForest  # lazy import

    cols = covariate_cols or COVARIATE_COLUMNS
    X = df[cols].values.astype(np.float64)
    # scikit-survival expects a structured array for y
    y = np.array(
        [(bool(e), t) for e, t in zip(df[event_col], df[duration_col])],
        dtype=[("event", bool), ("time", np.float64)],
    )
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rsf.fit(X, y)
    return rsf


# ===================================================================
# Evaluation
# ===================================================================

def concordance_index_cox(cph, df: pd.DataFrame) -> float:
    """Return the concordance index for a fitted CoxPH model."""
    return cph.concordance_index_


def concordance_index_rsf(
    rsf,
    df: pd.DataFrame,
    duration_col: str = "chapter_index",
    event_col: str = "event_observed",
    covariate_cols: list[str] | None = None,
) -> float:
    """Return the concordance index for a fitted RSF on given data."""
    from sksurv.metrics import concordance_index_censored  # lazy import

    cols = covariate_cols or COVARIATE_COLUMNS
    X = df[cols].values.astype(np.float64)
    y_event = df[event_col].astype(bool).values
    y_time = df[duration_col].astype(np.float64).values
    risk = rsf.predict(X)
    c_index = concordance_index_censored(y_event, y_time, risk)[0]
    return c_index


def select_model(
    cph,
    rsf,
    c_cox: float,
    c_rsf: float,
    threshold: float = C_INDEX_THRESHOLD,
) -> Tuple[Any, str]:
    """Select between CoxPH and RSF based on C-index delta.

    Returns
    -------
    (model, model_name) : Tuple[Any, str]
    """
    if c_rsf - c_cox > threshold:
        return rsf, "RandomSurvivalForest"
    return cph, "CoxPH"


# ===================================================================
# Hazard scoring
# ===================================================================

def predict_hazard_scores(
    model,
    model_name: str,
    df: pd.DataFrame,
    covariate_cols: list[str] | None = None,
) -> np.ndarray:
    """Predict hazard scores for every row using the selected model.

    For CoxPH the partial hazard is used (higher ⇒ more at risk).
    For RSF the risk score (predicted cumulative hazard) is used.

    Parameters
    ----------
    model : CoxPHFitter or RandomSurvivalForest
    model_name : str
    df : pd.DataFrame
    covariate_cols : list[str]

    Returns
    -------
    np.ndarray  — 1-D array of hazard scores, shape ``(len(df),)``.
    """
    cols = covariate_cols or COVARIATE_COLUMNS
    if model_name == "CoxPH":
        scores = model.predict_partial_hazard(df[cols]).values.ravel()
    else:
        X = df[cols].values.astype(np.float64)
        scores = model.predict(X)
    # Normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-9:
        scores = (scores - s_min) / (s_max - s_min)
    return scores


# ===================================================================
# I/O helpers
# ===================================================================

def save_hazard_parquet(
    df: pd.DataFrame,
    hazard_scores: np.ndarray,
    path: Path | None = None,
) -> Path:
    """Save ``(user_id, item_id, chapter_index, hazard_score)`` to Parquet."""
    out = (path or HAZARD_PARQUET)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = df[["user_id", "item_id", "chapter_index"]].copy()
    result["hazard_score"] = hazard_scores
    result.to_parquet(out, index=False)
    return out


def save_model(model, model_name: str, path: Path | None = None) -> Path:
    """Pickle the selected model alongside its name tag."""
    out = path or MODEL_PKL
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"model": model, "model_name": model_name}, fh)
    return out


def load_model(path: Path | None = None) -> Tuple[Any, str]:
    """Load a previously saved model.  Returns ``(model, model_name)``."""
    with open(path or MODEL_PKL, "rb") as fh:
        blob = pickle.load(fh)
    return blob["model"], blob["model_name"]


# ===================================================================
# Synthetic data generation
# ===================================================================

def generate_synthetic_survival_data(
    n_users: int = 200,
    n_items: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a realistic synthetic survival dataset for demo / testing.

    Each user reads a subset of items.  Abandonment probability rises
    with higher gap-trend, lower completion-proximity, and for the
    ``cliff`` / ``abandon_early`` curve shapes.

    Parameters
    ----------
    n_users : int
    n_items : int
    seed : int

    Returns
    -------
    pd.DataFrame  — Ready for ``prepare_survival_data``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    total_chapters_pool = rng.integers(5, 30, size=n_items)
    curve_shapes = list(COMPLETION_CURVE_CATEGORIES)

    for uid in tqdm(range(n_users), desc="Generating synthetic survival data"):
        # Each user reads 3-8 items
        n_read = rng.integers(3, 9)
        item_ids = rng.choice(n_items, size=n_read, replace=False)
        user_hazard_base = rng.uniform(0.05, 0.40)

        for iid in item_ids:
            total_ch = int(total_chapters_pool[iid])
            shape = rng.choice(curve_shapes)
            hiatus = float(rng.exponential(30))
            gap_trend = float(rng.normal(0.0, 1.0))
            vel_acc = float(rng.normal(0.0, 0.5))

            # Hazard increases with gap_trend, cliff/abandon_early shapes
            shape_risk = 0.3 if shape in ("cliff", "abandon_early") else -0.1
            linear_risk = 0.2 * gap_trend + shape_risk + 0.01 * hiatus + 0.5 * user_hazard_base

            # Determine chapter at which user drops out (or completes)
            for ch in range(1, total_ch + 1):
                prox = ch / total_ch
                hazard_prob = 1 / (1 + np.exp(-(linear_risk - 1.5 * prox + vel_acc * 0.3)))
                if rng.random() < hazard_prob * 0.15:
                    # Abandoned at this chapter
                    rows.append({
                        "user_id": f"u_{uid:04d}",
                        "item_id": f"i_{iid:04d}",
                        "chapter_index": ch,
                        "total_chapters": total_ch,
                        "inter_chapter_gap_trend": gap_trend,
                        "completion_curve_shape": shape,
                        "author_hiatus_days": hiatus,
                        "user_genre_hazard_baseline": user_hazard_base,
                        "velocity_acceleration": vel_acc,
                        "event_observed": 1,
                    })
                    break
            else:
                # Completed — right-censored at final chapter
                rows.append({
                    "user_id": f"u_{uid:04d}",
                    "item_id": f"i_{iid:04d}",
                    "chapter_index": total_ch,
                    "total_chapters": total_ch,
                    "inter_chapter_gap_trend": gap_trend,
                    "completion_curve_shape": shape,
                    "author_hiatus_days": hiatus,
                    "user_genre_hazard_baseline": user_hazard_base,
                    "velocity_acceleration": vel_acc,
                    "event_observed": 0,
                })

    return pd.DataFrame(rows)


# ===================================================================
# Main pipeline
# ===================================================================

def run_survival_pipeline(
    df_raw: pd.DataFrame | None = None,
    verbose: bool = True,
) -> Tuple[Any, str, pd.DataFrame]:
    """End-to-end survival modelling pipeline.

    1. Prepare data  2. Fit CoxPH  3. Fit RSF  4. Compare C-indices
    5. Select best model  6. Score all rows  7. Save outputs.

    Parameters
    ----------
    df_raw : pd.DataFrame, optional
        Raw survival data.  If *None*, synthetic data is generated.
    verbose : bool
        Whether to print diagnostic output.

    Returns
    -------
    (selected_model, model_name, hazard_df) : Tuple
    """
    # --- 0. Data ---
    if df_raw is None:
        df_raw = generate_synthetic_survival_data()
    df = prepare_survival_data(df_raw)
    if verbose:
        n_events = df["event_observed"].sum()
        print(f"[SurvivalModel] Observations: {len(df)} | "
              f"Events (abandoned): {n_events} | "
              f"Censored: {len(df) - n_events}")

    # --- 1. Fit CoxPH ---
    cph = fit_cox_ph(df)
    c_cox = concordance_index_cox(cph, df)
    if verbose:
        print(f"\n{'=' * 60}")
        print("CoxPH Hazard Ratios")
        print(f"{'=' * 60}")
        cph.print_summary(columns=["coef", "exp(coef)", "p"])
        print(f"\nCoxPH C-index: {c_cox:.4f}")

    # --- 2. Fit RSF when scikit-survival is available ---
    rsf = None
    c_rsf = float("-inf")
    try:
        rsf = fit_rsf(df)
        c_rsf = concordance_index_rsf(rsf, df)
        if verbose:
            print(f"RSF   C-index: {c_rsf:.4f}")
            print(f"Delta (RSF - Cox): {c_rsf - c_cox:+.4f}")
    except ImportError as exc:
        if verbose:
            print(f"[WARN] RSF skipped: {exc}")

    # --- 3. Select model ---
    model, model_name = select_model(cph, rsf, c_cox, c_rsf) if rsf is not None else (cph, "CoxPH")
    if verbose:
        print(f"\n→ Selected model: {model_name} "
              f"(threshold={C_INDEX_THRESHOLD})")

    # --- 4. Score ---
    hazard_scores = predict_hazard_scores(model, model_name, df)
    if verbose:
        print(f"\nHazard score stats — "
              f"min={hazard_scores.min():.4f}, "
              f"mean={hazard_scores.mean():.4f}, "
              f"max={hazard_scores.max():.4f}")

    # --- 5. Save ---
    haz_path = save_hazard_parquet(df, hazard_scores)
    mdl_path = save_model(model, model_name)
    if verbose:
        print(f"\n✓ Hazard parquet → {haz_path}")
        print(f"✓ Model pickle   → {mdl_path}")

    hazard_df = df[["user_id", "item_id", "chapter_index"]].copy()
    hazard_df["hazard_score"] = hazard_scores
    return model, model_name, hazard_df


# ===================================================================
# Standalone demo
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Narrative Intelligence Platform — Survival Dropout Model")
    print("=" * 70, "\n")

    # Try loading session_features if available
    session_path = _PROJECT_ROOT / "data" / "processed" / "session_features.parquet"
    if session_path.exists():
        print(f"[INFO] Found session features at {session_path} — "
              "deriving survival data …")
        sf = pd.read_parquet(session_path)
        # Derive survival columns from session_features
        # (simplified: treat each session as one observation)
        sf = sf.rename(columns={
            "final_completion_pct": "_fcp",
        })
        sf["total_chapters"] = sf.groupby("item_id")["chapter_index"].transform("max").clip(lower=1)
        sf["event_observed"] = (sf["_fcp"] < 0.9).astype(int)
        for col in ["inter_chapter_gap_trend", "author_hiatus_days",
                     "user_genre_hazard_baseline"]:
            if col not in sf.columns:
                sf[col] = np.random.default_rng(0).normal(size=len(sf))
        if "completion_curve_shape" not in sf.columns:
            sf["completion_curve_shape"] = np.random.default_rng(0).choice(
                COMPLETION_CURVE_CATEGORIES, size=len(sf)
            )
        if "velocity_acceleration" not in sf.columns:
            sf["velocity_acceleration"] = 0.0
        df_raw = sf
    else:
        print("[INFO] No session_features.parquet found — "
              "generating synthetic survival data …\n")
        df_raw = None

    model, name, hazard_df = run_survival_pipeline(df_raw, verbose=True)

    print(f"\n{'=' * 70}")
    print("  Sample hazard predictions (first 10)")
    print("=" * 70)
    print(hazard_df.head(10).to_string(index=False))

    print("\n✅ Survival model pipeline complete.")
