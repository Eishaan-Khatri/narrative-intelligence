"""
Narrative Intelligence Platform — Session Feature Extraction Pipeline
=====================================================================

Transforms raw telemetry events (from ``data/synthetic/events.parquet``)
into per-session feature vectors (``data/processed/session_features.parquet``).

Pipeline Steps
--------------
1.  **Session reconstruction** — Events are grouped by
    ``(user_id, item_id, chapter_index)`` and sorted by timestamp.  Within
    each group, sessions are split at 30-minute inactivity gaps
    (``INACTIVITY_GAP_SEC = 1800``) using the **cumsum-on-gap** trick:
    consecutive events with a time delta > 1800 s are assigned different
    session IDs.

2.  **Per-session feature computation** — For every reconstructed session:

    - ``session_duration_sec`` — last timestamp minus first timestamp.
    - ``reading_velocity_wpm`` — ``chapter_word_count / (total_dwell / 60)``,
      clamped to ``[50, 600]`` WPM.
    - ``velocity_acceleration`` — average velocity in the second half of the
      session minus average velocity in the first half (positive = speeding up).
    - ``completion_curve`` — 5-element vector of scroll depth at 20 / 40 / 60 /
      80 / 100 % of elapsed session time, using linear interpolation on
      SCROLL_TICK events.
    - ``completion_curve_shape`` — rule-based classifier:
        * *cliff*: depth at 80% ≥ 0.8 AND depth at 100% ≤ 0.3
        * *decay*: monotonically decreasing deltas, no single drop > 0.3
        * *abandon_early*: depth at 20% ≤ 0.15
        * *steady*: everything else
    - ``re_read_ratio`` — count(RE_SCROLL) / count(SCROLL_TICK).
    - ``final_completion_pct`` — last ``scroll_depth_pct`` divided by 100.
    - ``exit_reason`` — from EXIT event metadata, or ``chapter_end`` if a
      CHAPTER_COMPLETE event is present.

Inputs
------
- ``data/synthetic/events.parquet``
- ``data/synthetic/catalog.parquet`` (for ``avg_chapter_word_count``)

Outputs
-------
- ``data/processed/session_features.parquet``
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from feature_store.schema import (  # noqa: E402
    COMPLETION_CURVE_POINTS,
    INACTIVITY_GAP_SEC,
    READING_VELOCITY_MAX_WPM,
    READING_VELOCITY_MIN_WPM,
    CompletionCurveShape,
    EventType,
    ExitReason,
)

# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------


def reconstruct_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """
    Split events into sessions using 30-minute inactivity gaps.

    Strategy (cumsum-on-gap):
    1.  Sort by ``(user_id, item_id, chapter_index, timestamp)``.
    2.  Compute time deltas between consecutive events within each
        ``(user_id, item_id, chapter_index)`` group.
    3.  Where the delta exceeds ``INACTIVITY_GAP_SEC``, increment a
        session counter → assign new ``reconstructed_session_id``.

    Parameters
    ----------
    events : pd.DataFrame
        Raw event table with at least: user_id, item_id, chapter_index,
        timestamp, session_id.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional ``reconstructed_session_id``
        column.
    """
    df = events.sort_values(
        ["user_id", "item_id", "chapter_index", "timestamp"]
    ).copy()

    # Compute time deltas within each (user, item, chapter) group
    group_cols = ["user_id", "item_id", "chapter_index"]
    df["_ts_numeric"] = df["timestamp"].astype(np.int64) // 10**9  # seconds
    df["_delta"] = df.groupby(group_cols)["_ts_numeric"].diff().fillna(0)

    # Mark session boundaries where gap > INACTIVITY_GAP_SEC
    df["_new_session"] = (df["_delta"] > INACTIVITY_GAP_SEC).astype(int)

    # Cumsum within each group gives session index
    df["_session_idx"] = df.groupby(group_cols)["_new_session"].cumsum()

    # Composite session id: user_item_chapter_idx
    df["reconstructed_session_id"] = (
        df["user_id"] + "_" +
        df["item_id"] + "_" +
        df["chapter_index"].astype(str) + "_" +
        df["_session_idx"].astype(str)
    )

    # Cleanup temp columns
    df.drop(columns=["_ts_numeric", "_delta", "_new_session", "_session_idx"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Completion curve computation
# ---------------------------------------------------------------------------


def _compute_completion_curve(
    scroll_ticks: pd.DataFrame,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
) -> list[float]:
    """
    Compute a 5-element completion curve via linear interpolation.

    The curve captures scroll depth at 20%, 40%, 60%, 80%, and 100% of
    elapsed session time.  Uses ``np.interp`` on the SCROLL_TICK events.

    Parameters
    ----------
    scroll_ticks : pd.DataFrame
        Filtered to SCROLL_TICK events only, sorted by timestamp.
        Must have ``timestamp`` and ``scroll_depth_pct`` columns.
    session_start : pd.Timestamp
        Earliest event timestamp in the session.
    session_end : pd.Timestamp
        Latest event timestamp in the session.

    Returns
    -------
    list[float]
        5 depth values in [0, 1] (i.e. scroll_depth_pct / 100).
    """
    if scroll_ticks.empty or session_start == session_end:
        return [0.0] * COMPLETION_CURVE_POINTS

    # Elapsed fractions for each scroll tick
    total_dur = (session_end - session_start).total_seconds()
    if total_dur < 1e-6:
        return [0.0] * COMPLETION_CURVE_POINTS

    tick_elapsed = (scroll_ticks["timestamp"] - session_start).dt.total_seconds().values
    tick_fractions = tick_elapsed / total_dur  # in [0, 1]
    tick_depths = scroll_ticks["scroll_depth_pct"].values / 100.0  # normalise to [0, 1]

    # Query points: 20%, 40%, 60%, 80%, 100%
    query_points = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    curve = np.interp(query_points, tick_fractions, tick_depths).tolist()

    return [round(v, 4) for v in curve]


# ---------------------------------------------------------------------------
# Completion curve shape classification
# ---------------------------------------------------------------------------


def _classify_curve_shape(curve: list[float]) -> str:
    """
    Rule-based classifier for completion curve shape.

    Rules (applied in order):
    1. **abandon_early** — depth at point 0 (20%) ≤ 0.15
    2. **cliff**         — depth at point 3 (80%) ≥ 0.8  AND  depth at point 4 (100%) ≤ 0.3
    3. **decay**         — all deltas are non-positive (monotonically decreasing)
                           AND no single drop > 0.3
    4. **steady**        — everything else

    Parameters
    ----------
    curve : list[float]
        5-element completion curve (values in [0, 1]).

    Returns
    -------
    str
        One of: ``abandon_early``, ``cliff``, ``decay``, ``steady``.
    """
    if len(curve) < COMPLETION_CURVE_POINTS:
        return CompletionCurveShape.STEADY.value

    # Rule 1: abandon early
    if curve[0] <= 0.15:
        return CompletionCurveShape.ABANDON_EARLY.value

    # Rule 2: cliff
    if curve[3] >= 0.8 and curve[4] <= 0.3:
        return CompletionCurveShape.CLIFF.value

    # Rule 3: decay — monotonically decreasing deltas, no large drop
    deltas = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    if all(d <= 0 for d in deltas) and all(abs(d) <= 0.3 for d in deltas):
        return CompletionCurveShape.DECAY.value

    # Rule 4: steady
    return CompletionCurveShape.STEADY.value


# ---------------------------------------------------------------------------
# Per-session feature extraction
# ---------------------------------------------------------------------------


def _extract_exit_reason(session_events: pd.DataFrame) -> str:
    """
    Determine exit reason from session events.

    Priority:
    1. CHAPTER_COMPLETE event present → ``chapter_end``
    2. EXIT event with ``exit_reason`` in metadata → that reason
    3. Fallback → ``mid_chapter``

    Parameters
    ----------
    session_events : pd.DataFrame
        All events in a single session.

    Returns
    -------
    str
        Exit reason string.
    """
    if (session_events["event_type"] == EventType.CHAPTER_COMPLETE.value).any():
        return ExitReason.CHAPTER_END.value

    exit_rows = session_events[session_events["event_type"] == EventType.EXIT.value]
    if not exit_rows.empty:
        meta = exit_rows.iloc[-1]["metadata"]
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError):
                meta = {}
        if isinstance(meta, dict) and "exit_reason" in meta:
            return meta["exit_reason"]

    return ExitReason.MID_CHAPTER.value


def compute_session_features(
    events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-session features from reconstructed events.

    Parameters
    ----------
    events : pd.DataFrame
        Events with ``reconstructed_session_id`` from
        ``reconstruct_sessions()``.
    catalog : pd.DataFrame
        Book metadata with ``item_id`` and ``avg_chapter_word_count``.

    Returns
    -------
    pd.DataFrame
        One row per session conforming to ``SessionFeatures`` schema.
    """
    # Build word-count lookup
    word_counts = catalog.set_index("item_id")["avg_chapter_word_count"].to_dict()

    grouped = events.groupby("reconstructed_session_id")
    records: list[dict[str, Any]] = []

    for session_id, grp in tqdm(grouped, desc="Extracting features", unit="session"):
        grp = grp.sort_values("timestamp")
        if len(grp) < 2:
            continue  # Need at least 2 events for meaningful features

        user_id = grp["user_id"].iloc[0]
        item_id = grp["item_id"].iloc[0]
        chapter_index = int(grp["chapter_index"].iloc[0])
        device_type = grp["device_type"].mode().iloc[0] if not grp["device_type"].mode().empty else "mobile"

        ts_start = grp["timestamp"].iloc[0]
        ts_end = grp["timestamp"].iloc[-1]

        # ---- session_duration_sec ----
        session_duration = (ts_end - ts_start).total_seconds()

        # ---- reading_velocity_wpm ----
        chapter_words = word_counts.get(item_id, 3500)
        total_dwell = grp["dwell_time_sec"].sum()
        if total_dwell > 0:
            velocity_wpm = chapter_words / (total_dwell / 60.0)
        else:
            velocity_wpm = 0.0
        velocity_wpm = float(np.clip(velocity_wpm, READING_VELOCITY_MIN_WPM, READING_VELOCITY_MAX_WPM))

        # ---- velocity_acceleration ----
        scroll_ticks = grp[grp["event_type"] == EventType.SCROLL_TICK.value].copy()
        velocity_accel = 0.0
        if len(scroll_ticks) >= 4:
            midpoint = len(scroll_ticks) // 2
            first_half = scroll_ticks.iloc[:midpoint]
            second_half = scroll_ticks.iloc[midpoint:]

            def _half_velocity(half_df: pd.DataFrame) -> float:
                half_dwell = half_df["dwell_time_sec"].sum()
                if half_dwell > 0:
                    # Approximate words read from scroll delta
                    depth_range = half_df["scroll_depth_pct"].iloc[-1] - half_df["scroll_depth_pct"].iloc[0]
                    words_in_half = abs(depth_range) / 100.0 * chapter_words
                    return words_in_half / (half_dwell / 60.0)
                return 0.0

            v1 = _half_velocity(first_half)
            v2 = _half_velocity(second_half)
            velocity_accel = v2 - v1

        # ---- completion_curve ----
        curve = _compute_completion_curve(scroll_ticks, ts_start, ts_end)

        # ---- completion_curve_shape ----
        curve_shape = _classify_curve_shape(curve)

        # ---- re_read_ratio ----
        n_rescroll = (grp["event_type"] == EventType.RE_SCROLL.value).sum()
        n_scroll = (grp["event_type"] == EventType.SCROLL_TICK.value).sum()
        re_read_ratio = n_rescroll / max(n_scroll, 1)

        # ---- final_completion_pct ----
        depth_cols = grp[grp["scroll_depth_pct"].notna()]["scroll_depth_pct"]
        final_completion_pct = (
            float(np.clip(depth_cols.iloc[-1] / 100.0, 0.0, 1.0))
            if len(depth_cols) > 0
            else 0.0
        )

        # ---- exit_reason ----
        exit_reason = _extract_exit_reason(grp)

        records.append({
            "session_id": session_id,
            "user_id": user_id,
            "item_id": item_id,
            "chapter_index": chapter_index,
            "session_duration_sec": round(session_duration, 2),
            "reading_velocity_wpm": round(velocity_wpm, 2),
            "velocity_acceleration": round(velocity_accel, 2),
            "completion_curve": curve,
            "completion_curve_shape": curve_shape,
            "re_read_ratio": round(re_read_ratio, 4),
            "final_completion_pct": round(final_completion_pct, 2),
            "exit_reason": exit_reason,
            "device_type": device_type,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    events_path: Path | None = None,
    catalog_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full extraction pipeline: load events, reconstruct sessions, compute
    features, persist.

    Parameters
    ----------
    events_path : Path or None
        Path to events Parquet file.
    catalog_path : Path or None
        Path to catalog Parquet file.
    output_path : Path or None
        Output path for session features Parquet file.

    Returns
    -------
    pd.DataFrame
        Session features.
    """
    if events_path is None:
        events_path = _PROJECT_ROOT / "data" / "synthetic" / "events.parquet"
    if catalog_path is None:
        catalog_path = _PROJECT_ROOT / "data" / "synthetic" / "catalog.parquet"
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "processed" / "session_features.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("> Loading events ...")
    events = pd.read_parquet(events_path)
    print(f"  {len(events):,} events loaded")

    print("> Loading catalog ...")
    catalog = pd.read_parquet(catalog_path)
    print(f"  {len(catalog):,} items loaded")

    # Ensure timestamp is datetime
    if events["timestamp"].dtype == "object":
        events["timestamp"] = pd.to_datetime(events["timestamp"])

    print("> Reconstructing sessions ...")
    events = reconstruct_sessions(events)
    n_sessions = events["reconstructed_session_id"].nunique()
    print(f"  {n_sessions:,} sessions reconstructed")

    print("> Computing session features ...")
    session_features = compute_session_features(events, catalog)
    print(f"  {len(session_features):,} session feature rows computed")

    # Convert completion_curve list to string for Parquet storage
    session_features_save = session_features.copy()
    session_features_save["completion_curve"] = session_features_save["completion_curve"].apply(str)
    session_features_save.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[OK] Saved {output_path}")

    return session_features


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    import textwrap

    # Reconfigure stdout for Windows cp1252 compatibility
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    print(textwrap.dedent("""\
    +------------------------------------------------------------------+
    |  Narrative Intelligence Platform -- Session Feature Extraction    |
    +------------------------------------------------------------------+
    """))

    # Check if events exist; if not, run the simulator first
    events_path = _PROJECT_ROOT / "data" / "synthetic" / "events.parquet"
    if not events_path.exists():
        print("[WARN] No events.parquet found -- running Markov simulator first ...\n")
        from feature_store.simulator.markov_event_simulator import generate_synthetic_dataset
        generate_synthetic_dataset(
            num_users=1000, num_items=500,
            avg_chapters=15, sessions_per_user=30, seed=42,
        )
        print()

    session_features = run_pipeline()

    # ---- Report ----
    print("\n" + "=" * 60)
    print("SESSION FEATURE STATISTICS")
    print("=" * 60)

    # Completion curve shape distribution
    print("\nCompletion curve shape distribution:")
    shape_counts = session_features["completion_curve_shape"].value_counts()
    for shape, cnt in shape_counts.items():
        pct = cnt / len(session_features) * 100
        print(f"  {shape:<20s} {cnt:>8,}  ({pct:5.1f}%)")

    # Reading velocity stats
    print("\nReading velocity (WPM):")
    vel = session_features["reading_velocity_wpm"]
    print(f"  Mean:    {vel.mean():>8.1f}")
    print(f"  Median:  {vel.median():>8.1f}")
    print(f"  Std:     {vel.std():>8.1f}")
    print(f"  Min:     {vel.min():>8.1f}")
    print(f"  Max:     {vel.max():>8.1f}")

    # Final completion percentage histogram (10% buckets)
    print("\nFinal completion % distribution (10% buckets):")
    bins = [i / 10.0 for i in range(0, 11)]
    labels = [f"{int(lo * 100):>3d}-{int(hi * 100):>3d}%" for lo, hi in zip(bins[:-1], bins[1:])]
    completion_hist = pd.cut(
        session_features["final_completion_pct"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    ).value_counts().sort_index()
    for bucket, cnt in completion_hist.items():
        bar = "#" * max(1, int(cnt / max(completion_hist.values) * 40))
        print(f"  {bucket}  {cnt:>7,}  {bar}")

    # Exit reason distribution
    print("\nExit reason distribution:")
    exit_counts = session_features["exit_reason"].value_counts()
    for reason, cnt in exit_counts.items():
        pct = cnt / len(session_features) * 100
        print(f"  {reason:<20s} {cnt:>8,}  ({pct:5.1f}%)")

    # Session duration stats
    print("\nSession duration (sec):")
    dur = session_features["session_duration_sec"]
    print(f"  Mean:    {dur.mean():>10.1f}")
    print(f"  Median:  {dur.median():>10.1f}")
    print(f"  Std:     {dur.std():>10.1f}")

    print("\n[DONE] Session feature extraction complete.")
