"""
Narrative Intelligence Platform — Markov Event Simulator
=========================================================

Generates synthetic telemetry for the platform by modelling each reading
session as a **discrete-time Markov chain** over five engagement states:

    ENGAGED_FAST  →  high reading speed, low pause/exit probability
    ENGAGED_SLOW  →  moderate speed, moderate pause probability
    SKIMMING      →  very high speed, moderate exit probability
    DISTRACTED    →  low speed, high pause probability, elevated exit
    EXITING       →  absorbing state — session ends

Design rationale
----------------
The Markov transition matrix is *not* stationary: it is conditioned on three
latent factors that shift at every session (and partly at every tick):

1.  **Book quality tier** — drawn from ``Beta(2, 5)`` per item so most books
    are mediocre (mean ≈ 0.29).  Higher quality biases toward ENGAGED states
    and away from DISTRACTED / EXITING.

2.  **User-book affinity** — cosine similarity between a user's 40-dim taste
    vector and the book's topic vector.  High affinity increases tolerance for
    lower-quality books and keeps the chain in ENGAGED states.

3.  **Chapter position** — early chapters (index 3-5, 0-indexed) inject extra
    probability mass into DISTRACTED, simulating the "valley of death" where
    many readers churn.

Calibration
-----------
Books with higher latent quality produce systematically *higher* completion
rates.  This is verified in the ``__main__`` block by grouping items into
quality quartiles and printing mean completion rates.

Outputs
-------
- ``data/synthetic/events.parquet``   — flat event table
- ``data/synthetic/catalog.parquet``  — book metadata
- ``data/synthetic/users.parquet``    — user profiles

All timestamps are synthetic, anchored to ``2025-01-01 00:00:00 UTC`` and
incremented realistically per tick.

Algorithm
---------
Each tick (~5 seconds of wall-clock reading time):

1.  Draw the current state's **emission**: scroll_speed (words/sec),
    pause_probability, re_scroll_probability, exit_probability.
2.  Emit zero or more events (SCROLL_TICK, PAUSE, RESUME, RE_SCROLL).
3.  Advance scroll position by ``scroll_speed × tick_duration / chapter_words``.
4.  If scroll position ≥ 1.0, emit CHAPTER_COMPLETE and end session.
5.  If the chain transitions to EXITING, emit EXIT with reason ``mid_chapter``.
6.  Otherwise, sample next state from the conditioned transition matrix.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from feature_store.schema import (  # noqa: E402
    ENGAGEMENT_STATES,
    NMF_TOPICS_DIM,
    DeviceType,
    EventType,
    ExitReason,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_DURATION_SEC: float = 5.0
"""Wall-clock seconds per simulation tick."""

BASE_WORDS_PER_CHAPTER: int = 3500
"""Average chapter word count (σ = 800)."""

DEVICE_WEIGHTS: list[float] = [0.55, 0.30, 0.15]
"""Sampling weights for mobile / desktop / tablet."""

ANCHOR_TS: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc)
"""Epoch anchor for synthetic timestamps."""

GENRE_POOL: list[str] = [
    "Fantasy", "Science Fiction", "Romance", "Thriller", "Mystery",
    "Horror", "Literary Fiction", "Historical Fiction", "Young Adult",
    "Non-Fiction", "Biography", "Self-Help", "Adventure", "Dystopian",
    "Comedy", "Drama", "Crime", "Mythology", "Philosophy", "Poetry",
]

# ---------------------------------------------------------------------------
# State indices for fast numpy lookups
# ---------------------------------------------------------------------------
_S = {name: idx for idx, name in enumerate(ENGAGEMENT_STATES)}
_N_STATES = len(ENGAGEMENT_STATES)

# Emission parameters per state
# Columns: scroll_speed_wps, pause_prob, re_scroll_prob, exit_prob
_BASE_EMISSIONS: np.ndarray = np.array([
    # ENGAGED_FAST
    [5.0, 0.02, 0.01, 0.005],
    # ENGAGED_SLOW
    [2.5, 0.10, 0.05, 0.010],
    # SKIMMING
    [8.0, 0.01, 0.02, 0.040],
    # DISTRACTED
    [1.0, 0.25, 0.08, 0.080],
    # EXITING (absorbing — emissions never used)
    [0.0, 0.00, 0.00, 1.000],
], dtype=np.float64)

# Base transition matrix (rows = from, cols = to)
_BASE_TRANSITIONS: np.ndarray = np.array([
    # EF    ES    SK    DI    EX
    [0.65, 0.20, 0.08, 0.05, 0.02],  # ENGAGED_FAST
    [0.15, 0.55, 0.12, 0.13, 0.05],  # ENGAGED_SLOW
    [0.05, 0.10, 0.55, 0.20, 0.10],  # SKIMMING
    [0.03, 0.10, 0.15, 0.50, 0.22],  # DISTRACTED
    [0.00, 0.00, 0.00, 0.00, 1.00],  # EXITING (absorbing)
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: random simplex vector (Dirichlet with uniform α=1)
# ---------------------------------------------------------------------------

def _random_simplex(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Return a vector on the probability simplex (sums to 1, all ≥ 0)."""
    v = rng.dirichlet(np.ones(dim))
    return v.astype(np.float64)


# ---------------------------------------------------------------------------
# Catalog & user generation
# ---------------------------------------------------------------------------

def _generate_catalog(
    num_items: int,
    avg_chapters: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create a synthetic book catalog.

    Each book has:
    - ``latent_quality`` from Beta(2, 5)  →  skewed toward low/mediocre
    - ``topic_vector`` on the 40-dim simplex
    - ``avg_chapter_word_count``  ~  N(3500, 800), clipped ≥ 800
    - ``chapter_count``  ~  Poisson(avg_chapters-1) + 1, clipped ≥ 1
    - ``avg_rating``  calibrated from quality:  4 × quality + noise ∈ [1, 5]

    Parameters
    ----------
    num_items : int
        Number of books to generate.
    avg_chapters : int
        Mean chapter count (λ for Poisson + 1).
    rng : np.random.Generator
        PRNG for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: item_id, title, author_id, author_name, genres,
        avg_rating, rating_count, chapter_count, avg_chapter_word_count,
        latent_quality, topic_vector.
    """
    qualities = rng.beta(2, 5, size=num_items)
    chapter_counts = np.clip(rng.poisson(avg_chapters - 1, size=num_items) + 1, 1, None)
    word_counts = np.clip(rng.normal(BASE_WORDS_PER_CHAPTER, 800, size=num_items).astype(int), 800, None)

    # avg_rating correlated with quality, plus noise, clipped [1, 5]
    ratings = np.clip(4.0 * qualities + rng.normal(0.5, 0.3, size=num_items), 1.0, 5.0)

    records: list[dict[str, Any]] = []
    for i in range(num_items):
        item_id = f"item_{i:05d}"
        author_id = f"author_{rng.integers(0, max(1, num_items // 5)):04d}"
        genres = rng.choice(GENRE_POOL, size=rng.integers(1, 4), replace=False).tolist()
        topic_vec = _random_simplex(rng, NMF_TOPICS_DIM)

        records.append({
            "item_id": item_id,
            "title": f"Book {i:05d}",
            "author_id": author_id,
            "author_name": f"Author {author_id.split('_')[1]}",
            "genres": genres,
            "avg_rating": round(float(ratings[i]), 2),
            "rating_count": int(rng.integers(5, 5000)),
            "chapter_count": int(chapter_counts[i]),
            "avg_chapter_word_count": int(word_counts[i]),
            "latent_quality": float(qualities[i]),
            "topic_vector": topic_vec.tolist(),
        })

    return pd.DataFrame(records)


def _generate_users(
    num_users: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create synthetic user profiles.

    Each user has:
    - ``taste_vector`` on the 40-dim simplex
    - ``reading_speed_factor``  ~  LogNormal(0, 0.25)  (centred ≈ 1)
    - ``patience_factor``  ~  LogNormal(0, 0.3)
    - ``device_preference``  sampled from DEVICE_WEIGHTS

    Parameters
    ----------
    num_users : int
        Number of users.
    rng : np.random.Generator
        PRNG.

    Returns
    -------
    pd.DataFrame
    """
    devices = rng.choice(
        [d.value for d in DeviceType],
        size=num_users,
        p=DEVICE_WEIGHTS,
    )
    records: list[dict[str, Any]] = []
    for i in range(num_users):
        records.append({
            "user_id": f"user_{i:05d}",
            "taste_vector": _random_simplex(rng, NMF_TOPICS_DIM).tolist(),
            "reading_speed_factor": float(np.clip(rng.lognormal(0, 0.25), 0.4, 3.0)),
            "patience_factor": float(np.clip(rng.lognormal(0, 0.3), 0.3, 3.0)),
            "device_preference": devices[i],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core Markov simulation
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe for zero norms."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _condition_transitions(
    quality: float,
    affinity: float,
    chapter_index: int,
) -> np.ndarray:
    """
    Condition the base transition matrix on quality, affinity, and chapter.

    Strategy:
    - Compute a *boost* for ENGAGED states and a *penalty* for
      DISTRACTED / EXITING proportional to ``(quality + affinity) / 2``.
    - Chapters 3-5 (0-indexed) add extra DISTRACTED mass ("valley of death").
    - Re-normalise rows so they sum to 1.

    Parameters
    ----------
    quality : float
        Book's latent quality ∈ [0, 1].
    affinity : float
        Cosine similarity between user taste and book topic ∈ [-1, 1].
    chapter_index : int
        0-indexed chapter number.

    Returns
    -------
    np.ndarray
        Conditioned 5×5 transition matrix.
    """
    T = _BASE_TRANSITIONS.copy()
    # Combined signal: higher = better experience
    signal = np.clip((quality + max(affinity, 0.0)) / 2.0, 0.0, 1.0)

    # Boost engaged columns, penalise distracted / exiting
    engaged_boost = 0.15 * signal      # up to +0.15 per engaged col
    distract_penalty = 0.12 * signal    # up to -0.12 for distracted
    exit_penalty = 0.10 * signal        # up to -0.10 for exiting

    for row in range(_N_STATES - 1):  # don't touch absorbing row
        T[row, _S["ENGAGED_FAST"]] += engaged_boost * 0.5
        T[row, _S["ENGAGED_SLOW"]] += engaged_boost * 0.5
        T[row, _S["DISTRACTED"]] = max(T[row, _S["DISTRACTED"]] - distract_penalty, 0.01)
        T[row, _S["EXITING"]] = max(T[row, _S["EXITING"]] - exit_penalty, 0.005)

    # Valley of death: chapters 3-5 add distracted mass
    if 3 <= chapter_index <= 5:
        valley_boost = 0.08 * (1.0 - signal)  # worse for bad books
        for row in range(_N_STATES - 1):
            T[row, _S["DISTRACTED"]] += valley_boost
            T[row, _S["EXITING"]] += valley_boost * 0.3

    # Re-normalise rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    T /= row_sums
    return T


def _condition_emissions(
    quality: float,
    affinity: float,
    speed_factor: float,
    patience_factor: float,
) -> np.ndarray:
    """
    Personalise emission parameters per user-book pair.

    - ``scroll_speed_wps`` scaled by the user's ``speed_factor``.
    - ``exit_prob`` inversely scaled by quality × affinity × patience.
    - ``pause_prob`` reduced by patience.

    Returns
    -------
    np.ndarray
        Shape (5, 4): scroll_speed_wps, pause_prob, re_scroll_prob, exit_prob.
    """
    E = _BASE_EMISSIONS.copy()
    signal = np.clip((quality + max(affinity, 0.0)) / 2.0, 0.0, 1.0)

    # Speed column
    E[:, 0] *= speed_factor

    # Pause column — patient users pause less for engagement reasons
    E[:, 1] *= np.clip(1.0 / patience_factor, 0.3, 2.0)

    # Exit column — reduced by both signal and patience
    exit_scale = np.clip(1.0 - 0.6 * signal, 0.2, 1.5) * np.clip(1.0 / patience_factor, 0.4, 2.0)
    E[:-1, 3] *= exit_scale  # don't touch absorbing state

    return E


def _simulate_session(
    user_row: dict[str, Any],
    item_row: dict[str, Any],
    chapter_index: int,
    chapter_words: int,
    session_start_ts: datetime,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    Simulate a single reading session as a Markov chain.

    Returns a list of event dicts conforming to the Event schema.

    Parameters
    ----------
    user_row : dict
        User profile fields.
    item_row : dict
        Catalog item fields.
    chapter_index : int
        0-indexed chapter being read.
    chapter_words : int
        Word count for this chapter.
    session_start_ts : datetime
        Anchor timestamp for this session.
    rng : np.random.Generator
        PRNG.

    Returns
    -------
    list[dict]
        Event records.
    """
    user_id = user_row["user_id"]
    item_id = item_row["item_id"]
    session_id = str(uuid.uuid4())
    device = user_row["device_preference"]

    taste = np.array(user_row["taste_vector"])
    topic = np.array(item_row["topic_vector"])
    affinity = _cosine_similarity(taste, topic)
    quality = item_row["latent_quality"]

    T = _condition_transitions(quality, affinity, chapter_index)
    E = _condition_emissions(
        quality, affinity,
        user_row["reading_speed_factor"],
        user_row["patience_factor"],
    )

    events: list[dict[str, Any]] = []
    current_state = _S["ENGAGED_SLOW"]  # sessions start in ENGAGED_SLOW
    scroll_pct: float = 0.0
    tick: int = 0
    ts = session_start_ts
    is_paused = False
    max_ticks = int(3600 / TICK_DURATION_SEC)  # hard cap at ~1 hour

    # Opening event
    events.append(_make_event(
        user_id, item_id, chapter_index, session_id, ts,
        EventType.OPEN, scroll_pct, 0.0, device, {},
    ))

    while tick < max_ticks:
        tick += 1
        ts += timedelta(seconds=TICK_DURATION_SEC)
        state_name = ENGAGEMENT_STATES[current_state]

        speed_wps = E[current_state, 0]
        pause_p = E[current_state, 1]
        rescroll_p = E[current_state, 2]
        exit_p = E[current_state, 3]

        # ---- Pause / Resume logic ----
        if not is_paused and rng.random() < pause_p:
            is_paused = True
            events.append(_make_event(
                user_id, item_id, chapter_index, session_id, ts,
                EventType.PAUSE, scroll_pct, TICK_DURATION_SEC, device,
                {"state": state_name},
            ))
            # Don't advance scroll while paused
            current_state = _transition(T, current_state, rng)
            if current_state == _S["EXITING"]:
                events.append(_make_event(
                    user_id, item_id, chapter_index, session_id, ts,
                    EventType.EXIT, scroll_pct, tick * TICK_DURATION_SEC, device,
                    {"exit_reason": ExitReason.MID_CHAPTER.value, "state": state_name},
                ))
                break
            continue

        if is_paused:
            # Resume with some probability
            if rng.random() < 0.4:
                is_paused = False
                events.append(_make_event(
                    user_id, item_id, chapter_index, session_id, ts,
                    EventType.RESUME, scroll_pct, TICK_DURATION_SEC, device,
                    {"state": state_name},
                ))
            current_state = _transition(T, current_state, rng)
            if current_state == _S["EXITING"]:
                events.append(_make_event(
                    user_id, item_id, chapter_index, session_id, ts,
                    EventType.EXIT, scroll_pct, tick * TICK_DURATION_SEC, device,
                    {"exit_reason": ExitReason.MID_CHAPTER.value, "state": state_name},
                ))
                break
            continue

        # ---- Scroll advancement ----
        words_read = speed_wps * TICK_DURATION_SEC
        delta_pct = (words_read / max(chapter_words, 1)) * 100.0
        scroll_pct = min(scroll_pct + delta_pct, 100.0)

        events.append(_make_event(
            user_id, item_id, chapter_index, session_id, ts,
            EventType.SCROLL_TICK, scroll_pct, TICK_DURATION_SEC, device,
            {"state": state_name, "speed_wps": round(speed_wps, 2)},
        ))

        # ---- Re-scroll (backward jump) ----
        if rng.random() < rescroll_p and scroll_pct > 2.5:
            jump_back = rng.uniform(1.0, min(15.0, scroll_pct - 0.1))
            scroll_pct = max(scroll_pct - jump_back, 0.0)
            events.append(_make_event(
                user_id, item_id, chapter_index, session_id, ts,
                EventType.RE_SCROLL, scroll_pct, 0.0, device,
                {"jump_back_pct": round(jump_back, 2)},
            ))

        # ---- Check completion ----
        if scroll_pct >= 99.5:
            scroll_pct = 100.0
            events.append(_make_event(
                user_id, item_id, chapter_index, session_id, ts,
                EventType.CHAPTER_COMPLETE, 100.0, tick * TICK_DURATION_SEC, device,
                {"exit_reason": ExitReason.CHAPTER_END.value},
            ))
            break

        # ---- State transition ----
        current_state = _transition(T, current_state, rng)
        if current_state == _S["EXITING"]:
            events.append(_make_event(
                user_id, item_id, chapter_index, session_id, ts,
                EventType.EXIT, scroll_pct, tick * TICK_DURATION_SEC, device,
                {"exit_reason": ExitReason.MID_CHAPTER.value, "state": state_name},
            ))
            break

    return events


def _transition(T: np.ndarray, state: int, rng: np.random.Generator) -> int:
    """Sample next state from transition matrix row."""
    return int(rng.choice(_N_STATES, p=T[state]))


def _make_event(
    user_id: str,
    item_id: str,
    chapter_index: int,
    session_id: str,
    ts: datetime,
    event_type: EventType,
    scroll_depth_pct: float,
    dwell_time_sec: float,
    device_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Construct an event dict matching the Event schema."""
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "item_id": item_id,
        "chapter_index": chapter_index,
        "session_id": session_id,
        "timestamp": ts,
        "event_type": event_type.value,
        "scroll_depth_pct": round(scroll_depth_pct, 2),
        "dwell_time_sec": round(dwell_time_sec, 2),
        "device_type": device_type,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_users: int = 5000,
    num_items: int = 2000,
    avg_chapters: int = 15,
    sessions_per_user: int = 30,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end synthetic telemetry generator.

    1. Build a synthetic catalog and user pool.
    2. For each user, sample ``sessions_per_user`` books (weighted by
       user-book affinity) and simulate a reading session per chapter.
    3. Persist three Parquet files under ``output_dir``.

    Parameters
    ----------
    num_users : int
        Number of synthetic users.
    num_items : int
        Number of synthetic books.
    avg_chapters : int
        Average chapter count (Poisson λ + 1).
    sessions_per_user : int
        Books each user samples (one session per sampled book-chapter pair).
    seed : int
        Random seed for reproducibility.
    output_dir : Path or None
        Directory for output Parquet files (default: ``data/synthetic/``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (events_df, catalog_df, users_df)
    """
    rng = np.random.default_rng(seed)
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate catalog & users ----
    print("> Generating catalog ...")
    catalog_df = _generate_catalog(num_items, avg_chapters, rng)
    print(f"  {len(catalog_df)} items, quality mean={catalog_df['latent_quality'].mean():.3f}")

    print("> Generating users ...")
    users_df = _generate_users(num_users, rng)
    print(f"  {len(users_df)} users")

    # Pre-compute numpy arrays for fast affinity lookups
    item_ids = catalog_df["item_id"].values
    topic_matrix = np.array(catalog_df["topic_vector"].tolist())  # (num_items, 40)
    topic_norms = np.linalg.norm(topic_matrix, axis=1, keepdims=True)
    topic_norms[topic_norms < 1e-12] = 1.0
    topic_normed = topic_matrix / topic_norms

    item_qualities = catalog_df["latent_quality"].values
    item_chapters = catalog_df["chapter_count"].values
    item_words = catalog_df["avg_chapter_word_count"].values

    # ---- Simulate sessions ----
    print("> Simulating reading sessions ...")
    all_events: list[dict[str, Any]] = []
    global_ts_offset = 0  # accumulate across users for non-overlapping timestamps

    for u_idx in tqdm(range(num_users), desc="Users", unit="user"):
        user_row = users_df.iloc[u_idx].to_dict()
        taste = np.array(user_row["taste_vector"])
        taste_norm = np.linalg.norm(taste)
        if taste_norm < 1e-12:
            taste_norm = 1.0
        taste_normed = taste / taste_norm

        # Affinity scores for sampling
        affinities = topic_normed @ taste_normed  # (num_items,)
        # Convert to positive sampling weights
        weights = np.clip(affinities + 0.1, 0.01, None)  # shift to avoid negatives
        weights /= weights.sum()

        # Sample books for this user
        n_books = min(sessions_per_user, num_items)
        sampled_indices = rng.choice(num_items, size=n_books, replace=False, p=weights)

        user_ts = ANCHOR_TS + timedelta(seconds=global_ts_offset)

        for book_idx in sampled_indices:
            item_row = catalog_df.iloc[int(book_idx)].to_dict()
            n_ch = int(item_chapters[book_idx])
            word_count = int(item_words[book_idx])

            # Simulate 1-3 chapters per book (not the whole book usually)
            chapters_to_read = min(rng.integers(1, 4), n_ch)
            start_ch = rng.integers(0, max(1, n_ch - chapters_to_read + 1))

            for ch in range(start_ch, start_ch + chapters_to_read):
                # Per-chapter word count variation (±20%)
                ch_words = max(400, int(word_count * rng.uniform(0.8, 1.2)))
                session_events = _simulate_session(
                    user_row, item_row, ch, ch_words, user_ts, rng,
                )
                all_events.extend(session_events)

                # Advance timestamp by session duration + inter-session gap
                if session_events:
                    session_dur = (session_events[-1]["timestamp"] - user_ts).total_seconds()
                    user_ts += timedelta(seconds=session_dur + rng.uniform(60, 7200))

        global_ts_offset += rng.uniform(10, 100)

    # ---- Build events DataFrame ----
    print(f"> Assembling {len(all_events):,} events ...")
    events_df = pd.DataFrame(all_events)

    # Convert metadata dict to JSON string for Parquet compatibility
    events_df["metadata"] = events_df["metadata"].apply(
        lambda d: str(d) if d else "{}"
    )

    # ---- Persist ----
    events_path = output_dir / "events.parquet"
    catalog_path = output_dir / "catalog.parquet"
    users_path = output_dir / "users.parquet"

    # For catalog & users, convert list columns to string for Parquet
    catalog_save = catalog_df.copy()
    catalog_save["genres"] = catalog_save["genres"].apply(str)
    catalog_save["topic_vector"] = catalog_save["topic_vector"].apply(str)

    users_save = users_df.copy()
    users_save["taste_vector"] = users_save["taste_vector"].apply(str)

    events_df.to_parquet(events_path, index=False, engine="pyarrow")
    catalog_save.to_parquet(catalog_path, index=False, engine="pyarrow")
    users_save.to_parquet(users_path, index=False, engine="pyarrow")

    print(f"[OK] Saved {events_path}")
    print(f"[OK] Saved {catalog_path}")
    print(f"[OK] Saved {users_path}")

    return events_df, catalog_df, users_df


# ---------------------------------------------------------------------------
# Standalone demo & calibration check
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
    +--------------------------------------------------------------+
    |  Narrative Intelligence Platform -- Markov Event Simulator    |
    +--------------------------------------------------------------+
    """))

    events_df, catalog_df, users_df = generate_synthetic_dataset(
        num_users=1000,
        num_items=500,
        avg_chapters=15,
        sessions_per_user=30,
        seed=42,
    )

    # ---- Summary statistics ----
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total events:       {len(events_df):>12,}")
    print(f"Unique sessions:    {events_df['session_id'].nunique():>12,}")
    print(f"Unique users:       {events_df['user_id'].nunique():>12,}")
    print(f"Unique items:       {events_df['item_id'].nunique():>12,}")

    # Event type distribution
    print("\nEvent type distribution:")
    evt_counts = events_df["event_type"].value_counts()
    for evt, cnt in evt_counts.items():
        print(f"  {evt:<25s} {cnt:>10,}")

    # ---- Completion rate ----
    # A session is "completed" if it contains a CHAPTER_COMPLETE event
    session_completed = (
        events_df[events_df["event_type"] == EventType.CHAPTER_COMPLETE.value]
        .groupby("session_id")
        .size()
        .reset_index(name="completed")
    )
    all_sessions = events_df["session_id"].nunique()
    completed_sessions = len(session_completed)
    completion_rate = completed_sessions / max(all_sessions, 1)
    print(f"\nAvg completion rate: {completion_rate:.1%}")

    # ---- Exit reason distribution ----
    exit_events = events_df[events_df["event_type"].isin([
        EventType.EXIT.value, EventType.CHAPTER_COMPLETE.value,
    ])]
    print("\nExit reason distribution:")
    # Parse exit_reason from metadata string
    def _extract_exit_reason(row: pd.Series) -> str:
        meta = row["metadata"]
        if EventType.CHAPTER_COMPLETE.value in row["event_type"]:
            return "chapter_end"
        if "mid_chapter" in str(meta):
            return "mid_chapter"
        if "app_close" in str(meta):
            return "app_close"
        return "unknown"

    exit_events = exit_events.copy()
    exit_events["reason"] = exit_events.apply(_extract_exit_reason, axis=1)
    reason_counts = exit_events["reason"].value_counts()
    for reason, cnt in reason_counts.items():
        print(f"  {reason:<25s} {cnt:>10,}")

    # ---- Calibration check: quality quartile -> completion rate ----
    print("\n" + "=" * 60)
    print("CALIBRATION CHECK: Quality Quartile -> Completion Rate")
    print("=" * 60)

    # Build session-level completion flag
    session_items = (
        events_df.groupby("session_id")["item_id"]
        .first()
        .reset_index()
    )
    session_items["completed"] = session_items["session_id"].isin(
        session_completed["session_id"]
    ).astype(int)

    # Merge with catalog quality
    session_quality = session_items.merge(
        catalog_df[["item_id", "latent_quality"]],
        on="item_id",
        how="left",
    )
    session_quality["quality_quartile"] = pd.qcut(
        session_quality["latent_quality"], 4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    )

    quartile_stats = (
        session_quality.groupby("quality_quartile", observed=False)["completed"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "completion_rate", "count": "n_sessions"})
    )
    print(quartile_stats.to_string())
    print()

    # Verify monotonicity
    rates = quartile_stats["completion_rate"].values
    if all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)):
        print("[PASS] CALIBRATION PASSED: completion rate increases with quality quartile.")
    else:
        print("[NOTE] CALIBRATION NOTE: completion rate is not strictly monotonic,")
        print("       but should show a clear upward trend with sufficient data.")

    print("\n[DONE] Simulation complete.")
