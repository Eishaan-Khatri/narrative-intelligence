"""
Narrative Intelligence Platform — Unified Data Schema
=====================================================
Pydantic models for the unified event schema and all feature store tables.
This is the single source of truth that every pipeline script imports.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Event Schema
# ---------------------------------------------------------------------------

class EventType(str, enum.Enum):
    IMPRESSION = "IMPRESSION"
    OPEN = "OPEN"
    SCROLL_TICK = "SCROLL_TICK"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    RE_SCROLL = "RE_SCROLL"
    CHAPTER_COMPLETE = "CHAPTER_COMPLETE"
    EXIT = "EXIT"
    RATING = "RATING"
    REVIEW = "REVIEW"
    FOLLOW_AUTHOR = "FOLLOW_AUTHOR"
    SAVE = "SAVE"
    BOOKMARK = "BOOKMARK"
    SHARE = "SHARE"


class DeviceType(str, enum.Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class ExitReason(str, enum.Enum):
    CHAPTER_END = "chapter_end"
    MID_CHAPTER = "mid_chapter"
    APP_CLOSE = "app_close"


class CompletionCurveShape(str, enum.Enum):
    CLIFF = "cliff"
    DECAY = "decay"
    STEADY = "steady"
    ABANDON_EARLY = "abandon_early"


class Event(BaseModel):
    """A single telemetry event in the unified event stream."""
    event_id: str
    user_id: str
    item_id: str
    chapter_index: int = 0  # 0 for single-shot items (e.g. ratings)
    session_id: str
    timestamp: datetime
    event_type: EventType
    scroll_depth_pct: Optional[float] = None  # 0-100
    dwell_time_sec: Optional[float] = None
    device_type: DeviceType = DeviceType.MOBILE
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Feature Store Table Schemas
# ---------------------------------------------------------------------------

class SessionFeatures(BaseModel):
    """One row per reading session (Section 3.1)."""
    session_id: str
    user_id: str
    item_id: str
    chapter_index: int
    session_duration_sec: float
    reading_velocity_wpm: float  # capped [50, 600]
    velocity_acceleration: float  # positive = speeding up
    completion_curve: list[float]  # 5-element vector
    completion_curve_shape: CompletionCurveShape
    re_read_ratio: float
    final_completion_pct: float
    exit_reason: ExitReason
    device_type: DeviceType
    timestamp_start: datetime
    timestamp_end: datetime


class UserTemporalFeatures(BaseModel):
    """One row per user per weekly time bucket (Section 3.2)."""
    user_id: str
    time_bucket: str  # ISO week string, e.g. "2024-W05"
    engagement_profile_vector: list[float]  # 8-dim
    inter_chapter_gap_trend: float
    genre_drift_flag: bool = False


class ItemFingerprint(BaseModel):
    """81-dimensional item fingerprint (Section 3.3)."""
    item_id: str
    topic_vector: list[float]  # 40-dim (NMF)
    author_embedding: list[float]  # 32-dim
    quality_score: float  # 1-dim PCA composite
    structural_features: list[float]  # 8-dim


class DropoutHazard(BaseModel):
    """Per (user, item, chapter) hazard score (Section 3.4)."""
    user_id: str
    item_id: str
    chapter_index: int
    hazard_score: float


# ---------------------------------------------------------------------------
# 3. Catalog / Metadata Schemas
# ---------------------------------------------------------------------------

class BookMetadata(BaseModel):
    """Metadata for a single book/story in the catalog."""
    item_id: str
    title: str
    author_id: str
    author_name: str
    description: str = ""
    genres: list[str] = Field(default_factory=list)
    avg_rating: float = 0.0
    rating_count: int = 0
    review_count: int = 0
    page_count: int = 0
    chapter_count: int = 1
    publish_date: Optional[datetime] = None
    series_id: Optional[str] = None
    series_position: Optional[int] = None
    language: str = "en"


class UserProfile(BaseModel):
    """Lightweight user profile for simulation."""
    user_id: str
    taste_vector: list[float] = Field(default_factory=list)  # latent NMF-dim taste
    reading_speed_factor: float = 1.0  # multiplier on base reading speed
    patience_factor: float = 1.0  # multiplier on tolerance for low-quality
    device_preference: DeviceType = DeviceType.MOBILE


# ---------------------------------------------------------------------------
# 4. Utility Constants
# ---------------------------------------------------------------------------

# Feature dimensions
NMF_TOPICS_DIM = 40
AUTHOR_EMBEDDING_DIM = 32
STRUCTURAL_FEATURES_DIM = 8
ITEM_FINGERPRINT_DIM = NMF_TOPICS_DIM + AUTHOR_EMBEDDING_DIM + 1 + STRUCTURAL_FEATURES_DIM  # 81
USER_EMBEDDING_DIM = 128
ITEM_EMBEDDING_DIM = 128
ENGAGEMENT_PROFILE_DIM = 8

# User tower input dimensions
USER_TOWER_ITEM_POOL_DIM = ITEM_EMBEDDING_DIM  # 128 (or ITEM_FINGERPRINT_DIM=81 during bootstrap)
USER_TOWER_CONTEXT_DIM = 10  # time_of_day(5) + device(2) + session_intent(3)
USER_TOWER_INPUT_DIM = USER_TOWER_ITEM_POOL_DIM + ENGAGEMENT_PROFILE_DIM + USER_TOWER_CONTEXT_DIM  # 146

# Item tower input dimensions
ITEM_TOWER_INPUT_DIM = ITEM_FINGERPRINT_DIM + 2  # 83 (fingerprint + log_interactions + velocity)

# Simulation defaults
INACTIVITY_GAP_SEC = 1800  # 30-minute session boundary
READING_VELOCITY_MIN_WPM = 50
READING_VELOCITY_MAX_WPM = 600
COMPLETION_CURVE_POINTS = 5

# Markov simulator states
ENGAGEMENT_STATES = ["ENGAGED_FAST", "ENGAGED_SLOW", "SKIMMING", "DISTRACTED", "EXITING"]


def flatten_fingerprint(fp: ItemFingerprint) -> np.ndarray:
    """Flatten an ItemFingerprint into a single 81-dim numpy array."""
    return np.concatenate([
        np.array(fp.topic_vector),
        np.array(fp.author_embedding),
        np.array([fp.quality_score]),
        np.array(fp.structural_features),
    ])


if __name__ == "__main__":
    # Quick validation: create sample instances to verify schema integrity
    sample_event = Event(
        event_id="evt_001",
        user_id="u_001",
        item_id="i_001",
        chapter_index=0,
        session_id="sess_001",
        timestamp=datetime.now(),
        event_type=EventType.SCROLL_TICK,
        scroll_depth_pct=45.2,
        dwell_time_sec=12.5,
        device_type=DeviceType.MOBILE,
    )
    print(f"✓ Event schema valid: {sample_event.event_type.value}")

    sample_fp = ItemFingerprint(
        item_id="i_001",
        topic_vector=[0.0] * NMF_TOPICS_DIM,
        author_embedding=[0.0] * AUTHOR_EMBEDDING_DIM,
        quality_score=0.72,
        structural_features=[0.0] * STRUCTURAL_FEATURES_DIM,
    )
    vec = flatten_fingerprint(sample_fp)
    print(f"✓ ItemFingerprint dim: {vec.shape[0]} (expected {ITEM_FINGERPRINT_DIM})")

    sample_session = SessionFeatures(
        session_id="sess_001", user_id="u_001", item_id="i_001",
        chapter_index=0, session_duration_sec=300.0,
        reading_velocity_wpm=220.0, velocity_acceleration=-15.0,
        completion_curve=[0.1, 0.3, 0.55, 0.78, 0.92],
        completion_curve_shape=CompletionCurveShape.STEADY,
        re_read_ratio=0.05, final_completion_pct=0.92,
        exit_reason=ExitReason.CHAPTER_END,
        device_type=DeviceType.MOBILE,
        timestamp_start=datetime.now(), timestamp_end=datetime.now(),
    )
    print(f"✓ SessionFeatures valid: shape={sample_session.completion_curve_shape.value}")

    print("\n✅ All schemas validated successfully.")
