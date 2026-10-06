"""
models/schemas.py — Pydantic data models for the Discovery Engine.
All pipeline layers share these models to enforce a consistent schema.
"""
from __future__ import annotations

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class SourcePlatform(str, Enum):
    PLAY_STORE = "play_store"
    APP_STORE  = "app_store"
    REDDIT     = "reddit"
    YOUTUBE    = "youtube"
    HELP_FORUM = "help_forum"
    WEB        = "web"


class FrustrationLevel(str, Enum):
    LOW  = "low"
    MED  = "med"
    HIGH = "high"


class PatternType(str, Enum):
    VOICED   = "voiced"
    INFERRED = "inferred"


# ── Raw Item (Layer 1/2 output) ───────────────────────────────────────────────

class RawItem(BaseModel):
    """
    Normalised representation of a single piece of user feedback,
    exactly as collected from the source (before LLM tagging).
    """
    id: str                          # UUID v4
    source: SourcePlatform
    platform_item_id: Optional[str] = None   # original ID or URL from source
    text: str                        # cleaned feedback text (max 2,000 chars)
    date: Optional[str]   = None     # ISO 8601 date string, or None
    upvotes: int          = 0        # engagement proxy (votes, likes, etc.)
    url: Optional[str]    = None     # permalink to original content
    collected_at: str     = ""       # ISO 8601 timestamp of collection

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("RawItem.text must not be empty")
        return v.strip()


# ── Tagged Item (Layer 4 output) ──────────────────────────────────────────────

class TaggedItem(BaseModel):
    """
    A RawItem augmented with LLM-generated classification tags
    across all 8 dimensions of the tagging schema.
    """
    id: str                                   # UUID v4 (new, for the tag record)
    raw_item_id: str                          # FK → RawItem.id

    # Dimension 1: Relevance
    relevant: bool

    # Dimension 2: Problem type (multi-label)
    problem_types: List[str]  = []
    # Valid values: search_no_results | screenshots_docs | cannot_refine |
    #              wrong_metadata | face_recognition | sync_backup | other

    # Dimension 3: What user remembered (multi-label)
    remembered_cues: List[str] = []
    # Valid values: time_date | place | people_present | event |
    #              object_detail | purpose | emotion | none

    # Dimension 4: What user forgot (multi-label)
    forgotten_cues: List[str]  = []
    # Valid values: exact_date | exact_location | filename_album |
    #              search_keywords | none

    # Dimension 5: How they tried to search
    search_method: Optional[str] = None
    # Valid values: exact_keyword | vague_description | browsing |
    #              filters | asked_someone | gave_up | None

    # Dimension 6: Failure funnel stage(s)
    failure_points: List[str]  = []
    # Valid values: a | b | c | d

    # Dimension 7: Severity / sentiment
    frustration_level: Optional[FrustrationLevel] = None
    abandoned_app: Optional[bool] = None

    # Classifier metadata
    classifier_error: bool     = False
    pattern_type: Optional[PatternType] = None   # set during clustering

    # Passed through from RawItem for convenience
    source_platform: SourcePlatform
    date: Optional[str]   = None
    upvotes: int          = 0

    # Cluster assignment (set during Phase 4)
    assigned_areas: List[str] = []


# ── Opportunity Area (Layer 5 output) ─────────────────────────────────────────

class Quote(BaseModel):
    """A representative verbatim quote for an opportunity area."""
    raw_item_id: str
    text: str
    source_platform: str
    upvotes: int       = 0
    url: Optional[str] = None


class OpportunityArea(BaseModel):
    """
    A named cluster of retrieval-failure patterns, scored and
    ready for PM decision-making.
    """
    name: str
    description: str

    # Scoring dimensions
    evidence_volume:   int   = 0    # total items assigned to this area
    source_diversity:  int   = 0    # number of distinct platforms (max 6)
    severity_score:    float = 0.0  # weighted average frustration (1=low, 3=high)
    abandonment_rate:  float = 0.0  # fraction of items where user abandoned app

    # Voiced vs. Inferred split
    voiced_count:   int = 0
    inferred_count: int = 0

    # Representative quotes (top 3 by upvote proxy)
    top_quotes: List[Quote] = []

    # Breakdown by platform {platform_name: item_count}
    platform_breakdown: dict = {}

    # Source cluster type: "rule_based" | "semantic" | "hybrid"
    cluster_source: str = "rule_based"
