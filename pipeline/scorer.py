"""
pipeline/scorer.py — Opportunity Scoring & Open-Question Flagging (Phase 4).

Computes all scoring dimensions for each OpportunityArea:
  - evidence_volume   : total items assigned
  - source_diversity  : distinct platform count (1–6)
  - severity_score    : weighted avg frustration (low=1, med=2, high=3)
  - abandonment_rate  : fraction where user abandoned the app
  - voiced_count      : items with VOICED pattern type
  - inferred_count    : items with INFERRED pattern type
  - top_quotes        : 3 highest-upvote verbatim quotes
  - platform_breakdown: {platform: count}

Composite rank score = severity_score × log1p(evidence_volume) × source_diversity
(Higher is a stronger signal for PM prioritisation)

Open-question flagging (4.6):
  - inferred > voiced      → "Hidden latent need"
  - high volume, 1 source  → "Potentially platform-specific noise"
  - high abandonment (>30%)→ "Critical experience break"
  - 0 top quotes           → "Needs manual quote curation"
"""
from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from statistics import mean
from typing import Dict, List, Tuple

from loguru import logger

from models.schemas import (
    TaggedItem,
    OpportunityArea,
    Quote,
    PatternType,
    FrustrationLevel,
)
from pipeline.clusterer import (
    OPPORTUNITY_RULES,
    UNCATEGORIZED_AREA,
    UNCATEGORIZED_DESC,
)

# Map area_name → description from the rule definitions
AREA_DESCRIPTIONS: Dict[str, str] = {
    name: desc for _, _, name, desc in OPPORTUNITY_RULES
}
AREA_DESCRIPTIONS[UNCATEGORIZED_AREA] = UNCATEGORIZED_DESC

FRUSTRATION_WEIGHTS = {
    FrustrationLevel.LOW:  1.0,
    FrustrationLevel.MED:  2.0,
    FrustrationLevel.HIGH: 3.0,
}


# ── 4.4 Opportunity Scorer ────────────────────────────────────────────────────

def score_opportunity_areas(
    area_assignments: Dict[str, List[TaggedItem]],
    raw_texts: Dict[str, str],          # raw_item_id → text (for quotes)
    raw_urls:  Dict[str, str],          # raw_item_id → url
    cluster_source: str = "rule_based", # "rule_based" | "semantic" | "hybrid"
) -> List[OpportunityArea]:
    """
    Build a fully-scored OpportunityArea object for each cluster.

    Args:
        area_assignments : {area_name → list of TaggedItems}
        raw_texts        : {raw_item_id → original text} — needed for quote text
        raw_urls         : {raw_item_id → url} — needed for quote links
        cluster_source   : how this area was produced

    Returns:
        List of OpportunityArea objects, sorted by composite rank (desc).
    """
    scored: List[OpportunityArea] = []

    for area_name, items in area_assignments.items():
        if not items:
            continue

        # ── Scoring dimensions ────────────────────────────────────────────
        evidence_volume  = len(items)
        source_diversity = len({i.source_platform for i in items})

        # Severity score — weighted mean (items with no frustration_level skipped)
        fl_weights = [
            FRUSTRATION_WEIGHTS[i.frustration_level]
            for i in items
            if i.frustration_level is not None
        ]
        severity_score = round(mean(fl_weights), 3) if fl_weights else 0.0

        # Abandonment rate
        abandoned_items = [
            i for i in items if i.abandoned_app is True
        ]
        abandonment_rate = round(len(abandoned_items) / evidence_volume, 4)

        # Voiced / Inferred split
        voiced_count   = sum(
            1 for i in items if i.pattern_type == PatternType.VOICED
        )
        inferred_count = sum(
            1 for i in items if i.pattern_type == PatternType.INFERRED
        )

        # Platform breakdown
        platform_breakdown = dict(
            Counter(i.source_platform.value for i in items)
        )

        # Top 3 quotes by upvotes (only items whose text we have)
        top_quotes = _pick_top_quotes(items, raw_texts, raw_urls, n=3)

        # Area description (fall back to generic if unlabelled)
        description = AREA_DESCRIPTIONS.get(
            area_name,
            f"Auto-discovered cluster: {area_name}.",
        )

        # ── Determine cluster_source for this area ────────────────────────
        area_source = cluster_source
        if area_name in {name for _, _, name, _ in OPPORTUNITY_RULES}:
            area_source = "rule_based"
        elif area_name == UNCATEGORIZED_AREA:
            area_source = "rule_based"
        else:
            area_source = "semantic"

        scored.append(
            OpportunityArea(
                name=area_name,
                description=description,
                evidence_volume=evidence_volume,
                source_diversity=source_diversity,
                severity_score=severity_score,
                abandonment_rate=abandonment_rate,
                voiced_count=voiced_count,
                inferred_count=inferred_count,
                top_quotes=top_quotes,
                platform_breakdown=platform_breakdown,
                cluster_source=area_source,
            )
        )

    # Sort by composite rank: severity × log1p(volume) × diversity
    scored.sort(
        key=lambda a: a.severity_score * math.log1p(a.evidence_volume) * a.source_diversity,
        reverse=True,
    )

    logger.info(
        f"[scorer] Scored {len(scored)} opportunity areas. "
        f"Top area: '{scored[0].name}' (score≈"
        f"{scored[0].severity_score * math.log1p(scored[0].evidence_volume) * scored[0].source_diversity:.2f})"
        if scored else "[scorer] No areas to score."
    )
    return scored


def _pick_top_quotes(
    items: List[TaggedItem],
    raw_texts: Dict[str, str],
    raw_urls: Dict[str, str],
    n: int = 3,
) -> List[Quote]:
    """
    Select the top n quotes from an area's items, ranked by upvotes.
    Only items whose text is present in raw_texts are eligible.
    Falls back to any available text if upvote data is sparse.
    """
    eligible = [i for i in items if i.raw_item_id in raw_texts]
    eligible.sort(key=lambda i: i.upvotes, reverse=True)
    selected = eligible[:n]

    quotes: List[Quote] = []
    for item in selected:
        text = raw_texts.get(item.raw_item_id, "")
        # Truncate quote to 300 chars for readability
        quote_text = text[:300].rstrip() + ("…" if len(text) > 300 else "")
        quotes.append(
            Quote(
                raw_item_id=item.raw_item_id,
                text=quote_text,
                source_platform=item.source_platform.value,
                upvotes=item.upvotes,
                url=raw_urls.get(item.raw_item_id),
            )
        )
    return quotes


# ── 4.5 Opportunity Comparison Table ─────────────────────────────────────────

def build_comparison_table(areas: List[OpportunityArea]) -> List[dict]:
    """
    Build a flat list of dicts (one per area) for CSV export and dashboard
    display. Includes the composite rank score.

    The returned list is already sorted by rank (desc).
    """
    rows = []
    for rank, area in enumerate(areas, 1):
        composite = area.severity_score * math.log1p(area.evidence_volume) * area.source_diversity
        rows.append(
            {
                "rank":               rank,
                "name":               area.name,
                "evidence_volume":    area.evidence_volume,
                "source_diversity":   area.source_diversity,
                "severity_score":     area.severity_score,
                "abandonment_rate":   area.abandonment_rate,
                "voiced_count":       area.voiced_count,
                "inferred_count":     area.inferred_count,
                "composite_score":    round(composite, 3),
                "platform_breakdown": json.dumps(area.platform_breakdown),
                "cluster_source":     area.cluster_source,
                "description":        area.description,
            }
        )
    return rows


# ── 4.6 Contradiction & Open-Question Flagging ───────────────────────────────

def flag_open_questions(areas: List[OpportunityArea]) -> List[dict]:
    """
    Identify areas that warrant further investigation. Returns a list of
    {area_name, flag_type, message} dicts for the synthesis report.

    Flag types:
      hidden_need          : inferred > voiced (latent, unexpressed need)
      platform_noise       : high volume but only 1 platform (potential bias)
      critical_break       : abandonment_rate > 0.30
      low_evidence         : evidence_volume < 10 (weak signal, needs more data)
      needs_quote_curation : no verbatim quotes available
    """
    flags: List[dict] = []

    # Sort by evidence_volume to reason about "high volume"
    volume_sorted = sorted(areas, key=lambda a: a.evidence_volume, reverse=True)
    high_volume_threshold = (
        volume_sorted[0].evidence_volume * 0.3  # top 30% of max
        if volume_sorted else 0
    )

    for area in areas:
        # 1. Hidden latent need
        if area.inferred_count > area.voiced_count and area.evidence_volume >= 5:
            flags.append(
                {
                    "area_name": area.name,
                    "flag_type": "hidden_need",
                    "message": (
                        f"Inferred patterns ({area.inferred_count}) outnumber voiced "
                        f"complaints ({area.voiced_count}). Users feel this pain but "
                        "don't explicitly demand a fix — high strategic value."
                    ),
                }
            )

        # 2. Platform-specific noise
        if area.evidence_volume >= high_volume_threshold and area.source_diversity == 1:
            flags.append(
                {
                    "area_name": area.name,
                    "flag_type": "platform_noise",
                    "message": (
                        f"High volume ({area.evidence_volume} items) from only 1 platform. "
                        "Cross-validate with other sources before treating as universal."
                    ),
                }
            )

        # 3. Critical experience break
        if area.abandonment_rate > 0.30:
            pct = f"{area.abandonment_rate:.0%}"
            flags.append(
                {
                    "area_name": area.name,
                    "flag_type": "critical_break",
                    "message": (
                        f"{pct} of users explicitly abandoned the app or switched "
                        "to a workaround after this failure. Highest urgency fix."
                    ),
                }
            )

        # 4. Weak signal
        if area.evidence_volume < 10:
            flags.append(
                {
                    "area_name": area.name,
                    "flag_type": "low_evidence",
                    "message": (
                        f"Only {area.evidence_volume} items. More data needed before "
                        "this area can be reliably ranked."
                    ),
                }
            )

        # 5. No quotes
        if not area.top_quotes:
            flags.append(
                {
                    "area_name": area.name,
                    "flag_type": "needs_quote_curation",
                    "message": (
                        "No verbatim quotes available. Manually add representative "
                        "quotes before presenting to stakeholders."
                    ),
                }
            )

    logger.info(f"[scorer] Flagged {len(flags)} open questions across all areas")
    return flags

