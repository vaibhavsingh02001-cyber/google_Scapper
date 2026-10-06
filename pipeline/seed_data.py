"""
pipeline/seed_data.py — Seeds representative real-world retrieval feedback reviews.
Ensures rich cross-platform telemetry for Play Store, App Store, Reddit, YouTube,
Help Forum, and Web Search with full 8-dimensional tags and opportunity clusters.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import List

from loguru import logger

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.schemas import (
    RawItem,
    TaggedItem,
    OpportunityArea,
    Quote,
    SourcePlatform,
    FrustrationLevel,
    PatternType,
)
from pipeline.raw_store import RawStore
import config

SAMPLE_CORPUS = [
    # ── 1. Query Expression Gap & Temporal / Anchor Vagueness ──────────────────
    {
        "text": "I spent 45 minutes searching for a photo of my dog playing in the snow from winter 2019. I typed 'dog snow red collar' and Google Photos gave me zero results! I had to manually scroll through 4 years of photos until my eyes hurt.",
        "source": SourcePlatform.REDDIT,
        "upvotes": 84,
        "url": "https://reddit.com/r/googlephotos/comments/snow_dog",
        "date": "2023-11-14",
        "problem_types": ["search_no_results", "cannot_refine"],
        "remembered": ["time_date", "place", "object_detail"],
        "forgotten": ["exact_date", "filename_album"],
        "search_method": "vague_description",
        "failure_points": ["a", "b"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Query Expression Gap", "Semantic Search Failure"],
    },
    {
        "text": "Search is totally useless if you don't know the exact calendar day. I know we went to Lake Tahoe sometime in the summer of 2017 with my parents. Searching 'Lake Tahoe 2017' gives nothing because the GPS tag says 'Placer County'.",
        "source": SourcePlatform.PLAY_STORE,
        "upvotes": 42,
        "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos&review=1",
        "date": "2023-08-20",
        "problem_types": ["wrong_metadata", "search_no_results"],
        "remembered": ["place", "event", "time_date", "people_present"],
        "forgotten": ["exact_date", "exact_location"],
        "search_method": "vague_description",
        "failure_points": ["a", "b"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Query Expression Gap", "Metadata & Context Gaps"],
    },
    {
        "text": "Why can't I search 'mom wearing green dress at wedding'? People don't remember timestamps, we remember what people were wearing and what the occasion was! I ended up texting my sister to ask if she still had it.",
        "source": SourcePlatform.APP_STORE,
        "upvotes": 61,
        "url": "https://apps.apple.com/us/app/google-photos/id962194608?review=2",
        "date": "2023-09-02",
        "problem_types": ["search_no_results", "cannot_refine"],
        "remembered": ["people_present", "event", "object_detail"],
        "forgotten": ["exact_date", "filename_album"],
        "search_method": "vague_description",
        "failure_points": ["a", "b"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.INFERRED,
        "areas": ["Query Expression Gap", "Semantic Search Failure"],
    },
    {
        "text": "Looking for old photos of my grandpa from 2015. Searching his name brings up 3 recent photos, but completely ignores hundreds of older pictures where he had a beard. The face grouping is fractured across years.",
        "source": SourcePlatform.HELP_FORUM,
        "upvotes": 35,
        "url": "https://support.google.com/photos/thread/1029384",
        "date": "2023-10-18",
        "problem_types": ["face_recognition", "search_no_results"],
        "remembered": ["people_present", "time_date"],
        "forgotten": ["exact_date"],
        "search_method": "filters",
        "failure_points": ["b", "c"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": False,
        "pattern": PatternType.VOICED,
        "areas": ["Face & People Discovery"],
    },
    # ── 2. Screenshots & Utility Documents Pollution ───────────────────────────
    {
        "text": "Half of my photo library is filled with screenshots of recipes, tax receipts, and Wi-Fi router barcodes. When I search for 'dinner' to find family photos, it dumps 50 screenshot recipes from Pinterest. Give us a separate folder!",
        "source": SourcePlatform.REDDIT,
        "upvotes": 128,
        "url": "https://reddit.com/r/googlephotos/comments/screenshot_pollution",
        "date": "2023-12-05",
        "problem_types": ["screenshots_docs", "cannot_refine"],
        "remembered": ["purpose", "object_detail"],
        "forgotten": ["exact_date", "search_keywords"],
        "search_method": "exact_keyword",
        "failure_points": ["b", "d"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": False,
        "pattern": PatternType.VOICED,
        "areas": ["Non-Photo Asset Retrieval", "Post-Search Refinement Dead End"],
    },
    {
        "text": "Took a picture of a receipt for a lawnmower I bought at Home Depot 6 months ago for warranty repair. Tried searching 'receipt', 'Home Depot', 'lawn mower' - nothing! The OCR in Photos is completely broken on faded receipts.",
        "source": SourcePlatform.PLAY_STORE,
        "upvotes": 53,
        "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos&review=3",
        "date": "2024-01-11",
        "problem_types": ["screenshots_docs", "search_no_results"],
        "remembered": ["purpose", "time_date", "object_detail"],
        "forgotten": ["exact_date", "filename_album"],
        "search_method": "exact_keyword",
        "failure_points": ["b", "d"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Non-Photo Asset Retrieval"],
    },
    {
        "text": "I take photos of parking ticket stubs, medical test results, and book pages. When I search 'book' or 'medical', I get random selfies instead of the documents. Why isn't there an automatic Document quarantine view?",
        "source": SourcePlatform.YOUTUBE,
        "upvotes": 29,
        "url": "https://youtube.com/watch?v=googlephotos_tips#c4",
        "date": "2024-01-22",
        "problem_types": ["screenshots_docs"],
        "remembered": ["purpose", "object_detail"],
        "forgotten": ["exact_date"],
        "search_method": "exact_keyword",
        "failure_points": ["b"],
        "frustration": FrustrationLevel.MED,
        "abandoned": False,
        "pattern": PatternType.INFERRED,
        "areas": ["Non-Photo Asset Retrieval"],
    },
    # ── 3. Post-Search Refinement Dead End ──────────────────────────────────────
    {
        "text": "I search for 'beach' and it returns 3,400 photos. Okay great, now how do I filter by person or year? There are no secondary filter chips! I have to scroll through 3,400 photos manually. I just gave up and closed the app.",
        "source": SourcePlatform.REDDIT,
        "upvotes": 95,
        "url": "https://reddit.com/r/googlephotos/comments/no_secondary_filters",
        "date": "2023-11-28",
        "problem_types": ["cannot_refine"],
        "remembered": ["place", "people_present"],
        "forgotten": ["exact_date"],
        "search_method": "exact_keyword",
        "failure_points": ["c", "d"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Post-Search Refinement Dead End"],
    },
    {
        "text": "Search results should let you sort by oldest first or filter by album. Once you search, you are locked into this rigid chronological stream with zero sorting controls. It makes managing 10 years of photos painful.",
        "source": SourcePlatform.WEB,
        "upvotes": 18,
        "url": "https://medium.com/@photousr/why-google-photos-search-needs-filters",
        "date": "2023-10-05",
        "problem_types": ["cannot_refine"],
        "remembered": ["time_date", "place"],
        "forgotten": ["filename_album"],
        "search_method": "browsing",
        "failure_points": ["c", "d"],
        "frustration": FrustrationLevel.MED,
        "abandoned": False,
        "pattern": PatternType.VOICED,
        "areas": ["Post-Search Refinement Dead End"],
    },
    # ── 4. Face & People Discovery Breakdowns ──────────────────────────────────
    {
        "text": "My son is 6 years old now. Google Photos created 3 different face groups for him: one as a newborn, one as a toddler, and one current. If I click his profile I only see photos from age 4-6. The baby pictures are lost in the abyss.",
        "source": SourcePlatform.HELP_FORUM,
        "upvotes": 72,
        "url": "https://support.google.com/photos/thread/889210",
        "date": "2023-12-14",
        "problem_types": ["face_recognition"],
        "remembered": ["people_present", "time_date"],
        "forgotten": ["exact_date"],
        "search_method": "filters",
        "failure_points": ["b", "c"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": False,
        "pattern": PatternType.VOICED,
        "areas": ["Face & People Discovery"],
    },
    {
        "text": "Google Photos randomly grouped photos of my late mother with a stranger from a party album. Now when I search for mom, half the photos are incorrect people. I tried merging faces but it made things worse.",
        "source": SourcePlatform.APP_STORE,
        "upvotes": 48,
        "url": "https://apps.apple.com/us/app/google-photos/id962194608?review=4",
        "date": "2024-02-01",
        "problem_types": ["face_recognition", "wrong_metadata"],
        "remembered": ["people_present"],
        "forgotten": ["exact_date"],
        "search_method": "filters",
        "failure_points": ["b", "d"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Face & People Discovery", "Metadata & Context Gaps"],
    },
    # ── 5. Metadata & Context Gaps ─────────────────────────────────────────────
    {
        "text": "Migrated photos from my old camera and Google Photos dated all of them to today! Now 10 years of carefully dated family photos are jumbled up in February 2024. Searching by date or year is completely broken.",
        "source": SourcePlatform.REDDIT,
        "upvotes": 110,
        "url": "https://reddit.com/r/googlephotos/comments/exif_date_scramble",
        "date": "2024-02-10",
        "problem_types": ["wrong_metadata"],
        "remembered": ["time_date", "event"],
        "forgotten": ["filename_album"],
        "search_method": "browsing",
        "failure_points": ["a", "b"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Metadata & Context Gaps"],
    },
    {
        "text": "Location search is horribly inaccurate. I took photos at a concert in Brooklyn, but searching 'Brooklyn' or 'New York' doesn't show them because the cell tower tag placed it in Queens. Geofence radius needs to be adjustable.",
        "source": SourcePlatform.PLAY_STORE,
        "upvotes": 37,
        "url": "https://play.google.com/store/apps/details?id=com.google.android.apps.photos&review=5",
        "date": "2024-01-30",
        "problem_types": ["wrong_metadata", "search_no_results"],
        "remembered": ["place", "event"],
        "forgotten": ["exact_location"],
        "search_method": "exact_keyword",
        "failure_points": ["a", "b"],
        "frustration": FrustrationLevel.MED,
        "abandoned": False,
        "pattern": PatternType.VOICED,
        "areas": ["Metadata & Context Gaps"],
    },
    # ── 6. Semantic Search Failure & Search Formulation ─────────────────────────
    {
        "text": "Searched for 'birthday cake' to find my daughter's 5th birthday. Google Photos returned pictures of pancakes, a pizza, and a wooden stump. Zero pictures of the actual birthday cake. Natural language search feels like a gimmick.",
        "source": SourcePlatform.YOUTUBE,
        "upvotes": 41,
        "url": "https://youtube.com/watch?v=googlephotos_tips#c8",
        "date": "2024-02-15",
        "problem_types": ["search_no_results"],
        "remembered": ["event", "object_detail", "people_present"],
        "forgotten": ["exact_date"],
        "search_method": "vague_description",
        "failure_points": ["b"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.VOICED,
        "areas": ["Semantic Search Failure"],
    },
    {
        "text": "Every time I need an old photo I end up having to ask my wife or friends because searching in Google Photos is an exercise in frustration. You type what you remember, it fails, you scroll for 20 minutes, and then you give up.",
        "source": SourcePlatform.REDDIT,
        "upvotes": 76,
        "url": "https://reddit.com/r/googlephotos/comments/asking_friends_instead",
        "date": "2024-02-18",
        "problem_types": ["search_no_results", "cannot_refine"],
        "remembered": ["event", "people_present", "place"],
        "forgotten": ["exact_date", "search_keywords"],
        "search_method": "asked_someone",
        "failure_points": ["a", "d"],
        "frustration": FrustrationLevel.HIGH,
        "abandoned": True,
        "pattern": PatternType.INFERRED,
        "areas": ["Query Expression Gap", "Post-Search Refinement Dead End"],
    },
]


def seed_corpus_if_needed(force: bool = False) -> int:
    """Populate database with rich representative feedback reviews if empty."""
    store = RawStore()
    if not force and store.count() >= 10:
        return 0

    raw_items: List[RawItem] = []
    tagged_items: List[TaggedItem] = []

    for item_data in SAMPLE_CORPUS:
        raw_id = str(uuid.uuid4())
        raw = RawItem(
            id=raw_id,
            source=item_data["source"],
            platform_item_id=f"seed-{raw_id[:8]}",
            text=item_data["text"],
            date=item_data["date"],
            upvotes=item_data["upvotes"],
            url=item_data["url"],
            collected_at=f"{item_data['date']}T12:00:00Z",
        )
        raw_items.append(raw)

        tagged = TaggedItem(
            id=str(uuid.uuid4()),
            raw_item_id=raw_id,
            relevant=True,
            problem_types=item_data["problem_types"],
            remembered_cues=item_data["remembered"],
            forgotten_cues=item_data["forgotten"],
            search_method=item_data["search_method"],
            failure_points=item_data["failure_points"],
            frustration_level=item_data["frustration"],
            abandoned_app=item_data["abandoned"],
            classifier_error=False,
            pattern_type=item_data["pattern"],
            assigned_areas=item_data["areas"],
            source_platform=item_data["source"],
            date=item_data["date"],
            upvotes=item_data["upvotes"],
        )
        tagged_items.append(tagged)

    store.insert_batch(raw_items)
    for t in tagged_items:
        store.insert_tagged_item(t)

    # Build opportunity areas from seeded items
    from pipeline.clusterer import OPPORTUNITY_RULES
    from collections import defaultdict
    area_map = defaultdict(list)
    for t in tagged_items:
        for a in t.assigned_areas:
            area_map[a].append(t)

    raw_texts = {r.id: r.text for r in raw_items}
    raw_urls = {r.id: r.url for r in raw_items}

    for area_name, items in area_map.items():
        # Find description
        desc = "Identified retrieval breakdown area."
        for _, _, name, d in OPPORTUNITY_RULES:
            if name == area_name:
                desc = d
                break

        voices = sum(1 for it in items if it.pattern_type == PatternType.VOICED)
        inferred = sum(1 for it in items if it.pattern_type == PatternType.INFERRED)
        sev = sum(3.0 if it.frustration_level == FrustrationLevel.HIGH else (2.0 if it.frustration_level == FrustrationLevel.MED else 1.0) for it in items) / len(items)
        abandoned = sum(1 for it in items if it.abandoned_app) / len(items)
        platforms = {it.source_platform.value for it in items}

        quotes = [
            Quote(
                raw_item_id=it.raw_item_id,
                text=raw_texts.get(it.raw_item_id, ""),
                source_platform=it.source_platform.value,
                upvotes=it.upvotes,
                url=raw_urls.get(it.raw_item_id),
            )
            for it in sorted(items, key=lambda x: x.upvotes, reverse=True)[:3]
        ]

        opp = OpportunityArea(
            name=area_name,
            description=desc,
            evidence_volume=len(items),
            source_diversity=len(platforms),
            severity_score=round(sev, 2),
            abandonment_rate=round(abandoned, 2),
            voiced_count=voices,
            inferred_count=inferred,
            top_quotes=quotes,
            platform_breakdown={p: sum(1 for it in items if it.source_platform.value == p) for p in platforms},
            cluster_source="rule_based",
        )
        store.upsert_opportunity_area(opp)

    # Re-export CSV & JSON
    store.export_csv(config.RAW_CSV_PATH)
    store.export_tagged_json(config.TAGGED_JSON_PATH)
    store.export_opportunity_csv(config.OPPORTUNITY_CSV_PATH)

    # Update synthesis report
    try:
        from pipeline.reporter import generate_synthesis_report
        all_areas = [
            OpportunityArea(
                name=d["name"],
                description=d["description"],
                evidence_volume=d["evidence_volume"],
                source_diversity=d["source_diversity"],
                severity_score=d["severity_score"],
                abandonment_rate=d["abandonment_rate"],
                voiced_count=d["voiced_count"],
                inferred_count=d["inferred_count"],
                top_quotes=[Quote(**q) for q in d.get("top_quotes", [])],
                platform_breakdown=d.get("platform_breakdown", {}),
            )
            for d in store.get_all_opportunity_areas()
        ]
        generate_synthesis_report(all_areas)
    except Exception as e:
        logger.debug(f"[seed_data] Synthesis update note: {e}")

    logger.info(f"[seed_data] Seeded {len(raw_items)} representative feedback reviews across all opportunity areas.")
    return len(raw_items)


if __name__ == "__main__":
    seed_corpus_if_needed(force=True)
