"""
pipeline/tagger.py — LLM Classification Layer (Phase 3).

Uses Groq's API (llama-3.1-8b-instant) to classify each RawItem
across all 8 tagging dimensions defined in the architecture:

  1. Relevance           — is this about retrieval/search?
  2. Problem Types       — multi-label categorical
  3. Remembered Cues     — what the user recalled
  4. Forgotten Cues      — what the user didn't know
  5. Search Method       — how they attempted to search
  6. Failure Points      — funnel stage(s) (a/b/c/d)
  7. Severity/Sentiment  — frustration level + app abandonment
  8. Pattern Type        — voiced vs. inferred (set during classification)

Guardrails:
  - temperature=0.0 for deterministic JSON
  - Retry up to MAX_RETRIES with exponential backoff on JSON parse errors
  - Sync/backup items are explicitly caught and tagged relevant=False
  - Grounding check: remembered_cues tags are verified against raw text
  - Items that fail all retries → classifier_error=True (never crash the batch)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from models.schemas import (
    RawItem,
    TaggedItem,
    SourcePlatform,
    FrustrationLevel,
    PatternType,
)
import config

# ── Allowed enum values (for validation after LLM response) ──────────────────
VALID_PROBLEM_TYPES = {
    "search_no_results", "screenshots_docs", "cannot_refine",
    "wrong_metadata", "face_recognition", "sync_backup", "other",
}
VALID_REMEMBERED_CUES = {
    "time_date", "place", "people_present", "event",
    "object_detail", "purpose", "emotion", "none",
}
VALID_FORGOTTEN_CUES = {
    "exact_date", "exact_location", "filename_album",
    "search_keywords", "none",
}
VALID_SEARCH_METHODS = {
    "exact_keyword", "vague_description", "browsing",
    "filters", "asked_someone", "gave_up",
}
VALID_FAILURE_POINTS = {"a", "b", "c", "d"}
VALID_FRUSTRATION    = {"low", "med", "high"}

# Lexical anchors used for grounding verification of remembered_cues
CUE_ANCHORS: Dict[str, List[str]] = {
    "time_date":      ["yesterday", "last year", "years ago", "2020", "2021", "2022",
                       "2023", "2024", "month ago", "week ago", "summer", "winter",
                       "january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november", "december",
                       "recently", "long time", "when", "old", "ago", "birthday"],
    "place":          ["beach", "park", "hotel", "trip", "vacation", "holiday",
                       "city", "town", "home", "house", "school", "office", "gym",
                       "restaurant", "at the", "in the", "went to", "visited"],
    "people_present": ["friend", "family", "wife", "husband", "daughter", "son",
                       "sister", "brother", "mom", "dad", "parent", "child",
                       "baby", "dog", "cat", "pet", "with my", "with our"],
    "event":          ["wedding", "birthday", "party", "concert", "christmas",
                       "thanksgiving", "halloween", "graduation", "anniversary",
                       "reunion", "event", "celebration", "ceremony"],
    "object_detail":  ["photo of", "picture of", "screenshot", "receipt",
                       "document", "car", "food", "ticket", "selfie", "landscape",
                       "meme", "image of"],
    "purpose":        ["took it", "saved it", "captured", "meant to", "used for",
                       "for work", "for fun", "remember", "remind", "proof"],
    "emotion":        ["funny", "hilarious", "beautiful", "amazing", "cute",
                       "happy", "sad", "laughing", "emotional", "proud",
                       "favorite", "memorable", "special"],
}

# Keywords triggering Voiced vs. Inferred classification
EXPLICIT_SEARCH_TERMS = [
    "search", "query", "filter", "results", "algorithm", "typed",
    "searched for", "can't find", "cannot find", "couldn't find",
    "doesn't work", "not working", "broken", "fails", "failed",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a UX research analyst classifying user feedback about Google Photos
search and retrieval failures. Your task is to tag the feedback below using
ONLY the schema provided. Return ONLY valid JSON — no markdown, no explanation.

SCHEMA:
{
  "relevant": bool,
  "problem_types": [],
  "remembered_cues": [],
  "forgotten_cues": [],
  "search_method": null,
  "failure_points": [],
  "frustration_level": null,
  "abandoned_app": false
}

FIELD RULES:
- relevant: true ONLY if the user cannot LOCATE a photo/video/screenshot that EXISTS
  in their library. False for: sync failures, photos deleted from cloud, pricing,
  editing bugs, backup issues, storage quota, or unrelated topics.
- problem_types (multi-label, use [] if none apply):
    search_no_results | screenshots_docs | cannot_refine |
    wrong_metadata | face_recognition | sync_backup | other
- remembered_cues (multi-label, ONLY if EXPLICITLY stated in text):
    time_date | place | people_present | event | object_detail |
    purpose | emotion | none
  ⚠ DO NOT infer cues. Only tag what the user explicitly mentions.
- forgotten_cues (multi-label): exact_date | exact_location |
    filename_album | search_keywords | none
- search_method (single value or null):
    exact_keyword | vague_description | browsing | filters |
    asked_someone | gave_up
- failure_points (multi-label, map to retrieval funnel):
    "a" = could not express what they remembered as a query
    "b" = search did not understand / match their clues
    "c" = results appeared but user could not identify the right one
    "d" = search failed with no way to refine
- frustration_level: low | med | high | null
- abandoned_app: true if user explicitly says they gave up, uninstalled,
    or switched to another app/service

DISAMBIGUATION:
- If photos are MISSING (deleted, not synced, storage issue): relevant=false,
  problem_types=["sync_backup"]
- If feedback is about editing tools, sharing, upload, or unrelated bugs:
  relevant=false, problem_types=["other"]
- Sarcasm like "great, search works SO well" → frustration_level=high
"""

USER_PROMPT_TEMPLATE = '''\
Classify this feedback:
"""
{text}
"""
'''


class LLMTagger:
    """
    Classifies RawItems using the Groq API (llama-3.1-8b-instant).
    Writes results to the tagged_items SQLite table via RawStore.
    """

    def __init__(self) -> None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        try:
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY)
        except ImportError:
            raise ImportError("groq not installed. Run: pip install groq")

        logger.debug(
            f"[tagger] Groq client initialised "
            f"(model={config.CLASSIFIER_MODEL})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify_batch(
        self,
        items: List[RawItem],
        store,   # RawStore instance — imported inside to avoid circular
    ) -> Dict[str, int]:
        """
        Classify a list of RawItems, writing each result to tagged_items.

        Args:
            items: List of RawItem objects to classify.
            store: RawStore instance for writing tagged results.

        Returns:
            Summary dict: {total, relevant, irrelevant, errors}.
        """
        summary = {"total": 0, "relevant": 0, "irrelevant": 0, "errors": 0}
        batch_size = config.BATCH_SIZE

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]

            for item in batch:
                tagged = self._classify_with_fallback(item)
                store.insert_tagged_item(tagged)
                summary["total"] += 1
                if tagged.classifier_error:
                    summary["errors"] += 1
                elif tagged.relevant:
                    summary["relevant"] += 1
                else:
                    summary["irrelevant"] += 1

            # 1-second inter-batch delay (Groq free tier: ~30 req/min)
            if batch_start + batch_size < len(items):
                time.sleep(1.0)

            logger.debug(
                f"[tagger] Batch {batch_start // batch_size + 1} done — "
                f"{summary['relevant']} relevant so far"
            )

        return summary

    def classify_single(self, raw_text: str) -> dict:
        """
        Classify a single text string and return the raw tag dict.
        Used by the evaluation harness (run_eval.py).
        """
        raw_tags = self._call_groq(raw_text)
        return raw_tags

    # ── Private: classification flow ──────────────────────────────────────────

    def _classify_with_fallback(self, item: RawItem) -> TaggedItem:
        """
        Classify one item. On total failure → return a classifier_error TaggedItem.
        """
        try:
            raw_tags = self._call_groq_with_retry(item.text)
            tagged   = self._build_tagged_item(item, raw_tags)
            tagged   = self._apply_grounding_check(item.text, tagged)
            tagged   = self._set_pattern_type(item.text, tagged)
            return tagged
        except Exception as exc:
            logger.warning(
                f"[tagger] classify_with_fallback failed for {item.id}: {exc}"
            )
            return self._error_tagged_item(item)

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1.5, min=2, max=20),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError, KeyError)),
        reraise=True,
    )
    def _call_groq_with_retry(self, text: str) -> dict:
        """Call Groq with retries on JSON parse failures."""
        return self._call_groq(text)

    def _call_groq(self, text: str) -> dict:
        """
        Single Groq API call. Parses JSON from the response.
        Raises ValueError if the response cannot be parsed.
        """
        # Use primary model; fall back to larger model on second-to-last retry
        response = self._client.chat.completions.create(
            model=config.CLASSIFIER_MODEL,
            temperature=config.CLASSIFIER_TEMPERATURE,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text[:2000])},
            ],
        )

        raw_content = response.choices[0].message.content or ""

        # Extract JSON even if the model wraps it in markdown fences
        json_str = self._extract_json(raw_content)
        tags = json.loads(json_str)

        # Basic type validation
        if not isinstance(tags.get("relevant"), bool):
            raise ValueError(f"'relevant' field missing or not bool: {tags}")

        return tags

    # ── Private: post-processing ──────────────────────────────────────────────

    def _build_tagged_item(self, item: RawItem, tags: dict) -> TaggedItem:
        """Convert raw tag dict + RawItem into a validated TaggedItem."""
        # Sanitise multi-label fields against allowed sets
        problem_types    = self._filter_list(tags.get("problem_types", []),    VALID_PROBLEM_TYPES)
        remembered_cues  = self._filter_list(tags.get("remembered_cues", []),  VALID_REMEMBERED_CUES)
        forgotten_cues   = self._filter_list(tags.get("forgotten_cues", []),   VALID_FORGOTTEN_CUES)
        failure_points   = self._filter_list(tags.get("failure_points", []),   VALID_FAILURE_POINTS)

        search_method = tags.get("search_method")
        if search_method not in VALID_SEARCH_METHODS:
            search_method = None

        frustration_raw = tags.get("frustration_level")
        frustration = (
            FrustrationLevel(frustration_raw)
            if frustration_raw in VALID_FRUSTRATION
            else None
        )

        abandoned = tags.get("abandoned_app")
        if not isinstance(abandoned, bool):
            abandoned = None

        return TaggedItem(
            id=str(uuid.uuid4()),
            raw_item_id=item.id,
            relevant=bool(tags.get("relevant", False)),
            problem_types=problem_types,
            remembered_cues=remembered_cues,
            forgotten_cues=forgotten_cues,
            search_method=search_method,
            failure_points=failure_points,
            frustration_level=frustration,
            abandoned_app=abandoned,
            classifier_error=False,
            source_platform=item.source,
            date=item.date,
            upvotes=item.upvotes,
        )

    def _apply_grounding_check(self, text: str, tagged: TaggedItem) -> TaggedItem:
        """
        Grounding guardrail: strip remembered_cues tags that have no
        lexical anchor in the raw text (prevents hallucinated cues).
        """
        text_lower = text.lower()
        verified   = []
        for cue in tagged.remembered_cues:
            anchors = CUE_ANCHORS.get(cue, [])
            if not anchors:
                # Cues without anchors (e.g. "none") pass through
                verified.append(cue)
            elif any(anchor in text_lower for anchor in anchors):
                verified.append(cue)
            else:
                logger.debug(
                    f"[tagger] Grounding stripped cue '{cue}' "
                    f"(no anchor found in text)"
                )
        return tagged.model_copy(update={"remembered_cues": verified})

    def _set_pattern_type(self, text: str, tagged: TaggedItem) -> TaggedItem:
        """
        Classify pattern as VOICED or INFERRED:
        - VOICED:   user explicitly uses search/failure vocabulary
        - INFERRED: user describes rich memory cues without explicit search language
        """
        if not tagged.relevant:
            return tagged

        text_lower = text.lower()
        has_explicit = any(term in text_lower for term in EXPLICIT_SEARCH_TERMS)

        inferred_cues = {"emotion", "purpose", "event", "people_present"}
        has_rich_cues = any(c in inferred_cues for c in tagged.remembered_cues)

        pattern = (
            PatternType.INFERRED
            if (has_rich_cues and not has_explicit)
            else PatternType.VOICED
        )
        return tagged.model_copy(update={"pattern_type": pattern})

    # ── Private: utilities ────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract the first JSON object from a string, even if the model
        wrapped it in markdown fences or prefixed it with text.
        """
        # Try to find a {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

    @staticmethod
    def _filter_list(values: Any, valid_set: set) -> List[str]:
        """Keep only values that are in the allowed set."""
        if not isinstance(values, list):
            return []
        return [v for v in values if isinstance(v, str) and v in valid_set]

    @staticmethod
    def _error_tagged_item(item: RawItem) -> TaggedItem:
        """Return a safe fallback TaggedItem when classification fails entirely."""
        return TaggedItem(
            id=str(uuid.uuid4()),
            raw_item_id=item.id,
            relevant=False,
            problem_types=[],
            remembered_cues=[],
            forgotten_cues=[],
            search_method=None,
            failure_points=[],
            frustration_level=None,
            abandoned_app=None,
            classifier_error=True,
            source_platform=item.source,
            date=item.date,
            upvotes=item.upvotes,
        )

