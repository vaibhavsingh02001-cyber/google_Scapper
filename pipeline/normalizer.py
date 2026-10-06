"""
pipeline/normalizer.py — Text normalisation utilities.
Strips HTML, normalises whitespace, truncates intelligently,
and applies the retrieval-keyword window extraction for long texts.
"""
from __future__ import annotations

import re
import html
from typing import List

from models.schemas import RawItem
import config

# Keywords used to find the most retrieval-relevant sentence window in long texts
RETRIEVAL_KEYWORDS = [
    "search", "find", "look for", "lost", "remember", "where is",
    "can't locate", "missing", "retrieve", "results", "query",
    "can't find", "cannot find", "couldn't find",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(raw_text: str) -> str:
    """
    Clean and normalise a raw text string:
    1. Unescape HTML entities (&amp; → &, etc.)
    2. Strip HTML tags
    3. Collapse whitespace
    4. Truncate to MAX_TEXT_LENGTH using retrieval-keyword window
    """
    if not raw_text:
        return ""

    # 1. Unescape HTML entities
    text = html.unescape(raw_text)

    # 2. Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # 4. Truncate intelligently
    if len(text) > config.MAX_TEXT_LENGTH:
        text = _smart_truncate(text, config.MAX_TEXT_LENGTH)

    return text


def _smart_truncate(text: str, max_len: int) -> str:
    """
    For long texts, extract a window around retrieval-related keywords
    rather than blindly taking the first max_len characters.
    """
    text_lower = text.lower()

    # Find the earliest retrieval keyword position
    best_pos = len(text)
    for kw in RETRIEVAL_KEYWORDS:
        pos = text_lower.find(kw)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(text):
        # No keyword found — take the first max_len chars
        return text[:max_len]

    # Centre the window on the keyword, taking max_len characters
    start = max(0, best_pos - (max_len // 3))
    end   = min(len(text), start + max_len)
    return text[start:end].strip()


def normalize_items(items: List[RawItem]) -> List[RawItem]:
    """Apply text normalization to a list of RawItem objects in-place."""
    normalized = []
    for item in items:
        clean = normalize_text(item.text)
        if not clean or len(clean) < config.MIN_TEXT_LENGTH:
            continue
        # Return new object with clean text (Pydantic models are immutable by default)
        normalized.append(item.model_copy(update={"text": clean}))
    return normalized
