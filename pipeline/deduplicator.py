"""
pipeline/deduplicator.py — Deduplication utilities.
Level 1: exact SHA-256 hash deduplication (source + text[:200]).
Level 2: near-duplicate detection via Jaccard token similarity.
"""
from __future__ import annotations

import hashlib
from typing import List, Set, Tuple

from loguru import logger
from models.schemas import RawItem


def _item_hash(item: RawItem) -> str:
    """Compute a stable hash key for exact deduplication."""
    key = f"{item.source.value}::{item.text[:200].lower().strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _token_set(text: str) -> Set[str]:
    """Simple whitespace tokenizer for Jaccard similarity."""
    return set(text.lower().split())


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate(
    new_items: List[RawItem],
    existing_hashes: Set[str],
    near_dup_threshold: float = 0.85,
) -> Tuple[List[RawItem], Set[str]]:
    """
    Remove duplicates from `new_items` relative to `existing_hashes`.

    Level 1 — Exact: SHA-256 hash (source + text[:200])
    Level 2 — Near:  Jaccard similarity on token sets > threshold

    Returns:
        unique_items   — items that passed both dedup checks
        updated_hashes — existing_hashes union new item hashes
    """
    unique: List[RawItem]     = []
    seen_hashes               = set(existing_hashes)
    seen_token_sets: List[Set[str]] = []

    exact_dups   = 0
    near_dups    = 0

    for item in new_items:
        h = _item_hash(item)

        # Level 1: exact hash check
        if h in seen_hashes:
            exact_dups += 1
            continue

        # Level 2: near-duplicate check against items accepted in this batch
        tset = _token_set(item.text)
        is_near_dup = any(
            _jaccard(tset, existing) >= near_dup_threshold
            for existing in seen_token_sets
        )
        if is_near_dup:
            near_dups += 1
            continue

        seen_hashes.add(h)
        seen_token_sets.append(tset)
        unique.append(item)

    if exact_dups or near_dups:
        logger.debug(
            f"[dedup] Removed {exact_dups} exact + {near_dups} near-duplicates "
            f"({len(unique)} unique items kept)"
        )

    return unique, seen_hashes
