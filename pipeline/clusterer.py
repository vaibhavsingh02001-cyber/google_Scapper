"""
pipeline/clusterer.py — Clustering Layer (Phase 4).

Two-pass clustering strategy:
  Pass 1 — Rule-Based: map (problem_types × failure_points) combinations to
            predefined named opportunity areas. Items may be assigned to
            multiple areas (multi-label).

  Pass 2 — Semantic (HDBSCAN): cluster raw embeddings from Chroma using
            HDBSCAN, then use Groq to generate a name + description for
            each discovered semantic cluster. Merged with rule-based areas;
            overlapping items tracked to avoid double-counting.

  Items that match no rule → assigned to "Emerging / Uncategorized".
  Voiced vs. Inferred pattern already set by tagger (Phase 3); this module
  reads those values and aggregates them per area.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from models.schemas import TaggedItem, OpportunityArea, PatternType
import config

# ── 4.1 Rule-Based Opportunity Map ───────────────────────────────────────────
# Keys: (problem_type_match, failure_point_match)
#   - problem_type_match: exact problem_type value OR "*" (any)
#   - failure_point_match: single failure point OR comma-separated OR "*"
# An item matches a rule if it has the problem_type AND any of the
# failure_points listed in the key.

OPPORTUNITY_RULES: List[Tuple[str, str, str, str]] = [
    # (problem_type, failure_points, area_name, area_description)
    (
        "*", "a",
        "Query Expression Gap",
        "Users cannot translate what they remember into a query the search box "
        "can understand. They have rich mental cues (time, place, people) but "
        "lack the vocabulary or interface affordances to express them.",
    ),
    (
        "search_no_results", "b",
        "Semantic Search Failure",
        "Users form a reasonable natural-language query but Google Photos returns "
        "empty or irrelevant results. The semantic model fails to bridge user intent "
        "with indexed content.",
    ),
    (
        "screenshots_docs", "b,d",
        "Non-Photo Asset Retrieval",
        "Users struggle to find screenshots, receipts, PDFs, and documents saved to "
        "Google Photos. Search treats all items as photographs, providing no "
        "type-aware filtering.",
    ),
    (
        "cannot_refine", "d",
        "Post-Search Refinement Dead End",
        "After an initial search, users have no practical way to narrow results. "
        "Filters are absent, opaque, or ineffective, forcing manual scrolling through "
        "large result sets.",
    ),
    (
        "wrong_metadata", "a,b",
        "Metadata & Context Gaps",
        "Location tags, date stamps, or auto-generated labels are wrong or missing, "
        "causing retrieval to fail even when the user provides correct cues.",
    ),
    (
        "face_recognition", "b,c",
        "Face & People Discovery",
        "Users cannot reliably find photos by person. Face grouping is incomplete, "
        "inconsistent across albums, or fails after app updates.",
    ),
]

UNCATEGORIZED_AREA = "Emerging / Uncategorized"
UNCATEGORIZED_DESC = (
    "Items that do not match any predefined rule-based opportunity area. "
    "These may represent novel or niche retrieval problems worth monitoring "
    "as the corpus grows."
)


# ── 4.1 Rule-Based Clustering ─────────────────────────────────────────────────

def rule_based_cluster(tagged_items: List[TaggedItem]) -> Dict[str, List[TaggedItem]]:
    """
    Assign each relevant TaggedItem to one or more named opportunity areas
    based on its problem_types and failure_points.

    Returns:
        Dict mapping area_name → list of assigned TaggedItems.
        Items may appear in multiple area lists (multi-label).
    """
    area_items: Dict[str, List[TaggedItem]] = defaultdict(list)
    uncategorized: List[TaggedItem] = []

    for item in tagged_items:
        matched_any = False

        for prob_match, fp_match, area_name, _ in OPPORTUNITY_RULES:
            # Check problem_type match
            if prob_match == "*":
                type_ok = True
            else:
                type_ok = prob_match in item.problem_types

            if not type_ok:
                continue

            # Check failure_point match
            if fp_match == "*":
                fp_ok = True
            else:
                rule_fps = {fp.strip() for fp in fp_match.split(",")}
                fp_ok = bool(rule_fps & set(item.failure_points))

            if type_ok and fp_ok:
                area_items[area_name].append(item)
                matched_any = True

        if not matched_any:
            uncategorized.append(item)

    if uncategorized:
        area_items[UNCATEGORIZED_AREA] = uncategorized

    logger.info(
        f"[clusterer] Rule-based: {len(area_items)} areas, "
        + ", ".join(f"{k}={len(v)}" for k, v in area_items.items())
    )
    return dict(area_items)


# ── 4.2 Semantic Clustering (HDBSCAN) ────────────────────────────────────────

def semantic_cluster(
    embedder,            # Embedder instance
    tagged_items: List[TaggedItem],
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> Dict[str, List[TaggedItem]]:
    """
    Cluster raw embeddings with HDBSCAN. For each cluster, call Groq to
    generate a human-readable name and description.

    Returns:
        Dict mapping auto-generated area_name → list of TaggedItems.
        Returns empty dict if Chroma is empty or HDBSCAN finds no clusters.
    """
    if embedder.count() == 0:
        logger.warning("[clusterer] Chroma is empty — skipping semantic clustering.")
        return {}

    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed. Run: pip install hdbscan")
        return {}

    # Fetch all embeddings from Chroma
    ids, embeddings, metadatas = embedder.get_all_embeddings()
    if not embeddings:
        logger.warning("[clusterer] No embeddings retrieved from Chroma.")
        return {}

    logger.info(f"[clusterer] Running HDBSCAN on {len(embeddings)} embeddings...")

    matrix = np.array(embeddings, dtype=np.float32)
    clusterer_hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer_hdb.fit_predict(matrix)

    unique_labels = set(labels) - {-1}  # -1 = noise
    logger.info(
        f"[clusterer] HDBSCAN found {len(unique_labels)} clusters, "
        f"{(labels == -1).sum()} noise points"
    )

    if not unique_labels:
        return {}

    # Build {label_id → list of chroma_ids}
    label_to_ids: Dict[int, List[str]] = defaultdict(list)
    for chroma_id, label in zip(ids, labels):
        if label != -1:
            label_to_ids[int(label)].append(chroma_id)

    # Build {raw_item_id → TaggedItem} lookup
    id_to_item = {item.raw_item_id: item for item in tagged_items}

    # For each cluster: get sample texts → call Groq for name+description
    semantic_areas: Dict[str, List[TaggedItem]] = {}

    for label_id, chroma_ids in label_to_ids.items():
        cluster_items = [
            id_to_item[cid]
            for cid in chroma_ids
            if cid in id_to_item
        ]
        if not cluster_items:
            continue

        # Pick up to 5 representative texts for the label prompt
        sample_texts = [
            id_to_item[cid].raw_item_id  # placeholder — we use item text below
            for cid in chroma_ids[:5]
            if cid in id_to_item
        ]
        sample_texts = [item.raw_item_id for item in cluster_items[:5]]

        area_name, area_desc = _label_cluster_with_groq(
            cluster_items[:5], label_id
        )
        semantic_areas[area_name] = cluster_items

    logger.info(
        f"[clusterer] Semantic: {len(semantic_areas)} labelled clusters"
    )
    return semantic_areas


def _label_cluster_with_groq(
    sample_items: List[TaggedItem],
    cluster_id: int,
) -> Tuple[str, str]:
    """
    Ask Groq to generate a short name and description for a semantic cluster,
    given 5 representative tagged items.
    Falls back to a numbered name if the LLM call fails.
    """
    if not config.GROQ_API_KEY:
        return f"Semantic Cluster {cluster_id}", "Auto-discovered cluster."

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        # Build a mini-summary of the cluster's tags
        item_summaries = []
        for i, item in enumerate(sample_items, 1):
            item_summaries.append(
                f"{i}. problem_types={item.problem_types} "
                f"failure_points={item.failure_points} "
                f"frustration={item.frustration_level}"
            )

        prompt = (
            "You are labelling a cluster of Google Photos user feedback items. "
            "The cluster has these tagged dimensions:\n\n"
            + "\n".join(item_summaries)
            + "\n\nReturn ONLY JSON with exactly two keys:\n"
            '{"name": "<5-7 word opportunity name>", '
            '"description": "<1-2 sentence description of the retrieval problem>"}'
        )

        response = client.chat.completions.create(
            model=config.CLASSIFIER_MODEL,
            temperature=0.2,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            name = data.get("name", f"Semantic Cluster {cluster_id}").strip()
            desc = data.get("description", "Auto-discovered cluster.").strip()
            return name, desc
    except Exception as exc:
        logger.warning(f"[clusterer] Groq label call failed for cluster {cluster_id}: {exc}")

    return f"Semantic Cluster {cluster_id}", "Auto-discovered cluster via HDBSCAN."


# ── 4.3 Voiced vs. Inferred Separator ────────────────────────────────────────

def classify_pattern_type(item: TaggedItem, raw_text: str = "") -> str:
    """
    VOICED:   user directly complains about search failing.
    INFERRED: user describes memory cues (emotion/purpose/event)
              without explicitly using search/failure vocabulary.

    Note: This is already computed by the tagger (Phase 3).
    This function provides a standalone recalculation for audit/override.
    """
    inferred_cues = {"emotion", "purpose", "event", "people_present"}
    explicit_terms = [
        "search", "query", "filter", "results", "can't find",
        "cannot find", "couldn't find", "not working",
    ]
    has_rich_cues  = any(c in inferred_cues for c in item.remembered_cues)
    has_explicit   = any(t in raw_text.lower() for t in explicit_terms)

    if has_rich_cues and not has_explicit:
        return "inferred"
    return "voiced"


# ── 4.5 Merge Rule-Based + Semantic Areas ────────────────────────────────────

def merge_clusters(
    rule_areas: Dict[str, List[TaggedItem]],
    semantic_areas: Dict[str, List[TaggedItem]],
) -> Dict[str, List[TaggedItem]]:
    """
    Merge rule-based and semantic cluster dictionaries.
    If a semantic cluster name duplicates a rule area name, items are
    merged under the rule area name (rule areas take precedence).

    Returns the merged dict {area_name → items}.
    """
    merged: Dict[str, List[TaggedItem]] = dict(rule_areas)

    for name, items in semantic_areas.items():
        if name in merged:
            # Merge items — deduplicate by TaggedItem.id
            existing_ids = {i.id for i in merged[name]}
            new_items = [i for i in items if i.id not in existing_ids]
            merged[name].extend(new_items)
        else:
            merged[name] = items

    return merged


def update_assigned_areas(
    area_assignments: Dict[str, List[TaggedItem]],
) -> List[TaggedItem]:
    """
    Write back assigned_areas onto each TaggedItem (for DB persistence).
    Returns a flat de-duplicated list of TaggedItems with updated assigned_areas.
    """
    # Build {item_id → set of area names}
    item_areas: Dict[str, set] = defaultdict(set)
    item_lookup: Dict[str, TaggedItem] = {}

    for area_name, items in area_assignments.items():
        for item in items:
            item_areas[item.id].add(area_name)
            item_lookup[item.id] = item

    updated: List[TaggedItem] = []
    for item_id, area_set in item_areas.items():
        base_item = item_lookup[item_id]
        updated.append(
            base_item.model_copy(
                update={"assigned_areas": sorted(area_set)}
            )
        )
    return updated

