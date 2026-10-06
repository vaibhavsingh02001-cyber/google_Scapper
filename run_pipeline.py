"""
run_pipeline.py — Full pipeline CLI (Phases 3–6).

Usage:
    python run_pipeline.py                    # run all implemented steps
    python run_pipeline.py --step embed       # Phase 2: embed raw items into Chroma
    python run_pipeline.py --step tag         # Phase 3 (coming soon)
    python run_pipeline.py --step cluster     # Phase 4 (coming soon)
    python run_pipeline.py --step report      # Phase 6 (coming soon)
    python run_pipeline.py --status           # show DB + Chroma counts
"""
from __future__ import annotations

import argparse
import sys

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discovery Engine — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        choices=["collect", "embed", "tag", "cluster", "report"],
        help="Run a single pipeline step",
    )
    group.add_argument(
        "--from-step",
        choices=["collect", "embed", "tag", "cluster", "report"],
        dest="from_step",
        help="Run from this step onwards",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print DB and Chroma vector store status, then exit",
    )
    return parser.parse_args()


def run_embed() -> None:
    """Phase 2 — Embed all raw items into the Chroma vector store."""
    from pipeline.raw_store import RawStore
    from pipeline.embedder import Embedder

    store   = RawStore()
    embedder = Embedder()

    logger.info(f"[pipeline] Raw items in DB  : {store.count()}")
    logger.info(f"[pipeline] Vectors in Chroma: {embedder.count()}")

    # Load all raw items (not just untagged, so Chroma is always in sync)
    import sqlite3, config
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM raw_items").fetchall()
    conn.close()

    from models.schemas import RawItem, SourcePlatform
    items = [
        RawItem(
            id=r["id"],
            source=SourcePlatform(r["source"]),
            platform_item_id=r["platform_item_id"],
            text=r["text"],
            date=r["date"],
            upvotes=r["upvotes"] or 0,
            url=r["url"],
            collected_at=r["collected_at"] or "",
        )
        for r in rows
    ]

    stored = embedder.embed_and_store(items)
    logger.info(f"[pipeline] ✅ Phase 2 complete — {stored} new vectors stored.")
    logger.info(f"[pipeline]    Chroma total: {embedder.count()} vectors")

    # Phase 2 gate check
    info = embedder.collection_info()
    logger.info(f"[pipeline]    Collection : {info['collection']}")
    logger.info(f"[pipeline]    Distance   : {info['distance']}")
    logger.info(f"[pipeline]    Model      : {info['model']}")


def print_status() -> None:
    """Print current DB and Chroma store status."""
    from pipeline.raw_store import RawStore
    from pipeline.embedder import Embedder

    store    = RawStore()
    embedder = Embedder()

    logger.info("=" * 55)
    logger.info("  Discovery Engine — Status")
    logger.info("=" * 55)

    # SQLite raw_items summary
    summary = store.source_summary()
    logger.info("  SQLite raw_items:")
    for src, n in summary.items():
        logger.info(f"    {src:<15}: {n:>5}")
    logger.info(f"    {'TOTAL':<15}: {store.count():>5}")

    # tagged_items stats
    stats = store.get_tagging_stats()
    logger.info(f"  tagged_items:")
    logger.info(f"    {'total':<15}: {stats['total']:>5}")
    logger.info(f"    {'relevant':<15}: {stats['relevant']:>5}")
    logger.info(f"    {'irrelevant':<15}: {stats['irrelevant']:>5}")
    logger.info(f"    {'errors':<15}: {stats['errors']:>5}  ({stats['error_rate']:.1%} error rate)")

    # Chroma summary
    info = embedder.collection_info()
    logger.info(f"  Chroma vectors  : {info['vector_count']}")
    logger.info(f"  Collection      : {info['collection']}")
    logger.info("=" * 55)


def run_tag() -> None:
    """
    Phase 3 — Classify all untagged raw items with the Groq LLM tagger.
    Re-run safe: already-tagged items are skipped automatically.
    """
    import sys
    from tqdm import tqdm
    from pipeline.raw_store import RawStore
    from pipeline.tagger import LLMTagger
    import config

    store  = RawStore()
    tagger = LLMTagger()

    untagged = store.get_untagged_items()
    if not untagged:
        logger.info("[pipeline] No untagged items found — tagging already complete.")
        _print_tag_gate(store)
        return

    logger.info(f"[pipeline] Items to classify : {len(untagged)}")
    logger.info(f"[pipeline] Batch size        : {config.BATCH_SIZE}")
    estimated_batches = (len(untagged) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    logger.info(f"[pipeline] Estimated batches : {estimated_batches}")

    # Progress bar tracks individual items
    with tqdm(total=len(untagged), desc="Classifying", unit="item") as pbar:
        for batch_start in range(0, len(untagged), config.BATCH_SIZE):
            batch = untagged[batch_start : batch_start + config.BATCH_SIZE]

            for item in batch:
                tagged = tagger._classify_with_fallback(item)
                store.insert_tagged_item(tagged)
                pbar.update(1)

            # Inter-batch delay (Groq free tier ~30 req/min)
            import time
            if batch_start + config.BATCH_SIZE < len(untagged):
                time.sleep(1.0)

    # Print relevance summary
    stats = store.get_tagging_stats()
    logger.info("\n" + "=" * 55)
    logger.info("  PHASE 3 — TAGGING SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Total tagged    : {stats['total']}")
    logger.info(f"  Relevant        : {stats['relevant']}")
    logger.info(f"  Irrelevant      : {stats['irrelevant']}")
    logger.info(f"  Classifier err  : {stats['errors']}  ({stats['error_rate']:.1%})")
    logger.info("=" * 55)

    # Export tagged_items.json
    export_path = store.export_tagged_json(config.TAGGED_JSON_PATH)
    logger.info(f"  Exported: {export_path}")

    # Phase 3 gate check
    _print_tag_gate(store)


def _print_tag_gate(store) -> None:
    """Print Phase 3 done-when gate results."""
    stats = store.get_tagging_stats()

    gate_relevant = stats["relevant"] >= 200
    gate_errors   = stats["error_rate"] < 0.05

    logger.info("\n  Phase 3 Gate Check:")
    logger.info(f"    ≥ 200 relevant items : {'✅' if gate_relevant else '❌'}  ({stats['relevant']})")
    logger.info(f"    < 5% error rate      : {'✅' if gate_errors   else '❌'}  ({stats['error_rate']:.1%})")

    if gate_relevant and gate_errors:
        logger.info("\n  ✅ Phase 3 COMPLETE — proceed to Phase 4 (cluster)")
    else:
        if not gate_relevant:
            logger.warning(
                f"\n  ⚠️  Only {stats['relevant']} relevant items. "
                "Collect more data with run_collectors.py."
            )
        if not gate_errors:
            logger.warning(
                f"\n  ⚠️  Error rate {stats['error_rate']:.1%} exceeds 5% threshold. "
                "Check GROQ_API_KEY and retry failed items."
            )


def run_cluster() -> None:
    """
    Phase 4 — Cluster tagged items into opportunity areas, score each area,
    write back assigned_areas, and export opportunity_areas.csv.
    """
    import pandas as pd
    from pipeline.raw_store import RawStore
    from pipeline.embedder import Embedder
    from pipeline.clusterer import (
        rule_based_cluster,
        semantic_cluster,
        merge_clusters,
        update_assigned_areas,
    )
    from pipeline.scorer import (
        score_opportunity_areas,
        build_comparison_table,
        flag_open_questions,
    )
    import config

    store    = RawStore()
    embedder = Embedder()

    # Load all relevant tagged items
    tagged_items = store.get_relevant_tagged_items()
    if not tagged_items:
        logger.warning(
            "[pipeline] No relevant tagged items found. "
            "Run --step tag first."
        )
        return

    logger.info(f"[pipeline] Relevant items for clustering: {len(tagged_items)}")

    # Fetch raw texts + urls for quote construction
    raw_texts, raw_urls = store.get_raw_texts_and_urls()

    # ── Pass 1: Rule-based clustering ─────────────────────────────────────
    logger.info("[pipeline] Running rule-based clustering...")
    rule_areas = rule_based_cluster(tagged_items)

    # ── Pass 2: Semantic clustering (HDBSCAN) ─────────────────────────────
    sem_areas: dict = {}
    if embedder.count() > 0:
        logger.info("[pipeline] Running HDBSCAN semantic clustering...")
        sem_areas = semantic_cluster(embedder, tagged_items)
    else:
        logger.info(
            "[pipeline] Chroma is empty — skipping semantic clustering. "
            "Run --step embed first to enable this."
        )

    # ── Merge ──────────────────────────────────────────────────────────────
    merged = merge_clusters(rule_areas, sem_areas)
    logger.info(f"[pipeline] Merged areas: {list(merged.keys())}")

    # Write back assigned_areas to each TaggedItem (DB + JSON export)
    updated_items = update_assigned_areas(merged)
    for item in updated_items:
        store.update_tagged_item_areas(item.id, item.assigned_areas)

    # ── Score ──────────────────────────────────────────────────────────────
    logger.info("[pipeline] Scoring opportunity areas...")
    scored_areas = score_opportunity_areas(
        merged,
        raw_texts=raw_texts,
        raw_urls=raw_urls,
        cluster_source="hybrid" if sem_areas else "rule_based",
    )

    # ── Persist to DB ──────────────────────────────────────────────────────
    for area in scored_areas:
        store.upsert_opportunity_area(area)
    logger.info(f"[pipeline] Persisted {len(scored_areas)} opportunity areas to DB")

    # ── Export opportunity_areas.csv ───────────────────────────────────────
    comparison_rows = build_comparison_table(scored_areas)
    df = pd.DataFrame(comparison_rows)
    df.to_csv(config.OPPORTUNITY_CSV_PATH, index=False, encoding="utf-8")
    logger.info(f"[pipeline] Exported: {config.OPPORTUNITY_CSV_PATH}")

    # Also re-export tagged_items.json with updated assigned_areas
    store.export_tagged_json(config.TAGGED_JSON_PATH)
    logger.info(f"[pipeline] Re-exported: {config.TAGGED_JSON_PATH}")

    # ── Open questions ─────────────────────────────────────────────────────
    open_qs = flag_open_questions(scored_areas)

    # ── Print summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 4 — CLUSTERING SUMMARY")
    logger.info("=" * 60)
    for row in comparison_rows:
        logger.info(
            f"  #{row['rank']} {row['name']:<35} "
            f"vol={row['evidence_volume']:>4}  "
            f"sev={row['severity_score']:.2f}  "
            f"div={row['source_diversity']}  "
            f"score={row['composite_score']:.2f}"
        )
    logger.info("=" * 60)

    if open_qs:
        logger.info(f"\n  Open Questions ({len(open_qs)} flags):")
        for flag in open_qs:
            logger.info(f"  [{flag['flag_type']}] {flag['area_name']}: {flag['message'][:80]}…")

    # Phase 4 gate check
    named_areas = [a for a in scored_areas if a.name != "Emerging / Uncategorized"]
    gate_areas = len(named_areas) >= 5
    gate_voiced = any(a.inferred_count > 0 for a in scored_areas)

    logger.info("\n  Phase 4 Gate Check:")
    logger.info(f"    ≥ 5 named areas     : {'✅' if gate_areas  else '❌'}  ({len(named_areas)})")
    logger.info(f"    ≥ 1 inferred area   : {'✅' if gate_voiced else '❌'}")

    if gate_areas and gate_voiced:
        logger.info("\n  ✅ Phase 4 COMPLETE — proceed to Phase 5 (dashboard)")
    else:
        logger.warning("\n  ⚠️  Gate not met. Collect more data or adjust clustering rules.")


def run_collect() -> None:
    """
    Phase 1 — Run data collection across all configured sources,
    normalize, deduplicate, and persist to SQLite and data/raw_items.csv.
    """
    from collectors import ALL_COLLECTORS
    from pipeline.normalizer import normalize_items
    from pipeline.deduplicator import deduplicate
    from pipeline.raw_store import RawStore
    import config

    store = RawStore()
    logger.info("[pipeline] Starting Phase 1 data collection...")

    collected_all = []
    for name, collector_cls in ALL_COLLECTORS.items():
        limit = config.SOURCE_LIMITS.get(name, 100)
        logger.info(f"  [{name}] Collecting up to {limit} items...")
        try:
            collector = collector_cls()
            items = collector.collect(limit=limit)
            logger.info(f"  [{name}] Harvested {len(items)} items")
            collected_all.extend(items)
        except Exception as e:
            logger.warning(f"  [{name}] Collector encountered error: {e}")

    logger.info(f"[pipeline] Normalizing {len(collected_all)} raw items...")
    normalized = normalize_items(collected_all)

    existing_hashes = store.get_all_hashes()
    unique_items, dupes = deduplicate(normalized, existing_hashes=existing_hashes)
    logger.info(f"[pipeline] Unique items: {len(unique_items)}, duplicates removed: {dupes}")

    inserted = store.insert_batch(unique_items)
    logger.info(f"[pipeline] Inserted {inserted} new items into SQLite")

    csv_path = store.export_csv(config.RAW_CSV_PATH)
    logger.info(f"[pipeline] Exported raw items to {csv_path}")

    # Phase 1 Gate Check
    total_raw = store.count()
    sources_summary = store.source_summary()
    gate_500 = total_raw >= 500
    gate_sources = len(sources_summary) >= 4

    logger.info("\n" + "=" * 55)
    logger.info("  PHASE 1 — COLLECTION SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Total raw items in DB: {total_raw}")
    logger.info(f"  Platforms active     : {len(sources_summary)}/6")
    for src, cnt in sources_summary.items():
        logger.info(f"    - {src:<15}: {cnt}")
    logger.info("=" * 55)

    logger.info("\n  Phase 1 Gate Check:")
    logger.info(f"    ≥ 500 raw items     : {'✅' if gate_500 else '⚠️'}  ({total_raw})")
    logger.info(f"    Multi-source spread : {'✅' if gate_sources else '⚠️'}  ({len(sources_summary)} sources)")


def run_report() -> None:
    """
    Phase 6 — Generate executive synthesis report (reports/synthesis_report.md)
    from scored opportunity areas and open questions.
    """
    import os
    import config
    from pipeline.raw_store import RawStore
    from pipeline.scorer import flag_open_questions
    from pipeline.reporter import generate_synthesis_report
    from models.schemas import OpportunityArea, Quote

    store = RawStore()
    area_dicts = store.get_all_opportunity_areas()

    if not area_dicts:
        logger.warning(
            "[pipeline] No opportunity areas found in DB. "
            "Run --step cluster first."
        )
        return

    logger.info(f"[pipeline] Loaded {len(area_dicts)} opportunity areas from DB.")

    # Reconstruct OpportunityArea models
    areas = []
    for d in area_dicts:
        quotes = [
            Quote(
                raw_item_id=q.get("raw_item_id", ""),
                text=q.get("text", ""),
                source_platform=q.get("source_platform", "unknown"),
                upvotes=q.get("upvotes", 0) or 0,
                url=q.get("url"),
            )
            for q in d.get("top_quotes", [])
        ]
        areas.append(
            OpportunityArea(
                name=d["name"],
                description=d.get("description", ""),
                evidence_volume=d.get("evidence_volume", 0),
                source_diversity=d.get("source_diversity", 0),
                severity_score=d.get("severity_score", 0.0),
                abandonment_rate=d.get("abandonment_rate", 0.0),
                voiced_count=d.get("voiced_count", 0),
                inferred_count=d.get("inferred_count", 0),
                top_quotes=quotes,
                platform_breakdown=d.get("platform_breakdown", {}),
                cluster_source=d.get("cluster_source", "rule_based"),
            )
        )

    # Compile open questions from contradiction/risk flags
    open_questions = flag_open_questions(areas)

    # Generate synthesis report
    logger.info("[pipeline] Generating executive synthesis report...")
    report_text = generate_synthesis_report(areas, open_questions=open_questions)

    # Phase 6 Gate Check
    report_exists = os.path.exists(config.SYNTHESIS_REPORT_PATH)
    named_areas = [a for a in areas if a.name != "Emerging / Uncategorized"]
    has_min_areas = len(named_areas) >= 3
    has_inferred = any(a.inferred_count > 0 for a in areas)

    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 6 — SYNTHESIS REPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Report file   : {config.SYNTHESIS_REPORT_PATH}")
    logger.info(f"  Report size   : {len(report_text):,} characters")
    logger.info(f"  Areas covered : {len(areas)}")
    logger.info(f"  Open questions: {len(open_questions)} items flagged")
    logger.info("=" * 60)

    logger.info("\n  Phase 6 Gate Check:")
    logger.info(f"    Report generated    : {'✅' if report_exists else '❌'}")
    logger.info(f"    ≥ 3 opportunity areas: {'✅' if has_min_areas else '❌'}  ({len(named_areas)})")
    logger.info(f"    ≥ 1 inferred insight : {'✅' if has_inferred else '❌'}")

    if report_exists and has_min_areas and has_inferred:
        logger.info("\n  ✅ Phase 6 COMPLETE — Full pipeline deliverables generated successfully!")
    else:
        logger.warning("\n  ⚠️  Phase 6 gate check incomplete. Review clustering and data inputs.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
    )
    logger.add(
        f"logs/pipeline.log",
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    args = parse_args()

    if args.status:
        print_status()
        return

    STEPS = ["collect", "embed", "tag", "cluster", "report"]

    def should_run(target_step: str) -> bool:
        """Return True if this step should execute given the CLI args."""
        if args.status:
            return False
        if args.step:
            return args.step == target_step
        if args.from_step:
            return STEPS.index(target_step) >= STEPS.index(args.from_step)
        return True  # no flag = run all

    if should_run("collect"):
        logger.info("▶ Phase 1: Collecting feedback from all platforms...")
        run_collect()

    if should_run("embed"):
        logger.info("▶ Phase 2: Embedding raw items into Chroma...")
        run_embed()

    if should_run("tag"):
        logger.info("▶ Phase 3: Classifying items with Groq LLM tagger...")
        run_tag()

    if should_run("cluster"):
        logger.info("▶ Phase 4: Clustering, scoring & opportunity comparison...")
        run_cluster()

    if should_run("report"):
        logger.info("▶ Phase 6: Generating executive synthesis report...")
        run_report()


if __name__ == "__main__":
    main()

