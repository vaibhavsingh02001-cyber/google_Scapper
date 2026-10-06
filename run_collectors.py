"""
run_collectors.py — CLI entry point for Phase 1 data collection.

Usage:
    python run_collectors.py --sources all --limit 200
    python run_collectors.py --sources reddit,youtube --limit 100
    python run_collectors.py --sources play_store --limit 300
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from collectors import ALL_COLLECTORS
from pipeline.normalizer import normalize_items
from pipeline.deduplicator import deduplicate
from pipeline.raw_store import RawStore
import config


# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
logger.add(
    f"{config.LOGS_DIR}/collection.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discovery Engine — Data Collection Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_collectors.py --sources all --limit 200
  python run_collectors.py --sources reddit,youtube --limit 100
  python run_collectors.py --sources play_store --limit 300
        """,
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=(
            'Comma-separated source names, or "all". '
            f'Available: {", ".join(ALL_COLLECTORS.keys())}'
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max items to collect per source (default: 200)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export raw_items.csv after collection (default: True)",
    )
    return parser.parse_args()


def resolve_sources(sources_arg: str) -> list[str]:
    if sources_arg.strip().lower() == "all":
        return list(ALL_COLLECTORS.keys())
    names = [s.strip() for s in sources_arg.split(",")]
    invalid = [n for n in names if n not in ALL_COLLECTORS]
    if invalid:
        logger.error(f"Unknown sources: {invalid}. Valid: {list(ALL_COLLECTORS.keys())}")
        sys.exit(1)
    return names


def main() -> None:
    args = parse_args()
    sources = resolve_sources(args.sources)

    logger.info("=" * 60)
    logger.info("  Discovery Engine — Data Collection")
    logger.info(f"  Sources : {sources}")
    logger.info(f"  Limit   : {args.limit} per source")
    logger.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    store = RawStore()
    existing_hashes = store.get_all_hashes()

    summary: dict[str, int] = {}

    for source_name in tqdm(sources, desc="Collecting sources", unit="source"):
        logger.info(f"\n▶ Collecting from: {source_name}")
        CollectorClass = ALL_COLLECTORS[source_name]
        collector = CollectorClass()

        try:
            # 1. Collect raw items
            raw_items = collector.collect(limit=args.limit)
            logger.info(f"  Raw items fetched : {len(raw_items)}")

            if not raw_items:
                summary[source_name] = 0
                continue

            # 2. Normalize text
            normalized = normalize_items(raw_items)
            logger.info(f"  After normalization: {len(normalized)}")

            # 3. Deduplicate (against DB + within-batch)
            unique, existing_hashes = deduplicate(normalized, existing_hashes)
            logger.info(f"  After dedup       : {len(unique)}")

            # 4. Persist to SQLite
            inserted = store.insert_batch(unique)
            summary[source_name] = inserted
            logger.info(f"  Inserted          : {inserted} new items")

        except Exception as exc:
            logger.error(f"  [{source_name}] Collection failed: {exc}")
            summary[source_name] = 0

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  COLLECTION SUMMARY")
    logger.info("=" * 60)
    total = 0
    for src, count in summary.items():
        logger.info(f"  {src:<15} : {count:>5} items")
        total += count
    logger.info(f"  {'TOTAL':<15} : {total:>5} items")
    logger.info(f"  DB Total       : {store.count():>5} items")

    if args.export_csv:
        path = store.export_csv()
        logger.info(f"  CSV exported   : {path}")

    logger.info("=" * 60)

    if store.count() < config.TARGET_RAW_COUNT:
        logger.warning(
            f"\n⚠️  Only {store.count()} items collected. "
            f"Target is {config.TARGET_RAW_COUNT}. "
            "Consider broadening queries or running additional sources."
        )
    else:
        logger.info(
            f"\n✅ Phase 1 target met: {store.count()} items in DB "
            f"(target: {config.TARGET_RAW_COUNT})"
        )


if __name__ == "__main__":
    main()
