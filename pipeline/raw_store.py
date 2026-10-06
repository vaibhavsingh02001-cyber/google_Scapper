"""
pipeline/raw_store.py — SQLite persistence layer for raw collected items.
Manages the `raw_items` table (Phase 1) and all 4 tables (Phase 2+).
Thread-safe with WAL mode enabled.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Generator, List, Optional, Set

import pandas as pd
from loguru import logger

from models.schemas import RawItem, TaggedItem, SourcePlatform, FrustrationLevel
import config


class RawStore:
    """
    Manages the discovery_engine SQLite database.
    Tables: raw_items, tagged_items, opportunity_areas, quotes.
    """

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Connection management ─────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields an open connection and commits on exit."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL for concurrent read/write safety
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema initialisation ─────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create all tables if they do not already exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS raw_items (
                    id                TEXT PRIMARY KEY,
                    source            TEXT NOT NULL,
                    platform_item_id  TEXT,
                    text              TEXT NOT NULL,
                    date              TEXT,
                    upvotes           INTEGER DEFAULT 0,
                    url               TEXT,
                    collected_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS tagged_items (
                    id                TEXT PRIMARY KEY,
                    raw_item_id       TEXT NOT NULL,
                    relevant          INTEGER NOT NULL,
                    problem_types     TEXT DEFAULT '[]',
                    remembered_cues   TEXT DEFAULT '[]',
                    forgotten_cues    TEXT DEFAULT '[]',
                    search_method     TEXT,
                    failure_points    TEXT DEFAULT '[]',
                    frustration_level TEXT,
                    abandoned_app     INTEGER,
                    classifier_error  INTEGER DEFAULT 0,
                    pattern_type      TEXT,
                    assigned_areas    TEXT DEFAULT '[]',
                    source_platform   TEXT,
                    date              TEXT,
                    upvotes           INTEGER DEFAULT 0,
                    FOREIGN KEY (raw_item_id) REFERENCES raw_items(id)
                );

                CREATE TABLE IF NOT EXISTS opportunity_areas (
                    name               TEXT PRIMARY KEY,
                    description        TEXT,
                    evidence_volume    INTEGER DEFAULT 0,
                    source_diversity   INTEGER DEFAULT 0,
                    severity_score     REAL DEFAULT 0.0,
                    abandonment_rate   REAL DEFAULT 0.0,
                    voiced_count       INTEGER DEFAULT 0,
                    inferred_count     INTEGER DEFAULT 0,
                    top_quotes         TEXT DEFAULT '[]',
                    platform_breakdown TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS quotes (
                    id               TEXT PRIMARY KEY,
                    area_name        TEXT NOT NULL,
                    text             TEXT NOT NULL,
                    source_platform  TEXT,
                    upvotes          INTEGER DEFAULT 0,
                    url              TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_raw_source   ON raw_items(source);
                CREATE INDEX IF NOT EXISTS idx_tagged_relevant ON tagged_items(relevant);
                CREATE INDEX IF NOT EXISTS idx_tagged_raw_id  ON tagged_items(raw_item_id);
            """)
        logger.debug(f"[raw_store] Database initialised at {self.db_path}")

    # ── Raw Items ─────────────────────────────────────────────────────────────

    def insert_batch(self, items: List[RawItem]) -> int:
        """
        Upsert a batch of RawItems into the raw_items table.
        Returns the number of newly inserted rows.
        """
        if not items:
            return 0

        rows = [
            (
                item.id,
                item.source.value,
                item.platform_item_id,
                item.text,
                item.date,
                item.upvotes,
                item.url,
                item.collected_at,
            )
            for item in items
        ]

        with self._conn() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO raw_items
                    (id, source, platform_item_id, text, date, upvotes, url, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = cursor.rowcount

        logger.debug(f"[raw_store] Inserted {inserted}/{len(items)} items (ignoring dupes)")
        return inserted

    def count(self, source: Optional[str] = None) -> int:
        """Return the total number of raw items (optionally filtered by source)."""
        with self._conn() as conn:
            if source:
                row = conn.execute(
                    "SELECT COUNT(*) FROM raw_items WHERE source = ?", (source,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()
        return row[0] if row else 0

    def get_all_hashes(self) -> Set[str]:
        """
        Return the set of SHA-256 hashes for all existing raw items,
        used by the deduplicator on incremental runs.
        """
        import hashlib

        with self._conn() as conn:
            rows = conn.execute("SELECT source, text FROM raw_items").fetchall()

        hashes: Set[str] = set()
        for row in rows:
            key = f"{row['source']}::{row['text'][:200].lower().strip()}"
            hashes.add(hashlib.sha256(key.encode()).hexdigest())
        return hashes

    def get_untagged_items(self) -> List[RawItem]:
        """Fetch raw items that have no entry in tagged_items yet."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT r.* FROM raw_items r
                LEFT JOIN tagged_items t ON t.raw_item_id = r.id
                WHERE t.id IS NULL
                """
            ).fetchall()

        return [
            RawItem(
                id=row["id"],
                source=SourcePlatform(row["source"]),
                platform_item_id=row["platform_item_id"],
                text=row["text"],
                date=row["date"],
                upvotes=row["upvotes"] or 0,
                url=row["url"],
                collected_at=row["collected_at"] or "",
            )
            for row in rows
        ]

    # ── Export utilities ──────────────────────────────────────────────────────

    def export_csv(self, path: str = config.RAW_CSV_PATH) -> str:
        """Export the full raw_items table to a CSV file."""
        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM raw_items", conn)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"[raw_store] Exported {len(df)} raw items to {path}")
        return path

    def export_tagged_json(self, path: str) -> str:
        """Export the full tagged_items table to a JSON file."""
        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM tagged_items", conn)

        # Deserialise JSON array columns
        for col in ["problem_types", "remembered_cues", "forgotten_cues",
                    "failure_points", "assigned_areas"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.loads(v) if isinstance(v, str) else v
                )

        records = df.to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"[raw_store] Exported {len(records)} tagged items to {path}")
        return path

    def export_opportunity_csv(self, path: str) -> str:
        """Export the opportunity_areas table to a CSV file."""
        with self._conn() as conn:
            df = pd.read_sql_query("SELECT * FROM opportunity_areas", conn)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"[raw_store] Exported {len(df)} opportunity areas to {path}")
        return path

    def source_summary(self) -> dict:
        """Return a dict of {source: count} for all raw items."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) as n FROM raw_items GROUP BY source"
            ).fetchall()
        return {row["source"]: row["n"] for row in rows}

    # ── Tagged Items (Phase 3) ────────────────────────────────────────────────

    def insert_tagged_item(self, item: TaggedItem) -> None:
        """
        Upsert a single TaggedItem into the tagged_items table.
        JSON array fields are serialised before storage.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tagged_items (
                    id, raw_item_id, relevant,
                    problem_types, remembered_cues, forgotten_cues,
                    search_method, failure_points,
                    frustration_level, abandoned_app,
                    classifier_error, pattern_type, assigned_areas,
                    source_platform, date, upvotes
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    item.id,
                    item.raw_item_id,
                    1 if item.relevant else 0,
                    json.dumps(item.problem_types),
                    json.dumps(item.remembered_cues),
                    json.dumps(item.forgotten_cues),
                    item.search_method,
                    json.dumps(item.failure_points),
                    item.frustration_level.value if item.frustration_level else None,
                    1 if item.abandoned_app else (0 if item.abandoned_app is False else None),
                    1 if item.classifier_error else 0,
                    item.pattern_type.value if item.pattern_type else None,
                    json.dumps(item.assigned_areas),
                    item.source_platform.value,
                    item.date,
                    item.upvotes,
                ),
            )

    def get_tagging_stats(self) -> dict:
        """
        Return counts for Phase 3 QA gate:
        {total, relevant, irrelevant, errors, error_rate}.
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                      AS total,
                    SUM(CASE WHEN relevant = 1 THEN 1 ELSE 0 END) AS relevant,
                    SUM(CASE WHEN relevant = 0 AND classifier_error = 0
                             THEN 1 ELSE 0 END)                   AS irrelevant,
                    SUM(classifier_error)                         AS errors
                FROM tagged_items
                """
            ).fetchone()

        total    = row["total"]    or 0
        relevant = row["relevant"] or 0
        irrelevant = row["irrelevant"] or 0
        errors   = row["errors"]   or 0
        error_rate = (errors / total) if total else 0.0

        return {
            "total":      total,
            "relevant":   relevant,
            "irrelevant": irrelevant,
            "errors":     errors,
            "error_rate": round(error_rate, 4),
        }

    def get_relevant_tagged_items(self) -> List[TaggedItem]:
        """
        Return all tagged items where relevant=True and classifier_error=False.
        Used by the clustering phase (Phase 4).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT t.*, r.url
                FROM tagged_items t
                LEFT JOIN raw_items r ON r.id = t.raw_item_id
                WHERE t.relevant = 1 AND t.classifier_error = 0
                """
            ).fetchall()

        items: List[TaggedItem] = []
        for row in rows:
            fl_val = row["frustration_level"]
            pt_val = row["pattern_type"]
            items.append(
                TaggedItem(
                    id=row["id"],
                    raw_item_id=row["raw_item_id"],
                    relevant=True,
                    problem_types=json.loads(row["problem_types"] or "[]"),
                    remembered_cues=json.loads(row["remembered_cues"] or "[]"),
                    forgotten_cues=json.loads(row["forgotten_cues"] or "[]"),
                    search_method=row["search_method"],
                    failure_points=json.loads(row["failure_points"] or "[]"),
                    frustration_level=FrustrationLevel(fl_val) if fl_val else None,
                    abandoned_app=bool(row["abandoned_app"]) if row["abandoned_app"] is not None else None,
                    classifier_error=False,
                    pattern_type=None if not pt_val else (
                        __import__("models.schemas", fromlist=["PatternType"]).PatternType(pt_val)
                    ),
                    assigned_areas=json.loads(row["assigned_areas"] or "[]"),
                    source_platform=SourcePlatform(row["source_platform"]),
                    date=row["date"],
                    upvotes=row["upvotes"] or 0,
                )
            )
        return items

    # ── Opportunity Areas & Quotes (Phase 4) ──────────────────────────────────

    def upsert_opportunity_area(self, area) -> None:
        """
        Insert or replace an OpportunityArea in the opportunity_areas table.
        top_quotes and platform_breakdown are serialised as JSON.
        """
        from models.schemas import OpportunityArea
        quotes_json = json.dumps(
            [q.model_dump() for q in area.top_quotes]
        )
        breakdown_json = json.dumps(area.platform_breakdown)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO opportunity_areas (
                    name, description, evidence_volume, source_diversity,
                    severity_score, abandonment_rate, voiced_count, inferred_count,
                    top_quotes, platform_breakdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    area.name,
                    area.description,
                    area.evidence_volume,
                    area.source_diversity,
                    area.severity_score,
                    area.abandonment_rate,
                    area.voiced_count,
                    area.inferred_count,
                    quotes_json,
                    breakdown_json,
                ),
            )

        # Persist quotes to the quotes table
        for q in area.top_quotes:
            import uuid as _uuid
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO quotes
                        (id, area_name, text, source_platform, upvotes, url)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(_uuid.uuid4()),
                        area.name,
                        q.text,
                        q.source_platform,
                        q.upvotes,
                        q.url,
                    ),
                )

    def update_tagged_item_areas(self, item_id: str, assigned_areas: List[str]) -> None:
        """Write back the assigned_areas list to a tagged_items row."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE tagged_items SET assigned_areas = ? WHERE id = ?",
                (json.dumps(assigned_areas), item_id),
            )

    def get_raw_texts_and_urls(self) -> tuple:
        """
        Return two dicts: {raw_item_id → text} and {raw_item_id → url}.
        Used by the scorer to build Quote objects with verbatim text.
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT id, text, url FROM raw_items").fetchall()
        texts = {row["id"]: row["text"] for row in rows}
        urls  = {row["id"]: row["url"]  for row in rows}
        return texts, urls

    def get_all_opportunity_areas(self) -> List[dict]:
        """
        Return all opportunity_areas rows as a list of dicts
        (top_quotes and platform_breakdown deserialised from JSON).
        Used by the dashboard.
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM opportunity_areas").fetchall()
        areas = []
        for row in rows:
            d = dict(row)
            d["top_quotes"]         = json.loads(d.get("top_quotes") or "[]")
            d["platform_breakdown"] = json.loads(d.get("platform_breakdown") or "{}")
            areas.append(d)
        return areas
