"""
pipeline/export_bundle.py — Pre-compiles the entire Discovery Engine dataset into a single
high-performance bundle.json for the Vercel web application.
"""
from __future__ import annotations

import json
import os
import sqlite3
import pandas as pd

import config
from pipeline.raw_store import RawStore
from pipeline.dashboard_data import (
    get_kpi_stats,
    get_opportunity_areas_df,
    load_synthesis_report_markdown,
)
from pipeline.chat_engine import get_corpus_cognitive_metrics


def build_bundle() -> dict:
    """Build unified data bundle for the web application."""
    store = RawStore()

    # 1. KPIs & High-level stats
    kpi_stats = get_kpi_stats()

    # 2. Opportunity Areas
    opp_areas_df = get_opportunity_areas_df()
    opp_areas = opp_areas_df.to_dict(orient="records") if not opp_areas_df.empty else []

    # 3. Cognitive memory metrics
    cognitive_metrics = get_corpus_cognitive_metrics()

    # 4. Merged Raw + Tagged Items
    items = []
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT 
                    r.id, r.source, r.text, r.date, r.upvotes, r.url,
                    t.relevant, t.problem_types, t.remembered_cues, t.forgotten_cues,
                    t.search_method, t.failure_points, t.frustration_level,
                    t.abandoned_app, t.pattern_type, t.assigned_areas
                FROM raw_items r
                LEFT JOIN tagged_items t ON t.raw_item_id = r.id
                ORDER BY r.upvotes DESC
                """
            ).fetchall()

            for r in rows:
                pts = json.loads(r["problem_types"] or "[]") if r["problem_types"] else []
                rcs = json.loads(r["remembered_cues"] or "[]") if r["remembered_cues"] else []
                fcs = json.loads(r["forgotten_cues"] or "[]") if r["forgotten_cues"] else []
                fps = json.loads(r["failure_points"] or "[]") if r["failure_points"] else []
                areas = json.loads(r["assigned_areas"] or "[]") if r["assigned_areas"] else []

                items.append({
                    "id": r["id"],
                    "source": r["source"],
                    "text": r["text"],
                    "date": r["date"] or "",
                    "upvotes": r["upvotes"] or 0,
                    "url": r["url"] or "",
                    "relevant": bool(r["relevant"]),
                    "problem_types": pts,
                    "remembered_cues": rcs,
                    "forgotten_cues": fcs,
                    "search_method": r["search_method"] or "",
                    "failure_points": fps,
                    "frustration_level": r["frustration_level"] or "med",
                    "abandoned_app": bool(r["abandoned_app"]),
                    "pattern_type": r["pattern_type"] or "voiced",
                    "assigned_areas": areas,
                })
    except Exception as e:
        print(f"Error merging items: {e}")

    # 5. Synthesis Report Markdown
    report_markdown = load_synthesis_report_markdown()

    bundle = {
        "kpis": kpi_stats,
        "cognitive_metrics": cognitive_metrics,
        "opportunity_areas": opp_areas,
        "items": items,
        "report_markdown": report_markdown,
    }

    out_path = os.path.join(config.DATA_DIR, "bundle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"Exported bundle to {out_path} ({len(items)} items, {len(opp_areas)} areas)")
    return bundle


if __name__ == "__main__":
    build_bundle()
