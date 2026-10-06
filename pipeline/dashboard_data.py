"""
pipeline/dashboard_data.py — Data access and analytics layer for Streamlit dashboard.
Provides cached queries, aggregations, and fallback logic for all dashboard pages.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

import config
from pipeline.raw_store import RawStore


def get_db_connection() -> sqlite3.Connection:
    """Return a read-only or standard sqlite3 connection with Row factory."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_kpi_stats() -> Dict[str, Any]:
    """
    Retrieve headline metrics across raw items, tagged items, and opportunity areas.
    Safe against empty or non-existent tables.
    """
    stats = {
        "total_raw": 0,
        "total_relevant": 0,
        "source_count": 0,
        "total_sources_possible": 6,
        "areas_count": 0,
        "voiced_total": 0,
        "inferred_total": 0,
        "avg_severity": 0.0,
        "has_data": False,
    }

    if not os.path.exists(config.DB_PATH):
        return stats

    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Raw items count and distinct sources
            try:
                row = cursor.execute(
                    "SELECT COUNT(*) as n, COUNT(DISTINCT source) as sources FROM raw_items"
                ).fetchone()
                if row:
                    stats["total_raw"] = row["n"] or 0
                    stats["source_count"] = row["sources"] or 0
            except sqlite3.OperationalError:
                pass

            # Tagged items count
            try:
                row = cursor.execute(
                    """
                    SELECT 
                        SUM(CASE WHEN relevant = 1 THEN 1 ELSE 0 END) as relevant,
                        SUM(CASE WHEN pattern_type = 'voiced' THEN 1 ELSE 0 END) as voiced,
                        SUM(CASE WHEN pattern_type = 'inferred' THEN 1 ELSE 0 END) as inferred
                    FROM tagged_items
                    """
                ).fetchone()
                if row and row["relevant"] is not None:
                    stats["total_relevant"] = row["relevant"] or 0
                    stats["voiced_total"] = row["voiced"] or 0
                    stats["inferred_total"] = row["inferred"] or 0
            except sqlite3.OperationalError:
                pass

            # Opportunity areas count and average severity
            try:
                row = cursor.execute(
                    "SELECT COUNT(*) as n, AVG(severity_score) as avg_sev FROM opportunity_areas"
                ).fetchone()
                if row and row["n"]:
                    stats["areas_count"] = row["n"] or 0
                    stats["avg_severity"] = round(row["avg_sev"] or 0.0, 2)
            except sqlite3.OperationalError:
                pass

        stats["has_data"] = stats["total_raw"] > 0
    except Exception as e:
        logger.warning(f"[dashboard_data] Error fetching KPI stats: {e}")

    return stats


def get_opportunity_areas_df() -> pd.DataFrame:
    """
    Fetch all opportunity areas from SQLite and return as a sorted DataFrame.
    Includes composite score calculation: severity × log1p(volume) × diversity.
    """
    store = RawStore()
    areas = store.get_all_opportunity_areas()
    if not areas:
        return pd.DataFrame()

    records = []
    for a in areas:
        vol = a.get("evidence_volume", 0) or 0
        sev = a.get("severity_score", 0.0) or 0.0
        div = a.get("source_diversity", 0) or 0
        voiced = a.get("voiced_count", 0) or 0
        inferred = a.get("inferred_count", 0) or 0
        abandonment = a.get("abandonment_rate", 0.0) or 0.0

        # Composite priority score
        composite_score = round(sev * math.log1p(vol) * div, 2)

        records.append({
            "Opportunity Area": a.get("name", "Unnamed"),
            "Description": a.get("description", ""),
            "Evidence Volume": vol,
            "Source Diversity": f"{div}/6",
            "Diversity Raw": div,
            "Severity Score": round(sev, 2),
            "Abandonment Rate": f"{round(abandonment * 100, 1)}%",
            "Abandonment Raw": abandonment,
            "Voiced Count": voiced,
            "Inferred Count": inferred,
            "Composite Score": composite_score,
            "Top Quotes": a.get("top_quotes", []),
            "Platform Breakdown": a.get("platform_breakdown", {}),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Composite Score", ascending=False).reset_index(drop=True)
    return df


def get_opportunity_area_detail(area_name: str) -> Dict[str, Any]:
    """
    Fetch comprehensive details for a single opportunity area:
    - Metadata and scores
    - Failure points (A, B, C, D) × Platform matrix for heatmap
    - Quotes with verbatim text, upvotes, and URLs
    - Voiced vs Inferred breakdown
    """
    detail: Dict[str, Any] = {
        "name": area_name,
        "description": "",
        "evidence_volume": 0,
        "source_diversity": 0,
        "severity_score": 0.0,
        "abandonment_rate": 0.0,
        "voiced_count": 0,
        "inferred_count": 0,
        "quotes": [],
        "heatmap_df": pd.DataFrame(),
        "stages_present": [],
    }

    if not os.path.exists(config.DB_PATH):
        return detail

    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Area record
            row = cursor.execute(
                "SELECT * FROM opportunity_areas WHERE name = ?", (area_name,)
            ).fetchone()
            if row:
                detail["name"] = row["name"]
                detail["description"] = row["description"] or ""
                detail["evidence_volume"] = row["evidence_volume"] or 0
                detail["source_diversity"] = row["source_diversity"] or 0
                detail["severity_score"] = round(row["severity_score"] or 0.0, 2)
                detail["abandonment_rate"] = round((row["abandonment_rate"] or 0.0) * 100, 1)
                detail["voiced_count"] = row["voiced_count"] or 0
                detail["inferred_count"] = row["inferred_count"] or 0
                detail["top_quotes"] = json.loads(row["top_quotes"] or "[]")

            # Fetch matching tagged items for this area to compute the heatmap
            # assigned_areas is stored as JSON array: e.g. ["Temporal & Anchor Vagueness"]
            search_pattern = f'%"{area_name}"%'
            tagged_rows = cursor.execute(
                """
                SELECT t.failure_points, t.source_platform, t.pattern_type, t.upvotes,
                       r.text, r.url, r.id as raw_id
                FROM tagged_items t
                JOIN raw_items r ON r.id = t.raw_item_id
                WHERE t.assigned_areas LIKE ?
                ORDER BY t.upvotes DESC
                LIMIT 500
                """,
                (search_pattern,),
            ).fetchall()

            platforms = ["play_store", "app_store", "reddit", "youtube", "help_forum", "web"]
            stages = ["a", "b", "c", "d"]
            stage_labels = {
                "a": "Stage A (Query Formation)",
                "b": "Stage B (Visual Search / Indexing)",
                "c": "Stage C (Refinement & Browsing)",
                "d": "Stage D (Trust & Verification)",
            }

            # Initialize 2D grid: stages × platforms
            counts = {stage_labels[s]: {p: 0 for p in platforms} for s in stages}
            stages_found = set()
            quotes_list = []

            for tr in tagged_rows:
                src = tr["source_platform"]
                fps = json.loads(tr["failure_points"] or "[]")
                for fp in fps:
                    fp_clean = str(fp).lower().strip()
                    if fp_clean in stage_labels:
                        stages_found.add(fp_clean)
                        if src in counts[stage_labels[fp_clean]]:
                            counts[stage_labels[fp_clean]][src] += 1

                # Collect quotes
                quotes_list.append({
                    "text": tr["text"],
                    "source": tr["source_platform"],
                    "upvotes": tr["upvotes"] or 0,
                    "url": tr["url"],
                })

            detail["stages_present"] = sorted(list(stages_found))
            detail["heatmap_df"] = pd.DataFrame.from_dict(counts, orient="index")
            detail["quotes"] = quotes_list if quotes_list else detail.get("top_quotes", [])

    except Exception as e:
        logger.warning(f"[dashboard_data] Error building area detail for '{area_name}': {e}")

    return detail


def get_filtered_corpus(
    sources: Optional[List[str]] = None,
    problem_types: Optional[List[str]] = None,
    failure_points: Optional[List[str]] = None,
    frustration_levels: Optional[List[str]] = None,
    search_query: Optional[str] = None,
    search_mode: str = "keyword",
    page: int = 1,
    page_size: int = 50,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Query the corpus with multi-select filters and keyword/semantic search.
    Returns:
        (page_df, total_matched_count, full_export_df)
    """
    empty_df = pd.DataFrame(
        columns=[
            "ID", "Source", "Feedback Text", "Problem Types",
            "Failure Points", "Frustration", "Upvotes", "URL", "Date"
        ]
    )
    if not os.path.exists(config.DB_PATH):
        return empty_df, 0, empty_df

    matched_ids: Optional[List[str]] = None

    # Semantic search via Chroma if requested
    if search_query and search_query.strip() and search_mode == "semantic":
        try:
            from pipeline.embedder import Embedder
            embedder = Embedder()
            if embedder.count() > 0:
                results = embedder.query_similar(search_query.strip(), n=250)
                matched_ids = [r["id"] for r in results]
                if not matched_ids:
                    # Semantic search yielded no hits
                    matched_ids = ["__NO_MATCH__"]
        except Exception as e:
            logger.info(f"[dashboard_data] Semantic search fallback to keyword: {e}")
            search_mode = "keyword"

    # Build SQL query
    where_clauses: List[str] = []
    params: List[Any] = []

    # Filter: Source Platform
    if sources:
        placeholders = ",".join("?" for _ in sources)
        where_clauses.append(f"r.source IN ({placeholders})")
        params.extend(sources)

    # Filter: Frustration Level
    if frustration_levels:
        placeholders = ",".join("?" for _ in frustration_levels)
        where_clauses.append(f"t.frustration_level IN ({placeholders})")
        params.extend(frustration_levels)

    # Filter: Problem Types (multi-label JSON string search)
    if problem_types:
        sub_clauses = []
        for pt in problem_types:
            sub_clauses.append("t.problem_types LIKE ?")
            params.append(f'%"{pt}"%')
        where_clauses.append(f"({' OR '.join(sub_clauses)})")

    # Filter: Failure Points (multi-label JSON string search)
    if failure_points:
        sub_clauses = []
        for fp in failure_points:
            sub_clauses.append("t.failure_points LIKE ?")
            params.append(f'%"{fp}"%')
        where_clauses.append(f"({' OR '.join(sub_clauses)})")

    # Search query
    if matched_ids is not None:
        placeholders = ",".join("?" for _ in matched_ids)
        where_clauses.append(f"r.id IN ({placeholders})")
        params.extend(matched_ids)
    elif search_query and search_query.strip() and search_mode == "keyword":
        where_clauses.append("r.text LIKE ?")
        params.append(f"%{search_query.strip()}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    base_from = """
        FROM raw_items r
        LEFT JOIN tagged_items t ON t.raw_item_id = r.id
    """

    count_sql = f"SELECT COUNT(*) {base_from} {where_sql}"

    select_sql = f"""
        SELECT 
            r.id,
            r.source,
            r.text,
            r.date,
            r.upvotes,
            r.url,
            t.relevant,
            t.problem_types,
            t.failure_points,
            t.frustration_level,
            t.pattern_type,
            t.assigned_areas
        {base_from}
        {where_sql}
        ORDER BY r.upvotes DESC, r.date DESC
    """

    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Count total
            total_count = cursor.execute(count_sql, params).fetchone()[0]

            if total_count == 0:
                return empty_df, 0, empty_df

            # Fetch all for export (limit to 5000 max for memory safety)
            export_rows = cursor.execute(f"{select_sql} LIMIT 5000", params).fetchall()

            # Fetch page slice
            offset = max(0, (page - 1) * page_size)
            page_rows = cursor.execute(
                f"{select_sql} LIMIT ? OFFSET ?", params + [page_size, offset]
            ).fetchall()

        def parse_row(r: sqlite3.Row) -> dict:
            pts = json.loads(r["problem_types"] or "[]") if r["problem_types"] else []
            fps = json.loads(r["failure_points"] or "[]") if r["failure_points"] else []
            areas = json.loads(r["assigned_areas"] or "[]") if r["assigned_areas"] else []
            return {
                "ID": r["id"],
                "Source": r["source"],
                "Feedback Text": r["text"],
                "Problem Types": ", ".join(pts) if pts else "—",
                "Failure Points": ", ".join([str(p).upper() for p in fps]) if fps else "—",
                "Frustration": (r["frustration_level"] or "—").capitalize(),
                "Pattern": (r["pattern_type"] or "—").capitalize(),
                "Assigned Areas": ", ".join(areas) if areas else "—",
                "Upvotes": r["upvotes"] or 0,
                "URL": r["url"] or "",
                "Date": r["date"] or "",
            }

        page_df = pd.DataFrame([parse_row(r) for r in page_rows])
        export_df = pd.DataFrame([parse_row(r) for r in export_rows])

        return page_df, total_count, export_df

    except Exception as e:
        logger.warning(f"[dashboard_data] Error querying filtered corpus: {e}")
        return empty_df, 0, empty_df


def load_synthesis_report_markdown() -> str:
    """
    Read reports/synthesis_report.md.
    If the file does not exist yet (prior to Phase 6), returns a cleanly formatted
    pipeline synthesis preview with active data from the database.
    """
    report_path = config.SYNTHESIS_REPORT_PATH
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"[dashboard_data] Error reading report file: {e}")

    # Fallback preview if Phase 6 hasn't run yet
    kpis = get_kpi_stats()
    areas_df = get_opportunity_areas_df()

    preview = [
        "# 📑 Google Photos Retrieval Failure — Executive Synthesis Report",
        "",
        "> **Status:** Preliminary Pipeline Summary (Run Phase 6 to generate full LLM synthesis).",
        "",
        "## Executive Overview",
        f"- **Total Evidence Corpus:** {kpis['total_raw']:,} raw feedback items collected across {kpis['source_count']}/6 platforms.",
        f"- **Domain Relevant Failures:** {kpis['total_relevant']:,} items classified as specific photo retrieval breakdowns.",
        f"- **Opportunity Clusters Identified:** {kpis['areas_count']} opportunity areas scored across severity, volume, and source diversity.",
        "",
        "## Opportunity Areas Ranking Summary",
        "",
    ]

    if not areas_df.empty:
        summary_df = areas_df[[
            "Opportunity Area", "Composite Score", "Evidence Volume",
            "Severity Score", "Abandonment Rate", "Source Diversity"
        ]]
        try:
            preview.append(summary_df.to_markdown(index=False))
        except Exception:
            # Fallback simple markdown table generator
            headers = list(summary_df.columns)
            preview.append("| " + " | ".join(headers) + " |")
            preview.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, row in summary_df.iterrows():
                preview.append("| " + " | ".join(str(val) for val in row.values) + " |")
    else:
        preview.append(
            "_No opportunity clusters computed yet. Execute `python run_pipeline.py --step all` to build clusters._"
        )

    preview.extend([
        "",
        "---",
        "### How to generate the final Phase 6 LLM report:",
        "```bash",
        "python run_pipeline.py --step report",
        "```",
    ])

    return "\n".join(preview)


def inject_custom_css():
    """Inject cohesive Google Photos Material 3 aesthetic matching Wishlist Intelligence Dashboard."""
    import streamlit as st
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');
        
        :root {
            --gp-blue: #1a73e8;
            --gp-blue-hover: #1557b0;
            --gp-blue-soft: #e8f0fe;
            --gp-red: #ea4335;
            --gp-red-soft: #fce8e6;
            --gp-yellow: #fbbc04;
            --gp-yellow-soft: #fef7e0;
            --gp-green: #34a853;
            --gp-green-soft: #e6f4ea;
            --gp-purple: #a142f4;
            --gp-purple-soft: #f3e8fd;
            --gp-teal: #24c1e0;
            --gp-teal-soft: #e4f7fb;
            --canvas-bg: #f0f3f8;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --text-dark: #1e293b;
            --text-muted: #64748b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* App Background */
        .stApp {
            background-color: var(--canvas-bg);
            color: var(--text-dark);
        }

        /* Sidebar Styling (Dark Material Theme) */
        section[data-testid="stSidebar"] {
            background-color: #151821 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label {
            color: #94a3b8 !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Primary Action Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button[kind="primary"], .gp-btn-primary {
            background-color: var(--gp-blue) !important;
            color: #ffffff !important;
            border: none !important;
        }
        .stButton > button[kind="primary"]:hover, .gp-btn-primary:hover {
            background-color: var(--gp-blue-hover) !important;
            box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3) !important;
        }

        /* Google Photos Metric KPI Card */
        .gp-kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .gp-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        .gp-kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        .kpi-blue::before { background: var(--gp-blue); }
        .kpi-red::before { background: var(--gp-red); }
        .kpi-yellow::before { background: var(--gp-yellow); }
        .kpi-green::before { background: var(--gp-green); }
        .kpi-purple::before { background: var(--gp-purple); }

        .gp-kpi-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }
        .gp-kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--text-dark);
            line-height: 1.1;
        }
        .gp-kpi-subtext {
            font-size: 0.82rem;
            color: #64748b;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Opportunity Card (3-Column Grid) */
        .gp-opp-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.2s ease;
        }
        .gp-opp-card:hover {
            border-color: var(--gp-blue);
            box-shadow: 0 8px 24px rgba(26, 115, 232, 0.1);
            transform: translateY(-2px);
        }
        .gp-opp-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .gp-rank-pill {
            background: #1e293b;
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
        }
        .gp-opp-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
            line-height: 1.35;
        }
        .gp-opp-desc {
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.5;
            margin-bottom: 16px;
            flex-grow: 1;
        }
        .gp-opp-metrics {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #f1f5f9;
            padding-top: 12px;
            font-size: 0.8rem;
            color: #64748b;
        }

        /* Pill Badges */
        .gp-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 16px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-right: 4px;
            margin-bottom: 4px;
        }
        .pill-play_store { background: var(--gp-blue-soft); color: #1557b0; }
        .pill-reddit { background: #feefe3; color: #b03512; }
        .pill-app_store { background: var(--gp-purple-soft); color: #7627bb; }
        .pill-youtube { background: var(--gp-red-soft); color: #c5221f; }
        .pill-help_forum { background: var(--gp-green-soft); color: #137333; }
        .pill-web { background: var(--gp-teal-soft); color: #007b83; }

        .pill-high { background: var(--gp-red-soft); color: #c5221f; font-weight: 700; }
        .pill-med { background: var(--gp-yellow-soft); color: #b06000; font-weight: 700; }
        .pill-low { background: var(--gp-green-soft); color: #137333; font-weight: 700; }
        
        .pill-voiced { background: var(--gp-blue-soft); color: #1557b0; }
        .pill-inferred { background: var(--gp-purple-soft); color: #7627bb; }

        /* Pipeline Live Status Indicator */
        .pipeline-status-box {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 20px;
        }
        .status-dot {
            height: 8px;
            width: 8px;
            background-color: #34a853;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
            box-shadow: 0 0 8px #34a853;
        }

        /* Evidence Drawer Cards */
        .evidence-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--gp-blue);
            border-radius: 0 8px 8px 0;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .evidence-quote {
            font-style: italic;
            font-size: 0.9rem;
            color: #334155;
            line-height: 1.5;
            margin-bottom: 8px;
        }
        .evidence-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: #64748b;
        }

        /* Custom Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            font-size: 0.95rem;
            font-weight: 600;
            color: #64748b;
            border: none !important;
            background-color: transparent !important;
            padding: 0 4px;
        }
        .stTabs [aria-selected="true"] {
            color: var(--gp-blue) !important;
            border-bottom: 3px solid var(--gp-blue) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
