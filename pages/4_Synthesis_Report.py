"""
pages/4_Synthesis_Report.py — Executive Synthesis Report & Research Deliverables.
Renders markdown synthesis report, provides PDF/HTML export, and enables
one-click downloads of tagged_items.json and opportunity_areas.csv.
"""
from __future__ import annotations

import os
import json
import streamlit as st
import pandas as pd

import config
from pipeline.raw_store import RawStore
from pipeline.dashboard_data import load_synthesis_report_markdown, inject_custom_css

# Page Configuration
st.set_page_config(page_title="Synthesis Report — Discovery Engine", page_icon="📑", layout="wide")
inject_custom_css()

# Header
st.markdown(
    """
    <div style="padding: 1rem 0 1rem 0;">
        <h1 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 0.2rem;">
            📑 Executive Synthesis Report
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8;">
            Strategic research synthesis, failure funnel diagnostics, and quantified product opportunity spaces.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch report markdown
report_md = load_synthesis_report_markdown()

# ── Deliverables & Downloads Bar ──────────────────────────────────────────────
st.subheader("📦 Research Deliverables & Data Exports")

d_col1, d_col2, d_col3, d_col4 = st.columns(4)

# 1. Download Markdown Report
with d_col1:
    st.download_button(
        label="📄 Download Report (.md)",
        data=report_md.encode("utf-8"),
        file_name="google_photos_synthesis_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

# 2. PDF / Printable HTML Export
with d_col2:
    # Build clean printable HTML with print stylesheet
    html_printable = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Google Photos Retrieval Failure - Synthesis Report</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #1e293b;
                max-width: 850px;
                margin: 40px auto;
                padding: 0 20px;
            }}
            h1, h2, h3, h4 {{ color: #0f172a; margin-top: 1.5em; }}
            h1 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3em; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1.5em 0; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; font-size: 0.9em; }}
            th {{ background-color: #f1f5f9; }}
            blockquote {{ border-left: 4px solid #6366f1; margin: 1.5em 0; padding-left: 16px; color: #475569; }}
            code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
            @media print {{
                body {{ max-width: 100%; margin: 0; padding: 15mm; }}
                @page {{ margin: 15mm; }}
            }}
        </style>
    </head>
    <body>
        <div style="text-align: right; margin-bottom: 20px;">
            <button onclick="window.print()" style="padding: 8px 16px; background: #6366f1; color: white; border: none; border-radius: 6px; cursor: pointer;">
                Print to PDF / Save as PDF
            </button>
        </div>
        <div id="content">
            {report_md}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            document.getElementById('content').innerHTML = marked.parse(`{report_md.replace('`', '\\`')}`);
        </script>
    </body>
    </html>
    """
    st.download_button(
        label="🖨️ Export PDF / Printable",
        data=html_printable.encode("utf-8"),
        file_name="google_photos_synthesis_report_printable.html",
        mime="text/html",
        help="Downloads an executive HTML document that opens in your browser with 1-click 'Print to PDF' formatting.",
        use_container_width=True,
    )

# 3. Download tagged_items.json
with d_col3:
    tagged_path = config.TAGGED_JSON_PATH
    tagged_data = b"[]"
    if os.path.exists(tagged_path):
        with open(tagged_path, "rb") as f:
            tagged_data = f.read()
    else:
        # Export dynamically from SQLite if file hasn't been saved yet
        store = RawStore()
        try:
            store.export_tagged_json(tagged_path)
            with open(tagged_path, "rb") as f:
                tagged_data = f.read()
        except Exception:
            tagged_data = b"[]"

    st.download_button(
        label="📥 tagged_items.json",
        data=tagged_data,
        file_name="tagged_items.json",
        mime="application/json",
        use_container_width=True,
    )

# 4. Download opportunity_areas.csv
with d_col4:
    opp_path = config.OPPORTUNITY_CSV_PATH
    opp_data = b""
    if os.path.exists(opp_path):
        with open(opp_path, "rb") as f:
            opp_data = f.read()
    else:
        # Export dynamically from SQLite if file hasn't been saved yet
        store = RawStore()
        try:
            store.export_opportunity_csv(opp_path)
            with open(opp_path, "rb") as f:
                opp_data = f.read()
        except Exception:
            opp_data = b""

    st.download_button(
        label="📥 opportunity_areas.csv",
        data=opp_data,
        file_name="opportunity_areas.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ── Render Markdown Report ────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 32px 40px; margin-top: 10px;">
    """,
    unsafe_allow_html=True,
)

st.markdown(report_md, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
