"""
pages/1_Overview.py — Opportunity Areas Overview & Comparison.
Features sortable comparison matrix, evidence volume bar charts, and voiced vs. inferred breakdown.
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from pipeline.dashboard_data import get_kpi_stats, get_opportunity_areas_df, inject_custom_css

# Page Configuration
st.set_page_config(page_title="Overview — Discovery Engine", page_icon="📊", layout="wide")
inject_custom_css()

# Header
st.markdown(
    """
    <div style="padding: 1rem 0 1.5rem 0;">
        <h1 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 0.2rem;">
            📊 Opportunity Areas Overview
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8;">
            Comparative analysis of Google Photos retrieval failure spaces ranked across volume, severity, and platform diversity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch stats and opportunity areas
stats = get_kpi_stats()
areas_df = get_opportunity_areas_df()

# 4 KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="gp-kpi-card kpi-blue">
            <div class="gp-kpi-label">Total Evidence Items</div>
            <div class="gp-kpi-value">{stats["total_raw"]:,}</div>
            <div class="gp-kpi-subtext">All feedback collected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="gp-kpi-card kpi-purple">
            <div class="gp-kpi-label">Domain Relevant Items</div>
            <div class="gp-kpi-value">{stats["total_relevant"]:,}</div>
            <div class="gp-kpi-subtext">Verified retrieval failures</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="gp-kpi-card kpi-yellow">
            <div class="gp-kpi-label">Sources Covered</div>
            <div class="gp-kpi-value">{stats["source_count"]}/{stats["total_sources_possible"]}</div>
            <div class="gp-kpi-subtext">Independent channels</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="gp-kpi-card kpi-green">
            <div class="gp-kpi-label">Opportunity Clusters</div>
            <div class="gp-kpi-value">{stats["areas_count"]}</div>
            <div class="gp-kpi-subtext">Prioritized failure spaces</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if areas_df.empty:
    st.info(
        "💡 **No opportunity areas discovered yet.**\n\n"
        "Run the collection and pipeline clustering first to populate this view:\n"
        "```bash\n"
        "python run_collectors.py --sources all --limit 200\n"
        "python run_pipeline.py --step all\n"
        "```"
    )
    st.stop()

# ── Visualizations ────────────────────────────────────────────────────────────
st.subheader("📈 Evidence Volume & Pattern Distribution")
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    # Bar Chart: Evidence volume per opportunity area
    fig_vol = px.bar(
        areas_df.sort_values(by="Evidence Volume", ascending=True),
        x="Evidence Volume",
        y="Opportunity Area",
        orientation="h",
        title="<b>Evidence Volume per Opportunity Area</b>",
        labels={"Evidence Volume": "Feedback Count", "Opportunity Area": ""},
        color="Evidence Volume",
        color_continuous_scale=["#6366f1", "#a855f7", "#ec4899"],
    )
    fig_vol.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=20, t=40, b=20),
        height=380,
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with chart_col2:
    # Stacked Bar: Voiced vs. Inferred breakdown
    fig_stacked = go.Figure()
    sorted_df = areas_df.sort_values(by="Evidence Volume", ascending=True)

    fig_stacked.add_trace(
        go.Bar(
            y=sorted_df["Opportunity Area"],
            x=sorted_df["Voiced Count"],
            name="Voiced (Explicit Complaint)",
            orientation="h",
            marker=dict(color="#3b82f6"),
        )
    )
    fig_stacked.add_trace(
        go.Bar(
            y=sorted_df["Opportunity Area"],
            x=sorted_df["Inferred Count"],
            name="Inferred (Unarticulated Need)",
            orientation="h",
            marker=dict(color="#10b981"),
        )
    )

    fig_stacked.update_layout(
        barmode="stack",
        template="plotly_dark",
        title="<b>Voiced vs. Inferred Breakdown</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=20, t=40, b=20),
        height=380,
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

st.divider()

# ── Sortable Opportunity Comparison Table ─────────────────────────────────────
st.subheader("📑 Opportunity Area Comparison Matrix")

# Controls for sorting and filtering
ctrl_col1, ctrl_col2 = st.columns([2, 1])

with ctrl_col1:
    sort_by = st.selectbox(
        "Sort areas by:",
        options=[
            "Composite Score (Recommended: Severity × Volume × Diversity)",
            "Evidence Volume (Total items)",
            "Severity Score (Frustration intensity)",
            "Source Diversity (Cross-platform spread)",
            "Abandonment Rate (User churn signal)",
        ],
        index=0,
    )

with ctrl_col2:
    search_term = st.text_input("Filter table by name or keyword:", placeholder="e.g. search, backup, screenshot")

# Map selected sort to column
sort_mapping = {
    "Composite Score (Recommended: Severity × Volume × Diversity)": ("Composite Score", False),
    "Evidence Volume (Total items)": ("Evidence Volume", False),
    "Severity Score (Frustration intensity)": ("Severity Score", False),
    "Source Diversity (Cross-platform spread)": ("Diversity Raw", False),
    "Abandonment Rate (User churn signal)": ("Abandonment Raw", False),
}

sort_col, ascending = sort_mapping[sort_by]
display_df = areas_df.copy()

if search_term.strip():
    kw = search_term.strip().lower()
    display_df = display_df[
        display_df["Opportunity Area"].str.lower().str.contains(kw)
        | display_df["Description"].str.lower().str.contains(kw)
    ]

display_df = display_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

# Select columns to display in table
columns_to_show = [
    "Opportunity Area",
    "Composite Score",
    "Evidence Volume",
    "Severity Score",
    "Source Diversity",
    "Abandonment Rate",
    "Voiced Count",
    "Inferred Count",
    "Description",
]

st.dataframe(
    display_df[columns_to_show],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Composite Score": st.column_config.ProgressColumn(
            "Priority Score",
            help="severity × log1p(volume) × diversity",
            format="%.1f",
            min_value=0,
            max_value=float(areas_df["Composite Score"].max() if not areas_df.empty else 100),
        ),
        "Severity Score": st.column_config.NumberColumn(
            "Severity",
            help="1.0 = Low, 2.0 = Medium, 3.0 = High frustration",
            format="%.2f / 3.0",
        ),
        "Evidence Volume": st.column_config.NumberColumn(
            "Volume",
            help="Total tagged feedback items",
            format="%d",
        ),
        "Source Diversity": st.column_config.TextColumn(
            "Diversity",
            help="Number of distinct platforms where this failure occurs",
        ),
        "Abandonment Rate": st.column_config.TextColumn(
            "App Abandonment",
            help="Fraction of users reporting they stopped using search or left the app",
        ),
    },
)

# Deep dive link hint
st.caption("💡 Select any opportunity area above and navigate to **2. Opportunity Detail** in the sidebar for full failure heatmaps and verbatim quotes.")
