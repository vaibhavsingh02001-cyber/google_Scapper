"""
pages/2_Opportunity_Detail.py — Deep-Dive Opportunity Detail View.
Features failure funnel stage badges, failure point × platform heatmap,
voiced vs. inferred donut breakdown, and representative verbatim customer quotes.
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from pipeline.dashboard_data import (
    get_opportunity_areas_df,
    get_opportunity_area_detail,
    inject_custom_css,
)

# Page configuration
st.set_page_config(page_title="Opportunity Detail — Discovery Engine", page_icon="🔎", layout="wide")
inject_custom_css()

# Header
st.markdown(
    """
    <div style="padding: 1rem 0 1rem 0;">
        <h1 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 0.2rem;">
            🔎 Opportunity Area Detail
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8;">
            Granular inspection of retrieval breakdown stages, platform distribution, and customer verbatim quotes.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load available opportunity areas
areas_df = get_opportunity_areas_df()

if areas_df.empty:
    st.info(
        "💡 **No opportunity areas discovered yet.**\n\n"
        "Run the clustering pipeline first to populate this view:\n"
        "```bash\n"
        "python run_pipeline.py --step cluster\n"
        "```"
    )
    st.stop()

# Selector for opportunity area
area_names = areas_df["Opportunity Area"].tolist()
selected_area_name = st.selectbox(
    "Select Opportunity Area to Inspect:",
    options=area_names,
    index=0,
)

# Fetch detailed information for selected area
detail = get_opportunity_area_detail(selected_area_name)

# ── Area Header & Funnel Stage Badges ──────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 22px; margin-bottom: 20px;">
        <h2 style="margin: 0 0 10px 0; font-size: 1.6rem; color: #f8fafc;">{detail['name']}</h2>
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; margin-bottom: 16px;">
            {detail['description']}
        </p>
        <div style="margin-bottom: 8px;">
            <span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-right: 12px;">
                Active Failure Funnel Stages:
            </span>
            <span class="stage-badge {'badge-stage-a' if 'a' in detail['stages_present'] else ''}" style="opacity: {1.0 if 'a' in detail['stages_present'] else 0.3}">
                Stage A: Query Formation
            </span>
            <span class="stage-badge {'badge-stage-b' if 'b' in detail['stages_present'] else ''}" style="opacity: {1.0 if 'b' in detail['stages_present'] else 0.3}">
                Stage B: Visual Indexing
            </span>
            <span class="stage-badge {'badge-stage-c' if 'c' in detail['stages_present'] else ''}" style="opacity: {1.0 if 'c' in detail['stages_present'] else 0.3}">
                Stage C: Refinement & Browsing
            </span>
            <span class="stage-badge {'badge-stage-d' if 'd' in detail['stages_present'] else ''}" style="opacity: {1.0 if 'd' in detail['stages_present'] else 0.3}">
                Stage D: Trust & Verification
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric Badges
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evidence Volume", f"{detail['evidence_volume']} items")
m2.metric("Severity Score", f"{detail['severity_score']} / 3.0")
m3.metric("Platform Diversity", f"{detail['source_diversity']} / 6 sources")
m4.metric("Abandonment Rate", f"{detail['abandonment_rate']}%")

st.write("")

# ── Visualizations: Heatmap & Donut ───────────────────────────────────────────
viz_col1, viz_col2 = st.columns([1.3, 1])

with viz_col1:
    st.subheader("🔥 Failure Stage × Platform Distribution")
    heatmap_df = detail["heatmap_df"]

    if not heatmap_df.empty and heatmap_df.values.sum() > 0:
        fig_heat = px.imshow(
            heatmap_df,
            labels=dict(x="Source Platform", y="Failure Funnel Stage", color="Item Count"),
            x=[c.replace("_", " ").title() for c in heatmap_df.columns],
            y=heatmap_df.index.tolist(),
            color_continuous_scale="Purples",
            aspect="auto",
            text_auto=True,
        )
        fig_heat.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            font=dict(family="Inter, sans-serif"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("ℹ️ Heatmap data will automatically populate once feedback items are assigned to this cluster.")

with viz_col2:
    st.subheader("🍩 Voiced vs. Inferred Signal")
    voiced = detail["voiced_count"]
    inferred = detail["inferred_count"]

    if voiced > 0 or inferred > 0:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Voiced (Explicit Complaint)", "Inferred (Unarticulated Need)"],
                    values=[voiced, inferred],
                    hole=0.55,
                    marker=dict(colors=["#6366f1", "#10b981"]),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )
        fig_pie.update_layout(
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=20),
            height=340,
            font=dict(family="Inter, sans-serif"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>{voiced + inferred}</b><br>Total",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("ℹ️ No voiced/inferred items recorded for this area.")

st.divider()

# ── Representative Customer Quotes ────────────────────────────────────────────
st.subheader("💬 Representative Verbatim Customer Quotes")
st.caption("Ground-truth evidence tagged and ranked by engagement (upvotes/likes) to eliminate hallucination.")

quotes = detail["quotes"]

if quotes:
    # Filter and sort controls
    qc1, qc2, qc3 = st.columns([1, 1, 2])
    available_sources = sorted(list({q.get("source", q.get("source_platform", "unknown")) for q in quotes}))

    with qc1:
        source_filter = st.selectbox("Filter by Source:", options=["All Sources"] + available_sources)

    with qc2:
        sort_order = st.selectbox("Sort by Upvotes:", options=["High to Low", "Low to High"])

    with qc3:
        search_quote = st.text_input("Search quotes for keywords:", placeholder="e.g. impossible, search, date")

    # Apply filters
    filtered_quotes = quotes
    if source_filter != "All Sources":
        filtered_quotes = [
            q for q in filtered_quotes
            if q.get("source", q.get("source_platform", "")) == source_filter
        ]

    if search_quote.strip():
        sq = search_quote.strip().lower()
        filtered_quotes = [q for q in filtered_quotes if sq in q.get("text", "").lower()]

    # Sort
    filtered_quotes = sorted(
        filtered_quotes,
        key=lambda x: x.get("upvotes", 0) or 0,
        reverse=(sort_order == "High to Low"),
    )

    if filtered_quotes:
        st.write(f"Showing **{len(filtered_quotes)}** quotes:")
        for idx, q in enumerate(filtered_quotes[:15]):
            source_name = q.get("source", q.get("source_platform", "unknown")).replace("_", " ").title()
            upvotes = q.get("upvotes", 0) or 0
            url = q.get("url")
            url_link = f"<a href='{url}' target='_blank' style='color: #818cf8; text-decoration: none;'>View Source ↗</a>" if url else ""

            st.markdown(
                f"""
                <div class="quote-card">
                    <div class="quote-text">
                        "{q.get('text', '')}"
                    </div>
                    <div class="quote-meta">
                        <span>🏷️ <b>Platform:</b> {source_name}</span>
                        <span>👍 <b>Upvotes:</b> {upvotes}</span>
                        {f'<span>🔗 {url_link}</span>' if url_link else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("No quotes match your filter criteria.")
else:
    st.info("ℹ️ No verbatim quotes stored for this opportunity area yet.")
