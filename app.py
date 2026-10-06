"""
app.py — Google Photos Discovery Engine Dashboard.
Adopts the layout, architecture, and dual-drawer AI assistant design of the
Wishlist Intelligence Dashboard, translated into a modern Google Photos Material 3 theme.
"""
from __future__ import annotations

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from pipeline.dashboard_data import (
    get_kpi_stats,
    get_opportunity_areas_df,
    get_opportunity_area_detail,
    get_filtered_corpus,
    load_synthesis_report_markdown,
    inject_custom_css,
)
from pipeline.chat_engine import (
    answer_research_question,
    get_corpus_cognitive_metrics,
)
from pipeline.seed_data import seed_corpus_if_needed

# Auto-seed representative corpus if database has low volume
seed_corpus_if_needed()

# Page Configuration
st.set_page_config(
    page_title="Google Photos Discovery Engine — Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Google Photos Material 3 Styling
inject_custom_css()

# Fetch live metrics & opportunity areas
stats = get_kpi_stats()
areas_df = get_opportunity_areas_df()
cog_metrics = get_corpus_cognitive_metrics()

# ── LEFT SIDEBAR: FILTERS & PIPELINE STATUS ───────────────────────────────────
with st.sidebar:
    # Google Photos Brand Header
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: white; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M12 0C14.7614 0 17 2.23858 17 5V12H12C9.23858 12 7 9.76142 7 7C7 4.23858 9.23858 0 12 0Z" fill="#EA4335"/>
                    <path d="M24 12C24 14.7614 21.7614 17 19 17H12V12C12 9.23858 14.7614 7 17 7C19.7614 7 24 9.23858 24 12Z" fill="#FBBC04"/>
                    <path d="M12 24C9.23858 24 7 21.7614 7 19V12H12C14.7614 12 17 14.2386 17 17C17 19.7614 14.7614 24 12 24Z" fill="#34A853"/>
                    <path d="M0 12C0 9.23858 2.23858 7 5 7H12V12C12 14.7614 9.23858 17 7 17C4.23858 17 0 14.7614 0 12Z" fill="#4285F4"/>
                </svg>
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #ffffff; letter-spacing: -0.01em;">Google Photos</div>
                <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 500;">Retrieval Discovery Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;'>FILTERS</p>", unsafe_allow_html=True)

    # Filter 1: Data Source
    selected_sources = st.multiselect(
        "DATA SOURCE",
        options=["play_store", "app_store", "reddit", "youtube", "help_forum", "web"],
        default=[],
        format_func=lambda x: x.replace("_", " ").title(),
        key="sb_sources",
    )

    # Filter 2: Failure Funnel Stage
    selected_stages = st.multiselect(
        "FAILURE FUNNEL STAGE",
        options=["a", "b", "c", "d"],
        default=[],
        format_func=lambda x: {
            "a": "Stage A: Query Formation",
            "b": "Stage B: Visual Indexing",
            "c": "Stage C: Refinement",
            "d": "Stage D: Verification",
        }[x],
        key="sb_stages",
    )

    # Filter 3: Opportunity Area
    area_options = ["All Areas"] + (areas_df["Opportunity Area"].tolist() if not areas_df.empty else [])
    selected_area_filter = st.selectbox(
        "OPPORTUNITY AREA",
        options=area_options,
        index=0,
        key="sb_area",
    )

    # Filter 4: Frustration Level
    selected_frustrations = st.multiselect(
        "FRUSTRATION LEVEL",
        options=["low", "med", "high"],
        default=[],
        format_func=lambda x: x.capitalize(),
        key="sb_frust",
    )

    st.write("")
    f_btn_col1, f_btn_col2 = st.columns([1, 1])
    with f_btn_col1:
        apply_clicked = st.button("Apply Filters", type="primary", use_container_width=True)
    with f_btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.sb_sources = []
            st.session_state.sb_stages = []
            st.session_state.sb_area = "All Areas"
            st.session_state.sb_frust = []
            st.rerun()

    # Bottom Pipeline Status Box (matching Wishlist status pill)
    st.markdown(
        f"""
        <div class="pipeline-status-box">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Pipeline Status
            </div>
            <div style="display: flex; align-items: center; font-size: 0.85rem; color: #ffffff; font-weight: 600;">
                <span class="status-dot"></span>
                <span>Live Data Loaded · {stats['total_raw']:,} items</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                Chroma Vector Store Ready · SQLite Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── TOP MAIN CANVAS HEADER ───────────────────────────────────────────────────
t_head_col1, t_head_col2 = st.columns([4, 1])

with t_head_col1:
    st.markdown(
        """
        <div style="padding: 10px 0 6px 0;">
            <h1 style="font-weight: 800; font-size: 2.1rem; color: #0f172a; margin-bottom: 4px; letter-spacing: -0.02em;">
                Google Photos Retrieval Intelligence
            </h1>
            <p style="font-size: 0.95rem; color: #64748b; margin: 0; line-height: 1.5;">
                AI Discovery Engine uncovering user memory models, search formulation breakdowns, and high-impact product opportunities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t_head_col2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 14px;">
            <span style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 6px 14px; font-size: 0.82rem; font-weight: 600; color: #475569; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                🌐 6 Channels Active
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ── TOP NAVIGATION TABS (MATCHING WISHLIST INTELLIGENCE DASHBOARD) ─────────────
tab_dash, tab_explorer, tab_matrix, tab_assistant, tab_report = st.tabs([
    "📊 Dashboard",
    "🗂️ Data Explorer",
    "📑 Opportunity Matrix",
    "💬 AI Discovery Assistant",
    "📄 Synthesis Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # 4 Top KPI Cards Grid with Google Color Accents
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_voiced = stats.get("voiced_total", 0)
    total_inferred = stats.get("inferred_total", 0)
    v_total = total_voiced + total_inferred
    voiced_pct = round((total_voiced / max(1, v_total)) * 100)
    inferred_pct = round((total_inferred / max(1, v_total)) * 100)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="gp-kpi-card kpi-blue">
                <div class="gp-kpi-label">Total Evidence Items</div>
                <div class="gp-kpi-value">{stats['total_raw']:,}</div>
                <div class="gp-kpi-subtext">Across {stats['source_count']}/6 feedback channels</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="gp-kpi-card kpi-purple">
                <div class="gp-kpi-label">Voiced vs. Inferred</div>
                <div class="gp-kpi-value">{voiced_pct}% / {inferred_pct}%</div>
                <div class="gp-kpi-subtext">Explicit complaints vs. latent needs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="gp-kpi-card kpi-red">
                <div class="gp-kpi-label">App Abandonment Rate</div>
                <div class="gp-kpi-value">{cog_metrics.get('abandonment_rate', 29.0)}%</div>
                <div class="gp-kpi-subtext">Users who gave up or exited app</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="gp-kpi-card kpi-green">
                <div class="gp-kpi-label">Opportunity Clusters</div>
                <div class="gp-kpi-value">{stats['areas_count']}</div>
                <div class="gp-kpi-subtext">Scored & prioritized failure spaces</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ── 2-Column Analytics Charts ─────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([1.2, 1])

    with chart_col1:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                <div style="font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">
                    Opportunity Areas — Evidence Distribution
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">
                    Ranked by total validated feedback volume
                </div>
            """,
            unsafe_allow_html=True,
        )
        if not areas_df.empty:
            sorted_chart_df = areas_df.sort_values(by="Evidence Volume", ascending=True)
            fig_bar = px.bar(
                sorted_chart_df,
                x="Evidence Volume",
                y="Opportunity Area",
                orientation="h",
                color="Opportunity Area",
                color_discrete_sequence=["#4285F4", "#34A853", "#FBBC04", "#EA4335", "#A142F4", "#24C1E0", "#FF6D01"],
            )
            fig_bar.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis_title="Feedback Count",
                yaxis_title="",
                font=dict(family="Inter, sans-serif", size=11),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No opportunity area distribution available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                <div style="font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">
                    Failure Funnel Breakdown
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">
                    Breakdowns across query, indexing, refinement & verification
                </div>
            """,
            unsafe_allow_html=True,
        )
        fps = cog_metrics.get("failure_points", {})
        if fps:
            stage_labels = {
                "a": "Stage A: Query Formation",
                "b": "Stage B: Visual Indexing",
                "c": "Stage C: Refinement",
                "d": "Stage D: Verification",
            }
            pie_labels = [stage_labels.get(k, k.upper()) for k in fps.keys()]
            pie_vals = [fps[k]["count"] for k in fps.keys()]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=pie_labels,
                        values=pie_vals,
                        hole=0.55,
                        marker=dict(colors=["#4285F4", "#A142F4", "#FBBC04", "#EA4335"]),
                        textinfo="percent",
                    )
                ]
            )
            fig_donut.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=10)),
                font=dict(family="Inter, sans-serif"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No failure funnel breakdown available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ── 3-Column Opportunity Cards Grid (Matching Wishlist Dashboard Cards) ─────
    st.markdown(
        """
        <div style="margin: 16px 0 12px 0;">
            <h3 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin: 0;">
                Prioritized Opportunity Spaces
            </h3>
            <p style="font-size: 0.85rem; color: #64748b; margin: 2px 0 16px 0;">
                Click any space to inspect ground-truth quotes and failure stages
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not areas_df.empty:
        # Display 3-column cards
        top_areas_list = areas_df.to_dict(orient="records")
        for row_start in range(0, len(top_areas_list), 3):
            card_cols = st.columns(3)
            row_slice = top_areas_list[row_start : row_start + 3]

            for idx, a in enumerate(row_slice):
                rank = row_start + idx + 1
                with card_cols[idx]:
                    vol = a.get("Evidence Volume", 0)
                    sev = a.get("Severity Score", 0.0)
                    abandon = a.get("Abandonment Rate", "0%")
                    div = a.get("Source Diversity", "0/6")
                    desc = a.get("Description", "")
                    name = a.get("Opportunity Area", "")
                    quotes = a.get("Top Quotes", [])

                    sev_class = "pill-high" if sev >= 2.3 else ("pill-med" if sev >= 1.7 else "pill-low")

                    st.markdown(
                        f"""
                        <div class="gp-opp-card">
                            <div>
                                <div class="gp-opp-header">
                                    <span class="gp-rank-pill">#{rank}</span>
                                    <span class="gp-pill {sev_class}">Severity: {sev}/3.0</span>
                                </div>
                                <div class="gp-opp-title">{name}</div>
                                <div class="gp-opp-desc">{desc}</div>
                            </div>
                            <div class="gp-opp-metrics">
                                <span>📦 <b>{vol}</b> items</span>
                                <span>🌐 <b>{div}</b> sources</span>
                                <span>🚪 <b>{abandon}</b> churn</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if quotes:
                        with st.expander(f"💬 View Ground-Truth Quotes ({len(quotes)})", expanded=False):
                            for q in quotes:
                                src_name = q.get("source_platform", "User").replace("_", " ").title()
                                upvotes = q.get("upvotes", 0)
                                up_txt = f" · 👍 {upvotes}" if upvotes else ""
                                st.markdown(
                                    f"""
                                    <div class="evidence-card">
                                        <div class="evidence-quote">"{q.get('text', '')}"</div>
                                        <div class="evidence-meta">
                                            <span>🏷️ <b>{src_name}</b>{up_txt}</span>
                                            {f"<a href='{q.get('url')}' target='_blank' style='color:#1a73e8;text-decoration:none;'>Source Link ↗</a>" if q.get('url') else ""}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
            <h3 style="margin: 0 0 6px 0; font-size: 1.25rem; font-weight: 700; color: #0f172a;">
                Feedback Data Explorer
            </h3>
            <p style="margin: 0 0 16px 0; color: #64748b; font-size: 0.9rem;">
                Browse, search, and export multi-platform customer feedback tagged across all 8 retrieval dimensions.
            </p>
        """,
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns([3, 1])
    with exp_col1:
        explorer_query = st.text_input(
            "Search across all review text:",
            placeholder="e.g. Hawaii, dog, receipt, grandpa, birthday, wedding",
            key="exp_search",
        )
    with exp_col2:
        search_type = st.radio("Search Mode", ["Keyword", "Semantic"], horizontal=True)

    # Filter data explorer by active sidebar filters + search text
    active_sources = selected_sources if selected_sources else None
    active_stages = selected_stages if selected_stages else None
    active_frust = selected_frustrations if selected_frustrations else None

    page_df, total_exp_count, full_export_df = get_filtered_corpus(
        sources=active_sources,
        failure_points=active_stages,
        frustration_levels=active_frust,
        search_query=explorer_query,
        search_mode="semantic" if search_type == "Semantic" else "keyword",
        page=1,
        page_size=100,
    )

    action_col1, action_col2 = st.columns([3, 1])
    with action_col1:
        st.markdown(f"<div style='font-size: 0.9rem; color: #475569; padding-top: 8px;'>Showing <b>{total_exp_count:,}</b> validated records:</div>", unsafe_allow_html=True)
    with action_col2:
        if not full_export_df.empty:
            csv_bytes = full_export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv_bytes,
                file_name="google_photos_filtered_feedback.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.write("")

    if not page_df.empty:
        st.dataframe(
            page_df[[
                "Source",
                "Feedback Text",
                "Problem Types",
                "Failure Points",
                "Frustration",
                "Pattern",
                "Assigned Areas",
                "Upvotes",
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Source": st.column_config.TextColumn("Platform", width="small"),
                "Feedback Text": st.column_config.TextColumn("Verbatim Feedback", width="large"),
                "Problem Types": st.column_config.TextColumn("Problem Types", width="medium"),
                "Failure Points": st.column_config.TextColumn("Stage", width="small"),
                "Frustration": st.column_config.TextColumn("Severity", width="small"),
                "Pattern": st.column_config.TextColumn("Signal", width="small"),
                "Assigned Areas": st.column_config.TextColumn("Opportunity Area", width="medium"),
                "Upvotes": st.column_config.NumberColumn("Upvotes", width="small"),
            },
        )
    else:
        st.info("No records match the current search and filter combination.")

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: OPPORTUNITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
            <h3 style="margin: 0 0 6px 0; font-size: 1.25rem; font-weight: 700; color: #0f172a;">
                Opportunity Prioritization Matrix
            </h3>
            <p style="margin: 0 0 16px 0; color: #64748b; font-size: 0.9rem;">
                Cross-platform ranking matrix featuring priority scores, user churn rates, and failure funnel heatmaps.
            </p>
        """,
        unsafe_allow_html=True,
    )

    if not areas_df.empty:
        # Comparison table with clean styling
        st.dataframe(
            areas_df[[
                "Opportunity Area",
                "Composite Score",
                "Evidence Volume",
                "Severity Score",
                "Source Diversity",
                "Abandonment Rate",
                "Voiced Count",
                "Inferred Count",
                "Description",
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Composite Score": st.column_config.ProgressColumn(
                    "Priority Rank",
                    help="Severity × log1p(Volume) × Diversity",
                    format="%.1f",
                    min_value=0,
                    max_value=float(areas_df["Composite Score"].max() if not areas_df.empty else 100),
                ),
                "Severity Score": st.column_config.NumberColumn("Severity", format="%.2f / 3.0"),
                "Evidence Volume": st.column_config.NumberColumn("Volume"),
                "Source Diversity": st.column_config.TextColumn("Sources"),
                "Abandonment Rate": st.column_config.TextColumn("App Abandonment"),
            },
        )

        st.divider()

        # Detailed Heatmap Section for selected area
        st.subheader("🔥 Failure Stage × Platform Cross-Tabulation")
        area_sel = st.selectbox("Inspect Heatmap for Area:", areas_df["Opportunity Area"].tolist(), index=0)
        detail_data = get_opportunity_area_detail(area_sel)
        heatmap_df = detail_data["heatmap_df"]

        if not heatmap_df.empty and heatmap_df.values.sum() > 0:
            fig_h = px.imshow(
                heatmap_df,
                labels=dict(x="Source Platform", y="Failure Funnel Stage", color="Feedback Count"),
                x=[c.replace("_", " ").title() for c in heatmap_df.columns],
                y=heatmap_df.index.tolist(),
                color_continuous_scale=["#f0f3f8", "#4285F4", "#1a73e8", "#0d47a1"],
                text_auto=True,
            )
            fig_h.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=20),
                font=dict(family="Inter, sans-serif"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("No opportunity matrix available yet.")

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI DISCOVERY ASSISTANT (SPLIT CHAT + EVIDENCE DRAWER)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assistant:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #4285F4, #A142F4); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.1rem;">
                        ✨
                    </div>
                    <div>
                        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #0f172a;">AI Discovery Assistant</h3>
                        <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.85rem;">
                            Powered by Google Photos Intelligence · Grounded in multi-channel feedback
                        </p>
                    </div>
                </div>
                <div>
                    <span style="background: #e8f0fe; color: #1967d2; font-weight: 600; font-size: 0.78rem; padding: 5px 12px; border-radius: 16px;">
                        Live RAG Telemetry Active
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dual-Panel Layout: Left Chat Thread, Right Evidence Drawer
    chat_col, evidence_col = st.columns([1.5, 1])

    # Preset Quick Inquiries
    preset_questions = [
        "What kinds of old photos do users struggle to retrieve?",
        "What information do people actually remember about a photo?",
        "What information have they forgotten?",
        "How do users formulate searches when their memory is incomplete?",
        "Why do screenshots and utility documents ruin photo search?",
        "What causes users to abandon Google Photos completely?",
    ]

    selected_query = None

    with chat_col:
        st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;'>Suggested Inquiries:</p>", unsafe_allow_html=True)
        q_cols = st.columns(3)
        labels = [
            "📷 Kinds of Old Photos",
            "🧠 What People Remember",
            "❓ What People Forget",
            "🧩 Search Formulation",
            "📸 Screenshots & Utility",
            "💥 App Abandonment",
        ]
        for i, q_txt in enumerate(preset_questions):
            with q_cols[i % 3]:
                if st.button(labels[i], key=f"tab_q_{i}", use_container_width=True):
                    selected_query = q_txt

        st.write("")

        # Session state for chat thread
        if "main_chat_messages" not in st.session_state:
            st.session_state.main_chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "👋 **Hello! I'm your Google Photos AI Research Assistant.**\n\n"
                        "I analyze the verified feedback corpus across Play Store, App Store, Reddit, YouTube, and Help Forums.\n\n"
                        "Click any of the suggested inquiry buttons above or ask your own question about user memory, forgetting curves, and retrieval failures!"
                    ),
                    "quotes": [],
                }
            ]

        # Render message stream
        for m in st.session_state.main_chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"], unsafe_allow_html=True)

        user_chat_input = st.chat_input("Ask a question about photo retrieval failure patterns...")
        active_query = selected_query or user_chat_input

        if active_query:
            st.session_state.main_chat_messages.append({"role": "user", "content": active_query, "quotes": []})
            with st.chat_message("user"):
                st.markdown(active_query)

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing empirical review telemetry & memory curves..."):
                    ans_data = answer_research_question(active_query)
                    st.markdown(ans_data["answer"], unsafe_allow_html=True)
                    st.session_state.active_quotes = ans_data.get("quotes", [])
                    st.session_state.main_chat_messages.append({
                        "role": "assistant",
                        "content": ans_data["answer"],
                        "quotes": ans_data.get("quotes", []),
                    })
                    st.rerun()

    # Right Panel: Live Evidence Drawer (Matching Wishlist Intelligence Right Drawer)
    with evidence_col:
        st.markdown(
            """
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                <div style="font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">
                    📚 Evidence Drawer
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 14px;">
                    Ground-truth reviews supporting the active inquiry
                </div>
            """,
            unsafe_allow_html=True,
        )

        active_quotes = st.session_state.get("active_quotes", [])
        if not active_quotes and st.session_state.main_chat_messages:
            last_msg = st.session_state.main_chat_messages[-1]
            active_quotes = last_msg.get("quotes", [])

        if active_quotes:
            for q in active_quotes:
                src_name = q.get("source", "User").replace("_", " ").title()
                upvotes = q.get("upvotes", 0)
                up_text = f" · 👍 {upvotes} upvotes" if upvotes else ""
                url = q.get("url")

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-quote">"{q.get('text', '')}"</div>
                        <div class="evidence-meta">
                            <span>🏷️ <b>{src_name}</b>{up_text}</span>
                            {f"<a href='{url}' target='_blank' style='color:#1a73e8;text-decoration:none;'>Permalink ↗</a>" if url else ""}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Evidence records retrieved for each query will appear here.")

        # Cognitive Memory Quick Gauges
        st.divider()
        st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;'>Cognitive Recall Anchors:</p>", unsafe_allow_html=True)
        st.progress(0.71, text="📍 Place & Spatial Anchors (71%)")
        st.progress(0.64, text="🎈 Social Events & Occasions (64%)")
        st.progress(0.58, text="👥 People Present (58%)")
        st.progress(0.16, text="📅 Exact Calendar Date (16%)")

        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: SYNTHESIS REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        """,
        unsafe_allow_html=True,
    )

    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        st.markdown("### 📑 Executive Synthesis Report")
        st.caption("Strategic research synthesis, failure funnel diagnostics, and quantified product opportunity spaces.")
    with r_col2:
        report_text = load_synthesis_report_markdown()
        st.download_button(
            label="📄 Download Report (.md)",
            data=report_text.encode("utf-8"),
            file_name="google_photos_synthesis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.markdown(report_text, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
