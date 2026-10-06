"""
pages/5_AI_Chatbot.py — AI Discovery Research Assistant ("Ask the Corpus").
Features a Google Photos Material 3 dual-panel layout with chat thread, quick inquiry chips,
and a live Evidence Drawer showing verbatim customer reviews with platform and upvote citations.
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pipeline.dashboard_data import inject_custom_css
from pipeline.chat_engine import (
    answer_research_question,
    get_corpus_cognitive_metrics,
)

# Page Configuration
st.set_page_config(
    page_title="AI Research Assistant — Google Photos",
    page_icon="💬",
    layout="wide",
)
inject_custom_css()

# Fetch cognitive metrics
metrics = get_corpus_cognitive_metrics()

# ── Sidebar: Telemetry & Memory Charts ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: white; display: flex; align-items: center; justify-content: center;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M12 0C14.7614 0 17 2.23858 17 5V12H12C9.23858 12 7 9.76142 7 7C7 4.23858 9.23858 0 12 0Z" fill="#EA4335"/>
                    <path d="M24 12C24 14.7614 21.7614 17 19 17H12V12C12 9.23858 14.7614 7 17 7C19.7614 7 24 9.23858 24 12Z" fill="#FBBC04"/>
                    <path d="M12 24C9.23858 24 7 21.7614 7 19V12H12C14.7614 12 17 14.2386 17 17C17 19.7614 14.7614 24 12 24Z" fill="#34A853"/>
                    <path d="M0 12C0 9.23858 2.23858 7 5 7H12V12C12 14.7614 9.23858 17 7 17C4.23858 17 0 14.7614 0 12Z" fill="#4285F4"/>
                </svg>
            </div>
            <div>
                <div style="font-weight: 700; color: white; font-size: 0.95rem;">Memory Telemetry</div>
                <div style="font-size: 0.72rem; color: #94a3b8;">Cognitive Recall Distribution</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Remembered Cues Chart
    rem_data = metrics.get("remembered", {})
    if rem_data:
        rem_labels = [k.replace("_", " ").title() for k in rem_data.keys()][:5]
        rem_vals = [rem_data[k]["pct"] for k in rem_data.keys()][:5]

        fig_rem = px.bar(
            x=rem_vals,
            y=rem_labels,
            orientation="h",
            labels={"x": "% of Users Recalling", "y": ""},
            title="<b>What Users Remember</b>",
            color=rem_vals,
            color_continuous_scale=["#4285F4", "#1a73e8"],
        )
        fig_rem.update_layout(
            template="plotly_dark",
            margin=dict(l=5, r=10, t=35, b=20),
            height=210,
            coloraxis_showscale=False,
            font=dict(size=10, family="Inter, sans-serif"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rem, use_container_width=True)

    # Forgotten Cues Chart
    forg_data = metrics.get("forgotten", {})
    if forg_data:
        forg_labels = [k.replace("_", " ").title() for k in forg_data.keys()][:4]
        forg_vals = [forg_data[k]["pct"] for k in forg_data.keys()][:4]

        fig_forg = px.bar(
            x=forg_vals,
            y=forg_labels,
            orientation="h",
            labels={"x": "% of Users Forgetting", "y": ""},
            title="<b>What Users Forget</b>",
            color=forg_vals,
            color_continuous_scale=["#EA4335", "#c5221f"],
        )
        fig_forg.update_layout(
            template="plotly_dark",
            margin=dict(l=5, r=10, t=35, b=20),
            height=190,
            coloraxis_showscale=False,
            font=dict(size=10, family="Inter, sans-serif"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_forg, use_container_width=True)

    # Search Formulation Distribution
    sm_data = metrics.get("search_methods", {})
    if sm_data:
        sm_labels = [k.replace("_", " ").title() for k in sm_data.keys()][:4]
        sm_vals = [sm_data[k]["count"] for k in sm_data.keys()][:4]

        fig_sm = go.Figure(
            data=[
                go.Pie(
                    labels=sm_labels,
                    values=sm_vals,
                    hole=0.55,
                    marker=dict(colors=["#4285F4", "#34A853", "#FBBC04", "#EA4335"]),
                )
            ]
        )
        fig_sm.update_layout(
            template="plotly_dark",
            title="<b>Search Formulation Strategy</b>",
            showlegend=True,
            legend=dict(orientation="v", font=dict(size=9)),
            margin=dict(l=5, r=5, t=35, b=10),
            height=210,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sm, use_container_width=True)

    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.chatbot_page_messages = []
        st.rerun()

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #4285F4, #A142F4); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.1rem;">
                    ✨
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #0f172a;">AI Discovery Assistant</h2>
                    <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.88rem;">
                        Analyze memory models, forgetting curves, and retrieval failure patterns across multi-platform feedback.
                    </p>
                </div>
            </div>
            <div>
                <span style="background: #e8f0fe; color: #1967d2; font-weight: 600; font-size: 0.78rem; padding: 5px 12px; border-radius: 16px;">
                    Cognitive Telemetry Active
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Dual-Panel Layout (Left Chat Thread + Right Evidence Drawer) ──────────────
chat_col, evidence_col = st.columns([1.5, 1])

preset_questions = [
    "What kinds of old photos do users struggle to retrieve?",
    "What information do people actually remember about a photo?",
    "What information have they forgotten?",
    "How do users formulate searches when their memory is incomplete?",
    "Why do screenshots and utility documents ruin photo search?",
    "What causes users to abandon Google Photos completely?",
]

labels_map = {
    0: "📷 Kinds of Old Photos",
    1: "🧠 What People Remember",
    2: "❓ What People Forget",
    3: "🧩 Search Formulation",
    4: "📸 Screenshots & Utility",
    5: "💥 App Abandonment",
}

selected_query = None

with chat_col:
    st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 8px;'>Suggested Inquiries:</p>", unsafe_allow_html=True)
    q_cols = st.columns(3)
    for i, q_txt in enumerate(preset_questions):
        with q_cols[i % 3]:
            if st.button(labels_map.get(i, q_txt), key=f"p5_q_{i}", use_container_width=True):
                selected_query = q_txt

    st.write("")

    if "chatbot_page_messages" not in st.session_state:
        st.session_state.chatbot_page_messages = [
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

    for m in st.session_state.chatbot_page_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question about photo retrieval failures...")
    active_query = selected_query or user_input

    if active_query:
        st.session_state.chatbot_page_messages.append({"role": "user", "content": active_query, "quotes": []})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing empirical review telemetry & memory curves..."):
                ans_data = answer_research_question(active_query)
                st.markdown(ans_data["answer"], unsafe_allow_html=True)
                st.session_state.active_quotes = ans_data.get("quotes", [])
                st.session_state.chatbot_page_messages.append({
                    "role": "assistant",
                    "content": ans_data["answer"],
                    "quotes": ans_data.get("quotes", []),
                })
                st.rerun()

# ── Right Evidence Drawer ─────────────────────────────────────────────────────
with evidence_col:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
            <div style="font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">
                📚 Evidence Drawer
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 14px;">
                Ground-truth customer quotes supporting the active inquiry
            </div>
        """,
        unsafe_allow_html=True,
    )

    active_quotes = st.session_state.get("active_quotes", [])
    if not active_quotes and st.session_state.chatbot_page_messages:
        last_msg = st.session_state.chatbot_page_messages[-1]
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

    # Cognitive Memory Recall Anchors
    st.divider()
    st.markdown("<p style='font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;'>Cognitive Recall Anchors:</p>", unsafe_allow_html=True)
    st.progress(0.71, text="📍 Place & Spatial Anchors (71%)")
    st.progress(0.64, text="🎈 Social Events & Occasions (64%)")
    st.progress(0.58, text="👥 People Present (58%)")
    st.progress(0.16, text="📅 Exact Calendar Date (16%)")

    st.markdown("</div>", unsafe_allow_html=True)
