"""
VisionGuard — Page 3: Violation Records & Forensic Adjudication
Searchable, filterable audit registry of recorded violations with high-resolution
evidence inspection and human-in-the-loop review actions (Confirm, Reject, Inconclusive).
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DB_PATH, VIOLATION_TYPES, REVIEW_STATUSES
from modules.event_storage import EventStorage
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(page_title="Violation Records | VisionGuard", page_icon="📁", layout="wide")
inject_custom_css()

storage = EventStorage(DB_PATH)

render_surveillance_header(
    title="TRAFFIC VIOLATION EVIDENCE & AUDIT REGISTRY",
    subtitle="Forensic database of detected traffic infractions, optical snapshot evidence, and officer review actions.",
    tag="DATABASE ARCHIVE"
)

# Search & Filters Bar
f1, f2, f3 = st.columns([1.5, 1, 1])

with f1:
    search_query = st.text_input("🔍 Search Event ID, Tracking ID, or Notes:", placeholder="e.g. EV-2026, TRK-101")
with f2:
    cat_filter = st.selectbox("Filter by Category:", ["All"] + VIOLATION_TYPES)
with f3:
    status_filter = st.selectbox("Filter by Review Status:", ["All"] + REVIEW_STATUSES)

events = storage.list_events(category=cat_filter, status=status_filter, search=search_query, limit=150)

st.markdown(f"**Found {len(events)} Matching Records:**")

if events:
    col_table, col_inspect = st.columns([1.6, 1.4])

    with col_table:
        for ev in events:
            ev_id = ev["event_id"]
            status = ev["review_status"]

            badge_class = "badge-approved" if status == "Confirmed" else (
                "badge-critical" if status == "Rejected" else "badge-pending"
            )

            st.markdown(
                f"""
                <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #00E5FF; font-family: 'JetBrains Mono', monospace; font-size: 1.0rem;">{ev_id}</b>
                        <span class="{badge_class}">{status}</span>
                    </div>
                    <div style="color: #FFFFFF; font-size: 0.95rem; margin-top: 4px;">
                        <b>{ev['violation_category']}</b> | Entity: {str(ev['vehicle_type']).capitalize()} [{ev['tracking_id']}]
                    </div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">
                        Timestamp: {ev['timestamp']} | Source: {ev['camera_source']} | Conf: {ev['confidence_score']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"🔍 Inspect Case #{ev_id}", key=f"btn_ev_{ev['id']}"):
                st.session_state["selected_event_id"] = ev_id

    # Detailed Inspection & Adjudication View
    with col_inspect:
        sel_ev_id = st.session_state.get("selected_event_id", events[0]["event_id"] if events else None)
        active_ev = storage.get_event_by_id(sel_ev_id) if sel_ev_id else events[0]

        if active_ev:
            st.markdown(
                f"""
                <div style="background: #161F38; border: 1px solid #3A86FF55; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                    <h3 style="color: #00E5FF; margin: 0;">CASE FILE: {active_ev['event_id']}</h3>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Tracking Reference: <b>{active_ev['tracking_id']}</b> | Status: <b>{active_ev['review_status']}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display Evidence Image
            ev_img_path = active_ev.get("evidence_path", "")
            if ev_img_path and os.path.exists(ev_img_path):
                st.image(ev_img_path, use_container_width=True, caption=f"Evidence Capture: {os.path.basename(ev_img_path)}")
            else:
                st.info("Visual evidence frame snapshot not stored on local disk.")

            st.markdown("#### ⚖️ Human-in-the-Loop Adjudication")
            st.markdown(
                f"- **Violation Type:** `{active_ev['violation_category']}`\n"
                f"- **Entity:** `{str(active_ev['vehicle_type']).capitalize()}`\n"
                f"- **Detection Confidence:** `{active_ev['confidence_score']:.2f}`\n"
                f"- **Camera Source:** `{active_ev['camera_source']}`\n"
                f"- **Detection Timestamp:** `{active_ev['timestamp']}`"
            )

            officer_notes = st.text_area(
                "Reviewer Adjudication Remarks:",
                value=active_ev.get("reviewer_notes") or "",
                placeholder="Enter justification for confirmation or rejection..."
            )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅ Confirm Violation", use_container_width=True):
                    storage.update_review_status(active_ev["event_id"], "Confirmed", officer_notes)
                    st.success("Case confirmed by officer.")
                    st.rerun()
            with b2:
                if st.button("❌ Reject / False Positive", use_container_width=True):
                    storage.update_review_status(active_ev["event_id"], "Rejected", officer_notes)
                    st.error("Case dismissed as non-violation.")
                    st.rerun()
            with b3:
                if st.button("❓ Mark Inconclusive", use_container_width=True):
                    storage.update_review_status(active_ev["event_id"], "Inconclusive", officer_notes)
                    st.warning("Marked as inconclusive due to occlusion or low lighting.")
                    st.rerun()
else:
    st.info("No violation events found matching the specified filters.")
