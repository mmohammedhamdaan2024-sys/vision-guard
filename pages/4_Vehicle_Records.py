"""
Vision Guard — Page 4: Vehicle & Evidence Records
Searchable, filterable forensic registry of scanned vehicles, violations, and evidence snapshots.
"""

import os
import streamlit as st
import pandas as pd
from database.db_manager import DatabaseManager
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Vehicle Records | Vision Guard", page_icon="📁", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")

render_surveillance_header(
    title="VEHICLE & EVIDENCE RECORDS ARCHIVE",
    subtitle="Forensic database of scanned vehicles, optical evidence crops, plate detections, and audit histories.",
    tag="DATABASE REGISTRY"
)

# Search & Filters
fcol1, fcol2, fcol3 = st.columns([1.5, 1, 1])
with fcol1:
    search_query = st.text_input("🔍 Search by Registration Plate or Tracking ID", placeholder="e.g. DL 03, TRK-101")
with fcol2:
    status_filter = st.selectbox("Filter by Review Status", ["All", "Pending Review", "Approved", "Rejected", "Under Investigation"])
with fcol3:
    type_filter = st.selectbox("Filter by Violation Category", [
        "All", "Helmet Violation", "Seat-belt Violation", "Dark Window Tint Review", "Mirror Obscured", "Plate Unreadable"
    ])

violations = db.list_violations(status=status_filter, violation_type=type_filter, limit=100)

# Filter by search
if search_query:
    q = search_query.strip().lower()
    violations = [
        v for v in violations
        if q in str(v.get("plate_number", "")).lower() or q in str(v.get("tracking_id", "")).lower()
    ]

st.markdown(f"**Found {len(violations)} matching records in registry:**")

if violations:
    # Master-detail layout
    col_list, col_detail = st.columns([1.6, 1.4])

    with col_list:
        selected_record = None
        for v in violations:
            v_id = v["id"]
            status = v["status"]
            badge_class = "badge-approved" if status == "Approved" else ("badge-critical" if status == "Rejected" else "badge-pending")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b style="color: #00E5FF; font-size: 1.05rem;">{v.get('plate_number', 'UNREADABLE')}</b>
                                <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 8px;">[{v.get('tracking_id')}]</span>
                            </div>
                            <span class="{badge_class}">{status}</span>
                        </div>
                        <div style="color: #FFFFFF; font-size: 0.9rem; margin-top: 4px;">
                            <b>{v.get('violation_type')}</b> (Conf: {v.get('confidence', 0.0):.2f})
                        </div>
                        <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">
                            Logged on: {v.get('timestamp')} | Vehicle: {str(v.get('vehicle_type', 'Vehicle')).capitalize()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"🔎 Inspect Evidence #{v_id}", key=f"btn_v_{v_id}"):
                    st.session_state["selected_violation_id"] = v_id

    # Detail View on Right
    with col_detail:
        sel_id = st.session_state.get("selected_violation_id", violations[0]["id"] if violations else None)
        active_rec = next((v for v in violations if v["id"] == sel_id), violations[0] if violations else None)

        if active_rec:
            st.markdown(
                f"""
                <div style="background: #161F38; border: 1px solid #3A86FF55; border-radius: 10px; padding: 18px;">
                    <h3 style="color: #00E5FF; margin-bottom: 4px;">CASE FILE: VIO-{active_rec['id']:04d}</h3>
                    <div style="color: #94A3B8; font-size: 0.85rem;">Tracking Unit: {active_rec['tracking_id']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br/>", unsafe_allow_html=True)

            evid_path = active_rec.get("evidence_path", "")
            if evid_path and os.path.exists(evid_path):
                st.image(evid_path, use_container_width=True, caption=f"Forensic Snapshot: {os.path.basename(evid_path)}")
            else:
                st.info("Visual evidence archive snapshot not present on local disk.")

            st.markdown("#### 📑 Case Diagnostics")
            st.markdown(f"- **Registration Plate:** `{active_rec.get('plate_number', 'UNREADABLE')}`")
            st.markdown(f"- **Vehicle Type:** `{str(active_rec.get('vehicle_type', 'Unknown')).capitalize()}`")
            st.markdown(f"- **Violation Category:** `{active_rec.get('violation_type')}`")
            st.markdown(f"- **Severity Level:** `{active_rec.get('severity')}`")
            st.markdown(f"- **Detection Confidence:** `{active_rec.get('confidence', 0.0):.2f}`")
            st.markdown(f"- **Current Review Status:** `{active_rec.get('status')}`")
            st.markdown(f"- **Recorded Notes:** {active_rec.get('review_notes', 'N/A')}")
else:
    st.info("No records match the selected filter criteria.")
