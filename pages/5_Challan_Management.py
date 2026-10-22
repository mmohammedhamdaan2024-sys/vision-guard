"""
Vision Guard — Page 5: Simulated Challan Management
Mock citation adjudication, officer approvals/rejections, printable PDF generation,
and simulated mobile SMS notification preview.
"""

import os
import streamlit as st
import pandas as pd
from database.db_manager import DatabaseManager
from modules.challan_manager import ChallanManager
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Challan Management | Vision Guard", page_icon="🧾", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")
challan_mgr = ChallanManager(db, reports_dir="data/reports")

render_surveillance_header(
    title="SIMULATED E-CHALLAN ADJUDICATION PORTAL",
    subtitle="Review recorded violations, approve/reject citations, export printable PDFs, and simulate owner notification.",
    tag="CHALLAN DISPATCH"
)

# Search & Filters
f1, f2 = st.columns([1.5, 1])
with f1:
    search_term = st.text_input("🔍 Search Challan by Reference ID or Vehicle Plate", placeholder="e.g. VG-2026-CH, DL 03")
with f2:
    status_filter = st.selectbox("Filter by Adjudication Status", ["All", "Pending Review", "Approved", "Rejected", "Under Investigation"])

challans = challan_mgr.list_challans(status=status_filter, search=search_term, limit=100)

st.markdown(f"**Registry: {len(challans)} Challan Records**")

if challans:
    col_table, col_adjudicate = st.columns([1.5, 1.5])

    with col_table:
        selected_ref = None
        for c in challans:
            c_ref = c["challan_ref"]
            status = c["review_status"]
            badge_class = "badge-approved" if status == "Approved" else ("badge-critical" if status == "Rejected" else "badge-pending")

            st.markdown(
                f"""
                <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #00E5FF; font-family: 'JetBrains Mono', monospace;">{c_ref}</b>
                        <span class="{badge_class}">{status}</span>
                    </div>
                    <div style="color: #FFFFFF; font-size: 0.95rem; margin-top: 4px;">
                        <b>Vehicle:</b> {c['vehicle_plate']} | <b>Fine:</b> ₹{c['fine_amount']:,.2f}
                    </div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;">
                        Violation: {c['violation_type']} | Issued: {c['issued_at']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button(f"📄 Select {c_ref}", key=f"sel_{c['id']}"):
                st.session_state["active_challan_ref"] = c_ref

    # Adjudication & Details Panel
    with col_adjudicate:
        active_ref = st.session_state.get("active_challan_ref", challans[0]["challan_ref"] if challans else None)
        active_challan = challan_mgr.get_challan(active_ref)

        if active_challan:
            st.markdown(
                f"""
                <div style="background: #161F38; border: 1px solid #3A86FF55; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                    <h3 style="color: #00E5FF; margin: 0;">CITATION: {active_challan['challan_ref']}</h3>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Status: <b>{active_challan['review_status']}</b> | Payment: <b>{active_challan.get('payment_status', 'Unpaid')}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            evid_path = active_challan.get("evidence_path", "")
            if evid_path and os.path.exists(evid_path):
                st.image(evid_path, use_container_width=True, caption="Visual Evidence Citation Crop")

            st.markdown("#### ⚖️ Officer Adjudication Action")
            current_user = st.session_state.get("username", "Officer")
            reviewer_notes = st.text_area("Officer Review Comments:", value=active_challan.get("notes") or "", placeholder="Enter justification for decision...")

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅ Approve Citation", use_container_width=True):
                    challan_mgr.process_review_action(active_challan["id"], "Approved", current_user, reviewer_notes)
                    st.success("Challan Approved! Status updated.")
                    st.rerun()
            with b2:
                if st.button("❌ Dismiss / Reject", use_container_width=True):
                    challan_mgr.process_review_action(active_challan["id"], "Rejected", current_user, reviewer_notes)
                    st.error("Challan Dismissed.")
                    st.rerun()
            with b3:
                if st.button("🔍 Mark For Review", use_container_width=True):
                    challan_mgr.process_review_action(active_challan["id"], "Under Investigation", current_user, reviewer_notes)
                    st.warning("Marked Under Investigation.")
                    st.rerun()

            st.markdown("---")
            # PDF Export
            st.markdown("#### 🖨️ Official Notice PDF Generation")
            if st.button("📥 Generate & Download Printable PDF Challan", use_container_width=True):
                pdf_path = challan_mgr.export_pdf_challan(active_challan["challan_ref"])
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Official e-Challan Document (.pdf)",
                            data=f.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.error("Failed to compile PDF document.")

            st.markdown("---")
            # Simulated SMS Dispatch
            st.markdown("#### 📱 Simulated Mobile Notification")
            phone_input = st.text_input("Owner Contact Number (Simulation):", value="+91 98765 43210")
            if st.button("📲 Simulate Mobile SMS Notification", use_container_width=True):
                notif_res = challan_mgr.simulate_mobile_notification(active_challan["id"], phone_input)
                if notif_res["success"]:
                    st.markdown(
                        f"""
                        <div style="background: #0B132B; border: 1px solid #10B981; border-radius: 8px; padding: 14px; margin-top: 10px;">
                            <div style="color: #10B981; font-weight: 700; font-size: 0.85rem;">
                                ✔️ SIMULATED SMS PREVIEW [DELIVERED]
                            </div>
                            <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 2px;">
                                Sender: <b>{notif_res['sender_id']}</b> | To: <b>{notif_res['phone_number']}</b> | Time: {notif_res['timestamp']}
                            </div>
                            <div style="background: #1C2541; border-radius: 6px; padding: 10px; margin-top: 8px; color: #FFFFFF; font-size: 0.85rem;">
                                {notif_res['message_body']}
                            </div>
                            <div style="color: #F59E0B; font-size: 0.72rem; margin-top: 6px; font-style: italic;">
                                ⚠️ {notif_res['disclaimer']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.info("No challans match the specified filter criteria.")
