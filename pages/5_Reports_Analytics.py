"""
VisionGuard — Page 4: Reports & Macro Safety Analytics
Provides aggregate safety telemetry, temporal violation trends, category distribution,
compliance scores, CSV registry export, and printable PDF audit report generation.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import DB_PATH, REPORTS_DIR, VIOLATION_TYPES, REVIEW_STATUSES
from modules.event_storage import EventStorage
from modules.report_generator import generate_executive_audit_pdf
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(page_title="Reports & Analytics | VisionGuard", page_icon="📊", layout="wide")
inject_custom_css()

storage = EventStorage(DB_PATH)
storage.seed_initial_demo_events()

render_surveillance_header(
    title="MACRO ROAD SAFETY TELEMETRY & AUDIT REPORTS",
    subtitle="Aggregate violation metrics, statistical compliance trends, CSV data export, and official PDF audit reporting.",
    tag="ANALYTICS ENGINE"
)

# Fetch stats and events from SQLite
stats = storage.get_statistics()
df_all = storage.export_events_df()

total_v = stats.get("total_violations", 0)
helmet_v = stats.get("helmet_events", 0)
seatbelt_v = stats.get("seatbelt_events", 0)
redlight_v = stats.get("redlight_events", 0)
confirmed_v = stats.get("confirmed_events", 0)
pending_v = stats.get("pending_reviews", 0)

conf_rate = (confirmed_v / total_v * 100) if total_v > 0 else 0.0

# -------------------------------------------------------------
# 1. Top KPI Metric Cards
# -------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL VIOLATIONS</div>
            <div class="metric-value">{total_v}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">Audited events</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">NO HELMET ALERTS</div>
            <div class="metric-value" style="color: #FF0055;">{helmet_v}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">Two-wheeler riders</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SEAT-BELT ALERTS</div>
            <div class="metric-value" style="color: #FFE600;">{seatbelt_v}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">Cabin occupants</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">RED LIGHT INCURSIONS</div>
            <div class="metric-value" style="color: #FF7B00;">{redlight_v}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">Stop-line breaches</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CONFIRMATION RATE</div>
            <div class="metric-value" style="color: #00E5FF;">{conf_rate:.1f}%</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">Human adjudication</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Interactive Analytical Visualizations
# -------------------------------------------------------------
c_left, c_right = st.columns([1.5, 1])

with c_left:
    st.markdown("### 📈 Temporal Incursion Timeline (Last 14 Days)")
    daily_trends = stats.get("daily_trends", [])
    if daily_trends:
        trend_df = pd.DataFrame(daily_trends)
        fig_trend = px.area(
            trend_df,
            x="date",
            y="count",
            markers=True,
            labels={"date": "Surveillance Date", "count": "Violations Detected"},
            color_discrete_sequence=["#00E5FF"]
        )
        fig_trend.update_layout(
            paper_bgcolor="#0B132B",
            plot_bgcolor="#1C2541",
            font_color="#E0E6ED",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor="#2A385B"),
            yaxis=dict(showgrid=True, gridcolor="#2A385B"),
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No daily temporal records available.")

with c_right:
    st.markdown("### 🍩 Violation Category Breakdown")
    cat_counts = stats.get("category_counts", [])
    if cat_counts:
        cat_df = pd.DataFrame(cat_counts)
        fig_cat = px.pie(
            cat_df,
            names="violation_category",
            values="count",
            hole=0.55,
            color_discrete_sequence=["#FF0055", "#FFE600", "#FF7B00", "#7000FF", "#00E5FF"]
        )
        fig_cat.update_layout(
            paper_bgcolor="#0B132B",
            font_color="#E0E6ED",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No category data available.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Vehicle Classes & Review Status Visuals
# -------------------------------------------------------------
v_col1, v_col2 = st.columns([1, 1])

with v_col1:
    st.markdown("### 🚗 Violations by Vehicle Classification")
    veh_counts = stats.get("vehicle_counts", [])
    if veh_counts:
        v_df = pd.DataFrame(veh_counts)
        fig_veh = px.bar(
            v_df,
            x="vehicle_type",
            y="count",
            color="vehicle_type",
            labels={"vehicle_type": "Vehicle Type", "count": "Violations"},
            color_discrete_sequence=["#4361EE", "#3A86FF", "#00E5FF", "#4CC9F0"]
        )
        fig_veh.update_layout(
            paper_bgcolor="#0B132B",
            plot_bgcolor="#1C2541",
            font_color="#E0E6ED",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#2A385B"),
            height=260
        )
        st.plotly_chart(fig_veh, use_container_width=True)
    else:
        st.info("No vehicle breakdown data.")

with v_col2:
    st.markdown("### ⚖️ Forensic Review & Adjudication Status")
    if not df_all.empty and "review_status" in df_all.columns:
        stat_df = df_all["review_status"].value_counts().reset_index()
        stat_df.columns = ["status", "count"]
        color_map = {
            "Confirmed": "#00FF66",
            "Pending Review": "#FFE600",
            "Rejected": "#FF0055",
            "Inconclusive": "#94A3B8"
        }
        fig_stat = px.bar(
            stat_df,
            x="status",
            y="count",
            color="status",
            color_discrete_map=color_map,
            labels={"status": "Adjudication Decision", "count": "Events"}
        )
        fig_stat.update_layout(
            paper_bgcolor="#0B132B",
            plot_bgcolor="#1C2541",
            font_color="#E0E6ED",
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#2A385B"),
            height=260
        )
        st.plotly_chart(fig_stat, use_container_width=True)
    else:
        st.info("No adjudication status data.")

st.markdown("---")

# -------------------------------------------------------------
# 4. Export & Printable PDF Audit Report Section
# -------------------------------------------------------------
st.markdown("### 📑 Data Export & Official Audit Report Generation")

exp_col1, exp_col2 = st.columns([1, 1])

with exp_col1:
    st.markdown(
        """
        <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 18px;">
            <h4 style="color: #00E5FF; margin-top: 0;">📊 Export Full Event Registry (CSV)</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Download complete audit trail including event identifiers, timestamps, violation categories, 
                detection confidence ratings, and officer adjudication notes for external database archiving or Excel analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Audit Registry (.CSV)",
        data=csv_bytes,
        file_name=f"VisionGuard_Violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with exp_col2:
    st.markdown(
        """
        <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 18px;">
            <h4 style="color: #00E5FF; margin-top: 0;">📑 Generate Executive Audit Report (PDF)</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Compiles an official printable multi-page Road Safety & Enforcement Audit document featuring 
                executive summary telemetry, compliance KPI tables, recent violation records, and certified audit stamps.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚙️ Compile Executive PDF Audit Report", use_container_width=True):
        events_for_pdf = storage.list_events(limit=25)
        pdf_path = generate_executive_audit_pdf(stats, events_for_pdf)
        if os.path.exists(pdf_path):
            st.success(f"Audit report generated successfully!")
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Printable Audit Report (.PDF)",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.error("Failed to compile PDF audit document.")
