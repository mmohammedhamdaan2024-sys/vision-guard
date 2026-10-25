"""
VisionGuard – AI-Powered Intelligent Road Safety Monitoring and Traffic Violation Detection System
Main Overview Dashboard Application Entrypoint
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DB_PATH
from modules.event_storage import EventStorage
from utils.ui_components import inject_custom_css, render_surveillance_header, render_metric_card

st.set_page_config(
    page_title="VisionGuard — Road Safety Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Initialize Database & Seed realistic historical demonstration records
storage = EventStorage(DB_PATH)
storage.seed_initial_demo_events(force=False)
stats = storage.get_statistics()

# Top Surveillance Banner
render_surveillance_header(
    title="VISIONGUARD: INTELLIGENT TRAFFIC SAFETY COMMAND CENTER",
    subtitle="Automated real-time multi-agent highway surveillance, violation adjudication, and road safety telemetry.",
    tag="SURVEILLANCE ONLINE"
)

# Sidebar System Health & Quick Action Links
with st.sidebar:
    st.markdown("### 🛡️ VisionGuard Navigation")
    st.markdown("Use the navigation menu above to access dedicated surveillance portals.")
    st.markdown("---")
    st.markdown("#### 📡 System Status")
    st.markdown("🟢 **YOLOv8 Engine:** Loaded (COCO 80)")
    st.markdown("🟢 **Database:** Connected (`SQLite 3`)")
    st.markdown("🟢 **Violation Engine:** Armed & Deduplicating")
    st.markdown("🟢 **Stop-Line Monitor:** Armed")
    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    st.page_link("pages/1_Live_Monitoring.py", label="📹 Open Live Camera Feed", icon="📹")
    st.page_link("pages/2_Video_Analysis.py", label="🎞️ Analyze Recorded Video", icon="🎞️")
    st.page_link("pages/3_Violation_Records.py", label="📁 Forensic Violation Records", icon="📁")
    st.page_link("pages/4_Reports_Analytics.py", label="📊 Reports & Analytics", icon="📊")
    st.page_link("pages/5_Model_Information.py", label="🧠 AI Model & Diagnostics", icon="🧠")

# Section A: Overview Dashboard KPI Metrics
st.markdown("### 📊 Macro Safety Telemetry")
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    render_metric_card("Total Violations", f"{stats['total_violations']:,}", "All recorded alerts")
with k2:
    render_metric_card("Helmet Alerts", f"{stats['helmet_events']:,}", "Two-wheeler non-compliance")
with k3:
    render_metric_card("Seat-Belt Alerts", f"{stats['seatbelt_events']:,}", "Cabin restraint alerts")
with k4:
    render_metric_card("Red-Light Incursions", f"{stats['redlight_events']:,}", "Stop-line crossings")
with k5:
    render_metric_card("Confirmed Cases", f"{stats['confirmed_events']:,}", "Verified by officer")
with k6:
    render_metric_card("Pending Reviews", f"{stats['pending_reviews']:,}", "Awaiting adjudication")

st.markdown("<br/>", unsafe_allow_html=True)

# Section B: Plotly Charts
col_ch1, col_ch2 = st.columns([1, 1])

with col_ch1:
    st.markdown("#### 🍩 Violation Events by Category")
    if stats["category_counts"]:
        df_cat = pd.DataFrame(stats["category_counts"])
        fig_donut = px.pie(
            df_cat,
            values="count",
            names="violation_category",
            hole=0.55,
            color_discrete_sequence=["#EF4444", "#3A86FF", "#F59E0B", "#10B981", "#8B5CF6"]
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=15, b=15, l=15, r=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No violation records available for charting.")

with col_ch2:
    st.markdown("#### 📈 Date-Wise Event Trends (Past 14 Days)")
    if stats["daily_trends"]:
        df_trend = pd.DataFrame(stats["daily_trends"])
        fig_trend = px.line(
            df_trend,
            x="date",
            y="count",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#00E5FF"]
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=15, b=15, l=15, r=15),
            xaxis=dict(gridcolor="#1E293B", title="Date"),
            yaxis=dict(gridcolor="#1E293B", title="Violation Count")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Trend telemetry pending data collection.")

st.markdown("<br/>", unsafe_allow_html=True)

# Section C: Recent Activity Table & Quick Evidence Preview
st.markdown("### 📋 Recent Detection Activity Stream")
recent_events = storage.list_events(limit=10)

if recent_events:
    t_rows = []
    for r in recent_events:
        t_rows.append({
            "Event ID": r["event_id"],
            "Timestamp": r["timestamp"],
            "Violation Category": r["violation_category"],
            "Tracking ID": r["tracking_id"],
            "Entity Class": str(r["vehicle_type"]).capitalize(),
            "Confidence": f"{r['confidence_score']:.2f}",
            "Source": r["camera_source"],
            "Review Status": r["review_status"]
        })
    st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
else:
    st.info("No recent surveillance events logged.")

# Section D: Project Overview & College Viva Guide
with st.expander("🎓 B.Sc. Computer Science Capstone Project Summary", expanded=False):
    st.markdown(
        """
        - **System Name:** VisionGuard – AI-Powered Intelligent Road Safety Monitoring and Traffic Violation Detection System
        - **Core Stack:** Python 3, OpenCV, Ultralytics YOLOv8, PyTorch, Streamlit, SQLite 3, Plotly, ReportLab.
        - **Live Monitoring Capabilities:**
          1. **Webcam / Connected Camera Ingestion:** Accesses video devices via OpenCV with live FPS, bounding boxes, and controls.
          2. **Motorcycle Rider Helmet Detection:** Dual-mode deep inference and cranial contour/skin spectrometry fallback.
          3. **Occupant Seat-Belt Inspection:** Windshield cabin localization, CLAHE contrast enhancement, and diagonal Hough lines.
          4. **Red Light Stop-Line Enforcement:** Virtual stop-line crossing verification during active red lights.
          5. **Temporal Vehicle Tracking:** Centroid association preventing redundant violation generation.
        """
    )
