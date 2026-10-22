"""
Vision Guard — AI-Powered Traffic Violation Detection & Road Safety Monitoring System
Main Streamlit Application Entrypoint & Authentication Portal
"""

import os
import yaml
import streamlit as st
from database.db_manager import DatabaseManager
from utils.ui_components import inject_custom_css, render_surveillance_header, render_metric_card

st.set_page_config(
    page_title="Vision Guard — AI Traffic Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

# Initialize Database & Seed
db = DatabaseManager(config.get("storage", {}).get("database_path", "data/visionguard.db"))
db.seed_demo_data(force=False)

# Session Authentication State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""

inject_custom_css()

# -------------------------------------------------------------
# LOGIN PORTAL
# -------------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1C2541 0%, #0B132B 100%);
                        border: 1px solid #3A86FF55; border-radius: 14px; padding: 32px;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.5); text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 8px;">🛡️</div>
                <h1 style="color: #FFFFFF; font-size: 1.8rem; margin-bottom: 4px; font-weight: 700;">VISION GUARD</h1>
                <p style="color: #00E5FF; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-bottom: 24px;">
                    AI-POWERED TRAFFIC SURVEILLANCE & ROAD SAFETY SYSTEM
                </p>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 24px;">
                    B.Sc. Computer Science Final-Year Major Project<br/>
                    Surveillance Officer & Command Center Authentication
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("#### 🔒 Surveillance Officer Sign-In")
            username_input = st.text_input("Officer ID / Username", value="", placeholder="Enter officer username")
            password_input = st.text_input("Access Key / Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Authenticate & Access Command Center", use_container_width=True)

            if submit:
                demo_users = config.get("auth", {}).get("demo_users", {
                    "admin": {"password": "visionguard2026", "role": "Chief Surveillance Officer"},
                    "officer": {"password": "traffic123", "role": "Field Review Officer"}
                })

                if username_input in demo_users and password_input == demo_users[username_input]["password"]:
                    st.session_state.authenticated = True
                    st.session_state.username = username_input
                    st.session_state.role = demo_users[username_input]["role"]
                    st.success(f"Access Granted: Welcome, {st.session_state.role}!")
                    st.rerun()
                else:
                    st.error("Authentication Failed: Invalid Officer ID or Access Key.")

        # Quick demo credentials helper buttons
        st.markdown(
            """
            <div style="background: #161F38; border-radius: 8px; padding: 14px; margin-top: 16px; border: 1px dashed #3A86FF44;">
                <div style="color: #38BDF8; font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;">
                    DEMO ACCESS CREDENTIALS (VIVA / PRESENTATION)
                </div>
                <div style="color: #CBD5E1; font-size: 0.78rem;">
                    • <b>Administrator:</b> <code>admin</code> / <code>visionguard2026</code> (Chief Surveillance Officer)<br/>
                    • <b>Field Officer:</b> <code>officer</code> / <code>traffic123</code> (Field Review Officer)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("⚠️ Notice: For academic & demonstration evaluation only. Not connected to state databases.")

# -------------------------------------------------------------
# AUTHENTICATED COMMAND CENTER OVERVIEW
# -------------------------------------------------------------
else:
    # Sidebar HUD Diagnostics
    with st.sidebar:
        st.markdown("### 🛡️ Vision Guard")
        st.markdown(f"**Officer:** `{st.session_state.username}`")
        st.markdown(f"**Role:** `{st.session_state.role}`")
        st.markdown("---")

        st.markdown("#### 📡 System Status")
        st.markdown("🟢 **Surveillance Pipeline:** Online")
        st.markdown("🟢 **YOLOv8 Base Model:** Active (80 COCO Classes)")
        st.markdown("🟢 **EasyOCR Engine:** Ready")
        st.markdown("🟢 **SQLite Database:** Connected")
        st.markdown("🟢 **Challan Engine:** Ready")
        st.markdown("---")

        if st.button("🚪 Logout & Terminate Session", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.rerun()

    render_surveillance_header(
        title="SURVEILLANCE COMMAND CENTER OVERVIEW",
        subtitle=f"Operator: {st.session_state.username.upper()} ({st.session_state.role}) | System Status: ALL ENGINES OPERATIONAL",
        tag="LIVE SYSTEM OPERATIONAL"
    )

    metrics = db.get_dashboard_metrics()

    # Top KPI row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        render_metric_card("Vehicles Processed", f"{metrics['total_vehicles']:,}", "Analyzed in current registry")
    with mcol2:
        render_metric_card("Potential Violations", f"{metrics['total_violations']:,}", "Safety violations recorded")
    with mcol3:
        render_metric_card("Pending Officer Reviews", f"{metrics['pending_reviews']:,}", "Cases awaiting officer sign-off")
    with mcol4:
        render_metric_card("Approved Fines", f"₹{metrics['approved_fines']:,.2f}", "Verified demonstration citations")

    st.markdown("<br/>", unsafe_allow_html=True)

    # System Architecture & Navigation Guide
    st.markdown("### 🗺️ System Modules & Navigation")
    ncol1, ncol2, ncol3 = st.columns(3)

    with ncol1:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <h4 style="color: #00E5FF; margin-bottom: 8px;">📊 1. Main Dashboard</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Real-time aggregated visual telemetry, Plotly category distribution charts, 7-day trend analysis, and recent evidence snapshot gallery.
                </p>
                <hr style="border-color: #3A86FF22;"/>
                <h4 style="color: #00E5FF; margin-bottom: 8px;">📹 2. Traffic Monitoring</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Live-style camera feed simulation, video/image file ingestion, Centroid/IoU vehicle tracking HUD, and automatic violation capture.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with ncol2:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <h4 style="color: #00E5FF; margin-bottom: 8px;">🧪 3. Violation Lab</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Dedicated research sandbox for testing individual CV algorithms: Helmet Detection, Seat-belt Compliance, Window Tint, Mirror Visibility, and Plate OCR.
                </p>
                <hr style="border-color: #3A86FF22;"/>
                <h4 style="color: #00E5FF; margin-bottom: 8px;">📁 4. Vehicle & Evidence Records</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Searchable forensic archive of scanned vehicles, timestamps, recognized registration plates, and high-resolution evidence crops.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with ncol3:
        st.markdown(
            """
            <div class="metric-card" style="height: 100%;">
                <h4 style="color: #00E5FF; margin-bottom: 8px;">🧾 5. Challan Management</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Simulated e-challan adjudication workflow: approve/reject citations, generate printable PDF notices, and preview simulated SMS notifications.
                </p>
                <hr style="border-color: #3A86FF22;"/>
                <h4 style="color: #00E5FF; margin-bottom: 8px;">📈 6. Analytics & ⚙️ 7. Settings</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1;">
                    Comprehensive statistical audit reports (PDF & CSV exports), confidence threshold calibrations, and legal/ethical compliance guidelines.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # College Viva & Project Overview
    with st.expander("🎓 Project Objective & Viva Presentation Overview", expanded=False):
        st.markdown(
            """
            ### Vision Guard: Project Profile
            - **Degree:** B.Sc. Computer Science (Final-Year Major Project)
            - **Core Technology Stack:** Python 3, OpenCV, Ultralytics YOLOv8, EasyOCR, SQLite 3, Streamlit, Plotly, ReportLab.
            - **Key Scientific Innovations:**
              1. **Multi-Task Road Safety Surveillance:** Simultaneous detection of helmet compliance, occupant seat-belts, dark window tint, exterior mirror clarity, and registration plate OCR.
              2. **Intelligent Dual-Mode Inference:** Operates with specialized custom models when available, or seamlessly uses computer vision cranial morphological geometry and Hough strap profiling as fallback.
              3. **Temporal Deduplication:** Persistent vehicle tracking prevents redundant violation generation for the same vehicle across frames.
              4. **Human-in-the-Loop Decision Pipeline:** Multi-tier confidence classification ('Potential Violation', 'Compliant', 'Uncertain / Manual Review') prevents unverified citations.
            """
        )
