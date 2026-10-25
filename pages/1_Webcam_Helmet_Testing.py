"""
VisionGuard — Page 1: Personal Webcam Helmet Detection (Mode A)
Dedicated portal for testing the trained AI helmet detection model directly using a laptop webcam.
Continuously classifies: HELMET DETECTED, NO HELMET DETECTED, UNCERTAIN, or NO PERSON DETECTED.
Includes temporal frame smoothing to eliminate flickering and provides instant responsive updates
when putting on or removing a helmet.
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

from config import MODELS_DIR, HELMET_MODEL_PATH, EVIDENCE_DIR, COLORS
from modules.camera import CameraManager
from modules.helmet_detection import HelmetDetector
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(
    page_title="Webcam Helmet Testing | VisionGuard",
    page_icon="🪖",
    layout="wide"
)
inject_custom_css()

# Load Helmet Detector Engine
@st.cache_resource
def load_helmet_engine():
    return HelmetDetector(model_path=str(HELMET_MODEL_PATH), conf_threshold=0.35)

helmet_engine = load_helmet_engine()

render_surveillance_header(
    title="MODE A: PERSONAL WEBCAM HELMET DETECTION",
    subtitle="Direct optical testing portal for laptop webcams. Evaluates headgear compliance, draws high-contrast bounding boxes, and applies temporal smoothing.",
    tag="WEBCAM MODE READY"
)

# Manage Camera Session States
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "webcam_paused" not in st.session_state:
    st.session_state.webcam_paused = False
if "webcam_compliant_count" not in st.session_state:
    st.session_state.webcam_compliant_count = 0
if "webcam_violation_count" not in st.session_state:
    st.session_state.webcam_violation_count = 0

col_ctrl, col_feed = st.columns([1, 2.4])

with col_ctrl:
    st.markdown("### 🎛️ Webcam Controls")

    input_mode = st.radio(
        "Detection Input Mode:",
        ["📹 Live Laptop Webcam", "🖼️ Image Verification Bench"],
        help="Use 'Live Laptop Webcam' to test continuously with your camera. Use 'Image Verification Bench' to test with uploaded or sample photos."
    )

    if input_mode == "📹 Live Laptop Webcam":
        cam_idx = st.selectbox(
            "Select Camera Index:",
            [0, 1, 2],
            index=0,
            help="Device index for your built-in or USB webcam (usually 0)."
        )
        flip_h = st.checkbox("Mirror / Flip Camera Feed", value=True, help="Flips horizontally for a natural mirror view.")

        st.markdown("---")
        st.markdown("#### ⚡ Detection Sensitivity")
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.20,
            max_value=0.80,
            value=0.35,
            step=0.05,
            help="Higher values require stronger visual evidence; lower values increase sensitivity."
        )

        use_smoothing = st.checkbox("Temporal Smoothing", value=True, help="Smooths predictions across consecutive frames to suppress single-frame flickering.")
        window_size = st.slider("Smoothing Window (Frames)", min_value=3, max_value=9, value=5, step=2) if use_smoothing else 1

        st.markdown("---")
        # Action Buttons
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if not st.session_state.webcam_active:
                if st.button("▶️ Start Webcam", use_container_width=True, type="primary"):
                    st.session_state.webcam_active = True
                    st.session_state.webcam_paused = False
                    st.session_state.webcam_compliant_count = 0
                    st.session_state.webcam_violation_count = 0
                    helmet_engine.reset_history()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Webcam", use_container_width=True):
                    st.session_state.webcam_active = False
                    st.session_state.webcam_paused = False
                    st.rerun()

        with b_c2:
            if st.session_state.webcam_active:
                if not st.session_state.webcam_paused:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state.webcam_paused = True
                        st.rerun()
                else:
                    if st.button("▶️ Resume", use_container_width=True):
                        st.session_state.webcam_paused = False
                        st.rerun()

    else:
        st.markdown("#### 🧪 Test Image Selector")
        test_option = st.selectbox(
            "Choose a verification sample:",
            [
                "Sample: Motorcyclist With Helmet",
                "Sample: Pedestrian / Person Without Helmet",
                "Upload Custom Photo"
            ]
        )
        custom_file = None
        if test_option == "Upload Custom Photo":
            custom_file = st.file_uploader("Upload JPG/PNG photo:", type=["jpg", "jpeg", "png"])

        conf_thresh = st.slider("Confidence Threshold", 0.20, 0.80, 0.30, 0.05)
        flip_h = False
        use_smoothing = False
        window_size = 1

    # Model Status Card
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background: #1C2541; border: 1px solid #3A86FF33; border-radius: 8px; padding: 12px;">
            <div style="color: #64748B; font-size: 0.75rem; font-weight: 700;">ACTIVE HELMET MODEL</div>
            <div style="color: #00E5FF; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-top: 4px;">
                {'YOLOv8 Helmet Weights (Active)' if helmet_engine.has_model else 'Generic Weights (Fallback)'}
            </div>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 6px;">
                Classes: <code>With Helmet</code>, <code>Without Helmet</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_feed:
    st.markdown("### 🖥️ Live Optical Viewport & HUD")

    # Real-Time Telemetry Bar
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    stat_metric = m_col1.empty()
    fps_metric = m_col2.empty()
    comp_metric = m_col3.empty()
    viol_metric = m_col4.empty()

    frame_placeholder = st.empty()
    banner_placeholder = st.empty()

    if input_mode == "📹 Live Laptop Webcam":
        if st.session_state.webcam_active:
            cam_mgr = CameraManager(camera_index=cam_idx)
            success, msg = cam_mgr.start(cam_idx)

            if not success:
                st.error(f"❌ Webcam Failed to Open: {msg}")
                st.info("💡 Ensure no other application is using your webcam, check privacy settings, or try Camera Index 1.")
                st.session_state.webcam_active = False
            else:
                try:
                    # Update smoothing window
                    helmet_engine.webcam_history = collections.deque(maxlen=window_size)

                    while st.session_state.webcam_active:
                        if st.session_state.webcam_paused:
                            fps_metric.metric("FPS", "0.0")
                            stat_metric.metric("Live Status", "PAUSED")
                            time.sleep(0.1)
                            continue

                        ret, frame = cam_mgr.read_frame()
                        if not ret or frame is None:
                            st.warning("⚠️ Webcam disconnected or no video signal.")
                            break

                        if flip_h:
                            frame = cv2.flip(frame, 1)

                        # Run detection
                        res = helmet_engine.detect_personal_webcam(
                            frame, conf_threshold=conf_thresh, use_smoothing=use_smoothing
                        )

                        status = res["status"]
                        conf = res["confidence"]
                        annotated = res["annotated_frame"]

                        # Update metrics counters
                        if status == "Helmet detected":
                            st.session_state.webcam_compliant_count += 1
                        elif status == "No helmet detected":
                            st.session_state.webcam_violation_count += 1

                        # Update UI Cards
                        badge_color = "#00FF66" if status == "Helmet detected" else (
                            "#FF0055" if status == "No helmet detected" else (
                                "#FFE600" if status == "Uncertain" else "#94A3B8"
                            )
                        )
                        stat_metric.markdown(
                            f"""
                            <div style="background: #1C2541; padding: 6px 12px; border-radius: 6px; text-align: center; border-left: 3px solid {badge_color};">
                                <div style="font-size: 0.70rem; color: #64748B; font-weight: 700;">LIVE STATUS</div>
                                <div style="font-size: 0.95rem; font-weight: 700; color: {badge_color};">{status.upper()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        fps_metric.metric("FPS", f"{cam_mgr.fps:.1f}")
                        comp_metric.metric("Compliant Frames", f"{st.session_state.webcam_compliant_count}")
                        viol_metric.metric("Violation Frames", f"{st.session_state.webcam_violation_count}")

                        # Stream image to viewport
                        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        time.sleep(0.01)

                finally:
                    cam_mgr.stop()
        else:
            # Standby Screen
            stat_metric.metric("Live Status", "OFFLINE")
            fps_metric.metric("FPS", "0.0")
            comp_metric.metric("Compliant Frames", "0")
            viol_metric.metric("Violation Frames", "0")

            frame_placeholder.markdown(
                """
                <div style="background: #161F38; border: 2px dashed #3A86FF44; border-radius: 12px;
                            height: 440px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <div style="font-size: 3.5rem; margin-bottom: 8px;">🪖</div>
                    <div style="color: #FFFFFF; font-weight: 700; font-size: 1.3rem;">WEBCAM DETECTOR STANDBY</div>
                    <div style="color: #94A3B8; font-size: 0.9rem; max-width: 480px; margin-top: 8px;">
                        Click <b>'Start Webcam'</b> on the left panel to begin real-time testing.
                        The AI will track your head and continuously detect whether you are wearing a helmet or not.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        # Static Image Verification Bench
        test_img = None
        if test_option == "Sample: Motorcyclist With Helmet":
            sample_p = r"C:\Users\My PC\.cache\huggingface\hub\models--iam-tsr--yolov8n-helmet-detection\snapshots\59b68b03b60314badb0c19ebcc3b95baa8241cf8\assests\thumbnail.jpg"
            if os.path.exists(sample_p):
                test_img = cv2.imread(sample_p)
        elif test_option == "Sample: Pedestrian / Person Without Helmet":
            sample_p = "test_bus.jpg"
            if os.path.exists(sample_p):
                full_b = cv2.imread(sample_p)
                test_img = full_b[380:860, 200:360]  # unhelmeted person crop
        elif custom_file is not None:
            file_bytes = np.asarray(bytearray(custom_file.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if test_img is not None:
            res = helmet_engine.detect_personal_webcam(test_img, conf_threshold=conf_thresh, use_smoothing=False)
            status = res["status"]
            conf = res["confidence"]
            annotated = res["annotated_frame"]

            badge_color = "#00FF66" if status == "Helmet detected" else (
                "#FF0055" if status == "No helmet detected" else (
                    "#FFE600" if status == "Uncertain" else "#94A3B8"
                )
            )
            stat_metric.markdown(
                f"""
                <div style="background: #1C2541; padding: 6px 12px; border-radius: 6px; text-align: center; border-left: 3px solid {badge_color};">
                    <div style="font-size: 0.70rem; color: #64748B; font-weight: 700;">TEST RESULT</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: {badge_color};">{status.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            fps_metric.metric("Confidence", f"{conf:.2f}")
            comp_metric.metric("Mode", "Static Bench")
            viol_metric.metric("Engine", "YOLOv8")

            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("Upload a photo or select a verification sample on the left to test the helmet model.")
