"""
VisionGuard — Page 1: Live Monitoring Portal
Real-time webcam and camera feed surveillance with interactive Start/Stop/Pause controls,
dynamic YOLO bounding boxes, helmet/seatbelt analysis, and virtual stop-line monitoring.
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

from config import DB_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_STOP_LINE_Y_RATIO, COLORS, SAMPLES_DIR
from modules.camera import CameraManager
from modules.vehicle_detection import VehicleDetector
from modules.object_tracking import ObjectTracker
from modules.helmet_detection import HelmetDetector
from modules.seatbelt_detection import SeatbeltDetector
from modules.traffic_signal_detection import TrafficSignalMonitor
from modules.violation_engine import ViolationEngine
from modules.event_storage import EventStorage
from utils.image_utils import draw_hud_box
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Live Monitoring | VisionGuard", page_icon="📹", layout="wide")
inject_custom_css()

# Initialize Persistence & Engines
@st.cache_resource
def load_surveillance_engines():
    v_det = VehicleDetector(conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD)
    h_det = HelmetDetector()
    s_det = SeatbeltDetector()
    storage = EventStorage(DB_PATH)
    v_eng = ViolationEngine(storage)
    return v_det, h_det, s_det, storage, v_eng

v_detector, h_detector, s_detector, storage, v_engine = load_surveillance_engines()

render_surveillance_header(
    title="REAL-TIME LIVE CAMERA SURVEILLANCE FEED",
    subtitle="Live optical stream processing: object detection, rider helmet inspection, seatbelt analysis, and stop-line enforcement.",
    tag="LIVE FEED READY"
)

# Manage Camera Session States
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "camera_paused" not in st.session_state:
    st.session_state.camera_paused = False
if "selected_cam_idx" not in st.session_state:
    st.session_state.selected_cam_idx = 0

# Layout: Controls & Feeds
col_ctrl, col_feed = st.columns([1, 2.4])

with col_ctrl:
    st.markdown("### 🎛️ Camera & Stream Controls")

    feed_source_type = st.radio(
        "Select Video Input:",
        ["Live Hardware Camera (Webcam)", "Demo Mode (Sample Highway Video)"]
    )

    cam_index = 0
    if feed_source_type == "Live Hardware Camera (Webcam)":
        cam_index = st.selectbox("Select Camera Index:", [0, 1, 2], index=0, help="Device index for connected USB/webcam.")
    else:
        st.info("ℹ️ Running in **Demo Mode** using bundled prerecorded traffic clip.")

    st.markdown("---")
    st.markdown("#### 🎯 Detection Mode")
    detection_mode = st.selectbox(
        "Active Safety Rules:",
        [
            "All Safety Checks (Full Surveillance)",
            "Motorcycle Helmet Compliance Only",
            "Occupant Seat-Belt Compliance Only",
            "Traffic Signal & Stop-Line Incursions Only"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🚦 Stop-Line & Signal Controls")
    signal_state = st.selectbox("Current Signal State:", ["RED", "YELLOW", "GREEN"], index=0)
    stop_line_ratio = st.slider("Virtual Stop-Line Position (Y-Axis %)", min_value=0.20, max_value=0.85, value=0.65, step=0.05)

    st.markdown("---")
    # Action Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if not st.session_state.camera_active:
            if st.button("▶️ Start Camera", use_container_width=True, type="primary"):
                st.session_state.camera_active = True
                st.session_state.camera_paused = False
                st.rerun()
        else:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state.camera_active = False
                st.session_state.camera_paused = False
                st.rerun()

    with c_btn2:
        if st.session_state.camera_active:
            if not st.session_state.camera_paused:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.camera_paused = True
                    st.rerun()
            else:
                if st.button("▶️ Resume", use_container_width=True):
                    st.session_state.camera_paused = False
                    st.rerun()

    # Telemetry Card
    st.markdown("---")
    conf_slider = st.slider("Detection Confidence", min_value=0.25, max_value=0.85, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05)


with col_feed:
    st.markdown("### 🖥️ Optical Viewport & HUD")

    # Real-Time Telemetry Bar
    hud_stat_col1, hud_stat_col2, hud_stat_col3, hud_stat_col4 = st.columns(4)
    fps_metric = hud_stat_col1.empty()
    veh_metric = hud_stat_col2.empty()
    person_metric = hud_stat_col3.empty()
    viol_metric = hud_stat_col4.empty()

    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.camera_active:
        cam_mgr = CameraManager(camera_index=cam_index)
        signal_monitor = TrafficSignalMonitor(stop_line_y_ratio=stop_line_ratio, manual_signal_state=signal_state)
        tracker = ObjectTracker(max_disappeared=15, max_distance=85.0)

        # Source selection
        if feed_source_type == "Live Hardware Camera (Webcam)":
            target_source = cam_index
        else:
            target_source = str(SAMPLES_DIR / "sample_traffic_clip.mp4")

        success, msg = cam_mgr.start(target_source)

        if not success:
            st.error(f"❌ Camera Initialization Failed: {msg}")
            st.info("💡 Try changing the camera index to `1` or select 'Demo Mode (Sample Highway Video)' above.")
            st.session_state.camera_active = False
        else:
            live_violations_count = 0
            recent_alerts = []

            try:
                while st.session_state.camera_active:
                    if st.session_state.camera_paused:
                        status_msg = "⏸️ Feed Paused by Operator"
                        fps_metric.metric("FPS", "0.0")
                        time.sleep(0.1)
                        continue

                    ret, frame = cam_mgr.read_frame()
                    if not ret or frame is None:
                        # Loop sample video in demo mode
                        if feed_source_type != "Live Hardware Camera (Webcam)":
                            cam_mgr.start(target_source)
                            continue
                        else:
                            st.warning("⚠️ End of video stream or camera disconnected.")
                            break

                    h, w = frame.shape[:2]
                    signal_monitor.set_signal_state(signal_state)
                    signal_monitor.set_stop_line_ratio(stop_line_ratio)

                    # 1. Detect objects with YOLOv8
                    raw_dets = v_detector.detect(frame, conf=conf_slider)

                    # 2. Track objects across frames
                    tracked_objects = tracker.update(raw_dets)

                    # Counters
                    v_count = sum(1 for d in raw_dets if d["class_name"] in ("car", "motorcycle", "bus", "truck"))
                    p_count = sum(1 for d in raw_dets if d["class_name"] == "person")

                    annotated = frame.copy()

                    # Render Stop-Line & Signal HUD if traffic signal mode active
                    if "Traffic Signal" in detection_mode or "All Safety" in detection_mode:
                        annotated = signal_monitor.draw_hud(annotated)

                    # 3. Process Safety Rules for Each Tracked Object
                    for obj in tracked_objects:
                        x1, y1, x2, y2 = obj.bbox
                        vtype = obj.class_name

                        # Base vehicle HUD box
                        annotated = draw_hud_box(
                            annotated, obj.bbox, vtype.upper(),
                            conf=obj.conf, track_id=obj.track_id,
                            is_violation=len(obj.violations_flagged) > 0
                        )

                        # A. Motorcycle Rider Helmet Detection
                        h_res = None
                        if vtype == "motorcycle" and ("Helmet" in detection_mode or "All Safety" in detection_mode):
                            h_res = h_detector.analyze(frame, obj.bbox, raw_dets)
                            if h_res.get("head_box"):
                                hx1, hy1, hx2, hy2 = [int(v) for v in h_res["head_box"]]
                                box_color = h_res.get("box_color", COLORS["uncertain"])
                                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), box_color, 2)
                                cv2.putText(annotated, f"{h_res['status']} ({h_res['confidence']:.2f})",
                                            (hx1, max(15, hy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1)

                        # B. Occupant Seat-Belt Detection
                        s_res = None
                        if vtype in ("car", "truck", "bus") and ("Seat-Belt" in detection_mode or "All Safety" in detection_mode):
                            s_res = s_detector.analyze(frame, obj.bbox)
                            if s_res.get("cabin_box"):
                                cx1, cy1, cx2, cy2 = [int(v) for v in s_res["cabin_box"]]
                                box_color = s_res.get("box_color", COLORS["uncertain"])
                                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), box_color, 2)
                                cv2.putText(annotated, f"{s_res['status']} ({s_res['confidence']:.2f})",
                                            (cx1, max(18, cy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1)

                        # C. Red Light Stop-Line Crossing
                        sig_result = None
                        if ("Traffic Signal" in detection_mode or "All Safety" in detection_mode) and len(obj.history) >= 2:
                            prev_cent = obj.history[-2]
                            curr_cent = obj.history[-1]
                            sig_result = signal_monitor.check_stop_line_crossing(h, prev_cent, curr_cent, obj.bbox)

                        # D. Evaluate Violations via Engine (Deduplicated)
                        new_viols = v_engine.evaluate(
                            frame=frame,
                            tracked_object=obj,
                            helmet_result=h_res,
                            seatbelt_result=s_res,
                            traffic_signal_result=sig_result,
                            camera_source="Live Camera" if feed_source_type == "Live Hardware Camera (Webcam)" else "Demo Stream"
                        )

                        if new_viols:
                            live_violations_count += len(new_viols)
                            for v in new_viols:
                                recent_alerts.insert(0, f"🚨 [{v['category']}] ID: {obj.track_id} | {v['notes']}")

                    # Update Real-time Telemetry Metrics
                    fps_metric.metric("FPS", f"{cam_mgr.fps:.1f}")
                    veh_metric.metric("Vehicles Detected", f"{v_count}")
                    person_metric.metric("People Detected", f"{p_count}")
                    viol_metric.metric("Violations Logged", f"{live_violations_count}")

                    # Display Frame in Streamlit
                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if recent_alerts:
                        alert_placeholder.markdown("**Latest Real-Time Alerts:**\n" + "\n".join([f"- {a}" for a in recent_alerts[:4]]))

                    time.sleep(0.01)

            finally:
                cam_mgr.stop()
    else:
        # Standby Screen
        fps_metric.metric("FPS", "0.0")
        veh_metric.metric("Vehicles Detected", "0")
        person_metric.metric("People Detected", "0")
        viol_metric.metric("Violations Logged", "0")

        frame_placeholder.markdown(
            """
            <div style="background: #161F38; border: 2px dashed #3A86FF44; border-radius: 12px;
                        height: 420px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <div style="font-size: 3.5rem; margin-bottom: 8px;">📹</div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.3rem;">CAMERA FEED OFFLINE</div>
                <div style="color: #94A3B8; font-size: 0.9rem; max-width: 450px; margin-top: 8px;">
                    Select your preferred video input on the left panel (Live Webcam or Demo Mode) and click <b>'Start Camera'</b> to initiate real-time AI road safety detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
