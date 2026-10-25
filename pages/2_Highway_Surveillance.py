"""
VisionGuard — Page 2: Highway & Road Safety Surveillance (Mode B)
Multi-agent highway and intersection surveillance portal for connected CCTV feeds,
road cameras, and highway video streams.
Features:
- Multi-vehicle classification & persistent tracking (ByteTrack / Centroid)
- Multi-motorcycle & rider spatial association with helmet compliance inspection
- Cabin occupant seat-belt detection
- Configurable virtual stop-line & traffic signal state enforcement
- Real-time violation engine with duplicate-event suppression and SQLite storage
"""

import os
import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image

from config import (
    DB_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_STOP_LINE_Y_RATIO,
    COLORS, SAMPLES_DIR, HELMET_MODEL_PATH
)
from modules.camera import CameraManager
from modules.vehicle_detection import VehicleDetector
from modules.object_tracking import ObjectTracker
from modules.helmet_detection import HelmetDetector
from modules.seatbelt_detection import SeatbeltDetector
from modules.traffic_signal_detection import TrafficSignalMonitor
from modules.violation_engine import ViolationEngine
from modules.event_storage import EventStorage
from utils.image_utils import draw_hud_box
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(
    page_title="Highway Surveillance | VisionGuard",
    page_icon="🛣️",
    layout="wide"
)
inject_custom_css()

# Initialize Persistence & Multi-Stage Surveillance Engines
@st.cache_resource
def load_highway_engines():
    v_det = VehicleDetector(conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD)
    h_det = HelmetDetector(model_path=str(HELMET_MODEL_PATH))
    s_det = SeatbeltDetector()
    storage = EventStorage(DB_PATH)
    v_eng = ViolationEngine(storage)
    return v_det, h_det, s_det, storage, v_eng

v_detector, h_detector, s_detector, storage, v_engine = load_highway_engines()

render_surveillance_header(
    title="MODE B: HIGHWAY & INTERSECTION TRAFFIC SURVEILLANCE",
    subtitle="Automated CCTV stream analysis: multi-vehicle tracking, rider-motorcycle association, helmet enforcement, seatbelts, and red-light stop-line incursions.",
    tag="HIGHWAY SURVEILLANCE READY"
)

# Manage Highway Session States
if "hwy_active" not in st.session_state:
    st.session_state.hwy_active = False
if "hwy_paused" not in st.session_state:
    st.session_state.hwy_paused = False

col_ctrl, col_feed = st.columns([1, 2.4])

with col_ctrl:
    st.markdown("### 🎛️ Highway Stream Controls")

    feed_source_type = st.radio(
        "Surveillance Video Input:",
        ["Bundled Highway Footage (Demo)", "External USB / IP Traffic Camera", "Upload Custom Video"],
        help="Select video source for highway surveillance analysis."
    )

    custom_video_path = None
    cam_index = 0

    if feed_source_type == "External USB / IP Traffic Camera":
        cam_index = st.selectbox("Camera Hardware Index:", [0, 1, 2], index=0)
    elif feed_source_type == "Upload Custom Video":
        up_file = st.file_uploader("Upload Traffic MP4/AVI:", type=["mp4", "avi", "mov"])
        if up_file:
            temp_path = os.path.join("samples", f"temp_upload_{up_file.name}")
            with open(temp_path, "wb") as f:
                f.write(up_file.read())
            custom_video_path = temp_path
    else:
        st.info("ℹ️ Using verified bundled highway intersection recording (`sample_traffic_clip.mp4`).")

    st.markdown("---")
    st.markdown("#### 🎯 Active Surveillance Rules")
    enforce_helmet = st.checkbox("Motorcycle Rider Helmet Compliance", value=True)
    enforce_seatbelt = st.checkbox("Vehicle Occupant Seat-Belt Analysis", value=True)
    enforce_stopline = st.checkbox("Traffic Signal & Red-Light Incursions", value=True)

    st.markdown("---")
    st.markdown("#### 🚦 Signal & Geometry Calibration")
    signal_state = st.selectbox("Active Traffic Light State:", ["RED", "YELLOW", "GREEN"], index=0)
    stop_line_ratio = st.slider("Virtual Stop-Line Position (Y %)", 0.20, 0.85, 0.65, 0.05)
    conf_slider = st.slider("YOLO Detection Confidence", 0.25, 0.85, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    # Action Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if not st.session_state.hwy_active:
            if st.button("▶️ Start Surveillance", use_container_width=True, type="primary"):
                st.session_state.hwy_active = True
                st.session_state.hwy_paused = False
                st.rerun()
        else:
            if st.button("⏹️ Stop Stream", use_container_width=True):
                st.session_state.hwy_active = False
                st.session_state.hwy_paused = False
                st.rerun()

    with c_btn2:
        if st.session_state.hwy_active:
            if not st.session_state.hwy_paused:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.hwy_paused = True
                    st.rerun()
            else:
                if st.button("▶️ Resume", use_container_width=True):
                    st.session_state.hwy_paused = False
                    st.rerun()

with col_feed:
    st.markdown("### 🖥️ Surveillance Viewport & Telemetry HUD")

    # Real-Time Telemetry Cards
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    fps_metric = stat_c1.empty()
    veh_metric = stat_c2.empty()
    person_metric = stat_c3.empty()
    viol_metric = stat_c4.empty()

    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.hwy_active:
        cam_mgr = CameraManager(camera_index=cam_index)
        signal_monitor = TrafficSignalMonitor(stop_line_y_ratio=stop_line_ratio, manual_signal_state=signal_state)
        tracker = ObjectTracker(max_disappeared=15, max_distance=85.0)

        # Determine target source
        if feed_source_type == "External USB / IP Traffic Camera":
            target_source = cam_index
        elif feed_source_type == "Upload Custom Video" and custom_video_path:
            target_source = custom_video_path
        else:
            target_source = str(SAMPLES_DIR / "sample_traffic_clip.mp4")

        success, msg = cam_mgr.start(target_source)

        if not success:
            st.error(f"❌ Failed to Initialize Video Source: {msg}")
            st.session_state.hwy_active = False
        else:
            live_violations_count = 0
            recent_alerts = []

            try:
                while st.session_state.hwy_active:
                    if st.session_state.hwy_paused:
                        fps_metric.metric("FPS", "0.0")
                        time.sleep(0.1)
                        continue

                    ret, frame = cam_mgr.read_frame()
                    if not ret or frame is None:
                        # Auto-loop sample video
                        if feed_source_type == "Bundled Highway Footage (Demo)":
                            cam_mgr.start(target_source)
                            continue
                        else:
                            st.warning("⚠️ Stream finished or camera disconnected.")
                            break

                    h, w = frame.shape[:2]
                    signal_monitor.set_signal_state(signal_state)
                    signal_monitor.set_stop_line_ratio(stop_line_ratio)

                    # 1. Detect objects with YOLOv8
                    raw_dets = v_detector.detect(frame, conf=conf_slider)

                    # 2. Track objects across frames
                    tracked_objects = tracker.update(raw_dets)

                    # Global helmet scanner on frame
                    global_helmets = h_detector.detect_global_helmets(frame, conf_threshold=0.22) if enforce_helmet else []

                    # Counters
                    v_count = sum(1 for d in raw_dets if d["class_name"] in ("car", "motorcycle", "bus", "truck"))
                    p_count = sum(1 for d in raw_dets if d["class_name"] == "person")

                    annotated = frame.copy()

                    # Render Stop-Line & Signal HUD
                    if enforce_stopline:
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
                        if vtype == "motorcycle" and enforce_helmet:
                            h_res = h_detector.analyze_highway_rider(
                                frame=frame,
                                vehicle_box=obj.bbox,
                                all_detections=raw_dets,
                                frame_helmet_dets=global_helmets
                            )
                            if h_res.get("head_box"):
                                hx1, hy1, hx2, hy2 = [int(v) for v in h_res["head_box"]]
                                box_color = h_res.get("box_color", COLORS["uncertain"])
                                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), box_color, 2)
                                cv2.putText(
                                    annotated, f"Rider: {h_res['status']} ({h_res['confidence']:.2f})",
                                    (hx1, max(15, hy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1
                                )

                        # B. Occupant Seat-Belt Detection
                        s_res = None
                        if vtype in ("car", "truck", "bus") and enforce_seatbelt:
                            s_res = s_detector.analyze(frame, obj.bbox)
                            if s_res.get("cabin_box"):
                                cx1, cy1, cx2, cy2 = [int(v) for v in s_res["cabin_box"]]
                                box_color = s_res.get("box_color", COLORS["uncertain"])
                                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), box_color, 2)
                                cv2.putText(
                                    annotated, f"{s_res['status']} ({s_res['confidence']:.2f})",
                                    (cx1, max(18, cy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1
                                )

                        # C. Red Light Stop-Line Crossing
                        sig_result = None
                        if enforce_stopline and len(obj.history) >= 2:
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
                            camera_source="Highway Surveillance"
                        )

                        if new_viols:
                            live_violations_count += len(new_viols)
                            for v in new_viols:
                                recent_alerts.insert(0, f"🚨 [{v['category']}] ID: {obj.track_id} | {v['notes']}")

                    # Update Real-time Telemetry Metrics
                    fps_metric.metric("FPS", f"{cam_mgr.fps:.1f}")
                    veh_metric.metric("Vehicles Tracked", f"{v_count}")
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
        veh_metric.metric("Vehicles Tracked", "0")
        person_metric.metric("People Detected", "0")
        viol_metric.metric("Violations Logged", "0")

        frame_placeholder.markdown(
            """
            <div style="background: #161F38; border: 2px dashed #3A86FF44; border-radius: 12px;
                        height: 440px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <div style="font-size: 3.5rem; margin-bottom: 8px;">🛣️</div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.3rem;">HIGHWAY SURVEILLANCE STANDBY</div>
                <div style="color: #94A3B8; font-size: 0.9rem; max-width: 480px; margin-top: 8px;">
                    Select your surveillance input on the left panel (Bundled Footage or External Camera) and click <b>'Start Surveillance'</b> to begin monitoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
