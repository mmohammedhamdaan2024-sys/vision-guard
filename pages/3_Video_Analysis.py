"""
VisionGuard — Page 2: Video Analysis Module
Analyzes recorded traffic videos (MP4/AVI/MOV), applies automated violation detection,
tracks vehicles, displays processing progress, and permits downloading annotated output.
"""

import os
import cv2
import time
import tempfile
import streamlit as st
import numpy as np

from config import DB_PATH, DEFAULT_CONFIDENCE_THRESHOLD, SAMPLES_DIR, COLORS
from modules.vehicle_detection import VehicleDetector
from modules.object_tracking import ObjectTracker
from modules.helmet_detection import HelmetDetector
from modules.seatbelt_detection import SeatbeltDetector
from modules.traffic_signal_detection import TrafficSignalMonitor
from modules.violation_engine import ViolationEngine
from modules.event_storage import EventStorage
from utils.image_utils import draw_hud_box
from utils.video_utils import get_video_metadata, frame_generator
from utils.validation import validate_file
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(page_title="Video Analysis | VisionGuard", page_icon="🎞️", layout="wide")
inject_custom_css()

# Cache Models
@st.cache_resource
def load_analysis_engines():
    v_det = VehicleDetector(conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD)
    h_det = HelmetDetector()
    s_det = SeatbeltDetector()
    storage = EventStorage(DB_PATH)
    v_eng = ViolationEngine(storage)
    return v_det, h_det, s_det, storage, v_eng

v_detector, h_detector, s_detector, storage, v_engine = load_analysis_engines()

render_surveillance_header(
    title="RECORDED TRAFFIC VIDEO ANALYSIS & AUDITING",
    subtitle="Ingest traffic footage, detect safety violations across video frames, and export annotated evidence files.",
    tag="BATCH VIDEO AUDIT"
)

v_col1, v_col2 = st.columns([1, 2.2])

with v_col1:
    st.markdown("### 📁 Select Video Footage")
    v_source = st.radio(
        "Choose Video Origin:",
        ["Use Bundled Traffic Signal Video (Instant)", "Upload Custom Video File (MP4/AVI)"]
    )

    video_path_to_process = None

    if v_source == "Use Bundled Traffic Signal Video (Instant)":
        b_path = str(SAMPLES_DIR / "sample_traffic_clip.mp4")
        if os.path.exists(b_path):
            video_path_to_process = b_path
            meta = get_video_metadata(b_path)
            st.success(f"Loaded: {meta['width']}x{meta['height']} @ {meta['fps']} FPS ({meta['duration_sec']}s)")
        else:
            st.error("Bundled sample video not found on disk.")
    else:
        uploaded_file = st.file_uploader("Upload Traffic Recording:", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file:
            v_bytes = uploaded_file.read()
            valid, msg = validate_file(uploaded_file.name, v_bytes, is_video=True)
            if valid:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(v_bytes)
                tfile.close()
                video_path_to_process = tfile.name
                st.success(f"Uploaded: {uploaded_file.name}")
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("#### ⚙️ Analysis Parameters")
    sample_step = st.slider("Sampling Step (Frame Skip)", min_value=1, max_value=4, value=2,
                            help="Skips intermediate frames to speed up CPU inference.")
    conf_level = st.slider("Detection Confidence", 0.25, 0.85, 0.40, 0.05)
    sig_state = st.selectbox("Simulated Signal State:", ["RED", "YELLOW", "GREEN"], index=0)
    stop_ratio = st.slider("Virtual Stop-Line Height (%)", 0.30, 0.85, 0.65, 0.05)

    start_proc_btn = st.button("🚀 Execute Video Audit", type="primary", use_container_width=True)


with v_col2:
    st.markdown("### 🖥️ Processing Stream & Summary")
    video_box = st.empty()
    progress_bar = st.empty()
    status_label = st.empty()
    summary_box = st.empty()

    if start_proc_btn and video_path_to_process:
        meta = get_video_metadata(video_path_to_process)
        total_frames = meta.get("total_frames", 60)

        tracker = ObjectTracker(max_disappeared=15, max_distance=85.0)
        signal_monitor = TrafficSignalMonitor(stop_line_y_ratio=stop_ratio, manual_signal_state=sig_state)

        # Output video writer setup
        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_out_path = temp_out.name
        temp_out.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        detected_events = []
        frame_idx = 0

        pbar = progress_bar.progress(0)
        status_label.markdown("⏳ **Initializing Object Detection Pipeline...**")

        for f_no, frame in frame_generator(video_path_to_process, frame_skip=sample_step):
            frame_idx += 1
            h, w = frame.shape[:2]

            if writer is None:
                writer = cv2.VideoWriter(temp_out_path, fourcc, 15.0, (w, h))

            # 1. Detection
            raw_dets = v_detector.detect(frame, conf=conf_level)

            # 2. Tracking
            tracked = tracker.update(raw_dets)

            annotated = frame.copy()
            annotated = signal_monitor.draw_hud(annotated)

            # 3. Safety Analysis
            for obj in tracked:
                vtype = obj.class_name

                # Motorcycle Helmet Analysis
                h_res = None
                if vtype == "motorcycle":
                    h_res = h_detector.analyze(frame, obj.bbox, raw_dets)
                    if h_res.get("head_box"):
                        hx1, hy1, hx2, hy2 = [int(v) for v in h_res["head_box"]]
                        color = h_res.get("box_color", COLORS["uncertain"])
                        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)
                        cv2.putText(annotated, f"{h_res['status']}", (hx1, max(15, hy1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

                # Car Seat-Belt Analysis
                s_res = None
                if vtype in ("car", "truck", "bus"):
                    s_res = s_detector.analyze(frame, obj.bbox)
                    if s_res.get("cabin_box"):
                        cx1, cy1, cx2, cy2 = [int(v) for v in s_res["cabin_box"]]
                        color = s_res.get("box_color", COLORS["uncertain"])
                        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 2)
                        cv2.putText(annotated, f"{s_res['status']}", (cx1, max(18, cy1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

                # Stop-Line Crossing
                sig_res = None
                if len(obj.history) >= 2:
                    sig_res = signal_monitor.check_stop_line_crossing(h, obj.history[-2], obj.history[-1], obj.bbox)

                # Evaluate Violations
                viols = v_engine.evaluate(
                    frame=frame,
                    tracked_object=obj,
                    helmet_result=h_res,
                    seatbelt_result=s_res,
                    traffic_signal_result=sig_res,
                    camera_source="Uploaded Video"
                )
                if viols:
                    detected_events.extend(viols)

                annotated = draw_hud_box(
                    annotated, obj.bbox, vtype.upper(),
                    conf=obj.conf, track_id=obj.track_id,
                    is_violation=len(obj.violations_flagged) > 0
                )

            # Write to output video
            writer.write(annotated)

            # Update display
            video_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            prog = min(1.0, (f_no + 1) / max(1, total_frames))
            pbar.progress(prog)
            status_label.markdown(f"🟢 **Processing:** Frame `{f_no}` / `{total_frames}` | Violations: `{len(detected_events)}`")

        if writer is not None:
            writer.release()

        pbar.progress(1.0)
        status_label.success("🏁 **Video Audit Complete! All records persisted to database.**")

        # Summary Breakdown
        summary_box.markdown(
            f"""
            #### 📊 Audit Summary
            - **Total Frames Analyzed:** `{frame_idx}`
            - **Total Violations Identified:** `{len(detected_events)}`
            - **Unique Vehicles Tracked:** `{len(tracker.tracks)}`
            """
        )

        # Download Processed Video
        if os.path.exists(temp_out_path) and os.path.getsize(temp_out_path) > 0:
            with open(temp_out_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video File (.mp4)",
                    data=vf.read(),
                    file_name="VisionGuard_Annotated_Audit.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    else:
        video_box.markdown(
            """
            <div style="background: #161F38; border: 2px dashed #3A86FF44; border-radius: 12px;
                        height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <div style="font-size: 3.2rem; margin-bottom: 8px;">🎞️</div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.2rem;">VIDEO ANALYSIS STANDBY</div>
                <div style="color: #94A3B8; font-size: 0.88rem; max-width: 420px; margin-top: 8px;">
                    Select the Bundled Sample Video or upload a custom video file on the left panel, then click <b>'Execute Video Audit'</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
