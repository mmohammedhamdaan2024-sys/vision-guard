"""
Vision Guard — Page 2: Live-Style Traffic Monitoring
Simulated highway surveillance camera, video/image frame processing, tracking HUD, and live alerts.
"""

import os
import cv2
import time
import tempfile
import streamlit as st
import numpy as np
from PIL import Image

from database.db_manager import DatabaseManager
from modules.helmet_detection import HelmetDetector
from modules.seatbelt_detection import SeatbeltDetector
from modules.tint_analysis import WindowTintAnalyzer
from modules.mirror_analysis import MirrorAnalyzer
from modules.number_plate_ocr import NumberPlateRecognizer
from modules.vehicle_tracking import VehicleTracker
from modules.violation_engine import ViolationDecisionEngine
from utils.image_utils import draw_hud_box
from utils.video_utils import frame_generator, get_video_metadata
from utils.validation import validate_file, sanitize_filename
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Traffic Monitoring | Vision Guard", page_icon="📹", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")
engine = ViolationDecisionEngine(db, evidence_dir="evidence")

# Initialize CV Engines
@st.cache_resource
def load_cv_pipelines():
    helmet_det = HelmetDetector("models/yolov8n.pt", "models/helmet_yolo.pt")
    seatbelt_det = SeatbeltDetector("models/yolov8n.pt", "models/seatbelt_yolo.pt")
    tint_ana = WindowTintAnalyzer(darkness_threshold=65.0)
    mirror_ana = MirrorAnalyzer(visibility_threshold=60.0)
    plate_ocr = NumberPlateRecognizer(use_gpu=False, min_conf=0.40)
    return helmet_det, seatbelt_det, tint_ana, mirror_ana, plate_ocr

helmet_det, seatbelt_det, tint_ana, mirror_ana, plate_ocr = load_cv_pipelines()

render_surveillance_header(
    title="SIMULATED LIVE HIGHWAY MONITORING",
    subtitle="Simulated multi-vehicle tracking, deep bounding boxes, and automated safety violation capture.",
    tag="CAMERA CAM-04 (HIGHWAY EXPRESSWAY)"
)

st.info(
    "ℹ️ **Operational Notice:** This panel runs in **simulated monitoring mode** on uploaded video footage "
    "or the bundled highway sample clip. It demonstrates automated vehicle tracking and violation capture without requiring physical camera hardware."
)

col_ctrl, col_display = st.columns([1, 2.5])

with col_ctrl:
    st.markdown("### 🎛️ Surveillance Feed Controls")
    source_choice = st.radio(
        "Select Feed Source:",
        ["Bundled Highway Sample Clip (Recommended)", "Upload Video (MP4 / AVI)", "Upload Static Traffic Image"]
    )

    video_path_to_process = None
    image_to_process = None

    if source_choice == "Bundled Highway Sample Clip (Recommended)":
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples", "sample_traffic_clip.mp4")
        if os.path.exists(sample_path):
            video_path_to_process = sample_path
            meta = get_video_metadata(sample_path)
            st.success(f"Loaded Bundled Clip: {meta['width']}x{meta['height']} @ {meta['fps']} FPS ({meta['duration_sec']}s)")
        else:
            st.error("Bundled sample clip missing.")

    elif source_choice == "Upload Video (MP4 / AVI)":
        uploaded_video = st.file_uploader("Upload Traffic Video File", type=["mp4", "avi", "mov"])
        if uploaded_video:
            v_bytes = uploaded_video.read()
            valid, msg = validate_file(uploaded_video.name, v_bytes, is_video=True)
            if valid:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(v_bytes)
                tfile.close()
                video_path_to_process = tfile.name
                st.success(f"Video uploaded: {uploaded_video.name}")
            else:
                st.error(msg)

    elif source_choice == "Upload Static Traffic Image":
        uploaded_img = st.file_uploader("Upload Traffic Scene Image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            i_bytes = uploaded_img.read()
            valid, msg = validate_file(uploaded_img.name, i_bytes, is_video=False)
            if valid:
                file_bytes = np.asarray(bytearray(i_bytes), dtype=np.uint8)
                image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.success(f"Image uploaded: {uploaded_img.name}")
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown("#### ⚙️ Processing Parameters")
    frame_skip = st.slider("Frame Sampling Step", min_value=1, max_value=5, value=2,
                           help="Process every Nth frame for smooth CPU throughput.")
    conf_thresh = st.slider("Vehicle Detection Confidence", min_value=0.25, max_value=0.85, value=0.45)

    start_btn = st.button("▶️ Launch Surveillance Stream", use_container_width=True)


with col_display:
    st.markdown("### 🖥️ Optical Feed & Real-Time Event Log")
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    events_placeholder = st.empty()

    if start_btn:
        if image_to_process is not None:
            # Process static image
            status_placeholder.markdown("🔍 **Analyzing Scene Elements...**")
            tracker = VehicleTracker()
            annotated = image_to_process.copy()

            # YOLO inference
            results = helmet_det.base_model(image_to_process, conf=conf_thresh, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            detections = []
            for b, c, score in zip(boxes, classes, confs):
                if c == 2:
                    detections.append((b, "car", score))
                elif c == 3:
                    detections.append((b, "motorcycle", score))
                elif c == 5:
                    detections.append((b, "bus", score))
                elif c == 7:
                    detections.append((b, "truck", score))

            tracked_vehicles = tracker.update(detections)
            events_logged = []

            for tv in tracked_vehicles:
                # Per-vehicle analysis
                v_crop = image_to_process[tv.bbox[1]:tv.bbox[3], tv.bbox[0]:tv.bbox[2]]
                h_res, s_res, t_res, m_res, p_res = None, None, None, None, None

                if tv.vehicle_type == "motorcycle":
                    h_res = helmet_det.analyze(image_to_process)
                elif tv.vehicle_type in ("car", "truck"):
                    s_res = seatbelt_det.analyze(image_to_process)
                    t_res = tint_ana.analyze(image_to_process, tv.bbox)
                    m_res = mirror_ana.analyze(image_to_process, tv.bbox)

                p_res = plate_ocr.recognize(image_to_process, tv.bbox)

                viols = engine.evaluate_vehicle_event(
                    frame=image_to_process,
                    tracking_id=tv.track_id,
                    vehicle_type=tv.vehicle_type,
                    vehicle_box=tv.bbox,
                    helmet_result=h_res,
                    seatbelt_result=s_res,
                    tint_result=t_res,
                    mirror_result=m_res,
                    plate_result=p_res
                )
                events_logged.extend(viols)

                annotated = draw_hud_box(
                    annotated, tv.bbox, tv.vehicle_type.upper(),
                    conf=tv.conf, track_id=tv.track_id, is_violation=len(viols) > 0
                )

            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            status_placeholder.success(f"✅ Scene Analysis Complete: {len(tracked_vehicles)} vehicles tracked, {len(events_logged)} violations identified.")

            if events_logged:
                st.markdown("#### 🚨 Captured Violation Events")
                for e in events_logged:
                    st.error(f"**[{e.get('challan_ref', 'ALERT')}] {e['violation_type']}** | Fine: ₹{e.get('fine', 0)} | {e['notes']}")

        elif video_path_to_process is not None:
            # Process video stream
            tracker = VehicleTracker(max_disappeared=12)
            progress_bar = st.progress(0)
            meta = get_video_metadata(video_path_to_process)
            total_frames = meta.get("total_frames", 60)

            event_history = []
            frame_counter = 0

            for f_idx, frame in frame_generator(video_path_to_process, frame_skip=frame_skip):
                frame_counter += 1
                annotated = frame.copy()

                # YOLO base detection
                results = helmet_det.base_model(frame, conf=conf_thresh, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                detections = []
                for b, c, score in zip(boxes, classes, confs):
                    if c == 2:
                        detections.append((b, "car", score))
                    elif c == 3:
                        detections.append((b, "motorcycle", score))
                    elif c == 5:
                        detections.append((b, "bus", score))
                    elif c == 7:
                        detections.append((b, "truck", score))

                # Track across frames
                tracked_vehicles = tracker.update(detections)

                for tv in tracked_vehicles:
                    is_viol = False
                    # Only evaluate once per vehicle track to avoid duplicates
                    if tv.frames_active == 2 or (tv.frames_active % 15 == 0 and len(tv.violations_flagged) == 0):
                        h_res = helmet_det.analyze(frame) if tv.vehicle_type == "motorcycle" else None
                        s_res = seatbelt_det.analyze(frame) if tv.vehicle_type in ("car", "truck") else None
                        t_res = tint_ana.analyze(frame, tv.bbox) if tv.vehicle_type in ("car", "truck") else None
                        m_res = mirror_ana.analyze(frame, tv.bbox) if tv.vehicle_type in ("car", "truck") else None
                        p_res = plate_ocr.recognize(frame, tv.bbox)

                        viols = engine.evaluate_vehicle_event(
                            frame=frame,
                            tracking_id=tv.track_id,
                            vehicle_type=tv.vehicle_type,
                            vehicle_box=tv.bbox,
                            helmet_result=h_res,
                            seatbelt_result=s_res,
                            tint_result=t_res,
                            mirror_result=m_res,
                            plate_result=p_res
                        )
                        if viols:
                            is_viol = True
                            for v in viols:
                                tv.mark_violation_flagged(v["violation_type"])
                                event_history.insert(0, f"[{v.get('challan_ref', 'REV')}] {tv.track_id}: {v['violation_type']} (₹{v.get('fine', 0)})")

                    annotated = draw_hud_box(
                        annotated, tv.bbox, tv.vehicle_type.upper(),
                        conf=tv.conf, track_id=tv.track_id, is_violation=len(tv.violations_flagged) > 0
                    )

                # Update Streamlit display
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                if total_frames > 0:
                    progress_bar.progress(min(1.0, (f_idx + 1) / total_frames))

                status_placeholder.markdown(
                    f"🟢 **Monitoring Active:** Frame `{f_idx}` | Tracked Vehicles: `{len(tracked_vehicles)}` | Violations Logged: `{len(event_history)}`"
                )

                if event_history:
                    events_placeholder.markdown(
                        "**Recent Alerts:**\n" + "\n".join([f"- 🚨 {e}" for e in event_history[:5]])
                    )

                time.sleep(0.04) # Realistic frame playback delay

            progress_bar.progress(1.0)
            status_placeholder.success("🏁 Stream processing finalized. All violation records and evidence persisted to SQLite.")
        else:
            st.warning("Please choose a valid feed source or upload a file.")
    else:
        # Default idle preview image
        video_placeholder.markdown(
            """
            <div style="background: #161F38; border: 2px dashed #3A86FF44; border-radius: 10px;
                        height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                <div style="font-size: 2.8rem; margin-bottom: 8px;">📹</div>
                <div style="color: #FFFFFF; font-weight: 600; font-size: 1.1rem;">SURVEILLANCE CAMERA STANDBY</div>
                <div style="color: #94A3B8; font-size: 0.85rem; max-width: 400px; margin-top: 6px;">
                    Select the Bundled Highway Sample Clip or upload footage, then click <b>'Launch Surveillance Stream'</b> to begin automated road safety monitoring.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
