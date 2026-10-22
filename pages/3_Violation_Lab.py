"""
Vision Guard — Page 3: Violation Detection Lab
Dedicated interactive computer vision research & diagnostic testing laboratory
for Helmet, Seat-belt, Window Tint, Mirror Visibility, and Number Plate OCR.
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules.helmet_detection import HelmetDetector
from modules.seatbelt_detection import SeatbeltDetector
from modules.tint_analysis import WindowTintAnalyzer
from modules.mirror_analysis import MirrorAnalyzer
from modules.number_plate_ocr import NumberPlateRecognizer
from utils.validation import validate_file
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Violation Lab | Vision Guard", page_icon="🧪", layout="wide")
inject_custom_css()
require_auth()

@st.cache_resource
def load_lab_engines():
    return (
        HelmetDetector("models/yolov8n.pt", "models/helmet_yolo.pt"),
        SeatbeltDetector("models/yolov8n.pt", "models/seatbelt_yolo.pt"),
        WindowTintAnalyzer(darkness_threshold=65.0),
        MirrorAnalyzer(visibility_threshold=60.0),
        NumberPlateRecognizer(use_gpu=False, min_conf=0.40)
    )

helmet_det, seatbelt_det, tint_ana, mirror_ana, plate_ocr = load_lab_engines()

render_surveillance_header(
    title="VIOLATION DETECTION & COMPUTER VISION LAB",
    subtitle="Interactive testing bench: verify individual CV models, inspect diagnostic metrics, and compare raw vs processed features.",
    tag="LAB BENCH ACTIVE"
)

tab_helmet, tab_belt, tab_tint, tab_mirror, tab_plate = st.tabs([
    "🪖 1. Helmet Detection",
    "🎗️ 2. Seat-Belt Detection",
    "🕶️ 3. Window Tint Analysis",
    "🪞 4. Mirror Visibility",
    "🔢 5. Number Plate OCR"
])

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")

# =============================================================
# TAB 1: HELMET DETECTION
# =============================================================
with tab_helmet:
    st.markdown("### 🪖 Motorcycle Rider Helmet Compliance")
    st.markdown(
        "Detects two-wheelers, associates visible riders, crops the cranial region, and classifies helmet status "
        "using deep inference or cranial contour and skin-tone geometry."
    )

    h_col1, h_col2 = st.columns([1, 1.8])
    with h_col1:
        h_source = st.radio("Helmet Test Image:", ["Use Bundled Sample (Rider Without Helmet)", "Upload Custom Image"], key="h_src")
        img_helmet = None
        if h_source == "Use Bundled Sample (Rider Without Helmet)":
            p = os.path.join(SAMPLES_DIR, "sample_helmet.jpg")
            if os.path.exists(p):
                img_helmet = cv2.imread(p)
                st.caption(f"Loaded: `{p}`")
        else:
            up_h = st.file_uploader("Upload Two-Wheeler / Rider Image", type=["jpg", "png", "jpeg"], key="up_h")
            if up_h:
                b = up_h.read()
                valid, msg = validate_file(up_h.name, b)
                if valid:
                    arr = np.asarray(bytearray(b), dtype=np.uint8)
                    img_helmet = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        run_h = st.button("🔬 Execute Helmet Analysis", key="btn_h", use_container_width=True)

    with h_col2:
        if run_h and img_helmet is not None:
            res = helmet_det.analyze(img_helmet)
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(img_helmet, cv2.COLOR_BGR2RGB), use_container_width=True)
            with r2:
                st.markdown("**Annotated Detection HUD**")
                st.image(cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### 📋 Diagnostic Assessment")
            status = res["status"]
            if status == "Helmet detected":
                st.success(f"Status: **{status}** | Confidence: **{res['confidence']:.2f}**")
            elif status == "No helmet detected":
                st.error(f"Potential Violation: **{status}** | Confidence: **{res['confidence']:.2f}**")
            else:
                st.warning(f"Status: **{status}** | Confidence: **{res['confidence']:.2f}**")

            if res.get("head_crop") is not None:
                st.markdown("**Extracted Cranial Evidence Crop:**")
                st.image(cv2.cvtColor(res["head_crop"], cv2.COLOR_BGR2RGB), width=180)

            with st.expander("🔍 Deep Technical Diagnostics"):
                st.json(res.get("details", {}))
        elif img_helmet is not None:
            st.image(cv2.cvtColor(img_helmet, cv2.COLOR_BGR2RGB), width=400, caption="Ready for analysis")

# =============================================================
# TAB 2: SEAT-BELT DETECTION
# =============================================================
with tab_belt:
    st.markdown("### 🎗️ Vehicle Occupant Seat-Belt Compliance")
    st.markdown(
        "Isolates front windshield cabin zones, applies CLAHE contrast enhancement, "
        "and profiles diagonal shoulder-to-hip strap edges using Hough Line geometry."
    )

    b_col1, b_col2 = st.columns([1, 1.8])
    with b_col1:
        b_source = st.radio("Seat-Belt Test Image:", ["Use Bundled Sample (Car Occupant Audit)", "Upload Custom Image"], key="b_src")
        img_belt = None
        if b_source == "Use Bundled Sample (Car Occupant Audit)":
            p = os.path.join(SAMPLES_DIR, "sample_seatbelt.jpg")
            if os.path.exists(p):
                img_belt = cv2.imread(p)
                st.caption(f"Loaded: `{p}`")
        else:
            up_b = st.file_uploader("Upload Car Cabin Image", type=["jpg", "png", "jpeg"], key="up_b")
            if up_b:
                b = up_b.read()
                valid, msg = validate_file(up_b.name, b)
                if valid:
                    arr = np.asarray(bytearray(b), dtype=np.uint8)
                    img_belt = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        run_b = st.button("🔬 Execute Seat-Belt Analysis", key="btn_b", use_container_width=True)

    with b_col2:
        if run_b and img_belt is not None:
            res_b = seatbelt_det.analyze(img_belt)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Scene**")
                st.image(cv2.cvtColor(img_belt, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**HUD Analysis Overlay**")
                st.image(cv2.cvtColor(res_b["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### 📋 Diagnostic Assessment")
            status_b = res_b["status"]
            if status_b == "Seat belt detected":
                st.success(f"Status: **{status_b}** | Confidence: **{res_b['confidence']:.2f}**")
            elif status_b == "Possible seat-belt violation":
                st.error(f"Potential Violation: **{status_b}** | Confidence: **{res_b['confidence']:.2f}**")
            else:
                st.warning(f"Status: **{status_b}** | Confidence: **{res_b['confidence']:.2f}**")

            if res_b.get("occupant_crop") is not None:
                st.markdown("**Extracted Cabin Region:**")
                st.image(cv2.cvtColor(res_b["occupant_crop"], cv2.COLOR_BGR2RGB), width=220)

            with st.expander("🔍 Deep Technical Diagnostics"):
                st.json(res_b.get("diagnostics", {}))
        elif img_belt is not None:
            st.image(cv2.cvtColor(img_belt, cv2.COLOR_BGR2RGB), width=400, caption="Ready for analysis")

# =============================================================
# TAB 3: WINDOW TINT ANALYSIS
# =============================================================
with tab_tint:
    st.markdown("### 🕶️ Vehicle Window Tint & Visibility Assessment")
    st.markdown(
        "Estimates apparent window darkness and derives an image-based Visible Light Transmission (VLT proxy). "
        "Flags heavily obscured or dark film windows for manual physical verification."
    )

    t_col1, t_col2 = st.columns([1, 1.8])
    with t_col1:
        t_thresh = st.slider("Experimental Darkness Review Threshold (%)", min_value=40.0, max_value=90.0, value=65.0, step=1.0)
        tint_ana.darkness_threshold = t_thresh

        t_source = st.radio("Window Test Image:", ["Use Bundled Sample (Dark Tint Profile)", "Upload Custom Image"], key="t_src")
        img_tint = None
        if t_source == "Use Bundled Sample (Dark Tint Profile)":
            p = os.path.join(SAMPLES_DIR, "sample_tint.jpg")
            if os.path.exists(p):
                img_tint = cv2.imread(p)
                st.caption(f"Loaded: `{p}`")
        else:
            up_t = st.file_uploader("Upload Vehicle Side Profile Image", type=["jpg", "png", "jpeg"], key="up_t")
            if up_t:
                b = up_t.read()
                valid, msg = validate_file(up_t.name, b)
                if valid:
                    arr = np.asarray(bytearray(b), dtype=np.uint8)
                    img_tint = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        run_t = st.button("🔬 Execute Window Tint Analysis", key="btn_t", use_container_width=True)

    with t_col2:
        if run_t and img_tint is not None:
            res_t = tint_ana.analyze(img_tint)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Profile**")
                st.image(cv2.cvtColor(img_tint, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**Tint Diagnostic HUD**")
                st.image(cv2.cvtColor(res_t["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### 📋 Assessment & Photometric Proxy")
            d_score = res_t["darkness_score"]
            vlt = res_t["estimated_vlt_proxy"]
            if res_t["status"] == "Flagged for Dark Window Review":
                st.error(f"Flagged for Review: Darkness Score = **{d_score}%** (VLT Proxy: **{vlt}%**)")
            else:
                st.success(f"Compliant Visibility: Darkness Score = **{d_score}%** (VLT Proxy: **{vlt}%**)")

            if res_t.get("window_crop") is not None:
                st.markdown("**Window Crop:**")
                st.image(cv2.cvtColor(res_t["window_crop"], cv2.COLOR_BGR2RGB), width=220)

            with st.expander("⚖️ Legal Limitation & Methodology"):
                st.json(res_t.get("details", {}))
        elif img_tint is not None:
            st.image(cv2.cvtColor(img_tint, cv2.COLOR_BGR2RGB), width=400, caption="Ready for analysis")

# =============================================================
# TAB 4: MIRROR VISIBILITY
# =============================================================
with tab_mirror:
    st.markdown("### 🪞 Exterior Mirror Visibility Assessment")
    st.markdown(
        "Analyzes vehicle lateral flank zones to detect presence and structural clarity of exterior rear-view mirrors."
    )

    m_col1, m_col2 = st.columns([1, 1.8])
    with m_col1:
        m_thresh = st.slider("Experimental Visibility Threshold (%)", min_value=30.0, max_value=85.0, value=60.0, step=5.0)
        mirror_ana.visibility_threshold = m_thresh

        m_source = st.radio("Mirror Test Image:", ["Use Bundled Sample (Car Profile)", "Upload Custom Image"], key="m_src")
        img_mirror = None
        if m_source == "Use Bundled Sample (Car Profile)":
            p = os.path.join(SAMPLES_DIR, "sample_seatbelt.jpg")
            if os.path.exists(p):
                img_mirror = cv2.imread(p)
                st.caption(f"Loaded: `{p}`")
        else:
            up_m = st.file_uploader("Upload Vehicle Front/Side Image", type=["jpg", "png", "jpeg"], key="up_m")
            if up_m:
                b = up_m.read()
                valid, msg = validate_file(up_m.name, b)
                if valid:
                    arr = np.asarray(bytearray(b), dtype=np.uint8)
                    img_mirror = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        run_m = st.button("🔬 Execute Mirror Analysis", key="btn_m", use_container_width=True)

    with m_col2:
        if run_m and img_mirror is not None:
            res_m = mirror_ana.analyze(img_mirror)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Image**")
                st.image(cv2.cvtColor(img_mirror, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**Mirror HUD Inspection**")
                st.image(cv2.cvtColor(res_m["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### 📋 Assessment")
            status_m = res_m["status"]
            score_m = res_m["visibility_score"]
            if status_m == "Mirror region detected":
                st.success(f"Status: **{status_m}** | Clarity Score: **{score_m}%**")
            elif status_m == "Mirror potentially obscured or not clearly visible":
                st.warning(f"Review Required: **{status_m}** | Clarity Score: **{score_m}%**")
            else:
                st.info(f"Status: **{status_m}**")

            with st.expander("🔍 Deep Technical Diagnostics"):
                st.json(res_m.get("details", {}))
        elif img_mirror is not None:
            st.image(cv2.cvtColor(img_mirror, cv2.COLOR_BGR2RGB), width=400, caption="Ready for analysis")

# =============================================================
# TAB 5: NUMBER PLATE RECOGNITION & OCR
# =============================================================
with tab_plate:
    st.markdown("### 🔢 Number Plate Detection & Optical Character Recognition")
    st.markdown(
        "Isolates license plate regions using rectangular aspect-ratio filters, applies bilateral denoising "
        "and adaptive thresholding, runs EasyOCR, and permits officer manual correction."
    )

    p_col1, p_col2 = st.columns([1, 1.8])
    with p_col1:
        p_source = st.radio("Plate Test Image:", ["Use Bundled Sample (License Plate)", "Upload Custom Image"], key="p_src")
        img_plate = None
        if p_source == "Use Bundled Sample (License Plate)":
            p = os.path.join(SAMPLES_DIR, "sample_plate.jpg")
            if os.path.exists(p):
                img_plate = cv2.imread(p)
                st.caption(f"Loaded: `{p}`")
        else:
            up_p = st.file_uploader("Upload Vehicle Plate Image", type=["jpg", "png", "jpeg"], key="up_p")
            if up_p:
                b = up_p.read()
                valid, msg = validate_file(up_p.name, b)
                if valid:
                    arr = np.asarray(bytearray(b), dtype=np.uint8)
                    img_plate = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        run_p = st.button("🔬 Execute Number Plate OCR", key="btn_p", use_container_width=True)

    with p_col2:
        if run_p and img_plate is not None:
            res_p = plate_ocr.recognize(img_plate)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Scene**")
                st.image(cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**Localized Plate HUD**")
                st.image(cv2.cvtColor(res_p["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### 📋 Recognition Results")
            st.markdown(f"Recognized Plate Number: `<h2 style='color:#00E5FF; display:inline;'>{res_p['plate_number']}</h2>`", unsafe_allow_html=True)
            st.markdown(f"Raw OCR Output: `{res_p['raw_ocr_text']}` | Confidence Score: **{res_p['confidence']:.2f}** | Status: **{res_p['status']}**")

            k1, k2 = st.columns(2)
            with k1:
                if res_p.get("plate_crop") is not None:
                    st.markdown("**Cropped Plate:**")
                    st.image(cv2.cvtColor(res_p["plate_crop"], cv2.COLOR_BGR2RGB), width=200)
            with k2:
                if res_p.get("preprocessed_crop") is not None:
                    st.markdown("**Adaptive Morphological Filter:**")
                    st.image(res_p["preprocessed_crop"], width=200)

            # Manual Correction Box
            st.markdown("---")
            st.markdown("##### ✏️ Officer Manual Correction Override")
            manual_plate = st.text_input("Correct Registration Plate:", value=res_p['plate_number'])
            if st.button("Save Manual Correction"):
                st.success(f"Registration number corrected to '{manual_plate}' for audit registry.")
        elif img_plate is not None:
            st.image(cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB), width=400, caption="Ready for analysis")
