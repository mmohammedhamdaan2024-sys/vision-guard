"""
Vision Guard — Page 7: Settings & System Diagnostics
Threshold calibration, demonstration fine schedule, privacy policies, and technical viva guide.
"""

import os
import yaml
import streamlit as st
from database.db_manager import DatabaseManager
from utils.ui_components import inject_custom_css, render_surveillance_header, require_auth

st.set_page_config(page_title="Settings & About | Vision Guard", page_icon="⚙️", layout="wide")
inject_custom_css()
require_auth()

db = DatabaseManager("data/visionguard.db")

render_surveillance_header(
    title="SETTINGS & SYSTEM DIAGNOSTICS",
    subtitle="Calibrate detection thresholds, configure demonstration fines, inspect model status, and view legal disclaimers.",
    tag="CONFIG & AUDIT"
)

tab_thresh, tab_fines, tab_models, tab_privacy, tab_about = st.tabs([
    "🎚️ 1. Threshold Configuration",
    "💰 2. Demonstration Fines",
    "🤖 3. Model & Diagnostics",
    "🛡️ 4. Privacy & Ethics Policy",
    "🎓 5. Project & Viva Guide"
])

# =============================================================
# TAB 1: THRESHOLD CONFIGURATION
# =============================================================
with tab_thresh:
    st.markdown("### 🎚️ Computer Vision Inference Thresholds")
    st.markdown("Calibrate minimum confidence scores required to register safety alerts.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Primary Detection Confidences")
        c_veh = st.slider("Vehicle Detection Confidence (YOLOv8)", 0.20, 0.90, 0.45, 0.05)
        c_helm = st.slider("Helmet Violation Trigger Confidence", 0.30, 0.95, 0.50, 0.05)
        c_belt = st.slider("Seat-belt Violation Confidence", 0.30, 0.95, 0.50, 0.05)
        c_plate = st.slider("Number Plate OCR Confidence", 0.20, 0.90, 0.40, 0.05)

    with t2:
        st.markdown("#### Experimental Image Assessment Thresholds")
        st.caption("ℹ️ These are experimental heuristic thresholds for research and demonstration.")
        t_tint = st.slider("Dark Window Apparent Darkness Threshold (%)", 40.0, 90.0, 65.0, 1.0,
                           help="Windows exceeding this darkness percentage are flagged for manual review.")
        t_mirr = st.slider("Mirror Clarity Minimum Threshold (%)", 30.0, 85.0, 60.0, 5.0,
                           help="Lateral mirror scores below this are marked as potentially obscured.")

    if st.button("💾 Save Calibration Settings"):
        db.set_setting("vehicle_conf", str(c_veh))
        db.set_setting("helmet_conf", str(c_helm))
        db.set_setting("seatbelt_conf", str(c_belt))
        db.set_setting("plate_conf", str(c_plate))
        db.set_setting("dark_tint_threshold", str(t_tint))
        db.set_setting("mirror_threshold", str(t_mirr))
        st.success("✅ Threshold settings saved to persistent SQLite configuration registry.")

# =============================================================
# TAB 2: DEMONSTRATION FINES
# =============================================================
with tab_fines:
    st.markdown("### 💰 Demonstration Fine Schedule")
    st.warning("⚠️ **Notice:** All monetary amounts are illustrative values used solely for demonstration.")

    f1, f2 = st.columns(2)
    with f1:
        f_helm = st.number_input("Helmet Non-Compliance Fine (₹)", min_value=100, max_value=10000, value=1000, step=100)
        f_belt = st.number_input("Seat-belt Non-Compliance Fine (₹)", min_value=100, max_value=10000, value=1000, step=100)
        f_tint = st.number_input("Dark Window Film Review Fine (₹)", min_value=100, max_value=10000, value=750, step=50)

    with f2:
        f_mirr = st.number_input("Obscured Mirror Review Fine (₹)", min_value=100, max_value=10000, value=500, step=50)
        f_plate = st.number_input("Unreadable / Obscured Plate Fine (₹)", min_value=100, max_value=10000, value=2000, step=100)

    if st.button("💾 Update Demonstration Fine Schedule"):
        db.set_setting("fine_helmet", str(f_helm))
        db.set_setting("fine_seatbelt", str(f_belt))
        db.set_setting("fine_tint", str(f_tint))
        db.set_setting("fine_mirror", str(f_mirr))
        db.set_setting("fine_plate", str(f_plate))
        st.success("✅ Fine schedule updated.")

# =============================================================
# TAB 3: MODEL & SYSTEM DIAGNOSTICS
# =============================================================
with tab_models:
    st.markdown("### 🤖 Installed Deep Learning Models & Runtimes")

    base_exists = os.path.exists("models/yolov8n.pt")
    helmet_custom = os.path.exists("models/helmet_yolo.pt")
    seatbelt_custom = os.path.exists("models/seatbelt_yolo.pt")

    st.markdown(
        f"""
        | Component | Type | Status | Supported Classes / Mode |
        | :--- | :--- | :--- | :--- |
        | **YOLOv8 Base** | Object Detection | `{'🟢 Active' if base_exists else '🔴 Missing'}` | 80 COCO Classes (Car, Motorcycle, Bus, Truck, Person) |
        | **Helmet Engine** | Headgear Analyzer | `{'🟢 Custom Deep Model' if helmet_custom else '🟡 Intelligent CV Fallback'}` | `[Helmet, No Helmet, Uncertain]` via cranial contour & skin tone |
        | **Seat-Belt Engine** | Occupant Restraint | `{'🟢 Custom Deep Model' if seatbelt_custom else '🟡 Intelligent CV Fallback'}` | CLAHE contrast enhancement & diagonal Hough geometry |
        | **Window Tint** | Image Processing | `🟢 Active` | HSV / CIELAB apparent darkness & VLT proxy estimation |
        | **Mirror Engine** | Image Processing | `🟢 Active` | Lateral flank gradient & Laplacian protrusion variance |
        | **EasyOCR** | Text Recognition | `🟢 Active` | Latin Alphanumeric plate OCR + bilateral denoising |
        | **SQLite Engine** | Persistent DB | `🟢 Active` | `data/visionguard.db` |
        """
    )

    st.markdown("---")
    st.markdown("#### 🗄️ Database Management")
    if st.button("🔄 Reset & Re-Seed Demonstration Database"):
        db.seed_demo_data(force=True)
        st.success("Database restored to fresh demonstration state with realistic sample records.")
        st.rerun()

# =============================================================
# TAB 4: PRIVACY & ETHICS POLICY
# =============================================================
with tab_privacy:
    st.markdown("### 🛡️ Privacy, Ethics, and Statutory Limitations")
    st.markdown(
        """
        Vision Guard is engineered with strict ethical and data privacy principles:

        1. **Local Execution by Default:** All computer vision processing, video decoding, and OCR occurs entirely on the local deployment machine. No video frames, license plate texts, or telemetry data are transmitted to third-party cloud services.
        2. **Human-in-the-Loop Mandate:** Automated algorithmic findings are classified as *Potential Violations* or *Manual Review Flags*. No legally binding citation or punitive fine can be issued without explicit review and approval by an authorized enforcement officer.
        3. **No Facial Recognition:** The application intentionally does NOT implement facial biometric recognition, mass identity matching, or tracking of individuals outside vehicle safety contexts.
        4. **Simulated Notification Safety:** The mobile notification module operates strictly as a mock preview simulation. It does not contact real cellular networks or send unsolicited SMS messages.
        5. **Scientific Disclaimer on Photometric Limits:** Image-derived window darkness scores and mirror visibility values are visual approximations affected by camera sensor dynamics, atmospheric lighting, and viewing angles. They cannot substitute certified physical instruments (e.g. photometer tint meters).
        """
    )

# =============================================================
# TAB 5: PROJECT & VIVA GUIDE
# =============================================================
with tab_about:
    st.markdown("### 🎓 B.Sc. Computer Science Final-Year Major Project Guide")
    st.markdown(
        """
        #### Project Information
        - **Project Title:** Vision Guard — AI-Powered Traffic Violation Detection and Road Safety Monitoring System
        - **Curriculum:** B.Sc. Computer Science (Final-Year Major / Capstone Project)
        - **Primary Technologies:** Python 3, OpenCV, Ultralytics YOLOv8, EasyOCR, SQLite 3, Streamlit, Plotly, ReportLab.

        ---

        #### Frequently Asked Viva Examination Questions & Answers
        1. **Q: How does the system handle tracking without duplicating violations?**
           *A:* The `VehicleTracker` class computes Euclidean centroid distances between successive frames to maintain persistent tracking IDs (`TRK-101`, `TRK-102`). Each tracked vehicle maintains a `violations_flagged` set, preventing duplicate citations for the same vehicle during its transit.

        2. **Q: What happens if YOLOv8 nano does not have a dedicated 'helmet' class?**
           *A:* Vision Guard implements an intelligent dual-mode architecture. If a custom fine-tuned model (`helmet_yolo.pt`) is present, it executes deep classification. Otherwise, it isolates the rider's cranial region and uses morphological contour circularity, edge density, and HSV skin-tone masking to classify helmet presence or flag uncertain frames.

        3. **Q: Can window tint darkness derived from RGB images be used in court?**
           *A:* No. RGB camera values are influenced by ambient sunlight, tinted glass coatings, and interior shadows. Vision Guard explicitly labels this an *experimental VLT proxy* and mandates physical testing with a calibrated photometric tint meter before any statutory enforcement.

        4. **Q: Why was EasyOCR selected over traditional Tesseract?**
           *A:* EasyOCR leverages deep CRAFT text detection and ResNet-based recognition models with PyTorch, providing superior robustness against motion blur, low illumination, and skewed license plates common in highway surveillance.
        """
    )
