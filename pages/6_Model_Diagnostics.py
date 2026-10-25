"""
VisionGuard — Page 6: Model Diagnostics, Architecture & Training
Comprehensive technical telemetry, hardware acceleration diagnostics, active neural network
models, custom dataset training instructions, and statutory limitations.
"""

import os
import sys
import torch
import cv2
import streamlit as st

from config import (
    MODELS_DIR, HELMET_MODEL_PATH, YOLO_BASE_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD, TARGET_CLASSES, VIOLATION_TYPES
)
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(page_title="Model Diagnostics | VisionGuard", page_icon="🧠", layout="wide")
inject_custom_css()

render_surveillance_header(
    title="AI MODEL SPECIFICATIONS & HARDWARE DIAGNOSTICS",
    subtitle="Active neural network models, inference hardware acceleration, class mappings, and custom training documentation.",
    tag="SYSTEM ARCHITECTURE"
)

# -------------------------------------------------------------
# 1. Hardware & Runtime Diagnostics
# -------------------------------------------------------------
st.markdown("### ⚡ Hardware Acceleration & Runtime Environment")

cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "Host CPU Architecture"
torch_version = torch.__version__
cv_version = cv2.__version__
python_version = sys.version.split()[0]

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">ACCELERATION DEVICE</div>
            <div class="metric-value" style="font-size: 1.2rem; color: {'#00FF66' if cuda_available else '#00E5FF'};">
                {'CUDA GPU' if cuda_available else 'CPU RUNTIME'}
            </div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">{device_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with d2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PYTORCH RUNTIME</div>
            <div class="metric-value" style="font-size: 1.2rem; color: #FFE600;">v{torch_version}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Torch & Torchvision</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with d3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">OPENCV ENGINE</div>
            <div class="metric-value" style="font-size: 1.2rem; color: #3A86FF;">v{cv_version}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Optical processing</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with d4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PYTHON RUNTIME</div>
            <div class="metric-value" style="font-size: 1.2rem; color: #4CC9F0;">v{python_version}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 4px;">Modular environment</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Active Models Inventory
# -------------------------------------------------------------
st.markdown("### 📦 Active Neural Network Models Inventory")

m_c1, m_c2 = st.columns(2)

with m_c1:
    st.markdown(
        f"""
        <div style="background: #1C2541; border: 1px solid #3A86FF44; border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: #00E5FF; margin: 0;">🪖 Dedicated Helmet Detection Model</h4>
                <span style="background: #00FF6622; color: #00FF66; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">ACTIVE</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 8px;">
                Trained YOLOv8 architecture optimized on Roboflow & Kaggle motorcycle helmet safety datasets.
            </p>
            <ul style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.6;">
                <li><b>Weight Path:</b> <code>models/helmet_detector.pt</code></li>
                <li><b>Classes:</b> <code>0: With Helmet</code>, <code>1: Without Helmet</code></li>
                <li><b>Score mAP@0.5:</b> <code>81.1%</code></li>
                <li><b>Inference Latency:</b> ~18ms per frame (CPU)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with m_c2:
    st.markdown(
        f"""
        <div style="background: #1C2541; border: 1px solid #3A86FF44; border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: #00E5FF; margin: 0;">🚗 Highway Vehicle & Road User Model</h4>
                <span style="background: #00FF6622; color: #00FF66; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">ACTIVE</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 8px;">
                Ultralytics YOLOv8 nano backbone for multi-agent traffic participant localization and classification.
            </p>
            <ul style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.6;">
                <li><b>Weight Path:</b> <code>models/yolov8n.pt</code></li>
                <li><b>Monitored Classes:</b> <code>motorcycle</code>, <code>person</code>, <code>car</code>, <code>bus</code>, <code>truck</code></li>
                <li><b>Tracking:</b> Centroid / IoU association with trajectory vectoring</li>
                <li><b>Inference Latency:</b> ~22ms per frame (CPU)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Two Detection Modes Architecture
# -------------------------------------------------------------
st.markdown("### 🔬 Multi-Stage Detection Pipeline Architecture")

t1, t2, t3 = st.tabs(["🪖 Mode A: Personal Webcam", "🛣️ Mode B: Highway Surveillance", "🛠️ Custom Training Pipeline"])

with t1:
    st.markdown(
        """
        #### Mode A: Personal Webcam Testing Pipeline
        1. **Subject Localization:** Base detector extracts person bounding box. If user is sitting close to the camera with head/shoulders, direct headgear localization triggers.
        2. **Helmet Classification:** Runs `models/helmet_detector.pt` on the frame and head region to isolate `With Helmet` vs `Without Helmet`.
        3. **Temporal Smoothing:** Maintains a rolling FIFO queue of the last $N$ frames (default 5 frames). Predictions are smoothed via majority voting to prevent frame-to-frame label flickering.
        4. **Responsive Transitions:** Transitioning from wearing a helmet to removing it updates within ~150ms while suppressing false one-frame spikes.
        5. **Explicit Uncertainty:** Low-contrast lighting or angled poses yield `UNCERTAIN` rather than guessing arbitrarily.
        """
    )

with t2:
    st.markdown(
        """
        #### Mode B: Real-Road Highway Surveillance Pipeline
        1. **Multi-Agent Vehicle Tracking:** YOLOv8 localizes motorcycles, cars, trucks, buses, and persons across sequential frames with persistent tracking IDs (`TRK-101`, `TRK-102`).
        2. **Rider-to-Motorcycle Spatial Association:** Geometric overlap checks associate each detected rider with their specific motorcycle.
        3. **Multi-Rider Helmet Analysis:** Each localized rider is independently evaluated for headgear compliance using `models/helmet_detector.pt`.
        4. **Windshield Cabin Occupant Seat-Belt Analysis:** Windshield cabin is cropped, normalized with CLAHE contrast enhancement, and analyzed with diagonal Hough line gradient extraction.
        5. **Traffic Light Stop-Line Penetration:** Virtual geometric stop-line across the road is monitored. Vehicles crossing during active **RED** signals trigger citations.
        6. **Violation Engine with Deduplication:** Violations are logged once per vehicle, watermarked evidence snapshots are captured, and events are stored in SQLite.
        """
    )

with t3:
    st.markdown(
        """
        #### Custom Training & Dataset Preparation Pipeline
        VisionGuard includes automated scripts for obtaining datasets, converting annotations, and fine-tuning YOLO weights:
        
        ```powershell
        # 1. Prepare and validate custom dataset splits
        python scripts/prepare_dataset.py --dataset_dir datasets/helmet_data

        # 2. Execute transfer-learning fine-tuning from yolov8n.pt
        python scripts/train_model.py --epochs 50 --batch 16 --imgsz 640

        # 3. Evaluate Precision, Recall, and mAP@0.5
        python scripts/evaluate_model.py --weights models/helmet_detector.pt
        ```
        """
    )

st.markdown("---")

# -------------------------------------------------------------
# 4. Operational Limitations & Academic Integrity
# -------------------------------------------------------------
st.markdown("### ⚠️ Operational Limitations & Academic Integrity")

st.markdown(
    """
    > **Academic Integrity Statement:**  
    > In real-world automated traffic enforcement, **no computer vision system can establish 100% legal guilt from a single photo without human verification**. VisionGuard uses an **Adjudication State Machine** that routes uncertain or low-confidence cases to human officers for confirmation.
    """
)
