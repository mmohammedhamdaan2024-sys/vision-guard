"""
VisionGuard — Page 5: AI Model & Architecture Information
Technical specifications, computer vision pipeline diagnostics, hardware acceleration,
class mappings, model limitations, and custom dataset training documentation.
"""

import os
import sys
import torch
import cv2
import streamlit as st
import numpy as np

from config import (
    BASE_DIR, MODEL_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_STOP_LINE_Y_RATIO, TARGET_CLASSES, VIOLATION_TYPES
)
from utils.ui_components import inject_custom_css, render_surveillance_header

st.set_page_config(page_title="Model Information | VisionGuard", page_icon="🧠", layout="wide")
inject_custom_css()

render_surveillance_header(
    title="AI MODEL SPECIFICATIONS & VISION ARCHITECTURE",
    subtitle="Technical telemetry, neural network architecture, hardware diagnostics, and pipeline methodologies.",
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
# 2. Pipeline Architecture & Active Subsystems
# -------------------------------------------------------------
st.markdown("### 🔬 Multi-Stage Vision Pipeline Architecture")

t1, t2, t3 = st.tabs(["🎯 Detection & Tracking", "🪖 Safety Analysis Modules", "🚦 Stop-Line & Geometry"])

with t1:
    st.markdown(
        """
        #### 1. Vehicle & Road User Detection (YOLOv8)
        - **Primary Backbone:** Ultralytics YOLOv8 Nano (`yolov8n.pt`) with pretrained weights.
        - **Monitored Class Map:** `car`, `motorcycle`, `bus`, `truck`, `bicycle`, `person`.
        - **Inference Precision:** Real-time bounding box extraction, class probability distribution, and confidence thresholding (`conf >= 0.40`).
        - **Tracking Algorithm:** Centroid-distance and IoU matching with a persistence window of up to 30 absent frames.
        - **Motion Vectoring:** Continuous displacement tracking across sequential frames calculates instantaneous trajectory angle and direction (`downward`, `upward`, `stationary`).
        """
    )
    st.code(
        f"Target Monitored Classes: {list(TARGET_CLASSES.values())}\n"
        f"Inference Threshold: {DEFAULT_CONFIDENCE_THRESHOLD}\n"
        f"Tracking Metric: Euclidean Centroid Matching + Directional Displacement",
        language="yaml"
    )

with t2:
    st.markdown(
        """
        #### 2. Motorcycle Helmet Analysis Module
        - **Region of Interest (ROI):** Isolates the top 35% of motorcycle rider bounding boxes.
        - **Geometric & Colorimetric Assessment:**
          - Evaluates cranial contour curvature, circularity index, and boundary smoothness.
          - Analyzes HSV color distribution to detect exposed skin, hair tones, or synthetic helmet shells.
        - **Dual Mode Operation:** Automatically checks for custom-trained YOLO weights (`models/helmet_detector.pt`). If custom weights are present, runs multi-class helmet inference; otherwise falls back to the robust geometric heuristic.
        - **Honest Uncertainty:** Cases with low contrast, backlighting, or severe occlusion are classified as `Uncertain / not visible` rather than generating false positive citations.

        #### 3. Vehicle Occupant Seat-Belt Analysis Module
        - **Cabin Isolation:** Crops the upper 45% windshield area of detected passenger cars and commercial vehicles.
        - **Contrast Enhancement (CLAHE):** Applies Contrast Limited Adaptive Histogram Equalization to neutralize windshield glare, sky reflections, and tinted glass attenuation.
        - **Diagonal Gradient & Hough Transform:** Computes morphological gradients and detects diagonal linear segments oriented between 20° and 70°.
        - **Classification Output:** `Seat belt detected`, `Possible seat-belt violation`, or `Unable to determine`.
        """
    )

with t3:
    st.markdown(
        """
        #### 4. Traffic Signal & Virtual Stop-Line Monitor
        - **Virtual Stop Line:** Calibrated geometric boundary placed across the road corridor at a configurable vertical ratio (default `y = 65%`).
        - **Signal State Tracking:** Multi-band HSV color segmentation detects active signal state (`RED`, `YELLOW`, `GREEN`) from the traffic light mast.
        - **Incursion Invariant:** An infraction is triggered if and only if:
          1. The active traffic signal is **RED**.
          2. The lower centroid edge of a vehicle crosses the virtual stop line from above to below.
          3. The vehicle ID has not previously been cited for the same event (duplicate suppression).
        """
    )

st.markdown("---")

# -------------------------------------------------------------
# 3. Model Limitations & Engineering Integrity
# -------------------------------------------------------------
st.markdown("### ⚠️ Operational Limitations & Academic Integrity")

st.markdown(
    """
    > **Academic Integrity Statement:**  
    > In real-world automated traffic enforcement, **no single-camera computer vision system can establish 100% legal guilt without human verification**. VisionGuard is deliberately designed with an **Adjudication State Machine** that routes uncertain or low-confidence frames for human officer confirmation.
    """
)

lim_col1, lim_col2 = st.columns(2)

with lim_col1:
    st.markdown(
        """
        <div style="background: #1C2541; border-left: 4px solid #FFE600; padding: 14px 18px; border-radius: 4px; margin-bottom: 12px;">
            <b style="color: #FFE600;">Windshield Reflections & Nighttime Bloom</b>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                Direct sunlight reflections on curved laminated windshields can completely obscure cabin interiors. 
                In such conditions, seatbelt visibility drops below confidence thresholds and the system outputs <i>Unable to determine</i>.
            </p>
        </div>
        <div style="background: #1C2541; border-left: 4px solid #FFE600; padding: 14px 18px; border-radius: 4px;">
            <b style="color: #FFE600;">Occlusion in Heavy Congestion</b>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                When two-wheelers ride closely behind larger commercial trucks or buses, the cranial region may be partially occluded.
                The system avoids false penalties by verifying tracking persistence over multiple frames.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with lim_col2:
    st.markdown(
        """
        <div style="background: #1C2541; border-left: 4px solid #3A86FF; padding: 14px 18px; border-radius: 4px; margin-bottom: 12px;">
            <b style="color: #3A86FF;">Optical Perspective & Camera Angles</b>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                Seat-belt detection requires a forward-facing camera tilted at 15°–25° toward oncoming windshields.
                Perpendicular overhead gantry cameras cannot observe cabin occupants and must disable seat-belt checks.
            </p>
        </div>
        <div style="background: #1C2541; border-left: 4px solid #3A86FF; padding: 14px 18px; border-radius: 4px;">
            <b style="color: #3A86FF;">Custom Domain Fine-Tuning</b>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                Generic YOLO COCO weights do not include explicit 'seatbelt' or 'helmet' classes.
                VisionGuard includes complete training scripts for users wishing to fine-tune on domain-specific datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Custom Model Training & Dataset Pipeline
# -------------------------------------------------------------
st.markdown("### 🛠️ Custom Training Scripts & Dataset Pipeline")

st.markdown(
    """
    VisionGuard includes production-grade automation scripts for dataset downloading, YOLO annotation conversion,
    and transfer-learning fine-tuning:
    """
)

s_c1, s_c2, s_c3 = st.columns(3)

with s_c1:
    st.markdown(
        """
        **1. Dataset Preparation**  
        `scripts/prepare_dataset.py`  
        - Validates image dimensions and YOLO bounding box labels.  
        - Splits data into 70% Train, 20% Val, 10% Test.  
        - Generates dataset configuration YAML.
        """
    )
with s_c2:
    st.markdown(
        """
        **2. Fine-Tuning Execution**  
        `scripts/train_model.py`  
        - Configures transfer learning from `yolov8n.pt`.  
        - Hyperparameters: AdamW optimizer, lr0=0.01, cosine lr.  
        - Exports checkpoint weights directly to `models/`.
        """
    )
with s_c3:
    st.markdown(
        """
        **3. Evaluation & Metrics**  
        `scripts/evaluate_model.py`  
        - Calculates Precision, Recall, mAP@0.5, and mAP@0.5:0.95.  
        - Generates confusion matrices and PR curves.  
        - Validates latency and inference FPS.
        """
    )
