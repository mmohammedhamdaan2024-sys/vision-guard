"""
VisionGuard — Configuration Module
Centralized settings for detection models, thresholds, camera configuration, and storage paths.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVIDENCE_DIR = BASE_DIR / "evidence"
REPORTS_DIR = DATA_DIR / "reports"
SAMPLES_DIR = BASE_DIR / "samples"
DATABASE_DIR = BASE_DIR / "database"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "visionguard.db"

# Deep Learning Models
YOLO_BASE_MODEL = MODELS_DIR / "yolov8n.pt"
HELMET_MODEL_PATH = MODELS_DIR / "helmet_yolo.pt"
SEATBELT_MODEL_PATH = MODELS_DIR / "seatbelt_yolo.pt"

# Object Detection Classes (COCO indices for traffic surveillance)
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light"
}

# Detection & Violation Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
HELMET_CONFIDENCE_THRESHOLD = 0.50
SEATBELT_CONFIDENCE_THRESHOLD = 0.50
PLATE_CONFIDENCE_THRESHOLD = 0.40

# Traffic Signal & Stop Line Configuration
DEFAULT_STOP_LINE_Y_RATIO = 0.65  # Stop line at 65% of frame height
DEFAULT_SIGNAL_STATE = "RED"      # 'RED', 'YELLOW', 'GREEN'

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_TARGET_FPS = 30

# Violation Categories
VIOLATION_TYPES = [
    "Helmet Violation",
    "Seat-Belt Violation",
    "Red Light Stop-Line Violation",
    "Wrong-Way Movement",
    "Restricted Zone Entry"
]

# Review Statuses
REVIEW_STATUSES = [
    "Pending Review",
    "Confirmed",
    "Rejected",
    "Inconclusive"
]

# Color Palette (BGR for OpenCV and Hex for UI)
COLORS = {
    "compliant": (0, 255, 0),        # Neon Green
    "violation": (0, 0, 255),        # Bright Red
    "uncertain": (0, 200, 255),      # Amber/Yellow
    "vehicle": (255, 200, 0),        # Cyan
    "person": (255, 120, 0),         # Orange-Blue
    "stop_line_red": (0, 0, 255),    # Red stop line
    "stop_line_green": (0, 255, 0),  # Green stop line
    "hud_bg": (20, 24, 34)           # Dark HUD background
}
