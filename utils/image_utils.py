"""
Vision Guard — Image Processing & Visualization Utilities
Surveillance-grade HUD overlays, bounding boxes, crops, and synthetic sample generation.
"""

import os
import cv2
import numpy as np
import base64
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont


def draw_hud_box(image: np.ndarray, bbox: Tuple[int, int, int, int], label: str,
                 conf: float = 0.0, color: Tuple[int, int, int] = (0, 255, 0),
                 track_id: Optional[str] = None, is_violation: bool = False) -> np.ndarray:
    """
    Draws a surveillance HUD-style bounding box with corner brackets and tactical badges.
    color: BGR tuple (e.g., (0, 0, 255) for red alert, (255, 200, 0) for cyan)
    """
    img = image.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    box_color = (0, 0, 255) if is_violation else color
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)

    # Tactical corner brackets
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    if corner_len > 3:
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), box_color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), box_color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), box_color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), box_color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), box_color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), box_color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), box_color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), box_color, thickness)

    # Label text & tag badge
    badge_text = label
    if track_id:
        badge_text = f"[{track_id}] {badge_text}"
    if conf > 0:
        badge_text += f" {conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

    # Background badge
    badge_y1 = max(0, y1 - th - 8)
    badge_y2 = y1
    cv2.rectangle(img, (x1, badge_y1), (x1 + tw + 10, badge_y2), (20, 24, 34), -1)
    cv2.rectangle(img, (x1, badge_y1), (x1 + tw + 10, badge_y2), box_color, 1)

    # Text
    cv2.putText(img, badge_text, (x1 + 5, y1 - 4), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return img


def safe_crop(image: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.05) -> np.ndarray:
    """Safely extracts a region crop from an image with optional margin padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin)
    my = int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    return image[y1:y2, x1:x2].copy()


def image_to_base64(image_path_or_arr) -> str:
    """Converts an image file or numpy array to a base64 string."""
    if isinstance(image_path_or_arr, str):
        if not os.path.exists(image_path_or_arr):
            return ""
        with open(image_path_or_arr, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_path_or_arr, np.ndarray):
        success, encoded = cv2.imencode(".jpg", image_path_or_arr)
        if success:
            return base64.b64encode(encoded).decode("utf-8")
    return ""


def create_synthetic_sample(sample_type: str, output_path: str) -> str:
    """
    Creates high-quality synthetic visual test images for demonstration when
    external assets are unavailable.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas = np.zeros((480, 720, 3), dtype=np.uint8)

    # Road background
    canvas[:] = (45, 52, 54) # Asphalt
    # Lane dividers
    for x in range(30, 720, 90):
        cv2.rectangle(canvas, (x, 235), (x + 50, 245), (255, 255, 255), -1)

    if sample_type == "helmet":
        # Draw motorcycle + rider
        # Motorcycle chassis
        cv2.rectangle(canvas, (260, 240), (460, 360), (30, 30, 180), -1)
        cv2.circle(canvas, (290, 360), 45, (10, 10, 10), -1) # Wheels
        cv2.circle(canvas, (430, 360), 45, (10, 10, 10), -1)
        cv2.circle(canvas, (290, 360), 20, (180, 180, 180), -1)
        cv2.circle(canvas, (430, 360), 20, (180, 180, 180), -1)

        # Rider torso
        cv2.rectangle(canvas, (330, 150), (400, 270), (50, 50, 50), -1) # Dark jacket
        # Head (No helmet: exposed hair / skin tone)
        cv2.circle(canvas, (365, 115), 32, (120, 160, 210), -1) # Skin tone BGR
        cv2.ellipse(canvas, (365, 95), (28, 18), 0, 0, 180, (20, 20, 20), -1) # Hair
        cv2.putText(canvas, "TEST SAMPLE: MOTORCYCLE RIDER (NO HELMET)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif sample_type == "seatbelt":
        # Vehicle windshield & occupant
        cv2.rectangle(canvas, (180, 100), (540, 400), (120, 120, 120), -1) # Car front
        cv2.rectangle(canvas, (220, 130), (500, 270), (190, 210, 220), -1) # Windshield
        # Driver occupant
        cv2.circle(canvas, (300, 180), 30, (130, 170, 220), -1) # Driver head
        cv2.rectangle(canvas, (270, 210), (350, 270), (220, 80, 50), -1) # Orange shirt (No belt across)
        # Passenger
        cv2.circle(canvas, (420, 180), 30, (130, 170, 220), -1)
        cv2.rectangle(canvas, (390, 210), (470, 270), (60, 140, 60), -1)
        cv2.putText(canvas, "TEST SAMPLE: CAR OCCUPANT (SEATBELT AUDIT)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif sample_type == "tint":
        # Vehicle side profile with dark windows
        cv2.rectangle(canvas, (150, 180), (580, 360), (200, 200, 200), -1) # Silver car body
        # Heavily tinted dark windows
        cv2.rectangle(canvas, (240, 190), (340, 260), (15, 15, 15), -1) # Front side window (Very dark)
        cv2.rectangle(canvas, (360, 190), (480, 260), (10, 10, 10), -1) # Rear window
        cv2.circle(canvas, (220, 360), 40, (20, 20, 20), -1) # Wheels
        cv2.circle(canvas, (510, 360), 40, (20, 20, 20), -1)
        cv2.putText(canvas, "TEST SAMPLE: VEHICLE DARK TINT PROFILE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif sample_type == "plate":
        # Vehicle rear bumper with registration plate
        cv2.rectangle(canvas, (140, 120), (580, 380), (160, 30, 30), -1) # Red Car
        # License plate box
        cv2.rectangle(canvas, (270, 280), (450, 340), (255, 255, 255), -1) # White Plate
        cv2.rectangle(canvas, (270, 280), (450, 340), (0, 0, 0), 3) # Border
        # Plate blue strip
        cv2.rectangle(canvas, (270, 280), (290, 340), (200, 50, 0), -1) # IND Blue strip
        cv2.putText(canvas, "DL01AB1234", (295, 322),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, "TEST SAMPLE: VEHICLE LICENSE PLATE OCR", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(output_path, canvas)
    return output_path
