"""
VisionGuard — Seat Belt Detection Module
Monitors visible front-seat vehicle occupants for safety belt compliance.
Uses cabin localization, CLAHE contrast enhancement, and diagonal Hough strap geometry.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO

from config import SEATBELT_MODEL_PATH, SEATBELT_CONFIDENCE_THRESHOLD, COLORS


class SeatbeltDetector:
    """Analyzes passenger vehicle cabin for driver and front occupant seat-belt presence."""

    def __init__(self, custom_model_path: Optional[str] = None,
                 conf_threshold: float = SEATBELT_CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.custom_model_path = custom_model_path or str(SEATBELT_MODEL_PATH)
        self.has_custom_model = os.path.exists(self.custom_model_path)
        self.custom_model: Optional[YOLO] = None

        if self.has_custom_model:
            try:
                self.custom_model = YOLO(self.custom_model_path)
            except Exception as e:
                print(f"Warning: Could not load custom seatbelt model: {e}")
                self.has_custom_model = False

    def analyze(self, frame: np.ndarray, vehicle_box: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyzes a car/vehicle for occupant seat belt compliance.
        Returns:
            - status: 'Seat belt detected' | 'Possible seat-belt violation' | 'Unable to determine'
            - confidence: float
            - cabin_box: [x1, y1, x2, y2]
            - occupant_crop: np.ndarray
            - box_color: (B, G, R) tuple
            - details: dict
        """
        if frame is None:
            return {"status": "Unable to determine", "confidence": 0.0, "cabin_box": None, "occupant_crop": None}

        h, w = frame.shape[:2]

        if vehicle_box is not None:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
        else:
            vx1, vy1, vx2, vy2 = 0, 0, w, h

        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)
        vw = vx2 - vx1
        vh = vy2 - vy1

        if vw < 40 or vh < 40:
            return {
                "status": "Unable to determine",
                "confidence": 0.30,
                "cabin_box": None,
                "occupant_crop": None,
                "box_color": COLORS["uncertain"],
                "details": {"reason": "Vehicle resolution too low"}
            }

        # Windshield / Cabin Zone: upper 15% to 62% of vehicle height, middle 70% of width
        cx1 = max(0, vx1 + int(vw * 0.15))
        cx2 = min(w, vx2 - int(vw * 0.15))
        cy1 = max(0, vy1 + int(vh * 0.14))
        cy2 = min(h, vy1 + int(vh * 0.62))

        cabin_box = [cx1, cy1, cx2, cy2]
        cabin_crop = frame[cy1:cy2, cx1:cx2].copy() if (cx2 > cx1 and cy2 > cy1) else None

        if cabin_crop is None or cabin_crop.size == 0 or cabin_crop.shape[0] < 20 or cabin_crop.shape[1] < 20:
            return {
                "status": "Unable to determine",
                "confidence": 0.35,
                "cabin_box": cabin_box,
                "occupant_crop": None,
                "box_color": COLORS["uncertain"],
                "details": {"reason": "Windshield cabin crop insufficient"}
            }

        # Check lighting conditions and glare
        gray = cv2.cvtColor(cabin_crop, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        if mean_lum < 25 or mean_lum > 238 or contrast_std < 10:
            return {
                "status": "Unable to determine",
                "confidence": 0.40,
                "cabin_box": cabin_box,
                "occupant_crop": cabin_crop,
                "box_color": COLORS["uncertain"],
                "details": {"reason": "Adverse lighting, reflection, or opaque window tint"}
            }

        # Deep Model vs Diagonal Hough Edge Profiler
        if self.has_custom_model and self.custom_model is not None:
            try:
                preds = self.custom_model(cabin_crop, verbose=False, conf=0.35)[0]
                if len(preds.boxes) > 0:
                    cls_id = int(preds.boxes.cls[0].item())
                    score = float(preds.boxes.conf[0].item())
                    status = "Seat belt detected" if cls_id == 0 else "Possible seat-belt violation"
                    color = COLORS["compliant"] if cls_id == 0 else COLORS["violation"]
                    return {
                        "status": status,
                        "confidence": round(score, 2),
                        "cabin_box": cabin_box,
                        "occupant_crop": cabin_crop,
                        "box_color": color,
                        "details": {"mode": "deep_yolo_weights"}
                    }
            except Exception as e:
                print(f"Custom seatbelt model fallback: {e}")

        # Computer Vision Diagonal Strap Edge Profiling
        status, conf, diag = self._cv_strap_analyzer(cabin_crop)
        box_color = COLORS["compliant"] if status == "Seat belt detected" else (
            COLORS["violation"] if status == "Possible seat-belt violation" else COLORS["uncertain"]
        )

        return {
            "status": status,
            "confidence": round(conf, 2),
            "cabin_box": cabin_box,
            "occupant_crop": cabin_crop,
            "box_color": box_color,
            "details": diag
        }

    def _cv_strap_analyzer(self, cabin_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Enhances windshield contrast with CLAHE and searches for diagonal line segments
        angled between 28° and 68° corresponding to shoulder-to-opposite-hip strap orientation.
        """
        ch, cw = cabin_bgr.shape[:2]
        gray = cv2.cvtColor(cabin_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        edges = cv2.Canny(enhanced, 45, 140)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=22, minLineLength=int(ch * 0.25), maxLineGap=12)

        diagonal_straps = 0
        max_strap_ratio = 0.0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx == 0:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                if 28 <= angle <= 68:
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    if length > 0.28 * ch:
                        diagonal_straps += 1
                        max_strap_ratio = max(max_strap_ratio, length / ch)

        diag = {
            "diagonal_straps_found": diagonal_straps,
            "strap_length_ratio": round(float(max_strap_ratio), 3),
            "method": "CLAHE contrast + Hough directional line transformation"
        }

        if diagonal_straps >= 1 and max_strap_ratio > 0.35:
            conf = min(0.93, 0.68 + (max_strap_ratio * 0.25))
            return "Seat belt detected", conf, diag
        elif diagonal_straps == 0 and max_strap_ratio < 0.15:
            return "Possible seat-belt violation", 0.82, diag
        else:
            return "Unable to determine", 0.50, diag
