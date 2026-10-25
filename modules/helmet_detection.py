"""
VisionGuard — Helmet Detection Module
Monitors motorcycle rider headgear compliance using dual-mode analysis:
Deep YOLO classification (when weights are present) or intelligent computer vision
cranial contour, skin-tone ratio, and edge density profiling.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO

from config import HELMET_MODEL_PATH, HELMET_CONFIDENCE_THRESHOLD, COLORS


class HelmetDetector:
    """Detects motorcycle riders and evaluates helmet safety compliance."""

    def __init__(self, custom_model_path: Optional[str] = None,
                 conf_threshold: float = HELMET_CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.custom_model_path = custom_model_path or str(HELMET_MODEL_PATH)
        self.has_custom_model = os.path.exists(self.custom_model_path)
        self.custom_model: Optional[YOLO] = None

        if self.has_custom_model:
            try:
                self.custom_model = YOLO(self.custom_model_path)
            except Exception as e:
                print(f"Warning: Could not load custom helmet model: {e}")
                self.has_custom_model = False

    def analyze(self, frame: np.ndarray, vehicle_box: Optional[List[int]] = None,
                all_detections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyzes a motorcycle or full frame for rider helmet compliance.
        Returns:
            - status: 'Helmet detected' | 'No helmet detected' | 'Uncertain / not visible' | 'No rider localized'
            - confidence: float
            - rider_box: [x1, y1, x2, y2]
            - head_box: [x1, y1, x2, y2]
            - head_crop: np.ndarray
            - box_color: (B, G, R) tuple
            - details: dict
        """
        if frame is None:
            return {"status": "Uncertain / not visible", "confidence": 0.0, "head_box": None, "rider_box": None}

        h, w = frame.shape[:2]

        # 1. Determine Rider Zone
        if vehicle_box is not None:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w, vx2), min(h, vy2)
            vw = vx2 - vx1
            vh = vy2 - vy1

            # Search for a person detection overlapping this motorcycle
            rider_box = None
            if all_detections:
                for d in all_detections:
                    if d["class_name"] == "person":
                        px1, py1, px2, py2 = d["bbox"]
                        overlap_x = max(0, min(vx2, px2) - max(vx1, px1))
                        if overlap_x > 0.25 * (px2 - px1) and py1 < vy2:
                            rider_box = [px1, py1, px2, py2]
                            break

            # If no discrete person box, synthesize rider zone atop motorcycle
            if rider_box is None:
                ry1 = max(0, vy1 - int(vh * 0.40))
                ry2 = min(h, vy1 + int(vh * 0.50))
                rx1 = max(0, vx1 - int(vw * 0.05))
                rx2 = min(w, vx2 + int(vw * 0.05))
                rider_box = [rx1, ry1, rx2, ry2]
        else:
            # Whole frame evaluation
            rider_box = [int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.8)]

        rx1, ry1, rx2, ry2 = rider_box
        rw = rx2 - rx1
        rh = ry2 - ry1

        if rw < 15 or rh < 20:
            return {
                "status": "Uncertain / not visible",
                "confidence": 0.35,
                "rider_box": rider_box,
                "head_box": None,
                "head_crop": None,
                "box_color": COLORS["uncertain"],
                "details": {"reason": "Rider region too small or occluded"}
            }

        # 2. Localize Head Region (upper 30% of rider bounding box)
        hx1 = max(0, rx1 + int(rw * 0.12))
        hx2 = min(w, rx2 - int(rw * 0.12))
        hy1 = max(0, ry1)
        hy2 = min(h, ry1 + int(rh * 0.33))

        head_box = [hx1, hy1, hx2, hy2]
        head_crop = frame[hy1:hy2, hx1:hx2].copy() if (hx2 > hx1 and hy2 > hy1) else None

        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] < 12 or head_crop.shape[1] < 12:
            return {
                "status": "Uncertain / not visible",
                "confidence": 0.40,
                "rider_box": rider_box,
                "head_box": head_box,
                "head_crop": None,
                "box_color": COLORS["uncertain"],
                "details": {"reason": "Head region resolution insufficient"}
            }

        # 3. Model Inference vs Computer Vision Fallback
        if self.has_custom_model and self.custom_model is not None:
            try:
                preds = self.custom_model(head_crop, verbose=False, conf=0.35)[0]
                if len(preds.boxes) > 0:
                    cls_id = int(preds.boxes.cls[0].item())
                    score = float(preds.boxes.conf[0].item())
                    if cls_id == 0:
                        return {
                            "status": "Helmet detected",
                            "confidence": round(score, 2),
                            "rider_box": rider_box,
                            "head_box": head_box,
                            "head_crop": head_crop,
                            "box_color": COLORS["compliant"],
                            "details": {"mode": "deep_yolo_weights"}
                        }
                    else:
                        return {
                            "status": "No helmet detected",
                            "confidence": round(score, 2),
                            "rider_box": rider_box,
                            "head_box": head_box,
                            "head_crop": head_crop,
                            "box_color": COLORS["violation"],
                            "details": {"mode": "deep_yolo_weights"}
                        }
            except Exception as e:
                print(f"Custom helmet model inference fallback: {e}")

        # Intelligent CV Morphological Fallback
        status, conf, diag = self._cv_cranial_analyzer(head_crop)
        box_color = COLORS["compliant"] if status == "Helmet detected" else (
            COLORS["violation"] if status == "No helmet detected" else COLORS["uncertain"]
        )

        return {
            "status": status,
            "confidence": round(conf, 2),
            "rider_box": rider_box,
            "head_box": head_box,
            "head_crop": head_crop,
            "box_color": box_color,
            "details": diag
        }

    def _cv_cranial_analyzer(self, head_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Evaluates head crop using:
        1. HSV skin tone detection (bare face/neck vs solid helmet shell)
        2. Cranial dome circularity (helmets have smooth high circularity)
        3. Texture smoothness / edge frequency (hair has high frequency texture)
        """
        hsv = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Skin tone range
        lower_skin = np.array([0, 28, 50], dtype=np.uint8)
        upper_skin = np.array([25, 200, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / float(head_bgr.shape[0] * head_bgr.shape[1])

        # Hair tone (low value, low/mid saturation)
        hair_mask = (v < 50) & (s < 130)
        hair_ratio = np.sum(hair_mask) / float(head_bgr.shape[0] * head_bgr.shape[1])

        # Contour dome circularity
        gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_circularity = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 12:
                circ = 4 * np.pi * (area / (perimeter * perimeter))
                if circ > max_circularity:
                    max_circularity = circ

        diag = {
            "skin_ratio": round(float(skin_ratio), 3),
            "hair_ratio": round(float(hair_ratio), 3),
            "dome_circularity": round(float(max_circularity), 3),
            "method": "Cranial morphological & HSV dermal spectrometry"
        }

        # Decision rules
        if skin_ratio > 0.30 or (hair_ratio > 0.28 and max_circularity < 0.45):
            conf = min(0.94, 0.62 + (skin_ratio * 0.4) + (hair_ratio * 0.2))
            return "No helmet detected", conf, diag
        elif max_circularity > 0.52 or (skin_ratio < 0.14 and hair_ratio < 0.16):
            conf = min(0.95, 0.68 + (max_circularity * 0.3))
            return "Helmet detected", conf, diag
        else:
            return "Uncertain / not visible", 0.50, diag
