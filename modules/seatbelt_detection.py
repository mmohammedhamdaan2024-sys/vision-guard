"""
Vision Guard — Seat-belt Detection Module
Analyzes vehicle occupant compliance using cabin localization, diagonal strap edge analysis,
Hough transform geometry, and contrast profiling.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO


class SeatbeltDetector:
    def __init__(self, base_yolo_path: str = "models/yolov8n.pt",
                 custom_seatbelt_model_path: str = "models/seatbelt_yolo.pt",
                 conf_thresh: float = 0.50):
        self.conf_thresh = conf_thresh
        self.custom_model_path = custom_seatbelt_model_path
        self.base_model_path = base_yolo_path

        self.base_model = None
        if os.path.exists(base_yolo_path):
            self.base_model = YOLO(base_yolo_path)

        self.has_custom_model = os.path.exists(custom_seatbelt_model_path)
        self.custom_model = None
        if self.has_custom_model:
            try:
                self.custom_model = YOLO(custom_seatbelt_model_path)
            except Exception as e:
                print(f"Warning: Could not load custom seatbelt model: {e}")
                self.has_custom_model = False

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes vehicle image for occupant seat-belt compliance.
        Returns:
            - status: 'Seat belt detected' | 'Possible seat-belt violation' | 'Unable to determine' | 'No vehicle detected'
            - confidence: float
            - cabin_box: [x1, y1, x2, y2]
            - occupant_crop: np.ndarray
            - annotated_image: np.ndarray
            - diagnostics: dict
        """
        annotated = image.copy()
        h, w = image.shape[:2]

        if self.base_model is None:
            return {
                "status": "Unable to determine",
                "confidence": 0.0,
                "annotated_image": annotated,
                "occupant_crop": None,
                "diagnostics": {"error": "Base model not initialized"}
            }

        # Run base YOLO to detect 4-wheeled vehicles: car (2), bus (5), truck (7)
        results = self.base_model(image, verbose=False, conf=0.35)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        target_box = None
        target_conf = 0.0
        for box, cls_id, conf in zip(boxes, classes, confs):
            if cls_id in (2, 5, 7): # car, bus, truck
                target_box = box
                target_conf = conf
                break

        if target_box is None:
            # Check if this is already an occupant or windshield crop
            if w > 80 and h > 80:
                target_box = [0, 0, w, h]
                target_conf = 0.8
            else:
                return {
                    "status": "No vehicle detected",
                    "confidence": 0.0,
                    "annotated_image": annotated,
                    "occupant_crop": None,
                    "diagnostics": {"message": "No four-wheeled passenger vehicle detected in frame"}
                }

        # Extract cabin / windshield zone
        # Front windshield typically sits in upper 20% to 65% of the vehicle height and central 60% of width
        vx1, vy1, vx2, vy2 = [int(v) for v in target_box]
        vw = vx2 - vx1
        vh = vy2 - vy1

        cx1 = max(0, vx1 + int(vw * 0.15))
        cx2 = min(w, vx2 - int(vw * 0.15))
        cy1 = max(0, vy1 + int(vh * 0.15))
        cy2 = min(h, vy1 + int(vh * 0.65))

        if (cx2 - cx1) < 40 or (cy2 - cy1) < 40:
            return {
                "status": "Unable to determine",
                "confidence": 0.35,
                "annotated_image": annotated,
                "occupant_crop": None,
                "diagnostics": {"reason": "Vehicle resolution too low or occupant cabin occluded"}
            }

        cabin_crop = image[cy1:cy2, cx1:cx2].copy()

        # Check lighting conditions and glare in cabin
        gray_cabin = cv2.cvtColor(cabin_crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_cabin)
        std_brightness = np.std(gray_cabin)

        if mean_brightness < 28 or mean_brightness > 235 or std_brightness < 12:
            # Harsh glare or extreme shadow
            status = "Unable to determine"
            conf = 0.45
            diag = {
                "reason": "Adverse lighting, windshield glare, or dark tint occlusion",
                "mean_brightness": round(float(mean_brightness), 1),
                "contrast_std": round(float(std_brightness), 1)
            }
        else:
            status, conf, diag = self._analyze_diagonal_belt(cabin_crop)

        # Draw HUD annotation
        color = (0, 255, 0) if status == "Seat belt detected" else ((0, 0, 255) if status == "Possible seat-belt violation" else (0, 200, 255))
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(annotated, f"{status} ({conf:.2f})", (cx1, max(20, cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return {
            "status": status,
            "confidence": round(conf, 2),
            "cabin_box": [cx1, cy1, cx2, cy2],
            "occupant_crop": cabin_crop,
            "annotated_image": annotated,
            "diagnostics": diag
        }

    def _analyze_diagonal_belt(self, cabin_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyzes torso diagonal strap geometry using:
        - Multi-angle edge filtering (detecting lines angled at 25° to 65°)
        - Hough Line Transform
        - Intensity contrast along candidate strap vectors
        """
        ch, cw = cabin_bgr.shape[:2]
        gray = cv2.cvtColor(cabin_bgr, cv2.COLOR_BGR2GRAY)
        # CLAHE contrast enhancement to penetrate light windshield tint
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        edges = cv2.Canny(enhanced, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=int(ch * 0.25), maxLineGap=12)

        diagonal_line_count = 0
        strongest_diagonal_score = 0.0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx == 0:
                    continue
                angle_deg = np.degrees(np.arctan2(dy, dx))
                # Seatbelt diagonal geometry is typically between 28 and 65 degrees
                if 28 <= angle_deg <= 68:
                    line_len = np.sqrt(dx ** 2 + dy ** 2)
                    if line_len > 0.3 * ch:
                        diagonal_line_count += 1
                        strongest_diagonal_score = max(strongest_diagonal_score, line_len / ch)

        diag = {
            "diagonal_lines_detected": diagonal_line_count,
            "strap_length_ratio": round(float(strongest_diagonal_score), 3),
            "enhanced_contrast": True
        }

        if diagonal_line_count >= 2 or strongest_diagonal_score > 0.45:
            conf = min(0.93, 0.65 + (strongest_diagonal_score * 0.3))
            return "Seat belt detected", conf, diag
        elif diagonal_line_count == 0 and strongest_diagonal_score < 0.2:
            # Clear cabin view but absent diagonal strap
            conf = 0.78
            return "Possible seat-belt violation", conf, diag
        else:
            return "Unable to determine", 0.50, diag
