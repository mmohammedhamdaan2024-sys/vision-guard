"""
Vision Guard — Helmet Detection Module
Detects motorcycles, riders, and analyzes helmet compliance with dual-mode inference
(dedicated deep learning model if available, with intelligent computer vision fallback).
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO


class HelmetDetector:
    def __init__(self, base_yolo_path: str = "models/yolov8n.pt",
                 custom_helmet_model_path: str = "models/helmet_yolo.pt",
                 conf_thresh: float = 0.50):
        self.conf_thresh = conf_thresh
        self.custom_model_path = custom_helmet_model_path
        self.base_model_path = base_yolo_path

        # Load Base Model (COCO classes: person=0, motorcycle=3)
        self.base_model = None
        if os.path.exists(base_yolo_path):
            self.base_model = YOLO(base_yolo_path)

        # Check for dedicated custom-trained helmet model
        self.has_custom_model = os.path.exists(custom_helmet_model_path)
        self.custom_model = None
        if self.has_custom_model:
            try:
                self.custom_model = YOLO(custom_helmet_model_path)
            except Exception as e:
                print(f"Warning: Could not load custom helmet model: {e}")
                self.has_custom_model = False

    def is_custom_model_available(self) -> bool:
        return self.has_custom_model

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes an image for motorcycles, riders, and helmet safety status.
        Returns:
            - status: 'Helmet detected' | 'No helmet detected' | 'Uncertain / not visible' | 'No motorcycle detected'
            - confidence: float
            - motorcycle_box: [x1, y1, x2, y2]
            - rider_box: [x1, y1, x2, y2]
            - head_box: [x1, y1, x2, y2]
            - head_crop: np.ndarray
            - annotated_image: np.ndarray
            - details: dict
        """
        if self.base_model is None:
            return {
                "status": "Uncertain / not visible",
                "confidence": 0.0,
                "error": "Base YOLO model not loaded",
                "annotated_image": image.copy(),
                "details": {"mode": "error"}
            }

        annotated = image.copy()
        h, w = image.shape[:2]

        # 1. Base inference for motorcycle (class 3) and person (class 0)
        results = self.base_model(image, verbose=False, conf=0.35)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        motorcycles = []
        persons = []

        for box, cls_id, conf in zip(boxes, classes, confs):
            if cls_id == 3: # motorcycle
                motorcycles.append((box, conf))
            elif cls_id == 0: # person
                persons.append((box, conf))

        if not motorcycles:
            # If no motorcycle detected, check if image was already a cropped motorcycle image
            # In that case, look for person or treat upper region as rider candidate
            if persons:
                # Treat person as candidate rider
                rider_box, r_conf = persons[0]
                return self._evaluate_rider_head(image, annotated, rider_box, r_conf, None)
            
            return {
                "status": "No motorcycle detected",
                "confidence": 0.0,
                "motorcycle_box": None,
                "rider_box": None,
                "head_box": None,
                "head_crop": None,
                "annotated_image": annotated,
                "details": {"message": "No two-wheeler or rider found in frame"}
            }

        # Associate rider with motorcycle by bounding box overlap
        best_pair = None
        for m_box, m_conf in motorcycles:
            mx1, my1, mx2, my2 = m_box
            for p_box, p_conf in persons:
                px1, py1, px2, py2 = p_box
                # Rider usually overlaps or sits above the motorcycle
                # Check horizontal overlap and vertical proximity
                x_overlap = max(0, min(mx2, px2) - max(mx1, px1))
                if x_overlap > 0.2 * (px2 - px1) and py1 < my2:
                    best_pair = (m_box, p_box, m_conf, p_conf)
                    break
            if best_pair:
                break

        if not best_pair:
            # Motorcycle detected without clearly localized separate person box
            # Rider might be merged inside the motorcycle box (common in rear shots)
            m_box, m_conf = motorcycles[0]
            # Synthesize upper 45% of motorcycle box as rider zone
            mx1, my1, mx2, my2 = m_box
            synth_rider = [mx1, my1, mx2, my1 + (my2 - my1) * 0.55]
            return self._evaluate_rider_head(image, annotated, synth_rider, m_conf * 0.8, m_box)

        m_box, p_box, m_conf, p_conf = best_pair
        return self._evaluate_rider_head(image, annotated, p_box, p_conf, m_box)

    def _evaluate_rider_head(self, image: np.ndarray, annotated: np.ndarray,
                             rider_box: List[float], rider_conf: float,
                             moto_box: Optional[List[float]]) -> Dict[str, Any]:
        """Evaluates rider head region using custom model or CV fallback."""
        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = [int(v) for v in rider_box]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        rw = rx2 - rx1
        rh = ry2 - ry1

        if rw <= 10 or rh <= 15:
            return {
                "status": "Uncertain / not visible",
                "confidence": 0.3,
                "motorcycle_box": moto_box,
                "rider_box": rider_box,
                "head_box": None,
                "head_crop": None,
                "annotated_image": annotated,
                "details": {"reason": "Rider region too small or occluded"}
            }

        # Head region is approximately the top 25-30% of rider bounding box
        hx1 = max(0, rx1 + int(rw * 0.15))
        hx2 = min(w, rx2 - int(rw * 0.15))
        hy1 = max(0, ry1)
        hy2 = min(h, ry1 + int(rh * 0.32))

        head_crop = image[hy1:hy2, hx1:hx2].copy() if (hx2 > hx1 and hy2 > hy1) else None

        # Mode A: If custom YOLO helmet model exists
        if self.has_custom_model and self.custom_model is not None and head_crop is not None:
            c_res = self.custom_model(head_crop, verbose=False, conf=0.3)[0]
            if len(c_res.boxes) > 0:
                cls_id = int(c_res.boxes.cls[0].item())
                conf = float(c_res.boxes.conf[0].item())
                # Assume class 0=helmet, 1=no_helmet
                if cls_id == 0:
                    status = "Helmet detected"
                else:
                    status = "No helmet detected"

                self._draw_overlay(annotated, [hx1, hy1, hx2, hy2], status, conf)
                return {
                    "status": status,
                    "confidence": round(conf, 2),
                    "motorcycle_box": moto_box,
                    "rider_box": rider_box,
                    "head_box": [hx1, hy1, hx2, hy2],
                    "head_crop": head_crop,
                    "annotated_image": annotated,
                    "details": {"mode": "deep_custom_yolo", "model": self.custom_model_path}
                }

        # Mode B: Computer Vision Morphological & Color Space Analysis Fallback
        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] < 12 or head_crop.shape[1] < 12:
            status = "Uncertain / not visible"
            conf = 0.40
        else:
            status, conf, diag = self._cv_head_analyzer(head_crop)

        self._draw_overlay(annotated, [hx1, hy1, hx2, hy2], status, conf)

        return {
            "status": status,
            "confidence": round(conf, 2),
            "motorcycle_box": moto_box,
            "rider_box": rider_box,
            "head_box": [hx1, hy1, hx2, hy2],
            "head_crop": head_crop,
            "annotated_image": annotated,
            "details": {
                "mode": "cv_morphological_fallback",
                "notice": "Generic YOLO weights do not contain helmet class; evaluated via cranial contour geometry & skin tone distribution.",
                "diagnostics": diag if 'diag' in locals() else {}
            }
        }

    def _cv_head_analyzer(self, head_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyzes head crop using:
        1. HSV skin tone detection (bare face/neck vs solid shell)
        2. Circular dome contour detection (helmets have high circularity and smooth edges)
        3. Texture smoothness / edge frequency
        """
        hsv = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Standard human skin tone mask in HSV
        lower_skin = np.array([0, 28, 50], dtype=np.uint8)
        upper_skin = np.array([25, 200, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / float(head_bgr.shape[0] * head_bgr.shape[1])

        # Dark hair mask
        hair_mask = (v < 45) & (s < 120)
        hair_ratio = np.sum(hair_mask) / float(head_bgr.shape[0] * head_bgr.shape[1])

        # Gray edge variance
        gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / float(head_bgr.shape[0] * head_bgr.shape[1])

        # Circular dome contour inspection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_circularity = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 15:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > max_circularity:
                    max_circularity = circularity

        diag = {
            "skin_ratio": round(float(skin_ratio), 3),
            "hair_ratio": round(float(hair_ratio), 3),
            "edge_density": round(float(edge_density), 3),
            "dome_circularity": round(float(max_circularity), 3)
        }

        # Decision rule
        # If significant exposed skin or dark textured hair with low dome circularity -> No Helmet
        if skin_ratio > 0.32 or (hair_ratio > 0.28 and max_circularity < 0.45):
            conf = min(0.92, 0.60 + (skin_ratio * 0.4) + (hair_ratio * 0.2))
            return "No helmet detected", conf, diag
        elif max_circularity > 0.55 or (skin_ratio < 0.15 and hair_ratio < 0.18):
            conf = min(0.94, 0.65 + (max_circularity * 0.3))
            return "Helmet detected", conf, diag
        else:
            return "Uncertain / not visible", 0.52, diag

    def _draw_overlay(self, image: np.ndarray, head_box: List[int], status: str, conf: float):
        x1, y1, x2, y2 = head_box
        color = (0, 255, 0) # Green for helmet
        if status == "No helmet detected":
            color = (0, 0, 255) # Red for violation
        elif status == "Uncertain / not visible":
            color = (0, 200, 255) # Amber for review

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{status} ({conf:.2f})"
        cv2.putText(image, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
