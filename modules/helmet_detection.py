"""
VisionGuard — Helmet Detection Module
Real-time AI headgear safety compliance analyzer using trained YOLOv8 weights.
Supports:
1. Mode A: Personal Webcam Helmet Testing (individual subject, temporal smoothing, responsive helmet vs no-helmet detection)
2. Mode B: Highway Motorcycle & Rider Surveillance (spatial association of riders and motorcycles, multi-rider support)
"""

import os
import cv2
import collections
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO

from config import MODELS_DIR, HELMET_MODEL_PATH, COLORS


class HelmetDetector:
    """Trained YOLOv8 Helmet Detection Engine for personal webcam and highway surveillance."""

    def __init__(self, model_path: Optional[str] = None,
                 custom_model_path: Optional[str] = None,
                 base_model_path: Optional[str] = None,
                 conf_threshold: float = 0.35,
                 **kwargs):
        self.conf_threshold = conf_threshold
        target_path = model_path or custom_model_path or str(HELMET_MODEL_PATH)
        self.model_path = str(target_path)
        self.custom_model_path = self.model_path
        self.base_model_path = str(base_model_path or (MODELS_DIR / "yolov8n.pt"))

        # Load trained helmet model
        self.model: Optional[YOLO] = None
        self.has_model = False
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.has_model = True
                print(f"[HelmetDetector] Successfully loaded trained helmet model: {self.model_path}")
            except Exception as e:
                print(f"[HelmetDetector] Error loading helmet model: {e}")

        # Load base detector for person localization
        self.base_model: Optional[YOLO] = None
        if os.path.exists(self.base_model_path):
            try:
                self.base_model = YOLO(self.base_model_path)
            except Exception as e:
                print(f"[HelmetDetector] Error loading base model: {e}")

        # Temporal smoothing history for live webcam feed (Mode A)
        self.webcam_history = collections.deque(maxlen=5)
        self.last_status = "Uncertain"
        self.last_confidence = 0.0

    def reset_history(self):
        """Clears temporal smoothing history between camera restarts."""
        self.webcam_history.clear()
        self.last_status = "Uncertain"
        self.last_confidence = 0.0

    def detect_personal_webcam(self, frame: np.ndarray,
                               conf_threshold: Optional[float] = None,
                               use_smoothing: bool = True) -> Dict[str, Any]:
        """
        Analyzes a live webcam frame for personal helmet compliance.
        Designed for a person sitting/standing in front of a webcam.

        Returns:
            Dict containing:
                - status: 'Helmet detected' | 'No helmet detected' | 'Uncertain' | 'No person detected'
                - label: 'HELMET DETECTED' | 'NO HELMET DETECTED' | 'UNCERTAIN' | 'NO PERSON DETECTED'
                - confidence: float
                - person_box: [x1, y1, x2, y2] or None
                - head_box: [x1, y1, x2, y2] or None
                - box_color: (B, G, R)
                - annotated_frame: np.ndarray with bounding boxes and HUD
        """
        if frame is None or frame.size == 0:
            return {
                "status": "No person detected",
                "label": "NO PERSON DETECTED",
                "confidence": 0.0,
                "person_box": None,
                "head_box": None,
                "box_color": (148, 163, 184),
                "annotated_frame": frame
            }

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Step 1: Detect Person or Direct Headgear in Webcam Frame
        person_box = None
        person_conf = 0.0
        if self.base_model:
            base_res = self.base_model(frame, conf=0.25, verbose=False)[0]
            max_area = 0
            for b in base_res.boxes:
                if int(b.cls[0].item()) == 0:  # Class 0: person
                    bx = [int(v) for v in b.xyxy[0].tolist()]
                    area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    if area > max_area and area > (w * h * 0.03):
                        max_area = area
                        person_box = bx
                        person_conf = float(b.conf[0].item())

        # If full person detector didn't trigger (e.g. user sitting close with only head/shoulders visible),
        # check if helmet/head detector directly detects a head/helmet in the webcam feed
        direct_head_detection = None
        if self.has_model and self.model is not None:
            pre_h_res = self.model(frame, conf=0.20, verbose=False)[0]
            if len(pre_h_res.boxes) > 0:
                # Find best box
                best_pre = None
                max_pre_c = 0.0
                for b in pre_h_res.boxes:
                    c = int(b.cls[0].item())
                    bc = float(b.conf[0].item())
                    if bc > max_pre_c:
                        max_pre_c = bc
                        best_pre = (c, bc, [int(v) for v in b.xyxy[0].tolist()])
                if best_pre:
                    direct_head_detection = best_pre
                    if person_box is None:
                        # Synthesize subject box around the head
                        hbx = best_pre[2]
                        pad_w = int((hbx[2] - hbx[0]) * 0.4)
                        person_box = [
                            max(0, hbx[0] - pad_w),
                            max(0, hbx[1] - 20),
                            min(w, hbx[2] + pad_w),
                            min(h, hbx[3] + int((hbx[3] - hbx[1]) * 1.5))
                        ]
                        person_conf = best_pre[1]

        # If neither a person nor head/helmet was localized in front of the camera:
        if person_box is None:
            self.webcam_history.append(("No person detected", 0.0))
            # Draw standby HUD
            cv2.rectangle(annotated, (10, 10), (w - 10, 50), (20, 25, 45), -1)
            cv2.putText(annotated, "STATUS: WAITING FOR SUBJECT | POSITION HEAD IN CAMERA VIEW",
                        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA)
            return {
                "status": "No person detected",
                "label": "NO PERSON DETECTED",
                "confidence": 0.0,
                "person_box": None,
                "head_box": None,
                "box_color": (148, 163, 184),
                "annotated_frame": annotated
            }

        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        head_region_y2 = min(h, py1 + int(ph * 0.45))

        # Draw subtle person bounding box
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (70, 100, 140), 1)
        cv2.putText(annotated, f"Subject (Conf: {person_conf:.2f})", (px1, max(15, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

        # Step 2: Run Trained Helmet Model
        raw_status = "Uncertain"
        raw_confidence = 0.0
        head_box = [px1 + int(pw * 0.15), py1, px2 - int(pw * 0.15), head_region_y2]

        if self.has_model and self.model is not None:
            # Inference on frame
            h_res = self.model(frame, conf=0.20, verbose=False)[0]

            best_match = None
            highest_conf = 0.0

            for b in h_res.boxes:
                cls_id = int(b.cls[0].item())
                b_conf = float(b.conf[0].item())
                bx = [int(v) for v in b.xyxy[0].tolist()]
                bcx = (bx[0] + bx[2]) // 2
                bcy = (bx[1] + bx[3]) // 2

                # Verify detection lies within subject's upper head area
                if (px1 - 25 <= bcx <= px2 + 25) and (py1 - 35 <= bcy <= head_region_y2 + 35):
                    if b_conf > highest_conf:
                        highest_conf = b_conf
                        best_match = (cls_id, b_conf, bx)

            if best_match is not None:
                cls_id, b_conf, bx = best_match
                head_box = bx
                raw_confidence = b_conf

                # Class 0 = 'With Helmet', Class 1 = 'Without Helmet'
                if cls_id == 0:
                    if b_conf >= max(0.28, conf):
                        raw_status = "Helmet detected"
                    else:
                        raw_status = "Uncertain"
                elif cls_id == 1:
                    if b_conf >= max(0.25, conf - 0.05):
                        raw_status = "No helmet detected"
                    else:
                        raw_status = "Uncertain"
            else:
                # No helmet-specific box found on subject -> Evaluate head crop
                head_crop = frame[py1:head_region_y2, px1:px2]
                if head_crop.size > 0 and head_crop.shape[0] > 20 and head_crop.shape[1] > 20:
                    crop_res = self.model(head_crop, conf=0.20, verbose=False)[0]
                    if len(crop_res.boxes) > 0:
                        b = crop_res.boxes[0]
                        c_id = int(b.cls[0].item())
                        c_conf = float(b.conf[0].item())
                        raw_confidence = c_conf
                        if c_id == 0 and c_conf >= conf:
                            raw_status = "Helmet detected"
                        elif c_id == 1 and c_conf >= conf - 0.05:
                            raw_status = "No helmet detected"
                        else:
                            raw_status = "Uncertain"
                    else:
                        raw_status = "Uncertain"
                        raw_confidence = 0.40

        # Step 3: Temporal Smoothing Across Consecutive Frames
        if use_smoothing:
            self.webcam_history.append((raw_status, raw_confidence))
            # Majority voting in window
            status_counts = collections.Counter([s for s, _ in self.webcam_history])
            most_common_status, count = status_counts.most_common(1)[0]

            # Only switch status if it appears consistently in at least 3 of last 5 frames
            if count >= 3 or len(self.webcam_history) < 3:
                effective_status = most_common_status
            else:
                effective_status = self.last_status

            # Calculate average confidence for the effective status
            confs = [c for s, c in self.webcam_history if s == effective_status]
            effective_confidence = float(np.mean(confs)) if confs else raw_confidence
        else:
            effective_status = raw_status
            effective_confidence = raw_confidence

        self.last_status = effective_status
        self.last_confidence = effective_confidence

        # Step 4: Formatting Labels & Colors
        if effective_status == "Helmet detected":
            label_text = f"HELMET DETECTED ({effective_confidence * 100:.1f}%)"
            box_color = (0, 255, 102)  # High-contrast Green (BGR)
            banner_bg = (10, 45, 20)
            banner_text = f"COMPLIANT: HELMET DETECTED | CONFIDENCE: {effective_confidence:.2f}"
            badge_status = "Helmet detected"
        elif effective_status == "No helmet detected":
            label_text = f"NO HELMET DETECTED ({effective_confidence * 100:.1f}%)"
            box_color = (50, 50, 255)  # Bright Red (BGR)
            banner_bg = (20, 15, 55)
            banner_text = f"VIOLATION: NO HELMET DETECTED | CONFIDENCE: {effective_confidence:.2f}"
            badge_status = "No helmet detected"
        else:
            label_text = f"UNCERTAIN ({effective_confidence * 100:.1f}%)"
            box_color = (0, 230, 255)  # Amber / Yellow (BGR)
            banner_bg = (15, 35, 45)
            banner_text = f"UNCERTAIN / VERIFYING | ADJUST POSITION OR LIGHTING"
            badge_status = "Uncertain"

        # Draw Head / Helmet Bounding Box with Label Tab
        hx1, hy1, hx2, hy2 = head_box
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(w, hx2), min(h, hy2)

        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), box_color, 3)

        # Label Tab
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tab_y1 = max(0, hy1 - th - 10)
        cv2.rectangle(annotated, (hx1, tab_y1), (hx1 + tw + 12, hy1), box_color, -1)
        cv2.putText(annotated, label_text, (hx1 + 6, hy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Top HUD Status Banner
        cv2.rectangle(annotated, (10, 10), (w - 10, 48), banner_bg, -1)
        cv2.rectangle(annotated, (10, 10), (w - 10, 48), box_color, 2)
        cv2.putText(annotated, banner_text, (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

        return {
            "status": badge_status,
            "label": label_text,
            "confidence": round(effective_confidence, 2),
            "person_box": person_box,
            "head_box": head_box,
            "box_color": box_color,
            "annotated_frame": annotated
        }

    def analyze_highway_rider(self, frame: np.ndarray,
                               vehicle_box: List[int],
                               all_detections: Optional[List[Dict[str, Any]]] = None,
                               frame_helmet_dets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyzes a motorcycle rider in highway surveillance footage.
        Associates helmet detections with the specific rider and motorcycle.
        """
        if frame is None or frame.size == 0 or not vehicle_box:
            return {"status": "Uncertain", "confidence": 0.0, "head_box": None, "rider_box": None}

        h, w = frame.shape[:2]
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)
        vw = vx2 - vx1
        vh = vy2 - vy1

        # 1. Localize Rider Box
        rider_box = None
        if all_detections:
            for d in all_detections:
                if d.get("class_name") == "person":
                    px1, py1, px2, py2 = d["bbox"]
                    overlap_x = max(0, min(vx2, px2) - max(vx1, px1))
                    # Rider torso/base overlaps motorcycle bounds
                    if overlap_x > 0.20 * (px2 - px1) and py1 < vy2:
                        rider_box = [px1, py1, px2, py2]
                        break

        # If no separate person box detected, approximate rider area atop motorcycle
        if rider_box is None:
            rx1 = max(0, vx1 - int(vw * 0.05))
            rx2 = min(w, vx2 + int(vw * 0.05))
            ry1 = max(0, vy1 - int(vh * 0.35))
            ry2 = min(h, vy1 + int(vh * 0.45))
            rider_box = [rx1, ry1, rx2, ry2]

        rx1, ry1, rx2, ry2 = rider_box
        rw = rx2 - rx1
        rh = ry2 - ry1

        head_box = [rx1 + int(rw * 0.15), ry1, rx2 - int(rw * 0.15), ry1 + int(rh * 0.40)]

        # 2. Match with Helmet Model Detections
        status = "Uncertain"
        conf = 0.45

        # If precomputed global helmet detections are passed:
        if frame_helmet_dets is not None:
            best_match = None
            max_c = 0.0
            for h_det in frame_helmet_dets:
                bx = h_det["box"]
                bcx = (bx[0] + bx[2]) // 2
                bcy = (bx[1] + bx[3]) // 2
                if (rx1 - 25 <= bcx <= rx2 + 25) and (ry1 - 25 <= bcy <= ry1 + int(rh * 0.50)):
                    if h_det["conf"] > max_c:
                        max_c = h_det["conf"]
                        best_match = h_det

            if best_match:
                head_box = best_match["box"]
                conf = best_match["conf"]
                if best_match["class_id"] == 0 and conf >= 0.30:
                    status = "Helmet detected"
                elif best_match["class_id"] == 1 and conf >= 0.28:
                    status = "No helmet detected"
        elif self.has_model and self.model is not None:
            # Direct inference on rider upper-body
            hx1, hy1, hx2, hy2 = head_box
            if hx2 > hx1 and hy2 > hy1:
                crop = frame[hy1:hy2, hx1:hx2]
                if crop.size > 0 and crop.shape[0] > 15 and crop.shape[1] > 15:
                    preds = self.model(crop, conf=0.20, verbose=False)[0]
                    if len(preds.boxes) > 0:
                        b = preds.boxes[0]
                        c_id = int(b.cls[0].item())
                        c_conf = float(b.conf[0].item())
                        conf = c_conf
                        if c_id == 0 and c_conf >= 0.30:
                            status = "Helmet detected"
                        elif c_id == 1 and c_conf >= 0.28:
                            status = "No helmet detected"

        box_color = COLORS["compliant"] if status == "Helmet detected" else (
            COLORS["violation"] if status == "No helmet detected" else COLORS["uncertain"]
        )

        return {
            "status": status,
            "confidence": round(conf, 2),
            "rider_box": rider_box,
            "head_box": head_box,
            "box_color": box_color
        }

    def detect_global_helmets(self, frame: np.ndarray, conf_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """Runs global helmet detection on the entire frame for highway multi-rider scanning."""
        if not self.has_model or self.model is None or frame is None:
            return []

        res = self.model(frame, conf=conf_threshold, verbose=False)[0]
        results = []
        for b in res.boxes:
            c = int(b.cls[0].item())
            b_conf = float(b.conf[0].item())
            bx = [int(v) for v in b.xyxy[0].tolist()]
            results.append({
                "class_id": c,
                "class_name": self.model.names[c],
                "conf": b_conf,
                "box": bx
            })
        return results
