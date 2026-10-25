"""
VisionGuard — Violation Engine Module
Rule-based decision logic integrating detection results across helmet, seatbelt,
traffic signal stop-line crossing, and vehicle motion modules.
Enforces duplicate-event suppression and saves watermarked forensic evidence images.
"""

import os
import cv2
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from config import EVIDENCE_DIR, COLORS
from modules.event_storage import EventStorage


class ViolationEngine:
    """Combines detection pipeline outputs, applies safety rules, and generates violation records."""

    def __init__(self, event_storage: EventStorage, evidence_dir: Optional[str] = None):
        self.storage = event_storage
        self.evidence_dir = str(evidence_dir or EVIDENCE_DIR)
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Feature flags for additional road safety modules
        self.wrong_way_enabled = True
        self.expected_direction = "downward"  # Standard traffic flow direction in camera FOV

    def evaluate(self, frame: np.ndarray, tracked_object: Any,
                 helmet_result: Optional[Dict[str, Any]] = None,
                 seatbelt_result: Optional[Dict[str, Any]] = None,
                 traffic_signal_result: Optional[Tuple[bool, Dict[str, Any]]] = None,
                 camera_source: str = "Live Camera") -> List[Dict[str, Any]]:
        """
        Evaluates a tracked road user for road safety infractions.
        Returns a list of generated violation items.
        """
        generated_violations = []
        track_id = tracked_object.track_id
        vtype = tracked_object.class_name
        vbox = tracked_object.bbox

        # -------------------------------------------------------------
        # 1. Motorcycle Rider Helmet Check
        # -------------------------------------------------------------
        if vtype == "motorcycle" and helmet_result:
            h_status = helmet_result.get("status")
            h_conf = helmet_result.get("confidence", 0.0)

            if h_status == "No helmet detected" and not tracked_object.has_violation("Helmet Violation"):
                evid_path = self._save_evidence_crop(frame, vbox, track_id, "HELMET_VIOLATION")
                notes = f"Two-wheeler rider identified without protective helmet (Confidence: {h_conf:.2f})."
                event_id = self.storage.log_event(
                    violation_category="Helmet Violation",
                    tracking_id=track_id,
                    vehicle_type=vtype,
                    confidence_score=h_conf,
                    evidence_path=evid_path,
                    camera_source=camera_source,
                    review_status="Pending Review",
                    reviewer_notes=notes
                )
                tracked_object.mark_violation("Helmet Violation")
                generated_violations.append({
                    "event_id": event_id,
                    "category": "Helmet Violation",
                    "confidence": h_conf,
                    "evidence_path": evid_path,
                    "notes": notes
                })

        # -------------------------------------------------------------
        # 2. Vehicle Occupant Seat-Belt Check
        # -------------------------------------------------------------
        if vtype in ("car", "truck", "bus") and seatbelt_result:
            s_status = seatbelt_result.get("status")
            s_conf = seatbelt_result.get("confidence", 0.0)

            if s_status == "Possible seat-belt violation" and not tracked_object.has_violation("Seat-Belt Violation"):
                evid_path = self._save_evidence_crop(frame, vbox, track_id, "SEATBELT_VIOLATION")
                notes = f"Visible front-seat occupant unfastened across cabin (Confidence: {s_conf:.2f})."
                event_id = self.storage.log_event(
                    violation_category="Seat-Belt Violation",
                    tracking_id=track_id,
                    vehicle_type=vtype,
                    confidence_score=s_conf,
                    evidence_path=evid_path,
                    camera_source=camera_source,
                    review_status="Pending Review",
                    reviewer_notes=notes
                )
                tracked_object.mark_violation("Seat-Belt Violation")
                generated_violations.append({
                    "event_id": event_id,
                    "category": "Seat-Belt Violation",
                    "confidence": s_conf,
                    "evidence_path": evid_path,
                    "notes": notes
                })

        # -------------------------------------------------------------
        # 3. Red Light Stop-Line Violation
        # -------------------------------------------------------------
        if traffic_signal_result:
            is_red_viol, sig_details = traffic_signal_result
            if is_red_viol and not tracked_object.has_violation("Red Light Stop-Line Violation"):
                evid_path = self._save_evidence_crop(frame, vbox, track_id, "RED_LIGHT_VIOLATION")
                notes = f"Vehicle traversed past virtual stop-line during active RED traffic light."
                event_id = self.storage.log_event(
                    violation_category="Red Light Stop-Line Violation",
                    tracking_id=track_id,
                    vehicle_type=vtype,
                    confidence_score=0.92,
                    evidence_path=evid_path,
                    camera_source=camera_source,
                    review_status="Pending Review",
                    reviewer_notes=notes
                )
                tracked_object.mark_violation("Red Light Stop-Line Violation")
                generated_violations.append({
                    "event_id": event_id,
                    "category": "Red Light Stop-Line Violation",
                    "confidence": 0.92,
                    "evidence_path": evid_path,
                    "notes": notes
                })

        # -------------------------------------------------------------
        # 4. Optional Module: Wrong-Way Movement
        # -------------------------------------------------------------
        if self.wrong_way_enabled and tracked_object.frames_active > 5:
            if tracked_object.direction != "stationary" and tracked_object.direction != self.expected_direction:
                if not tracked_object.has_violation("Wrong-Way Movement"):
                    evid_path = self._save_evidence_crop(frame, vbox, track_id, "WRONG_WAY")
                    notes = f"Vehicle moving {tracked_object.direction.upper()} contrary to expected {self.expected_direction.upper()} traffic flow."
                    event_id = self.storage.log_event(
                        violation_category="Wrong-Way Movement",
                        tracking_id=track_id,
                        vehicle_type=vtype,
                        confidence_score=0.88,
                        evidence_path=evid_path,
                        camera_source=camera_source,
                        review_status="Pending Review",
                        reviewer_notes=notes
                    )
                    tracked_object.mark_violation("Wrong-Way Movement")
                    generated_violations.append({
                        "event_id": event_id,
                        "category": "Wrong-Way Movement",
                        "confidence": 0.88,
                        "evidence_path": evid_path,
                        "notes": notes
                    })

        return generated_violations

    def _save_evidence_crop(self, frame: np.ndarray, bbox: List[int],
                            track_id: str, tag: str) -> str:
        """Crops the vehicle/violator with padding, stamps watermarks, and writes to disk."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Margin
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            crop = np.zeros((100, 100, 3), dtype=np.uint8)

        # Watermark
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(crop, f"VISIONGUARD EVIDENCE | {track_id} | {t_str}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        fname = f"EVID_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track_id}_{tag}.jpg"
        fpath = os.path.join(self.evidence_dir, fname)
        cv2.imwrite(fpath, crop)
        return fpath
