"""
Vision Guard — Violation Decision Engine
Aggregates detections across helmet, seat-belt, window tint, mirror, and plate OCR modules.
Enforces multi-tier confidence rules, handles uncertainty, saves evidence images,
and persists violation and challan records.
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from database.db_manager import DatabaseManager
from utils.image_utils import safe_crop


class ViolationDecisionEngine:
    def __init__(self, db_manager: DatabaseManager,
                 evidence_dir: str = "evidence",
                 config_thresholds: Optional[Dict[str, Any]] = None,
                 config_fines: Optional[Dict[str, Any]] = None):
        self.db = db_manager
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Thresholds
        self.thresholds = config_thresholds or {
            "helmet_confidence": 0.50,
            "seatbelt_confidence": 0.50,
            "dark_tint_threshold": 65.0,
            "mirror_visibility_threshold": 60.0,
            "plate_confidence": 0.40
        }

        # Demonstration Fine Schedule
        self.fines = config_fines or {
            "helmet_violation": 1000,
            "seatbelt_violation": 1000,
            "dark_tint_violation": 750,
            "mirror_obscured": 500,
            "unregistered_plate": 2000
        }

    def evaluate_vehicle_event(self,
                               frame: np.ndarray,
                               tracking_id: str,
                               vehicle_type: str,
                               vehicle_box: List[int],
                               helmet_result: Optional[Dict[str, Any]] = None,
                               seatbelt_result: Optional[Dict[str, Any]] = None,
                               tint_result: Optional[Dict[str, Any]] = None,
                               mirror_result: Optional[Dict[str, Any]] = None,
                               plate_result: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluates a detected/tracked vehicle against all road safety rules.
        Returns a list of violation/review records generated for this event.
        """
        generated_violations = []

        # Plate metadata
        plate_number = "UNREADABLE"
        plate_conf = 0.0
        if plate_result:
            plate_number = plate_result.get("plate_number", "UNREADABLE")
            plate_conf = plate_result.get("confidence", 0.0)

        # Upsert vehicle record in DB
        existing_vehicle = self.db.get_vehicle_by_tracking_id(tracking_id)
        if not existing_vehicle:
            vehicle_crop = safe_crop(frame, vehicle_box, margin=0.1)
            snapshot_filename = f"VEH_{tracking_id}_{int(time.time())}.jpg"
            snapshot_path = os.path.join(self.evidence_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, vehicle_crop)

            vehicle_id = self.db.add_vehicle(
                tracking_id=tracking_id,
                vehicle_type=vehicle_type,
                plate_number=plate_number,
                plate_confidence=plate_conf,
                snapshot_path=snapshot_path
            )
        else:
            vehicle_id = existing_vehicle["id"]
            if plate_number != "UNREADABLE" and existing_vehicle["plate_number"] == "UNREADABLE":
                self.db.update_vehicle_plate(tracking_id, plate_number, plate_conf)

        # Rule 1: Motorcycle Rider Helmet Check
        if vehicle_type == "motorcycle" and helmet_result:
            h_status = helmet_result.get("status")
            h_conf = helmet_result.get("confidence", 0.0)

            if h_status == "No helmet detected":
                severity = "Critical" if h_conf >= self.thresholds["helmet_confidence"] else "Review"
                review_status = "Pending Review"
                evid_path = self._save_evidence_crop(frame, vehicle_box, tracking_id, "HELMET_VIOLATION")
                notes = f"Rider identified without protective headgear (Detection confidence: {h_conf:.2f})."
                fine = self.fines.get("helmet_violation", 1000)

                v_id = self.db.add_violation(
                    vehicle_id=vehicle_id,
                    tracking_id=tracking_id,
                    violation_type="Helmet Violation",
                    severity=severity,
                    confidence=h_conf,
                    evidence_path=evid_path,
                    status=review_status,
                    review_notes=notes
                )
                challan_ref = self.db.create_challan(
                    violation_id=v_id,
                    vehicle_plate=plate_number,
                    violation_type="Helmet Violation",
                    fine_amount=fine,
                    evidence_path=evid_path,
                    notes=notes
                )
                generated_violations.append({
                    "violation_id": v_id,
                    "challan_ref": challan_ref,
                    "violation_type": "Helmet Violation",
                    "severity": severity,
                    "status": review_status,
                    "fine": fine,
                    "notes": notes,
                    "evidence_path": evid_path
                })

            elif h_status == "Uncertain / not visible":
                evid_path = self._save_evidence_crop(frame, vehicle_box, tracking_id, "HELMET_UNCERTAIN")
                notes = f"Head region obscured or confidence below threshold ({h_conf:.2f}). Manual review required."
                v_id = self.db.add_violation(
                    vehicle_id=vehicle_id,
                    tracking_id=tracking_id,
                    violation_type="Helmet Uncertain Review",
                    severity="Review",
                    confidence=h_conf,
                    evidence_path=evid_path,
                    status="Pending Review",
                    review_notes=notes
                )
                generated_violations.append({
                    "violation_id": v_id,
                    "violation_type": "Helmet Uncertain Review",
                    "severity": "Review",
                    "status": "Pending Review",
                    "notes": notes,
                    "evidence_path": evid_path
                })

        # Rule 2: Occupant Seat-belt Check (Cars, Buses, Trucks)
        if vehicle_type in ("car", "bus", "truck") and seatbelt_result:
            s_status = seatbelt_result.get("status")
            s_conf = seatbelt_result.get("confidence", 0.0)

            if s_status == "Possible seat-belt violation":
                severity = "Critical" if s_conf >= self.thresholds["seatbelt_confidence"] else "Review"
                review_status = "Pending Review"
                evid_path = self._save_evidence_crop(frame, vehicle_box, tracking_id, "SEATBELT_VIOLATION")
                notes = f"Cabin occupant observed without visible fastened diagonal safety belt (Confidence: {s_conf:.2f})."
                fine = self.fines.get("seatbelt_violation", 1000)

                v_id = self.db.add_violation(
                    vehicle_id=vehicle_id,
                    tracking_id=tracking_id,
                    violation_type="Seat-belt Violation",
                    severity=severity,
                    confidence=s_conf,
                    evidence_path=evid_path,
                    status=review_status,
                    review_notes=notes
                )
                challan_ref = self.db.create_challan(
                    violation_id=v_id,
                    vehicle_plate=plate_number,
                    violation_type="Seat-belt Violation",
                    fine_amount=fine,
                    evidence_path=evid_path,
                    notes=notes
                )
                generated_violations.append({
                    "violation_id": v_id,
                    "challan_ref": challan_ref,
                    "violation_type": "Seat-belt Violation",
                    "severity": severity,
                    "status": review_status,
                    "fine": fine,
                    "notes": notes,
                    "evidence_path": evid_path
                })

            elif s_status == "Unable to determine":
                # Do not trigger confirmed fine for reflection/adverse lighting
                pass

        # Rule 3: Window Tint & Visibility Assessment
        if tint_result and tint_result.get("status") == "Flagged for Dark Window Review":
            darkness = tint_result.get("darkness_score", 0.0)
            evid_path = self._save_evidence_crop(frame, vehicle_box, tracking_id, "TINT_REVIEW")
            notes = (
                f"Experimental apparent window darkness of {darkness:.1f}% exceeds demonstration threshold "
                f"({self.thresholds['dark_tint_threshold']}%). Flagged for physical officer inspection."
            )
            fine = self.fines.get("dark_tint_violation", 750)

            v_id = self.db.add_violation(
                vehicle_id=vehicle_id,
                tracking_id=tracking_id,
                violation_type="Dark Window Tint Review",
                severity="Warning",
                confidence=round(darkness / 100.0, 2),
                evidence_path=evid_path,
                status="Pending Review",
                review_notes=notes
            )
            challan_ref = self.db.create_challan(
                violation_id=v_id,
                vehicle_plate=plate_number,
                violation_type="Dark Window Tint Review",
                fine_amount=fine,
                evidence_path=evid_path,
                notes=notes
            )
            generated_violations.append({
                "violation_id": v_id,
                "challan_ref": challan_ref,
                "violation_type": "Dark Window Tint Review",
                "severity": "Warning",
                "status": "Pending Review",
                "fine": fine,
                "notes": notes,
                "evidence_path": evid_path
            })

        # Rule 4: Mirror Visibility Assessment
        if mirror_result and mirror_result.get("status") == "Mirror potentially obscured or not clearly visible":
            vis_score = mirror_result.get("visibility_score", 0.0)
            evid_path = self._save_evidence_crop(frame, vehicle_box, tracking_id, "MIRROR_OBSCURED")
            notes = f"Exterior wing mirror region obscured or missing protrusion clarity (Clarity score: {vis_score:.1f}%)."
            fine = self.fines.get("mirror_obscured", 500)

            v_id = self.db.add_violation(
                vehicle_id=vehicle_id,
                tracking_id=tracking_id,
                violation_type="Mirror Obscured",
                severity="Review",
                confidence=round(1.0 - (vis_score / 100.0), 2),
                evidence_path=evid_path,
                status="Under Investigation",
                review_notes=notes
            )
            challan_ref = self.db.create_challan(
                violation_id=v_id,
                vehicle_plate=plate_number,
                violation_type="Mirror Obscured",
                fine_amount=fine,
                evidence_path=evid_path,
                notes=notes
            )
            generated_violations.append({
                "violation_id": v_id,
                "challan_ref": challan_ref,
                "violation_type": "Mirror Obscured",
                "severity": "Review",
                "status": "Under Investigation",
                "fine": fine,
                "notes": notes,
                "evidence_path": evid_path
            })

        return generated_violations

    def _save_evidence_crop(self, frame: np.ndarray, bbox: List[int],
                            tracking_id: str, tag: str) -> str:
        """Saves a high-resolution evidence crop to disk."""
        crop = safe_crop(frame, bbox, margin=0.15)
        # Add timestamp & watermark
        cv2.putText(crop, f"VISION GUARD EVIDENCE | {tracking_id} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        filename = f"EVID_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tracking_id}_{tag}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(filepath, crop)
        return filepath
