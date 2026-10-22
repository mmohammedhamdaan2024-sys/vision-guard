"""
Vision Guard — Mirror Visibility Analysis Module
Experimental module for vehicle exterior rear-view mirror presence, clarity, and obstruction assessment.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class MirrorAnalyzer:
    def __init__(self, visibility_threshold: float = 60.0):
        # Experimental threshold: estimated clarity/visibility score percentage
        self.visibility_threshold = visibility_threshold

    def analyze(self, image: np.ndarray, vehicle_box: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """
        Analyzes side flank zones of a vehicle to detect exterior wing mirror regions and estimate clarity.
        Returns:
            - status: 'Mirror region detected' | 'Mirror potentially obscured or not clearly visible' | 'Unable to determine'
            - visibility_score: float (0 - 100%)
            - mirror_crops: list of crops
            - annotated_image: np.ndarray
            - details: dict
        """
        annotated = image.copy()
        h, w = image.shape[:2]

        if vehicle_box is not None:
            vx1, vy1, vx2, vy2 = vehicle_box
        else:
            vx1, vy1, vx2, vy2 = 0, 0, w, h

        vw = vx2 - vx1
        vh = vy2 - vy1

        if vw < 60 or vh < 40:
            return {
                "status": "Unable to determine",
                "visibility_score": 0.0,
                "mirror_crops": [],
                "annotated_image": annotated,
                "details": {"reason": "Vehicle scale is too small for mirror resolution."}
            }

        # Exterior wing mirrors typically reside at the lateral edges (left 15% and right 15%)
        # at the lower boundary of the windshield / upper door level (35% to 55% vehicle height)
        mw = int(vw * 0.16)
        mh = int(vh * 0.20)
        my1 = vy1 + int(vh * 0.35)
        my2 = my1 + mh

        # Left mirror candidate zone
        lx1 = max(0, vx1 - int(mw * 0.2))
        lx2 = min(w, vx1 + int(mw * 1.0))

        # Right mirror candidate zone
        rx1 = max(0, vx2 - int(mw * 1.0))
        rx2 = min(w, vx2 + int(mw * 0.2))

        crops = []
        scores = []

        for side, (x1, y1, x2, y2) in [("Left Flank", (lx1, my1, lx2, my2)), ("Right Flank", (rx1, my1, rx2, my2))]:
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = image[y1:y2, x1:x2].copy()
                crops.append(crop)

                # Assess structural protrusion and clarity via Laplacian sharpness and gradient contrast
                gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray_c, cv2.CV_64F).var()
                sobelx = cv2.Sobel(gray_c, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_c, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))

                # Normalize to an experimental 0-100% score
                clarity_score = min(100.0, (lap_var / 80.0) * 45.0 + (grad_mag / 40.0) * 55.0)
                scores.append(round(clarity_score, 1))

                # Draw candidate mirror zone on HUD
                color = (0, 255, 0) if clarity_score >= self.visibility_threshold else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                cv2.putText(annotated, f"Mirror: {clarity_score:.0f}%", (x1, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if not scores:
            return {
                "status": "Unable to determine",
                "visibility_score": 0.0,
                "mirror_crops": [],
                "annotated_image": annotated,
                "details": {"reason": "Could not extract lateral flank candidate coordinates."}
            }

        avg_score = round(float(np.mean(scores)), 1)
        if avg_score >= self.visibility_threshold:
            status = "Mirror region detected"
        elif avg_score >= 30.0:
            status = "Mirror potentially obscured or not clearly visible"
        else:
            status = "Unable to determine"

        return {
            "status": status,
            "visibility_score": avg_score,
            "threshold_used": self.visibility_threshold,
            "mirror_crops": crops,
            "annotated_image": annotated,
            "details": {
                "individual_flank_scores": scores,
                "method": "Lateral flank gradient & Laplacian edge structural protrusion analysis",
                "disclaimer": "Experimental project parameter. Not an official homologation test."
            }
        }
