"""
Vision Guard — Window Tint & Visibility Analysis Module
Experimental image-processing module to assess apparent vehicle window darkness
and estimate visual light transmission (VLT proxy).
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class WindowTintAnalyzer:
    def __init__(self, darkness_threshold: float = 65.0):
        # Experimental threshold: apparent darkness percentage above which manual review is triggered
        self.darkness_threshold = darkness_threshold

    def analyze(self, image: np.ndarray, vehicle_box: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """
        Analyzes window region of vehicle for apparent darkness and estimated visibility.
        Returns:
            - darkness_score: float (0 - 100%)
            - estimated_vlt_proxy: float (0 - 100%)
            - status: 'Compliant / Adequate Visibility' | 'Flagged for Dark Window Review' | 'Unable to Determine'
            - window_crop: np.ndarray
            - annotated_image: np.ndarray
            - details: dict explaining method, parameters, and legal disclaimers
        """
        annotated = image.copy()
        h, w = image.shape[:2]

        if vehicle_box is not None:
            vx1, vy1, vx2, vy2 = vehicle_box
        else:
            vx1, vy1, vx2, vy2 = 0, 0, w, h

        vw = vx2 - vx1
        vh = vy2 - vy1

        # Candidate window region: upper-middle portion of the vehicle profile
        wx1 = max(0, vx1 + int(vw * 0.15))
        wx2 = min(w, vx2 - int(vw * 0.15))
        wy1 = max(0, vy1 + int(vh * 0.12))
        wy2 = min(h, vy1 + int(vh * 0.52))

        if (wx2 - wx1) < 30 or (wy2 - wy1) < 20:
            return {
                "darkness_score": 0.0,
                "estimated_vlt_proxy": 100.0,
                "status": "Unable to Determine",
                "window_crop": None,
                "annotated_image": annotated,
                "details": {"reason": "Vehicle window bounding region too small or occluded."}
            }

        window_crop = image[wy1:wy2, wx1:wx2].copy()

        # Image-based darkness assessment via HSV Value channel and Lab Luminance
        hsv = cv2.cvtColor(window_crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(window_crop, cv2.COLOR_BGR2LAB)

        v_channel = hsv[:, :, 2]
        l_channel = lab[:, :, 0]

        mean_v = float(np.mean(v_channel))
        mean_l = float(np.mean(l_channel))

        # Normalized darkness score: 0% (crystal clear / bright) to 100% (pitch black film)
        # Value 255 = completely bright (0% dark), Value 0 = completely dark (100% dark)
        darkness_score = round(100.0 - ((mean_v / 255.0) * 100.0), 2)
        # Estimated VLT Proxy (Visible Light Transmission approximation)
        vlt_proxy = round(max(0.0, min(100.0, 100.0 - darkness_score)), 2)

        # Standard deviation reflects transparency vs opaque tint
        # Tinted windows have low interior texture variance compared to transparent windows
        variance = float(np.std(v_channel))

        if darkness_score >= self.darkness_threshold:
            status = "Flagged for Dark Window Review"
            color = (0, 0, 255) # Red
        else:
            status = "Compliant / Adequate Visibility"
            color = (0, 255, 0) # Green

        # Overlay HUD on image
        cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), color, 2)
        label = f"Tint Darkness: {darkness_score}% (VLT Proxy: {vlt_proxy}%)"
        cv2.putText(annotated, label, (wx1, max(20, wy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return {
            "darkness_score": darkness_score,
            "estimated_vlt_proxy": vlt_proxy,
            "status": status,
            "threshold_used": self.darkness_threshold,
            "window_crop": window_crop,
            "annotated_image": annotated,
            "details": {
                "method": "HSV Value & CIELAB Luminance Channel Spectrometry Proxy",
                "mean_brightness_v": round(mean_v, 2),
                "interior_texture_variance": round(variance, 2),
                "legal_disclaimer": (
                    "CRITICAL LEGAL NOTICE: An image-based darkness score is an experimental visual "
                    "proxy and does NOT constitute a certified photometric Visible Light Transmission (VLT) "
                    "measurement. Official enforcement requires a physical calibrated tint-meter (e.g. Tint-Chek/Inspector)."
                )
            }
        }
