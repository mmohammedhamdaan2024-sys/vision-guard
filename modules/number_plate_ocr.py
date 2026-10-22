"""
Vision Guard — Number Plate Detection & OCR Recognition Module
Extracts vehicle registration plates, applies OpenCV morphological filtering,
runs EasyOCR, normalizes text, and marks unreadable plates for manual officer review.
"""

import os
import cv2
import numpy as np
import re
from typing import Dict, Any, Tuple, Optional, List
from utils.validation import normalize_license_plate


class NumberPlateRecognizer:
    _reader_instance = None

    def __init__(self, use_gpu: bool = False, min_conf: float = 0.40):
        self.min_conf = min_conf
        self.use_gpu = use_gpu

    @classmethod
    def get_ocr_reader(cls, use_gpu: bool = False):
        """Lazy singleton loader for EasyOCR reader to optimize performance."""
        if cls._reader_instance is None:
            try:
                import easyocr
                cls._reader_instance = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            except Exception as e:
                print(f"Warning: EasyOCR reader initialization failed: {e}")
                cls._reader_instance = None
        return cls._reader_instance

    def recognize(self, image: np.ndarray, vehicle_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Locates and recognizes registration number plate on a vehicle.
        Returns:
            - plate_number: str (normalized)
            - raw_ocr_text: str (unmodified original OCR detection)
            - confidence: float (0.0 to 1.0)
            - plate_box: [x1, y1, x2, y2]
            - plate_crop: np.ndarray
            - preprocessed_crop: np.ndarray
            - status: 'Plate Recognized' | 'Low Confidence / Review' | 'Unreadable Plate'
            - annotated_image: np.ndarray
        """
        annotated = image.copy()
        h, w = image.shape[:2]

        if vehicle_box is not None:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_box]
        else:
            vx1, vy1, vx2, vy2 = 0, 0, w, h

        vw = vx2 - vx1
        vh = vy2 - vy1

        # Locate candidate plate region
        plate_box, plate_crop = self._locate_plate_region(image, (vx1, vy1, vx2, vy2))

        if plate_crop is None or plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 20:
            return {
                "plate_number": "UNREADABLE",
                "raw_ocr_text": "",
                "confidence": 0.0,
                "plate_box": None,
                "plate_crop": None,
                "preprocessed_crop": None,
                "status": "Unreadable Plate",
                "annotated_image": annotated,
                "details": {"reason": "Could not identify candidate plate region"}
            }

        # Morphological preprocessing pipeline
        prep_crop = self._preprocess_plate(plate_crop)

        # Run OCR
        reader = self.get_ocr_reader(self.use_gpu)
        raw_text = ""
        conf = 0.0

        if reader is not None:
            try:
                # Run on preprocessed and original, pick best
                results_prep = reader.readtext(prep_crop, detail=1)
                results_orig = reader.readtext(plate_crop, detail=1)
                all_results = results_prep + results_orig

                if all_results:
                    # Pick result with best confidence and alphanumeric characters
                    valid_candidates = []
                    for bbox, text, score in all_results:
                        clean_candidate = re.sub(r'[^A-Za-z0-9]', '', text)
                        if len(clean_candidate) >= 3:
                            valid_candidates.append((score, text, clean_candidate))

                    if valid_candidates:
                        valid_candidates.sort(key=lambda x: x[0], reverse=True)
                        conf, raw_text, _ = valid_candidates[0]
            except Exception as e:
                print(f"OCR inference warning: {e}")

        clean_number = normalize_license_plate(raw_text)

        # Determine review status
        if clean_number != "UNREADABLE" and conf >= self.min_conf:
            status = "Plate Recognized"
            color = (0, 255, 0) # Green
        elif clean_number != "UNREADABLE" and conf < self.min_conf:
            status = "Low Confidence / Review"
            color = (0, 200, 255) # Amber
        else:
            status = "Unreadable Plate"
            color = (0, 0, 255) # Red

        # Draw HUD Box
        px1, py1, px2, py2 = plate_box
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
        cv2.putText(annotated, f"PLATE: {clean_number} ({conf:.2f})", (px1, max(20, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return {
            "plate_number": clean_number,
            "raw_ocr_text": raw_text,
            "confidence": round(float(conf), 2),
            "plate_box": plate_box,
            "plate_crop": plate_crop,
            "preprocessed_crop": prep_crop,
            "status": status,
            "annotated_image": annotated,
            "details": {
                "ocr_engine": "EasyOCR + Bilateral Adaptive Morphological Filter",
                "manual_verification_required": status != "Plate Recognized"
            }
        }

    def _locate_plate_region(self, image: np.ndarray,
                             vbox: Tuple[int, int, int, int]) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Locates candidate plate region using rectangular contour aspect ratio filtering,
        with fallback to standard lower bumper geometry.
        """
        h, w = image.shape[:2]
        vx1, vy1, vx2, vy2 = vbox
        vw = vx2 - vx1
        vh = vy2 - vy1

        # Search within the lower 45% of the vehicle
        search_y1 = vy1 + int(vh * 0.55)
        search_y2 = vy2
        search_x1 = vx1 + int(vw * 0.15)
        search_x2 = vx2 - int(vw * 0.15)

        search_y1 = max(0, search_y1)
        search_y2 = min(h, search_y2)
        search_x1 = max(0, search_x1)
        search_x2 = min(w, search_x2)

        if search_y2 <= search_y1 or search_x2 <= search_x1:
            # Default to lower vehicle zone
            px1 = max(0, vx1 + int(vw * 0.25))
            px2 = min(w, vx2 - int(vw * 0.25))
            py1 = max(0, vy1 + int(vh * 0.65))
            py2 = min(h, vy2 - int(vh * 0.05))
            return [px1, py1, px2, py2], image[py1:py2, px1:px2].copy()

        search_zone = image[search_y1:search_y2, search_x1:search_x2]
        gray = cv2.cvtColor(search_zone, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blur, 40, 140)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best_rect = None
        max_area = 0

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, rw, rh = cv2.boundingRect(cnt)
                aspect_ratio = rw / float(rh) if rh > 0 else 0
                area = rw * rh
                # Standard license plates have aspect ratio between 2.2 and 5.5
                if 2.0 <= aspect_ratio <= 6.0 and area > 600 and area > max_area:
                    max_area = area
                    best_rect = (x, y, rw, rh)

        if best_rect is not None:
            rx, ry, rw, rh = best_rect
            px1 = search_x1 + rx
            py1 = search_y1 + ry
            px2 = px1 + rw
            py2 = py1 + rh
        else:
            # Lower-central bumper fallback
            px1 = max(0, vx1 + int(vw * 0.28))
            px2 = min(w, vx2 - int(vw * 0.28))
            py1 = max(0, vy1 + int(vh * 0.68))
            py2 = min(h, vy2 - int(vh * 0.08))

        crop = image[py1:py2, px1:px2].copy() if (px2 > px1 and py2 > py1) else None
        return [px1, py1, px2, py2], crop

    def _preprocess_plate(self, plate_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances plate contrast and clarity:
        1. Grayscale conversion
        2. Bilateral filtering to preserve edges while clearing road dust
        3. Adaptive thresholding
        4. Morphological rectangular kernel closing
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return np.zeros((32, 64), dtype=np.uint8)

        # Upscale small plates for sharper character OCR
        ph, pw = plate_bgr.shape[:2]
        if ph < 50:
            scale = 60.0 / ph
            plate_bgr = cv2.resize(plate_bgr, (int(pw * scale), 60), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 60, 60)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return closed
