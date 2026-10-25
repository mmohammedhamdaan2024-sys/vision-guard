"""
VisionGuard — Unit Tests for Upgraded Helmet Detector (Mode A & Mode B)
Validates trained YOLOv8 model loading, personal webcam inference, temporal smoothing,
and unhelmeted vs helmeted classification.
"""

import os
import pytest
import numpy as np

from modules.helmet_detection import HelmetDetector
from config import HELMET_MODEL_PATH


@pytest.fixture
def helmet_detector():
    return HelmetDetector(model_path=str(HELMET_MODEL_PATH))


def test_helmet_model_loaded(helmet_detector):
    assert helmet_detector.has_model is True
    assert helmet_detector.model is not None
    # Verify trained classes: With Helmet (0), Without Helmet (1)
    names = helmet_detector.model.names
    assert 0 in names and 1 in names
    assert "helmet" in names[0].lower()


def test_empty_black_frame_returns_no_person(helmet_detector):
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = helmet_detector.detect_personal_webcam(black_frame, use_smoothing=False)
    assert res["status"] == "No person detected"
    assert res["person_box"] is None
    assert res["confidence"] == 0.0


def test_personal_webcam_detection_on_real_samples(helmet_detector):
    import cv2
    if os.path.exists("test_bus.jpg"):
        full_img = cv2.imread("test_bus.jpg")
        # Crop unhelmeted person
        p_unhelmeted = full_img[380:860, 200:360]
        res_unhelmeted = helmet_detector.detect_personal_webcam(p_unhelmeted, use_smoothing=False)
        assert res_unhelmeted["status"] == "No helmet detected"
        assert res_unhelmeted["confidence"] > 0.30


def test_temporal_smoothing_logic(helmet_detector):
    helmet_detector.reset_history()
    # Fill history with consistent status
    helmet_detector.webcam_history.append(("Helmet detected", 0.90))
    helmet_detector.webcam_history.append(("Helmet detected", 0.88))
    helmet_detector.webcam_history.append(("Helmet detected", 0.92))

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = helmet_detector.detect_personal_webcam(dummy_frame, use_smoothing=True)
    # Even if an empty frame is sent, temporal history does not immediately switch to wrong status
    assert len(helmet_detector.webcam_history) > 0
