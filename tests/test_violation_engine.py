"""
VisionGuard — Unit Tests for Violation Decision Rules & Engine
Tests safety evaluation for helmet, seatbelt, stop-line penetration, and duplicate suppression.
"""

import os
import shutil
import pytest
import numpy as np

from modules.event_storage import EventStorage
from modules.violation_engine import ViolationEngine
from modules.object_tracking import TrackedObject

TEST_DB_PATH = "data/test_engine_suite.db"
TEST_EVIDENCE_DIR = "evidence/test_evidence_suite"


@pytest.fixture
def engine():
    # Setup
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    if os.path.exists(TEST_EVIDENCE_DIR):
        try:
            shutil.rmtree(TEST_EVIDENCE_DIR)
        except Exception:
            pass

    storage = EventStorage(TEST_DB_PATH)
    v_engine = ViolationEngine(storage, evidence_dir=TEST_EVIDENCE_DIR)
    yield v_engine

    # Teardown
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    if os.path.exists(TEST_EVIDENCE_DIR):
        try:
            shutil.rmtree(TEST_EVIDENCE_DIR)
        except Exception:
            pass


def test_motorcycle_no_helmet_rule(engine):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obj = TrackedObject("TRK-501", "motorcycle", [100, 100, 250, 350], 0.90)

    helmet_result = {
        "status": "No helmet detected",
        "confidence": 0.89
    }

    viols = engine.evaluate(frame, obj, helmet_result=helmet_result)
    assert len(viols) == 1
    assert viols[0]["category"] == "Helmet Violation"
    assert viols[0]["confidence"] == 0.89
    assert obj.has_violation("Helmet Violation") is True

    # Duplicate suppression check on subsequent frame
    viols_subsequent = engine.evaluate(frame, obj, helmet_result=helmet_result)
    assert len(viols_subsequent) == 0


def test_seatbelt_violation_rule(engine):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obj = TrackedObject("TRK-502", "car", [150, 150, 400, 450], 0.92)

    seatbelt_result = {
        "status": "Possible seat-belt violation",
        "confidence": 0.85
    }

    viols = engine.evaluate(frame, obj, seatbelt_result=seatbelt_result)
    assert len(viols) == 1
    assert viols[0]["category"] == "Seat-Belt Violation"
    assert obj.has_violation("Seat-Belt Violation") is True

    # Duplicate suppression check
    viols_subsequent = engine.evaluate(frame, obj, seatbelt_result=seatbelt_result)
    assert len(viols_subsequent) == 0


def test_red_light_stop_line_violation(engine):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obj = TrackedObject("TRK-503", "truck", [100, 200, 300, 400], 0.94)

    traffic_signal_result = (True, {"state": "RED", "crossed": True})
    viols = engine.evaluate(frame, obj, traffic_signal_result=traffic_signal_result)
    assert len(viols) == 1
    assert viols[0]["category"] == "Red Light Stop-Line Violation"
    assert obj.has_violation("Red Light Stop-Line Violation") is True


def test_compliant_vehicle_no_violations(engine):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obj = TrackedObject("TRK-504", "motorcycle", [100, 100, 250, 350], 0.95)

    helmet_result = {
        "status": "Helmet detected",
        "confidence": 0.94
    }

    viols = engine.evaluate(frame, obj, helmet_result=helmet_result)
    assert len(viols) == 0
    assert obj.has_violation("Helmet Violation") is False
