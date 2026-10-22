"""
Vision Guard — Unit Tests for Violation Decision Rules
"""

import os
import pytest
import numpy as np
from database.db_manager import DatabaseManager
from modules.violation_engine import ViolationDecisionEngine

TEST_DB_PATH = "data/test_engine.db"


@pytest.fixture
def engine():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    db = DatabaseManager(TEST_DB_PATH)
    eng = ViolationDecisionEngine(db, evidence_dir="evidence/test_evid")
    yield eng
    # Teardown
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


def test_motorcycle_no_helmet_rule(engine):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    helmet_result = {
        "status": "No helmet detected",
        "confidence": 0.88
    }
    plate_result = {
        "plate_number": "KA 05 EF 1234",
        "confidence": 0.91
    }

    violations = engine.evaluate_vehicle_event(
        frame=dummy_frame,
        tracking_id="TRK-901",
        vehicle_type="motorcycle",
        vehicle_box=[100, 100, 300, 400],
        helmet_result=helmet_result,
        plate_result=plate_result
    )

    assert len(violations) == 1
    assert violations[0]["violation_type"] == "Helmet Violation"
    assert violations[0]["severity"] == "Critical"
    assert violations[0]["fine"] == 1000


def test_seatbelt_violation_rule(engine):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    seatbelt_result = {
        "status": "Possible seat-belt violation",
        "confidence": 0.84
    }
    plate_result = {
        "plate_number": "DL 02 ZZ 7777",
        "confidence": 0.85
    }

    violations = engine.evaluate_vehicle_event(
        frame=dummy_frame,
        tracking_id="TRK-902",
        vehicle_type="car",
        vehicle_box=[50, 50, 400, 350],
        seatbelt_result=seatbelt_result,
        plate_result=plate_result
    )

    assert len(violations) == 1
    assert violations[0]["violation_type"] == "Seat-belt Violation"
    assert violations[0]["severity"] == "Critical"
