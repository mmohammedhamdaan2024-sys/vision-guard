"""
Vision Guard — Unit Tests for Database Operations
"""

import os
import pytest
from database.db_manager import DatabaseManager

TEST_DB_PATH = "data/test_visionguard.db"


@pytest.fixture
def db():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
    db_inst = DatabaseManager(TEST_DB_PATH)
    yield db_inst
    # Teardown
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


def test_init_and_seed(db):
    db.seed_demo_data(force=True)
    metrics = db.get_dashboard_metrics()
    assert metrics["total_vehicles"] > 0
    assert metrics["total_violations"] > 0
    assert metrics["pending_reviews"] > 0


def test_vehicle_operations(db):
    vid = db.add_vehicle("TRK-999", "car", "DL 01 AB 9999", 0.95, "evidence/test.jpg")
    assert vid > 0
    v = db.get_vehicle_by_tracking_id("TRK-999")
    assert v is not None
    assert v["plate_number"] == "DL 01 AB 9999"


def test_violation_and_challan_flow(db):
    vid = db.add_vehicle("TRK-888", "motorcycle", "MH 12 CD 8888", 0.90)
    viol_id = db.add_violation(
        vehicle_id=vid,
        tracking_id="TRK-888",
        violation_type="Helmet Violation",
        severity="Critical",
        confidence=0.92,
        evidence_path="evidence/test.jpg",
        status="Pending Review"
    )
    assert viol_id > 0

    challan_ref = db.create_challan(
        violation_id=viol_id,
        vehicle_plate="MH 12 CD 8888",
        violation_type="Helmet Violation",
        fine_amount=1000.0,
        evidence_path="evidence/test.jpg"
    )
    assert challan_ref.startswith("VG-2026-CH-")

    # Update review status
    challan = db.get_challan_by_ref(challan_ref)
    assert challan is not None
    db.update_challan_review_status(challan["id"], "Approved", "Inspector Test", "Approved violation")

    updated_challan = db.get_challan_by_ref(challan_ref)
    assert updated_challan["review_status"] == "Approved"
