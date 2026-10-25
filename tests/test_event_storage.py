"""
VisionGuard — Unit Tests for EventStorage (SQLite)
Tests database initialization, event logging, query filtering, status updates, and stats.
"""

import os
import pytest
import pandas as pd

from modules.event_storage import EventStorage

TEST_DB_PATH = "data/test_storage_suite.db"


@pytest.fixture
def storage():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass

    stor = EventStorage(TEST_DB_PATH)
    yield stor

    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


def test_db_initialization(storage):
    assert os.path.exists(TEST_DB_PATH)
    events = storage.list_events()
    assert len(events) == 0


def test_log_and_retrieve_event(storage):
    ev_id = storage.log_event(
        violation_category="Helmet Violation",
        tracking_id="TRK-1001",
        vehicle_type="motorcycle",
        confidence_score=0.91,
        evidence_path="evidence/test.jpg",
        camera_source="Live Camera",
        review_status="Pending Review",
        reviewer_notes="Rider without helmet"
    )

    assert ev_id.startswith("EV-")

    retrieved = storage.get_event_by_id(ev_id)
    assert retrieved is not None
    assert retrieved["tracking_id"] == "TRK-1001"
    assert retrieved["violation_category"] == "Helmet Violation"
    assert retrieved["review_status"] == "Pending Review"


def test_update_review_status(storage):
    ev_id = storage.log_event(
        violation_category="Seat-Belt Violation",
        tracking_id="TRK-1002",
        vehicle_type="car",
        confidence_score=0.88,
        evidence_path="evidence/test2.jpg"
    )

    storage.update_review_status(ev_id, "Confirmed", "Verified by traffic officer")
    updated = storage.get_event_by_id(ev_id)
    assert updated["review_status"] == "Confirmed"
    assert "Verified by traffic officer" in updated["reviewer_notes"]


def test_statistics_and_export(storage):
    storage.seed_initial_demo_events(force=True)
    stats = storage.get_statistics()

    assert stats["total_violations"] > 0
    assert stats["helmet_events"] >= 0
    assert stats["seatbelt_events"] >= 0
    assert stats["pending_reviews"] >= 0

    df = storage.export_events_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == stats["total_violations"]
