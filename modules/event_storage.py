"""
VisionGuard — Event Storage Module
SQLite database manager storing violation events, evidence snapshots,
audit reviews, and macro road safety telemetry.
"""

import os
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from config import DB_PATH


class EventStorage:
    """Manages SQLite persistence for VisionGuard violation events and telemetry."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables and indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS violation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    violation_category TEXT NOT NULL,
                    tracking_id TEXT NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    evidence_path TEXT NOT NULL,
                    camera_source TEXT DEFAULT 'Live Camera',
                    review_status TEXT DEFAULT 'Pending Review' CHECK(review_status IN ('Pending Review', 'Confirmed', 'Rejected', 'Inconclusive')),
                    reviewer_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_cat ON violation_events(violation_category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_status ON violation_events(review_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_tracking ON violation_events(tracking_id);")
            conn.commit()

    def seed_initial_demo_events(self, force: bool = False):
        """Seeds realistic historical events across the past 7 days if empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violation_events;")
            count = cursor.fetchone()[0]
            if count > 0 and not force:
                return

            now = datetime.now()
            samples = [
                ("Helmet Violation", "TRK-101", "motorcycle", 0.92, "evidence/sample_helmet_violation.jpg", "Live Camera", "Confirmed", "Rider observed crossing junction without protective helmet.", 6),
                ("Seat-Belt Violation", "TRK-102", "car", 0.86, "evidence/sample_seatbelt_violation.jpg", "Uploaded Video", "Pending Review", "Front passenger unbuckled across highway corridor.", 5),
                ("Red Light Stop-Line Violation", "TRK-103", "car", 0.94, "evidence/sample_tint_violation.jpg", "Live Camera", "Confirmed", "Vehicle penetrated active red light stop line at 22:45.", 4),
                ("Helmet Violation", "TRK-104", "motorcycle", 0.89, "evidence/sample_helmet_violation.jpg", "Live Camera", "Pending Review", "Pillion rider without headgear.", 3),
                ("Seat-Belt Violation", "TRK-105", "truck", 0.83, "evidence/sample_seatbelt_violation.jpg", "Live Camera", "Inconclusive", "Windshield glare obscured diagonal strap visibility.", 3),
                ("Red Light Stop-Line Violation", "TRK-106", "car", 0.91, "evidence/sample_mirror_violation.jpg", "Uploaded Video", "Rejected", "Vehicle stopped 0.5m behind virtual threshold; false trigger.", 2),
                ("Wrong-Way Movement", "TRK-107", "motorcycle", 0.88, "evidence/sample_helmet_violation.jpg", "Live Camera", "Confirmed", "Motorcycle driving northbound on one-way southbound lane.", 1),
                ("Seat-Belt Violation", "TRK-108", "car", 0.87, "evidence/sample_seatbelt_violation.jpg", "Live Camera", "Pending Review", "Driver occupant seatbelt unfastened.", 0),
                ("Helmet Violation", "TRK-109", "motorcycle", 0.95, "evidence/sample_helmet_violation.jpg", "Live Camera", "Pending Review", "Speeding two-wheeler without helmet.", 0),
            ]

            for cat, trk, vtype, conf, evid, src, status, notes, d_ago in samples:
                e_id = f"EV-{now.strftime('%Y%m')}-{random.randint(10000, 99999)}"
                t_stamp = (now - timedelta(days=d_ago, hours=random.randint(1, 8), minutes=random.randint(5, 55))).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO violation_events (event_id, timestamp, violation_category, tracking_id, vehicle_type, confidence_score, evidence_path, camera_source, review_status, reviewer_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (e_id, t_stamp, cat, trk, vtype, conf, evid, src, status, notes)
                )
            conn.commit()

    def log_event(self, violation_category: str, tracking_id: str,
                  vehicle_type: str, confidence_score: float,
                  evidence_path: str, camera_source: str = "Live Camera",
                  review_status: str = "Pending Review",
                  reviewer_notes: str = "") -> str:
        """Records a new violation event into SQLite and returns the unique event ID."""
        event_id = f"EV-{datetime.now().strftime('%Y%m')}-{random.randint(10000, 99999)}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violation_events (event_id, violation_category, tracking_id, vehicle_type, confidence_score, evidence_path, camera_source, review_status, reviewer_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (event_id, violation_category, tracking_id, vehicle_type, confidence_score, evidence_path, camera_source, review_status, reviewer_notes)
            )
            conn.commit()
            return event_id

    def list_events(self, category: Optional[str] = None,
                    status: Optional[str] = None,
                    search: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves filtered violation records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM violation_events WHERE 1=1"
            params: List[Any] = []

            if category and category != "All":
                query += " AND violation_category = ?"
                params.append(category)
            if status and status != "All":
                query += " AND review_status = ?"
                params.append(status)
            if search:
                query += " AND (event_id LIKE ? OR tracking_id LIKE ? OR reviewer_notes LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM violation_events WHERE event_id = ? LIMIT 1;", (event_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_review_status(self, event_id: str, new_status: str, notes: Optional[str] = None):
        """Updates officer adjudication decision: Confirmed, Rejected, or Inconclusive."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if notes is not None:
                cursor.execute(
                    "UPDATE violation_events SET review_status = ?, reviewer_notes = ? WHERE event_id = ?;",
                    (new_status, notes, event_id)
                )
            else:
                cursor.execute(
                    "UPDATE violation_events SET review_status = ? WHERE event_id = ?;",
                    (new_status, event_id)
                )
            conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates macro metrics and aggregates from SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violation_events;")
            total_violations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violation_events WHERE violation_category = 'Helmet Violation';")
            helmet_events = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violation_events WHERE violation_category = 'Seat-Belt Violation';")
            seatbelt_events = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violation_events WHERE violation_category = 'Red Light Stop-Line Violation';")
            redlight_events = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violation_events WHERE review_status = 'Pending Review';")
            pending_reviews = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violation_events WHERE review_status = 'Confirmed';")
            confirmed_events = cursor.fetchone()[0]

            # Category distribution
            cursor.execute("SELECT violation_category, COUNT(*) as count FROM violation_events GROUP BY violation_category;")
            category_counts = [dict(r) for r in cursor.fetchall()]

            # Date-wise trends
            cursor.execute(
                """
                SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM violation_events 
                GROUP BY DATE(timestamp) 
                ORDER BY date ASC LIMIT 14;
                """
            )
            daily_trends = [dict(r) for r in cursor.fetchall()]

            # Vehicle distribution
            cursor.execute("SELECT vehicle_type, COUNT(*) as count FROM violation_events GROUP BY vehicle_type;")
            vehicle_counts = [dict(r) for r in cursor.fetchall()]

            return {
                "total_violations": total_violations,
                "helmet_events": helmet_events,
                "seatbelt_events": seatbelt_events,
                "redlight_events": redlight_events,
                "pending_reviews": pending_reviews,
                "confirmed_events": confirmed_events,
                "category_counts": category_counts,
                "daily_trends": daily_trends,
                "vehicle_counts": vehicle_counts
            }

    def export_events_df(self) -> pd.DataFrame:
        """Returns entire event registry as a pandas DataFrame."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM violation_events ORDER BY id DESC;", conn)
            return df
