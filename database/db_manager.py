"""
Vision Guard — Database Manager
Handles SQLite connection, schema execution, persistence, queries, and demo seeding.
"""

import os
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "visionguard.db")
DEFAULT_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self, schema_path: str = DEFAULT_SCHEMA_PATH):
        """Initializes tables and default settings from schema.sql."""
        if not os.path.exists(schema_path):
            schema_path = DEFAULT_SCHEMA_PATH
            
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
            
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def seed_demo_data(self, force: bool = False):
        """Seeds realistic demonstration records if database is empty or force=True."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM violations")
            count = cursor.fetchone()[0]
            if count > 0 and not force:
                return  # Already seeded

            # Seed realistic traffic data spanning the past 7 days
            sample_vehicles = [
                ("TRK-101", "motorcycle", "DL 03 AC 8821", 0.94),
                ("TRK-102", "car", "MH 12 QW 4412", 0.91),
                ("TRK-103", "car", "KA 05 MN 9010", 0.88),
                ("TRK-104", "motorcycle", "UP 16 ZY 3311", 0.95),
                ("TRK-105", "truck", "HR 26 BD 7720", 0.82),
                ("TRK-106", "car", "TN 09 ER 1055", 0.90),
                ("TRK-107", "motorcycle", "RJ 14 XY 6231", 0.86),
                ("TRK-108", "bus", "GJ 01 TT 5599", 0.93),
                ("TRK-109", "car", "CH 01 AF 4902", 0.79),
                ("TRK-110", "motorcycle", "WB 02 KK 1184", 0.89),
                ("TRK-111", "car", "KL 07 CD 3409", 0.92),
                ("TRK-112", "car", "TS 08 HG 8122", 0.87),
            ]

            vehicle_ids = []
            now = datetime.now()
            for trk_id, vtype, plate, pconf in sample_vehicles:
                days_ago = random.randint(0, 6)
                seen_time = (now - timedelta(days=days_ago, hours=random.randint(1, 10), minutes=random.randint(5, 50))).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO vehicles (tracking_id, vehicle_type, plate_number, plate_confidence, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (trk_id, vtype, plate, pconf, seen_time, seen_time)
                )
                vehicle_ids.append(cursor.lastrowid)

            sample_violations = [
                (vehicle_ids[0], "TRK-101", "Helmet Violation", "Critical", 0.91, "Approved", "evidence/sample_helmet_violation.jpg", 1000, "Rider detected without protective headgear at signal junction.", 6),
                (vehicle_ids[1], "TRK-102", "Seat-belt Violation", "Critical", 0.87, "Pending Review", "evidence/sample_seatbelt_violation.jpg", 1000, "Driver occupant seatbelt unfastened across torso.", 5),
                (vehicle_ids[2], "TRK-103", "Dark Window Tint Review", "Warning", 0.74, "Pending Review", "evidence/sample_tint_violation.jpg", 750, "Apparent darkness 78.4% exceeds demonstration threshold of 65%.", 4),
                (vehicle_ids[3], "TRK-104", "Helmet Violation", "Critical", 0.96, "Approved", "evidence/sample_helmet_violation.jpg", 1000, "Motorcycle rider without helmet on high-speed flyover.", 4),
                (vehicle_ids[4], "TRK-105", "Mirror Obscured", "Review", 0.68, "Under Investigation", "evidence/sample_mirror_violation.jpg", 500, "Passenger side rear-view mirror folded or occluded.", 3),
                (vehicle_ids[5], "TRK-106", "Seat-belt Violation", "Critical", 0.84, "Rejected", "evidence/sample_seatbelt_violation.jpg", 1000, "False positive appeal: Dark sash fabric mistaken for unbuckled belt.", 3),
                (vehicle_ids[6], "TRK-107", "Helmet Violation", "Critical", 0.89, "Pending Review", "evidence/sample_helmet_violation.jpg", 1000, "Pillion rider helmet missing; rider wearing helmet.", 2),
                (vehicle_ids[7], "TRK-108", "Plate Unreadable", "Review", 0.35, "Pending Review", "evidence/sample_plate_violation.jpg", 2000, "High mud splatter over rear registration plate.", 2),
                (vehicle_ids[8], "TRK-109", "Dark Window Tint Review", "Warning", 0.79, "Approved", "evidence/sample_tint_violation.jpg", 750, "Heavy opaque aftermarket black film on front side windows.", 1),
                (vehicle_ids[9], "TRK-110", "Helmet Violation", "Critical", 0.93, "Pending Review", "evidence/sample_helmet_violation.jpg", 1000, "Speeding rider with no safety helmet.", 0),
                (vehicle_ids[10], "TRK-111", "Seat-belt Violation", "Critical", 0.88, "Approved", "evidence/sample_seatbelt_violation.jpg", 1000, "Front passenger and driver both without seatbelt.", 0),
            ]

            for vid, trk, vtype, sev, conf, status, evid, fine, notes, d_ago in sample_violations:
                v_time = (now - timedelta(days=d_ago, hours=random.randint(1, 8), minutes=random.randint(10, 45))).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO violations (vehicle_id, tracking_id, violation_type, severity, confidence, status, evidence_path, review_notes, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (vid, trk, vtype, sev, conf, status, evid, notes, v_time)
                )
                viol_id = cursor.lastrowid
                
                # Fetch plate
                cursor.execute("SELECT plate_number FROM vehicles WHERE id = ?", (vid,))
                p_row = cursor.fetchone()
                plate = p_row[0] if p_row else "UNREADABLE"
                
                # Create corresponding Challan
                challan_ref = f"VG-2026-CH-{random.randint(100000, 999999)}"
                cursor.execute(
                    """
                    INSERT INTO challans (challan_ref, violation_id, vehicle_plate, violation_type, fine_amount, review_status, payment_status, evidence_path, issued_by, issued_at, mobile_notified, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (challan_ref, viol_id, plate, vtype, fine, status, "Unpaid" if status != "Approved" else "Paid", evid, "Vision Guard Highway Camera #04", v_time, 1 if status == "Approved" else 0, notes)
                )
                challan_id = cursor.lastrowid
                
                # Review log
                if status in ["Approved", "Rejected"]:
                    cursor.execute(
                        """
                        INSERT INTO review_logs (challan_id, reviewer_name, action, comments, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (challan_id, "Officer R. Sharma", status, f"Officer review completed: {status}.", v_time)
                    )

            conn.commit()

    # --- Vehicles ---
    def add_vehicle(self, tracking_id: str, vehicle_type: str, plate_number: str = "UNREADABLE",
                    plate_confidence: float = 0.0, snapshot_path: str = "") -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO vehicles (tracking_id, vehicle_type, plate_number, plate_confidence, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tracking_id, vehicle_type, plate_number, plate_confidence, snapshot_path)
            )
            conn.commit()
            return cursor.lastrowid

    def get_vehicle_by_tracking_id(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicles WHERE tracking_id = ? ORDER BY id DESC LIMIT 1", (tracking_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_vehicle_plate(self, tracking_id: str, plate_number: str, plate_confidence: float):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE vehicles 
                SET plate_number = ?, plate_confidence = ?, last_seen = CURRENT_TIMESTAMP
                WHERE tracking_id = ?
                """,
                (plate_number, plate_confidence, tracking_id)
            )
            conn.commit()

    def list_vehicles(self, limit: int = 100, search: str = "") -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if search:
                cursor.execute(
                    """
                    SELECT * FROM vehicles 
                    WHERE plate_number LIKE ? OR tracking_id LIKE ? OR vehicle_type LIKE ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (f"%{search}%", f"%{search}%", f"%{search}%", limit)
                )
            else:
                cursor.execute("SELECT * FROM vehicles ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Violations ---
    def add_violation(self, vehicle_id: Optional[int], tracking_id: str, violation_type: str,
                      severity: str, confidence: float, evidence_path: str,
                      status: str = "Pending Review", review_notes: str = "") -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO violations (vehicle_id, tracking_id, violation_type, severity, confidence, status, evidence_path, review_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (vehicle_id, tracking_id, violation_type, severity, confidence, status, evidence_path, review_notes)
            )
            conn.commit()
            return cursor.lastrowid

    def list_violations(self, status: Optional[str] = None, violation_type: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT v.*, veh.plate_number, veh.vehicle_type 
                FROM violations v
                LEFT JOIN vehicles veh ON v.vehicle_id = veh.id
                WHERE 1=1
            """
            params: List[Any] = []
            if status and status != "All":
                query += " AND v.status = ?"
                params.append(status)
            if violation_type and violation_type != "All":
                query += " AND v.violation_type = ?"
                params.append(violation_type)
            query += " ORDER BY v.id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def update_violation_status(self, violation_id: int, new_status: str, notes: str = ""):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE violations
                SET status = ?, review_notes = CASE WHEN ? != '' THEN ? ELSE review_notes END
                WHERE id = ?
                """,
                (new_status, notes, notes, violation_id)
            )
            conn.commit()

    # --- Challans ---
    def create_challan(self, violation_id: int, vehicle_plate: str, violation_type: str,
                       fine_amount: float, evidence_path: str = "", notes: str = "",
                       issued_by: str = "Vision Guard Automated System") -> str:
        challan_ref = f"VG-2026-CH-{random.randint(100000, 999999)}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challans (challan_ref, violation_id, vehicle_plate, violation_type, fine_amount, evidence_path, notes, issued_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (challan_ref, violation_id, vehicle_plate, violation_type, fine_amount, evidence_path, notes, issued_by)
            )
            conn.commit()
            return challan_ref

    def list_challans(self, status: Optional[str] = None, search: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT c.*, v.confidence, v.severity, v.tracking_id
                FROM challans c
                LEFT JOIN violations v ON c.violation_id = v.id
                WHERE 1=1
            """
            params: List[Any] = []
            if status and status != "All":
                query += " AND c.review_status = ?"
                params.append(status)
            if search:
                query += " AND (c.challan_ref LIKE ? OR c.vehicle_plate LIKE ? OR c.violation_type LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
            query += " ORDER BY c.id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_challan_by_ref(self, challan_ref: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.*, v.confidence, v.severity, v.tracking_id, v.evidence_path as viol_evidence
                FROM challans c
                LEFT JOIN violations v ON c.violation_id = v.id
                WHERE c.challan_ref = ?
                """,
                (challan_ref,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_challan_review_status(self, challan_id: int, new_status: str, reviewer_name: str, comments: str = ""):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE challans SET review_status = ?, notes = ? WHERE id = ?",
                (new_status, comments, challan_id)
            )
            # Sync with violation table
            cursor.execute(
                """
                UPDATE violations SET status = ?, review_notes = ?
                WHERE id = (SELECT violation_id FROM challans WHERE id = ?)
                """,
                (new_status, comments, challan_id)
            )
            cursor.execute(
                """
                INSERT INTO review_logs (challan_id, reviewer_name, action, comments)
                VALUES (?, ?, ?, ?)
                """,
                (challan_id, reviewer_name, new_status, comments)
            )
            conn.commit()

    def record_notification(self, challan_id: int, phone_number: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE challans 
                SET mobile_notified = 1, notification_phone = ?
                WHERE id = ?
                """,
                (phone_number, challan_id)
            )
            conn.commit()

    def get_review_logs(self, challan_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM review_logs WHERE challan_id = ? ORDER BY id DESC", (challan_id,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Analytics & Dashboard Aggregates ---
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vehicles")
            total_vehicles = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violations")
            total_violations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'Helmet Violation'")
            helmet_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type = 'Seat-belt Violation'")
            seatbelt_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM violations WHERE violation_type IN ('Dark Window Tint Review', 'Mirror Obscured')")
            tint_mirror_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM challans WHERE review_status = 'Pending Review'")
            pending_reviews = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(fine_amount) FROM challans WHERE review_status = 'Approved'")
            row_fines = cursor.fetchone()[0]
            approved_fines = row_fines if row_fines else 0.0

            return {
                "total_vehicles": total_vehicles,
                "total_violations": total_violations,
                "helmet_alerts": helmet_alerts,
                "seatbelt_alerts": seatbelt_alerts,
                "tint_mirror_alerts": tint_mirror_alerts,
                "pending_reviews": pending_reviews,
                "approved_fines": approved_fines
            }

    def get_violations_by_category(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT violation_type, COUNT(*) as count 
                FROM violations 
                GROUP BY violation_type 
                ORDER BY count DESC
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_daily_violation_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DATE(timestamp) as date, COUNT(*) as count,
                       SUM(CASE WHEN violation_type = 'Helmet Violation' THEN 1 ELSE 0 END) as helmet,
                       SUM(CASE WHEN violation_type = 'Seat-belt Violation' THEN 1 ELSE 0 END) as seatbelt,
                       SUM(CASE WHEN violation_type NOT IN ('Helmet Violation', 'Seat-belt Violation') THEN 1 ELSE 0 END) as others
                FROM violations
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
                LIMIT ?
                """,
                (days,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_setting(self, key: str, default: str = "") -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def set_setting(self, key: str, value: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO app_settings (key, value, updated_at) 
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (key, value)
            )
            conn.commit()
