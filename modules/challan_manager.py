"""
Vision Guard — Simulated Challan Management Module
Workflow handling mock citations, officer approvals/rejections, printable PDF generation,
and simulated mobile SMS notification previews.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from database.db_manager import DatabaseManager
from utils.report_generator import generate_challan_pdf
from utils.validation import format_currency


class ChallanManager:
    def __init__(self, db_manager: DatabaseManager, reports_dir: str = "data/reports"):
        self.db = db_manager
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def get_challan(self, challan_ref: str) -> Optional[Dict[str, Any]]:
        return self.db.get_challan_by_ref(challan_ref)

    def list_challans(self, status: str = "All", search: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        return self.db.list_challans(status=status, search=search, limit=limit)

    def process_review_action(self, challan_id: int, new_status: str,
                              reviewer_name: str, comments: str) -> bool:
        """
        Executes officer decision: 'Approved', 'Rejected', or 'Under Investigation'.
        Updates challan status and appends audit log in SQLite.
        """
        try:
            self.db.update_challan_review_status(
                challan_id=challan_id,
                new_status=new_status,
                reviewer_name=reviewer_name,
                comments=comments
            )
            return True
        except Exception as e:
            print(f"Error processing review action: {e}")
            return False

    def export_pdf_challan(self, challan_ref: str) -> Optional[str]:
        """
        Generates and returns the path to a high-quality printable e-Challan PDF.
        """
        challan_data = self.get_challan(challan_ref)
        if not challan_data:
            return None

        filename = f"Challan_{challan_ref.replace('-', '_')}.pdf"
        output_path = os.path.join(self.reports_dir, filename)

        try:
            generate_challan_pdf(challan_data, output_path)
            return output_path
        except Exception as e:
            print(f"PDF generation error: {e}")
            return None

    def simulate_mobile_notification(self, challan_id: int, phone_number: str) -> Dict[str, Any]:
        """
        Simulates an SMS notification dispatch to the vehicle owner.
        DOES NOT make any external telecommunication or API calls.
        Returns a rich preview template for UI display.
        """
        # Retrieve challan
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM challans WHERE id = ?", (challan_id,))
            row = cursor.fetchone()
            if not row:
                return {"success": False, "message": "Challan record not found."}
            challan = dict(row)

        # Mark in DB
        self.db.record_notification(challan_id, phone_number)

        # Generate mock message body
        msg_template = (
            f"GOVT TRAFFIC POLICE NOTICE: E-Challan {challan['challan_ref']} generated for "
            f"vehicle {challan['vehicle_plate']}. Violation: {challan['violation_type']}. "
            f"Illustrative Fine: ₹{challan['fine_amount']:,.2f}. Recorded on {challan['issued_at']}. "
            f"View visual evidence & file dispute online at https://visionguard.demo.gov/review"
        )

        return {
            "success": True,
            "sender_id": "VISIONGUARD-HQ",
            "phone_number": phone_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message_body": msg_template,
            "disclaimer": "SIMULATION ONLY: No actual SMS or external network transmission occurred."
        }
