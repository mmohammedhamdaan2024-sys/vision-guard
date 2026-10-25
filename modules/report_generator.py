"""
VisionGuard — PDF Report Generator Module
Creates comprehensive, printable PDF road safety audit reports with ReportLab.
Includes executive telemetry summaries, category tables, and visual evidence embeds.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.graphics.shapes import Drawing, Rect, String

from config import REPORTS_DIR


def generate_executive_audit_pdf(stats: Dict[str, Any], events_list: List[Dict[str, Any]],
                                 output_path: Optional[str] = None) -> str:
    """
    Generates a publication-quality PDF audit report summarizing traffic safety telemetry.
    """
    out_file = output_path or str(REPORTS_DIR / "VisionGuard_Road_Safety_Audit_Report.pdf")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    doc = SimpleDocTemplate(out_file, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AuditTitle", parent=styles["Heading1"], fontSize=18,
        textColor=colors.HexColor("#0B132B"), alignment=1, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "AuditSubtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#4361EE"), alignment=1, spaceAfter=16
    )
    h2_style = ParagraphStyle(
        "AuditH2", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#0B132B"), spaceBefore=10, spaceAfter=6
    )
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9)

    story = []

    # Title Banner
    story.append(Paragraph("VISIONGUARD — INTELLIGENT ROAD SAFETY MONITORING SYSTEM", title_style))
    story.append(Paragraph("EXECUTIVE SURVEILLANCE & TRAFFIC VIOLATION AUDIT REPORT", subtitle_style))
    story.append(Spacer(1, 10))

    # Executive Summary Table
    total_v = stats.get("total_violations", 0)
    helmet_v = stats.get("helmet_events", 0)
    seatbelt_v = stats.get("seatbelt_events", 0)
    redlight_v = stats.get("redlight_events", 0)
    pending_v = stats.get("pending_reviews", 0)
    confirmed_v = stats.get("confirmed_events", 0)

    summary_data = [
        [Paragraph("<b>Metric Parameter</b>", cell_style), Paragraph("<b>Recorded Audit Value</b>", cell_style)],
        [Paragraph("Total Safety Violations Logged", cell_style), Paragraph(str(total_v), cell_style)],
        [Paragraph("Motorcycle Helmet Non-Compliance Alerts", cell_style), Paragraph(str(helmet_v), cell_style)],
        [Paragraph("Occupant Seat-Belt Non-Compliance Alerts", cell_style), Paragraph(str(seatbelt_v), cell_style)],
        [Paragraph("Red Light & Stop-Line Incursions", cell_style), Paragraph(str(redlight_v), cell_style)],
        [Paragraph("Cases Confirmed by Officer Review", cell_style), Paragraph(str(confirmed_v), cell_style)],
        [Paragraph("Cases Pending Human Verification", cell_style), Paragraph(str(pending_v), cell_style)],
    ]
    st_table = Table(summary_data, colWidths=[4.2 * inch, 3.0 * inch])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B132B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(st_table)
    story.append(Spacer(1, 15))

    # Category Breakdown Table
    story.append(Paragraph("<b>Violation Events by Category</b>", h2_style))
    cat_rows = [[Paragraph("<b>Violation Category</b>", cell_style), Paragraph("<b>Total Recorded</b>", cell_style)]]
    for c in stats.get("category_counts", []):
        cat_rows.append([
            Paragraph(str(c.get("violation_category", "Unknown")), cell_style),
            Paragraph(str(c.get("count", 0)), cell_style)
        ])
    ct_table = Table(cat_rows, colWidths=[4.5 * inch, 2.7 * inch])
    ct_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4361EE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ct_table)
    story.append(Spacer(1, 15))

    # Sample Evidence Case Embeds
    story.append(Paragraph("<b>Recent Forensic Evidence Snapshot</b>", h2_style))
    if events_list:
        sample_evid = events_list[0].get("evidence_path", "")
        if sample_evid and os.path.exists(sample_evid):
            try:
                img = RLImage(sample_evid, width=4.5 * inch, height=2.4 * inch)
                story.append(img)
            except Exception:
                story.append(Paragraph("[Visual evidence frame preview unavailable]", cell_style))
        else:
            d = Drawing(400, 75)
            d.add(Rect(0, 0, 400, 75, fillColor=colors.HexColor("#1C2541"), strokeColor=colors.HexColor("#4361EE")))
            d.add(String(100, 32, "SURVEILLANCE EVIDENCE ARCHIVE ATTACHED", fontSize=10, fillColor=colors.white))
            story.append(d)

    story.append(Spacer(1, 20))
    disclaimer = (
        "<b>LEGAL & COMPLIANCE NOTICE:</b> This document is generated by the VisionGuard AI surveillance engine. "
        "Detections are advisory and must be verified by certified traffic personnel prior to statutory enforcement."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=styles["Italic"], fontSize=8, textColor=colors.gray, alignment=1)))

    doc.build(story)
    return out_file
