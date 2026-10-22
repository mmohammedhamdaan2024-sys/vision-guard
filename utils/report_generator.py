"""
Vision Guard — PDF Report Generator
Generates official-grade printable E-Challan notices and executive traffic safety audit reports using ReportLab.
"""

import os
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String


def generate_challan_pdf(challan_data: Dict[str, Any], output_path: str) -> str:
    """
    Generates a printable, professional e-Challan PDF document.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChallanTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0B132B"),
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ChallanSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#4361EE"),
        alignment=1,
        spaceAfter=15
    )
    label_style = ParagraphStyle(
        "FieldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#333333")
    )
    val_style = ParagraphStyle(
        "FieldValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#111111")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.HexColor("#777777"),
        alignment=1
    )

    story = []

    # Header Bar
    story.append(Paragraph("VISION GUARD — SMART HIGHWAY SURVEILLANCE", title_style))
    story.append(Paragraph("MINISTRY OF TRANSPORT & TRAFFIC ENFORCEMENT (SIMULATED)", subtitle_style))
    story.append(Spacer(1, 10))

    # Reference & Metadata Banner Table
    ref_num = challan_data.get("challan_ref", "VG-2026-CH-000000")
    status = challan_data.get("review_status", "Pending Review")
    status_color = colors.HexColor("#2EC4B6") if status == "Approved" else colors.HexColor("#E71D36")

    meta_table_data = [
        [
            Paragraph("<b>CHALLAN REFERENCE NO:</b>", label_style),
            Paragraph(f"<font color='#0B132B' size='11'><b>{ref_num}</b></font>", val_style),
            Paragraph("<b>STATUS:</b>", label_style),
            Paragraph(f"<b>{status.upper()}</b>", val_style)
        ],
        [
            Paragraph("<b>DATE & TIME:</b>", label_style),
            Paragraph(str(challan_data.get("issued_at", "N/A")), val_style),
            Paragraph("<b>ISSUED BY:</b>", label_style),
            Paragraph(str(challan_data.get("issued_by", "Vision Guard Automated Highway System")), val_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[1.8 * inch, 2.0 * inch, 1.2 * inch, 2.2 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Vehicle & Violation Details Table
    viol_type = challan_data.get("violation_type", "Traffic Safety Violation")
    plate = challan_data.get("vehicle_plate", "UNREADABLE")
    fine = challan_data.get("fine_amount", 1000)
    notes = challan_data.get("notes", "Visual evidence recorded by camera unit.")

    details_data = [
        [Paragraph("<b>Vehicle Registration Plate</b>", label_style), Paragraph(f"<font size='12'><b>{plate}</b></font>", val_style)],
        [Paragraph("<b>Violation Category</b>", label_style), Paragraph(f"<font color='#E71D36'><b>{viol_type}</b></font>", val_style)],
        [Paragraph("<b>Proposed Demonstration Fine</b>", label_style), Paragraph(f"<font size='11' color='#0B132B'><b>₹{fine:,.2f}</b></font>", val_style)],
        [Paragraph("<b>Officer Review / Detection Notes</b>", label_style), Paragraph(str(notes), val_style)],
    ]
    details_table = Table(details_data, colWidths=[2.5 * inch, 4.7 * inch])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#E2E8F0")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 15))

    # Visual Evidence Section
    story.append(Paragraph("<b>SURVEILLANCE VISUAL EVIDENCE CAPTURE</b>", label_style))
    story.append(Spacer(1, 6))

    evidence_path = challan_data.get("evidence_path", "")
    if evidence_path and os.path.exists(evidence_path):
        try:
            evid_img = RLImage(evidence_path, width=4.5 * inch, height=2.6 * inch)
            story.append(evid_img)
        except Exception:
            story.append(Paragraph("[Visual evidence frame preview unavailable on disk]", val_style))
    else:
        # Mock Evidence drawing placeholder
        d = Drawing(400, 100)
        d.add(Rect(0, 0, 400, 100, fillColor=colors.HexColor("#1C2541"), strokeColor=colors.HexColor("#4361EE")))
        d.add(String(70, 45, f"EVIDENCE ARCHIVE: {ref_num} [ATTACHED]", fontSize=11, fillColor=colors.white))
        story.append(d)

    story.append(Spacer(1, 20))

    # Payment & Notice Section
    notice_text = (
        "<b>LEGAL NOTICE & CITATION DISCLAIMER:</b><br/>"
        "1. This is a computer vision-assisted demonstration notice produced by Vision Guard AI.<br/>"
        "2. All proposed fines are illustrative demonstration amounts and are subject to mandatory review by an authorized enforcement officer under statutory traffic laws.<br/>"
        "3. Experimental image measurements (such as window darkness scores and mirror clarity) are visual approximations and cannot substitute calibrated photometric instruments."
    )
    story.append(Paragraph(notice_text, disclaimer_style))

    doc.build(story)
    return output_path


def generate_audit_report_pdf(metrics: Dict[str, Any], category_data: List[Dict[str, Any]],
                              output_path: str) -> str:
    """
    Generates an executive-level PDF system summary and compliance audit report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = []
    title_style = ParagraphStyle("RptTitle", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#0B132B"), alignment=1)
    sub_style = ParagraphStyle("RptSub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#4361EE"), alignment=1, spaceAfter=20)
    h2_style = ParagraphStyle("RptH2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0B132B"), spaceBefore=10, spaceAfter=6)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9)

    story.append(Paragraph("VISION GUARD — ROAD SAFETY SYSTEM AUDIT", title_style))
    story.append(Paragraph("Automated Surveillance Statistics & Enforcement Summary", sub_style))

    # Executive Summary Table
    summary_data = [
        [Paragraph("<b>Metric</b>", cell_style), Paragraph("<b>Recorded Value</b>", cell_style)],
        [Paragraph("Total Vehicles Scanned", cell_style), Paragraph(str(metrics.get("total_vehicles", 0)), cell_style)],
        [Paragraph("Total Violations Detected", cell_style), Paragraph(str(metrics.get("total_violations", 0)), cell_style)],
        [Paragraph("Helmet Non-Compliance Alerts", cell_style), Paragraph(str(metrics.get("helmet_alerts", 0)), cell_style)],
        [Paragraph("Seat-belt Non-Compliance Alerts", cell_style), Paragraph(str(metrics.get("seatbelt_alerts", 0)), cell_style)],
        [Paragraph("Window & Mirror Review Flags", cell_style), Paragraph(str(metrics.get("tint_mirror_alerts", 0)), cell_style)],
        [Paragraph("Pending Human Officer Reviews", cell_style), Paragraph(str(metrics.get("pending_reviews", 0)), cell_style)],
        [Paragraph("Total Approved Fines", cell_style), Paragraph(f"₹{metrics.get('approved_fines', 0):,.2f}", cell_style)],
    ]
    t = Table(summary_data, colWidths=[3.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B132B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Violations by Category
    story.append(Paragraph("<b>Violations Breakdown by Category</b>", h2_style))
    cat_rows = [[Paragraph("<b>Violation Category</b>", cell_style), Paragraph("<b>Count</b>", cell_style)]]
    for c in category_data:
        cat_rows.append([
            Paragraph(str(c.get("violation_type", "Other")), cell_style),
            Paragraph(str(c.get("count", 0)), cell_style)
        ])
    ct = Table(cat_rows, colWidths=[4.5 * inch, 2.5 * inch])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4361EE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ct)
    story.append(Spacer(1, 20))

    disclaimer = (
        "Report generated by Vision Guard Automated Surveillance Engine. "
        "Intended solely for research, statistical auditing, and educational demonstration."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=styles["Italic"], fontSize=8, textColor=colors.gray, alignment=1)))

    doc.build(story)
    return output_path
