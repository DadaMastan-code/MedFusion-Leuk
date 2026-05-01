"""Generate a realistic CBC blood report PDF for testing."""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from pathlib import Path

OUT_PATH = Path("data/test_samples/sample_blood_report.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=16, textColor=colors.HexColor("#1a3a6b"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#444444"),
        spaceAfter=2,
    )
    header_style = ParagraphStyle(
        "SectionHeader", parent=styles["Normal"],
        fontSize=11, textColor=colors.white,
        backColor=colors.HexColor("#1a3a6b"),
        spaceAfter=0, spaceBefore=8,
        leftIndent=4,
    )

    story = []

    # ── Hospital header ────────────────────────────────────────────────────
    story.append(Paragraph("MedFusion General Hospital", title_style))
    story.append(Paragraph("Department of Haematology & Laboratory Medicine", sub_style))
    story.append(Paragraph("123 Medical Drive, Clinictown, CA 90001 | Tel: +1-800-MED-LAB1", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a6b")))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("COMPLETE BLOOD COUNT (CBC) REPORT", ParagraphStyle(
        "RepTitle", parent=styles["Heading1"],
        fontSize=13, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a3a6b"),
    )))
    story.append(Spacer(1, 0.2*cm))

    # ── Patient info table ─────────────────────────────────────────────────
    patient_data = [
        ["Patient Name:", "Test Patient", "Report Date:", "2026-04-30"],
        ["Patient ID:", "MF-2026-00471", "Lab ID:", "LAB-2026-98312"],
        ["Date of Birth:", "1985-07-14", "Ordering Physician:", "Dr. A. Sharma"],
        ["Gender:", "Male", "Ward / Unit:", "Haematology OPD"],
    ]
    pt = Table(patient_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND",(0,0), (-1,-1), colors.HexColor("#f0f4fa")),
        ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f0f4fa"), colors.white]),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # ── CBC Results table ──────────────────────────────────────────────────
    story.append(Paragraph("  Complete Blood Count Results", header_style))
    story.append(Spacer(1, 0.1*cm))

    cbc_header = ["Parameter", "Result", "Units", "Reference Range", "Flag"]
    cbc_data = [
        cbc_header,
        ["WBC Count",          "95,000",  "/µL",       "4,000 – 11,000",   "*** HIGH ***"],
        ["RBC Count",          "3.1",     "×10⁶/µL",   "4.5 – 5.5",        "LOW"],
        ["Hemoglobin",         "7.8",     "g/dL",      "13.5 – 17.5",      "** LOW **"],
        ["Hematocrit (HCT)",   "24.2",    "%",         "41 – 53",          "LOW"],
        ["MCV",                "78.1",    "fL",        "80 – 100",         "LOW"],
        ["MCH",                "25.2",    "pg",        "27 – 33",          "LOW"],
        ["MCHC",               "32.2",    "g/dL",      "32 – 36",          "Normal"],
        ["Platelet Count",     "42,000",  "/µL",       "150,000 – 400,000","*** LOW ***"],
        ["Blast Percentage",   "72",      "%",         "0 – 5",            "*** CRITICAL ***"],
    ]
    cbc_tbl = Table(cbc_data, colWidths=[4.5*cm, 2.5*cm, 2.2*cm, 4.5*cm, 3.5*cm])
    cbc_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f8f9fa")]),
        ("TEXTCOLOR",    (4,1), (4,-1), colors.HexColor("#cc0000")),
        ("FONTNAME",     (4,1), (4,-1), "Helvetica-Bold"),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(cbc_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Chemistry panel ────────────────────────────────────────────────────
    story.append(Paragraph("  Biochemistry / Chemistry Panel", header_style))
    story.append(Spacer(1, 0.1*cm))

    chem_header = ["Test", "Result", "Units", "Reference Range", "Flag"]
    chem_data = [
        chem_header,
        ["LDH (Lactate Dehydrogenase)", "850",  "U/L",    "140 – 280",  "*** HIGH ***"],
        ["Uric Acid",                    "8.2",  "mg/dL",  "3.5 – 7.2",  "HIGH"],
        ["Serum Calcium",               "9.1",   "mg/dL",  "8.5 – 10.5", "Normal"],
        ["Total Protein",               "6.2",   "g/dL",   "6.3 – 8.2",  "Low"],
        ["Albumin",                     "3.4",   "g/dL",   "3.5 – 5.0",  "Low"],
    ]
    chem_tbl = Table(chem_data, colWidths=[5.5*cm, 2*cm, 2.2*cm, 4*cm, 3.5*cm])
    chem_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f8f9fa")]),
        ("TEXTCOLOR",     (4,1), (4,-1), colors.HexColor("#cc0000")),
        ("FONTNAME",      (4,1), (4,-1), "Helvetica-Bold"),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(chem_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Clinical comment ───────────────────────────────────────────────────
    story.append(Paragraph("  Haematologist's Comment", header_style))
    story.append(Spacer(1, 0.1*cm))
    comment = (
        "CRITICAL VALUES PRESENT. Markedly elevated WBC count of 95,000/µL with "
        "blast percentage of 72% is highly suggestive of Acute Myeloid Leukemia (AML). "
        "Severe anaemia (Hb 7.8 g/dL) and thrombocytopenia (Platelets 42,000/µL) "
        "consistent with bone marrow infiltration. Elevated LDH (850 U/L) and uric acid "
        "(8.2 mg/dL) indicate high tumour burden. Urgent haematology referral recommended. "
        "Bone marrow biopsy and cytogenetic studies advised."
    )
    story.append(Paragraph(comment, ParagraphStyle(
        "Comment", parent=styles["Normal"], fontSize=9,
        leftIndent=4, rightIndent=4, spaceAfter=4,
        backColor=colors.HexColor("#fff3cd"), borderPadding=6,
    )))
    story.append(Spacer(1, 0.5*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        "This report is for clinical decision support only and must be interpreted "
        "by a qualified hematologist. Report generated: 2026-04-30 09:47 UTC",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"PDF created: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_pdf()
