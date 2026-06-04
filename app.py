from __future__ import annotations

from io import BytesIO
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Alignment, Font, PatternFill

from models import CompanyProfile, ReviewOptions
from reviewer import build_ai_review_memo, findings_to_markdown, normalize_reporting_currency, review_pdf


def _metric_lines(value: object, empty: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text == empty:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _finding_rows(result) -> list[dict[str, str]]:
    rows = []
    for index, finding in enumerate(result.findings, start=1):
        metadata = finding.metadata or {}
        rows.append(
            {
            "ID": f"EX-{index:03d}",
            "Status": "Open",
            "Severity": finding.severity,
            "Category": finding.category,
            "Check type": finding.category,
            "Confidence": _finding_confidence(finding, result),
            "Page reference": _page_reference(finding.location, finding.evidence),
            "Statement": metadata.get("statement", ""),
            "Line item": metadata.get("line_item", ""),
            "Referenced note": metadata.get("referenced_note", ""),
            "Suggested note": metadata.get("suggested_note", ""),
            "Match confidence": metadata.get("match_confidence", finding.severity),
            "Reason": metadata.get("reason", ""),
            "Location": finding.location,
            "Issue": finding.issue,
            "Evidence": finding.evidence,
            "Recommendation": finding.recommendation,
            "Reviewer comment": "",
            "Prepared by": "",
            "Reviewed by": "",
            "Date cleared": "",
            }
        )
    return rows


def _finding_confidence(finding, result) -> str:
    if finding.metadata and finding.metadata.get("match_confidence"):
        return f"Review prompt / {finding.metadata['match_confidence']}"
    if finding.category == "Extraction quality":
        if finding.location in {"PDF extraction", "Table extraction", "Notes agreement"}:
            return f"Text {result.metrics.get('extraction_confidence', '0%')} / Table {result.metrics.get('table_confidence', '0%')}"
        return finding.severity
    if finding.severity == "High":
        return "High"
    if finding.severity == "Medium":
        return "Medium"
    return "Low"


def _page_reference(location: str, evidence: str = "") -> str:
    text = f"{location}\n{evidence}"
    pages = sorted({int(match) for match in re.findall(r"\bPage\s+(\d+)\b", text, flags=re.I)})
    if not pages:
        return ""
    if len(pages) == 1:
        return f"Page {pages[0]}"
    return "Pages " + ", ".join(str(page) for page in pages)


def _note_heading_rows(result) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in _metric_lines(result.metrics.get("note_headings"), "No note headings detected."):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3:
            rows.append(
                {
                    "Note": parts[0].replace("Note ", "", 1).strip(),
                    "Page": parts[1].replace("Page ", "", 1).strip(),
                    "Heading": " | ".join(parts[2:]).strip(),
                }
            )
    return rows


def _build_excel_export(result) -> bytes:
    output = BytesIO()
    summary_rows = [
        {"Metric": "Checks performed", "Value": result.metrics.get("checks_performed_count", 0)},
        {"Metric": "Checks passed", "Value": result.metrics.get("checks_passed_count", 0)},
        {"Metric": "Checks skipped", "Value": result.metrics.get("checks_skipped_count", 0)},
        {"Metric": "Findings", "Value": result.metrics.get("findings", 0)},
        {"Metric": "High findings", "Value": result.metrics.get("high", 0)},
        {"Metric": "Medium findings", "Value": result.metrics.get("medium", 0)},
        {"Metric": "Low findings", "Value": result.metrics.get("low", 0)},
        {"Metric": "Extraction confidence", "Value": result.metrics.get("extraction_confidence", "0%")},
        {"Metric": "Table confidence", "Value": result.metrics.get("table_confidence", "0%")},
        {"Metric": "cautious_note_validation_enabled", "Value": result.metrics.get("cautious_note_validation_enabled", False)},
        {"Metric": "note_validation_mode", "Value": result.metrics.get("note_validation_mode", "skipped")},
        {"Metric": "note_reference_rows_detected", "Value": result.metrics.get("note_reference_rows_detected", 0)},
        {"Metric": "note_headings_detected", "Value": result.metrics.get("note_headings_detected", 0)},
        {"Metric": "note_reference_findings", "Value": result.metrics.get("note_reference_findings", 0)},
    ]
    checks_performed = [{"Check performed": item} for item in _metric_lines(result.metrics.get("checks_performed"), "No deterministic checks completed.")]
    checks_skipped = [{"Check skipped": item} for item in _metric_lines(result.metrics.get("checks_skipped"), "No major checks skipped.")]
    detected_profile = result.metrics.get("detected_profile", {})
    profile_rows = [{"Field": key, "Detected value": value} for key, value in detected_profile.items()] if isinstance(detected_profile, dict) else []
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        exception_rows = _finding_rows(result) or [
            {
                "ID": "",
                "Status": "Noted",
                "Severity": "",
                "Category": "",
                "Check type": "",
                "Confidence": "",
                "Page reference": "",
                "Statement": "",
                "Line item": "",
                "Referenced note": "",
                "Suggested note": "",
                "Match confidence": "",
                "Reason": "",
                "Location": "",
                "Issue": "No automated findings were identified.",
                "Evidence": "",
                "Recommendation": "",
                "Reviewer comment": "",
                "Prepared by": "",
                "Reviewed by": "",
                "Date cleared": "",
            }
        ]
        pd.DataFrame(exception_rows).to_excel(writer, sheet_name="Exception register", index=False)
        pd.DataFrame(checks_performed).to_excel(writer, sheet_name="Checks performed", index=False)
        pd.DataFrame(checks_skipped).to_excel(writer, sheet_name="Checks skipped", index=False)
        pd.DataFrame(_note_heading_rows(result)).to_excel(writer, sheet_name="Notes detected", index=False)
        pd.DataFrame(profile_rows).to_excel(writer, sheet_name="Detected profile", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F2937")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value or "")) for cell in column_cells), default=10)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 70)
        _format_exception_register_sheet(writer.book["Exception register"])
    return output.getvalue()


def _format_exception_register_sheet(worksheet) -> None:
    max_row = max(worksheet.max_row, 2)
    max_col = worksheet.max_column
    table_ref = f"A1:{get_column_letter(max_col)}{max_row}"
    table = Table(displayName="ExceptionRegister", ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    worksheet.add_table(table)
    headers = {str(cell.value): cell.column for cell in worksheet[1]}
    status_col = headers.get("Status")
    severity_col = headers.get("Severity")
    if status_col:
        status_letter = get_column_letter(status_col)
        validation = DataValidation(type="list", formula1='"Open,Cleared,False Positive,Noted"', allow_blank=True)
        worksheet.add_data_validation(validation)
        validation.add(f"{status_letter}2:{status_letter}500")
    if severity_col:
        severity_letter = get_column_letter(severity_col)
        color_map = {
            "High": ("FEE2E2", "991B1B"),
            "Medium": ("FEF3C7", "92400E"),
            "Low": ("DBEAFE", "1E40AF"),
        }
        for severity, (fill, font) in color_map.items():
            worksheet.conditional_formatting.add(
                f"{severity_letter}2:{severity_letter}{max_row}",
                FormulaRule(
                    formula=[f'${severity_letter}2="{severity}"'],
                    fill=PatternFill("solid", fgColor=fill),
                    font=Font(color=font, bold=True),
                ),
            )
    for header in ("Evidence", "Recommendation", "Issue", "Reviewer comment"):
        column = headers.get(header)
        if column:
            letter = get_column_letter(column)
            worksheet.column_dimensions[letter].width = 55 if header in {"Evidence", "Recommendation", "Issue"} else 32
            for row in range(2, max_row + 1):
                worksheet[f"{letter}{row}"].alignment = Alignment(wrap_text=True, vertical="top")
    worksheet.auto_filter.ref = table_ref


def _docx_paragraph(text: str, style: str = "") -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{xml_escape(text)}</w:t></w:r></w:p>"


def _docx_table(rows: list[list[object]]) -> str:
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for value in row:
            display_value = "" if value is None else str(value)
            shading = '<w:shd w:fill="1F2937"/>' if row_index == 0 else ""
            color = '<w:color w:val="FFFFFF"/>' if row_index == 0 else ""
            bold = "<w:b/>" if row_index == 0 else ""
            cells.append(
                "<w:tc>"
                f"<w:tcPr>{shading}</w:tcPr>"
                f"<w:p><w:r><w:rPr>{bold}{color}</w:rPr><w:t>{xml_escape(display_value)}</w:t></w:r></w:p>"
                "</w:tc>"
            )
        table_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        "<w:tbl>"
        '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        f"{''.join(table_rows)}"
        "</w:tbl>"
    )


def _detected_profile_rows(result) -> list[list[str]]:
    detected = result.metrics.get("detected_profile", {})
    if not isinstance(detected, dict):
        detected = {}
    wanted = ("Company name", "Year end", "Currency", "Framework", "Entity type", "Extraction confidence")
    return [["Field", "Detected value"], *[[field, str(detected.get(field, "Not detected"))] for field in wanted]]


def _build_word_memo_export(result) -> bytes:
    memo = build_ai_review_memo(result)
    assurance = str(result.metrics.get("positive_assurance", ""))
    disclaimer = (
        "This automated review supports financial statement review procedures but does not replace professional "
        "judgement, firm methodology, or the auditor's responsibility to evaluate the report and underlying evidence."
    )
    body_parts = [
        _docx_paragraph("AI Audit Assistant Review Memo", "Title"),
        _docx_paragraph("Detected Company Profile", "Heading1"),
        _docx_table(_detected_profile_rows(result)),
        _docx_paragraph("Executive Memo", "Heading1"),
        _docx_paragraph(memo),
        _docx_paragraph(disclaimer),
    ]
    if assurance:
        body_parts.append(_docx_paragraph(assurance))
    body_parts.extend(
        [
            _docx_paragraph("Dashboard", "Heading1"),
            _docx_table(
                [
                    ["Metric", "Value"],
                    ["Checks performed", result.metrics.get("checks_performed_count", 0)],
                    ["Checks passed", result.metrics.get("checks_passed_count", 0)],
                    ["Checks skipped", result.metrics.get("checks_skipped_count", 0)],
                    ["Findings", result.metrics.get("findings", 0)],
                    ["High findings", result.metrics.get("high", 0)],
                    ["Medium findings", result.metrics.get("medium", 0)],
                    ["Low findings", result.metrics.get("low", 0)],
                    ["Extraction confidence", result.metrics.get("extraction_confidence", "0%")],
                    ["Table confidence", result.metrics.get("table_confidence", "0%")],
                ]
            ),
            _docx_paragraph("Checks Performed", "Heading1"),
        ]
    )
    body_parts.extend(_docx_paragraph(item) for item in _metric_lines(result.metrics.get("checks_performed"), "No deterministic checks completed."))
    body_parts.append(_docx_paragraph("Checks Skipped", "Heading1"))
    skipped = _metric_lines(result.metrics.get("checks_skipped"), "No major checks skipped.")
    body_parts.extend(_docx_paragraph(item) for item in skipped) if skipped else body_parts.append(_docx_paragraph("No major checks skipped."))
    body_parts.append(_docx_paragraph("Findings Summary", "Heading1"))
    if result.findings:
        grouped: dict[tuple[str, str], list[object]] = {}
        for finding in result.findings:
            grouped.setdefault((finding.severity, finding.category), []).append(finding)
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        for (severity, category), findings in sorted(grouped.items(), key=lambda item: (severity_order.get(item[0][0], 9), item[0][1])):
            body_parts.append(_docx_paragraph(f"{severity} | {category}", "Heading2"))
            body_parts.append(
                _docx_table(
                    [
                        ["Location", "Issue", "Evidence", "Recommendation"],
                        *[
                            [finding.location, finding.issue, finding.evidence, finding.recommendation]
                            for finding in findings
                        ],
                    ]
                )
            )
    else:
        body_parts.append(_docx_paragraph("No automated findings were identified."))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr></w:body>"
        "</w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document_xml)
    return output.getvalue()


st.set_page_config(page_title="AI Audit Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --ink: #101820;
        --muted: #627181;
        --line: #d9e0e8;
        --panel: #ffffff;
        --panel-soft: #f6f8fb;
        --navy: #071629;
        --navy-2: #0d243d;
        --gold: #b9934a;
        --gold-soft: #efe4cf;
        --green: #126c55;
        --red: #9d2433;
    }

    .stApp {
        background:
            linear-gradient(180deg, #f4f6f9 0%, #fbfcfd 38%, #ffffff 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244,246,249,.82);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }

    .premium-hero {
        background:
            linear-gradient(135deg, rgba(7,22,41,.98), rgba(13,36,61,.94)),
            radial-gradient(circle at 85% 12%, rgba(185,147,74,.28), transparent 34%);
        border: 1px solid rgba(185,147,74,.22);
        border-radius: 8px;
        padding: 34px 38px 30px;
        box-shadow: 0 24px 70px rgba(7,22,41,.18);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: var(--gold);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .premium-title {
        color: #ffffff !important;
        font-size: 42px;
        line-height: 1.05;
        font-weight: 700;
        margin: 0;
    }

    .premium-subtitle {
        color: #c9d5df !important;
        max-width: 780px;
        font-size: 16px;
        line-height: 1.6;
        margin: 14px 0 0;
    }

    .module-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 28px;
    }

    .module-card {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 18px 16px;
        min-height: 176px;
        box-shadow: 0 14px 38px rgba(16,24,32,.07);
    }

    .module-index {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .module-title {
        color: var(--ink);
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .module-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }

    .section-label {
        color: var(--ink);
        font-size: 18px;
        font-weight: 760;
        margin: 8px 0 8px;
    }

    .upload-shell {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px 8px;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
        margin-bottom: 22px;
    }

    .profile-shell {
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 20px 22px 8px;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
        margin-bottom: 24px;
    }

    .profile-heading {
        color: var(--ink);
        font-size: 18px;
        font-weight: 760;
        margin-bottom: 4px;
    }

    .profile-copy {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 16px;
    }

    .memo-panel {
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid var(--line);
        border-left: 4px solid var(--gold);
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 14px 38px rgba(16,24,32,.07);
        margin: 10px 0 18px;
    }

    .memo-kicker {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .memo-text {
        color: var(--ink);
        font-size: 15px;
        line-height: 1.65;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff, #f7f9fb);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 14px 34px rgba(16,24,32,.07);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 780;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #cbd5df !important;
        border-radius: 6px !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8a97a5 !important;
    }

    .stTextInput label p,
    .stTextArea label p,
    .stNumberInput label p,
    .stSelectbox label p,
    .stFileUploader label p,
    .stMultiSelect label p,
    .stSlider label p,
    .stToggle label p {
        color: var(--ink) !important;
        font-weight: 700;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    label,
    label *,
    .stToggle p,
    .stToggle span {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    .control-label {
        color: var(--ink);
        font-size: 14px;
        font-weight: 760;
        margin: 2px 0 8px;
    }

    .stToggle,
    .stToggle *,
    .stSlider,
    .stSlider *,
    .stNumberInput,
    .stNumberInput * {
        color: var(--ink) !important;
    }

    .stNumberInput button {
        background: #f3f6f9 !important;
        color: var(--ink) !important;
        border-color: #cbd5df !important;
        box-shadow: none !important;
    }

    .stSlider [data-baseweb="slider"] div {
        color: var(--ink) !important;
    }

    [data-testid="stTooltipIcon"] {
        color: #627181 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] section {
        background: #252630 !important;
        border: 1px solid #252630 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #f7f9fb !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #ffffff !important;
        border: 1px solid #d9e0e8 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileName"] *,
    [data-testid="stFileUploaderFileSize"],
    [data-testid="stFileUploaderFileSize"] * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] svg {
        color: #627181 !important;
        fill: #627181 !important;
    }

    .stDownloadButton button,
    .stButton button {
        background: linear-gradient(135deg, var(--navy), var(--navy-2));
        color: #ffffff;
        border: 1px solid rgba(185,147,74,.48);
        border-radius: 6px;
        min-height: 42px;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(7,22,41,.18);
    }

    .stDownloadButton button:hover,
    .stButton button:hover {
        border-color: var(--gold);
        color: #ffffff;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 16px 42px rgba(16,24,32,.07);
    }

    .streamlit-expanderHeader {
        font-weight: 700;
        color: var(--ink);
    }

    @media (max-width: 1100px) {
        .module-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .premium-title {
            font-size: 34px;
        }
    }

    @media (max-width: 680px) {
        .module-grid {
            grid-template-columns: 1fr;
        }
        .premium-hero {
            padding: 26px 22px;
        }
        .premium-title {
            font-size: 30px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="premium-hero">
        <div class="eyebrow">Financial Statement Intelligence</div>
        <div class="premium-title">AI Audit Assistant</div>
        <p class="premium-subtitle">
            A high-assurance review workspace for prepared financial statements, built to surface arithmetic,
            presentation, note-agreement, policy, and standards-checklist exceptions before final sign-off.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
        <div class="profile-heading">Review mode</div>
        <div class="profile-copy">Quick Review only needs the PDF. Advanced Review lets you override the detected engagement profile.</div>
        """,
        unsafe_allow_html=True,
    )
    review_mode = st.radio("Mode", ["Quick Review", "Advanced Review"], horizontal=True)
    company_name = ""
    industry = ""
    reporting_currency = "Not specified"
    presentation_standard = "IFRS"
    expected_policies_text = ""
    significant_transactions_text = ""
    checklist_areas_text = ""
    if review_mode == "Advanced Review":
        profile_cols = st.columns([1.2, 1, 0.8, 0.8])
        company_name = profile_cols[0].text_input("Company name")
        industry = profile_cols[1].text_input("Industry")
        currency_choice = profile_cols[2].selectbox(
            "Reporting currency",
            ["Not specified", "NGN", "USD", "GBP", "EUR", "ZAR", "GHS", "KES", "Other"],
            index=0,
        )
        reporting_currency = currency_choice
        if currency_choice == "Other":
            reporting_currency = profile_cols[2].text_input("Custom currency", placeholder="Example: NGN")
        presentation_standard = profile_cols[3].selectbox("Presentation standard", ["IFRS", "Local GAAP"], index=0)
        detail_cols = st.columns(3)
        expected_policies_text = detail_cols[0].text_area(
            "Expected policies",
            placeholder="Example: revenue, financial instruments, tax",
            help="Comma-separated policy areas that are expected even if balances are not obvious in the extracted PDF text.",
        )
        significant_transactions_text = detail_cols[1].text_area(
            "Significant transactions",
            placeholder="Example: leases, share-based payments, foreign currency loans",
            help="Comma-separated transactions or balances that should have tailored accounting policy coverage.",
        )
        checklist_areas_text = detail_cols[2].text_area(
            "Force checklist areas",
            placeholder="Example: IFRS 15, IFRS 16, revenue, leases, EPS",
            help="Comma-separated standards or areas to check even if the PDF text does not clearly trigger them.",
        )
    ocr_cols = st.columns([1, 1, 1])
    ocr_cols[0].markdown('<div class="control-label">OCR scanned PDFs</div>', unsafe_allow_html=True)
    use_ocr = ocr_cols[0].toggle(
        "Enable OCR for scanned PDFs",
        value=True,
        label_visibility="collapsed",
        help="When text coverage is low, render pages in memory and run local Tesseract OCR. No OCR output or images are saved.",
    )
    ocr_max_pages = ocr_cols[1].number_input(
        "OCR page limit",
        min_value=1,
        max_value=300,
        value=60,
        step=5,
        help="Limits OCR work for very large PDFs. Increase if the full report is scanned.",
    )
    ocr_dpi = ocr_cols[2].select_slider(
        "OCR quality",
        options=[150, 200, 250, 300],
        value=200,
        help="Higher DPI can improve OCR accuracy but takes longer.",
    )
    with st.expander("Advanced settings", expanded=review_mode == "Advanced Review"):
        cautious_note_agreement = st.checkbox(
            "Run cautious note-reference validation anyway",
            value=False,
            key="run_cautious_note_reference_validation_anyway",
            help=(
                "When note/table confidence is below 80%, detailed note-reference validation is normally skipped. "
                "Enable this to run a heading-based review-prompt check using detected note headings and line-extracted primary statement rows. "
                "Detailed amount agreement remains skipped until table confidence improves."
            ),
        )

st.markdown('<div class="section-label">Audit review modules</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="module-grid">
        <div class="module-card">
            <div class="module-index">Module 01</div>
            <div class="module-title">Totals and rounding</div>
            <div class="module-copy">Totals, subtotals, cross-footings, duplicate totals, and $000s / millions labels.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 02</div>
            <div class="module-title">Formatting</div>
            <div class="module-copy">Number styles, brackets for negatives, currency markers, comparatives, and headings.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 03</div>
            <div class="module-title">Notes agreement</div>
            <div class="module-copy">Face statement references, segment totals, EPS, tax, depreciation, and note totals.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 04</div>
            <div class="module-title">Accounting policies</div>
            <div class="module-copy">Irrelevant policies, boilerplate wording, missing policies, and superseded standards.</div>
        </div>
        <div class="module-card">
            <div class="module-index">Module 05</div>
            <div class="module-title">Standards checklist</div>
            <div class="module-copy">Triggered IFRS disclosure checks for presentation and significant transaction areas.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload prepared financial statement PDF", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.info("Upload a PDF to start the review.")
    st.stop()

expected_policies = tuple(item.strip() for item in expected_policies_text.split(",") if item.strip())
significant_transactions = tuple(item.strip() for item in significant_transactions_text.split(",") if item.strip())
checklist_areas = tuple(item.strip() for item in checklist_areas_text.split(",") if item.strip())
normalised_currency = "" if reporting_currency == "Not specified" else normalize_reporting_currency(reporting_currency)
if reporting_currency != "Not specified" and not normalised_currency:
    st.warning("Enter a valid reporting currency before running the review. Example: NGN, USD, GBP, EUR, ZAR, GHS, or KES.")
    st.stop()
profile = CompanyProfile(
    company_name=company_name.strip(),
    industry=industry.strip(),
    reporting_currency=normalised_currency,
    expected_policies=expected_policies,
    significant_transactions=significant_transactions,
    presentation_standard=presentation_standard,
    checklist_areas=checklist_areas,
)

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    temp_file.write(uploaded.getbuffer())
    temp_path = Path(temp_file.name)

try:
    with st.spinner("Extracting PDF text, running OCR if needed, and performing review checks..."):
        result = review_pdf(
            temp_path,
            profile,
            ReviewOptions(
                use_ocr=use_ocr,
                ocr_max_pages=int(ocr_max_pages),
                ocr_dpi=int(ocr_dpi),
                run_cautious_note_agreement=cautious_note_agreement,
            ),
        )
finally:
    temp_path.unlink(missing_ok=True)

st.markdown('<div class="section-label">Review dashboard</div>', unsafe_allow_html=True)
review_cols = st.columns(5)
review_cols[0].metric("Checks performed", result.metrics.get("checks_performed_count", 0))
review_cols[1].metric("Checks passed", result.metrics.get("checks_passed_count", 0))
review_cols[2].metric("Checks skipped", result.metrics.get("checks_skipped_count", 0))
review_cols[3].metric("Findings", result.metrics["findings"])
review_cols[4].metric("Extraction confidence", result.metrics.get("extraction_confidence", "0%"))

risk_cols = st.columns(7)
risk_cols[0].metric("High", result.metrics["high"])
risk_cols[1].metric("Medium", result.metrics["medium"])
risk_cols[2].metric("Low", result.metrics["low"])
risk_cols[3].metric("Pages", result.metrics["pages"])
risk_cols[4].metric("Table confidence", result.metrics.get("table_confidence", "0%"))
risk_cols[5].metric("OCR pages", result.metrics.get("ocr_pages", 0))
risk_cols[6].metric("Tables", result.metrics["tables"])

detected_profile = result.metrics.get("detected_profile", {})
if isinstance(detected_profile, dict):
    st.markdown('<div class="section-label">Detected profile after upload</div>', unsafe_allow_html=True)
    profile_rows = [
        {"Field": key, "Detected value": value}
        for key, value in detected_profile.items()
    ]
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

st.markdown(
    f"""
    <section class="memo-panel">
        <div class="memo-kicker">Executive review memo</div>
        <div class="memo-text">{build_ai_review_memo(result)}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

download_cols = st.columns(2)
download_cols[0].download_button(
    "Download Excel Exception Register",
    _build_excel_export(result),
    file_name="audit_exception_register.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
download_cols[1].download_button(
    "Download Word Review Memo",
    _build_word_memo_export(result),
    file_name="audit_review_memo.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)

with st.expander("Notes detected (debug)", expanded=False):
    note_rows = _note_heading_rows(result)
    if note_rows:
        st.dataframe(pd.DataFrame(note_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No note headings detected.")

with st.expander("Note validation debug", expanded=False):
    st.json(
        {
            "cautious_note_validation_enabled": result.metrics.get("cautious_note_validation_enabled", False),
            "note_validation_mode": result.metrics.get("note_validation_mode", "skipped"),
            "note_reference_rows_detected": result.metrics.get("note_reference_rows_detected", 0),
            "note_headings_detected": result.metrics.get("note_headings_detected", 0),
            "note_reference_findings": result.metrics.get("note_reference_findings", 0),
        }
    )

with st.expander("Developer/debug Markdown export", expanded=False):
    markdown_report = findings_to_markdown(result)
    st.download_button(
        "Download Markdown Developer Report",
        markdown_report,
        file_name="financial_statement_review_debug.md",
        mime="text/markdown",
    )

status_cols = st.columns(2)
with status_cols[0]:
    with st.expander("Checks performed", expanded=True):
        assurance = str(result.metrics.get("positive_assurance", ""))
        if assurance:
            st.success(assurance)
        st.write(str(result.metrics.get("checks_performed", "No deterministic checks completed.")))
with status_cols[1]:
    with st.expander("Checks skipped", expanded=False):
        st.write(str(result.metrics.get("checks_skipped", "No major checks skipped.")))

if not result.findings:
    st.success("No issues were detected by the automated checks.")
    st.stop()

severity_order = ["High", "Medium", "Low"]
filter_cols = st.columns([1.4, 1])
category_filter = filter_cols[0].multiselect(
    "Category",
    sorted({finding.category for finding in result.findings}),
)
severity_filter = filter_cols[1].multiselect("Severity", severity_order, default=severity_order)

filtered = [
    finding
    for finding in result.findings
    if (not category_filter or finding.category in category_filter)
    and (not severity_filter or finding.severity in severity_filter)
]

rows = [
    {
        "Severity": finding.severity,
        "Category": finding.category,
        "Location": finding.location,
        "Issue": finding.issue,
        "Evidence": finding.evidence,
        "Recommendation": finding.recommendation,
    }
    for finding in filtered
]
st.markdown('<div class="section-label">Exception register</div>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

for finding in filtered:
    with st.expander(f"{finding.severity}: {finding.issue}", expanded=finding.severity == "High"):
        st.write(f"**Category:** {finding.category}")
        st.write(f"**Location:** {finding.location}")
        st.write(f"**Evidence:** {finding.evidence}")
        st.write(f"**Recommendation:** {finding.recommendation}")
