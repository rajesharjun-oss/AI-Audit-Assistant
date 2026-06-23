from __future__ import annotations

from io import BytesIO
import re

import pandas as pd
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo

from export_utils import clean_name_consistency_rows


def metric_lines(value: object, empty: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text == empty:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_excel_export(result) -> bytes:
    output = BytesIO()
    summary_rows = [
        {"Metric": "Checks performed", "Value": result.metrics.get("checks_performed_count", 0)},
        {"Metric": "Checks passed", "Value": result.metrics.get("checks_passed_count", 0)},
        {"Metric": "Checks skipped", "Value": result.metrics.get("checks_skipped_count", 0)},
        {"Metric": "Findings", "Value": result.metrics.get("findings", 0)},
        {"Metric": "High findings", "Value": result.metrics.get("high", 0)},
        {"Metric": "Medium findings", "Value": result.metrics.get("medium", 0)},
        {"Metric": "Low findings", "Value": result.metrics.get("low", 0)},
        {"Metric": "OCR text coverage", "Value": result.metrics.get("ocr_text_coverage", result.metrics.get("extraction_coverage", "0%"))},
        {"Metric": "OCR table candidates", "Value": result.metrics.get("ocr_tables", 0)},
        {"Metric": "Statement structure confidence", "Value": result.metrics.get("statement_structure_confidence", "0%")},
        {"Metric": "Note structure confidence", "Value": result.metrics.get("note_structure_confidence", "0%")},
        {"Metric": "Table arithmetic confidence", "Value": table_arithmetic_display(result)},
        {"Metric": "notes_section_start_page", "Value": result.metrics.get("notes_section_start_page", "Not detected")},
        {"Metric": "notes_heading_snippet", "Value": result.metrics.get("notes_heading_snippet", "No reliable notes heading detected.")},
        {"Metric": "cautious_note_validation_enabled", "Value": result.metrics.get("cautious_note_validation_enabled", False)},
        {"Metric": "note_validation_mode", "Value": result.metrics.get("note_validation_mode", "skipped")},
        {"Metric": "note_reference_rows_detected", "Value": result.metrics.get("note_reference_rows_detected", 0)},
        {"Metric": "note_headings_detected", "Value": result.metrics.get("note_headings_detected", 0)},
        {"Metric": "note_reference_findings", "Value": result.metrics.get("note_reference_findings", 0)},
        {"Metric": "AI policy review status", "Value": result.metrics.get("ai_policy_review_status", "disabled")},
        {"Metric": "AI policy review message", "Value": result.metrics.get("ai_policy_review_message", "")},
        {"Metric": "AI policy review summary", "Value": result.metrics.get("ai_policy_review_summary", "")},
        {"Metric": "AI finding review status", "Value": result.metrics.get("ai_finding_review_status", "disabled")},
        {"Metric": "AI finding review message", "Value": result.metrics.get("ai_finding_review_message", "")},
        {"Metric": "AI finding review summary", "Value": result.metrics.get("ai_finding_review_summary", "")},
        {"Metric": "AI finding review count", "Value": result.metrics.get("ai_finding_reviewed", 0)},
        {"Metric": "AI finding review suppressed", "Value": result.metrics.get("ai_finding_suppressed", 0)},
    ]
    checks_performed = [{"Check performed": item} for item in metric_lines(result.metrics.get("checks_performed"), "No deterministic checks completed.")]
    checks_skipped = checks_skipped_rows(result) or [{"Check area": "None", "Reason skipped": "No major checks skipped."}]
    check_results = result.metrics.get("check_results", [])
    if not isinstance(check_results, list) or not check_results:
        check_results = [{"Check": "No deterministic checks completed.", "Result": "Skipped", "Severity": "", "Evidence": ""}]
    detected_profile = result.metrics.get("detected_profile", {})
    profile_rows = [{"Field": key, "Detected value": value} for key, value in detected_profile.items()] if isinstance(detected_profile, dict) else []
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(profile_rows).to_excel(writer, sheet_name="Detected profile", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        finding_summary_rows = finding_summary_rows_for_result(result) or [{"Severity": "", "Category": "", "Page reference": "", "Issue": "No automated findings were identified.", "Recommendation": ""}]
        pd.DataFrame(finding_summary_rows).to_excel(writer, sheet_name="Findings summary", index=False)
        ai_status = str(result.metrics.get("ai_policy_review_status", "disabled") or "disabled")
        ai_message = str(result.metrics.get("ai_policy_review_message", "") or "").strip()
        ai_default_title = {
            "disabled": "AI policy review not enabled.",
            "unavailable": "AI policy review was enabled but is unavailable in this environment.",
            "skipped": "AI policy review was enabled but no suitable policy/disclosure context was detected.",
            "error": "AI policy review was enabled but failed during execution.",
            "completed": "AI policy review completed but returned no observation rows.",
        }.get(ai_status, "AI policy review returned no rows.")
        ai_policy_rows = result.metrics.get("ai_policy_export", []) or [{"Title": ai_default_title, "Status": ai_status, "Message": ai_message}]
        pd.DataFrame(ai_policy_rows).to_excel(writer, sheet_name="AI policy judgement", index=False)
        ai_finding_status = str(result.metrics.get("ai_finding_review_status", "disabled") or "disabled")
        ai_finding_message = str(result.metrics.get("ai_finding_review_message", "") or "").strip()
        ai_finding_default_title = {
            "disabled": "AI finding review not enabled.",
            "unavailable": "AI finding review was enabled but is unavailable in this environment.",
            "skipped": "AI finding review was enabled but no weak deterministic findings were eligible.",
            "error": "AI finding review was enabled but failed during execution.",
            "deferred": "AI finding review was deferred due to API availability or rate limiting.",
            "completed": "AI finding review completed but returned no adjudication rows.",
        }.get(ai_finding_status, "AI finding review returned no rows.")
        ai_finding_rows = result.metrics.get("ai_finding_export", []) or [{"Finding ID": "", "Issue": ai_finding_default_title, "AI status": ai_finding_status, "Reason": ai_finding_message}]
        pd.DataFrame(ai_finding_rows).to_excel(writer, sheet_name="AI finding review", index=False)
        exception_rows = finding_rows(result) or [
            {
                "ID": "",
                "Status": "Noted",
                "Severity": "",
                "Category": "",
                "Check type": "",
                "Confidence": "",
                "Page reference": "",
                "Note reference": "",
                "Statement": "",
                "Line item": "",
                "Referenced note": "",
                "Suggested note": "",
                "Match confidence": "",
                "Reason": "",
                "Current year amount found?": "",
                "Prior year amount found?": "",
                "Amount found in note": "",
                "Alternative note found": "",
                "Amount match confidence": "",
                "AI review status": "",
                "AI review confidence": "",
                "AI review reason": "",
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

        note_agreement_rows = note_agreement_result_rows(result)
        primary_rows = note_agreement_rows or [{"Statement": "No lines detected"}]
        no_notes_rows = [r for r in note_agreement_rows if r.get("Has note?") == "No"] or [{"Statement": "None found"}]
        linked_rows = [r for r in note_agreement_rows if r.get("Has note?") == "Yes"] or [{"Statement": "None found"}]
        pd.DataFrame(primary_rows).to_excel(writer, sheet_name="Primary statement line items", index=False)
        pd.DataFrame(no_notes_rows).to_excel(writer, sheet_name="Items without notes summary", index=False)
        pd.DataFrame(linked_rows).to_excel(writer, sheet_name="Note-linked review", index=False)

        notes_detected_rows = note_heading_rows(result) or [{"Note": "None found"}]
        notes_heading_candidate_rows = notes_heading_candidate_rows_for_result(result)
        ocr_statement_rows = ocr_statement_row_rows(result)
        pd.DataFrame(notes_detected_rows).to_excel(writer, sheet_name="Notes detected", index=False)
        pd.DataFrame(notes_heading_candidate_rows).to_excel(writer, sheet_name="Notes heading candidates", index=False)
        pd.DataFrame(ocr_statement_rows).to_excel(writer, sheet_name="OCR statement rows", index=False)

        policy_rows = result.metrics.get("policy_export", []) or [{"Paragraph reviewed": "None found"}]
        pd.DataFrame(policy_rows).to_excel(writer, sheet_name="Notes 1 and 2 policy review", index=False)
        unref_rows = result.metrics.get("unreferenced_notes", []) or [{"Note": "None", "Heading": "None found", "Comment": "All notes referenced or filtered"}]
        pd.DataFrame(unref_rows).to_excel(writer, sheet_name="Unreferenced notes", index=False)

        cross_export = result.metrics.get("cross_page_export", {})
        amount_rows = [
            translate_row_page_fields(row, result, ("Pages checked", "Context", "Issue"))
            for row in (cross_export.get("key_amounts", []) or [{"Metric": "None found"}])
        ]
        name_rows = [
            translate_row_page_fields(row, result, ("Page 1", "Page 2"))
            for row in (clean_name_consistency_rows(cross_export.get("names", [])) or [{"Name variant 1": "None found"}])
        ]
        date_rows = [
            translate_row_page_fields(row, result, ("Page", "Comment"))
            for row in (cross_export.get("dates", []) or [{"Date found": "None found"}])
        ]
        grammar_rows = [
            translate_row_page_fields(row, result, ("Page", "Context"))
            for row in (cross_export.get("grammar", []) or [{"Page": "None found"}])
        ]
        pd.DataFrame(amount_rows).to_excel(writer, sheet_name="Key amount consistency", index=False)
        pd.DataFrame(name_rows).to_excel(writer, sheet_name="Name consistency", index=False)
        pd.DataFrame(date_rows).to_excel(writer, sheet_name="Date consistency", index=False)
        pd.DataFrame(grammar_rows).to_excel(writer, sheet_name="Grammar review", index=False)

        pd.DataFrame(check_results).to_excel(writer, sheet_name="Checks results", index=False)
        pd.DataFrame(checks_performed).to_excel(writer, sheet_name="Checks performed", index=False)
        pd.DataFrame(checks_skipped).to_excel(writer, sheet_name="Checks skipped", index=False)
        skipped_summary_rows = result.metrics.get("skipped_table_summary", []) or [{"Skipped check group": "None", "Reason skipped": "No table-specific skips recorded."}]
        pd.DataFrame(skipped_summary_rows).to_excel(writer, sheet_name="Skipped checks summary", index=False)
        skipped_table_rows = result.metrics.get("skipped_table_details", []) or [{"Page": "None", "Reason skipped": "No table-specific skips recorded."}]
        pd.DataFrame(skipped_table_rows).to_excel(writer, sheet_name="Skipped table details", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F2937")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for column_cells in worksheet.columns:
                max_length = max((len(str(cell.value or "")) for cell in column_cells), default=10)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 70)

        format_exception_register_sheet(writer.book["Exception register"])
        format_excel_table_sheet(writer.book["Findings summary"], "FindingsSummary")
        format_excel_table_sheet(writer.book["Primary statement line items"], "PrimaryLineItems")
        format_excel_table_sheet(writer.book["Items without notes summary"], "NoNotesSummary")
        format_excel_table_sheet(writer.book["Note-linked review"], "NoteLinkedReview")
        format_excel_table_sheet(writer.book["Notes detected"], "NotesDetected")
        format_excel_table_sheet(writer.book["Notes heading candidates"], "NotesHeadingCandidates")
        format_excel_table_sheet(writer.book["OCR statement rows"], "OCRStatementRows")
        format_excel_table_sheet(writer.book["Notes 1 and 2 policy review"], "PolicyReview")
        format_excel_table_sheet(writer.book["AI policy judgement"], "AIPolicyJudgement")
        format_excel_table_sheet(writer.book["AI finding review"], "AIFindingReview")
        format_excel_table_sheet(writer.book["Unreferenced notes"], "UnreferencedNotes")
        format_excel_table_sheet(writer.book["Key amount consistency"], "AmountConsistency")
        format_excel_table_sheet(writer.book["Name consistency"], "NameConsistency")
        format_excel_table_sheet(writer.book["Date consistency"], "DateConsistency")
        format_excel_table_sheet(writer.book["Grammar review"], "GrammarReview")
        format_excel_table_sheet(writer.book["Checks results"], "ChecksResults")
        format_excel_table_sheet(writer.book["Skipped checks summary"], "SkippedChecksSummary")
        format_excel_table_sheet(writer.book["Skipped table details"], "SkippedTableDetails")
    return output.getvalue()


def checks_skipped_rows(result) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in metric_lines(result.metrics.get("checks_skipped"), "No major checks skipped."):
        row = parse_skipped_check(item)
        rows.append(translate_row_page_fields(row, result, ("Page reference", "Original message", "Reason skipped")))
    return rows


def finding_rows(result) -> list[dict[str, str]]:
    rows = []
    for index, finding in enumerate(result.findings, start=1):
        metadata = finding.metadata or {}
        row = {
            "ID": f"EX-{index:03d}",
            "Status": "Open",
            "Severity": finding.severity,
            "Category": finding.category,
            "Check type": finding.category,
            "Confidence": finding_confidence(finding, result),
            "Page reference": page_reference_for_finding(finding, result),
            "Note reference": note_reference_for_finding(finding, result),
            "Statement": metadata.get("statement", ""),
            "Line item": metadata.get("line_item", ""),
            "Referenced note": metadata.get("referenced_note", ""),
            "Suggested note": metadata.get("suggested_note", ""),
            "Match confidence": metadata.get("match_confidence", finding.severity),
            "Reason": metadata.get("reason", ""),
            "Current year amount found?": metadata.get("current_year_amount_found", ""),
            "Prior year amount found?": metadata.get("prior_year_amount_found", ""),
            "Amount found in note": metadata.get("amount_found_in_note", ""),
            "Alternative note found": metadata.get("alternative_note_found", ""),
            "Amount match confidence": metadata.get("amount_match_confidence", ""),
            "AI review status": metadata.get("ai_review_status", ""),
            "AI review confidence": metadata.get("ai_review_confidence", ""),
            "AI review reason": metadata.get("ai_review_reason", ""),
            "Location": translate_page_tokens(finding.location, result),
            "Issue": finding.issue,
            "Evidence": translate_page_tokens(finding.evidence, result),
            "Recommendation": finding.recommendation,
            "Reviewer comment": "",
            "Prepared by": "",
            "Reviewed by": "",
            "Date cleared": "",
        }
        rows.append(row)
    return rows


def finding_summary_rows_for_result(result) -> list[dict[str, str]]:
    return [
        {
            "Severity": finding.severity,
            "Category": finding.category,
            "Page reference": page_reference_for_finding(finding, result),
            "Note reference": note_reference_for_finding(finding, result),
            "Issue": finding.issue,
            "Recommendation": finding.recommendation,
        }
        for finding in result.findings
    ]


def table_arithmetic_display(result) -> str:
    skipped_details = result.metrics.get("skipped_table_details", [])
    if isinstance(skipped_details, list) and skipped_details:
        return "0% (Skipped)"
    return result.metrics.get("table_arithmetic_confidence", "0%")


def exported_file_stem(result) -> str:
    detected = result.metrics.get("detected_profile", {})
    company_name = ""
    if isinstance(detected, dict):
        company_name = str(detected.get("Company name", "") or "").strip()
    if not company_name:
        company_name = "financial_statement"
    stem = re.sub(r"[^A-Za-z0-9]+", "_", company_name).strip("_").lower()
    return stem[:90] or "financial_statement"


def note_heading_rows(result) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in metric_lines(result.metrics.get("note_headings"), "No note headings detected."):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 4:
            rows.append(
                {
                    "Note": parts[0].replace("Note ", "", 1).strip(),
                    "Page": parts[1].replace("Page ", "", 1).strip(),
                    "Page range": parts[2].strip(),
                    "Heading": parts[3].strip(),
                    "Confidence": parts[4].replace("Confidence:", "", 1).strip() if len(parts) >= 5 else "",
                    "Source snippet": parts[5].replace("Source:", "", 1).strip() if len(parts) >= 6 else "",
                }
            )
        elif len(parts) >= 3:
            rows.append(
                {
                    "Note": parts[0].replace("Note ", "", 1).strip(),
                    "Page": parts[1].replace("Page ", "", 1).strip(),
                    "Page range": parts[1].strip(),
                    "Heading": " | ".join(parts[2:]).strip(),
                    "Confidence": "",
                    "Source snippet": "",
                }
            )
    return rows


def notes_heading_candidate_rows_for_result(result) -> list[dict[str, str]]:
    rows = result.metrics.get("notes_heading_candidates", [])
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    return [{"Page": "", "Raw OCR snippet": "No possible notes heading candidates detected.", "Normalized snippet": "", "Similarity score": "", "Accepted": "No", "Reason": ""}]


def note_agreement_result_rows(result) -> list[dict[str, str]]:
    rows = result.metrics.get("note_agreement_results", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def ocr_statement_row_rows(result) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in metric_lines(result.metrics.get("ocr_statement_rows"), "No OCR primary statement rows detected."):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 10:
            rows.append(
                {
                    "Statement": parts[0],
                    "Page": parts[1].replace("Page ", "", 1).strip(),
                    "Line item": parts[2],
                    "Note detected": parts[3],
                    "Current year amount": parts[4],
                    "Prior year amount": parts[5],
                    "Raw OCR line": parts[6],
                    "Parse confidence": parts[7],
                    "Correction applied?": parts[8],
                    "Correction reason": parts[9],
                }
            )
    return rows or [{"Statement": "", "Page": "", "Line item": "No OCR primary statement rows detected.", "Note detected": "", "Current year amount": "", "Prior year amount": "", "Raw OCR line": "", "Parse confidence": "", "Correction applied?": "", "Correction reason": ""}]


def printed_page_map(result) -> dict[int, int]:
    raw_map = result.metrics.get("printed_page_map", {})
    if not isinstance(raw_map, dict):
        return {}
    mapped: dict[int, int] = {}
    for key, value in raw_map.items():
        try:
            mapped[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mapped


def reviewer_page_number(result, page_number: int) -> int:
    return printed_page_map(result).get(page_number, page_number)


def translate_page_tokens(text: object, result) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    def replace_page_phrase(match: re.Match) -> str:
        prefix = match.group(1)
        body = match.group(2)
        translated: list[str] = []
        for token in re.split(r"(\D+)", body):
            if token.isdigit():
                translated.append(str(reviewer_page_number(result, int(token))))
            else:
                translated.append(token)
        return prefix + "".join(translated)

    return re.sub(r"\b(Pages?\s+)([\d,\-\sand]+)", replace_page_phrase, value, flags=re.I)


def translate_row_page_fields(row: dict[str, object], result, fields: tuple[str, ...]) -> dict[str, object]:
    translated = dict(row)
    for field in fields:
        if field in translated:
            translated[field] = translate_page_tokens(translated.get(field, ""), result)
    return translated


def parse_skipped_check(item: str) -> dict[str, str]:
    check_area = item
    page_reference = ""
    reason = item
    can_fix = "Partially"
    reviewer_action = "Review the referenced page or supporting detail sheet, then rerun after improving extraction if needed."
    statement_match = re.match(r"(.+?):\s+Page\s+(\d+)\s+skipped because\s+(.+)", item, flags=re.I)
    if statement_match:
        check_area = statement_match.group(1).strip()
        page_reference = f"Page {statement_match.group(2)}"
        reason = statement_match.group(3).rstrip(".")
        reviewer_action = "Open the page and inspect the statement manually; automated casting is withheld because extraction did not produce reliable rows/columns."
    elif item.lower().startswith("generic table arithmetic skipped"):
        check_area = "Generic table arithmetic"
        page_reference = "See Skipped table details"
        reason = "Low-confidence or non-standard tables are listed separately."
        reviewer_action = "Use Skipped checks summary and Skipped table details to inspect the affected note tables manually."
    elif "notes section start was not detected" in item.lower():
        check_area = "OCR note-reference validation"
        reason = "Notes section start was not detected reliably."
        can_fix = "Yes, if OCR/notes heading extraction improves"
        reviewer_action = "Review Notes heading candidates and confirm the page where notes begin."
    elif "table extraction confidence is below threshold" in item.lower():
        check_area = "Detailed note agreement"
        reason = "Table extraction confidence is below the safe threshold."
        reviewer_action = "Use Note-linked review and Note agreement results for cautious review prompts; rerun detailed agreement when table extraction improves."
    elif "limited-scope statement extract" in item.lower():
        check_area = "Full AFS completeness and note agreement"
        reason = "The upload appears to be a limited-scope statement extract."
        can_fix = "Yes, with complete AFS upload"
        reviewer_action = "Upload the complete financial statements to run full checklist, policy, and note-agreement checks."
    elif "pdf extraction is unreliable" in item.lower():
        check_area = "Primary statement checks"
        reason = "PDF extraction is unreliable."
        reviewer_action = "Review extraction confidence, OCR statement rows, and source pages before relying on automated checks."
    return {
        "Check area": check_area,
        "Page reference": page_reference,
        "Reason skipped": reason,
        "Can automated check be fixed?": can_fix,
        "Reviewer action": reviewer_action,
        "Original message": item,
    }


def finding_confidence(finding, result) -> str:
    if finding.metadata and finding.metadata.get("ai_review_confidence"):
        return f"AI-reviewed / {finding.metadata['ai_review_confidence']}"
    if finding.metadata and finding.metadata.get("match_confidence"):
        return f"Review prompt / {finding.metadata['match_confidence']}"
    if finding.category == "Extraction quality":
        if finding.location in {"PDF extraction", "Table extraction", "Notes agreement"}:
            return (
                f"OCR text {result.metrics.get('ocr_text_coverage', result.metrics.get('extraction_coverage', '0%'))} / "
                f"Statement {result.metrics.get('statement_structure_confidence', '0%')} / "
                f"Table arithmetic {table_arithmetic_display(result)}"
            )
        return finding.severity
    if finding.category == "Totals and rounding":
        return f"Table arithmetic {table_arithmetic_display(result)}"
    return finding.severity


def page_reference(location: str, evidence: str = "", result=None) -> str:
    text = f"{location}\n{evidence}"
    pages = set()
    for match in re.finditer(r"\bpages?\s*:?\s+([0-9,\sand]+)", text, flags=re.I):
        for number in re.findall(r"\d+", match.group(1)):
            page_number = int(number)
            pages.add(reviewer_page_number(result, page_number) if result is not None else page_number)
    for match in re.findall(r"\bPage\s+(\d+)\b", text, flags=re.I):
        page_number = int(match)
        pages.add(reviewer_page_number(result, page_number) if result is not None else page_number)
    pages = sorted(pages)
    if not pages:
        return ""
    return f"Page {pages[0]}" if len(pages) == 1 else "Pages " + ", ".join(str(page) for page in pages)


def page_reference_for_finding(finding, result) -> str:
    direct = page_reference(finding.location, finding.evidence, result)
    metadata = finding.metadata or {}
    if finding.category == "Notes agreement":
        pages: set[int] = set(int(match) for match in re.findall(r"\bPage\s+(\d+)\b", direct or ""))
        note_pages = note_page_reference_map(result)
        for key in ("referenced_note", "suggested_note", "alternative_note_found"):
            note_ref = str(metadata.get(key, "") or "").upper().strip()
            if not note_ref:
                continue
            page_text = note_pages.get(note_ref) or note_pages.get(re.sub(r"[A-Z]+$", "", note_ref))
            if page_text:
                pages.update(int(match) for match in re.findall(r"\d+", page_text))
        if pages:
            ordered = sorted(pages)
            return f"Page {ordered[0]}" if len(ordered) == 1 else "Pages " + ", ".join(str(page) for page in ordered)
    if direct:
        return direct
    return str(finding.location or "").strip() or "Document-wide"


def note_reference_for_finding(finding, result) -> str:
    metadata = finding.metadata or {}
    note_candidates: list[str] = []
    for key in ("referenced_note", "suggested_note", "alternative_note_found", "amount_found_in_note"):
        value = str(metadata.get(key, "") or "").strip().upper()
        if value:
            note_candidates.append(value if value.startswith("NOTE ") else f"Note {value}")
    if note_candidates:
        return ", ".join(dict.fromkeys(note_candidates))
    text = "\n".join(str(part or "") for part in (finding.location, finding.issue, finding.evidence, metadata.get("reason", "")))
    matches = re.findall(r"\bNote\s+(\d+[A-Z]?)\b", text, flags=re.I)
    if matches:
        return ", ".join(f"Note {match.upper()}" for match in dict.fromkeys(matches))
    return ""


def note_page_reference_map(result) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in note_heading_rows(result):
        note = str(row.get("Note", "")).upper().strip()
        page_range = str(row.get("Page range", "") or row.get("Page", "")).strip()
        if note and page_range:
            mapping[note] = page_range
    for row in note_agreement_result_rows(result):
        note = str(row.get("Note number", "")).upper().strip()
        page_range = str(row.get("Note section page range", "")).strip()
        if note and page_range:
            mapping[note] = page_range
    return mapping


def format_excel_table_sheet(worksheet, table_name: str) -> None:
    max_row = max(worksheet.max_row, 2)
    max_col = worksheet.max_column
    table_ref = f"A1:{get_column_letter(max_col)}{max_row}"
    table = Table(displayName=table_name, ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    worksheet.add_table(table)


def format_exception_register_sheet(worksheet) -> None:
    max_row = max(worksheet.max_row, 2)
    max_col = worksheet.max_column
    table_ref = f"A1:{get_column_letter(max_col)}{max_row}"
    table = Table(displayName="ExceptionRegister", ref=table_ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
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
        color_map = {"High": ("FEE2E2", "991B1B"), "Medium": ("FEF3C7", "92400E"), "Low": ("DBEAFE", "1E40AF")}
        for severity, (fill, font) in color_map.items():
            worksheet.conditional_formatting.add(
                f"{severity_letter}2:{severity_letter}{max_row}",
                FormulaRule(formula=[f'${severity_letter}2="{severity}"'], fill=PatternFill("solid", fgColor=fill), font=Font(color=font, bold=True)),
            )
    for header in ("Evidence", "Recommendation", "Issue", "Reviewer comment"):
        column = headers.get(header)
        if column:
            letter = get_column_letter(column)
            worksheet.column_dimensions[letter].width = 55 if header in {"Evidence", "Recommendation", "Issue"} else 32
            for row in range(2, max_row + 1):
                worksheet[f"{letter}{row}"].alignment = Alignment(wrap_text=True, vertical="top")
    worksheet.auto_filter.ref = table_ref
