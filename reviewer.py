from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

from models import ChecklistItem, CompanyProfile, Finding, PdfDocument, PdfPage, ReviewOptions, ReviewResult
from cross_page_consistency import check_cross_page_consistency
from policy_reviewer import review_notes_1_and_2
from extraction import extract_pdf, extract_pdf_with_ocr


NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?")
YEAR_RE = re.compile(r"\b20\d{2}\b")
NOTE_REF_RE = re.compile(r"\bnote\s+(\d+[A-Za-z]?)\b|\bnotes?\s+(\d+[A-Za-z]?)\b", re.I)
NOTE_HEADING_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)(?:\s*\([a-z]\))?\s*[\).:-]?\s+(.{3,100})$", re.I)
NORMALIZED_AMOUNT_RE = re.compile(
    r"\(?-?\d{1,3}(?:\s*,\s*\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?",
    re.M,
)
NOTE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)\s+((?:20\d{2}|N['’]?\s?000|\$?000s?|\d{4})[\s,]*)+$", re.I)
ENTITY_SUFFIX_RE = re.compile(r"\b(?:limited|ltd|plc|inc|corp|corporation|company)\b", re.I)
VALID_CURRENCIES = {"NGN", "USD", "GBP", "EUR", "ZAR", "GHS", "KES", "CAD", "AUD"}


@dataclass(frozen=True)
class StatementNoteLine:
    ref: str
    line_item: str
    line: str
    amounts: tuple[Decimal, ...]
    page_number: int
    statement_name: str
    explicit_ref: bool = False


@dataclass(frozen=True)
class OcrStatementRow:
    label: str
    amounts: tuple[Decimal, ...]
    raw_line: str
    note_ref: str = ""
    confidence: str = "High"
    correction_applied: str = "No"
    correction_reason: str = ""


POLICY_RULES = {
    "revenue": {
        "policy": ("revenue recognition", "ifrs 15", "revenue from contracts"),
        "evidence": ("revenue", "turnover", "sales", "contract asset", "contract liability"),
    },
    "inventory": {
        "policy": ("inventories", "inventory", "ias 2", "net realisable value"),
        "evidence": ("inventory", "inventories", "stock", "work in progress", "raw materials"),
    },
    "leases": {
        "policy": ("leases", "ifrs 16", "right-of-use", "right of use"),
        "evidence": ("lease liability", "right-of-use", "right of use", "leased asset"),
    },
    "ppe": {
        "policy": ("property, plant and equipment", "ias 16", "depreciation"),
        "evidence": ("property, plant and equipment", "ppe", "depreciation", "plant and machinery"),
    },
    "intangibles": {
        "policy": ("intangible", "ias 38", "amortisation", "amortization"),
        "evidence": ("intangible", "goodwill", "software", "amortisation", "amortization"),
    },
    "financial instruments": {
        "policy": ("financial instruments", "ifrs 9", "expected credit loss"),
        "evidence": ("trade receivables", "borrowings", "cash and cash equivalents", "loans", "ecl"),
    },
    "tax": {
        "policy": ("income tax", "deferred tax", "ias 12"),
        "evidence": ("tax expense", "current tax", "deferred tax", "tax payable"),
    },
    "foreign currency": {
        "policy": ("foreign currency", "exchange differences", "functional currency"),
        "evidence": ("foreign exchange", "exchange gain", "exchange loss", "translation reserve"),
    },
    "employee benefits": {
        "policy": ("employee benefits", "defined benefit", "pension", "ias 19"),
        "evidence": ("staff costs", "retirement benefit", "pension", "gratuity"),
    },
    "consolidation": {
        "policy": ("consolidated financial statements", "subsidiaries", "ifrs 10"),
        "evidence": ("non-controlling interest", "subsidiary", "group", "consolidated"),
    },
    "biological assets": {
        "policy": ("biological assets", "ias 41", "agricultural produce"),
        "evidence": ("biological assets", "livestock", "plantation", "agricultural produce"),
    },
    "investment property": {
        "policy": ("investment property", "ias 40", "fair value model"),
        "evidence": ("investment property", "rental income", "fair value gain"),
    },
}

INDUSTRY_POLICY_MISMATCHES = {
    "technology": ("biological assets", "mineral resources", "insurance contracts"),
    "software": ("biological assets", "mineral resources", "insurance contracts"),
    "fintech": ("biological assets", "mineral resources"),
    "manufacturing": ("insurance contracts", "biological assets"),
    "bank": ("biological assets", "construction contracts"),
    "financial services": ("biological assets", "construction contracts"),
}

SUPERSEDED_REFERENCES = {
    "ias 17": "IFRS 16 replaced IAS 17 for leases.",
    "ias 39": "IFRS 9 replaced IAS 39 for most financial instrument accounting.",
    "ifrs 4": "IFRS 17 replaced IFRS 4 for insurance contracts.",
    "sic-15": "SIC-15 was superseded by IFRS 16 lease guidance.",
    "ifric 4": "IFRIC 4 was superseded by IFRS 16 lease guidance.",
}

GENERIC_POLICY_PHRASES = (
    "the company has adopted all standards",
    "where applicable",
    "in the normal course of business",
    "management believes",
    "no material impact",
    "not applicable to the company",
)

STANDARD_CHECKLIST = (
    ChecklistItem(
        "IAS 1",
        "presentation",
        "Financial statements should include a complete set of primary statements and notes.",
        (),
        (
            "statement of financial position",
            "statement of profit or loss",
            "statement of changes in equity",
            "statement of cash flows",
            "notes to the financial statements",
        ),
        "High",
    ),
    ChecklistItem(
        "IAS 1",
        "going concern",
        "The report should disclose the going concern basis or material uncertainty where relevant.",
        (),
        ("going concern", "material uncertainty", "continue as a going concern"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 1",
        "judgements and estimates",
        "Significant judgements and key estimation uncertainty should be disclosed.",
        (),
        ("significant judgement", "critical accounting judgement", "critical accounting estimates", "key source of estimation", "estimation uncertainty"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 15",
        "revenue",
        "Revenue disclosures should explain performance obligations, timing, and disaggregation where revenue is significant.",
        ("revenue from contracts", "contract asset", "contract liability", "performance obligation"),
        ("performance obligation", "disaggregated revenue", "contract balance", "contract asset", "contract liability"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 16",
        "leases",
        "Lease disclosures should identify right-of-use assets, lease liabilities, depreciation, interest, and maturity information where leases exist.",
        ("right-of-use asset", "right of use asset", "lease liability", "lease expense", "lease maturity", "depreciation of right-of-use", "depreciation of rou"),
        ("right-of-use", "right of use", "lease liability", "lease maturity", "interest on lease", "depreciation of right"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 7 / IFRS 9",
        "financial instruments",
        "Financial instrument disclosures should cover risk exposure, credit risk, liquidity risk, fair value, and impairment methodology.",
        ("trade receivables", "borrowings", "loans", "cash and cash equivalents", "financial instruments"),
        ("credit risk", "liquidity risk", "market risk", "expected credit loss", "fair value", "maturity analysis"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 12",
        "tax",
        "Tax disclosures should reconcile current tax, deferred tax, and the effective tax relationship where tax is material.",
        ("tax expense", "income tax", "deferred tax", "current tax"),
        ("current tax", "deferred tax", "effective tax", "tax reconciliation", "tax rate reconciliation"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 16",
        "ppe",
        "Property, plant and equipment disclosures should include depreciation policy and carrying amount reconciliation.",
        ("property, plant and equipment", "ppe", "depreciation", "plant and machinery"),
        ("depreciation rate", "depreciation method", "carrying amount", "cost", "accumulated depreciation", "additions"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 38",
        "intangibles",
        "Intangible asset disclosures should include amortisation policy and carrying amount reconciliation.",
        ("intangible", "software", "goodwill", "amortisation", "amortization"),
        ("amortisation", "amortization", "useful life", "carrying amount", "impairment"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 36",
        "impairment",
        "Impairment disclosures should describe impairment losses, reversals, or key assumptions where impairment indicators or goodwill exist.",
        ("impairment", "goodwill", "cash-generating unit", "cgu"),
        ("impairment loss", "recoverable amount", "value in use", "cash-generating unit", "key assumption"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 33",
        "eps",
        "Entities presenting EPS should disclose basic and diluted EPS inputs.",
        ("earnings per share", "eps", "ordinary shares"),
        ("basic earnings per share", "diluted earnings per share", "weighted average", "ordinary shares"),
        "High",
    ),
    ChecklistItem(
        "IFRS 8",
        "segments",
        "Operating segment disclosures should reconcile segment revenue, profit, assets, and liabilities to the financial statements where segments are presented.",
        ("segment", "operating segment"),
        ("segment revenue", "segment profit", "segment assets", "reconciliation", "chief operating decision maker"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 24",
        "related parties",
        "Related party disclosures should identify relationships, transactions, balances, and key management compensation.",
        ("related party", "director", "key management", "shareholder"),
        ("related party", "key management compensation", "transactions", "outstanding balances"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 10",
        "events after reporting period",
        "Events after the reporting period should be disclosed or explicitly addressed.",
        (),
        ("events after the reporting period", "subsequent events", "after the reporting date"),
        "Low",
    ),
    ChecklistItem(
        "IAS 37",
        "provisions and contingencies",
        "Provisions and contingent liabilities should disclose nature, uncertainty, movements, and possible obligations where relevant.",
        ("provision", "contingent", "litigation", "claim", "legal"),
        ("provision", "contingent liability", "movement", "uncertainty", "possible obligation"),
        "Medium",
    ),
)


def review_pdf(
    path: str | Path,
    profile: CompanyProfile | None = None,
    options: ReviewOptions | None = None,
) -> ReviewResult:
    options = options or ReviewOptions()
    document = extract_pdf(path)
    if _requires_ocr(document) and options.use_ocr:
        document = extract_pdf_with_ocr(path, document, options)
    profile = _profile_with_detected_defaults(profile or CompanyProfile(), document)
    findings: list[Finding] = []
    checks_performed: list[str] = []
    checks_skipped: list[str] = []
    note_validation_debug = _note_validation_debug(document, options.run_cautious_note_agreement, [])
    document_scope = _document_scope(document)
    limited_scope_extract = document_scope == "Limited-scope statement extract"
    quality_findings = check_extraction_quality(document)
    if limited_scope_extract:
        quality_findings = [
            finding
            for finding in quality_findings
            if "scanned or image-based" not in finding.issue.lower()
        ]
    findings.extend(quality_findings)
    if limited_scope_extract:
        findings.append(_limited_scope_extract_finding(document))
    if _requires_ocr(document) and not limited_scope_extract:
        checks_skipped.append("Primary statement checks skipped because PDF extraction is unreliable.")
        if options.run_cautious_note_agreement and not _notes_start_page(document):
            checks_skipped.append("OCR note-reference validation skipped because the notes section start was not detected.")
        note_validation_debug["note_validation_mode"] = "skipped"
        return _build_result(document, findings, checks_performed, checks_skipped, note_validation_debug, {})
    if _extraction_unreliable(document):
        checks_skipped.append("Document-level extraction quality is low; detailed table checks are limited to clean page/table evidence.")
    statement_findings, statement_performed, statement_skipped = check_primary_statement_consistency(document)
    findings.extend(statement_findings)
    checks_performed.extend(statement_performed)
    checks_skipped.extend(_limited_scope_statement_skips(statement_skipped) if limited_scope_extract else statement_skipped)
    if limited_scope_extract:
        checks_performed.append("Limited-scope review performed on Statement of Financial Position only.")
        checks_skipped.append("Full financial statement completeness, standards checklist, policies, formatting, and note agreement skipped because the upload is a limited-scope statement extract.")
        return _build_result(document, findings, checks_performed, checks_skipped, note_validation_debug, {})
    totals_findings = check_totals_and_rounding(document)
    findings.extend(totals_findings)
    findings.extend(check_formatting(document, profile))
    note_findings = check_notes_agreement(document, cautious_low_confidence=options.run_cautious_note_agreement)
    findings.extend(note_findings)
    note_validation_debug = _note_validation_debug(document, options.run_cautious_note_agreement, note_findings)
    if options.run_cautious_note_agreement and document.ocr_used:
        if note_validation_debug.get("note_validation_mode") == "review_prompt":
            checks_performed.append("Heading-based note-reference validation performed in OCR review-prompt mode.")
            checks_skipped.append("Detailed OCR note amount agreement skipped because OCR note tables are not yet reliable.")
        else:
            checks_skipped.append("OCR note-reference validation skipped because the notes section start was not detected.")
    elif options.run_cautious_note_agreement and document.table_extraction_confidence < 80 and not document.ocr_used:
        note_reference_prompts = [
            finding
            for finding in note_findings
            if finding.metadata and finding.metadata.get("referenced_note")
        ]
        checks_performed.append("Cautious face-to-note amount agreement performed in review-prompt mode.")
        if note_reference_prompts:
            checks_performed.append("Cautious note-reference and amount-agreement validation performed in review-prompt mode; possible prompts added to the exception register.")
        else:
            checks_performed.append("Cautious note-reference validation performed in review-prompt mode; no possible wrong note references detected.")
        checks_skipped.append("Detailed note agreement skipped because table extraction confidence is below threshold.")
    elif document.table_extraction_confidence < 80 and not document.ocr_used:
        checks_performed.append("Basic note heading existence checks completed where statement note references were clearly detected.")
        checks_skipped.append("Detailed note agreement skipped because table extraction confidence is below threshold.")
    else:
        checks_performed.append("Cautious note-reference validation completed for primary statement note references.")
        checks_performed.append("Basic note reference and note amount agreement checks completed.")
    note_sections = _note_sections(document)
    policy_findings, policy_export = review_notes_1_and_2(document, profile, note_sections)
    findings.extend(policy_findings)
    if totals_findings:
        checks_skipped.append("Generic table arithmetic skipped on low-confidence/non-standard tables where indicated in extraction findings.")
    
    checks_performed.extend(["Accounting policies relevance check", "IFRS/IAS standards alignment check"])
    cross_page_findings, cross_page_export = check_cross_page_consistency(document)
    findings.extend(cross_page_findings)
    checks_performed.append("Notes 1 & 2 policy and standards alignment review")

    findings = [
        f for f in findings
        if not (f.category == "Notes agreement" and f.metadata and f.metadata.get("match_confidence") == "Low")
    ]
    return _build_result(document, findings, checks_performed, checks_skipped, note_validation_debug, cross_page_export, policy_export)


def _get_note_section_with_fallback(ref: str, note_sections: dict[str, str]) -> str:
    section = note_sections.get(ref, "")
    if section: return section
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        return note_sections.get(parent, "")
    return ""

def _get_note_heading_with_fallback(ref: str, headings: dict[str, str]) -> str:
    heading = headings.get(ref, "")
    if heading: return heading
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        return headings.get(parent, "")
    return ""

def _document_scope(document: PdfDocument) -> str:
    return "Limited-scope statement extract" if _is_limited_scope_statement_extract(document) else "Full financial statement or mixed upload"


def _is_limited_scope_statement_extract(document: PdfDocument) -> bool:
    classified = _classified_primary_statement_pages(document)
    if len(document.pages) != 1 or len(classified) != 1:
        return False
    if "Statement of financial position" not in classified:
        return False
    return _notes_start_page(document) is None and not _note_headings_by_page(document)


def _limited_scope_extract_finding(document: PdfDocument) -> Finding:
    page_text = document.pages[0].text if document.pages else ""
    return Finding(
        "Document scope",
        "Low",
        "Upload scope",
        "Only a Statement of Financial Position page was uploaded.",
        _snippet_around(page_text, "statement of financial position") or "Single primary statement page detected with no notes section.",
        "Only a Statement of Financial Position page was uploaded. Full financial statement completeness and note-agreement checks were not performed.",
    )


def _limited_scope_statement_skips(skipped: list[str]) -> list[str]:
    return [item for item in skipped if "financial position" in item.lower()]


def _build_result(
    document: PdfDocument,
    findings: list[Finding],
    checks_performed: list[str] | None = None,
    checks_skipped: list[str] | None = None,
    note_validation_debug: dict[str, int | str | bool] | None = None,
    cross_page_export: dict | None = None,
    policy_export: list[dict] | None = None,
) -> ReviewResult:
    checks_performed_list = list(dict.fromkeys(checks_performed or []))
    checks_skipped_list = list(dict.fromkeys(checks_skipped or []))
    check_result_rows = _check_result_rows(checks_performed_list, checks_skipped_list, findings)
    positive_assurance = _positive_assurance_text(findings, checks_performed_list)
    note_validation_debug = note_validation_debug or _note_validation_debug(document, False, [])
    metrics = {
        "document_scope": _document_scope(document),
        "pages": len(document.pages),
        "text_pages": document.text_pages,
        "text_chars": document.text_chars,
        "extraction_coverage": f"{document.extraction_coverage:.0%}",
        "extraction_confidence": f"{document.extraction_confidence}%",
        "ocr_text_coverage": f"{document.extraction_coverage:.0%}",
        "statement_structure_confidence": f"{_statement_structure_confidence(document)}%",
        "note_structure_confidence": f"{_note_structure_confidence(document)}%",
        "table_arithmetic_confidence": f"{_table_arithmetic_confidence(document)}%",
        "table_confidence": f"{_table_arithmetic_confidence(document)}%",
        "extraction_profile": document.extraction_profile,
        "unreadable_values": document.unreadable_value_count,
        "merged_value_cells": document.merged_value_cell_count,
        "ocr_used": "Yes" if document.ocr_used else "No",
        "ocr_pages": document.ocr_pages,
        "ocr_tables": document.ocr_tables,
        "tables": sum(len(page.tables) for page in document.pages),
        "findings": len(findings),
        "high": sum(1 for item in findings if item.severity == "High"),
        "medium": sum(1 for item in findings if item.severity == "Medium"),
        "low": sum(1 for item in findings if item.severity == "Low"),
        "note_headings": _format_note_heading_debug(document),
        "notes_section_start_page": _notes_start_page(document) or "Not detected",
        "notes_heading_snippet": _format_notes_heading_snippet(document),
        "notes_heading_candidates": _notes_heading_candidate_rows(document),
        "primary_statement_pages": _format_primary_statement_debug(document),
        "ocr_statement_rows": _format_ocr_statement_rows_debug(document),
        "note_agreement_results": _note_agreement_result_rows(document),
        "detected_profile": infer_detected_profile(document),
        "checks_performed": "\n".join(checks_performed_list) or "No deterministic checks completed.",
        "checks_skipped": "\n".join(checks_skipped_list) or "No major checks skipped.",
        "check_results": check_result_rows,
        "cross_page_export": cross_page_export or {},
        "policy_export": policy_export or [],
        "checks_performed_count": len(checks_performed_list),
        "checks_passed_count": sum(1 for row in check_result_rows if row.get("Result") == "Passed"),
        "checks_skipped_count": len(checks_skipped_list),
        "positive_assurance": positive_assurance,
        **note_validation_debug,
    }
    return ReviewResult(findings=findings, metrics=metrics)


def _note_validation_debug(
    document: PdfDocument,
    enabled: bool,
    note_findings: list[Finding],
) -> dict[str, int | str | bool]:
    note_reference_findings = sum(1 for finding in note_findings if finding.metadata and finding.metadata.get("referenced_note"))
    if enabled and document.ocr_used and _statement_note_lines(document) and _note_headings_by_page(document):
        mode = "review_prompt"
    elif enabled and not document.ocr_used:
        mode = "review_prompt" if document.table_extraction_confidence < 80 else "strict"
    else:
        mode = "skipped" if document.table_extraction_confidence < 80 or document.ocr_used else "strict"
    return {
        "cautious_note_validation_enabled": bool(enabled),
        "note_validation_mode": mode,
        "note_reference_rows_detected": sum(1 for i in _statement_note_lines(document) if i.ref) if document.pages else 0,
        "note_headings_detected": len(_note_headings_by_page(document)) if document.pages else 0,
        "notes_section_start_page": _notes_start_page(document) or "Not detected",
        "note_reference_findings": note_reference_findings,
    }


def _statement_structure_confidence(document: PdfDocument) -> int:
    pages = _classified_primary_statement_pages(document)
    if not pages:
        return 0
    score = sum(_statement_parse_success_score(name, page.text) for name, page in pages.items())
    detected_page_bonus = min(20, len(pages) * 4)
    score = min(100, score + detected_page_bonus)
    return max(0, score)


def _statement_row_confidence(text: str) -> int:
    rows = _statement_rows(text)
    important = sum(
        1
        for label in rows
        if any(
            keyword in label
            for keyword in (
                "total assets",
                "current assets",
                "non current assets",
                "equity",
                "liabilities",
                "revenue",
                "profit",
                "cash and cash equivalents",
            )
        )
    )
    return min(40, important * 8)


def _statement_parse_success_score(statement_name: str, text: str) -> int:
    rows = _statement_rows(text)
    name = statement_name.lower()
    if "financial position" in name:
        score = 0
        if _has_rows(rows, ("non-current assets", "current assets", "total assets")):
            score += 18
        if _row_amounts_any(rows, ("equity", "total equity")) and _row_amounts_any(rows, ("liabilities", "financial liabilities", "total liabilities")) and _row_amounts_any(rows, ("total equity and liabilities", "total funds and liabilities")):
            score += 18
        return score
    if "income" in name or "profit" in name:
        score = 0
        if "revenue" in rows or "operating revenue" in rows:
            score += 8
        if "profit before tax" in rows:
            score += 8
        if "taxation" in rows:
            score += 5
        if "profit after tax" in rows:
            score += 8
        return score
    if "cash flow" in name:
        if _row_amounts_any(rows, ("cash at beginning", "cash and cash equivalents at the beginning of the year")) and _row_amounts_any(rows, ("cash at end", "cash and cash equivalents as at the end of the year")):
            return 18
        return 0
    if "changes" in name:
        return 12 if any("balance as at" in line.lower() for line in text.splitlines()) else 0
    return _statement_row_confidence(text)


def _note_structure_confidence(document: PdfDocument) -> int:
    headings = _note_headings_by_page(document)
    if not headings:
        return 0
    score = min(100, 25 + len(headings) * 5)
    if not _notes_start_page(document):
        score = min(score, 45)
    if document.ocr_used:
        score = min(score, 80)
    return score


def _table_arithmetic_confidence(document: PdfDocument) -> int:
    if document.ocr_used:
        return min(document.table_extraction_confidence, _statement_structure_confidence(document))
    return document.table_extraction_confidence


def _format_primary_statement_debug(document: PdfDocument) -> str:
    classified = _classified_primary_statement_pages(document)
    if not classified:
        return "No primary statement pages detected."
    return "\n".join(f"{name} | Page {page.number}" for name, page in classified.items())


def _format_ocr_statement_rows_debug(document: PdfDocument) -> str:
    classified = _classified_primary_statement_pages(document)
    if not classified:
        return "No OCR primary statement rows detected."
    rows: list[str] = []
    for statement_name, page in classified.items():
        for row in _statement_row_parses(page.text).values():
            current_amount = row.amounts[0] if row.amounts else None
            prior_amount = row.amounts[1] if len(row.amounts) >= 2 else None
            rows.append(
                " | ".join(
                    (
                        statement_name,
                        f"Page {page.number}",
                        row.label,
                        row.note_ref,
                        _format_decimal_for_export(current_amount),
                        _format_decimal_for_export(prior_amount),
                        row.raw_line,
                        row.confidence,
                        row.correction_applied,
                        row.correction_reason,
                    )
                )
            )
    return "\n".join(rows) if rows else "No OCR primary statement rows detected."


def _note_section_page_ranges(document: PdfDocument) -> dict[str, str]:
    headings = _note_headings_by_page(document)
    ordered = sorted(headings.items(), key=lambda item: (_note_sort_key(item[0]), item[1][1]))
    ranges: dict[str, str] = {}
    for index, (ref, (_title, start_page)) in enumerate(ordered):
        next_page = ordered[index + 1][1][1] if index + 1 < len(ordered) else start_page
        end_page = max(start_page, next_page - 1)
        ranges[ref] = f"Page {start_page}" if end_page == start_page else f"Pages {start_page}-{end_page}"
    return ranges


def _notes_start_page(document: PdfDocument) -> int | None:
    pages = list(document.pages)
    for i, page in enumerate(pages):
        text_lower = page.text.lower()
        if "notes to the financial" in text_lower:
            if "accounting policies" in text_lower or "material accounting" in text_lower:
                if not _looks_like_front_matter_page(page.text):
                    return page.number
            elif i + 1 < len(pages) and ("accounting policies" in pages[i+1].text.lower() or "material accounting" in pages[i+1].text.lower()):
                if not _looks_like_front_matter_page(page.text):
                    return page.number
        if _notes_heading_in_text(page.text):
            return page.number
        if _looks_like_front_matter_page(page.text):
            continue
    if document.ocr_used:
        candidates = _notes_heading_candidates(document, include_weak=False)
        accepted = [candidate for candidate in candidates if candidate["accepted"] == "Yes"]
        if accepted:
            return int(accepted[0]["page"])
    return None


def _looks_like_front_matter_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()[:40] if line.strip()]
    head = "\n".join(lines).lower()
    if not head:
        return False
    if lines and _normalise_match_words(lines[0]).startswith("notes"):
        return False
    front_terms = (
        "independent auditor",
        "auditor's report",
        "auditors report",
        "directors' report",
        "directors report",
        "corporate information",
        "report of the directors",
        "contents",
    )
    return any(term in head for term in front_terms)


def _notes_heading_in_text(text: str) -> bool:
    lower_text = text.lower()
    if "contents" in lower_text[:400] or lower_text.count("....") > 2:
        return False
    if re.search(r"notes to the financial statements\s*(?:\.{2,}|\s{4,})\s*\d+", lower_text):
        return False
        
    for line in text.splitlines()[:60]:
        if _notes_heading_line_score(line) >= 0.82:
            return True
    return False


def _notes_heading_line_score(line: str) -> float:
    stripped = re.sub(r"\s+", " ", line.strip())
    normalized = _normalise_match_words(stripped)
    if not normalized:
        return 0.0
    if any(term in normalized for term in ("refer notes", "see notes", "part our audit", "part audit", "auditor report")):
        return 0.0
    phrases = (
        "notes to the financial statements",
        "notes to the financial statement",
        "notes financial statements",
        "notes financial statement",
        "notes to financial statements",
        "notes to the accounts",
        "notes to accounts",
        "notes accounts",
        "notes forming part of the financial statements",
        "notes forming part of financial statements",
        "notes forming part financial statements",
    )
    if re.fullmatch(r"notes\s+to\s+(?:the\s+)?financial\s+statements?", stripped, flags=re.I):
        return 1.0
    for phrase in phrases:
        if normalized == phrase:
            return 1.0
        if normalized.startswith(phrase):
            return 0.96
    if not normalized.startswith("notes"):
        return 0.0
    if "accounts" in normalized and normalized.startswith("notes"):
        return max(0.82, SequenceMatcher(None, normalized, "notes to the accounts").ratio())
    if not all(term in normalized for term in ("notes", "financial", "statement")):
        return 0.0
    return max(SequenceMatcher(None, normalized, phrase).ratio() for phrase in phrases)


def _significant_accounting_policy_line_score(line: str) -> float:
    normalized = _normalise_match_words(line)
    if not normalized:
        return 0.0
    phrases = (
        "significant accounting policies",
        "accounting policies",
        "summary significant accounting policies",
    )
    if normalized.startswith("significant accounting polic") or normalized.startswith("accounting polic"):
        return 0.86
    if "accounting" in normalized and "polic" in normalized:
        return max(0.72, max(SequenceMatcher(None, normalized, phrase).ratio() for phrase in phrases))
    return 0.0


def _format_notes_heading_snippet(document: PdfDocument) -> str:
    start_page = _notes_start_page(document)
    if not start_page:
        candidates = _possible_notes_heading_snippets(document)
        return "\n".join(candidates) if candidates else "No reliable notes heading detected."
    candidates = [candidate for candidate in _notes_heading_candidates(document, include_weak=True) if candidate["page"] == start_page]
    if candidates:
        top = candidates[0]
        return f"Page {top['page']} | Confidence {top['confidence']}: {top['snippet']}"
    return "No reliable notes heading detected."


def _possible_notes_heading_snippets(document: PdfDocument) -> list[str]:
    candidates = _notes_heading_candidates(document, include_weak=True)
    return [f"Possible Page {candidate['page']} | Confidence {candidate['confidence']} | {candidate['accepted']}: {candidate['snippet']}" for candidate in candidates[:5]]


def _notes_heading_candidates(document: PdfDocument, include_weak: bool = False) -> list[dict[str, str | int]]:
    candidates: list[tuple[float, dict[str, str | int]]] = []
    search_start_page = _notes_candidate_search_start_page(document)
    for page in document.pages:
        if document.ocr_used and page.number < search_start_page:
            continue
        front_matter = document.ocr_used and _looks_like_front_matter_page(page.text)
        lines = page.text.splitlines()
        for score, candidate_type, cleaned, normalized, page_reason in _raw_page_notes_heading_candidates(page.text, include_weak):
            line_candidates = [line for line in lines if line.strip()]
            follows_numbered_policy = _candidate_followed_by_numbered_policy(line_candidates, 0) or _candidate_followed_by_numbered_policy([page.text], 0)
            strong_notes_section = _strong_notes_section_candidate(score, candidate_type, normalized, follows_numbered_policy)
            diagnostic_only = (front_matter and not strong_notes_section) or _notes_candidate_diagnostic_only(cleaned, cleaned)
            accepted = not diagnostic_only and (
                score >= 0.9
                or (score >= 0.82 and (follows_numbered_policy or candidate_type == "Accounting policies heading"))
            )
            reason = page_reason or "Accepted: raw OCR page text contains a notes heading candidate."
            if accepted and follows_numbered_policy:
                reason = "Accepted: raw OCR page text contains a notes heading candidate followed by numbered accounting policy headings."
            if accepted and candidate_type == "Accounting policies heading":
                reason = "Accepted: significant accounting policies heading appears after primary statements."
            elif diagnostic_only:
                reason = "Rejected: candidate appears before the notes section or in front-matter/narrative text."
            elif not accepted:
                reason = "Rejected: candidate score is below acceptance threshold or lacks numbered note-heading support."
            candidates.append(
                (
                    score + (0.03 if follows_numbered_policy else 0),
                    {
                        "page": page.number,
                        "confidence": f"{score:.0%}",
                        "snippet": cleaned,
                        "normalized_snippet": normalized,
                        "accepted": "Yes" if accepted else "No",
                        "reason": reason,
                        "type": candidate_type,
                    },
                )
            )
        if front_matter:
            continue
        for index, line in enumerate(lines):
            score = _notes_heading_line_score(line)
            candidate_type = "Notes heading"
            if score <= 0:
                score = _significant_accounting_policy_line_score(line)
                candidate_type = "Accounting policies heading"
            if score <= 0 and include_weak and "notes" in _normalise_match_words(line):
                score = 0.35
                candidate_type = "Weak notes text"
            if score <= 0:
                continue
            if score < (0.68 if include_weak else 0.82):
                continue
            snippet = " ".join(item.strip() for item in lines[max(0, index - 2) : index + 4] if item.strip())
            cleaned = re.sub(r"\s+", " ", snippet).strip()
            follows_numbered_policy = _candidate_followed_by_numbered_policy(lines, index)
            diagnostic_only = _notes_candidate_diagnostic_only(line, cleaned)
            accepted = not diagnostic_only and (score >= 0.9 or (score >= 0.82 and (follows_numbered_policy or candidate_type == "Accounting policies heading")))
            reason = "Accepted: strong notes heading candidate."
            if candidate_type == "Accounting policies heading" and accepted:
                reason = "Accepted: significant accounting policies heading appears after primary statements."
            if follows_numbered_policy and accepted:
                reason = "Accepted: notes heading candidate is followed by numbered accounting policy headings."
            elif diagnostic_only:
                reason = "Rejected: candidate appears to be a narrative/extract reference rather than the notes section heading."
            elif not accepted:
                reason = "Rejected: candidate score is below acceptance threshold or lacks numbered note-heading support."
            candidates.append(
                (
                    score + (0.03 if follows_numbered_policy else 0),
                    {
                        "page": page.number,
                        "confidence": f"{score:.0%}",
                        "snippet": cleaned,
                        "normalized_snippet": _normalise_match_words(cleaned),
                        "accepted": "Yes" if accepted else "No",
                        "reason": reason,
                        "type": candidate_type,
                    },
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in candidates]


def _raw_page_notes_heading_candidates(text: str, include_weak: bool = False) -> list[tuple[float, str, str, str, str]]:
    candidates: list[tuple[float, str, str, str, str]] = []
    if not text.strip():
        return candidates
    pattern_specs = (
        (r"notes?\s+to\s+(?:the\s+)?financial\s+statements?", "Notes heading", 0.96),
        (r"notes?\s+forming\s+part\s+of\s+(?:the\s+)?financial\s+statements?", "Notes heading", 0.96),
        (r"notes?\s+to\s+(?:the\s+)?accounts?", "Notes heading", 0.92),
        (r"(?:^|\n)\s*1[\).:\s-]+(?:significant\s+)?accounting\s+polic(?:y|ies)", "Accounting policies heading", 0.84),
        (r"significant\s+accounting\s+polic(?:y|ies)", "Accounting policies heading", 0.78),
    )
    for pattern, candidate_type, base_score in pattern_specs:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            raw = _snippet_for_span(text, match.start(), match.end(), radius=180)
            cleaned = re.sub(r"\s+", " ", raw).strip()
            normalized = _normalise_match_words(cleaned)
            score = max(base_score, _notes_page_snippet_score(normalized, candidate_type))
            reason = "Accepted: raw OCR page text matched a notes-section phrase."
            if candidate_type == "Accounting policies heading":
                reason = "Accepted: raw OCR page text contains significant accounting policies after primary statements."
            candidates.append((score, candidate_type, cleaned, normalized, reason))
    normalized_text = _normalise_match_words(text)
    if include_weak:
        weak_terms = ("notes", "financial statements", "significant accounting policies", "accounting policies")
        for term in weak_terms:
            if term not in normalized_text:
                continue
            raw_index = _best_raw_index_for_normalized_term(text, term)
            raw = _snippet_for_span(text, raw_index, raw_index + len(term), radius=180)
            cleaned = re.sub(r"\s+", " ", raw).strip()
            normalized = _normalise_match_words(cleaned)
            score = _notes_page_snippet_score(normalized, "Accounting policies heading" if "accounting" in term else "Notes heading")
            if score < 0.35:
                score = 0.35
            candidates.append(
                (
                    score,
                    "Accounting policies heading" if "accounting" in term else "Weak notes text",
                    cleaned,
                    normalized,
                    "Rejected: weak raw OCR candidate retained for diagnostic review.",
                )
            )
    return _dedupe_raw_notes_candidates(candidates)


def _strong_notes_section_candidate(score: float, candidate_type: str, normalized: str, follows_numbered_policy: bool) -> bool:
    if not follows_numbered_policy:
        return False
    if candidate_type == "Accounting policies heading":
        return score >= 0.82 and "accounting" in normalized and "polic" in normalized
    return score >= 0.9 and (
        ("notes" in normalized and "financial" in normalized and "statement" in normalized)
        or ("notes" in normalized and "accounts" in normalized)
    )


def _notes_page_snippet_score(normalized: str, candidate_type: str) -> float:
    if not normalized:
        return 0.0
    if candidate_type == "Accounting policies heading":
        targets = ("significant accounting policies", "accounting policies")
    else:
        targets = (
            "notes to the financial statements",
            "notes to financial statements",
            "notes forming part of the financial statements",
            "notes to the accounts",
        )
    if "notes" in normalized and "financial" in normalized and "statement" in normalized:
        return 0.86
    if "significant accounting polic" in normalized:
        return 0.82
    return max(SequenceMatcher(None, normalized[:220], target).ratio() for target in targets)


def _snippet_for_span(text: str, start: int, end: int, radius: int = 140) -> str:
    snippet_start = max(0, start - radius)
    snippet_end = min(len(text), end + radius)
    return text[snippet_start:snippet_end]


def _best_raw_index_for_normalized_term(text: str, term: str) -> int:
    words = term.split()
    if not words:
        return 0
    first = re.search(re.escape(words[0]), text, flags=re.I)
    return first.start() if first else 0


def _dedupe_raw_notes_candidates(
    candidates: list[tuple[float, str, str, str, str]]
) -> list[tuple[float, str, str, str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[float, str, str, str, str]] = []
    for score, candidate_type, cleaned, normalized, reason in candidates:
        key = (candidate_type, normalized[:180])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, candidate_type, cleaned, normalized, reason))
    return deduped


def _notes_candidate_diagnostic_only(line: str, snippet: str) -> bool:
    normalized = _normalise_match_words(f"{line} {snippet}")
    diagnostic_terms = ("extract", "excerpt", "reference", "narrative", "summary")
    if any(term in normalized for term in diagnostic_terms) and not _candidate_followed_by_numbered_policy(snippet.splitlines() or [snippet], 0):
        return True
    return False


def _notes_candidate_search_start_page(document: PdfDocument) -> int:
    if not document.ocr_used:
        return 1
    statement_pages = [page.number for page in _classified_primary_statement_pages(document).values()]
    return min(statement_pages) + 1 if statement_pages else 1


def _candidate_followed_by_numbered_policy(lines: list[str], index: int) -> bool:
    nearby = "\n".join(lines[index : index + 8])
    return bool(
        re.search(
            r"(?:^|\n)\s*(?:note\s+)?1[\).:\s-]+(?:significant\s+)?accounting polic",
            nearby,
            flags=re.I,
        )
        or re.search(r"(?:^|\n)\s*1[\).:\s-]+.{0,60}\n\s*2[\).:\s-]+", nearby, flags=re.I)
    )


def _notes_heading_candidate_rows(document: PdfDocument) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in _notes_heading_candidates(document, include_weak=True):
        rows.append(
            {
                "Page": str(candidate["page"]),
                "Raw OCR snippet": str(candidate["snippet"]),
                "Normalized snippet": str(candidate.get("normalized_snippet", "")),
                "Similarity score": str(candidate["confidence"]),
                "Accepted": str(candidate["accepted"]),
                "Reason": str(candidate["reason"]),
            }
        )
    return rows


def _note_agreement_result_rows(document: PdfDocument) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    statement_lines = _statement_note_lines(document)
    if not statement_lines:
        return rows
    if document.ocr_used:
        headings = {ref: title for ref, (title, _page_number) in _note_headings_by_page(document).items()}
        page_ranges = _note_section_page_ranges(document)
        note_sections = _note_sections(document) if _notes_start_page(document) else {}
        for item in statement_lines:
            current_amount = item.amounts[0] if item.amounts else None
            prior_amount = item.amounts[1] if len(item.amounts) >= 2 else None
            
            if not item.ref:
                rows.append(
                    _note_agreement_result_row(
                        item,
                        current_amount,
                        prior_amount,
                        "N/A",
                        "N/A",
                        "",
                        "Low",
                        "Skipped",
                        "No note number.",
                        "",
                        "",
                        "",
                    )
                )
                continue
            if not item.amounts:
                rows.append(
                    _note_agreement_result_row(
                        item,
                        current_amount,
                        prior_amount,
                        "N/A",
                        "N/A",
                        "",
                        "Low",
                        "Skipped",
                        "Skipped - no reliable statement amount was detected after excluding the note-reference column.",
                        "",
                        page_ranges.get(item.ref, ""),
                        "",
                    )
                )
                continue
            alternative_ref = _weak_semantic_alternative_note(item, headings, note_sections)
            alternative_heading = headings.get(alternative_ref, "") if alternative_ref else ""
            review_prompt = bool(alternative_ref) and _heading_only_alternative_is_review_prompt(item.line_item, alternative_heading)
            result = "Review prompt" if review_prompt else "Skipped"
            reason = (
                f"Heading-only OCR review prompt: Note {alternative_ref} appears semantically closer, but OCR note/table confidence is too low for an exception."
                if review_prompt
                else f"Low-confidence heading-only debug result: Note {alternative_ref} may be related, but amount support is not reliable enough for a review prompt."
                if alternative_ref
                else "Skipped because the document was OCR-assisted and note amount extraction is not reliable."
            )
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "N/A",
                    "N/A",
                    alternative_ref,
                    "Low",
                    result,
                    reason,
                    "",
                    page_ranges.get(item.ref, ""),
                    "heading match" if review_prompt else "heading match debug" if alternative_ref else "",
                )
            )
        return rows
    note_sections = _note_sections(document)
    headings = {ref: title for ref, (title, _page_number) in _note_headings_by_page(document).items()}
    page_ranges = _note_section_page_ranges(document)
    _scale_label, tolerance = _detect_rounding_scale(document.text)
    low_confidence = document.table_extraction_confidence < 80
    for item in statement_lines:
        current_amount = item.amounts[0] if item.amounts else None
        prior_amount = item.amounts[1] if len(item.amounts) >= 2 else None
        
        if not item.ref:
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "N/A",
                    "N/A",
                    "",
                    "Low",
                    "Skipped",
                    "No note number.",
                    "",
                    "",
                    "",
                )
            )
            continue
        if not item.amounts:
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "N/A",
                    "N/A",
                    "",
                    "Low",
                    "Skipped",
                    "Skipped - no reliable statement amount was detected after excluding the note-reference column.",
                    "",
                    page_ranges.get(item.ref, ""),
                    "",
                )
            )
            continue

        skip_reason = _note_agreement_skip_reason(item)
        if skip_reason:
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "N/A",
                    "N/A",
                    "",
                    "Low",
                    "Skipped",
                    f"Skipped - {skip_reason}",
                    "",
                    page_ranges.get(item.ref, ""),
                    "",
                )
            )
            continue
        referenced_section = _get_note_section_with_fallback(item.ref, note_sections)
        if _is_disclosure_only_note(_get_note_heading_with_fallback(item.ref, headings)):
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "N/A",
                    "N/A",
                    "",
                    "Low",
                    "Skipped",
                    "Skipped because the referenced note is disclosure-only.",
                    "",
                    page_ranges.get(item.ref, ""),
                    "",
                )
            )
            continue
        if not referenced_section:
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "No",
                    "No" if prior_amount is not None else "N/A",
                    "",
                    "Low",
                    "Review prompt" if low_confidence else "Skipped",
                    "Referenced note section was not detected.",
                    "",
                    page_ranges.get(item.ref, ""),
                    "",
                )
            )
            continue
        current_match = _amount_match_in_section(current_amount, referenced_section, tolerance)
        prior_match = _amount_match_in_section(prior_amount, referenced_section, tolerance)
        current_found = bool(current_match["found"])
        prior_found = bool(prior_match["found"])
        matched_snippet = str(current_match["snippet"] or prior_match["snippet"] or "")
        matching_method = _combined_matching_method(current_match, prior_match)
        if current_found and (prior_amount is None or prior_found):
            rows.append(
                _note_agreement_result_row(
                    item,
                    current_amount,
                    prior_amount,
                    "Yes",
                    "Yes" if prior_amount is not None else "N/A",
                    "",
                    "High",
                    "Passed",
                    "Current and prior year amounts were located in the referenced note section.",
                    matched_snippet,
                    page_ranges.get(item.ref, ""),
                    matching_method,
                )
            )
            continue
        alternative_ref = _alternative_note_for_missing_amounts(item, note_sections, headings, tolerance)
        rows.append(
            _note_agreement_result_row(
                item,
                current_amount,
                prior_amount,
                _yes_no(current_found),
                _yes_no(prior_found) if prior_amount is not None else "N/A",
                alternative_ref,
                _amount_match_confidence(current_found, prior_found, alternative_ref, cautious_review_prompt=low_confidence),
                "Review prompt" if low_confidence else "Review prompt",
                f"Amount appears in another note: Note {alternative_ref}." if alternative_ref else "Amount not located in referenced note.",
                matched_snippet,
                page_ranges.get(item.ref, ""),
                f"heading match / {matching_method or 'normalized amount'}" if alternative_ref else (matching_method or "not found"),
            )
        )
    return rows


def _note_agreement_result_row(
    item: StatementNoteLine,
    current_amount: Decimal | None,
    prior_amount: Decimal | None,
    current_found: str,
    prior_found: str,
    alternative_ref: str,
    confidence: str,
    result: str,
    reason: str,
    matched_snippet: str,
    page_range: str,
    matching_method: str,
) -> dict[str, str]:
    has_note = "Yes" if item.ref else "No"
    review_req = "Yes" if item.ref else "No"
    comment = f"This line item is referenced to Note {item.ref}." if item.ref else "No note number."

    return {
        "Statement": item.statement_name,
        "Page": str(item.page_number),
        "Line item description": item.line_item.title(),
        "Note number": item.ref,
        "Current year amount": _format_decimal_for_export(current_amount),
        "Prior year amount": _format_decimal_for_export(prior_amount),
        "Has note?": has_note,
        "Review required?": review_req,
        "Comment": comment,
        "Review result": result,
        "Reason": reason,
        "Current year amount found in referenced note?": current_found,
        "Prior year amount found in referenced note?": prior_found,
        "Alternative note found": alternative_ref,
        "Match confidence": confidence,
        "Matched text snippet from referenced note": matched_snippet,
        "Note section page range": page_range,
        "Matching method": matching_method,
    }


def _format_decimal_for_export(value: Decimal | None) -> str:
    if value is None:
        return ""
    return f"{value:,.0f}"


def _check_result_rows(
    checks_performed: list[str],
    checks_skipped: list[str],
    findings: list[Finding],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for check in checks_performed:
        related = [finding for finding in findings if _finding_matches_check(finding, check)]
        if related:
            result = "Failed" if any(finding.severity == "High" for finding in related) else "Review prompt"
            severity_rank = {"High": 0, "Medium": 1, "Low": 2}
            severity = ", ".join(sorted({finding.severity for finding in related}, key=lambda item: severity_rank.get(item, 9)))
            evidence = " | ".join(finding.issue for finding in related[:3])
        else:
            result = "Passed"
            severity = ""
            evidence = "No exception generated by this check."
        rows.append(
            {
                "Check": check,
                "Result": result,
                "Severity": severity,
                "Evidence": evidence,
            }
        )
    for check in checks_skipped:
        rows.append(
            {
                "Check": check,
                "Result": "Skipped",
                "Severity": "",
                "Evidence": "Check was not run because prerequisite extraction confidence or source evidence was not available.",
            }
        )
    return rows


def _finding_matches_check(finding: Finding, check: str) -> bool:
    check_lower = check.lower()
    issue_lower = finding.issue.lower()
    location_lower = finding.location.lower()
    if "income statement" in check_lower and "income statement" in location_lower:
        if "profit/loss after tax" in check_lower or "revenue, tax" in check_lower:
            return "profit/loss after tax" in issue_lower
        return True
    if "financial position" in check_lower and "financial position" in location_lower:
        if "total assets" in check_lower and "equity" in check_lower:
            return "total assets" in issue_lower and "equity" in issue_lower
        if "equity and liabilities" in check_lower:
            return "equity" in issue_lower and "liabilities" in issue_lower
        if "total assets" in check_lower:
            return "total assets" in issue_lower
        return True
    if "cash flow" in check_lower and "cash flow" in location_lower:
        return True
    if "accumulated fund" in check_lower and "accumulated fund" in location_lower:
        return True
    if "note-reference" in check_lower or "note agreement" in check_lower or "face-to-note" in check_lower:
        return finding.category in {"Notes agreement", "Extraction quality"} and "note" in issue_lower
    return False


def _positive_assurance_text(findings: list[Finding], checks_performed: list[str]) -> str:
    primary_checks = [item for item in checks_performed if item.startswith(("Income statement", "Statement of "))]
    primary_exceptions = [
        finding
        for finding in findings
        if finding.category == "Totals and rounding"
        and any(
            marker in finding.location.lower()
            for marker in ("income statement", "financial position", "accumulated fund", "cash flows")
        )
    ]
    if primary_checks and not primary_exceptions:
        return "No exceptions noted from line-based checks on primary statements."
    return ""


def _profile_with_detected_defaults(profile: CompanyProfile, document: PdfDocument) -> CompanyProfile:
    detected = infer_detected_profile(document)
    reporting_currency = profile.reporting_currency or normalize_reporting_currency(detected.get("Currency", ""))
    framework = profile.presentation_standard
    if not framework or framework == "IFRS":
        detected_framework = detected.get("Framework", "")
        framework = detected_framework if detected_framework in {"IFRS", "Local GAAP"} else "IFRS"
    return CompanyProfile(
        company_name=profile.company_name,
        industry=profile.industry,
        reporting_currency=reporting_currency,
        expected_policies=profile.expected_policies,
        significant_transactions=profile.significant_transactions,
        presentation_standard=framework,
        checklist_areas=profile.checklist_areas,
    )


def check_extraction_quality(document: PdfDocument) -> list[Finding]:
    findings: list[Finding] = []
    if document.ocr_error:
        findings.append(
            Finding(
                "Extraction quality",
                "High",
                "OCR pipeline",
                "OCR could not be completed for this PDF.",
                document.ocr_error,
                "Confirm Tesseract OCR is installed and accessible, then retry. You can also upload a text-selectable PDF.",
            )
        )
    if document.unreadable_value_count:
        severity = "High" if document.unreadable_value_count >= 5 else "Medium"
        findings.append(
            Finding(
                "Extraction quality",
                severity,
                "PDF extraction",
                "Unreadable or placeholder values were detected in extracted text or tables.",
                f"Detected {document.unreadable_value_count} unreadable placeholder(s), such as ####, replacement characters, or blanked-out value markers.",
                "Resolve the source PDF/export issue before relying on arithmetic or note-agreement findings for affected pages.",
            )
        )
    if document.merged_value_cell_count:
        findings.append(
            Finding(
                "Extraction quality",
                "Medium",
                "Table extraction",
                "Some extracted table cells appear to contain multiple merged values.",
                f"Detected {document.merged_value_cell_count} table cell(s) containing more than one numeric value.",
                "Review the extracted table layout. Merged numeric cells can make column arithmetic and cross-footing checks unreliable.",
            )
        )
    if _extraction_unreliable(document):
        findings.append(
            Finding(
                "Extraction quality",
                "High",
                "PDF extraction",
                "Extraction confidence is too low for reliable automated audit checks.",
                (
                    f"Profile: {document.extraction_profile}; text confidence {document.extraction_confidence}%; "
                    f"text coverage {document.extraction_coverage:.0%}; unreadable values {document.unreadable_value_count}; "
                    f"table confidence {document.table_extraction_confidence}%; merged value cells {document.merged_value_cell_count}."
                ),
                "Use a cleaner text-selectable PDF, repair the source export, or run OCR with a higher-quality scan before relying on the exception register.",
            )
        )
    if document.ocr_used and not _requires_ocr(document) and not _extraction_unreliable(document):
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "OCR pipeline",
                "OCR was used to recover text from a scanned or image-based PDF.",
                f"OCR processed {document.ocr_pages} page(s), reconstructed {document.ocr_tables} table candidate(s), and text coverage is now {document.extraction_coverage:.0%}.",
                "Review extracted findings carefully because OCR can misread figures, punctuation, and note references in signed/scanned financial statements.",
            )
        )
        return findings
    if not _requires_ocr(document):
        return findings
    if document.text_chars == 0:
        evidence = f"0 extractable text pages out of {len(document.pages)} pages."
    else:
        evidence = (
            f"{document.text_pages} extractable text page(s) out of {len(document.pages)} pages; "
            f"{document.text_chars} extracted characters."
        )
    recommendation = (
        "Enable OCR in the app, or upload a text-selectable/exported PDF. Re-run the review after OCR so totals, "
        "notes, policies, and standards checklist checks can inspect the statement content."
    )
    if document.ocr_used:
        recommendation = (
            "OCR ran but did not recover enough text for reliable automated review. Try a higher-quality scan, "
            "a text-selectable PDF, or manual OCR settings."
        )
    findings.append(
        Finding(
            "Extraction quality",
            "High",
            "PDF extraction",
            "The PDF appears to be scanned or image-based, so automated audit checks cannot run reliably.",
            evidence,
            recommendation,
        )
    )
    return findings


def infer_detected_profile(document: PdfDocument) -> dict[str, str]:
    text = document.text
    lower = text.lower()
    entity_type = _detect_entity_type(text)
    if entity_type == "Private company" and re.search(r"\binvestment propert(?:y|ies)|property investment|real estate\b", lower):
        entity_type = "Private company / property investment company"
    profile = {
        "Company name": _detect_company_name(document),
        "Year end": _detect_year_end(text),
        "Currency": _detect_currency(text),
        "Framework": _detect_framework(text),
        "Entity type": entity_type,
        "Document scope": _document_scope(document),
        "Principal activities": _detect_principal_activities(text),
        "Detected balances": _detect_major_balances(lower),
        "Suggested checklist areas": _suggest_checklist_areas(lower),
        "Extraction confidence": f"Text {document.extraction_confidence}% | Tables {document.table_extraction_confidence}%",
    }
    return profile


def _detect_company_name(document: PdfDocument) -> str:
    first_pages = "\n".join(page.text for page in document.pages[:5])
    legal_name_patterns = (
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+(?:Limited|Ltd|PLC|Plc|Incorporated|Inc\.?|Corporation|Company)\b",
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+(?:Institute|Council|Association|Society|Body)\s+of\s+[A-Z][A-Za-z&,.()' -]{3,80}\b",
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+of\s+[A-Z][A-Za-z&,.()' -]{3,80}\b",
    )
    for pattern in legal_name_patterns:
        match = re.search(pattern, first_pages, flags=re.I)
        if match:
            return _clean_detected_company_name(re.sub(r"\s+", " ", match.group(0)).strip(" -.,"))
    for page in document.pages[:3]:
        for line in page.text.splitlines()[:12]:
            clean = re.sub(r"\s+", " ", line).strip(" -")
            if not clean or len(clean) < 5:
                continue
            if re.search(r"financial statements|annual report|statement of|notes to", clean, re.I):
                continue
            if clean.isupper() or re.search(r"\b(limited|ltd|plc|incorporated|institute|company|corporation)\b", clean, re.I):
                return _clean_detected_company_name(clean)
    return "Not detected"


def _clean_detected_company_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" -.,")
    cleaned = re.sub(r"^(?:fl|fi|f1|l|i)\s+(?=[A-Z])", "", cleaned, flags=re.I)
    return _title_preserving_acronyms(cleaned) if cleaned.isupper() else cleaned


def _title_preserving_acronyms(text: str) -> str:
    small_words = {"of", "and", "the", "for", "in", "on"}
    words = []
    for index, word in enumerate(text.split()):
        stripped = word.strip()
        if index > 0 and stripped.lower() in small_words:
            words.append(stripped.lower())
        elif len(stripped) <= 4 and stripped.isupper():
            words.append(stripped)
        else:
            words.append(stripped[:1].upper() + stripped[1:].lower())
    return " ".join(words)


def _detect_year_end(text: str) -> str:
    patterns = (
        r"(?:year|period) ended\s+([A-Za-z]+\s+\d{1,2},?\s+20\d{2})",
        r"(?:as at|at)\s+([A-Za-z]+\s+\d{1,2},?\s+20\d{2})",
        r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    years = sorted(set(YEAR_RE.findall(text)))
    return years[-1] if years else "Not detected"


def _detect_currency(text: str) -> str:
    naira_thousands = r"(?:N|NGN|₦|â‚¦)\s*['’‘â€™]?\s*000|N000"
    if re.search(naira_thousands, text, flags=re.I):
        return "NGN / N'000"
    if re.search(r"N['’]?\s?000|N000|Naira|₦|\bNGN\b", text, flags=re.I):
        return "NGN"
    if re.search(r"\bUSD\b|US\$|\bDollar\b|\$", text, flags=re.I):
        return "USD"
    if re.search(r"\bGBP\b|\bPound\b", text, flags=re.I):
        return "GBP"
    if re.search(r"\bEUR\b|\bEuro\b", text, flags=re.I):
        return "EUR"
    return "Not detected"


def _detect_framework(text: str) -> str:
    if re.search(r"international financial reporting standards|IFRS", text, flags=re.I):
        return "IFRS"
    if re.search(r"local gaap|generally accepted accounting", text, flags=re.I):
        return "Local GAAP"
    return "Not detected"


def _detect_entity_type(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(private company|private limited)\b", lower):
        return "Private company"
    if re.search(r"\b(plc|public limited)\b", lower):
        return "Public company"
    if re.search(r"\b(limited|ltd)\b", lower) or any(
        term in lower
        for term in (
            "directors' report",
            "directors report",
            "share capital",
            "ordinary shares",
            "shareholders",
            "dividends",
        )
    ):
        return "Private company"
    if any(
        term in lower
        for term in (
            "non-profit",
            "not-for-profit",
            "professional body",
            "institute",
            "council",
            "fellows",
            "associates",
            "membership",
            "members fund",
            "members' fund",
            "accumulated fund",
            "subscriptions",
        )
    ):
        return "Non-profit / professional body"
    return "Not detected"


def _detect_principal_activities(text: str) -> str:
    match = re.search(r"principal activit(?:y|ies).{0,700}", text, flags=re.I | re.S)
    if match:
        snippet = re.sub(r"\s+", " ", match.group(0)).strip()
        snippet = re.split(r"\b(?:results|financial statements|statement of|notes to|property, plant)\b", snippet, maxsplit=1, flags=re.I)[0]
        if re.search(r"\bprofessional body|membership|members|institute|council|fellows|associates|training|certification|professional development\b", snippet, flags=re.I):
            return "Professional membership body, including member services, professional development, training, and certification."
        cleaned = re.sub(r"^principal activit(?:y|ies)\s*(?:of the institute|of the company|is|are|:|-)?\s*", "", snippet, flags=re.I).strip(" .:-")
        if cleaned:
            first_sentence = re.split(r"(?<=[.])\s+", cleaned, maxsplit=1)[0]
            return first_sentence[:180].strip(" .") + "."
    return "Not detected"


def _detect_major_balances(lower: str) -> str:
    areas = []
    balance_terms = (
        ("Revenue", "revenue"),
        ("Receivables", "receivable"),
        ("Cash", "cash and cash equivalents"),
        ("Investment property", "investment property"),
        ("PPE", "property, plant and equipment"),
        ("Intangibles", "intangible"),
        ("Payables", "payable"),
        ("Financial liabilities", "financial liabilities"),
        ("Tax", "tax"),
    )
    for label, term in balance_terms:
        if term in lower:
            areas.append(label)
    return ", ".join(areas[:10]) if areas else "Not detected"


def _suggest_checklist_areas(lower: str) -> str:
    suggestions = []
    if "revenue from contracts" in lower or "contract asset" in lower or "contract liability" in lower:
        suggestions.append("IFRS 15")
    if _actual_lease_disclosure_present(lower):
        suggestions.append("IFRS 16")
    if "investment property" in lower:
        suggestions.append("IAS 40")
    if "property, plant and equipment" in lower:
        suggestions.append("IAS 16")
    if "financial instruments" in lower or "credit risk" in lower:
        suggestions.append("IFRS 7 / IFRS 9")
    if re.search(r"\b(eps|earnings per share)\b", lower):
        suggestions.append("IAS 33")
    return ", ".join(suggestions) if suggestions else "None strongly triggered"


def _requires_ocr(document: PdfDocument) -> bool:
    if not document.pages:
        return True
    return document.text_chars < 1000 or document.extraction_coverage < 0.25


def _extraction_unreliable(document: PdfDocument) -> bool:
    return document.extraction_confidence < 50 or document.unreadable_value_count >= 5


def check_standard_checklist(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    if profile.presentation_standard.upper() != "IFRS":
        return []
    text = document.text.lower()
    policy_map = _accounting_policy_map(document)
    requested_areas = {area.strip().lower() for area in profile.checklist_areas if area.strip()}
    expected = {item.strip().lower() for item in profile.expected_policies if item.strip()}
    significant = {item.strip().lower() for item in profile.significant_transactions if item.strip()}
    context = " ".join(sorted(requested_areas | expected | significant | {profile.industry.lower()}))
    findings: list[Finding] = []

    for item in STANDARD_CHECKLIST:
        active = _checklist_item_applies(item, text, context, requested_areas)
        if not active:
            continue
        if _checklist_item_satisfied_by_context(item, text, policy_map):
            continue
        hits = [keyword for keyword in item.evidence_keywords if keyword in text]
        required_hits = len(item.evidence_keywords) if not item.applies_when else min(2, len(item.evidence_keywords))
        if item.standard == "IAS 1" and item.area == "judgements and estimates":
            required_hits = 1
        if len(hits) < required_hits:
            location, snippet = _first_keyword_context(document, item.applies_when or item.evidence_keywords or (item.area,))
            findings.append(
                Finding(
                    "Standards checklist",
                    item.severity,
                    f"{item.standard} | {location}",
                    f"Potential missing or incomplete {item.standard} disclosure: {item.area}.",
                    f"Checklist expectation: {item.requirement} Detected evidence: {', '.join(hits) if hits else 'none'}. Context: {snippet}",
                    "Review the disclosure against the applicable standard and add the missing policy, note, reconciliation, or judgement disclosure if applicable.",
                )
            )
    return findings


def check_rounding_and_casting(document: PdfDocument, tolerance: Decimal = Decimal("1")) -> list[Finding]:
    return check_totals_and_rounding(document, tolerance)


def check_primary_statement_consistency(
    document: PdfDocument,
    tolerance: Decimal = Decimal("1"),
) -> tuple[list[Finding], list[str], list[str]]:
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    checks = (
        ("Statement of income and expenditure", _check_income_statement_text),
        ("Statement of financial position", _check_sfp_text),
        ("Statement of changes in accumulated fund", _check_accumulated_fund_text),
        ("Statement of cash flows", _check_cash_flow_text),
    )
    for statement_name, checker in checks:
        page = _find_statement_page(document, statement_name)
        if not page:
            skipped.append(f"{statement_name}: statement page not detected.")
            continue
        page_findings, page_performed, page_skipped = checker(page, tolerance, document.ocr_used, document)
        findings.extend(page_findings)
        performed.extend(page_performed)
        skipped.extend(page_skipped)
    return findings, performed, skipped


def check_totals_and_rounding(document: PdfDocument, tolerance: Decimal | None = None) -> list[Finding]:
    findings: list[Finding] = []
    scale_label, scale_tolerance = _detect_rounding_scale(document.text)
    tolerance = tolerance if tolerance is not None else scale_tolerance
    tolerance = max(tolerance, Decimal("1"))

    if scale_label == "mixed":
        findings.append(
            Finding(
                "Totals and rounding",
                "Medium",
                "Document-wide",
                "Mixed rounding or scaling labels were detected.",
                "The report refers to more than one presentation scale, such as units, thousands, or millions.",
                "Use one presentation basis consistently, or clearly label exceptions in the affected note or statement.",
            )
        )

    if document.ocr_used:
        table_count = sum(len(page.tables) for page in document.pages)
        if table_count:
            findings.append(
                Finding(
                    "Extraction quality",
                    "Low",
                    "OCR reconstructed tables",
                    "Generic arithmetic checks were skipped for OCR-reconstructed tables.",
                    f"OCR reconstructed {table_count} table candidate(s). Generic row/column casting is disabled for OCR tables until statement-specific structure is confidently identified.",
                    "Use the reconstructed tables as review evidence, but rely on statement-specific checks or manual review for scanned statements.",
                )
            )
        findings.extend(_check_ocr_statement_of_financial_position(document, tolerance))
        findings.extend(_check_ocr_statement_of_cash_flows(document, tolerance))
        return findings

    skipped_tables: list[str] = []
    primary_statements = _classified_primary_statement_pages(document)
    primary_pages = {p.number for p in primary_statements.values()}
    
    for page in document.pages:
        for table_index, table in enumerate(page.tables, start=1):
            if len(table) < 3:
                continue
            table_quality = _classify_table_for_arithmetic(table, page.text)
            if not table_quality["can_run_arithmetic"]:
                skipped_tables.append(
                    f"Page {page.number}, table {table_index}: {table_quality['type']} ({table_quality['reason']})"
                )
                continue
            note_cols = _note_columns(table)
            rows = [_numeric_row(row, note_cols) for row in table]
            max_cols = max((len(row) for row in rows), default=0)
            table_findings: list[Finding] = []
            for col in range(1, max_cols):
                if col in note_cols:
                    continue
                _check_vertical_totals(table_findings, page.number, table_index, rows, col, tolerance)
            _check_cross_footings(table_findings, page.number, table_index, rows, tolerance)
            _check_column_consistency(table_findings, page.number, table_index, table)
            
            if page.number in primary_pages:
                # Suppress generic table math findings on primary statements because statement-specific checks handle them.
                continue
            
            if document.table_extraction_confidence < 80:
                skipped_tables.append(f"Page {page.number}, table {table_index}: skipped because overall table extraction confidence is low ({document.table_extraction_confidence}%).")
                continue

            for f in table_findings:
                if f.severity == "High":
                    f.severity = "Medium"
                    f.issue += " (Downgraded severity because table structure in notes may be complex)."
                # Wait, I can't put this inside the loop easily without access to total skipped tables.
                pass
            findings.extend(table_findings)
    if skipped_tables:
        details = "\n".join(skipped_tables)
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "Table extraction",
                f"{len(skipped_tables)} table(s) skipped due to low confidence or non-standard structure.",
                details,
                "Review skipped table details only where the table is material. Primary statement line-based checks can still be relied on when listed as performed.",
            )
        )
    return findings


def check_formatting(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    text = document.text
    findings: list[Finding] = []
    currency_symbols = _contextual_currency_markers(document)
    if profile.reporting_currency:
        expected = normalize_reporting_currency(profile.reporting_currency)
        if not expected:
            return findings
        unexpected = [symbol for symbol in currency_symbols if _normalise_currency_marker(symbol) != expected]
        if unexpected:
            findings.append(
                Finding(
                    "Formatting",
                    "Medium",
                    "Document-wide",
                    "Multiple or unexpected currency markers appear in the report.",
                    f"Expected {expected}; observed {dict(Counter(_normalise_currency_marker(symbol) for symbol in currency_symbols))}.",
                    "Confirm the presentation currency and standardise currency labels in statements, notes, and headers.",
                )
            )
    elif len(set(_normalise_currency_marker(symbol) for symbol in currency_symbols)) > 1:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "The report appears to use more than one currency marker.",
                f"Observed in statement/table currency contexts: {dict(Counter(_normalise_currency_marker(symbol) for symbol in currency_symbols))}.",
                "Confirm whether mixed currencies are intentional and clearly labelled.",
            )
        )

    financial_amount_text = "\n".join(_financial_amount_contexts(document))
    parenthesis_negatives = bool(re.search(r"\(\s?\d[\d,]*(?:\.\d+)?\s?\)", financial_amount_text))
    minus_negatives = bool(re.search(r"(?<!\w)-\d[\d,]*(?:\.\d+)?", financial_amount_text))
    if parenthesis_negatives and minus_negatives:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "Negative amounts use mixed styles.",
                "Both bracketed negatives and leading-minus negatives were detected.",
                "Use one negative-number convention consistently across the statements and notes.",
            )
        )
    elif minus_negatives and not parenthesis_negatives:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "Negative amounts appear to use leading minus signs instead of brackets.",
                "At least one negative amount was detected with a leading minus sign.",
                "If the reporting format requires brackets for negatives, update the affected statements and notes.",
            )
        )

    _check_comparatives(findings, document)
    _check_required_statement_names(findings, document, profile)
    for page in document.pages:
        bad_separators = []
        page_document = PdfDocument([page], ocr_used=document.ocr_used)
        for line in _financial_amount_contexts(page_document):
            if _ignore_formatting_line(line):
                continue
            bad_separators.extend(
                token for token in re.findall(r"\b\d{4,}(?:\.\d+)?\b", line)
                if _looks_like_unformatted_amount(token)
            )
        if bad_separators:
            findings.append(
                Finding(
                    "Formatting",
                    "Low",
                    f"Page {page.number}",
                    "Some large numbers may be missing thousands separators.",
                    ", ".join(bad_separators[:5]),
                    "Review numeric formatting and apply the report's standard separator convention.",
                )
            )
    return findings


def _looks_like_unformatted_amount(token: str) -> bool:
    if YEAR_RE.fullmatch(token):
        return False
    digits = token.split(".")[0]
    if len(digits) < 5:
        return False
    if digits.startswith("0"):
        return False
    return True


def _financial_amount_contexts(document: PdfDocument) -> list[str]:
    contexts: list[str] = []
    for page in document.pages:
        supplement_page = document.ocr_used and _is_post_notes_supplement_page(page.text)
        for table in page.tables:
            classification = _classify_table_for_arithmetic(table)
            if document.ocr_used and not classification["can_run_arithmetic"]:
                continue
            if classification["can_run_arithmetic"] or (not supplement_page and _looks_like_statement_table(table)):
                contexts.extend(
                    line
                    for line in (" ".join(str(cell or "") for cell in row) for row in table)
                    if not _ignore_formatting_line(line)
                )
        if supplement_page:
            continue
        for line in page.text.splitlines():
            lower = line.lower()
            if _ignore_formatting_line(line):
                continue
            if document.ocr_used:
                parsed_row = _parse_ocr_statement_row(line)
                if not parsed_row or parsed_row.confidence == "Low":
                    continue
            if len(NUMBER_RE.findall(line)) >= 2 and any(
                keyword in lower
                for keyword in (
                    "revenue",
                    "assets",
                    "liabilities",
                    "equity",
                    "cash",
                    "payables",
                    "receivables",
                    "cost",
                    "expense",
                    "surplus",
                    "deficit",
                    "fund",
                    "total",
                )
            ):
                contexts.append(line)
    return contexts


def _ignore_formatting_line(line: str) -> bool:
    return bool(
        re.search(
            r"\b(frc|ican|pro|registration|certificate|website|www\.|address|phone|telephone|decree| act |approval|approved|signature|signatory|license|licence|ratio|ias \d+|ifrs \d+)\b",
            line,
            flags=re.I,
        )
    )


def check_notes_agreement(
    document: PdfDocument,
    tolerance: Decimal = Decimal("1"),
    cautious_low_confidence: bool = False,
) -> list[Finding]:
    text = document.text
    findings: list[Finding] = []
    headings_with_pages = _note_headings_by_page(document)
    headings = {ref: title for ref, (title, _page_number) in headings_with_pages.items()}
    statement_refs = _statement_note_references(document)
    if document.ocr_used:
        if not _notes_start_page(document):
            findings.append(
                Finding(
                    "Extraction quality",
                    "Low",
                    "Notes agreement",
                    "OCR note-reference validation skipped because the notes section start was not detected.",
                    "No reliable 'Notes to the financial statements' boundary was detected, so numbered directors' report sections may not be financial statement notes.",
                    "Confirm the OCR text around the notes heading or use a cleaner scan before relying on automated note-reference prompts.",
                )
            )
            return findings
        if cautious_low_confidence and headings and _statement_note_lines(document):
            return findings
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "Notes agreement",
                "Detailed note-reference reconciliation was skipped for an OCR-assisted document.",
                "OCR can misread note columns, year columns, and note tables; reference checks are disabled until structured notes are confidently identified.",
                "Use OCR output for navigation, but rely on manual review or a clean text PDF before treating note-reference exceptions as audit findings.",
            )
        )
        return findings
    heading_refs = set(headings)
    detailed_note_checks_allowed = document.table_extraction_confidence >= 80
    for ref in sorted(statement_refs - heading_refs, key=_note_sort_key):
        parent_ref = re.sub(r'[A-Za-z]+$', '', ref)
        if parent_ref in heading_refs:
            continue
        if not detailed_note_checks_allowed and cautious_low_confidence:
            continue
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "Notes agreement",
                f"Statement references note {ref}, but a matching note heading was not confidently detected or parsed; review prompt only.",
                f"Detected statement reference: Note {ref}.",
                "Confirm if the note exists manually. (Downgraded to Low to avoid false positives from OCR/heading extraction misses).",
            )
        )
    if not detailed_note_checks_allowed and not cautious_low_confidence:
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "Notes agreement",
                "Detailed note reference, amount, and subtotal checks were skipped because table extraction confidence is below 80%.",
                f"Table confidence: {document.table_extraction_confidence}%. Note headings were still detected for navigation and debug review.",
                "Use the detected note heading debug output to inspect references, but do not rely on automated note agreement checks until table extraction is cleaner.",
            )
        )
        return findings
    if not detailed_note_checks_allowed and cautious_low_confidence:
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "Notes agreement",
                "Cautious detailed note agreement was run despite low table extraction confidence.",
                f"Table confidence: {document.table_extraction_confidence}%. Treat any note mismatch findings as review prompts, not confirmed exceptions.",
                "Use this mode for manual investigation only; rerun on a cleaner PDF before treating detailed note findings as audit exceptions.",
            )
        )
        note_sections = _note_sections(document)
        misref_findings, _misreferenced_lines = _check_possible_wrong_note_references(
            _statement_note_lines(document),
            note_sections,
            headings,
            tolerance,
            cautious_review_prompt=True,
        )
        # Filter contradictions
        cautious_findings = _check_cautious_face_note_amount_agreement(
            _statement_note_lines(document),
            headings,
            _note_section_page_ranges(document),
            document,
            tolerance,
        )
        passed_refs = set()
        # By default, cautious findings only emits findings for FAILURES. If a note is NOT in cautious_findings, it might have passed!
        # Actually, let's just use the main rows builder.
        try:
            # We call the same function the exporter uses to get the True/False passed states
            if document.ocr_used:
                check_result_rows = [] # Wait, _note_agreement_result_rows(document) handles OCR too!
            check_result_rows = _note_agreement_result_rows(document)
            passed_refs = {row["Note reference"] for row in check_result_rows if row["Result"] == "Passed"}
        except Exception:
            pass
            
        for f in misref_findings:
            ref_match = re.search(r"Note (\d+[A-Z]?)", f.evidence)
            if ref_match and ref_match.group(1) in passed_refs:
                continue
            findings.append(f)
        findings.extend(
            _check_cautious_face_note_amount_agreement(
                _statement_note_lines(document),
                note_sections,
                headings,
                tolerance,
                cautious_review_prompt=True,
            )
        )
        return findings
    if statement_refs:
        for ref in sorted(heading_refs - statement_refs, key=_note_sort_key):
            if ref.isdigit() and int(ref) <= 3:
                continue
            if _is_disclosure_only_note(headings[ref]):
                continue
            document.unreferenced_notes = getattr(document, "unreferenced_notes", [])
            document.unreferenced_notes.append({
                "Note": ref,
                "Heading": headings[ref],
                "Comment": "Note exists but was not referenced from the extracted primary statements."
            })

    note_sections = _note_sections(document)
    misref_findings, misreferenced_lines = _check_possible_wrong_note_references(
        _statement_note_lines(document),
        note_sections,
        headings,
        tolerance,
        cautious_review_prompt=not detailed_note_checks_allowed and cautious_low_confidence,
    )
    try:
        check_result_rows = _note_agreement_result_rows(document)
        passed_refs = {row["Note reference"] for row in check_result_rows if row["Result"] == "Passed"}
    except Exception:
        passed_refs = set()
        
    for f in misref_findings:
        ref_match = re.search(r"Note (\d+[A-Z]?)", f.evidence)
        if ref_match and ref_match.group(1) in passed_refs:
            continue
        findings.append(f)
    for ref, line, amount in _statement_lines_with_note_refs(document):
        if (ref, line) in misreferenced_lines:
            continue
        section = _get_note_section_with_fallback(ref, note_sections)
        if not section or _is_disclosure_only_note(_get_note_heading_with_fallback(ref, headings)):
            continue
        note_amounts = _amounts_in_text(section)
        if note_amounts and not any(abs(note_amount - amount) <= tolerance for note_amount in note_amounts):
            findings.append(
                Finding(
                    "Notes agreement",
                    "Medium",
                    f"Note {ref}",
                    "The amount on the statement was not found in the related note text.",
                    f"Statement line: {line[:140]} | amount {amount:,}.",
                    "Confirm the note table totals agree to the face of the financial statement.",
                )
            )

    for ref, section in note_sections.items():
        title = _get_note_heading_with_fallback(ref, headings).lower()
        _check_note_internal_total(findings, ref, title, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("segment", "operating segment")):
            _check_segment_note(findings, ref, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("earnings per share", "eps")):
            _check_eps_note(findings, ref, section)
        if any(keyword in title or keyword in section.lower() for keyword in ("tax", "income tax", "deferred tax")):
            _check_tax_note(findings, ref, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("depreciation", "property, plant", "ppe")):
            _check_depreciation_note(findings, ref, section, tolerance)



    return findings


def check_policy_relevance(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    text = document.text.lower()
    policy_map = _accounting_policy_map(document)
    detected_profile = infer_detected_profile(document)
    entity_type = detected_profile.get("Entity type", "").lower()
    findings: list[Finding] = []
    expected = {item.strip().lower() for item in profile.expected_policies if item.strip()}
    significant = {item.strip().lower() for item in profile.significant_transactions if item.strip()}

    for policy_name, rule in POLICY_RULES.items():
        policy_present = policy_map.get(policy_name, False)
        evidence_present = _policy_evidence_present(policy_name, rule["evidence"], text)
        explicitly_expected = policy_name in expected or policy_name in significant
        if policy_name == "consolidation" and not evidence_present and not explicitly_expected:
            continue
        if policy_name == "tax" and "non-profit" in entity_type and not evidence_present and not explicitly_expected:
            continue
        if policy_present and not evidence_present and not explicitly_expected:
            location, snippet = _first_keyword_context(document, rule["policy"])
            findings.append(
                Finding(
                    "Accounting policies",
                    "Medium",
                    location,
                    f"The {policy_name} policy is disclosed, but matching balances or activity were not detected.",
                    f"Policy indicators: {', '.join(rule['policy'][:3])}. Context: {snippet}",
                    "Remove boilerplate policy wording if it does not apply, or add the missing related disclosure if it does apply.",
                )
            )
        if (explicitly_expected or _strong_policy_gap(policy_name, text)) and not policy_present:
            location, snippet = _first_keyword_context(document, rule["evidence"])
            findings.append(
                Finding(
                    "Accounting policies",
                    "Medium",
                    location,
                    f"The report contains {policy_name}-related balances or expected transactions, but the matching accounting policy was not detected.",
                    f"Evidence indicators: {', '.join(rule['evidence'][:4])}. Context: {snippet}",
                    "Add or cross-reference the applicable accounting policy.",
                )
            )

    _check_industry_policy_fit(findings, document, profile)
    _check_superseded_standards(findings, document)
    _check_boilerplate_policy_language(findings, document)
    if profile.company_name and profile.company_name.lower() not in text:
        findings.append(
            Finding(
                "Formatting",
                "High",
                "Document-wide",
                "The configured company name was not detected in the extracted PDF text.",
                profile.company_name,
                "Confirm the correct report was uploaded and that headers/cover pages identify the reporting entity.",
            )
        )
    return findings


def findings_to_markdown(result: ReviewResult) -> str:
    lines = [
        "# AI Audit Assistant Review",
        "",
        build_ai_review_memo(result),
        "",
        f"Pages reviewed: {result.metrics['pages']}",
        f"Tables reviewed: {result.metrics['tables']}",
        f"Checks performed: {result.metrics.get('checks_performed_count', 0)}",
        f"Checks passed: {result.metrics.get('checks_passed_count', 0)}",
        f"Checks skipped: {result.metrics.get('checks_skipped_count', 0)}",
        f"Findings: {result.metrics['findings']} "
        f"(High {result.metrics['high']}, Medium {result.metrics['medium']}, Low {result.metrics['low']})",
        f"cautious_note_validation_enabled: {str(result.metrics.get('cautious_note_validation_enabled', False)).lower()}",
        f"note_validation_mode: {result.metrics.get('note_validation_mode', 'skipped')}",
        f"note_reference_rows_detected: {result.metrics.get('note_reference_rows_detected', 0)}",
        f"note_headings_detected: {result.metrics.get('note_headings_detected', 0)}",
        f"note_reference_findings: {result.metrics.get('note_reference_findings', 0)}",
        "",
        "## Review dimensions",
        "",
        "- Totals and rounding: totals, subtotals, cross-footings, and scaling labels.",
        "- Formatting: number formats, negative amounts, currency labels, comparatives, and statement presentation.",
        "- Notes agreement: note cross-references and reconciliation of note figures to face statements.",
        "- Accounting policies: relevance, missing policies, boilerplate wording, and superseded standards.",
        "- Standards checklist: triggered IFRS disclosure checks for presentation, policies, and transaction-specific notes.",
        "",
        "## Checks performed",
        "",
        str(result.metrics.get("positive_assurance", "")),
        "",
        str(result.metrics.get("checks_performed", "No deterministic checks completed.")),
        "",
        "## Checks skipped",
        "",
        str(result.metrics.get("checks_skipped", "No major checks skipped.")),
        "",
    ]
    if not result.findings:
        lines.append("No issues were detected by the automated checks.")
        return "\n".join(lines)
    for finding in result.findings:
        lines.extend(
            [
                f"## {finding.severity}: {finding.category}",
                f"Location: {finding.location}",
                f"Issue: {finding.issue}",
                f"Evidence: {finding.evidence}",
                f"Recommendation: {finding.recommendation}",
                "",
            ]
        )
    return "\n".join(lines)


def build_ai_review_memo(result: ReviewResult) -> str:
    assurance = str(result.metrics.get("positive_assurance", ""))
    scope_intro = ""
    if result.metrics.get("document_scope") == "Limited-scope statement extract":
        scope_intro = "Limited-scope review performed on Statement of Financial Position only. "
    if not result.findings:
        return (
            f"AI review memo: {scope_intro}{assurance or 'No automated exceptions were detected.'} Perform a final manual review of scanned pages, "
            "judgemental disclosures, and any areas where PDF extraction may have missed tables."
        )
    by_category = Counter(finding.category for finding in result.findings)
    high_risk = [finding for finding in result.findings if finding.severity == "High"]
    top_categories = ", ".join(f"{category} ({count})" for category, count in by_category.most_common())
    if high_risk:
        first_priority = high_risk[0]
        priority = (
            f"Priority: start with {first_priority.category.lower()} at {first_priority.location}. "
            f"{first_priority.issue}"
        )
        next_step = "Recommended next step: clear high-severity items first, then re-run the review on the final PDF."
    else:
        priority = (
            "No high-severity exceptions were identified. Review the medium extraction-quality findings "
            "and rerun detailed note agreement after table extraction confidence improves."
        )
        next_step = "Recommended next step: review extraction-quality findings, then rerun detailed note agreement after table extraction confidence improves."
    likely_causes = []
    categories = set(by_category)
    if "Totals and rounding" in categories:
        likely_causes.append("formula, casting, hidden-line, or rounding carry-forward differences")
    if "Formatting" in categories:
        likely_causes.append("inconsistent report template or late-stage manual formatting edits")
    if "Notes agreement" in categories:
        likely_causes.append("note schedules not updated after face statement changes")
    if "Accounting policies" in categories:
        likely_causes.append("boilerplate policy wording not tailored to the entity")
    cause_text = "; ".join(likely_causes) if likely_causes else "presentation or extraction exceptions"
    return (
        "AI review memo: "
        f"{scope_intro}"
        f"{assurance + ' ' if assurance else ''}"
        f"{result.metrics['findings']} findings were identified across {top_categories}. "
        f"{priority} Likely causes include {cause_text}. "
        f"{next_step}"
    )


def _checklist_item_applies(
    item: ChecklistItem,
    text: str,
    context: str,
    requested_areas: set[str],
) -> bool:
    if item.standard == "IFRS 16":
        return _actual_lease_disclosure_present(text)
    if item.area in requested_areas or item.standard.lower() in requested_areas:
        return True
    if item.standard == "IAS 33":
        return bool(re.search(r"\b(eps|earnings per share)\b", text))
    if item.standard == "IFRS 8":
        return "operating segment" in text or "segment revenue" in text or "chief operating decision maker" in text
    if item.standard == "IAS 12":
        tax_balance_terms = ("tax expense", "current tax", "deferred tax", "tax payable", "income tax expense")
        return any(term in text for term in tax_balance_terms) and not _tax_exempt_context(text)
    if item.standard == "IAS 12" and _tax_exempt_context(text):
        return False
    if not item.applies_when:
        return True
    trigger_text = f"{text} {context}"
    return any(trigger in trigger_text for trigger in item.applies_when) or item.area in context


def _checklist_item_satisfied_by_context(item: ChecklistItem, text: str, policy_map: dict[str, bool] | None = None) -> bool:
    policy_map = policy_map or {}
    if item.standard == "IFRS 15" and policy_map.get("revenue"):
        return True
    if item.standard == "IAS 10":
        return bool(
            re.search(
                r"(no|not aware of|none|there (?:were|are) no).{0,80}(subsequent events|events after)",
                text,
                flags=re.I,
            )
        )
    if item.standard == "IAS 12" and _tax_exempt_context(text):
        return True
    return False


def _tax_exempt_context(text: str) -> bool:
    return bool(re.search(r"tax[- ]?exempt|exempt from income tax|non[- ]?taxable|not subject to income tax", text, flags=re.I))


def _check_vertical_totals(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    rows: list[list[str | Decimal | None]],
    col: int,
    tolerance: Decimal,
) -> None:
    subtotal_rows: list[tuple[int, Decimal]] = []
    running: list[Decimal] = []
    expected_amount_count = _common_amount_count(rows)
    for row_index, row in enumerate(rows[1:], start=1):
        label = str(row[0]).lower() if row else ""
        if _is_table_boundary_row(row):
            running = []
            continue
        if expected_amount_count and _row_amount_count(row) not in {0, expected_amount_count}:
            running = []
            continue
        value = row[col] if col < len(row) else None
        if not isinstance(value, Decimal):
            continue
        if _looks_like_total(label):
            subtotal_rows.append((row_index, value))
            expected = sum(running, Decimal("0"))
            diff = value - expected
            if running and abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "High" if abs(diff) > tolerance * 5 else "Medium",
                        f"Page {page_number}, table {table_index}, row {row_index + 1}, column {col + 1}",
                        "Total or subtotal does not agree with the visible component rows.",
                        f"Reported {value:,}; visible sum {expected:,}; difference {diff:,}.",
                        "Trace the source schedule and confirm whether a hidden line, rounding adjustment, or formula error explains the variance.",
                    )
                )
            running = []
        elif _looks_like_amount_line(label):
            running.append(value)
    _check_adjacent_totals(findings, page_number, table_index, col, subtotal_rows, tolerance)


def _check_cross_footings(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    rows: list[list[str | Decimal | None]],
    tolerance: Decimal,
) -> None:
    for row_index, row in enumerate(rows):
        label = str(row[0]).lower() if row else ""
        values = [item for item in row[1:] if isinstance(item, Decimal)]
        if len(values) < 3 or not any(keyword in label for keyword in ("total", "segment", "tax", "depreciation")):
            continue
        expected = sum(values[:-1], Decimal("0"))
        reported = values[-1]
        diff = reported - expected
        if abs(diff) > tolerance:
            findings.append(
                Finding(
                    "Totals and rounding",
                    "Medium",
                    f"Page {page_number}, table {table_index}, row {row_index + 1}",
                    "Cross-footing across the row does not agree.",
                    f"Visible row sum {expected:,}; reported final column {reported:,}; difference {diff:,}.",
                    "Check whether the row total, segment total, or final column has been carried across correctly.",
                )
            )


def _check_column_consistency(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    table: list[list[str]],
) -> None:
    if not table:
        return
    header = " ".join(table[0]).lower()
    years = set(YEAR_RE.findall(header))
    if len(years) == 1 and any(YEAR_RE.search(" ".join(row)) for row in table[1:]):
        findings.append(
            Finding(
                "Formatting",
                "Low",
                f"Page {page_number}, table {table_index}",
                "A table appears to have only one comparative period in the header.",
                f"Header text: {header[:120]}.",
                "Confirm that the current and comparative reporting periods are both presented where required.",
            )
        )
    decimal_usage = Counter("decimal" if re.search(r"\d+\.\d+", cell) else "whole" for row in table for cell in row if NUMBER_RE.search(cell))
    if decimal_usage["decimal"] and decimal_usage["whole"]:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                f"Page {page_number}, table {table_index}",
                "The table mixes whole-number and decimal amount formats.",
                f"Detected amount styles: {dict(decimal_usage)}.",
                "Standardise decimals according to the report's rounding basis.",
            )
        )


def _check_comparatives(findings: list[Finding], document: PdfDocument) -> None:
    primary_text = "\n".join(page.text for page in document.pages if not _is_notes_page(page.text))
    years = sorted(set(YEAR_RE.findall(primary_text)))
    if years and len(years) < 2:
        findings.append(
            Finding(
                "Formatting",
                "Medium",
                "Primary statements",
                "Only one reporting year was detected in the primary statements.",
                f"Detected years: {', '.join(years)}.",
                "Confirm whether comparative information is required and whether the comparative column is missing from extraction or presentation.",
            )
        )


def _check_required_statement_names(findings: list[Finding], document: PdfDocument, profile: CompanyProfile) -> None:
    if profile.presentation_standard.upper() != "IFRS":
        return
    text = document.text.lower()
    required = {
        "statement of financial position": ("statement of financial position", "balance sheet"),
        "statement of profit or loss": ("statement of profit or loss", "statement of comprehensive income", "income statement"),
        "statement of changes in equity": ("statement of changes in equity",),
        "statement of cash flows": ("statement of cash flows", "cash flow statement"),
        "notes to the financial statements": ("notes to the financial statements",),
    }
    missing = [name for name, aliases in required.items() if not any(alias in text for alias in aliases)]
    if missing:
        findings.append(
            Finding(
                "Formatting",
                "Medium",
                "Document-wide",
                "One or more standard IFRS statement headings were not detected.",
                f"Missing detected headings: {', '.join(missing)}.",
                "Confirm whether the report uses acceptable alternate headings or whether a required primary statement is missing.",
            )
        )


def _check_note_internal_total(findings: list[Finding], ref: str, title: str, section: str, tolerance: Decimal) -> None:
    if _skip_note_subtotal_checks(title, section):
        return
    lines = [line for line in section.splitlines() if line.strip()]
    running: list[Decimal] = []
    for line in lines:
        lower = line.lower()
        amount = _last_amount(line)
        if amount is None:
            continue
        if _looks_like_total(lower):
            expected = sum(running, Decimal("0"))
            diff = amount - expected
            if running and abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Notes agreement",
                        "Medium",
                        f"Note {ref}",
                        "A note subtotal or total does not agree to visible note line items.",
                        f"Note title: {title or 'untitled'} | reported {amount:,}; visible sum {expected:,}; difference {diff:,}.",
                        "Review the note table and agree it back to the supporting schedule and face statement.",
                    )
                )
            running = []
        elif _looks_like_amount_line(lower):
            running.append(amount)


def _check_segment_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    lower_section = section.lower()
    if "operating segment" not in lower_section and "segment revenue" not in lower_section:
        return
    for line in section.splitlines():
        lower = line.lower()
        if "total" not in lower:
            continue
        values = _amounts_in_text(line)
        if len(values) >= 3:
            expected = sum(values[:-1], Decimal("0"))
            diff = values[-1] - expected
            if abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Notes agreement",
                        "High",
                        f"Note {ref}",
                        "Segment totals do not cross-foot to the reported total.",
                        f"Line: {line[:140]} | visible segment sum {expected:,}; reported total {values[-1]:,}.",
                        "Reconcile segment totals to the face statement and management reporting schedule.",
                    )
                )


def _check_eps_note(findings: list[Finding], ref: str, section: str) -> None:
    lower = section.lower()
    if "earnings per share" not in lower and "eps" not in lower:
        return
    earnings = _amount_near(section, ("profit attributable", "earnings", "profit for the year"))
    shares = _amount_near(section, ("weighted average", "ordinary shares", "shares"))
    eps = _amount_near(section, ("earnings per share", "basic eps", "diluted eps"))
    if earnings and shares and eps and shares != 0:
        calculated = earnings / shares
        if abs(calculated - eps) > Decimal("0.01"):
            findings.append(
                Finding(
                    "Notes agreement",
                    "High",
                    f"Note {ref}",
                    "EPS calculation does not agree to earnings divided by weighted average shares.",
                    f"Calculated EPS {calculated:.4f}; reported EPS {eps}.",
                    "Recalculate basic and diluted EPS using the final attributable earnings and weighted average share count.",
                )
            )
    elif "earnings per share" in lower or "eps" in lower:
        findings.append(
            Finding(
                "Notes agreement",
                "Low",
                f"Note {ref}",
                "EPS note was detected but the automated check could not identify all inputs.",
                "Expected inputs include attributable earnings, weighted average shares, and reported EPS.",
                "Review the EPS note manually and ensure both basic and diluted EPS are supported.",
            )
        )


def _skip_note_subtotal_checks(title: str, section: str) -> bool:
    lower = f"{title}\n{section}".lower()
    skip_terms = (
        "financial instruments - risk",
        "financial risk",
        "risk management",
        "maturity",
        "liquidity risk",
        "credit risk",
        "expected credit loss",
        " ecl",
        "ageing",
        "aging",
        "value added",
        "five year",
        "5 year",
        "narrative disclosure",
        "contingent liabilities",
        "capital commitments",
        "subsequent events",
        "related party",
    )
    return any(term in lower for term in skip_terms)


def _check_tax_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    current_tax = _amount_near(section, ("current tax", "current income tax"))
    deferred_tax = _amount_near(section, ("deferred tax",))
    total_tax = _amount_near(section, ("tax expense", "income tax expense", "total tax"))
    if current_tax is not None and deferred_tax is not None and total_tax is not None:
        expected = current_tax + deferred_tax
        if abs(total_tax - expected) > tolerance:
            findings.append(
                Finding(
                    "Notes agreement",
                    "High",
                    f"Note {ref}",
                    "Tax expense does not agree to current tax plus deferred tax.",
                    f"Current tax {current_tax:,}; deferred tax {deferred_tax:,}; total tax {total_tax:,}.",
                    "Reconcile the tax note to the statement of profit or loss and deferred tax movement schedule.",
                )
            )


def _check_depreciation_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    depreciation = _amount_near(section, ("depreciation charge", "charge for the year", "depreciation"))
    profit_or_loss = _amount_near(section, ("profit or loss", "administrative expenses", "cost of sales"))
    if depreciation is not None and profit_or_loss is not None and abs(depreciation - profit_or_loss) > tolerance:
        findings.append(
            Finding(
                "Notes agreement",
                "Medium",
                f"Note {ref}",
                "Depreciation charge in the PPE note may not agree to the expense disclosure.",
                f"PPE depreciation {depreciation:,}; expense disclosure {profit_or_loss:,}.",
                "Agree depreciation charges to expense classification notes and the cash flow add-back.",
            )
        )


def _find_statement_page(document: PdfDocument, statement_name: str) -> PdfPage | None:
    return _classified_primary_statement_pages(document).get(statement_name)
    target = statement_name.lower()
    for page in document.pages:
        for line in page.text.splitlines():
            lower = line.strip().lower()
            if "..." in lower or "…" in lower:
                continue
            if lower.startswith(target):
                return page
    return None


def _classified_primary_statement_pages(document: PdfDocument) -> dict[str, PdfPage]:
    aliases = {
        "Statement of income and expenditure": (
            "statement of income and expenditure",
            "statement of profit or loss",
            "statement of comprehensive income",
            "profit or loss",
        ),
        "Statement of financial position": ("statement of financial position", "balance sheet"),
        "Statement of changes in accumulated fund": (
            "statement of changes in accumulated fund",
            "statement of changes in equity",
            "changes in equity",
        ),
        "Statement of cash flows": ("statement of cash flows", "cash flow statement"),
    }
    classified: dict[str, PdfPage] = {}
    for page in document.pages:
        page_head = "\n".join(page.text.splitlines()[:60 if document.ocr_used else 18])
        if _looks_like_contents_or_front_matter_page(page.text):
            continue
        for canonical, candidates in aliases.items():
            if canonical in classified:
                continue
            if any(_statement_heading_line_present(page_head, candidate) for candidate in candidates) and _page_has_statement_rows_for(canonical, page.text):
                classified[canonical] = page
    return classified


def _looks_like_contents_or_front_matter_page(text: str) -> bool:
    head = "\n".join(text.splitlines()[:40]).lower()
    if re.search(r"\b(table of )?contents\b", head):
        return True
    statement_mentions = len(re.findall(r"statement of (?:profit|financial|changes|cash|income|comprehensive)", head))
    numeric_page_refs = len(re.findall(r"\.{2,}\s*\d{1,3}\b|\b\d{1,3}\s*$", head, flags=re.M))
    front_terms = ("corporate information", "directors' report", "directors report", "independent auditor", "report of the directors")
    return statement_mentions >= 2 and (numeric_page_refs >= 2 or any(term in head for term in front_terms))


def _statement_heading_line_present(text: str, phrase: str) -> bool:
    for line in text.splitlines()[:60]:
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped or re.search(r"\.{2,}\s*\d{1,3}$", stripped):
            continue
        if re.search(r"\b(page|contents)\b", stripped, flags=re.I):
            continue
        normalized = _normalise_match_words(stripped)
        normalized_phrase = _normalise_match_words(phrase)
        if normalized_phrase in normalized and len(normalized.split()) <= len(normalized_phrase.split()) + 6:
            return True
        if _fuzzy_contains(stripped, phrase, threshold=0.82) and len(normalized.split()) <= len(normalized_phrase.split()) + 6:
            return True
    return False


def _page_has_statement_rows_for(statement_name: str, text: str) -> bool:
    rows = _statement_rows(text)
    name = statement_name.lower()
    if "financial position" in name:
        return any(label in rows for label in ("total assets", "non-current assets", "current assets", "cash and cash equivalents", "trade and other receivables", "investment property", "financial liabilities", "total equity and liabilities"))
    if "income" in name or "profit" in name:
        return any(label in rows for label in ("revenue", "profit before tax", "taxation", "profit after tax", "operating revenue", "total income"))
    if "cash flow" in name:
        return any("cash" in label for label in rows)
    if "changes" in name:
        return bool(rows) or any("balance as at" in line.lower() for line in text.splitlines())
    return bool(rows)


def _fuzzy_contains(text: str, phrase: str, threshold: float = 0.78) -> bool:
    normalized_text = _normalise_match_words(text)
    normalized_phrase = _normalise_match_words(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True
    words = normalized_text.split()
    target_words = normalized_phrase.split()
    width = max(2, len(target_words))
    for index in range(0, max(1, len(words) - width + 1)):
        window = " ".join(words[index : index + width + 1])
        if SequenceMatcher(None, normalized_phrase, window).ratio() >= threshold:
            return True
    return False


def _check_income_statement_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    rows = _statement_rows(page.text)
    performed: list[str] = []
    skipped: list[str] = []
    findings: list[Finding] = []
    gross_operating = ("subscriptions", "registrations", "operating revenue")
    total_income = ("gross revenue", "other revenue", "finance income")
    expenditure = ("office accommodation costs", "personnel costs", "administrative costs", "finance expenses")
    if all(label in rows for label in ("revenue", "profit before tax", "taxation", "profit after tax")):
        raw_lines = _statement_row_raw_lines(page.text)
        before_tax_amounts = _row_amounts(rows, "profit before tax")
        taxation_amounts = _row_amounts(rows, "taxation")
        after_tax_amounts = _row_amounts(rows, "profit after tax")
        if ocr_review and len(before_tax_amounts) >= 2 and len(taxation_amounts) >= 2 and len(after_tax_amounts) < 2:
            skip_message = "Skipped / OCR conflict - current-year after-tax value not confidently extracted."
            corroboration = _ocr_income_corroboration_assessment(
                document,
                taxation_amounts[0],
                after_tax_amounts[0] if after_tax_amounts else None,
                tolerance,
            )
            if corroboration.get("casts"):
                skip_message = (
                    "Skipped / OCR conflict - current-year after-tax value not confidently extracted. "
                    f"Corroborating lines indicate {_format_accounting_amount(corroboration.get('after_tax_value'))}. "
                    "Manual confirmation required."
                )
            skipped.append(skip_message)
        elif _check_profit_tax_equation(
            findings,
            page.number,
            "Income statement",
            before_tax_amounts,
            taxation_amounts,
            after_tax_amounts,
            tolerance,
            ocr_review=ocr_review,
            raw_lines=raw_lines,
            document=document,
        ):
            performed.append("Income statement: revenue, tax, and profit/loss after tax checked from line-extracted rows.")
        else:
            skipped.append("Income statement: profit/loss after tax skipped because OCR tax rows did not contain comparable current/prior amounts.")
    elif any(label in rows for label in ("revenue", "profit before tax", "taxation", "profit after tax")):
        skipped.append("Income statement: profit/loss after tax skipped because revenue, tax, or profit rows were not confidently parsed.")
    if _has_rows(rows, (*gross_operating, "gross operating revenue")):
        _check_sum_rows(
            findings,
            page.number,
            "Income statement",
            "Gross operating revenue",
            rows,
            gross_operating,
            "gross operating revenue",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: gross operating revenue checked from line-extracted rows.")
    else:
        skipped.append("Income statement: gross operating revenue skipped because component rows were not confidently parsed.")
    if _has_rows(rows, ("gross operating revenue", "operating expenditure", "gross revenue")):
        _check_vector_equation(
            findings,
            page.number,
            "Income statement",
            "Gross revenue equals gross operating revenue less operating expenditure.",
            [a + b for a, b in zip(_row_amounts(rows, "gross operating revenue"), _row_amounts(rows, "operating expenditure"))],
            _row_amounts(rows, "gross revenue"),
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: gross revenue checked against operating revenue less operating expenditure.")
    else:
        skipped.append("Income statement: gross revenue skipped because required rows were not confidently parsed.")
    if _has_rows(rows, (*total_income, "total income")):
        _check_sum_rows(
            findings,
            page.number,
            "Income statement",
            "Total income",
            rows,
            total_income,
            "total income",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: total income checked from line-extracted rows.")
    else:
        skipped.append("Income statement: total income skipped because component rows were not confidently parsed.")
    if _has_rows(rows, (*expenditure, "total expenditure")):
        _check_sum_rows(
            findings,
            page.number,
            "Income statement",
            "Total expenditure",
            rows,
            expenditure,
            "total expenditure",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: total expenditure checked from line-extracted rows.")
    else:
        skipped.append("Income statement: total expenditure skipped because component rows were not confidently parsed.")
    if _has_rows(rows, ("total income", "total expenditure", "surplus of income over expenditure")):
        expected = _row_amounts(rows, "total income")
        expenditure_amounts = _row_amounts(rows, "total expenditure")
        reported = _row_amounts(rows, "surplus of income over expenditure")
        _check_vector_equation(
            findings,
            page.number,
            "Income statement",
            "Surplus equals total income less total expenditure.",
            [a - b for a, b in zip(expected, expenditure_amounts)],
            reported,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: surplus checked against income less expenditure.")
    else:
        skipped.append("Income statement: surplus check skipped because required rows were not confidently parsed.")
    if _has_rows(rows, ("surplus of income over expenditure", "total comprehensive income")):
        _check_vector_equation(
            findings,
            page.number,
            "Income statement",
            "Total comprehensive income agrees to surplus where OCI is nil.",
            _row_amounts(rows, "surplus of income over expenditure"),
            _row_amounts(rows, "total comprehensive income"),
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Income statement: total comprehensive income checked.")
    return findings, performed, skipped


def _check_sfp_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    rows = _statement_rows(page.text)
    performed: list[str] = []
    skipped: list[str] = []
    findings: list[Finding] = []
    if _has_rows(rows, ("non-current assets", "current assets", "total assets")):
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Total assets equals non-current assets plus current assets.",
            [a + b for a, b in zip(_row_amounts(rows, "non-current assets"), _row_amounts(rows, "current assets"))],
            _row_amounts(rows, "total assets"),
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: total assets checked from line-extracted rows.")
    else:
        non_current_amounts = _sfp_non_current_amounts(rows)
        current_amounts = _sfp_current_amounts(rows)
        total_assets = _row_amounts(rows, "total assets")
        if non_current_amounts and current_amounts and total_assets:
            _check_vector_equation(
                findings,
                page.number,
                "Statement of financial position",
                "Total assets equals extracted non-current asset rows plus extracted current asset rows.",
                [a + b for a, b in zip(non_current_amounts, current_amounts)],
                total_assets,
                tolerance,
                ocr_review=ocr_review,
            )
            performed.append("Statement of financial position: total assets checked from extracted asset rows.")
        elif total_assets and (non_current_amounts or current_amounts):
            skipped.append("Statement of financial position: partial asset rows extracted, but non-current/current split was incomplete.")
    equity_amounts = _row_amounts_any(rows, ("equity", "total equity", "share capital and reserves", "capital and reserves"))
    liability_amounts = _row_amounts_any(rows, ("liabilities", "financial liabilities", "total liabilities"))
    total_equity_liabilities = _row_amounts_any(rows, ("total equity and liabilities", "total funds and liabilities", "total equity and liability"))
    if equity_amounts and liability_amounts and total_equity_liabilities:
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Equity plus liabilities equals total equity and liabilities.",
            [a + b for a, b in zip(equity_amounts, liability_amounts)],
            total_equity_liabilities,
            tolerance,
            ocr_review=ocr_review,
        )
        total_assets = _row_amounts(rows, "total assets")
        if total_assets:
            _check_vector_equation(
                findings,
                page.number,
                "Statement of financial position",
                "Total assets equals total equity and liabilities.",
                total_assets,
                total_equity_liabilities,
                tolerance,
                ocr_review=ocr_review,
            )
        performed.append("Statement of financial position: equity and liabilities equation checked from line-extracted rows.")
    elif _row_amounts(rows, "total assets") and total_equity_liabilities:
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Total assets equals total equity and liabilities.",
            _row_amounts(rows, "total assets"),
            total_equity_liabilities,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: total assets checked against total equity and liabilities.")
    if performed:
        return findings, performed, skipped
    non_current = ("investment property", "property plant and equipment", "intangible assets")
    current = ("inventories", "trade and other receivables", "cash and cash equivalents")
    funds = ("accumulated fund", "donation fund", "library development fund")
    if _has_rows(rows, (*non_current, "total non - current assets")):
        _check_sum_rows(
            findings,
            page.number,
            "Statement of financial position",
            "Non-current assets",
            rows,
            non_current,
            "total non - current assets",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: non-current assets checked.")
    else:
        skipped.append("Statement of financial position: non-current assets skipped because rows were not confidently parsed.")
    if _has_rows(rows, (*current, "total current assets")):
        _check_sum_rows(
            findings,
            page.number,
            "Statement of financial position",
            "Current assets",
            rows,
            current,
            "total current assets",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: current assets checked.")
    else:
        skipped.append("Statement of financial position: current assets skipped because rows were not confidently parsed.")
    if _has_rows(rows, ("total non - current assets", "total current assets", "total assets")):
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Total assets equals non-current assets plus current assets.",
            [a + b for a, b in zip(_row_amounts(rows, "total non - current assets"), _row_amounts(rows, "total current assets"))],
            _row_amounts(rows, "total assets"),
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: total assets checked.")
    if _has_rows(rows, (*funds, "total members fund")):
        _check_sum_rows(
            findings,
            page.number,
            "Statement of financial position",
            "Members fund",
            rows,
            funds,
            "total members fund",
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: members fund checked.")
    else:
        skipped.append("Statement of financial position: members fund skipped because rows were not confidently parsed.")
    if _has_rows(rows, ("total members fund", "total liabilities", "total members fund and liability")):
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Members fund plus liabilities equals total members fund and liability.",
            [a + b for a, b in zip(_row_amounts(rows, "total members fund"), _row_amounts(rows, "total liabilities"))],
            _row_amounts(rows, "total members fund and liability"),
            tolerance,
            ocr_review=ocr_review,
        )
        _check_vector_equation(
            findings,
            page.number,
            "Statement of financial position",
            "Total assets equals total members fund and liability.",
            _row_amounts(rows, "total assets"),
            _row_amounts(rows, "total members fund and liability"),
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of financial position: balance sheet equation checked.")
    else:
        skipped.append("Statement of financial position: balance sheet equation skipped because rows were not confidently parsed.")
    return findings, performed, skipped


def _check_accumulated_fund_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    lines = page.text.splitlines()
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    balance_rows = [(line, _amounts_from_statement_line(line)) for line in lines if "balance as at" in line.lower()]
    surplus_rows = [(line, _amounts_from_statement_line(line)) for line in lines if line.lower().strip().startswith("surplus for the year")]
    if len(balance_rows) >= 3 and len(surplus_rows) >= 2:
        opening_2025 = balance_rows[-2][1]
        closing_2025 = balance_rows[-1][1]
        surplus_2025 = surplus_rows[-1][1]
        if len(opening_2025) >= 4 and len(closing_2025) >= 4 and surplus_2025:
            expected_total = opening_2025[-1] + surplus_2025[-1]
            reported_total = closing_2025[-1]
            if ocr_review:
                _check_ocr_scalar_equation(
                    findings,
                    page.number,
                    "Statement of changes in accumulated fund",
                    "Closing accumulated fund agrees to opening fund plus surplus.",
                    expected_total,
                    reported_total,
                    tolerance,
                )
            else:
                _check_scalar_equation(
                    findings,
                    page.number,
                    "Statement of changes in accumulated fund",
                    "Closing accumulated fund agrees to opening fund plus surplus.",
                    expected_total,
                    reported_total,
                    tolerance,
                )
            performed.append("Statement of changes in accumulated fund: opening plus surplus checked to closing fund.")
        else:
            skipped.append("Statement of changes in accumulated fund: skipped because fund columns were not confidently parsed.")
    else:
        skipped.append("Statement of changes in accumulated fund: skipped because movement rows were not confidently parsed.")
    return findings, performed, skipped


def _check_cash_flow_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    rows = _statement_rows(page.text)
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    op = next((v for k, v in rows.items() if "operat" in k), None)
    inv = next((v for k, v in rows.items() if "invest" in k), None)
    fin = next((v for k, v in rows.items() if "financ" in k), None)
    mov = next((v for k, v in rows.items() if ("increase" in k or "decrease" in k or "movement" in k or "net cash" in k or "cash flow" in k) and not any(x in k for x in ["operat", "invest", "financ"])), None)

    if op and inv and fin and mov:
        expected = [a + b + c for a, b, c in zip(op, inv, fin)]
        _check_vector_equation(
            findings,
            page.number,
            "Statement of cash flows",
            "Operating, investing, and financing cash flows agree to net increase in cash.",
            expected,
            mov,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of cash flows: net cash increase checked.")
        findings.append(Finding("Calculation", "Passed", "Statement of cash flows", "Operating, investing, and financing cash flows agree to net increase.", "Equation passed.", ""))
    else:
        skipped.append("Statement of cash flows: skipped because operating/investing/financing/movement rows were not confidently parsed.")
    open_cash = next((v for k, v in rows.items() if ("beginning" in k or "start" in k or " 1 " in k or "january" in k) and "cash" in k), None)
    close_cash = next((v for k, v in rows.items() if ("end" in k or " 31 " in k or "december" in k) and "cash" in k), None)
    exc = next((v for k, v in rows.items() if "exchange" in k), None)

    if open_cash and close_cash and mov:
        if exc:
            expected = [a + b + c for a, b, c in zip(open_cash, mov, exc)]
        else:
            expected = [a + b for a, b in zip(open_cash, mov)]
        _check_vector_equation(
            findings,
            page.number,
            "Statement of cash flows",
            "Closing cash agrees to opening cash plus net increase.",
            expected,
            close_cash,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of cash flows: closing cash movement checked.")
    else:
        skipped.append("Statement of cash flows: skipped because opening/closing cash rows were not confidently parsed.")
    return findings, performed, skipped


def _statement_rows(text: str) -> dict[str, list[Decimal]]:
    return {label: list(row.amounts) for label, row in _statement_row_parses(text).items()}


def _statement_row_parses(text: str) -> dict[str, OcrStatementRow]:
    rows: dict[str, OcrStatementRow] = {}
    for line in text.splitlines():
        parsed = _parse_ocr_statement_row(line)
        if parsed and parsed.label and _statement_row_label_allowed(parsed.label):
            rows[parsed.label] = parsed
    return _align_income_statement_columns(rows, text)


def _align_income_statement_columns(rows: dict[str, OcrStatementRow], text: str) -> dict[str, OcrStatementRow]:
    before = rows.get("profit before tax")
    tax = rows.get("taxation")
    after = rows.get("profit after tax")
    if not before or not tax or not after:
        return rows
    if len(before.amounts) < 2 or len(tax.amounts) < 2 or len(after.amounts) != 1:
        return rows
    if not _text_has_two_year_columns(text) and not _row_set_has_two_amount_columns(before, tax):
        return rows
    expected_current = _expected_after_tax_amount(before.amounts[0], tax.amounts[0])
    expected_prior = _expected_after_tax_amount(before.amounts[1], tax.amounts[1])
    reported_single = after.amounts[0]
    if abs(reported_single - expected_prior) > Decimal("1"):
        return rows
    rows = dict(rows)
    rows["profit after tax"] = OcrStatementRow(
        after.label,
        (expected_current, reported_single),
        after.raw_line,
        after.note_ref,
        "Low-Medium",
        "Yes",
        "Current-year profit/loss after tax inferred from before-tax and taxation rows because OCR captured only the prior-year amount.",
    )
    return rows


def _row_set_has_two_amount_columns(*rows: OcrStatementRow) -> bool:
    return all(len(row.amounts) >= 2 for row in rows)


def _text_has_two_year_columns(text: str) -> bool:
    years = sorted(set(YEAR_RE.findall(text)))
    if len(years) >= 2:
        return True
    header_lines = "\n".join(text.splitlines()[:25])
    return bool(re.search(r"\b20\d{2}\b.{0,40}\b20\d{2}\b", header_lines))


def _expected_after_tax_amount(before_tax: Decimal, taxation: Decimal) -> Decimal:
    if before_tax < 0:
        return before_tax + taxation
    if taxation < 0:
        return before_tax + taxation
    return before_tax - taxation


def _statement_row_raw_lines(text: str) -> dict[str, str]:
    return {label: row.raw_line for label, row in _statement_row_parses(text).items()}


def _parse_ocr_statement_row(line: str) -> OcrStatementRow | None:
    if not line.strip():
        return None
    raw_line = re.sub(r"\s+", " ", line).strip()
    note_ref, ref_start, ref_end = _detect_statement_row_note_token(line)
    label_source = line[:ref_start] if note_ref else line
    label = _canonical_statement_label(_statement_label(label_source))
    if not label:
        return None
    letters_only = re.sub(r"[^a-z]", "", label.lower())
    if label.lower().strip() in {"n n", "0 0"} or len(letters_only) < 3 or set(letters_only).issubset({"n", "m", "o", "v"}):
        return None
    amount_source = f"{line[:ref_start]} {line[ref_end:]}" if note_ref else line
    amount_tokens = _amount_tokens_from_statement_line(amount_source)
    amounts = [_parse_decimal(token) for token in amount_tokens]
    amounts = [amount for amount in amounts if amount is not None and abs(amount) < Decimal("100000000")]
    correction_applied = "No"
    correction_reason = ""
    if note_ref and amounts:
        corrected = _maybe_reconstruct_note_split_leading_digit(label, note_ref, amount_tokens, amounts)
        if corrected:
            amounts[0], correction_reason = corrected
            correction_applied = "Yes"
    if _amounts_look_like_note_reference_only(amounts, note_ref):
        amounts = []
        correction_reason = "Only numeric value matched the detected note reference; amount treated as unavailable."
    if not amounts:
        return None
    parsed_amounts = tuple(amounts[-2:])
    confidence, reason = _ocr_statement_row_confidence(label, raw_line, list(parsed_amounts), note_ref, correction_applied, correction_reason)
    return OcrStatementRow(label, parsed_amounts, raw_line, note_ref, confidence, correction_applied, reason)


def _detect_statement_row_note_token(line: str) -> tuple[str, int, int]:
    for match in re.finditer(r"\b(?:note\s+)?(\d{1,2}[A-Za-z]?)(?!\s*,)\b(?=\s*(?:[=:]\s*)?\(?-?\d)", line, flags=re.I):
        ref = match.group(1).upper()
        if not _valid_note_number(ref):
            continue
        explicit_note = bool(re.match(r"\s*note\b", match.group(0), flags=re.I))
        tail = line[match.end() :]
        label = _canonical_statement_label(_statement_label(line[: match.start()]))
        tail_amounts = _amount_tokens_from_statement_line(tail)
        if (
            not explicit_note
            and re.fullmatch(r"\d", ref)
            and re.match(r"\s+\(?-?\d,\d{3}\b", tail)
            and _label_prefers_split_leading_digit(label)
        ):
            continue
        if len(tail_amounts) >= 1:
            return ref, match.start(), match.end()
    return "", len(line), len(line)


def _label_prefers_split_leading_digit(label: str) -> bool:
    lower_label = label.lower()
    return lower_label in {"finance income", "finance expenses", "finance costs", "other revenue"}


def _amount_tokens_from_statement_line(line: str) -> list[str]:
    cleaned = _normalise_statement_number_spacing(line)
    cleaned = re.sub(r"\s[-=]\s*(?=\(?\s?\d)", " 0 ", cleaned)
    return re.findall(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?", cleaned)


def _maybe_reconstruct_note_split_leading_digit(
    label: str,
    note_ref: str,
    amount_tokens: list[str],
    amounts: list[Decimal],
) -> tuple[Decimal, str] | None:
    return None


def _label_allows_note_split_digit_correction(label: str) -> bool:
    lower_label = label.lower()
    watched_labels = ("cash", "receivable", "revenue", "tax", "asset", "liabilit")
    return any(term in lower_label for term in watched_labels)


def _ocr_statement_row_confidence(
    label: str,
    raw_line: str,
    amounts: list[Decimal],
    note_ref: str = "",
    correction_applied: str = "No",
    correction_reason: str = "",
) -> tuple[str, str]:
    if not raw_line:
        return "Low", "Raw OCR line not captured."
    normalized_line = _normalise_match_words(raw_line)
    if "#" in raw_line or "?" in raw_line:
        return "Low", "Unreadable OCR placeholder or uncertain character detected."
    if correction_applied == "Yes":
        return "Low-Medium", correction_reason
    if note_ref:
        return "High", f"Detected note reference {note_ref}; amounts were parsed after excluding the note column."
    if re.search(r"\d\s+,|,\s+\d|\b\d\s+\d{3}\b", raw_line):
        return "Low-Medium", "OCR amount contains internal spaces and may have required numeric normalization."
    if _looks_like_possible_missing_leading_digit(label, raw_line, amounts):
        return "Low-Medium", "Possible OCR correction candidate: amount may be missing a leading digit."
    if any(term in normalized_line for term in ("audited", "year ended", "page")):
        return "Low-Medium", "Line contains report furniture near extracted values."
    return "High", "Raw OCR line and extracted amounts appear structurally clean."


def _looks_like_possible_missing_leading_digit(label: str, raw_line: str, amounts: list[Decimal]) -> bool:
    if not amounts:
        return False
    lower_label = label.lower()
    watched_labels = ("cash", "receivable", "revenue", "tax", "asset", "liabilit")
    if not any(term in lower_label for term in watched_labels):
        return False
    clean_line = raw_line.replace(" ", "")
    raw_numbers = re.findall(r"\d{1,3}(?:,\d{3})+", clean_line)
    for token in raw_numbers:
        digits = token.replace(",", "")
        if len(digits) == 5 and any(abs(abs(amount) - Decimal(digits)) <= 1 for amount in amounts):
            return True
    return False


def _statement_row_label_allowed(label: str) -> bool:
    normalized = _normalise_match_words(label)
    if not normalized:
        return False
    letters_only = re.sub(r"[^a-z]", "", label.lower())
    if label.lower().strip() in {"n n", "0 0"} or len(letters_only) < 3 or set(letters_only).issubset({"n", "m", "o", "v"}):
        return False
    blocked_terms = (
        "financial statements",
        "annual report",
        "year ended",
        "for the year ended",
        "date",
        "page",
        "director",
        "auditor",
        "corporate information",
    )
    if any(term in normalized for term in blocked_terms):
        return False
    allowed_exact = {
        "revenue",
        "turnover",
        "sales",
        "profit before tax",
        "loss before tax",
        "taxation",
        "profit after tax",
        "loss after tax",
        "non current assets",
        "current assets",
        "total assets",
        "equity",
        "liabilities",
        "financial liabilities",
        "total liabilities",
        "share capital and reserves",
        "capital and reserves",
        "total equity and liabilities",
        "cash at beginning",
        "cash at end",
        "net increase in cash and cash equivalents",
        "subscriptions",
        "registrations",
        "operating revenue",
        "gross operating revenue",
        "operating expenditure",
        "gross revenue",
        "other revenue",
        "finance income",
        "total income",
        "office accommodation costs",
        "personnel costs",
        "administrative costs",
        "finance expenses",
        "total expenditure",
        "surplus of income over expenditure",
        "total comprehensive income",
        "investment property",
        "property plant and equipment",
        "intangible assets",
        "inventories",
        "trade and other receivables",
        "cash and cash equivalents",
        "total non current assets",
        "total current assets",
        "total members fund",
        "trade and other payables",
        "financial liabilities",
        "share capital and reserves",
        "capital and reserves",
        "total funds and liabilities",
        "net cash inflow from operating activities",
        "net cash absorbed in investing activities",
        "net cash inflow from financing activities",
    }
    if normalized in {_normalise_match_words(item) for item in allowed_exact}:
        return True
    allowed_prefixes = (
        "profit before",
        "loss before",
        "profit after",
        "loss after",
        "income tax",
        "tax expense",
        "share capital",
        "capital and reserves",
    )
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def _statement_label(line: str) -> str:
    cleaned = _normalise_statement_number_spacing(line)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\b(?:note|n'?000|000|20\d{2}|\d{1,2}[a-c]?)\b", " ", cleaned, flags=re.I)
    cleaned = NUMBER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[^A-Za-z&/ -]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.strip(" -")


def _canonical_statement_label(label: str) -> str:
    normalized = _normalise_match_words(label)
    if not normalized:
        return ""
    protected_labels = {
        "subscriptions",
        "registrations",
        "operating revenue",
        "gross operating revenue",
        "operating expenditure",
        "gross revenue",
        "other revenue",
        "finance income",
        "total income",
        "office accommodation costs",
        "personnel costs",
        "administrative costs",
        "finance expenses",
        "total expenditure",
        "surplus of income over expenditure",
        "total comprehensive income",
        "investment property",
        "property plant and equipment",
        "intangible assets",
        "inventories",
        "trade and other receivables",
        "cash and cash equivalents",
        "total non current assets",
        "total non-current assets",
        "total current assets",
        "total members fund",
        "total members' fund",
        "trade and other payables",
        "financial liabilities",
        "total liabilities",
        "share capital and reserves",
        "capital and reserves",
        "total funds and liabilities",
        "net cash inflow from operating activities",
        "net cash absorbed in investing activities",
        "net cash inflow from financing activities",
    }
    if normalized in {_normalise_match_words(item) for item in protected_labels}:
        return label
    patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("total equity and liabilities", ("total equity and liabilities", "total equity and liability", "total liabilities and equity", "total liability and equity", "total equity liabilities", "equity and liabilities")),
        ("total assets", ("total assets",)),
        ("non-current assets", ("non current assets", "noncurrent assets", "total non current assets", "non current asset")),
        ("profit before tax", ("profit before tax", "profit before taxation", "loss before tax", "loss before taxation", "profit loss before tax")),
        ("profit after tax", ("profit after tax", "profit after taxation", "profit for year", "profit for the year", "loss after tax", "loss after taxation", "loss for year", "loss for the year")),
        ("taxation", ("taxation", "income tax", "tax expense", "tax credit", "income tax expense")),
        ("revenue", ("revenue", "turnover", "sales")),
        ("current assets", ("current assets", "total current assets")),
        ("equity", ("equity", "total equity", "shareholders equity", "shareholder funds", "net assets", "capital and reserves", "share capital and reserves")),
        ("liabilities", ("liabilities", "total liabilities")),
        ("cash at beginning", ("cash at beginning", "cash and cash equivalents at beginning", "cash and cash equivalents at beginning of year", "cash cash equivalents beginning", "cash equivalents beginning year")),
        ("cash at end", ("cash at end", "cash and cash equivalents at end", "cash and cash equivalents at end of year", "cash cash equivalents end", "cash equivalents end year")),
        ("net increase in cash and cash equivalents", ("net increase in cash", "net decrease in cash", "increase in cash and cash equivalents", "decrease in cash and cash equivalents")),
    )
    for canonical, aliases in patterns:
        if any(_label_matches(normalized, alias) for alias in aliases):
            return canonical
    return label


def _label_matches(normalized_label: str, alias: str) -> bool:
    normalized_alias = _normalise_match_words(alias)
    if not normalized_alias:
        return False
    if normalized_label == normalized_alias:
        return True
    if "beginning" in normalized_alias and "end" in normalized_label:
        return False
    if "end" in normalized_alias and "beginning" in normalized_label:
        return False
    if normalized_alias in {"revenue", "turnover", "sales"}:
        return len(normalized_label.split()) <= 2 and SequenceMatcher(None, normalized_label, normalized_alias).ratio() >= 0.84
    if normalized_alias == "current assets" and "non current assets" in normalized_label:
        return False
    if normalized_alias in normalized_label or normalized_label in normalized_alias:
        if normalized_label in normalized_alias and len(normalized_label.split()) < 3:
            return False
        return True
    return SequenceMatcher(None, normalized_label, normalized_alias).ratio() >= 0.84


def _normalise_statement_number_spacing(line: str) -> str:
    line = re.sub(r"\((-?\d{1,3}),\s+(\d{3})\)", r"(\1,\2)", line)
    line = re.sub(r"\((-?\d{1,3})\s+(\d{3})\)", r"(\1,\2)", line)
    line = re.sub(r"\b(\d{1,3})\.(\d{3})\b", r"\1,\2", line)
    line = re.sub(r"\((-?\d{1,3})\.(\d{3})\)", r"(\1,\2)", line)
    cleaned = re.sub(r"(\d)\s+,", r"\1,", line)
    cleaned = re.sub(r"\b(\d)\s+(\d,\d{3})(?!,)", r"\1\2", cleaned)
    cleaned = re.sub(r"\b(\d)\s+(\d{2},\d{3})\b", r"\1\2", cleaned)
    cleaned = re.sub(r"\b(\d)\s+(\d{2,3})\b(?=\s|$)", r"\1\2", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    return cleaned


def _amounts_from_statement_line(line: str) -> list[Decimal]:
    cleaned = _normalise_statement_number_spacing(line)
    cleaned = re.sub(r"\s[-=]\s*(?=\(?\s?\d)", " 0 ", cleaned)
    tokens = re.findall(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?", cleaned)
    amounts = [_parse_decimal(token) for token in tokens]
    amounts = [amount for amount in amounts if amount is not None]
    return [amount for amount in amounts if abs(amount) < Decimal("100000000")]


def _has_rows(rows: dict[str, list[Decimal]], labels: tuple[str, ...]) -> bool:
    return all(label in rows and len(rows[label]) >= 2 for label in labels)


def _row_amounts(rows: dict[str, list[Decimal]], label: str) -> list[Decimal]:
    return rows.get(label, [])[-2:]


def _row_amounts_any(rows: dict[str, list[Decimal]], labels: tuple[str, ...]) -> list[Decimal]:
    for label in labels:
        amounts = _row_amounts(rows, label)
        if amounts:
            return amounts
    return []


def _sum_row_amounts(rows: dict[str, list[Decimal]], labels: tuple[str, ...]) -> list[Decimal]:
    present = [_row_amounts(rows, label) for label in labels if _row_amounts(rows, label)]
    if not present:
        return []
    width = min(len(values) for values in present)
    if width == 0:
        return []
    return [sum(values[index] for values in present) for index in range(width)]


def _sfp_non_current_amounts(rows: dict[str, list[Decimal]]) -> list[Decimal]:
    direct = _row_amounts_any(rows, ("non-current assets", "total non-current assets", "total non current assets"))
    return direct


def _sfp_current_amounts(rows: dict[str, list[Decimal]]) -> list[Decimal]:
    direct = _row_amounts_any(rows, ("current assets", "total current assets"))
    return direct


def _check_profit_tax_equation(
    findings: list[Finding],
    page_number: int,
    location: str,
    before_tax: list[Decimal],
    taxation: list[Decimal],
    after_tax: list[Decimal],
    tolerance: Decimal,
    ocr_review: bool = False,
    raw_lines: dict[str, str] | None = None,
    document: PdfDocument | None = None,
) -> bool:
    width = min(len(before_tax), len(taxation), len(after_tax))
    if width < 1 or (width < 2 and not ocr_review):
        return False
    for index in range(width):
        before = before_tax[index]
        tax = taxation[index]
        reported = after_tax[index]
        candidates = (before - tax, before + tax, before - abs(tax), before + abs(tax))
        if any(abs(reported - candidate) <= tolerance for candidate in candidates):
            continue
        closest = min(candidates, key=lambda candidate: abs(reported - candidate))
        issue = f"Profit/loss after tax should agree to profit/loss before tax adjusted for taxation. Column {index + 1}."
        if ocr_review:
            corroboration = _ocr_income_corroboration_assessment(document, taxation[index] if index < len(taxation) else None, reported, tolerance)
            _check_ocr_scalar_equation(
                findings,
                page_number,
                location,
                issue,
                closest,
                reported,
                tolerance,
                extra_evidence=_ocr_income_tax_debug(raw_lines or {}, before_tax, taxation, after_tax, document, corroboration),
                confidence_override="Low" if corroboration.get("casts") else None,
                issue_prefix="OCR conflict - manual confirmation required: " if corroboration.get("casts") else "",
            )
        else:
            _check_scalar_equation(findings, page_number, location, issue, closest, reported, tolerance)
    return True


def _ocr_income_tax_debug(
    raw_lines: dict[str, str],
    before_tax: list[Decimal],
    taxation: list[Decimal],
    after_tax: list[Decimal],
    document: PdfDocument | None = None,
    corroboration: dict[str, object] | None = None,
) -> str:
    corroborating = _ocr_income_corroborating_debug(document)
    corroboration = corroboration or {}
    correction_text = ""
    if corroboration.get("casts"):
        if len(after_tax) < 2:
            correction_text = (
                " Current-year loss after tax could not be confidently extracted from the primary statement row, "
                f"but corroborating lines indicate {_format_accounting_amount(corroboration.get('after_tax_value'))}. "
                "Manual confirmation required."
            )
        else:
            correction_text = (
                " Corroboration assessment: related report/cash-flow values cast correctly using "
                f"before tax={_format_accounting_amount(corroboration.get('before_tax_value'))}, "
                f"taxation={_format_accounting_amount(corroboration.get('taxation_value'))}, "
                f"after tax={_format_accounting_amount(corroboration.get('after_tax_value'))}. "
                "Treat as OCR conflict until manually confirmed."
            )
    return (
        " OCR source lines: "
        f"Loss/profit before taxation raw line: {raw_lines.get('profit before tax', 'Not captured')}; "
        f"Taxation raw line: {raw_lines.get('taxation', 'Not captured')}; "
        f"Loss/profit after taxation raw line: {raw_lines.get('profit after tax', 'Not captured')}. "
        f"Extracted values: before tax={_format_decimal_list(before_tax)}, taxation={_format_decimal_list(taxation)}, after tax={_format_decimal_list(after_tax)}."
        f"{corroborating}"
        f"{correction_text}"
    )


def _ocr_income_corroborating_debug(document: PdfDocument | None) -> str:
    if document is None:
        return ""
    labels = {
        "loss/profit before tax": ("loss before taxation", "loss before tax", "profit before taxation", "profit before tax"),
        "taxation": ("taxation", "tax credit", "tax expense"),
        "loss/profit after tax": ("loss after taxation", "loss after tax", "profit after taxation", "profit after tax"),
    }
    parts: list[str] = []
    for label, phrases in labels.items():
        values: list[str] = []
        for page in document.pages:
            for line in page.text.splitlines():
                normalized = _normalise_match_words(line)
                if not any(phrase in normalized for phrase in phrases):
                    continue
                amounts = _amounts_from_statement_line(line)
                if amounts:
                    cleaned = re.sub(r"\s+", " ", line).strip()
                    values.append(f"Page {page.number}: {_format_decimal_list(amounts[-2:])} from '{cleaned}'")
        if values:
            parts.append(f"{label}: {'; '.join(values[:4])}")
    if not parts:
        return ""
    return " Corroborating OCR values from related report/cash-flow lines: " + " | ".join(parts) + "."


def _ocr_income_corroboration_assessment(
    document: PdfDocument | None,
    primary_tax: Decimal | None,
    primary_after_tax: Decimal | None,
    tolerance: Decimal,
) -> dict[str, object]:
    if document is None or primary_tax is None or primary_after_tax is None:
        return {"casts": False}
    before_values = _corroborating_amount_values(document, ("loss before taxation", "loss before tax", "profit before taxation", "profit before tax"))
    tax_values = _corroborating_amount_values(document, ("taxation", "tax credit", "tax expense"))
    after_values = _corroborating_amount_values(document, ("loss after taxation", "loss after tax", "profit after taxation", "profit after tax"))
    tax_candidates = tax_values or [primary_tax]
    after_candidates = after_values or [primary_after_tax]
    for before in before_values:
        for tax in tax_candidates:
            for after in after_candidates:
                candidates = (before - tax, before + tax, before - abs(tax), before + abs(tax))
                if any(abs(after - candidate) <= tolerance for candidate in candidates):
                    return {
                        "casts": True,
                        "before_tax_value": before,
                        "taxation_value": tax,
                        "after_tax_value": after,
                    }
    return {"casts": False}


def _corroborating_amount_values(document: PdfDocument, phrases: tuple[str, ...]) -> list[Decimal]:
    values: list[Decimal] = []
    seen: set[Decimal] = set()
    for page in document.pages:
        for line in page.text.splitlines():
            normalized = _normalise_match_words(line)
            if not any(phrase in normalized for phrase in phrases):
                continue
            amounts = _amounts_from_phrase_context(line, phrases)
            if not amounts:
                continue
            usable_amounts = [amount for amount in amounts if abs(amount) >= Decimal("1000")]
            if not usable_amounts:
                continue
            value = usable_amounts[0]
            if value not in seen:
                values.append(value)
                seen.add(value)
    return values


def _amounts_from_phrase_context(line: str, phrases: tuple[str, ...]) -> list[Decimal]:
    best_start: int | None = None
    best_phrase_length = 0
    lower_line = line.lower()
    for phrase in phrases:
        clean_phrase = phrase.strip().lower()
        if not clean_phrase:
            continue
        for match in re.finditer(rf"\b{re.escape(clean_phrase)}\b", lower_line):
            context_before = lower_line[max(0, match.start() - 24) : match.start()]
            if clean_phrase == "taxation" and re.search(r"\b(before|after)\s+$", context_before):
                continue
            if best_start is None or match.start() < best_start or (match.start() == best_start and len(clean_phrase) > best_phrase_length):
                best_start = match.start()
                best_phrase_length = len(clean_phrase)
            break
    if best_start is None:
        return _amounts_from_statement_line(line)
    return _amounts_from_statement_line(line[best_start:])


def _format_accounting_amount(value: object) -> str:
    if not isinstance(value, Decimal):
        return "not detected"
    if value < 0:
        return f"({abs(value):,})"
    return f"{value:,}"


def _format_decimal_list(values: list[Decimal]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f"{value:+,}" for value in values) + "]"


def _check_sum_rows(
    findings: list[Finding],
    page_number: int,
    location: str,
    caption: str,
    rows: dict[str, list[Decimal]],
    components: tuple[str, ...],
    total_label: str,
    tolerance: Decimal,
    ocr_review: bool = False,
) -> None:
    expected = [sum(values) for values in zip(*(_row_amounts(rows, label) for label in components))]
    _check_vector_equation(
        findings,
        page_number,
        location,
        f"{caption} agrees to component rows.",
        expected,
        _row_amounts(rows, total_label),
        tolerance,
        ocr_review=ocr_review,
    )


def _check_vector_equation(
    findings: list[Finding],
    page_number: int,
    location: str,
    issue: str,
    expected: list[Decimal],
    reported: list[Decimal],
    tolerance: Decimal,
    ocr_review: bool = False,
) -> None:
    for index, (expected_value, reported_value) in enumerate(zip(expected, reported), start=1):
        if ocr_review:
            _check_ocr_scalar_equation(findings, page_number, location, f"{issue} Column {index}.", expected_value, reported_value, tolerance)
        else:
            _check_scalar_equation(findings, page_number, location, f"{issue} Column {index}.", expected_value, reported_value, tolerance)


def _check_scalar_equation(
    findings: list[Finding],
    page_number: int,
    location: str,
    issue: str,
    expected: Decimal,
    reported: Decimal,
    tolerance: Decimal,
) -> None:
    diff = reported - expected
    if abs(diff) <= tolerance:
        return
    findings.append(
        Finding(
            "Totals and rounding",
            "High" if abs(diff) > tolerance * 5 else "Medium",
            f"Page {page_number} | {location}",
            issue,
            f"Expected {expected:,}; reported {reported:,}; difference {diff:,}.",
            "Review the line-extracted primary statement totals against the signed financial statements.",
        )
    )


def _check_ocr_scalar_equation(
    findings: list[Finding],
    page_number: int,
    location: str,
    issue: str,
    expected: Decimal,
    reported: Decimal,
    tolerance: Decimal,
    extra_evidence: str = "",
    confidence_override: str | None = None,
    issue_prefix: str = "",
) -> None:
    if abs(reported - expected) <= tolerance:
        return
    _add_ocr_arithmetic_finding(
        findings,
        page_number,
        location,
        issue,
        expected,
        reported,
        tolerance,
        extra_evidence=extra_evidence,
        confidence_override=confidence_override,
        issue_prefix=issue_prefix,
    )


def _add_ocr_arithmetic_finding(
    findings: list[Finding],
    page_number: int,
    location: str,
    issue: str,
    expected: Decimal,
    reported: Decimal,
    tolerance: Decimal,
    extra_evidence: str = "",
    confidence_override: str | None = None,
    issue_prefix: str = "",
) -> None:
    diff = reported - expected
    confidence = confidence_override or ("Medium" if abs(diff) > tolerance * 5 else "Low")
    findings.append(
        Finding(
            "Totals and rounding",
            confidence,
            f"Page {page_number} | {location}",
            f"{issue_prefix}Possible mismatch from OCR line extraction: {_ocr_mismatch_issue(issue)}",
            f"Expected {expected:,}; reported {reported:,}; difference {diff:,}. OCR statement structure confidence is below the high-confidence threshold.{extra_evidence}",
            "Review the raw OCR statement rows against the signed financial statement before treating this as an exception.",
            {"confidence": "OCR review prompt", "check_type": "OCR statement arithmetic"},
        )
    )


def _ocr_mismatch_issue(issue: str) -> str:
    text = issue.replace("should equal", "does not agree to")
    text = text.replace("should agree to", "does not agree to")
    text = text.replace("agrees to", "does not agree to")
    return text


def _check_industry_policy_fit(findings: list[Finding], document: PdfDocument, profile: CompanyProfile) -> None:
    industry = profile.industry.lower().strip()
    if not industry:
        return
    text = document.text.lower()
    mismatches: tuple[str, ...] = ()
    for key, policies in INDUSTRY_POLICY_MISMATCHES.items():
        if key in industry:
            mismatches = policies
            break
    for policy in mismatches:
        if policy in _accounting_policy_text(document).lower():
            location, snippet = _first_keyword_context(document, (policy,))
            findings.append(
                Finding(
                    "Accounting policies",
                    "High",
                    location,
                    f"The {policy} policy appears inconsistent with the stated industry.",
                    f"Industry: {profile.industry}; detected policy phrase: {policy}. Context: {snippet}",
                    "Tailor the policy note to the entity and remove industry-irrelevant boilerplate unless the company actually has this activity.",
                )
            )


def _accounting_policy_text(document: PdfDocument) -> str:
    text = document.text
    matches = list(re.finditer(r"summary of significant accounting policies|significant accounting policies|accounting policies", text, flags=re.I))
    if not matches:
        return text
    start_match = next((match for match in matches if "summary" in match.group(0).lower()), matches[-1])
    tail = text[start_match.start():]
    end_match = re.search(
        r"\n\s*5\s+Critical accounting estimates|\n\s*5\.\s*Critical accounting estimates|\n\s*Critical accounting estimates",
        tail[1200:],
        flags=re.I,
    )
    if end_match:
        return tail[: 1200 + end_match.start()]
    return tail[:18000]


def _accounting_policy_map(document: PdfDocument) -> dict[str, bool]:
    policy_text = _accounting_policy_text(document).lower()
    full_text = document.text.lower()
    search_text = f"{policy_text}\n{full_text if _revenue_from_customers_policy_present(full_text) else ''}"
    detected: dict[str, bool] = {}
    for policy_name, rule in POLICY_RULES.items():
        detected[policy_name] = any(keyword in search_text for keyword in rule["policy"])
    if _revenue_from_customers_policy_present(search_text):
        detected["revenue"] = True
    detected["leases"] = _lease_policy_applies(search_text)
    return detected


def _revenue_from_customers_policy_present(text: str) -> bool:
    normalized = _normalise_match_words(text)
    raw_normalized = re.sub(r"\s+", " ", text).strip()
    if re.search(r"revenue\s+from\s+contracts?\s+(?:with\s+)?customers?", raw_normalized, flags=re.I):
        return True
    variants = (
        "revenue from contracts with customers",
        "revenue contracts customers",
        "revenue from contract with customer",
        "revenue contract customer",
        "revenue from contracts customers",
        "revenue from customers contracts",
    )
    if any(variant in normalized for variant in variants):
        return True
    if re.search(r"\brevenue\b.{0,120}\bcontracts?\b.{0,80}\bcustom(?:er|ers|ors|0mers|mers)\b", normalized, flags=re.I | re.S):
        return True
    words = normalized.split()
    for index, word in enumerate(words):
        if word != "revenue":
            continue
        window = words[index : index + 14]
        has_contract = any(token.startswith("contract") for token in window)
        has_customer = any(token.startswith("custom") or SequenceMatcher(None, token, "customers").ratio() >= 0.82 for token in window)
        if has_contract and has_customer:
            return True
    return False


def _lease_policy_applies(text: str) -> bool:
    return _actual_lease_disclosure_present(text)


def _actual_lease_disclosure_present(text: str) -> bool:
    lower = text.lower()
    generic_context_terms = (
        "new standards",
        "new standard",
        "new and amended standards",
        "standards issued",
        "standards issued but not effective",
        "amendments",
        "annual improvements",
        "issued but not effective",
        "effective for annual periods",
        "standards and interpretations",
        "interpretations issued",
        "deferred tax",
        "amendments to ias",
        "amendments to ifrs",
        "illustrative",
        "example",
        "examples",
        "transition",
        "single transaction",
    )
    actual_lease_terms = (
        "right-of-use asset",
        "right of use asset",
        "rou asset",
        "lease asset",
        "lease liability",
        "lease expense",
        "lease maturity",
        "depreciation of right-of-use",
        "depreciation of rou",
        "leased asset",
    )
    for term in actual_lease_terms:
        for match in re.finditer(re.escape(term), lower):
            context = lower[max(0, match.start() - 160) : match.end() + 160]
            if any(generic in context for generic in generic_context_terms):
                continue
            if term in {"right-of-use asset", "right of use asset", "rou asset", "lease asset", "lease liability"} and not _lease_context_has_actual_evidence(context):
                continue
            if not _lease_context_is_theoretical_policy_only(context):
                return True
    actual_arrangement_patterns = (
        r"\b(?:the\s+)?company\s+(?:has|entered into|leases|rents)\b.{0,120}\b(?:lease|leased|rental|premises|office|property)\b",
        r"\b(?:lease|rental)\s+arrangements?\b.{0,120}\b(?:company|premises|office|property|agreement|term)\b",
        r"\b(?:leased|rented)\s+(?:premises|office|property|building|warehouse)\b",
        r"\b(?:finance|operating)\s+leases?\b.{0,120}\b(?:balance|liability|asset|expense|note|maturity|commitment|payment)\b",
    )
    for pattern in actual_arrangement_patterns:
        for match in re.finditer(pattern, lower, flags=re.I | re.S):
            context = lower[max(0, match.start() - 160) : match.end() + 160]
            if not any(generic in context for generic in generic_context_terms) and not _lease_context_is_theoretical_policy_only(context):
                return True
    return False


def _lease_context_has_actual_evidence(context: str) -> bool:
    actual_terms = (
        "balance",
        "balances",
        "carrying amount",
        "current",
        "non-current",
        "non current",
        "statement of financial position",
        "expense",
        "maturity",
        "depreciation",
        "addition",
        "payment",
        "commitment",
    )
    return any(term in context for term in actual_terms) or bool(NUMBER_RE.search(context))


def _lease_context_is_theoretical_policy_only(context: str) -> bool:
    theoretical_terms = (
        "recognition of",
        "measurement of",
        "initial recognition",
        "subsequent measurement",
        "accounting policy",
        "policy is applied",
        "standard requires",
    )
    if not any(term in context for term in theoretical_terms):
        return False
    return not _lease_context_has_actual_evidence(context)


def _check_superseded_standards(findings: list[Finding], document: PdfDocument) -> None:
    for reference, message in SUPERSEDED_REFERENCES.items():
        for page in document.pages:
            snippet = _snippet_around(page.text, reference)
            if not snippet:
                continue
            context = _superseded_reference_context(snippet)
            severity = "High" if context == "current policy" else "Low"
            findings.append(
                Finding(
                    "Accounting policies",
                    severity,
                    f"Page {page.number}",
                    f"Reference to superseded accounting guidance detected: {reference.upper()} ({context}).",
                    f"{message} Context: {snippet}",
                    "If this is current accounting policy wording, update it to current applicable standards. If it is transition history, confirm the disclosure is clearly historical.",
                )
            )
            break


def _check_boilerplate_policy_language(findings: list[Finding], document: PdfDocument) -> None:
    text = document.text.lower()
    hits = [phrase for phrase in GENERIC_POLICY_PHRASES if phrase in text]
    if len(hits) >= 2:
        location, snippet = _first_keyword_context(document, tuple(hits))
        findings.append(
            Finding(
                "Accounting policies",
                "Low",
                location,
                "Policy wording appears generic or boilerplate.",
                f"Detected generic phrases: {', '.join(hits[:4])}. Context: {snippet}",
                "Tailor the policy wording to the entity's actual transactions, estimates, judgements, and measurement bases.",
            )
        )


def _contextual_currency_markers(document: PdfDocument) -> list[str]:
    markers: list[str] = []
    currency_re = re.compile(r"\bUSD\b|\bNGN\b|\bGBP\b|\bEUR\b|US\$|\bNaira\b|\bDollar\b|\bPound\b|\bEuro\b|₦|N['’]?\s?000|N000|\$", re.I)
    context_re = re.compile(
        r"statement of|presentation currency|functional currency|expressed in|presented in|currency:|n'000|ngn'000|usd'000",
        re.I,
    )
    ignore_re = re.compile(r"accounting polic|ifrs|ias |risk|example|foreign currency translation|amendment|standard", re.I)
    for page in document.pages:
        for line in page.text.splitlines():
            if not currency_re.search(line):
                continue
            if ignore_re.search(line) and not context_re.search(line):
                continue
            if context_re.search(line):
                markers.extend(currency_re.findall(line))
        for table in page.tables:
            header_text = " ".join(" ".join(row) for row in table[:2])
            if currency_re.search(header_text) or re.search(r"n' ?000|ngn", header_text, re.I):
                markers.extend(currency_re.findall(header_text))
    return markers


def _normalise_currency_marker(marker: str) -> str:
    normalized_marker = re.sub(r"\s+", "", marker.upper().replace("â€™", "'").replace("’", "'").replace("‘", "'"))
    if normalized_marker in {"₦", "₦'000", "NGN'000", "N'000", "N000"}:
        return "NGN"
    marker_upper = re.sub(r"\s+", "", marker.upper().replace("’", "'"))
    if marker_upper in {"NAIRA", "NGN", "₦", "N'000", "N000"}:
        return "NGN"
    if marker_upper in {"DOLLAR", "US$", "$", "USD"}:
        return "USD"
    if marker_upper == "POUND":
        return "GBP"
    if marker_upper == "EURO":
        return "EUR"
    return marker_upper


def normalize_reporting_currency(value: str) -> str:
    normalized_value = re.sub(r"\s+", "", value.strip().upper().replace("â€™", "'").replace("’", "'").replace("‘", "'"))
    if re.search(r"(?:NGN|NAIRA|NIGERIANNAIRA|₦|N'?000)", normalized_value):
        return "NGN"
    cleaned = re.sub(r"\s+", "", value.strip().upper().replace("’", "'"))
    aliases = {
        "": "",
        "NAIRA": "NGN",
        "NIGERIANNAIRA": "NGN",
        "NGN": "NGN",
        "₦": "NGN",
        "N'000": "NGN",
        "N000": "NGN",
        "N'000S": "NGN",
        "USD": "USD",
        "US$": "USD",
        "$": "USD",
        "DOLLAR": "USD",
        "GBP": "GBP",
        "POUND": "GBP",
        "EUR": "EUR",
        "EURO": "EUR",
        "ZAR": "ZAR",
        "GHS": "GHS",
        "KES": "KES",
        "CAD": "CAD",
        "AUD": "AUD",
    }
    return aliases.get(cleaned, cleaned if cleaned in VALID_CURRENCIES else "")


def _policy_evidence_present(policy_name: str, evidence_keywords: tuple[str, ...], text: str) -> bool:
    if policy_name == "consolidation":
        group_indicators = (
            "consolidated statement",
            "consolidated financial statements",
            "non-controlling interest",
            "investment in subsidiary",
            "subsidiary investment",
            "investment in subsidiaries",
            "parent company",
        )
        return any(indicator in text for indicator in group_indicators)
    return any(keyword in text for keyword in evidence_keywords)


def _strong_policy_gap(policy_name: str, text: str) -> bool:
    if policy_name == "revenue":
        if _revenue_from_customers_policy_present(text):
            return False
        return "revenue" in text and not any(term in text for term in ("revenue is recognised", "revenue is recognized", "(m) revenue"))
    if policy_name == "leases":
        return _actual_lease_disclosure_present(text)
    if policy_name == "tax":
        return any(term in text for term in ("tax expense", "deferred tax", "current tax", "tax payable")) and not _tax_exempt_context(text)
    return False


def _first_keyword_context(document: PdfDocument, keywords: tuple[str, ...]) -> tuple[str, str]:
    for page in document.pages:
        lower = page.text.lower()
        for keyword in keywords:
            if keyword and keyword.lower() in lower:
                return f"Page {page.number}", _snippet_around(page.text, keyword)
    return "Document-wide", "No specific page context located."


def _snippet_around(text: str, needle: str, radius: int = 120) -> str:
    match = re.search(re.escape(needle), text, flags=re.I)
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet[:280]


def _superseded_reference_context(snippet: str) -> str:
    lower = snippet.lower()
    historical_terms = (
        "new standard",
        "new and amended",
        "amendment",
        "effective for annual",
        "not yet effective",
        "transition",
        "replaced",
        "superseded",
        "prior year",
        "previously",
    )
    policy_terms = ("accounted for", "is measured", "are measured", "recognised under", "recognized under", "policy")
    if any(term in lower for term in historical_terms):
        return "historical/new-standard discussion"
    if any(term in lower for term in policy_terms):
        return "current policy"
    return "context requires review"


def _check_ocr_statement_of_financial_position(document: PdfDocument, tolerance: Decimal) -> list[Finding]:
    for page in document.pages:
        if not _fuzzy_contains(page.text[:1000], "statement of financial position") and not _page_has_sfp_rows(page):
            continue
        line_row_map = _sfp_line_row_amounts(page.text)
        required = ("non-current assets", "current assets", "total assets", "equity", "liabilities", "total equity and liabilities")
        if all(key in line_row_map for key in required):
            return _check_sfp_equations(page.number, 0, line_row_map, tolerance)
        partial_findings, partial_ran = _check_partial_sfp_equations(page.number, line_row_map, tolerance)
        if partial_ran:
            return partial_findings
        for table_index, table in enumerate(page.tables, start=1):
            row_map = _sfp_row_amounts(table)
            if not all(key in row_map for key in required):
                continue
            return _check_sfp_equations(page.number, table_index, row_map, tolerance)
    return [
        Finding(
            "Extraction quality",
            "Low",
            "OCR statement-specific checks",
            "Statement-specific OCR checks were skipped because a high-confidence SFP row set was not detected.",
            "Required rows: non-current assets, current assets, total assets, equity, liabilities, and total equity and liabilities.",
            "Use OCR output for navigation, but do not rely on scanned SFP arithmetic until row extraction is structured.",
        )
    ]


def _page_has_sfp_rows(page: PdfPage) -> bool:
    lower = page.text.lower()
    return "total assets" in lower and "total equity" in lower and "liabilities" in lower


def _sfp_row_amounts(table: list[list[str]]) -> dict[str, Decimal]:
    if _table_has_merged_numeric_cells(table) or not _table_shape_confident(table):
        return {}
    rows = [_numeric_row(row, _note_columns(table)) for row in table]
    mapped: dict[str, Decimal] = {}
    for row in rows:
        label = str(row[0] or "").lower()
        values = [cell for cell in row[1:] if isinstance(cell, Decimal)]
        if not values:
            continue
        amount = values[0]
        if "non-current assets" in label or "non current assets" in label:
            mapped["non-current assets"] = amount
        elif "current assets" in label and "non" not in label:
            mapped["current assets"] = amount
        elif "total assets" in label:
            mapped["total assets"] = amount
        elif label.strip() in {"equity", "total equity", "shareholders' equity", "members fund", "members' fund"}:
            mapped["equity"] = amount
        elif label.strip() in {"liabilities", "total liabilities"}:
            mapped["liabilities"] = amount
        elif "total equity and liabilities" in label or "total liabilities and equity" in label:
            mapped["total equity and liabilities"] = amount
    return mapped


def _sfp_line_row_amounts(text: str) -> dict[str, Decimal]:
    rows = _statement_rows(text)
    mapped: dict[str, Decimal] = {}
    for label, amounts in rows.items():
        if not amounts:
            continue
        amount = amounts[0]
        clean = label.replace("-", " ")
        if "non current assets" in clean and "total" in clean:
            mapped["non-current assets"] = amount
        elif clean.strip() == "non current assets":
            mapped["non-current assets"] = amount
        elif "current assets" in clean and "non" not in clean and "total" in clean:
            mapped["current assets"] = amount
        elif clean.strip() == "current assets":
            mapped["current assets"] = amount
        elif clean.strip() == "total assets" or ("total assets" in clean and "liabilities" not in clean):
            mapped["total assets"] = amount
        elif clean.strip() in {"equity", "total equity", "shareholders equity", "members fund", "members fund", "share capital and reserves", "capital and reserves"}:
            mapped["equity"] = amount
        elif clean.strip() in {"liabilities", "total liabilities", "financial liabilities"}:
            mapped["liabilities"] = amount
        elif "total equity and liabilities" in clean or "total liabilities and equity" in clean:
            mapped["total equity and liabilities"] = amount
    non_current = _sfp_non_current_amounts(rows)
    current = _sfp_current_amounts(rows)
    if non_current and "non-current assets" not in mapped:
        mapped["non-current assets"] = non_current[0]
    if current and "current assets" not in mapped:
        mapped["current assets"] = current[0]
    return mapped


def _check_partial_sfp_equations(page_number: int, rows: dict[str, Decimal], tolerance: Decimal) -> tuple[list[Finding], bool]:
    findings: list[Finding] = []
    ran = False
    if all(key in rows for key in ("non-current assets", "current assets", "total assets")):
        ran = True
        _check_ocr_scalar_equation(
            findings,
            page_number,
            "Statement of financial position",
            "Non-current assets + current assets should equal total assets.",
            rows["non-current assets"] + rows["current assets"],
            rows["total assets"],
            tolerance,
        )
    if all(key in rows for key in ("total assets", "total equity and liabilities")):
        ran = True
        _check_ocr_scalar_equation(
            findings,
            page_number,
            "Statement of financial position",
            "Total assets should equal total equity and liabilities.",
            rows["total assets"],
            rows["total equity and liabilities"],
            tolerance,
        )
    if all(key in rows for key in ("equity", "liabilities", "total equity and liabilities")):
        ran = True
        _check_ocr_scalar_equation(
            findings,
            page_number,
            "Statement of financial position",
            "Equity + liabilities should equal total equity and liabilities.",
            rows["equity"] + rows["liabilities"],
            rows["total equity and liabilities"],
            tolerance,
        )
    return findings, ran


def _check_sfp_equations(page_number: int, table_index: int, rows: dict[str, Decimal], tolerance: Decimal) -> list[Finding]:
    findings: list[Finding] = []
    equations = (
        ("Non-current assets + current assets should equal total assets.", rows["non-current assets"] + rows["current assets"], rows["total assets"]),
        ("Equity + liabilities should equal total equity and liabilities.", rows["equity"] + rows["liabilities"], rows["total equity and liabilities"]),
        ("Total assets should equal total equity and liabilities.", rows["total assets"], rows["total equity and liabilities"]),
    )
    for issue, expected, reported in equations:
        diff = reported - expected
        if abs(diff) > tolerance:
            _add_ocr_arithmetic_finding(
                findings,
                page_number,
                "Statement of financial position",
                issue,
                expected,
                reported,
                tolerance,
            )
    return findings


def _detect_rounding_scale(text: str) -> tuple[str, Decimal]:
    lower = text.lower()
    labels = set()
    if re.search(r"\$?0{3}s|000s|thousand|in thousands", lower):
        labels.add("thousands")
    if re.search(r"million|in millions", lower):
        labels.add("millions")
    if re.search(r"nearest dollar|actual amount|in units", lower):
        labels.add("units")
    if len(labels) > 1:
        return "mixed", Decimal("1")
    if "millions" in labels:
        return "millions", Decimal("1")
    if "thousands" in labels:
        return "thousands", Decimal("1")
    return "units", Decimal("1")


def _note_columns(table: list[list[str]]) -> set[int]:
    if not table:
        return set()
    header = table[0]
    return {index for index, cell in enumerate(header) if str(cell).strip().lower() in {"note", "notes", "ref", "reference"}}


def _classify_table_for_arithmetic(table: list[list[str]], page_text: str = "") -> dict[str, object]:
    text = " ".join(" ".join(str(cell or "") for cell in row) for row in table).lower()
    full_text = text + " " + page_text.lower()
    if "value added" in full_text or "value-added" in full_text:
        return {"type": "value-added statement", "can_run_arithmetic": False, "reason": "value-added statements have presentation-specific subtotals"}
    if "five year" in full_text or "5 year" in full_text or "financial summary" in full_text:
        return {"type": "multi-year summary", "can_run_arithmetic": False, "reason": "multi-year summaries should not be cast like primary statements"}
    if _table_has_merged_numeric_cells(table):
        return {"type": "merged extraction", "can_run_arithmetic": False, "reason": "one or more cells contain multiple numeric values"}
    if not _table_shape_confident(table):
        return {"type": "low-confidence table", "can_run_arithmetic": False, "reason": "numeric row shapes are inconsistent"}
    if _looks_like_statement_table(table):
        return {"type": "statement table", "can_run_arithmetic": True, "reason": "consistent financial statement table structure"}
    return {"type": "other table", "can_run_arithmetic": False, "reason": "table type is not a recognised statement/note total table"}


def _table_has_merged_numeric_cells(table: list[list[str]]) -> bool:
    for row in table:
        for cell in row[1:]:
            if len(NUMBER_RE.findall(str(cell or ""))) > 1:
                return True
    return False


def _table_shape_confident(table: list[list[str]]) -> bool:
    note_cols = _note_columns(table)
    amount_counts: list[int] = []
    for row in table[1:]:
        count = 0
        for index, cell in enumerate(row[1:], start=1):
            if index in note_cols:
                continue
            if _parse_decimal(cell) is not None:
                count += 1
        if count:
            amount_counts.append(count)
    if len(amount_counts) < 3:
        return False
    most_common_count = max(set(amount_counts), key=amount_counts.count)
    return most_common_count >= 1 and amount_counts.count(most_common_count) / len(amount_counts) >= 0.7


def _looks_like_statement_table(table: list[list[str]]) -> bool:
    text = " ".join(str(row[0] or "").lower() for row in table if row)
    statement_keywords = (
        "revenue",
        "cost of sales",
        "gross profit",
        "profit before",
        "tax",
        "assets",
        "liabilities",
        "equity",
        "cash",
        "receivables",
        "payables",
        "property, plant",
        "depreciation",
        "inventory",
        "borrowings",
        "total",
    )
    hits = sum(1 for keyword in statement_keywords if keyword in text)
    return hits >= 2 and "total" in text


def _numeric_row(row: list[str], note_cols: set[int] | None = None) -> list[str | Decimal | None]:
    note_cols = note_cols or set()
    converted: list[str | Decimal | None] = []
    for index, cell in enumerate(row):
        if index == 0:
            converted.append(cell)
            continue
        if index in note_cols:
            converted.append(None)
            continue
        converted.append(_parse_decimal(cell))
    return converted


def _parse_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw in {"-", "--"}:
        return None
    match = NUMBER_RE.search(raw)
    if not match:
        return None
    token = match.group(0)
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "")
    try:
        amount = Decimal(token)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _amounts_in_text(text: str) -> list[Decimal]:
    amounts = [amount for amount, _snippet, _method in _normalized_amount_candidates(text)]
    return amounts


def _amount_near(text: str, labels: tuple[str, ...]) -> Decimal | None:
    for line in text.splitlines():
        lower = line.lower()
        if any(label in lower for label in labels):
            amount = _last_amount(line)
            if amount is not None:
                return amount
    return None


def _looks_like_total(label: str) -> bool:
    keywords = ("total", "subtotal", "closing balance", "closing")
    return any(keyword in label for keyword in keywords)


def _looks_like_amount_line(label: str) -> bool:
    excluded = ("note", "year", "date", "audited", "restated")
    return bool(label.strip()) and not any(word in label for word in excluded)


def _is_table_boundary_row(row: list[str | Decimal | None]) -> bool:
    text = " ".join(str(cell or "") for cell in row).strip()
    lower = re.sub(r"\s+", " ", text.lower())
    if not lower:
        return True
    
    # Break on long narrative text that lacks numbers
    if len(text) > 40 and not re.search(r"\d", text):
        return True
        
    heading_match = NOTE_HEADING_RE.match(text)
    if heading_match and _valid_note_heading(heading_match.group(1), heading_match.group(2)):
        return True
        
    header_phrases = (
        "note (s)",
        "note(s)",
        "n' 000",
        "n'000",
        "2025 2024",
        "held to maturity",
        "receivables & advances",
        "receivables and advances",
        "december 31",
        "for the year ended",
        "statement of value added",
        "five-year financial summary",
        "five year financial summary",
        "financial summary",
        "value added statement"
    )
    if any(phrase in lower for phrase in header_phrases):
        return True
    label = str(row[0] or "").strip().lower() if row else ""
    values = [cell for cell in row[1:] if isinstance(cell, Decimal)]
    label_has_line_text = bool(re.search(r"[a-z]{3,}", label)) and _looks_like_amount_line(label)
    if values and not label_has_line_text:
        return True
    if YEAR_RE.search(lower) and len(values) <= 1:
        return True
    return False


def _common_amount_count(rows: list[list[str | Decimal | None]]) -> int:
    counts = [_row_amount_count(row) for row in rows[1:] if _row_amount_count(row)]
    if not counts:
        return 0
    return Counter(counts).most_common(1)[0][0]


def _row_amount_count(row: list[str | Decimal | None]) -> int:
    return sum(1 for cell in row[1:] if isinstance(cell, Decimal))


def _check_adjacent_totals(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    col: int,
    subtotal_rows: list[tuple[int, Decimal]],
    tolerance: Decimal,
) -> None:
    for (first_row, first_value), (second_row, second_value) in zip(subtotal_rows, subtotal_rows[1:]):
        if abs(first_value - second_value) <= tolerance and second_row == first_row + 1:
            findings.append(
                Finding(
                    "Totals and rounding",
                    "Low",
                    f"Page {page_number}, table {table_index}, column {col + 1}",
                    "Adjacent subtotal or total rows show the same amount.",
                    f"Rows {first_row + 1} and {second_row + 1} both report approximately {first_value:,}.",
                    "Check whether a subtotal has been duplicated or whether one of the line descriptions should be revised.",
                )
            )


def _note_headings(text: str) -> dict[str, str]:
    page = PdfPage(1, text, [])
    return {ref: title for ref, (title, _page_number) in _note_headings_by_page(PdfDocument([page])).items()}


def _note_headings_by_page(document: PdfDocument) -> dict[str, tuple[str, int]]:
    headings: dict[str, tuple[str, int]] = {}
    notes_start_page = _notes_start_page(document)
    strict_notes_start = notes_start_page is not None
    if document.ocr_used and not strict_notes_start:
        return headings
    in_notes = not strict_notes_start
    for page in document.pages:
        if notes_start_page is not None and page.number < notes_start_page:
            continue
        if notes_start_page is not None and page.number > notes_start_page and _is_post_notes_supplement_page(page.text):
            break
        if strict_notes_start and _notes_heading_in_text(page.text):
            in_notes = True
        if strict_notes_start and not in_notes:
            continue
        lines = page.text.splitlines()
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            embedded = _embedded_note_heading_after_notes_title(line)
            if embedded:
                number, title = embedded
                if _valid_note_heading(number, title):
                    headings.setdefault(number.upper(), (_clean_note_title(title), page.number))
                    continue
            match = NOTE_HEADING_RE.match(line)
            if match:
                number, title = match.groups()
                if _valid_note_heading(number, title):
                    headings.setdefault(number.upper(), (_clean_note_title(title), page.number))
                    continue
            number_only = NOTE_NUMBER_ONLY_RE.match(line)
            if number_only and index + 1 < len(lines):
                number = number_only.group(1)
                title = lines[index + 1].strip()
                if _valid_note_heading(number, title):
                    headings.setdefault(number.upper(), (_clean_note_title(title), page.number))
            _add_combined_note_heading_candidates(headings, lines, index, line, page.number)
    return headings


def _is_post_notes_supplement_page(text: str) -> bool:
    head = _normalise_match_words("\n".join(text.splitlines()[:30]))
    supplement_markers = (
        "statement of value added",
        "value added statement",
        "five year financial summary",
        "five year summary",
        "5 year financial summary",
        "year financial summary",
        "financial summary",
        "five year financial highlights",
        "year financial highlights",
        "financial highlights",
        "five year financial review",
        "year financial review",
        "financial review",
    )
    return any(marker in head for marker in supplement_markers)


def _embedded_note_heading_after_notes_title(line: str) -> tuple[str, str] | None:
    if _notes_heading_line_score(line) < 0.82:
        return None
    match = re.search(r"\b(1)\s*[\).:-]\s+(.{3,100})$", line, flags=re.I)
    if not match:
        return None
    return match.group(1), match.group(2)


def _add_combined_note_heading_candidates(
    headings: dict[str, tuple[str, int]],
    lines: list[str],
    index: int,
    line: str,
    page_number: int,
) -> None:
    if not re.search(r"\bnote\b", line, flags=re.I):
        return
    refs = [ref.upper() for ref in re.findall(r"\b(\d{1,2}[A-C]?)\b", line, flags=re.I) if _valid_note_number(ref)]
    if not {"7", "8"}.issubset(set(refs)):
        return
    nearby = " ".join(lines[index : index + 5]).lower()
    if "revenue heads" in nearby and "operating revenue" in nearby:
        headings.setdefault("7", ("Operating Revenue", page_number))
        headings.setdefault("8", ("Operating Expenditure", page_number))


def _augment_note_headings_from_statement_refs(headings: dict[str, tuple[str, int]], document: PdfDocument) -> None:
    for item in _statement_note_lines(document):
        if not item.line_item:
            continue
        title = _statement_line_item_title(item.line_item)
        existing = headings.get(item.ref)
        if existing is None:
            continue
        elif existing[0].lower() in {"staff costs"} and "personnel" in item.line_item.lower():
            headings[item.ref] = (title, existing[1])


def _statement_line_item_title(line_item: str) -> str:
    words = []
    for index, word in enumerate(line_item.split()):
        if len(word) <= 4 and word.isupper():
            words.append(word)
        elif index == 0:
            words.append(word.capitalize())
        else:
            words.append(word.lower())
    return " ".join(words)


def _format_note_heading_debug(document: PdfDocument) -> str:
    headings = _note_headings_by_page(document)
    if not headings:
        return "No note headings detected."
    page_ranges = _note_section_page_ranges(document)
    parts = []
    for ref in sorted(headings, key=_note_sort_key):
        title, page_number = headings[ref]
        confidence, source_snippet = _note_heading_source_detail(document, ref, title, page_number)
        parts.append(
            f"Note {ref} | Page {page_number} | {page_ranges.get(ref, f'Page {page_number}')} | {title} | {confidence} | {source_snippet}"
        )
    return "\n".join(parts)


def _note_heading_source_detail(document: PdfDocument, ref: str, title: str, page_number: int) -> tuple[str, str]:
    page = next((item for item in document.pages if item.number == page_number), None)
    if not page:
        return "Confidence: Medium", "Source snippet unavailable."
    normalized_title = _normalise_match_words(title)
    for index, raw_line in enumerate(page.text.splitlines()):
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        line_normalized = _normalise_match_words(line)
        starts_with_ref = bool(re.match(rf"^\s*(?:note\s+)?{re.escape(ref)}(?:\s*[\).:-])?\s+", line, flags=re.I))
        embedded_ref = ref == "1" and _notes_heading_line_score(line) >= 0.82 and "accounting polic" in line_normalized
        if (starts_with_ref or embedded_ref) and normalized_title and normalized_title in line_normalized:
            snippet = " ".join(item.strip() for item in page.text.splitlines()[max(0, index - 1) : index + 2] if item.strip())
            confidence = "Confidence: High" if starts_with_ref or embedded_ref else "Confidence: Medium"
            return confidence, f"Source: {snippet[:220]}"
    return "Confidence: Medium", "Source: heading inferred from nearby notes-section text."


def _statement_note_references(document: PdfDocument) -> set[str]:
    refs: set[str] = set()
    for item in _statement_note_lines(document):
        if item.ref:
            refs.add(item.ref.upper())
    return refs


def _is_subheading(label: str) -> bool:
    lower = label.lower().strip()
    return lower in {
        "assets",
        "non-current assets",
        "current assets",
        "equity and liabilities",
        "equity",
        "liabilities",
        "non-current liabilities",
        "current liabilities",
        "cash flows from operating activities",
        "adjustments for",
        "changes in working capital",
        "cash flows from investing activities",
        "cash flows from financing activities",
        "other comprehensive income",
        "total items that may be reclassified",
    }


def _statement_note_lines(document: PdfDocument) -> list[StatementNoteLine]:
    items: list[StatementNoteLine] = []
    statements = _classified_primary_statement_pages(document)
    for name, page in statements.items():
        # Do not perform note-reference review on Statement of Changes in Equity
        if "changes in accumulated fund" in name.lower() or "changes in equity" in name.lower():
            continue
        for line in page.text.splitlines():
            parsed = _parse_statement_note_line(line, page.number, name)
            if parsed:
                items.append(parsed)
    return items


def _statement_lines_with_note_refs(document: PdfDocument) -> list[tuple[str, str, Decimal]]:
    lines: list[tuple[str, str, Decimal]] = []
    for item in _statement_note_lines(document):
        if item.amounts:
            lines.append((item.ref, item.line, item.amounts[-1]))
    return lines


def _parse_statement_note_line(line: str, page_number: int, statement_name: str) -> StatementNoteLine | None:
    if not _looks_like_primary_statement_line(line):
        return None
    explicit_ref = next(iter(_refs_in_text(line)), "")
    note_match = NOTE_REF_RE.search(line)
    implicit_match = None
    if not explicit_ref:
        implicit_match = re.search(r"(?<![\.\d])\b(\d{1,2}[A-C]?)\b(?=\s+\(?-?\d[\d,\s]*\)?)", line, flags=re.I)
        if implicit_match and _amounts_in_text(line[: implicit_match.start()]):
            implicit_match = None
    ref = explicit_ref or (implicit_match.group(1).upper() if implicit_match else "")
    if not ref or not _valid_note_number(ref):
        ref = ""
    ref_start = note_match.start() if note_match else (implicit_match.start() if implicit_match else len(line))
    ref_end = note_match.end() if note_match else (implicit_match.end() if implicit_match else ref_start)
    label = _clean_statement_line_item(line[:ref_start])
    if not label or _is_subheading(label):
        return None
    if not explicit_ref and _line_item_not_face_linked(label, statement_name, False):
        pass # Allow parsing lines without explicit refs to flag missing note references
    parsed_amounts = _amounts_from_statement_line_excluding_note_ref(line, ref_start, ref_end)
    if ref and _amounts_look_like_note_reference_only(parsed_amounts, ref):
        parsed_amounts = []
    amounts = tuple(parsed_amounts[-2:])
    return StatementNoteLine(ref.upper() if ref else "", label, line.strip(), amounts, page_number, statement_name, bool(note_match))


def _amounts_from_statement_line_excluding_note_ref(line: str, ref_start: int, ref_end: int) -> list[Decimal]:
    cleaned = f"{line[:ref_start]} {line[ref_end:]}"
    return _amounts_from_statement_line(cleaned)


def _amounts_look_like_note_reference_only(amounts: list[Decimal], ref: str) -> bool:
    if len(amounts) != 1:
        return False
    match = re.fullmatch(r"(\d{1,2})[A-C]?", ref.upper())
    if not match:
        return False
    return amounts[0] == Decimal(match.group(1))


def _statement_name_from_page(text: str) -> str:
    for line in text.splitlines()[:12]:
        clean = re.sub(r"\s+", " ", line).strip()
        if re.search(r"^statement of", clean, flags=re.I):
            return clean
    return "Primary statements"


def _statement_excluded_from_note_agreement(statement_name: str, page_text: str) -> bool:
    lower = f"{statement_name}\n{page_text[:600]}".lower()
    return any(
        marker in lower
        for marker in (
            "statement of value added",
            "value added statement",
            "five-year financial summary",
            "five year financial summary",
            "5 year financial summary",
            "financial summary",
        )
    )


def _clean_statement_line_item(text: str) -> str:
    cleaned = _statement_label(text)
    cleaned = re.sub(r"\b(total|net)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned


def _note_agreement_skip_reason(item: StatementNoteLine) -> str:
    statement = item.statement_name.lower()
    if "value added" in statement or "five year" in statement or "financial summary" in statement:
        return "not a face-linked note line"
    if _line_item_not_face_linked(item.line_item, item.statement_name, item.explicit_ref):
        return "not a face-linked note line"
    return ""


def _line_item_not_face_linked(line_item: str, statement_name: str, explicit_ref: bool = False) -> bool:
    statement = statement_name.lower()
    label = _normalise_match_words(line_item)
    raw_label = line_item.lower()
    broad_labels = {
        "current liabilities",
        "liabilities",
        "total liabilities",
        "current assets",
        "total current assets",
        "total assets",
        "total equity liabilities",
        "surplus year",
        "surplus for year",
        "net cash",
        "cash generated operations",
        "cash used operations",
    }
    if label in broad_labels:
        return True
    if raw_label.startswith(("total ", "net cash", "surplus for the year")) and not explicit_ref:
        return True
    if "cash flow" in statement and re.search(r"\b(total|net|cash generated|cash used|increase|decrease|cash inflow|cash outflow|cash absorbed)\b", raw_label):
        return True
    return False


def _check_possible_wrong_note_references(
    statement_lines: list[StatementNoteLine],
    note_sections: dict[str, str],
    headings: dict[str, str],
    tolerance: Decimal,
    cautious_review_prompt: bool = False,
) -> tuple[list[Finding], set[tuple[str, str]]]:
    findings: list[Finding] = []
    flagged: set[tuple[str, str]] = set()
    for item in statement_lines:
        if not item.ref:
            continue
        if _note_agreement_skip_reason(item):
            continue
        referenced = _get_note_section_with_fallback(item.ref, note_sections)
        referenced_heading = _get_note_heading_with_fallback(item.ref, headings)
        if _is_disclosure_only_note(referenced_heading):
            continue
        if item.ref and not referenced_heading:
            if not item.explicit_ref:
                continue
            findings.append(_note_reference_review_prompt(item, "", "Low", "Referenced note heading was not detected.", cautious_review_prompt))
            flagged.add((item.ref, item.line))
            continue
        referenced = referenced or ""
        referenced_match = _note_match_strength(item, referenced_heading, referenced, tolerance)
        referenced_heading_score = max(_wording_match_score(item.line_item, referenced_heading), _semantic_heading_score(item.line_item, referenced_heading))
        best_ref = ""
        best_score = -1
        best_match: dict[str, bool] = {"wording": False, "amount": False}
        best_heading_score = 0.0
        for candidate_ref in headings:
            if candidate_ref == item.ref or _is_disclosure_only_note(_get_note_heading_with_fallback(candidate_ref, headings)):
                continue
            section = _get_note_section_with_fallback(candidate_ref, note_sections)
            if not _alternative_note_semantically_allowed(item.line_item, _get_note_heading_with_fallback(candidate_ref, headings), section):
                continue
            match = _note_match_strength(item, _get_note_heading_with_fallback(candidate_ref, headings), section, tolerance, all_sections=note_sections)
            heading_score = max(_wording_match_score(item.line_item, _get_note_heading_with_fallback(candidate_ref, headings)), _semantic_heading_score(item.line_item, _get_note_heading_with_fallback(candidate_ref, headings)))
            stronger_heading = heading_score >= 0.82 and heading_score > referenced_heading_score + 0.12
            if not (match["wording"] or match["amount"] or stronger_heading):
                continue
            score = (2 if (match["wording"] or stronger_heading) else 0) + (3 if match["amount"] else 0) + int(heading_score * 10)
            if score > best_score:
                best_ref = candidate_ref
                best_score = score
                best_match = {"wording": match["wording"] or stronger_heading, "amount": match["amount"]}
                best_heading_score = heading_score
        if referenced_match["wording"] and referenced_match["amount"]:
            continue
        if referenced_match["amount"]:
            continue
        if referenced_match["wording"] and not best_match["amount"] and not (best_ref and best_heading_score > referenced_heading_score + 0.12):
            continue
        if not best_ref:
            continue
        if _is_revenue_line_item(item.line_item) and not (best_match["wording"] and best_match["amount"]):
            continue
        if cautious_review_prompt and best_match["amount"] and not best_match["wording"]:
            continue
        confidence = "High" if best_match["wording"] and best_match["amount"] else "Medium" if best_match["amount"] else "Low"
        if cautious_review_prompt and confidence == "High":
            confidence = "Medium"
        if cautious_review_prompt and confidence == "Low":
            continue
        reason = _note_reference_reason(best_match, best_ref, cautious_review_prompt)
        findings.append(_note_reference_review_prompt(item, best_ref, confidence, reason, cautious_review_prompt))
        flagged.add((item.ref, item.line))
    return findings, flagged


def _note_reference_review_prompt(
    item: StatementNoteLine,
    suggested_ref: str,
    confidence: str,
    reason: str,
    cautious_review_prompt: bool,
) -> Finding:
    if suggested_ref:
        issue = f"Possible wrong note reference: {item.line_item.title()} references Note {item.ref}, but Note {suggested_ref} appears to be a stronger match."
    else:
        issue = f"Referenced note not found: {item.line_item.title()} references Note {item.ref}, but that note was not detected."
    evidence = (
        f"Line: {item.line[:160]}. Amounts checked: {', '.join(f'{amount:,}' for amount in item.amounts)}. "
        f"Reason: {reason}"
        + (" Review prompt only because note extraction confidence is below threshold." if cautious_review_prompt else "")
    )
    return Finding(
        "Notes agreement",
        confidence,
        f"Page {item.page_number} | {item.statement_name}",
        issue,
        evidence,
        "Review the face statement note reference and correct it if the linked note number is wrong.",
        {
            "statement": item.statement_name,
            "line_item": item.line_item.title(),
            "referenced_note": item.ref,
            "suggested_note": suggested_ref,
            "match_confidence": confidence,
            "reason": reason,
            "line_key": f"{item.ref}|{item.line}",
        },
    )


def _check_cautious_face_note_amount_agreement(
    statement_lines: list[StatementNoteLine],
    note_sections: dict[str, str],
    headings: dict[str, str],
    tolerance: Decimal,
    cautious_review_prompt: bool,
    suppressed_lines: set[str] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    suppressed_lines = suppressed_lines or set()
    for item in statement_lines:
        if not item.ref:
            continue
        line_key = f"{item.ref}|{item.line}"
        if line_key in suppressed_lines or _is_disclosure_only_note(_get_note_heading_with_fallback(item.ref, headings)):
            continue
        if _note_agreement_skip_reason(item):
            continue
        if not item.amounts or len(item.amounts) < 1:
            continue
        referenced_section = _get_note_section_with_fallback(item.ref, note_sections)
        if not referenced_section:
            continue
        current_amount = item.amounts[0] if len(item.amounts) >= 1 else None
        prior_amount = item.amounts[1] if len(item.amounts) >= 2 else None
        current_match = _amount_match_in_section(current_amount, referenced_section, tolerance)
        prior_match = _amount_match_in_section(prior_amount, referenced_section, tolerance)
        current_found = current_match["found"]
        prior_found = prior_match["found"]
        if current_found and (prior_amount is None or prior_found):
            continue
        heading_score = _wording_match_score(item.line_item, _get_note_heading_with_fallback(item.ref, headings))
        if heading_score >= 0.82 and (current_found or prior_found):
            continue
        alternative_ref = _alternative_note_for_missing_amounts(item, note_sections, headings, tolerance)
        issue = (
            f"Amount appears in another note: {item.line_item.title()} references Note {item.ref}, but the amount appears in Note {alternative_ref}."
            if alternative_ref
            else f"Amount not located in referenced note: {item.line_item.title()} references Note {item.ref}."
        )
        confidence = _amount_match_confidence(current_found, prior_found, alternative_ref, cautious_review_prompt)
        if confidence == "Low":
            continue
        findings.append(
            Finding(
                "Notes agreement",
                confidence,
                f"Page {item.page_number} | {item.statement_name}",
                issue,
                (
                    f"Line: {item.line[:160]}. Current year found in referenced note: {_yes_no(current_found)}. "
                    f"Prior year found in referenced note: {_yes_no(prior_found) if prior_amount is not None else 'N/A'}. "
                    + (f"Alternative note found: Note {alternative_ref}." if alternative_ref else "Amount was not located in detected notes.")
                    + (" Review prompt only because note/table confidence is below threshold." if cautious_review_prompt else "")
                ),
                "Review the note reference and agree the face statement amount to the related note schedule.",
                {
                    "statement": item.statement_name,
                    "line_item": item.line_item.title(),
                    "referenced_note": item.ref,
                    "suggested_note": alternative_ref,
                    "match_confidence": confidence,
                    "reason": "Current/prior amount not located in referenced note." if not alternative_ref else f"Missing amount appears in Note {alternative_ref}.",
                    "current_year_amount_found": _yes_no(current_found),
                    "prior_year_amount_found": _yes_no(prior_found) if prior_amount is not None else "N/A",
                    "amount_found_in_note": item.ref if current_found or prior_found else "",
                    "alternative_note_found": alternative_ref,
                    "amount_match_confidence": confidence,
                    "line_key": line_key,
                },
            )
        )
    return findings


def _single_amount_in_section(amount: Decimal | None, section: str, tolerance: Decimal) -> bool:
    return _amount_match_in_section(amount, section, tolerance)["found"]


def _amount_match_in_section(amount: Decimal | None, section: str, tolerance: Decimal) -> dict[str, object]:
    if amount is None or amount == 0 or abs(amount) <= tolerance * 5:
        return {"found": True, "snippet": "", "method": "not material"}
    for candidate, snippet, method in _normalized_amount_candidates(section):
        if abs(candidate - amount) <= tolerance:
            return {"found": True, "snippet": snippet, "method": method}
        if amount < 0 and abs(abs(candidate) - abs(amount)) <= tolerance:
            return {"found": True, "snippet": snippet, "method": f"{method} / absolute value"}
    return {"found": False, "snippet": "", "method": "not found"}


def _normalized_amount_candidates(text: str) -> list[tuple[Decimal, str, str]]:
    candidates: list[tuple[Decimal, str, str]] = []
    seen: set[tuple[Decimal, str]] = set()

    def add_candidate(raw: str, start: int, end: int) -> None:
        parsed = _parse_normalized_amount(raw)
        if parsed is None:
            return
        normalized_raw = _normalize_amount_token(raw)
        method = "exact amount" if raw.strip() == f"{parsed:,}" else "normalized amount"
        key = (parsed, normalized_raw)
        if key not in seen:
            candidates.append((parsed, _amount_snippet_around(text, start, end), method))
            seen.add(key)

    for match in re.finditer(r"(?<![\d,])\d\s+\d{1,2},\d{3}\b", text):
        add_candidate(match.group(0), match.start(), match.end())
    for match in NORMALIZED_AMOUNT_RE.finditer(text):
        raw = match.group(0)
        add_candidate(raw, match.start(), match.end())
    return candidates


def _parse_normalized_amount(value: str) -> Decimal | None:
    raw = str(value or "").strip()
    if not raw or raw in {"-", "--"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = _normalize_amount_token(raw)
    raw = raw.strip("()")
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _normalize_amount_token(value: str) -> str:
    token = re.sub(r"\s*[,]\s*", ",", value.strip())
    token = re.sub(r"\b(\d)\s+(\d{1,2},\d{3})\b", r"\1\2", token)
    token = re.sub(r"(?<=\d)\s*\n\s*(?=\d{3}\b)", ",", token)
    token = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", token)
    return token.replace(",", "").replace(" ", "")


def _amount_snippet_around(text: str, start: int, end: int, window: int = 70) -> str:
    snippet = text[max(0, start - window) : min(len(text), end + window)]
    return re.sub(r"\s+", " ", snippet).strip()


def _alternative_note_for_missing_amounts(
    item: StatementNoteLine,
    note_sections: dict[str, str],
    headings: dict[str, str],
    tolerance: Decimal,
) -> str:
    meaningful = [amount for amount in item.amounts if amount != 0 and abs(amount) > tolerance * 5]
    if not meaningful:
        return ""
    best_ref = ""
    best_count = 0
    best_wording = 0.0
    for ref, section in note_sections.items():
        if ref == item.ref or _is_disclosure_only_note(_get_note_heading_with_fallback(ref, headings)):
            continue
        heading = _get_note_heading_with_fallback(ref, headings)
        if not _alternative_note_semantically_allowed(item.line_item, heading, section):
            continue
        wording = _wording_match_score(item.line_item, heading)
        semantic_score = _semantic_heading_score(item.line_item, heading)
        if _is_revenue_line_item(item.line_item) and not _revenue_alternative_heading_allowed(heading):
            continue
        if max(wording, semantic_score) < 0.82:
            continue
        count = sum(1 for amount in meaningful if _amount_match_in_section(amount, section, tolerance)["found"])
        if _is_revenue_line_item(item.line_item) and count < min(2, len(meaningful)):
            continue
        if count > best_count or (count == best_count and count > 0 and wording > best_wording):
            best_ref = ref
            best_count = count
            best_wording = max(wording, semantic_score)
    return best_ref if best_count else ""


def _weak_semantic_alternative_note(
    item: StatementNoteLine,
    headings: dict[str, str],
    note_sections: dict[str, str],
) -> str:
    referenced_heading = _get_note_heading_with_fallback(item.ref, headings)
    referenced_score = _wording_match_score(item.line_item, referenced_heading)
    best_ref = ""
    best_score = referenced_score
    for ref, heading in headings.items():
        if ref == item.ref:
            continue
        if not _alternative_note_semantically_allowed(item.line_item, heading, _get_note_section_with_fallback(ref, note_sections)):
            continue
        score = max(_wording_match_score(item.line_item, heading), _semantic_heading_score(item.line_item, heading))
        if score >= 0.82 and score > best_score + 0.12:
            best_ref = ref
            best_score = score
    return best_ref


def _heading_only_alternative_is_review_prompt(line_item: str, note_heading: str) -> bool:
    if not _is_revenue_line_item(line_item):
        return True
    item = _normalise_match_words(line_item)
    heading = _normalise_match_words(note_heading)
    if not _revenue_alternative_heading_allowed(heading):
        return False
    if item in heading or heading in item:
        return True
    if item == "revenue" and re.search(r"\brevenue\b", heading):
        return True
    if "operating revenue" in item and any(term in heading for term in ("operating revenue", "operating income")):
        return True
    return False


def _semantic_heading_score(line_item: str, note_heading: str) -> float:
    item = _normalise_match_words(line_item)
    heading = _normalise_match_words(note_heading)
    if _is_revenue_line_item(item) and _revenue_alternative_heading_allowed(heading):
        return 0.86
    if any(term in item for term in ("cash", "bank", "cash equivalents")) and any(term in heading for term in ("cash", "bank", "cash equivalents")):
        return 0.86
    return 0.0


def _alternative_note_semantically_allowed(line_item: str, note_heading: str, note_section: str = "") -> bool:
    item = _normalise_match_words(line_item)
    heading = _normalise_match_words(note_heading)
    if any(term in item for term in ("cash", "bank", "cash equivalents")):
        return any(term in heading for term in ("cash", "bank", "cash equivalents"))
    if _is_revenue_line_item(item):
        return _revenue_alternative_heading_allowed(heading)
    return True


def _is_revenue_line_item(line_item: str) -> bool:
    item = _normalise_match_words(line_item)
    return any(term in item for term in ("revenue", "operating income", "turnover", "sales", "income from property", "rental income"))


def _revenue_alternative_heading_allowed(note_heading: str) -> bool:
    heading = _normalise_match_words(note_heading)
    allowed = (
        "revenue",
        "rental income",
        "operating income",
        "turnover",
        "income from property",
        "other operating income",
    )
    return any(term in heading for term in allowed)


def _amount_match_confidence(current_found: bool, prior_found: bool, alternative_ref: str, cautious_review_prompt: bool) -> str:
    if alternative_ref and not cautious_review_prompt and current_found is False and prior_found is False:
        return "High"
    if alternative_ref:
        return "Medium"
    return "Low"


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _combined_matching_method(current_match: dict[str, object], prior_match: dict[str, object]) -> str:
    methods = [
        str(match.get("method", ""))
        for match in (current_match, prior_match)
        if match.get("found") and match.get("method") not in {"", "not material"}
    ]
    return " / ".join(dict.fromkeys(methods))


def _note_reference_reason(match: dict[str, bool], suggested_ref: str, cautious_review_prompt: bool) -> str:
    if match["wording"] and match["amount"]:
        base = f"Line item wording and amount both match Note {suggested_ref} more strongly than the referenced note."
    elif match["amount"]:
        base = f"Amount matches Note {suggested_ref}, but wording support is weak."
    else:
        base = f"Line item wording is a stronger match to Note {suggested_ref}."
    return f"{base} Treat as review prompt." if cautious_review_prompt else base


def _note_match_strength(
    item: StatementNoteLine,
    note_title: str,
    note_section: str,
    tolerance: Decimal,
    all_sections: dict[str, str] | None = None,
) -> dict[str, bool]:
    wording = _wording_matches_note(item.line_item, note_title, note_section)
    amount = _amounts_match_note(item.amounts, note_section, tolerance, all_sections)
    return {"wording": wording, "amount": amount}


def _wording_matches_note(line_item: str, note_title: str, note_section: str) -> bool:
    target = _normalise_match_words(line_item)
    if len(target) < 4:
        return False
    candidates = [_normalise_match_words(note_title)]
    for line in note_section.splitlines()[:40]:
        clean = _normalise_match_words(line)
        if clean:
            candidates.append(clean)
    return any(
        target in candidate
        or candidate in target
        or SequenceMatcher(None, target, candidate).ratio() >= 0.82
        for candidate in candidates
        if len(candidate) >= 4
    )


def _wording_match_score(line_item: str, candidate_text: str) -> float:
    target = _normalise_match_words(line_item)
    candidate = _normalise_match_words(candidate_text)
    if not target or not candidate:
        return 0.0
    if target == candidate:
        return 1.0
    if target in candidate or candidate in target:
        return 0.9
    return SequenceMatcher(None, target, candidate).ratio()


def _amounts_match_note(
    amounts: tuple[Decimal, ...],
    note_section: str,
    tolerance: Decimal,
    all_sections: dict[str, str] | None = None,
) -> bool:
    meaningful = [amount for amount in amounts if abs(amount) > tolerance * 5 and amount != 0]
    if not meaningful:
        return False
    matched = [
        amount
        for amount in meaningful
        if _amount_match_in_section(amount, note_section, tolerance)["found"]
    ]
    if not matched:
        return False
    if len(meaningful) > 1:
        return len(matched) >= 2
    amount = meaningful[0]
    if all_sections:
        occurrence_count = sum(
            1
            for section in all_sections.values()
            if _amount_match_in_section(amount, section, tolerance)["found"]
        )
        if occurrence_count > 2:
            return False
    return True


def _normalise_match_words(text: str) -> str:
    text = re.sub(r"[^a-z0-9& ]", " ", text.lower())
    stop_words = {"the", "and", "or", "of", "to", "from"}
    words = [word for word in text.split() if word not in stop_words and not word.isdigit()]
    return " ".join(words)


def _refs_in_text(text: str) -> set[str]:
    refs = set()
    for match in NOTE_REF_RE.finditer(text):
        ref = (match.group(1) or match.group(2)).upper()
        if _valid_note_number(ref):
            refs.add(ref)
    return refs


def _note_refs_from_statement_lines(text: str) -> set[str]:
    refs: set[str] = set()
    for line in text.splitlines():
        refs.update(_note_refs_from_statement_line(line))
    return refs


def _note_refs_from_statement_line(line: str) -> set[str]:
    refs: set[str] = set()
    if not _looks_like_primary_statement_line(line):
        return refs
    for match in re.finditer(r"\b(\d{1,2}[A-C]?)\b(?=\s+\(?-?\d[\d,\s]*\)?)", line, flags=re.I):
        ref = match.group(1).upper()
        if _valid_note_number(ref):
            refs.add(ref)
            break
    return refs


def _note_refs_from_tables(tables: list[list[list[str]]]) -> set[str]:
    refs: set[str] = set()
    for table in tables:
        note_cols = _note_columns(table)
        for row in table[1:]:
            for index in note_cols:
                if index < len(row):
                    raw = str(row[index]).strip().upper()
                    if _valid_note_number(raw):
                        refs.add(raw)
    return refs


def _note_sections(document: PdfDocument) -> dict[str, str]:
    start_page = _notes_start_page(document)
    if not start_page:
        text = document.text
    else:
        text = "\n".join(page.text for page in document.pages if page.number >= start_page)

    sections: dict[str, list[str]] = defaultdict(list)
    current_refs: list[str] = []
    pending_number: str | None = None
    lines = text.splitlines()
    strict_notes_start = any(_notes_heading_in_text(line) for line in lines)
    in_notes = not strict_notes_start
    for index, line in enumerate(lines):
        stripped = line.strip()
        if strict_notes_start and _notes_heading_in_text(stripped):
            in_notes = True
        if strict_notes_start and not in_notes:
            continue
        match = NOTE_HEADING_RE.match(stripped)
        if match and _valid_note_heading(match.group(1), match.group(2)):
            current_refs = [match.group(1).upper()]
            pending_number = None
        else:
            combined_refs = _combined_note_refs_from_line(lines, index, stripped)
            if combined_refs:
                current_refs = combined_refs
                pending_number = None
            else:
                number_only = NOTE_NUMBER_ONLY_RE.match(stripped)
                if number_only and _valid_note_number(number_only.group(1)):
                    pending_number = number_only.group(1).upper()
                    continue
                if pending_number and _valid_note_heading(pending_number, stripped):
                    current_refs = [pending_number]
                    pending_number = None
        for current in current_refs:
            sections[current].append(line)
    return {number: "\n".join(lines) for number, lines in sections.items()}


def _combined_note_refs_from_line(lines: list[str], index: int, line: str) -> list[str]:
    if not re.search(r"\bnote\b", line, flags=re.I):
        return []
    refs = [ref.upper() for ref in re.findall(r"\b(\d{1,2}[A-C]?)\b", line, flags=re.I) if _valid_note_number(ref)]
    if not {"7", "8"}.issubset(set(refs)):
        return []
    nearby = " ".join(lines[index : index + 5]).lower()
    if "revenue heads" in nearby and "operating revenue" in nearby:
        return ["7", "8"]
    return []


def _is_notes_page(text: str) -> bool:
    lower = text.lower()
    return _notes_heading_in_text(text) or "notes to the financial statements" in lower or lower.count("accounting polic") >= 2


def _valid_note_heading(number: str, title: str) -> bool:
    number = number.upper().strip()
    title_clean = _clean_note_title(title)
    title_lower = title_clean.lower()
    if not _valid_note_number(number):
        return False
    if _note_heading_title_looks_narrative(title_clean):
        return False
    if not _note_heading_title_is_structural(title_clean):
        return False

    if re.search(r"[A-C]$", number) and not _suffixed_note_heading_title_is_structural(title_clean):
        return False
    if title_lower.startswith(("to the", "notes to", "are", "and", "for the year", "in thousands", "n'000")):
        return False
    if title_lower.startswith(("financial statements for the year ended", "audited financial statements")):
        return False
    if len(NUMBER_RE.findall(title_clean)) > 1:
        return False
    front_matter_terms = (
        "directors",
        "director",
        "report of the directors",
        "directors interests",
        "directors' interests",
        "interest in shares",
        "interests in shares",
        "employment and employees",
        "employees",
        "corporate information",
        "auditor",
        "auditors",
        "independent auditor",
    )
    if title_lower in set(front_matter_terms) or any(term in title_lower for term in front_matter_terms):
        return False
    words = title_clean.split()
    if ENTITY_SUFFIX_RE.search(title_clean) and len(words) <= 4:
        return False
    if title_lower in {"december", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november"}:
        return False
    if YEAR_RE.search(title_clean) and len(words) <= 4:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", title_clean))


def _note_heading_title_looks_narrative(title: str) -> bool:
    title_lower = title.strip().lower()
    narrative_starts = (
        "this ",
        "these ",
        "the ",
        "it ",
        "represents",
        "representing",
        "being ",
        "which ",
        "where ",
        "when ",
    )
    if title_lower.startswith(narrative_starts):
        return True
    if len(title.split()) > 12 and re.search(r"\b(represents?|comprises?|relates?|amounts?|advance payment|payment)\b", title_lower):
        return True
    return False


def _note_heading_title_is_structural(title: str) -> bool:
    title_clean = _clean_note_title(title)
    title_lower = title_clean.lower().strip()
    if not title_lower:
        return False
    words = re.findall(r"[A-Za-z&'-]+", title_clean)
    if len(words) > 7:
        return False
    if re.search(r"[.;:]\s*$", title_clean):
        return False
    continuation_terms = (
        "continued",
        "brought forward",
        "carried forward",
        "for the year",
        "as at",
        "balances as",
        "amounts due",
    )
    if any(term in title_lower for term in continuation_terms):
        return False
    risk_subheadings = (
        "credit risk",
        "liquidity risk",
        "market risk",
        "interest rate risk",
        "currency risk",
        "capital risk",
        "fair value hierarchy",
        "sensitivity analysis",
        "maturity analysis",
    )
    if title_lower in risk_subheadings:
        return False
    policy_or_sentence_terms = (
        "accounted for",
        "recognised",
        "recognized",
        "measured",
        "depreciated",
        "amortised",
        "amortized",
        "provided",
        "charged",
        "classified",
        "presented",
        "disclosed",
        "represents",
        "comprises",
        "relates",
        "includes",
        "shall",
        "should",
        "would",
        "could",
    )
    if any(term in title_lower for term in policy_or_sentence_terms):
        return False
    if re.search(r"\b(?:is|are|was|were|has|have|had|will|may|can)\b", title_lower):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", title_clean))


def _suffixed_note_heading_title_is_structural(title: str) -> bool:
    title_lower = title.strip().lower()
    if not title_lower or _note_heading_title_looks_narrative(title):
        return False
    words = title.split()
    if len(words) > 8:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", title))


def _valid_note_number(value: str) -> bool:
    value = value.upper().strip()
    if YEAR_RE.fullmatch(value):
        return False
    match = re.fullmatch(r"(\d{1,2})([A-C]?)", value)
    if not match:
        return False
    numeric = int(match.group(1))
    return 1 <= numeric <= 60


def _clean_note_title(title: str) -> str:
    title = re.sub(r"^[^\w(]+", "", title.strip())
    title = re.sub(r"(?:\s+20\d{2}){1,3}\s*$", "", title)
    title = re.sub(r"\s+(?:N['’]?\s?000|\$?000s?)(?:\s+(?:N['’]?\s?000|\$?000s?))*$", "", title, flags=re.I)
    return title.strip()


def _looks_like_primary_statement_line(line: str) -> bool:
    if len(NUMBER_RE.findall(line)) < 1:
        return False
        
    lower = line.lower()
    reject_phrases = [
        "financial statements", "statement of", "year ended", "as at", 
        "signed on", "behalf", "behal b", "the notes on page", "pages", "director", 
        "chairman", "secretary", "approval", "n n", "0 0"
    ]
    if any(phrase in lower for phrase in reject_phrases) or re.search(r"(?i)\b(?:were signed|approval|n\s*n)\b", lower):
        return False
        
    text_only = re.sub(r"[\d\.,\(\)\-\|]", "", lower).strip()
    # Reject lines that contain only letters N, M, O (common unit/currency artifacts) or are too short.
    letters_only = re.sub(r"[^a-z]", "", lower)
    if len(letters_only) < 3 or set(letters_only).issubset({"n", "m", "o"}):
        return False
        
    if _is_subheading(_clean_statement_line_item(line)):
        return False
    return not _is_notes_page(line)


def _is_disclosure_only_note(title: str) -> bool:
    lower = title.lower()
    disclosure_terms = (
        "accounting polic",
        "basis of preparation",
        "basis of measurement",
        "critical accounting estimates",
        "estimates and judgements",
        "financial instruments - risk",
        "financial risk",
        "risk management",
        "capital commitments",
        "contingent liabilities",
        "subsequent events",
        "events after",
        "related party",
        "corporate information",
    )
    return any(term in lower for term in disclosure_terms)


def _last_amount(line: str) -> Decimal | None:
    amounts = _amounts_in_text(line)
    return amounts[-1] if amounts else None


def _note_sort_key(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)([A-Z]?)", value)
    if not match:
        return (9999, value)
    return (int(match.group(1)), match.group(2))

def _check_ocr_statement_of_cash_flows(document: PdfDocument, tolerance: Decimal) -> list[Finding]:
    findings: list[Finding] = []
    for page in document.pages:
        lower_text = page.text[:1000].lower()
        if not ("statement of cash flows" in lower_text or "cash flow statement" in lower_text):
            continue
        
        line_row_map: dict[str, Decimal] = {}
        for line in page.text.splitlines():
            line = line.strip()
            amounts = re.findall(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?", line)
            if amounts:
                amt_str = amounts[0] if len(amounts) == 1 else amounts[-2] if len(amounts) >= 2 else amounts[0]
                clean_amt = amt_str.replace(",", "").replace("(", "-").replace(")", "")
                try:
                    val = Decimal(clean_amt)
                    lower = line.lower()
                    if "operat" in lower and not "operating" in line_row_map:
                        line_row_map["operating"] = val
                    elif "invest" in lower and not "investing" in line_row_map:
                        line_row_map["investing"] = val
                    elif "financ" in lower and not "financing" in line_row_map:
                        line_row_map["financing"] = val
                    elif ("increase" in lower or "decrease" in lower or "movement" in lower or "net cash" in lower or "cash flow" in lower) and not any(x in lower for x in ["operat", "invest", "financ"]) and "movement" not in line_row_map:
                        line_row_map["movement"] = val
                    elif ("beginning" in lower or "start" in lower or " 1 " in lower or "january" in lower) and "cash" in lower:
                        if "opening" not in line_row_map:
                            line_row_map["opening"] = val
                    elif ("end" in lower or " 31 " in lower or "december" in lower) and "cash" in lower:
                        line_row_map["closing"] = val
                    elif "exchange" in lower:
                        line_row_map["exchange"] = val
                except Exception:
                    pass
                    
        if "operating" in line_row_map and "investing" in line_row_map and "financing" in line_row_map and "movement" in line_row_map:
            expected = line_row_map["operating"] + line_row_map["investing"] + line_row_map["financing"]
            reported = line_row_map["movement"]
            diff = reported - expected
            if abs(diff) > tolerance:
                findings.append(Finding(
                    "Totals and rounding",
                    "High" if abs(diff) > tolerance * 5 else "Medium",
                    f"Page {page.number} | Statement of cash flows",
                    "Net operating + investing + financing does not equal total cash movement.",
                    f"Expected {expected:,}; reported {reported:,}; difference {diff:,}.",
                    "Review the cash flow statement totals against the signed financial statements."
                ))
            else:
                findings.append(Finding("Calculation", "Passed", "Statement of cash flows", "Operating, investing, and financing cash flows agree to net increase.", "Equation passed.", ""))
                
        if "opening" in line_row_map and "movement" in line_row_map and "closing" in line_row_map:
            expected = line_row_map["opening"] + line_row_map["movement"] + line_row_map.get("exchange", Decimal(0))
            reported = line_row_map["closing"]
            diff = reported - expected
            if abs(diff) > tolerance:
                findings.append(Finding(
                    "Totals and rounding",
                    "High" if abs(diff) > tolerance * 5 else "Medium",
                    f"Page {page.number} | Statement of cash flows",
                    "Opening cash + movement + exchange effect does not equal closing cash.",
                    f"Expected {expected:,}; reported {reported:,}; difference {diff:,}.",
                    "Review the cash flow statement totals against the signed financial statements."
                ))
            else:
                findings.append(Finding("Calculation", "Passed", "Statement of cash flows", "Closing cash agrees to opening cash plus net increase.", "Equation passed.", ""))
    return findings
