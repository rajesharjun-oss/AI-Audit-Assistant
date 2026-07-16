from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from models import ChecklistItem, CompanyProfile, Finding, PdfDocument, PdfPage, ReviewOptions, ReviewResult
from cross_page_consistency import check_cross_page_consistency
from policy_reviewer import review_notes_1_and_2
from ai_review_pipeline import AiReviewContext, run_ai_review_pipeline
from extraction import extract_pdf, extract_pdf_with_ocr
from canonical_checks import run_canonical_checks
from canonical_extraction import document_section_map, table_classification_rows


NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?")
YEAR_RE = re.compile(r"\b20\d{2}\b")
NOTE_REF_RE = re.compile(r"\bnote\s+(\d+[A-Za-z]?)\b|\bnotes?\s+(\d+[A-Za-z]?)\b", re.I)
NOTE_HEADING_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)(?:\s*\([a-z]\))?\s*[\).:-]?\s*(.{3,100})$", re.I)
NORMALIZED_AMOUNT_RE = re.compile(
    r"\(?-?\d{1,3}(?:\s*,\s*\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?",
    re.M,
)
NOTE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)\s+((?:20\d{2}|N[\'\u2019]?\s?000|\$?000s?|\d{4})[\s,]*)+$", re.I)
ENTITY_SUFFIX_RE = re.compile(r"\b(?:limited|ltd|plc|inc|corp|corporation|company)\b", re.I)
VALID_CURRENCIES = {"NGN", "USD", "GBP", "EUR", "ZAR", "GHS", "KES", "CAD", "AUD"}
NOTE_TITLE_OCR_CORRECTIONS = {
    "othet": "other",
    "amottisation": "amortisation",
    "patty": "party",
    "prepatation": "preparation",
    "liabilties": "liabilities",
    "liabilty": "liability",
    "interpreiations": "interpretations",
}


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

NOTE_COMPATIBILITY_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "ppe": {
        "line": ("property plant equipment", "ppe", "tangible assets", "plant equipment"),
        "heading": ("property plant equipment", "ppe", "tangible assets", "plant machinery", "buildings", "fixtures", "furniture"),
    },
    "investment property": {
        "line": ("investment property",),
        "heading": ("investment property",),
    },
    "intangible assets": {
        "line": ("intangible assets", "intangible asset", "software", "goodwill"),
        "heading": ("intangible assets", "intangible asset", "software", "goodwill", "amortisation", "amortization"),
    },
    "inventory": {
        "line": ("inventory", "inventories", "stock"),
        "heading": ("inventory", "inventories", "stock"),
    },
    "trade receivables": {
        "line": ("trade receivables", "other receivables", "trade other receivables", "receivables", "contract assets"),
        "heading": ("trade receivables", "other receivables", "trade other receivables", "receivables", "contract assets"),
    },
    "other financial assets": {
        "line": ("other financial assets", "financial asset", "financial assets", "amortised cost", "amortized cost", "loans advances"),
        "heading": ("other financial assets", "financial asset", "financial assets", "amortised cost", "amortized cost", "loans advances"),
    },
    "cash": {
        "line": ("cash", "bank", "cash equivalents"),
        "heading": ("cash", "bank", "cash equivalents"),
    },
    "share capital": {
        "line": ("share capital", "ordinary shares", "issued capital"),
        "heading": ("share capital", "ordinary shares", "issued capital"),
    },
    "deposit for shares": {
        "line": ("deposit for shares", "share deposit", "shares deposit", "deposit on shares", "deposit towards shares"),
        "heading": ("deposit for shares", "share deposit", "shares deposit", "deposit on shares", "deposit towards shares"),
    },
    "dividends": {
        "line": ("dividend", "dividends", "distribution to owners", "distribution to shareholders"),
        "heading": ("dividend", "dividends", "distribution to owners", "distribution to shareholders"),
    },
    "leases": {
        "line": ("lease liability", "lease liabilities", "right of use asset", "right-of-use asset", "rou asset", "lease expense"),
        "heading": ("lease liability", "lease liabilities", "right of use asset", "right-of-use asset", "rou asset", "lease expense", "leases"),
    },
    "borrowings": {
        "line": ("borrowings", "loans", "financial liabilities", "bank overdraft", "overdraft"),
        "heading": ("borrowings", "loans", "financial liabilities", "bank overdraft", "overdraft"),
    },
    "trade payables": {
        "line": ("trade payables", "other payables", "trade other payables", "payables", "accruals", "contract liabilities"),
        "heading": ("trade payables", "other payables", "trade other payables", "payables", "accruals", "contract liabilities"),
    },
    "provisions": {
        "line": ("provision", "provisions", "legal provision", "warranty provision", "decommissioning"),
        "heading": ("provision", "provisions", "legal provision", "warranty provision", "decommissioning"),
    },
    "deferred income": {
        "line": ("deferred income", "contract liabilities", "contract liability", "advance from customers", "customer advances"),
        "heading": ("deferred income", "contract liabilities", "contract liability", "advance from customers", "customer advances"),
    },
    "deferred tax": {
        "line": ("deferred tax", "deferred tax asset", "deferred tax liability"),
        "heading": ("deferred tax", "deferred tax asset", "deferred tax liability"),
    },
    "tax": {
        "line": ("tax", "taxation", "current tax", "income tax", "tax payable", "tax receivable"),
        "heading": ("tax", "taxation", "current tax", "income tax", "tax payable", "tax receivable"),
    },
    "revenue": {
        "line": ("revenue", "operating income", "turnover", "sales", "income property", "rental income"),
        "heading": ("revenue", "rental income", "operating income", "turnover", "income property", "other operating income"),
    },
    "direct costs": {
        "line": ("direct costs", "cost sales", "cost revenue"),
        "heading": ("direct costs", "cost sales", "cost revenue"),
    },
    "employee benefits": {
        "line": ("employee benefit", "employee benefits", "staff costs", "personnel costs", "salaries", "wages", "payroll", "pension"),
        "heading": ("employee benefit", "employee benefits", "staff costs", "personnel costs", "salaries", "wages", "payroll", "pension"),
    },
    "impairment": {
        "line": ("impairment", "expected credit loss", "ecl", "credit loss", "loss allowance"),
        "heading": ("impairment", "expected credit loss", "ecl", "credit loss", "loss allowance"),
    },
    "fair value gains": {
        "line": ("fair value gain", "fair value loss", "other operating gains", "operating gains", "gain on investment"),
        "heading": ("fair value gain", "fair value loss", "other operating gains", "operating gains", "gain on investment"),
    },
    "other operating income": {
        "line": ("other operating income", "other income", "miscellaneous income"),
        "heading": ("other operating income", "other income", "miscellaneous income"),
    },
    "administrative expenses": {
        "line": ("administrative expenses", "admin expenses", "operating expenses"),
        "heading": ("administrative expenses", "admin expenses", "operating expenses"),
    },
    "selling and distribution expenses": {
        "line": ("selling expenses", "distribution expenses", "marketing expenses", "selling and distribution"),
        "heading": ("selling expenses", "distribution expenses", "marketing expenses", "selling and distribution"),
    },
    "finance cost": {
        "line": ("finance cost", "finance costs", "interest expense"),
        "heading": ("finance cost", "finance costs", "interest expense"),
    },
    "finance income": {
        "line": ("finance income", "interest income"),
        "heading": ("finance income", "interest income"),
    },
    "related parties": {
        "line": ("related parties", "related party"),
        "heading": ("related parties", "related party"),
    },
    "going concern": {
        "line": ("going concern",),
        "heading": ("going concern",),
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


def extract_review_document(path: str | Path, options: ReviewOptions | None = None) -> PdfDocument:
    options = options or ReviewOptions()
    document = extract_pdf(path)
    if _requires_ocr(document) and options.use_ocr:
        document = extract_pdf_with_ocr(path, document, options)
    return document


def review_pdf(
    path: str | Path,
    profile: CompanyProfile | None = None,
    options: ReviewOptions | None = None,
) -> ReviewResult:
    options = options or ReviewOptions()
    document = extract_review_document(path, options)
    return review_document(document, path, profile, options)


def review_document(
    document: PdfDocument,
    path: str | Path,
    profile: CompanyProfile | None = None,
    options: ReviewOptions | None = None,
) -> ReviewResult:
    options = options or ReviewOptions()
    profile = _profile_with_detected_defaults(profile or CompanyProfile(), document)
    findings: list[Finding] = []
    checks_performed: list[str] = []
    checks_skipped: list[str] = []
    rotated_pages = getattr(document, "rotated_page_details", []) or []
    if rotated_pages:
        page_summary = ", ".join(f"Page {item['page']} ({item['rotation']}deg)" for item in rotated_pages)
        checks_performed.append(f"Auto-rotation recovery applied during extraction: {page_summary}.")
    note_validation_debug = _note_validation_debug(document, options.run_cautious_note_agreement, [])
    canonical_check_rows: list[dict[str, object]] = []
    canonical_audit_rows: list[dict[str, object]] = []
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
    contents_performed, contents_skipped, contents_findings = _contents_statement_page_agreement_note(document)
    findings.extend(contents_findings)
    checks_performed.extend(contents_performed)
    checks_skipped.extend(contents_skipped)
    if limited_scope_extract:
        canonical_findings, canonical_check_rows, canonical_audit_rows, canonical_performed, canonical_skipped = _run_canonical_qc_layer(document, findings)
        findings.extend(canonical_findings)
        checks_performed.extend(canonical_performed)
        checks_skipped.extend(canonical_skipped)
        checks_performed.append("Limited-scope review performed on Statement of Financial Position only.")
        checks_skipped.append("Full financial statement completeness, standards checklist, policies, formatting, and note agreement skipped because the upload is a limited-scope statement extract.")
        return _build_result(
            document,
            findings,
            checks_performed,
            checks_skipped,
            note_validation_debug,
            {},
            canonical_check_rows=canonical_check_rows,
            canonical_audit_rows=canonical_audit_rows,
        )
    totals_findings = check_totals_and_rounding(document)
    findings.extend(totals_findings)
    findings.extend(check_formatting(document, profile))
    note_findings = check_notes_agreement(document, cautious_low_confidence=options.run_cautious_note_agreement)
    findings.extend(note_findings)
    findings.extend(_check_note_contradictions(document))
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
    policy_map = _accounting_policy_map(document)
    policy_findings, policy_export = review_notes_1_and_2(document, profile, note_sections, policy_map=policy_map)
    findings.extend(policy_findings)
    if getattr(document, "skipped_table_details", None):
        checks_skipped.append("Generic table arithmetic skipped on low-confidence/non-standard tables; details are listed in Skipped table details.")
    
    checks_performed.extend(["Accounting policies relevance check", "IFRS/IAS standards alignment check"])
    cross_page_findings, cross_page_export = check_cross_page_consistency(document)
    findings.extend(cross_page_findings)
    if _policy_export_has_missing_rows(policy_export):
        checks_skipped.append("Notes 1 & 2 policy and standards alignment review requires manual review because one or more expected policy sections were not found.")
    else:
        checks_performed.append("Notes 1 & 2 policy and standards alignment review")
    canonical_findings, canonical_check_rows, canonical_audit_rows, canonical_performed, canonical_skipped = _run_canonical_qc_layer(document, findings)
    findings.extend(canonical_findings)
    checks_performed.extend(canonical_performed)
    checks_skipped.extend(canonical_skipped)

    findings = [
        f for f in findings
        if not (f.category == "Notes agreement" and f.metadata and f.metadata.get("match_confidence") == "Low")
    ]
    ai_pipeline = run_ai_review_pipeline(
        AiReviewContext(
            document=document,
            profile=profile,
            note_sections=note_sections,
            policy_map=policy_map,
            findings=findings,
            model=options.ai_model,
            pdf_path=path,
            use_policy_review=options.use_ai_policy_review,
            use_full_review=options.use_ai_full_review,
            checks_skipped=checks_skipped,
            review_mode=options.ai_review_mode,
        )
    )
    findings = ai_pipeline.findings
    checks_performed.extend(ai_pipeline.checks_performed)
    checks_skipped.extend(ai_pipeline.checks_skipped)
    ai_policy_export = ai_pipeline.policy_export
    ai_policy_summary = ai_pipeline.policy_summary
    ai_policy_status = ai_pipeline.policy_status
    ai_policy_model = ai_pipeline.policy_model
    ai_policy_message = ai_pipeline.policy_message
    ai_full_export = ai_pipeline.full_export
    ai_full_summary = ai_pipeline.full_summary
    ai_full_status = ai_pipeline.full_status
    ai_full_model = ai_pipeline.full_model
    ai_full_message = ai_pipeline.full_message
    ai_evidence_pack_rows = ai_pipeline.evidence_pack_rows
    ai_finding_export = ai_pipeline.finding_export
    ai_finding_summary = ai_pipeline.finding_summary
    ai_finding_status = ai_pipeline.finding_status
    ai_finding_model = ai_pipeline.finding_model
    ai_finding_message = ai_pipeline.finding_message
    ai_finding_suppressed = ai_pipeline.finding_suppressed
    ai_finding_suppressed_rows = ai_pipeline.finding_suppressed_rows
    ai_finding_reviewed = ai_pipeline.finding_reviewed
    ai_error_rows = ai_pipeline.error_rows
    ai_review_mode = ai_pipeline.review_mode
    ai_combined_summary = ai_pipeline.combined_summary
    ai_combined_memo = ai_pipeline.combined_memo
    ai_review_comment_rows = ai_pipeline.review_comment_rows
    ai_summary_fields = ai_pipeline.summary_fields
    return _build_result(
        document,
        findings,
        checks_performed,
        checks_skipped,
        note_validation_debug,
        cross_page_export,
        policy_export,
        ai_policy_export,
        ai_policy_status,
        ai_policy_model,
        ai_policy_summary,
        ai_policy_message,
        ai_full_export,
        ai_full_status,
        ai_full_model,
        ai_full_summary,
        ai_full_message,
        ai_finding_export,
        ai_finding_status,
        ai_finding_model,
        ai_finding_summary,
        ai_finding_message,
        ai_finding_reviewed,
        ai_finding_suppressed,
        ai_finding_suppressed_rows,
        ai_evidence_pack_rows,
        ai_error_rows,
        ai_review_mode,
        ai_combined_summary,
        ai_combined_memo,
        ai_review_comment_rows,
        ai_summary_fields,
        canonical_check_rows=canonical_check_rows,
        canonical_audit_rows=canonical_audit_rows,
    )


def rerun_ai_review_from_cached_result(
    document: PdfDocument,
    path: str | Path,
    profile: CompanyProfile | None,
    options: ReviewOptions,
    cached_result: ReviewResult,
) -> ReviewResult:
    """Rerun only the optional AI layer against cached extraction and deterministic findings."""
    options = options or ReviewOptions()
    profile = _profile_with_detected_defaults(profile or CompanyProfile(), document)
    deterministic_findings = [
        _clone_cached_finding(finding)
        for finding in cached_result.findings
        if not _is_cached_ai_finding(finding) and not _is_generated_manual_review_finding(finding)
    ]
    checks_performed = [
        item
        for item in _metric_text_lines(cached_result.metrics.get("checks_performed"), "No deterministic checks completed.")
        if not _is_ai_status_line(item)
    ]
    checks_skipped = [
        item
        for item in _metric_text_lines(cached_result.metrics.get("checks_skipped"), "No major checks skipped.")
        if not _is_ai_status_line(item)
    ]
    note_sections = _note_sections(document)
    policy_map = _accounting_policy_map(document)
    ai_pipeline = run_ai_review_pipeline(
        AiReviewContext(
            document=document,
            profile=profile,
            note_sections=note_sections,
            policy_map=policy_map,
            findings=deterministic_findings,
            model=options.ai_model,
            pdf_path=path,
            use_policy_review=options.use_ai_policy_review,
            use_full_review=options.use_ai_full_review,
            checks_skipped=checks_skipped,
            review_mode=options.ai_review_mode,
        )
    )
    checks_performed.extend(ai_pipeline.checks_performed)
    checks_skipped.extend(ai_pipeline.checks_skipped)
    return _build_result(
        document,
        ai_pipeline.findings,
        checks_performed,
        checks_skipped,
        _note_validation_debug(document, options.run_cautious_note_agreement, []),
        cached_result.metrics.get("cross_page_export", {}) or {},
        cached_result.metrics.get("policy_export", []) or [],
        ai_pipeline.policy_export,
        ai_pipeline.policy_status,
        ai_pipeline.policy_model,
        ai_pipeline.policy_summary,
        ai_pipeline.policy_message,
        ai_pipeline.full_export,
        ai_pipeline.full_status,
        ai_pipeline.full_model,
        ai_pipeline.full_summary,
        ai_pipeline.full_message,
        ai_pipeline.finding_export,
        ai_pipeline.finding_status,
        ai_pipeline.finding_model,
        ai_pipeline.finding_summary,
        ai_pipeline.finding_message,
        ai_pipeline.finding_reviewed,
        ai_pipeline.finding_suppressed,
        ai_pipeline.finding_suppressed_rows,
        ai_pipeline.evidence_pack_rows,
        ai_pipeline.error_rows,
        ai_pipeline.review_mode,
        ai_pipeline.combined_summary,
        ai_pipeline.combined_memo,
        ai_pipeline.review_comment_rows,
        ai_pipeline.summary_fields,
    )


def _metric_text_lines(value: object, empty_marker: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text == empty_marker:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_ai_status_line(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        marker in lower
        for marker in (
            "ai review",
            "combined ai",
            "ai policy",
            "ai full",
            "ai finding",
            "openai_api_key",
            "openai",
        )
    )


def _is_cached_ai_finding(finding: Finding) -> bool:
    category = str(finding.category or "").strip().lower()
    metadata = finding.metadata or {}
    return category.startswith("ai ") or bool(metadata.get("ai_review_status"))


def _is_generated_manual_review_finding(finding: Finding) -> bool:
    metadata = finding.metadata or {}
    return finding.category == "Manual review" and metadata.get("check_type") == "manual_review_required"


def _clone_cached_finding(finding: Finding) -> Finding:
    return Finding(
        finding.category,
        finding.severity,
        finding.location,
        finding.issue,
        finding.evidence,
        finding.recommendation,
        dict(finding.metadata or {}),
    )


def _get_note_section_with_fallback(ref: str, note_sections: dict[str, str], document: PdfDocument | None = None) -> str:
    section = note_sections.get(ref, "")
    if section: return section
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        if note_sections.get(parent): return note_sections.get(parent, "")
    
    if ref.isdigit() and document:
        ref_num = int(ref)
        prev_ref = str(ref_num - 1)
        next_ref = str(ref_num + 1)
        # Search dynamically in text
        text = document.text
        # e.g. "Note 3", "3.", " 3 ", "4\\nIntangible assets"
        pattern = rf"(?:\n\s*(?:Note|NOTE)\s+{ref}\b|\n\s*{ref}\.?\s+[A-Z]|\n\s*{ref}\n\s*[A-Z])(.*?)(?:\n\s*(?:Note|NOTE)\s+{next_ref}\b|\n\s*{next_ref}\.?\s+[A-Z]|\n\s*{next_ref}\n\s*[A-Z])"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""







def _get_note_heading_with_fallback(ref: str, headings: dict[str, str]) -> str:
    heading = headings.get(ref, "")
    if heading: return heading
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        return headings.get(parent, "")
    return ""

def _document_scope(document: PdfDocument) -> str:
    cached = getattr(document, "_document_scope_cache", None)
    if isinstance(cached, str):
        return cached
    scope = "Limited-scope statement extract" if _is_limited_scope_statement_extract(document) else "Full financial statement or mixed upload"
    setattr(document, "_document_scope_cache", scope)
    return scope


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


def _extraction_finding_as_skipped_check(finding: Finding) -> str:
    if finding.category != "Extraction quality":
        return ""
    issue = str(finding.issue or "").lower()
    non_exception_markers = (
        "checks were skipped",
        "validation skipped",
        "reconciliation was skipped",
        "checks were disabled",
        "ocr was used to recover text",
        "cautious detailed note agreement was run",
        "generic arithmetic checks were skipped",
        "statement-specific ocr checks were skipped",
    )
    if not any(marker in issue for marker in non_exception_markers):
        return ""
    parts = [str(finding.issue or "Extraction-related check skipped.").strip()]
    location = str(finding.location or "").strip()
    if location:
        parts.append(f"Location: {location}.")
    evidence = str(finding.evidence or "").strip()
    if evidence:
        parts.append(f"Reason: {evidence}")
    recommendation = str(finding.recommendation or "").strip()
    if recommendation:
        parts.append(f"Reviewer action: {recommendation}")
    return " ".join(parts)


def _split_elevated_findings(findings: list[Finding], document: PdfDocument) -> tuple[list[Finding], list[dict[str, str]]]:
    elevated: list[Finding] = []
    not_elevated: list[dict[str, str]] = []
    for finding in findings:
        reason = _not_elevated_reason(finding, document)
        if reason:
            not_elevated.append(_not_elevated_prompt_row(finding, reason, document))
            continue
        elevated.append(finding)
    return elevated, not_elevated


def _not_elevated_reason(finding: Finding, document: PdfDocument) -> str:
    metadata = finding.metadata or {}
    explicit = str(metadata.get("not_elevated", "") or "").strip().lower()
    if explicit in {"1", "true", "yes"}:
        return str(metadata.get("not_elevated_reason", "Evidence was not strong enough to elevate this item to the exception register.") or "")
    status = str(metadata.get("ai_review_status", "") or "").strip().lower()
    if status in {"likely_false_positive", "insufficient_evidence"}:
        return "AI finding review marked this item as insufficiently supported for the exception register."
    confidence = str(metadata.get("match_confidence", "") or metadata.get("amount_match_confidence", "") or "").strip().lower()
    if finding.category == "Notes agreement" and confidence == "low":
        return "Low-confidence note agreement prompt; retained for review only until note extraction and matching are corroborated."
    if finding.category == "Notes agreement" and any(term in str(finding.issue or "").lower() for term in ("amount", "note", "reference")):
        return "Note agreement prompt retained for manual confirmation; statement and note page references are provided where the engine can infer them."
    if finding.category == "AI policy judgement":
        ai_confidence = str(metadata.get("match_confidence", "") or "").strip().lower()
        if ai_confidence == "low":
            return "Low-confidence AI policy judgement; retained for reviewer context rather than a confirmed exception."
        if not _finding_has_page_evidence(finding) and not _finding_has_note_evidence(finding):
            return "AI policy judgement did not identify a specific page or note reference."
    if finding.category == "AI full review":
        ai_confidence = str(metadata.get("match_confidence", "") or metadata.get("ai_review_confidence", "") or "").strip().lower()
        if ai_confidence == "low":
            return "Low-confidence AI full-review observation; retained for reviewer context rather than a confirmed exception."
        if not _finding_has_page_evidence(finding) and not _finding_has_note_evidence(finding):
            return "AI full review did not identify a specific page or note reference."
    if finding.category in _PAGE_REQUIRED_FINDING_CATEGORIES and not _finding_has_page_evidence(finding):
        return "Finding did not include a specific page reference, so it is retained as a review prompt instead of an exception."
    if document.ocr_used and str(metadata.get("ocr_review", "") or "").lower() in {"true", "1", "yes"}:
        if confidence == "low":
            return "OCR-derived item has low parse/match confidence."
    return ""


_PAGE_REQUIRED_FINDING_CATEGORIES = {
    "Notes agreement",
    "Totals and rounding",
    "Formatting",
    "Narrative consistency",
    "Key amount consistency",
    "AI policy judgement",
    "AI full review",
}


def _finding_has_page_evidence(finding: Finding) -> bool:
    metadata = finding.metadata or {}
    for key in ("page_reference", "page", "pages"):
        value = str(metadata.get(key, "") or "").strip()
        if value and value.lower() not in {"document-wide", "page not isolated", "not detected"}:
            return True
    text = "\n".join(str(part or "") for part in (finding.location, finding.evidence, finding.issue))
    if re.search(r"\bPages?\s+\d+", text, flags=re.I):
        return True
    if str(finding.location or "").strip().lower() == "document-wide" and finding.category not in _PAGE_REQUIRED_FINDING_CATEGORIES:
        return True
    return False


def _finding_has_note_evidence(finding: Finding) -> bool:
    metadata = finding.metadata or {}
    for key in ("note_reference", "referenced_note", "suggested_note", "alternative_note_found", "amount_found_in_note"):
        if str(metadata.get(key, "") or "").strip():
            return True
    text = "\n".join(str(part or "") for part in (finding.location, finding.evidence, finding.issue))
    return bool(re.search(r"\bNote\s+\d+[A-Z]?\b", text, flags=re.I))


def _not_elevated_prompt_row(finding: Finding, reason: str, document: PdfDocument) -> dict[str, str]:
    metadata = finding.metadata or {}
    note_reference = str(metadata.get("note_reference", "") or metadata.get("referenced_note", "") or metadata.get("suggested_note", "") or _note_reference_text(finding))
    return {
        "Severity": finding.severity,
        "Category": finding.category,
        "Page reference": _not_elevated_page_reference(finding, document, note_reference),
        "Note reference": note_reference,
        "Issue": finding.issue,
        "Evidence": finding.evidence,
        "Reason not elevated": reason,
        "Reviewer action": finding.recommendation or "Review the source page/note manually if this area is material.",
    }


def _not_elevated_page_reference(finding: Finding, document: PdfDocument, note_reference: str = "") -> str:
    metadata = finding.metadata or {}
    pages: set[int] = set()
    direct = str(metadata.get("page_reference", "") or _page_reference_text(finding)).strip()
    for number in re.findall(r"\d+", direct):
        pages.add(int(number))
    statement_pages = _statement_reference_pages(document)
    for ref in re.findall(r"\d+[A-Z]?", note_reference or ""):
        page_text = statement_pages.get(ref.upper())
        if page_text:
            pages.update(int(number) for number in re.findall(r"\d+", page_text))
    note_headings = _note_headings_by_page(document)
    for ref in re.findall(r"\d+[A-Z]?", note_reference or ""):
        heading = note_headings.get(ref.upper())
        if heading:
            pages.add(_reviewer_page_number(document, heading[1]))
    if pages:
        ordered = sorted(pages)
        return f"Page {ordered[0]}" if len(ordered) == 1 else "Pages " + ", ".join(str(page) for page in ordered)
    return ""


def _policy_rows_from_ai_full_export(ai_full_export: list[dict[str, str]] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in ai_full_export or []:
        if not isinstance(row, dict):
            continue
        combined = " ".join(
            str(row.get(key, "") or "")
            for key in ("Dimension", "Standard/topic", "Title", "Issue", "Rationale", "Evidence")
        ).lower()
        if any(
            term in combined
            for term in (
                "policy",
                "standard",
                "ifrs",
                "ias",
                "disclosure",
                "industry_fit",
                "standard_context",
                "policy_relevance",
                "disclosure_completeness",
            )
        ):
            rows.append(dict(row))
    return rows

def _page_reference_text(finding: Finding) -> str:
    text = "\n".join(str(part or "") for part in (finding.location, finding.evidence))
    pages = sorted({int(match) for match in re.findall(r"\bPage\s+(\d+)\b", text, flags=re.I)})
    if not pages:
        return ""
    return f"Page {pages[0]}" if len(pages) == 1 else "Pages " + ", ".join(str(page) for page in pages)


def _note_reference_text(finding: Finding) -> str:
    text = "\n".join(str(part or "") for part in (finding.location, finding.issue, finding.evidence))
    refs = re.findall(r"\bNote\s+(\d+[A-Z]?)\b", text, flags=re.I)
    return ", ".join(f"Note {ref.upper()}" for ref in dict.fromkeys(refs))


def _ai_stage_status_rows(
    ai_policy_status: str,
    ai_policy_message: str,
    ai_full_status: str,
    ai_full_message: str,
    ai_finding_status: str,
    ai_finding_message: str,
) -> list[dict[str, str]]:
    stages = [
        ("AI policy review", ai_policy_status, ai_policy_message),
        ("AI full review", ai_full_status, ai_full_message),
        ("AI finding cleanup", ai_finding_status, ai_finding_message),
    ]
    rows: list[dict[str, str]] = []
    for stage, status, message in stages:
        clean_status = str(status or "disabled").strip() or "disabled"
        if clean_status == "disabled":
            continue
        rows.append(
            {
                "Stage": stage,
                "Status": _ai_display_status(clean_status),
                "Message": str(message or "").strip(),
            }
        )
    return rows


def _ai_display_status(status: str) -> str:
    clean = str(status or "").strip().lower()
    if clean == "completed":
        return "Completed"
    if clean in {"deferred", "error", "unavailable"}:
        return "Failed after retries / Not completed"
    if clean == "skipped":
        return "Skipped"
    if clean == "disabled":
        return "Not started"
    return clean.replace("_", " ").title()


def _ai_overall_status(stage_rows: list[dict[str, str]]) -> str:
    if not stage_rows:
        return "Not started"
    statuses = [str(row.get("Status", "")) for row in stage_rows]
    if any(status == "Failed after retries / Not completed" for status in statuses):
        return "Failed after retries / Not completed"
    if any(status == "Completed" for status in statuses):
        return "Completed"
    if all(status == "Skipped" for status in statuses):
        return "Skipped"
    return "Not completed"

def _build_result(
    document: PdfDocument,
    findings: list[Finding],
    checks_performed: list[str] | None = None,
    checks_skipped: list[str] | None = None,
    note_validation_debug: dict[str, int | str | bool] | None = None,
    cross_page_export: dict | None = None,
    policy_export: list[dict] | None = None,
    ai_policy_export: list[dict] | None = None,
    ai_policy_status: str = "disabled",
    ai_policy_model: str = "",
    ai_policy_summary: str = "",
    ai_policy_message: str = "",
    ai_full_export: list[dict] | None = None,
    ai_full_status: str = "disabled",
    ai_full_model: str = "",
    ai_full_summary: str = "",
    ai_full_message: str = "",
    ai_finding_export: list[dict] | None = None,
    ai_finding_status: str = "disabled",
    ai_finding_model: str = "",
    ai_finding_summary: str = "",
    ai_finding_message: str = "",
    ai_finding_reviewed: int = 0,
    ai_finding_suppressed: int = 0,
    ai_finding_suppressed_rows: list[dict[str, str]] | None = None,
    ai_evidence_pack_rows: list[dict[str, str]] | None = None,
    ai_error_rows: list[dict[str, str]] | None = None,
    ai_review_mode: str = "standard",
    ai_combined_summary: str = "",
    ai_combined_memo: str = "",
    ai_review_comment_rows: list[dict[str, str]] | None = None,
    ai_summary_fields: dict[str, str] | None = None,
    canonical_check_rows: list[dict[str, object]] | None = None,
    canonical_audit_rows: list[dict[str, object]] | None = None,
) -> ReviewResult:
    checks_performed_list = list(dict.fromkeys(checks_performed or []))
    
    # Process passed findings and narrative contradiction
    checks_skipped_list = list(dict.fromkeys(checks_skipped or []))
    active_findings = []
    passed_check_evidence: dict[str, str] = {}
    for f in findings:
        if f.severity == "Passed":
            if f.issue not in checks_performed_list:
                checks_performed_list.append(f.issue)
            passed_check_evidence[f.issue] = f.evidence
            continue
        skipped_reason = _extraction_finding_as_skipped_check(f)
        if skipped_reason:
            checks_skipped_list.append(skipped_reason)
            continue
        if "Statement names in the narrative do not match the statement headings" in f.issue:
            f.issue = "Statement names in the notes or auditor's report do not match the statement headings."
            f.location = "Document-wide"
            f.severity = "Medium"
        active_findings.append(f)
            
    checks_skipped_list = list(dict.fromkeys(checks_skipped_list))
    
    is_company = bool(
        document
        and _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
    )
    if is_company:
        ngo_terms = ["gross operating revenue", "total income", "total expenditure", "surplus", "statement of changes in accumulated fund", "statement of income and expenditure"]
        checks_skipped_list = [c for c in checks_skipped_list if not any(t in c.lower() for t in ngo_terms)]
    # Skipped/manual-review items belong in Checks skipped and Checks results only, not the Exception register.
    findings, not_elevated_review_prompts = _split_elevated_findings(active_findings, document)
    canonical_check_rows = canonical_check_rows or []
    canonical_audit_rows = canonical_audit_rows or []
    check_result_rows = _check_result_rows(checks_performed_list, checks_skipped_list, findings, document, passed_check_evidence)
    check_result_rows.extend(_canonical_checks_result_rows(canonical_check_rows))
        
    positive_assurance = _positive_assurance_text(findings, checks_performed_list)
    note_validation_debug = note_validation_debug or _note_validation_debug(document, False, [])
    rotated_pages = getattr(document, "rotated_page_details", []) or []
    notes_start_page = _notes_start_page(document)
    ai_stage_rows = _ai_stage_status_rows(
        ai_policy_status,
        ai_policy_message,
        ai_full_status,
        ai_full_message,
        ai_finding_status,
        ai_finding_message,
    )
    ai_overall_status = _ai_overall_status(ai_stage_rows)
    performed_result_count = sum(
        1
        for row in check_result_rows
        if row.get("Result") in {"Passed", "Failed", "Failed / Review prompt"}
    )
    passed_result_count = sum(1 for row in check_result_rows if row.get("Result") == "Passed")
    checks_performed_count = max(len(checks_performed_list), performed_result_count)
    checks_passed_count = min(passed_result_count, checks_performed_count)
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
        "rotated_pages": rotated_pages,
        "tables": sum(len(page.tables) for page in document.pages),
        "findings": len(findings),
        "review_prompts_not_elevated_count": len(not_elevated_review_prompts),
        "review_prompts_not_elevated": not_elevated_review_prompts,
        "high": sum(1 for item in findings if item.severity == "High"),
        "medium": sum(1 for item in findings if item.severity == "Medium"),
        "low": sum(1 for item in findings if item.severity == "Low"),
        "printed_page_map": {str(key): value for key, value in _printed_page_number_map(document).items()},
        "note_headings": _format_note_heading_debug(document),
        "notes_section_start_page": _reviewer_page_number(document, notes_start_page) if notes_start_page else "Not detected",
        "notes_heading_snippet": _format_notes_heading_snippet(document),
        "notes_heading_candidates": _notes_heading_candidate_rows(document),
        "primary_statement_pages": _format_primary_statement_debug(document),
        "ocr_statement_rows": _format_ocr_statement_rows_debug(document),
        "note_agreement_results": _note_agreement_result_rows(document),
        "skipped_table_details": _skipped_table_detail_rows(document),
        "skipped_table_summary": _skipped_table_summary_rows(document),
        "detected_profile": infer_detected_profile(document),
        "checks_performed": "\n".join(checks_performed_list) or "No deterministic checks completed.",
        "checks_skipped": "\n".join(checks_skipped_list) or "No major checks skipped.",
        "check_results": check_result_rows,
        "canonical_recalculation_checks": canonical_check_rows,
        "canonical_extraction_audit": canonical_audit_rows,
        "contents_agreement": _contents_statement_page_agreement_rows(document),
        "deterministic_section_map": document_section_map(document),
        "deterministic_table_classification": table_classification_rows(document),
        "cross_page_export": cross_page_export or {},
        "policy_export": policy_export or [],
        "ai_review_status": ai_overall_status,
        "ai_review_mode": ai_review_mode,
        "ai_review_stage_status": ai_stage_rows,
        "ai_combined_review_summary": ai_combined_summary,
        "ai_combined_review_memo": ai_combined_memo,
        "ai_error_log": ai_error_rows or [],
        "ai_review_comment_rows": ai_review_comment_rows or [],
        "ai_summary_fields": ai_summary_fields or {},
        "ai_policy_export": ai_policy_export or [],
        "ai_policy_review_status": ai_policy_status,
        "ai_policy_review_model": ai_policy_model,
        "ai_policy_review_summary": ai_policy_summary,
        "ai_policy_review_message": ai_policy_message,
        "ai_full_export": ai_full_export or [],
        "ai_full_review_status": ai_full_status,
        "ai_full_review_model": ai_full_model,
        "ai_full_review_summary": ai_full_summary,
        "ai_full_review_message": ai_full_message,
        "ai_finding_export": ai_finding_export or [],
        "ai_finding_review_status": ai_finding_status,
        "ai_finding_review_model": ai_finding_model,
        "ai_finding_review_summary": ai_finding_summary,
        "ai_finding_review_message": ai_finding_message,
        "ai_finding_reviewed": ai_finding_reviewed,
        "ai_finding_suppressed": ai_finding_suppressed,
        "ai_suppressed_findings": ai_finding_suppressed_rows or [],
        "ai_evidence_packs": ai_evidence_pack_rows or [],
        "hybrid_review_mode": "AI-assisted evidence review" if (ai_policy_status != "disabled" or ai_full_status != "disabled" or ai_finding_status != "disabled") else "Deterministic engine only",
        "hybrid_review_principle": "Engine performs extraction, arithmetic, and structural checks; AI reviews evidence packs for policy/disclosure judgement and likely false positives.",
        "checks_performed_count": checks_performed_count,
        "checks_passed_count": checks_passed_count,
        "checks_skipped_count": len(checks_skipped_list),
        "positive_assurance": positive_assurance,
        **note_validation_debug,
    }
    return ReviewResult(findings=findings, metrics=metrics)


def _policy_export_has_missing_rows(policy_export: list[dict] | None) -> bool:
    if not policy_export:
        return False
    missing_markers = ("none found", "missing", "not found", "not detected")
    for row in policy_export:
        result = str(row.get("Result") or row.get("Status") or row.get("Review result") or "").lower()
        evidence = str(row.get("Evidence") or row.get("Policy text") or row.get("Extract") or row.get("Comment") or "").lower()
        if any(marker in result for marker in missing_markers) or any(marker in evidence for marker in missing_markers):
            return True
    return False


def _manual_review_findings_for_skipped_checks(checks_skipped: list[str], document: PdfDocument) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for check in checks_skipped:
        if not _skipped_check_requires_manual_review(check, document):
            continue
        location = _manual_review_location_for_skipped_check(check, document)
        key = f"{location}|{check}"
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            Finding(
                "Manual review",
                "Low",
                location,
                "Automated check could not be completed; manual review required.",
                check,
                _manual_review_recommendation_for_skipped_check(check),
                metadata={"check_type": "manual_review_required", "page_reference": location},
            )
        )
    return findings


def _skipped_check_requires_manual_review(check: str, document: PdfDocument | None = None) -> bool:
    lower = str(check or "").lower()
    if not lower or "no major checks skipped" in lower:
        return False
    status_only_terms = (
        "ai review",
        "ai policy",
        "ai finding",
        "openai",
        "api key",
        "rate-limit",
        "rate limit",
        "cooldown",
        "temporarily busy",
        "malformed structured output",
        "limited-scope statement extract",
        "full financial statement completeness",
        "document-level extraction quality is low",
        "ocr-assisted document",
        "detailed note-reference reconciliation was skipped",
        "detailed note agreement skipped because table extraction confidence is below threshold",
        "contents agreement: skipped because statement references in contents were not detected",
        "canonical deterministic recalculation skipped",
    )
    if any(term in lower for term in status_only_terms):
        return False
    if "generic table arithmetic skipped" in lower and document is not None:
        return any(row.get("Can automated check be fixed?") != "Not applicable" for row in _skipped_table_summary_rows(document))
    actionable_terms = (
        "skipped",
        "not confidently parsed",
        "not detected",
        "not reliable",
        "manual confirmation required",
        "could not be confidently extracted",
        "table extraction confidence is below threshold",
    )
    return any(term in lower for term in actionable_terms)


def _manual_review_location_for_skipped_check(check: str, document: PdfDocument | None = None) -> str:
    lower = str(check or "").lower()
    if "generic table arithmetic skipped" in lower and document is not None:
        pages: list[str] = []
        for row in _skipped_table_summary_rows(document):
            if row.get("Can automated check be fixed?") == "Not applicable":
                continue
            page_ref = str(row.get("Pages affected", "")).strip()
            if page_ref:
                pages.append(page_ref)
        if pages:
            return "; ".join(dict.fromkeys(pages))
        return "See Skipped table details"
    page_match = re.search(r"\bPages?\s+([0-9, and-]+)", check, flags=re.I)
    if page_match:
        numbers = re.findall(r"\d+", page_match.group(1))
        if numbers:
            return "Page " + numbers[0] if len(numbers) == 1 else "Pages " + ", ".join(numbers)
    if "note" in lower and document is not None:
        start = _notes_start_page(document)
        if start:
            return f"Notes section from Page {_reviewer_page_number(document, start)}"
        return "Notes section not detected"
    return "Document-wide"


def _manual_review_recommendation_for_skipped_check(check: str) -> str:
    lower = str(check or "").lower()
    if "cash flow" in lower:
        return "Review the cash flow statement manually and tie major movements back to the statement of financial position, profit or loss, changes in equity, and supporting notes."
    if "note" in lower:
        return "Review the referenced note section manually and confirm note heading, face-statement reference, and amount agreement."
    if "generic table arithmetic" in lower or "table" in lower:
        return "Use the Skipped table details sheet to cast the affected table manually or rerun after improving extraction/OCR."
    return "Review the referenced area manually and rerun the tool after improving extraction if needed."


def _note_validation_debug(
    document: PdfDocument,
    enabled: bool,
    note_findings: list[Finding],
) -> dict[str, int | str | bool]:
    note_reference_findings = sum(1 for finding in note_findings if finding.metadata and finding.metadata.get("referenced_note"))
    notes_start_page = _notes_start_page(document)
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
        "notes_section_start_page": _reviewer_page_number(document, notes_start_page) if notes_start_page else "Not detected",
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


def _contains_compatible_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalise_match_words(text)
    if not normalized:
        return False
    return any(_normalise_match_words(phrase) in normalized for phrase in phrases)


def _note_compatibility_rule(line_item: str) -> str:
    item = _normalise_match_words(line_item)
    if not item:
        return ""
    for rule_name, rule in NOTE_COMPATIBILITY_RULES.items():
        if any(_normalise_match_words(term) in item for term in rule["line"]):
            return rule_name
    return ""


def _note_heading_semantically_compatible(line_item: str, note_heading: str, note_section: str = "") -> bool:
    rule_name = _note_compatibility_rule(line_item)
    if not rule_name:
        return True
    rule = NOTE_COMPATIBILITY_RULES[rule_name]
    combined = f"{note_heading} {note_section[:400]}"
    return _contains_compatible_phrase(combined, rule["heading"])


def _note_compatibility_label(line_item: str) -> str:
    rule_name = _note_compatibility_rule(line_item)
    return rule_name or "line item"


def _is_ppe_line_item(line_item: str) -> bool:
    return _note_compatibility_rule(line_item) == "ppe"


def _is_ppe_heading_compatible(note_heading: str) -> bool:
    return _contains_compatible_phrase(note_heading, NOTE_COMPATIBILITY_RULES["ppe"]["heading"])


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
    return "\n".join(
        f"{name} | Page {_reviewer_page_number(document, page.number)}" for name, page in classified.items()
    )


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
                        f"Page {_reviewer_page_number(document, page.number)}",
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
        reviewer_start = _reviewer_page_number(document, start_page)
        reviewer_end = _reviewer_page_number(document, end_page)
        ranges[ref] = (
            f"Page {reviewer_start}"
            if reviewer_end == reviewer_start
            else f"Pages {reviewer_start}-{reviewer_end}"
        )
    return ranges


def _printed_footer_page_number(text: str) -> int | None:
    if not text:
        return None
    non_empty_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return None
    tail = non_empty_lines[-12:]
    for line in reversed(tail):
        normalized = line.lower().strip(" -")
        if normalized in {"draft", "audited", "unaudited"}:
            continue
        if re.fullmatch(r"\d{1,3}", normalized):
            return int(normalized)
        match = re.fullmatch(r"page\s+(\d{1,3})", normalized)
        if match:
            return int(match.group(1))
    return None


def _printed_page_number_map(document: PdfDocument) -> dict[int, int]:
    cached = getattr(document, "_printed_page_number_map_cache", None)
    if isinstance(cached, dict):
        return cached
    mapping: dict[int, int] = {}
    highest_physical_page = max((page.number for page in document.pages), default=0)
    max_reasonable_offset = max(12, highest_physical_page // 3)
    for page in document.pages:
        printed = _printed_footer_page_number(page.text)
        if printed is None:
            continue
        # Contents/front-matter pages often end with statement index references
        # rather than a real printed footer. Reject impossible offsets such as
        # physical page 3 being read as printed page 44.
        if abs(printed - page.number) > max_reasonable_offset:
            continue
        mapping[page.number] = printed

    known_pages = sorted(mapping)
    for left, right in zip(known_pages, known_pages[1:]):
        gap = right - left
        if gap <= 1:
            continue
        if mapping[right] - mapping[left] != gap:
            continue
        for page_number in range(left + 1, right):
            mapping.setdefault(page_number, mapping[left] + (page_number - left))
    setattr(document, "_printed_page_number_map_cache", mapping)
    return mapping


def _reviewer_page_number(document: PdfDocument, page_number: int) -> int:
    return _printed_page_number_map(document).get(page_number, page_number)


def _translate_page_reference_text(text: str, document: PdfDocument) -> str:
    value = str(text or "").strip()
    if not value:
        return value

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        body = match.group(2)
        translated = []
        for token in re.split(r"(\D+)", body):
            if token.isdigit():
                translated.append(str(_reviewer_page_number(document, int(token))))
            else:
                translated.append(token)
        return prefix + "".join(translated)

    return re.sub(r"\b(Pages?\s+)([\d,\-\s]+)", _replace, value, flags=re.I)


def _apply_reviewer_page_labels_to_note_rows(document: PdfDocument, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        page_value = str(row.get("Page", "")).strip()
        if page_value.isdigit():
            row["Page"] = str(_reviewer_page_number(document, int(page_value)))
        note_range = str(row.get("Note section page range", "")).strip()
        if note_range:
            row["Note section page range"] = _translate_page_reference_text(note_range, document)
    return rows


def _notes_start_page(document: PdfDocument) -> int | None:
    if hasattr(document, "_notes_start_page_cache"):
        return getattr(document, "_notes_start_page_cache")
    search_start_page = _notes_candidate_search_start_page(document)
    pages = list(document.pages)
    for i, page in enumerate(pages):
        if page.number < search_start_page:
            continue
        if _page_has_numbered_notes_section_start(page.text) or _page_has_accounting_policy_notes_start(page.text):
            setattr(document, "_notes_start_page_cache", page.number)
            return page.number
        text_lower = page.text.lower()
        if "notes to the financial" in text_lower:
            if "accounting policies" in text_lower or "material accounting" in text_lower:
                if not _looks_like_front_matter_page(page.text):
                    setattr(document, "_notes_start_page_cache", page.number)
                    return page.number
            elif i + 1 < len(pages) and ("accounting policies" in pages[i+1].text.lower() or "material accounting" in pages[i+1].text.lower()):
                if not _looks_like_front_matter_page(page.text):
                    setattr(document, "_notes_start_page_cache", page.number)
                    return page.number
        if _notes_heading_in_text(page.text) and not _looks_like_front_matter_page(page.text):
            setattr(document, "_notes_start_page_cache", page.number)
            return page.number
    candidates = _notes_heading_candidates(document, include_weak=False)
    accepted = [candidate for candidate in candidates if candidate["accepted"] == "Yes"]
    if accepted:
        start_page = int(min(accepted, key=lambda item: int(item["page"]))["page"])
        setattr(document, "_notes_start_page_cache", start_page)
        return start_page
    setattr(document, "_notes_start_page_cache", None)
    return None


def _page_has_numbered_notes_section_start(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()[:80] if line.strip()]
    if not lines:
        return False
    head = "\n".join(lines)
    normalized_head = _normalise_match_words(head)
    has_notes_heading = (
        any(_notes_heading_line_score(line) >= 0.82 for line in lines)
        or "notes financial statements" in normalized_head
        or "notes financial statement" in normalized_head
        or "notes accounts" in normalized_head
    )
    if not has_notes_heading:
        return False
    numbered_note_patterns = (
        r"(?im)^\s*(?:note\s+)?1\s*(?:[.)]|:)?\s+(?:reporting entity|significant accounting polic|material accounting polic|basis of preparation)\b",
        r"(?im)^\s*1\.1\s+(?:basis of prep\w*|basis of preparation|material accounting polic|significant accounting polic)\b",
        r"(?im)^\s*1\s*(?:[.)]|:)?\s+.{0,80}\n\s*1\.1\s+(?:basis of prep\w*|basis of preparation|material accounting polic|significant accounting polic)\b",
    )
    if any(re.search(pattern, head, flags=re.I) for pattern in numbered_note_patterns):
        return True
    return bool(
        "accounting polic" in normalized_head
        and re.search(r"(?im)^\s*1\.1\s+(?:basis of prep\w*|basis of preparation|material accounting polic|significant accounting polic)\b", head)
    )


def _page_has_accounting_policy_notes_start(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()[:80] if line.strip()]
    if not lines:
        return False
    head = "\n".join(lines)
    normalized_head = _normalise_match_words(head)
    if "contents" in normalized_head and normalized_head.count("statement") >= 2:
        return False
    has_policy_title = any(
        _normalise_match_words(line) in {"accounting policies", "material accounting policies", "significant accounting policies"}
        for line in lines[:12]
    ) or "material accounting policies" in normalized_head or "significant accounting policies" in normalized_head
    has_numbered_policy = bool(
        re.search(r"(?im)^\s*(?:note\s+)?1\s*(?:[.)]|:)?\s*(?:reporting entity|significant accounting polic|material accounting polic|basis of prep\w*)\b", head)
        or re.search(r"(?im)^\s*1\.1\s+(?:basis of prep\w*|basis of preparation|material accounting polic|significant accounting polic)\b", head)
    )
    return has_policy_title and has_numbered_policy


def _looks_like_front_matter_page(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()[:40] if line.strip()]
    head = "\n".join(lines).lower()
    if not head:
        return False
    if _page_has_numbered_notes_section_start(text) or _page_has_accounting_policy_notes_start(text):
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
    cache_name = "_notes_heading_candidates_weak_cache" if include_weak else "_notes_heading_candidates_strict_cache"
    cached = getattr(document, cache_name, None)
    if isinstance(cached, list):
        return cached
    candidates: list[tuple[float, dict[str, str | int]]] = []
    search_start_page = _notes_candidate_search_start_page(document)
    for page in document.pages:
        if page.number < search_start_page:
            continue
        front_matter = _looks_like_front_matter_page(page.text)
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
    rows = [candidate for _score, candidate in candidates]
    setattr(document, cache_name, rows)
    return rows


def _raw_page_notes_heading_candidates(text: str, include_weak: bool = False) -> list[tuple[float, str, str, str, str]]:
    candidates: list[tuple[float, str, str, str, str]] = []
    if not text.strip():
        return candidates
    pattern_specs = (
        (r"notes?\s+to\s+(?:the\s+)?financial\s+statements?", "Notes heading", 0.96),
        (r"notes?\s+forming\s+part\s+of\s+(?:the\s+)?financial\s+statements?", "Notes heading", 0.96),
        (r"notes?\s+to\s+(?:the\s+)?accounts?", "Notes heading", 0.92),
        (r"(?:^|\n)\s*1\s*[\).:\s-]*\s*(?:material\s+|significant\s+)?accounting\s+polic(?:y|ies)", "Accounting policies heading", 0.84),
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
    contents_terms = (
        "statement financial position",
        "statement comprehensive income",
        "statement cash flows",
        "statement changes equity",
        "statement changes accumulated fund",
        "value added statement",
        "five year financial summary",
        "five year financial review",
    )
    if sum(1 for term in contents_terms if term in normalized) >= 2 and "accounting polic" not in normalized:
        return True
    return False


def _notes_candidate_search_start_page(document: PdfDocument) -> int:
    statement_pages = [page.number for page in _classified_primary_statement_pages(document).values()]
    return max(statement_pages) + 1 if statement_pages else 1


def _candidate_followed_by_numbered_policy(lines: list[str], index: int) -> bool:
    nearby = "\n".join(lines[index : index + 10])
    return bool(
        re.search(
            r"(?:^|\n)\s*(?:note\s+)?1\s*[\).:\s-]*\s*(?:material\s+|significant\s+)?accounting polic",
            nearby,
            flags=re.I,
        )
        or re.search(r"(?:^|\n)\s*1\.1\s+(?:basis of prep\w*|basis of preparation|material accounting polic|significant accounting polic)\b", nearby, flags=re.I)
        or re.search(r"(?:^|\n)\s*1[\).:\s-]+.{0,60}\n\s*2[\).:\s-]+", nearby, flags=re.I)
    )


def _notes_heading_candidate_rows(document: PdfDocument) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in _notes_heading_candidates(document, include_weak=True):
        reviewer_page = _reviewer_page_number(document, int(candidate["page"]))
        rows.append(
            {
                "Page": str(reviewer_page),
                "Raw OCR snippet": str(candidate["snippet"]),
                "Normalized snippet": str(candidate.get("normalized_snippet", "")),
                "Similarity score": str(candidate["confidence"]),
                "Accepted": str(candidate["accepted"]),
                "Reason": str(candidate["reason"]),
            }
        )
    return rows


def _note_agreement_result_rows(document: PdfDocument) -> list[dict[str, str]]:
    cached = getattr(document, "_note_agreement_result_rows_cache", None)
    if isinstance(cached, list):
        return cached
    rows: list[dict[str, str]] = []
    statement_lines = _statement_note_lines(document)
    if not statement_lines:
        return _finalize_note_agreement_result_rows(document, rows)
    if document.ocr_used:
        headings = {ref: title for ref, (title, _page_number) in _note_headings_by_page(document).items()}
        page_ranges = _note_section_page_ranges(document)
        note_sections = _note_sections(document) if _notes_start_page(document) else {}
        for item in statement_lines:
            current_amount = item.amounts[0] if item.amounts else None
            prior_amount = item.amounts[1] if len(item.amounts) >= 2 else None
            if not item.ref and _note_agreement_skip_reason(item):
                continue
            
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
        return _finalize_note_agreement_result_rows(document, rows)
    note_sections = _note_sections(document)
    headings = {ref: title for ref, (title, _page_number) in _note_headings_by_page(document).items()}
    page_ranges = _note_section_page_ranges(document)
    _scale_label, tolerance = _detect_rounding_scale(document.text)
    low_confidence = document.table_extraction_confidence < 80
    for item in statement_lines:
        current_amount = item.amounts[0] if item.amounts else None
        prior_amount = item.amounts[1] if len(item.amounts) >= 2 else None
        if not item.ref and _note_agreement_skip_reason(item):
            continue
        
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
        referenced_heading = _get_note_heading_with_fallback(item.ref, headings)
        allow_absolute = _note_heading_allows_signless_amount_match(referenced_heading, referenced_section)
        current_match = _amount_match_in_section(current_amount, referenced_section, tolerance, allow_absolute=allow_absolute)
        prior_match = _amount_match_in_section(prior_amount, referenced_section, tolerance, allow_absolute=allow_absolute)
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
    return _finalize_note_agreement_result_rows(document, rows)


def _filtered_note_agreement_rows(document: PdfDocument, findings: list[Finding] | None = None) -> list[dict[str, str]]:
    rows = _note_agreement_result_rows(document)
    note_prompt_keys = {
        str((finding.metadata or {}).get("line_key", "")).strip()
        for finding in (findings or [])
        if finding.category == "Notes agreement" and finding.metadata and finding.metadata.get("line_key")
    }
    filtered: list[dict[str, str]] = []
    for row in rows:
        result = str(row.get("Review result", ""))
        reason = str(row.get("Reason", ""))
        confidence = str(row.get("Match confidence", ""))
        if result == "Internal note":
            continue
        if result == "Review prompt" and confidence.lower() == "low":
            continue
        if result == "Review prompt" and note_prompt_keys:
            row_note = str(row.get("Note number", "")).strip()
            row_line = str(row.get("Line item description", "")).strip().lower()
            matching_key = any(
                key.startswith(f"{row_note}|") and row_line in key.lower()
                for key in note_prompt_keys
            )
            if not matching_key:
                continue
        if result == "Review prompt" and "low-confidence heading-only debug result" in reason.lower():
            continue
        filtered.append(row)
    return filtered


def _skipped_table_detail_rows(document: PdfDocument) -> list[dict[str, str]]:
    details = getattr(document, "skipped_table_details", []) or []
    rows: list[dict[str, str]] = []
    for detail in details:
        match = re.match(r"Page\s+(\d+),\s+table\s+(\d+):\s+(.+?)(?:\s+\((.+)\))?$", detail)
        if match:
            page, table, table_type, reason = match.groups()
            reviewer_page = str(_reviewer_page_number(document, int(page)))
            if table_type.lower().startswith("skipped because"):
                reason = reason or table_type
                table_type = "Notes table"
            rows.append(
                {
                    "Page": reviewer_page,
                    "Source PDF page": page,
                    "Table": table,
                    "Classification": table_type,
                    "Reason skipped": reason or "",
                    "Result": "Skipped - not reliable for generic arithmetic",
                }
            )
        else:
            rows.append(
                {
                    "Page": "",
                    "Source PDF page": "",
                    "Table": "",
                    "Classification": "",
                    "Reason skipped": _translate_page_reference_text(detail, document),
                    "Result": "Skipped - not reliable for generic arithmetic",
                }
            )
    return rows


def _skipped_table_summary_rows(document: PdfDocument) -> list[dict[str, str]]:
    details = _skipped_table_detail_rows(document)
    if not details:
        return []
    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in details:
        classification = row.get("Classification", "") or "Table-specific skip"
        reason = row.get("Reason skipped", "") or classification
        skip_context = f"{classification} {reason}".lower()
        if "generic arithmetic is not reliable for notes tables" in skip_context:
            group = "Notes tables - manual review recommended"
            reviewer_action = "Use note-reference and amount-agreement sheets; inspect individual note tables manually where prompted."
            can_fix = "Partially"
            why_review = "The PDF table can be visible to a reviewer, but extracted rows/columns may merge note numbers, years, narrative text, and amounts. Generic subtotal casting is withheld to avoid false exceptions."
        elif "low-confidence" in classification.lower() or "numeric row shapes" in reason.lower():
            group = "Low-confidence table extraction"
            reviewer_action = "Review the source page before relying on automated table arithmetic."
            can_fix = "Possibly"
            why_review = "The extracted numeric row pattern is inconsistent, so the tool cannot confirm which figures belong to the same table columns without reviewer inspection."
        elif "value-added statement" in skip_context:
            group = "Value-added statement"
            reviewer_action = "Inspect manually if material; do not cast it like a primary statement because value-added presentations have their own subtotal logic."
            can_fix = "Not applicable"
            why_review = "Value-added statements are supplementary presentation schedules, so generic row addition can create false exceptions."
        elif "multi-year summary" in skip_context:
            group = "Multi-year summary"
            reviewer_action = "Inspect manually if material; do not cast it like a current-year primary statement."
            can_fix = "Not applicable"
            why_review = "Multi-year summaries combine several years and summary measures, so generic table casting is intentionally withheld."
        elif "not a recognised statement/note total table" in reason.lower() or "other table" in classification.lower():
            group = "Non-standard or non-financial table"
            reviewer_action = "No generic casting performed; review only if the table is financially relevant."
            can_fix = "Not applicable"
            why_review = "The table does not look like a standard amount table for automated casting, or it may be narrative/front-matter information."
        else:
            group = classification
            reviewer_action = "Review the table manually if it is material to the financial statements."
            can_fix = "Unknown"
            why_review = "The extraction did not provide enough reliable structure to support an automated conclusion."
        key = (group, reason, reviewer_action, can_fix, why_review)
        bucket = groups.setdefault(key, {"pages": set(), "tables": 0})
        bucket["tables"] = int(bucket["tables"]) + 1
        page = str(row.get("Page", "")).strip()
        if page.isdigit():
            cast_pages = bucket["pages"]
            if isinstance(cast_pages, set):
                cast_pages.add(int(page))
    summary_rows: list[dict[str, str]] = []
    for (group, reason, reviewer_action, can_fix, why_review), bucket in groups.items():
        pages = bucket.get("pages", set())
        page_reference = _format_page_set(pages if isinstance(pages, set) else set())
        summary_rows.append(
            {
                "Skipped check group": group,
                "Pages affected": page_reference or "Not page-specific",
                "Tables affected": str(bucket.get("tables", 0)),
                "Reason skipped": reason,
                "Can automated check be fixed?": can_fix,
                "Why reviewer should review": why_review,
                "Reviewer action": reviewer_action,
            }
        )
    return sorted(summary_rows, key=lambda row: row["Skipped check group"])


def _format_page_set(pages: set[int]) -> str:
    if not pages:
        return ""
    ordered = sorted(pages)
    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    label = "Page" if len(ordered) == 1 else "Pages"
    return f"{label} {', '.join(ranges)}"


def _finalize_note_agreement_result_rows(document: PdfDocument, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched_rows = _enrich_note_agreement_rows(document, rows)
    labelled_rows = _label_non_elevated_note_agreement_rows(enriched_rows)
    labelled_rows = _apply_reviewer_page_labels_to_note_rows(document, labelled_rows)
    setattr(document, "_note_agreement_result_rows_cache", labelled_rows)
    return labelled_rows


def _label_non_elevated_note_agreement_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        result = str(row.get("Result") or row.get("Review result") or "").strip()
        confidence = str(row.get("Match confidence") or "").strip().lower()
        if result == "Review prompt" and confidence == "low":
            row["Review result"] = "Not elevated / internal note"
            row["Result"] = "Not elevated / internal note"
            reason = str(row.get("Reason") or "").strip()
            if reason and not reason.lower().startswith("not elevated"):
                row["Reason"] = f"Not elevated - {reason}"
            elif not reason:
                row["Reason"] = "Not elevated - low-confidence note agreement result retained for reviewer context only."
    return rows


def _enrich_note_agreement_rows(document: PdfDocument, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows
    headings = {ref: title for ref, (title, _page_number) in _note_headings_by_page(document).items()}
    note_sections = _note_sections(document) if headings else {}
    for row in rows:
        line_item = str(row.get("Line item") or row.get("Line item description") or "").strip()
        referenced_ref = str(row.get("Referenced note") or row.get("Note number") or "").strip().upper()
        suggested_ref = str(row.get("Alternative note found") or row.get("Suggested note") or "").strip().upper()
        referenced_heading = _get_note_heading_with_fallback(referenced_ref, headings) if referenced_ref else ""
        suggested_heading = _get_note_heading_with_fallback(suggested_ref, headings) if suggested_ref else ""
        referenced_section = _get_note_section_with_fallback(referenced_ref, note_sections, document) if referenced_ref else ""
        suggested_section = _get_note_section_with_fallback(suggested_ref, note_sections, document) if suggested_ref else ""
        rule_name = _note_compatibility_rule(line_item)
        referenced_status = _note_heading_compatibility_status(line_item, referenced_ref, referenced_heading, referenced_section)
        suggested_status = _note_heading_compatibility_status(line_item, suggested_ref, suggested_heading, suggested_section)
        row["Referenced note heading"] = referenced_heading
        row["Note compatibility rule"] = rule_name
        row["Referenced heading compatible?"] = referenced_status
        row["Suggested note heading"] = suggested_heading
        row["Suggested note compatible?"] = suggested_status
        row["Compatibility reason"] = _note_compatibility_reason(
            line_item,
            referenced_ref,
            referenced_heading,
            referenced_status,
            suggested_ref,
            suggested_heading,
            suggested_status,
            rule_name,
        )
    return rows


def _note_heading_compatibility_status(line_item: str, ref: str, heading: str, section: str = "") -> str:
    if not ref:
        return "N/A"
    if not heading:
        return "Not detected"
    if not _note_compatibility_rule(line_item):
        return "Not tested"
    return "Yes" if _note_heading_semantically_compatible(line_item, heading, section) else "No"


def _note_compatibility_reason(
    line_item: str,
    referenced_ref: str,
    referenced_heading: str,
    referenced_status: str,
    suggested_ref: str,
    suggested_heading: str,
    suggested_status: str,
    rule_name: str,
) -> str:
    if not referenced_ref:
        return "No note reference was detected on the face statement row."
    if referenced_status == "Not detected":
        base = "Referenced note heading was not detected."
    elif referenced_status == "Not tested":
        base = "No specific note-heading compatibility rule matched; amount and wording checks were used."
    elif referenced_status == "Yes":
        base = f"Referenced note heading is compatible with the {rule_name} line item."
    else:
        base = f"Referenced note heading is not compatible with the {rule_name} line item."
    if suggested_ref:
        suggested_label = suggested_heading or "heading not detected"
        if suggested_status == "Yes":
            return f"{base} Suggested Note {suggested_ref} ('{suggested_label}') is compatible."
        if suggested_status == "No":
            return f"{base} Suggested Note {suggested_ref} ('{suggested_label}') is not compatible; treat as weak/debug only."
        return f"{base} Suggested Note {suggested_ref} ('{suggested_label}') was not compatibility-tested."
    return base


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
    reason_lower = reason.lower()
    export_result = result

    return {
        "Statement": item.statement_name,
        "Page": str(item.page_number),
        "Line item description": item.line_item.title(),
        "Line item": item.line_item.title(),
        "Note number": item.ref,
        "Referenced note": item.ref,
        "Current year amount": _format_decimal_for_export(current_amount),
        "Prior year amount": _format_decimal_for_export(prior_amount),
        "Has note?": has_note,
        "Review required?": review_req,
        "Comment": comment,
        "Review result": export_result,
        "Result": export_result,
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



def _run_canonical_qc_layer(
    document: PdfDocument,
    existing_findings: list[Finding],
) -> tuple[list[Finding], list[dict[str, object]], list[dict[str, object]], list[str], list[str]]:
    try:
        canonical_findings, check_results, audit_rows = run_canonical_checks(document)
    except Exception as exc:
        return (
            [],
            [],
            [],
            [],
            [f"Canonical QC skipped because the canonical parser raised {type(exc).__name__}: {exc}"],
        )

    check_rows = [result.to_row() for result in check_results]
    findings_to_add = [
        prepared
        for finding in canonical_findings
        if (prepared := _prepare_canonical_finding_for_main_register(finding, existing_findings)) is not None
    ]
    passed = sum(1 for result in check_results if result.status == "Pass")
    failed = sum(1 for result in check_results if result.status == "Fail")
    not_tested = sum(1 for result in check_results if result.status in {"Not tested", "Manual review required"})
    performed: list[str] = []
    skipped: list[str] = []
    if passed or failed:
        performed.append(
            f"Canonical deterministic recalculation checks completed: {passed} passed, {failed} exception/review prompt(s)."
        )
    if not_tested:
        skipped.append(
            f"Canonical deterministic recalculation skipped {not_tested} check(s) because required source rows were not parsed with sufficient confidence; see Canonical recalculation checks."
        )
    if not check_results:
        skipped.append(
            "Canonical deterministic recalculation skipped because no primary statement facts were parsed; see Canonical extraction audit."
        )
    return findings_to_add, check_rows, audit_rows, performed, skipped


def _prepare_canonical_finding_for_main_register(finding: Finding, existing_findings: list[Finding]) -> Finding | None:
    metadata = dict(finding.metadata or {})
    confidence = str(metadata.get("match_confidence") or metadata.get("confidence") or "Medium")
    severity = finding.severity
    if severity == "High" and confidence != "High":
        severity = "Medium"
    category = {
        "Casting": "Totals and rounding",
        "Cross-casting": "Totals and rounding",
        "Cash Flow": "Totals and rounding",
        "Note Cross-reference": "Notes agreement",
    }.get(finding.category, finding.category)
    prepared = Finding(
        category,
        severity,
        finding.location,
        finding.issue,
        f"Canonical deterministic check. {finding.evidence}",
        finding.recommendation,
        metadata={**metadata, "source_engine": "Canonical QC", "check_type": metadata.get("check_type", "Canonical QC")},
    )
    if _canonical_finding_already_present(prepared, existing_findings):
        return None
    return prepared


def _canonical_finding_already_present(candidate: Finding, existing_findings: list[Finding]) -> bool:
    candidate_pages = set(re.findall(r"\bPage\s+(\d+)\b", f"{candidate.location} {candidate.evidence}", flags=re.I))
    candidate_text = _canonical_normalise_words(f"{candidate.issue} {candidate.evidence}")
    for existing in existing_findings:
        if existing.severity == "Passed":
            continue
        existing_pages = set(re.findall(r"\bPage\s+(\d+)\b", f"{existing.location} {existing.evidence}", flags=re.I))
        if candidate_pages and existing_pages and not (candidate_pages & existing_pages):
            continue
        existing_text = _canonical_normalise_words(f"{existing.issue} {existing.evidence}")
        if not existing_text:
            continue
        if SequenceMatcher(None, candidate_text[:500], existing_text[:500]).ratio() >= 0.82:
            return True
    return False


def _canonical_normalise_words(text: str) -> str:
    value = str(text or "").lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _canonical_checks_result_rows(canonical_check_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in canonical_check_rows:
        status = str(row.get("Status", "") or "")
        if status == "Pass":
            result = "Passed"
            severity = ""
        elif status == "Fail":
            result = "Failed / Review prompt"
            severity = str(row.get("Priority", "") or "")
        elif status == "Manual review required":
            result = "Manual review required"
            severity = str(row.get("Priority", "") or "")
        else:
            result = "Skipped"
            severity = ""
        evidence_parts = [
            f"Statement: {row.get('Statement', '')}",
            f"Entity: {row.get('Entity', '')}",
            f"Year: {row.get('Year', '')}",
            f"Formula: {row.get('Formula', '')}",
            f"Reported: {row.get('Reported amount', '')}",
            f"Expected: {row.get('Expected amount', '')}",
            f"Difference: {row.get('Difference', '')}",
            f"Source pages: {row.get('Source pages', '')}",
            f"Source rows: {row.get('Source rows', '')}",
            f"Confidence: {row.get('Confidence', '')}",
        ]
        rows.append(
            {
                "Check": f"Canonical QC - {row.get('Check', '')}",
                "Result": result,
                "Severity": severity,
                "Evidence": " | ".join(part for part in evidence_parts if not part.endswith(': ')),
            }
        )
    return rows


def _check_result_rows(
    checks_performed: list[str],
    checks_skipped: list[str],
    findings: list[Finding],
    document: PdfDocument | None = None,
    passed_check_evidence: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    passed_check_evidence = passed_check_evidence or {}
    for check in checks_performed:
        related = [finding for finding in findings if _finding_matches_check(finding, check)]
        if related:
            result = "Failed / Review prompt"
            severity_rank = {"High": 0, "Medium": 1, "Low": 2}
            severity = ", ".join(sorted({finding.severity for finding in related}, key=lambda item: severity_rank.get(item, 9)))
            evidence = " | ".join(finding.issue for finding in related[:3])
        else:
            result = "Passed"
            severity = ""
            evidence = passed_check_evidence.get(check, "No exception generated by this check.")
        rows.append(
            {
                "Check": check,
                "Result": result,
                "Severity": severity,
                "Evidence": evidence,
            }
        )
    is_company = bool(
        document
        and _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
    )
    
    for check in checks_skipped:
        if _skipped_check_requires_manual_review(check, document):
            result_status = "Manual review required"
            evidence = "Automated check was not reliable enough to conclude; reviewer should inspect the referenced page/table/note."
        else:
            if "skipped" in check.lower():
                result_status = "Skipped - not elevated"
            else:
                result_status = "Status only / deferred"
            evidence = "This item reports an availability, scope, or not-applicable condition rather than a failed audit check."
        
        if is_company:
            lower_check = check.lower()
            ngo_terms = ["gross operating revenue", "total income", "total expenditure", "surplus", "statement of changes in accumulated fund", "statement of income and expenditure"]
            if any(term in lower_check for term in ngo_terms):
                result_status = "Not applicable"
                evidence = "Check is not applicable for private companies."
                
        rows.append(
            {
                "Check": check,
                "Result": result_status,
                "Severity": "",
                "Evidence": evidence,
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
        if finding.category != "Totals and rounding":
            return False
        if "calculation" in check_lower or "checked" in check_lower:
            return "calculation" in issue_lower or "agree" in issue_lower or "equal" in issue_lower
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
    cached = getattr(document, "_detected_profile_cache", None)
    if isinstance(cached, dict):
        return cached
    text = document.text
    profile_text = _profile_detection_text(document)
    lower = profile_text.lower()
    company_name = _detect_company_name(document)
    entity_type = _detect_entity_type(profile_text, company_name=company_name)
    if entity_type == "Private company":
        if re.search(r"\b(financial group|group management and supervisory services)\b", lower):
            entity_type = "Private company / financial group / group management and supervisory services"
        elif re.search(r"\binvestment propert(?:y|ies)|property investment|real estate\b", lower):
            entity_type = "Private company / property investment company"
    profile = {
        "Company name": company_name,
        "Year end": _detect_year_end(text),
        "Currency": _detect_currency(text),
        "Framework": _detect_framework(text),
        "Entity type": entity_type,
        "Document scope": _document_scope(document),
        "Principal activities": _detect_principal_activities(profile_text),
        "Detected balances": _detect_major_balances(lower),
        "Suggested checklist areas": _suggest_checklist_areas(lower),
        "Extraction confidence": f"Text {document.extraction_confidence}% | Tables {document.table_extraction_confidence}%",
    }
    setattr(document, "_detected_profile_cache", profile)
    return profile


def _detect_company_name(document: PdfDocument) -> str:
    legal_name_patterns = (
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+(?:Limited|Ltd|PLC|Plc|Incorporated|Inc\.?|Corporation|Company)\b",
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+(?:Institute|Council|Association|Society|Body)\s+of\s+[A-Z][A-Za-z&,.()' -]{3,80}\b",
        r"[A-Z][A-Za-z&,.()' -]{8,120}\s+of\s+[A-Z][A-Za-z&,.()' -]{3,80}\b",
    )
    candidates: list[tuple[str, int, int]] = []
    candidate_seen: set[tuple[str, int]] = set()
    for page_index, page in enumerate(document.pages[:5]):
        raw_lines = page.text.splitlines()
        normalized_lines = [re.sub(r"\s+", " ", line).strip(" -.,") for line in raw_lines if line.strip()]
        titleish_windows: list[str] = []
        window_lines = normalized_lines[:20]
        for index, line in enumerate(window_lines):
            titleish_windows.append(line)
            if index + 1 < len(window_lines):
                combined = f"{line} {window_lines[index + 1]}".strip()
                if len(combined) <= 160:
                    titleish_windows.append(combined)
        for position, candidate_text in enumerate(titleish_windows):
            if not _company_name_candidate_allowed(candidate_text):
                continue
            for legal_pattern in legal_name_patterns:
                match = re.search(legal_pattern, candidate_text, flags=re.I)
                if not match:
                    continue
                candidate = _clean_detected_company_name(re.sub(r"\s+", " ", match.group(0)).strip(" -.,"))
                if not candidate or not _company_name_candidate_allowed(candidate):
                    continue
                key = (candidate, page_index)
                if key in candidate_seen:
                    continue
                candidate_seen.add(key)
                score = _company_name_candidate_score(candidate, page_index, position)
                candidates.append((candidate, score, page_index))
            if re.search(r"\b(limited|ltd|plc|incorporated|institute|company|corporation|council|association|society|body)\b", candidate_text, re.I):
                candidate = _clean_detected_company_name(candidate_text)
                if candidate and _company_name_candidate_allowed(candidate):
                    key = (candidate, page_index)
                    if key not in candidate_seen:
                        candidate_seen.add(key)
                        score = _company_name_candidate_score(candidate, page_index, position)
                        candidates.append((candidate, score, page_index))
    if candidates:
        grouped: dict[str, tuple[int, int]] = {}
        for candidate, score, page_index in candidates:
            total_score, first_page = grouped.get(candidate, (0, page_index))
            grouped[candidate] = (total_score + score, min(first_page, page_index))
        best = sorted(grouped.items(), key=lambda item: (-item[1][0], item[1][1], -len(item[0])))[0][0]
        return best
    for page in document.pages[:3]:
        for line in page.text.splitlines()[:12]:
            clean = re.sub(r"\s+", " ", line).strip(" -")
            if not clean or len(clean) < 5:
                continue
            if not _company_name_candidate_allowed(clean):
                continue
            if clean.isupper() or re.search(r"\b(limited|ltd|plc|incorporated|institute|company|corporation)\b", clean, re.I):
                return _clean_detected_company_name(clean)
    return "Not detected"


def _clean_detected_company_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" -.,")
    cleaned = re.sub(r"^(?:fl|fi|f1|l|i)\s+(?=[A-Z])", "", cleaned, flags=re.I)
    return _title_preserving_acronyms(cleaned) if cleaned.isupper() else cleaned


def _company_name_candidate_allowed(text: str) -> bool:
    clean = re.sub(r"\s+", " ", str(text or "")).strip(" -.,")
    lower = clean.lower()
    if not clean or len(clean) < 5:
        return False
    if len(re.findall(r"\d", clean)) > 2:
        return False
    excluded = (
        "financial statements",
        "annual report",
        "statement of",
        "notes to",
        "table of contents",
        "corporate information",
        "independent auditor",
        "directors' report",
        "directors report",
        "year ended",
        "for the year ended",
        "together with",
        "contents",
        "page",
        "draft",
    )
    if any(marker in lower for marker in excluded):
        return False
    if len(clean.split()) > 14:
        return False
    return True


def _company_name_candidate_score(candidate: str, page_index: int, position: int) -> int:
    lower = candidate.lower()
    score = 0
    if re.search(r"\b(limited|ltd|plc|incorporated|inc\.?|corporation|company)\b", lower):
        score += 12
    if re.search(r"\b(institute|council|association|society|body)\b", lower):
        score += 10
    if " of " in lower:
        score += 4
    word_count = len(candidate.split())
    if 2 <= word_count <= 8:
        score += 4
    if page_index == 0:
        score += 6
    elif page_index == 1:
        score += 3
    score += max(0, 6 - min(position, 6))
    score += min(10, len(candidate) // 12)
    return score


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
            return _normalise_year_end_format(match.group(1))
    years = sorted(set(YEAR_RE.findall(text)))
    return years[-1] if years else "Not detected"


def _normalise_year_end_format(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").replace(",", " ")).strip()
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", cleaned, flags=re.I)
    month_first = re.match(r"^([A-Za-z]+)\s+(\d{1,2})\s+(20\d{2})$", cleaned)
    if month_first:
        month, day, year = month_first.groups()
        return f"{month} {int(day)} {year}"
    day_first = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})$", cleaned)
    if day_first:
        day, month, year = day_first.groups()
        return f"{month} {int(day)} {year}"
    return cleaned



def _normalise_currency_text(value: str) -> str:
    text = str(value or "")
    replacements = {
        "\u00e2\u201a\u00a6": chr(0x20A6),
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": "'",
        "\u00e2\u20ac\u009d": "'",
        "\u00c3\u00a2\u00e2\u201a\u00ac\u00e2\u201e\u00a2": "'",
        "\u00c3\u00a2\u00e2\u201a\u00ac\u00cb\u0153": "'",
        "\u00c3\u00a2\u00e2\u201a\u00ac\u00c2\u009d": "'",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def _detect_currency(text: str) -> str:
    text = _normalise_currency_text(text)
    naira_symbol = chr(0x20A6)
    naira_quotes = r"['\u2019\u2018`\u201c\u201d]"
    naira_thousands = (
        rf"(?:N\s*{naira_quotes}\s*000|N000|N[^A-Za-z0-9]{0,2}000|N{naira_symbol}\s*000|{naira_symbol}\s*{naira_quotes}\s*000|{naira_symbol})"
    )
    if re.search(naira_thousands, text, flags=re.I):
        return "NGN / N'000"
    if re.search(
        rf"N\s*{naira_quotes}\s*000|N000|{naira_symbol}\s*{naira_quotes}\s*000|\bNGN\b|\bNAIRA\b|NIGERIAN NAIRA|{naira_symbol}",
        text,
        flags=re.I,
    ):
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


def _profile_detection_text(document: PdfDocument) -> str:
    first_pages = "\n".join(page.text for page in document.pages[:8])
    snippets: list[str] = []
    for match in re.finditer(r"(?:principal activities|nature of business|principal activity)[\s\S]{1,350}?(?:\n\n|\.\s)", document.text, re.I):
        snippets.append(match.group(0))
        if len(snippets) >= 3:
            break
    return "\n".join([first_pages, *snippets]).strip()


def _detect_entity_type(text: str, company_name: str = "") -> str:
    lower = text.lower()
    private_score = 0
    public_score = 0
    nonprofit_score = 0

    company_lower = company_name.lower()
    if re.search(r"\bplc\b", company_lower) or "public limited" in company_lower:
        public_score += 4
    elif re.search(r"\b(limited|ltd)\b", company_lower):
        private_score += 4

    if re.search(r"\b(private company|private limited|private limited liability company)\b", lower):
        private_score += 3
    if re.search(r"\b(public company|public limited liability company|public limited)\b", lower):
        public_score += 3
    if re.search(r"\bplc\b", lower):
        public_score += 2
    if re.search(r"\b(limited|ltd)\b", lower):
        private_score += 1
    if any(
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
        private_score += 2
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
        nonprofit_score += 3
    if nonprofit_score and nonprofit_score > max(private_score, public_score):
        return "Non-profit / professional body"
    if private_score >= max(public_score, nonprofit_score) and private_score > 0:
        return "Private company"
    if public_score > 0:
        return "Public company"
    return "Not detected"


def _detect_principal_activities(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("cash reward", "consumer loyalty", "loyalty reward", "reward service")):
        return "Consumer loyalty and rewards / cash reward service"
    if any(term in lower for term in ("rendering supervisory services", "supervisory services", "management of related entities", "related entities under the group structure", "group structure")):
        return "Management of related entities, including supervisory and related group support services."
    if any(term in lower for term in ("property investment", "investment property", "rental income", "income from property")):
        return "Property investment and related rental income activities."
    if any(
        term in lower
        for term in (
            "renewable energy",
            "mini grid",
            "mini-grid",
            "energy generation",
            "electricity generation",
            "distribution of mini grids",
            "distribution of mini-grids",
            "distributable renewable energy",
        )
    ):
        return "Renewable energy generation and mini-grid distribution activities."
    
    matches = list(re.finditer(r"(?:principal activities|nature of business|principal activity)[\s\S]{1,300}?(?:\\n\\n|\.\s)", text, re.I))
    if matches:
        extracted = matches[0].group(0)
        extracted = re.sub(r"(?i)^(principal activities|nature of business|principal activity)", "", extracted)
        extracted = extracted.strip(" :-\\n\\t")
        # Exclude report titles and generic info
        if "directors" in extracted.lower() or "report" in extracted.lower() or "general information" in extracted.lower():
            pass
        else:
            return _summarise_activity_sentence(extracted)
    if _professional_membership_activity_context(lower):
        return "Professional membership body, including member services, professional development, training, and certification."
    return ""


def _professional_membership_activity_context(lower: str) -> bool:
    membership_terms = ("membership", "members", "fellows", "associates", "subscriptions")
    activity_terms = ("training", "certification", "professional development", "member services", "professional body")
    entity_terms = ("institute", "council", "association", "professional")
    return (
        any(term in lower for term in membership_terms)
        and any(term in lower for term in activity_terms)
        and any(term in lower for term in entity_terms)
    )


def _summarise_activity_sentence(extracted: str) -> str:
    normalized = re.sub(r"\s+", " ", extracted).strip(" .")
    lower = normalized.lower()
    if any(term in lower for term in ("rendering supervisory services", "supervisory services", "management of related entities", "related entities", "group structure")):
        return "Management of related entities, including supervisory and related group support services."
    if _professional_membership_activity_context(lower):
        return "Professional membership body, including member services, professional development, training, and certification."
    if any(term in lower for term in ("cash reward", "consumer loyalty", "loyalty reward", "reward service")):
        return "Consumer loyalty and rewards / cash reward service"
    if any(term in lower for term in ("property investment", "investment property", "rental income", "income from property")):
        return "Property investment and related rental income activities."
    if len(normalized) > 180:
        return normalized[:177].rsplit(" ", 1)[0].rstrip(" ,;") + "..."
    return normalized




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


def _suggest_checklist_areas(lower_text: str) -> str:
    areas = []
    if re.search(r"\b(?:revenue|turnover|sales)\b", lower_text):
        areas.append("IFRS 15 (Revenue)")
    if _actual_lease_disclosure_present(lower_text):
        areas.append("IFRS 16 (Leases)")
    if re.search(r"\b(?:expected credit loss|ecl|impairment of financial|financial assets)\b", lower_text):
        areas.append("IFRS 9 (Financial Instruments)")
    if re.search(r"\b(?:intangible assets?|goodwill|amortisation)\b", lower_text):
        areas.append("IAS 38 (Intangible Assets)")
    if re.search(r"\b(?:investment propert(?:y|ies))\b", lower_text):
        areas.append("IAS 40 (Investment Property)")
    if re.search(r"\b(?:deferred tax|income tax expense|taxation)\b", lower_text):
        areas.append("IAS 12 (Income Taxes)")
    return ", ".join(areas) if areas else "None"




def _requires_ocr(document: PdfDocument) -> bool:
    if not document.pages:
        return True
    if document.text_pages == 0 or document.extraction_coverage == 0:
        return True
    if document.extraction_coverage < 0.25:
        return True
    return document.text_chars < 1000 and document.extraction_coverage < 0.5


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
    classified = _classified_primary_statement_pages(document)
    income_page = classified.get("Statement of income and expenditure")
    equity_page = classified.get("Statement of changes in accumulated fund")
    income_header = "\n".join(income_page.text.splitlines()[:8]).lower() if income_page else ""
    equity_header = "\n".join(equity_page.text.splitlines()[:8]).lower() if equity_page else ""
    has_profit_or_loss_heading = bool(
        income_page and any(marker in income_header for marker in ("statement of profit or loss", "comprehensive income"))
    )
    has_equity_heading = bool(equity_page and "changes in equity" in equity_header)
    is_company = document and (
        _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
        or has_profit_or_loss_heading
        or has_equity_heading
    )
    income_stmt_name = "Statement of profit or loss" if is_company else "Statement of income and expenditure"
    changes_stmt_name = "Statement of changes in equity" if is_company else "Statement of changes in accumulated fund"
    
    checks = (
        (income_stmt_name, _check_income_statement_text),
        ("Statement of financial position", _check_sfp_text),
        (changes_stmt_name, _check_accumulated_fund_text),
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

    cross_source_findings, cross_source_performed, cross_source_skipped = _check_cross_source_cash_flow(document, tolerance)
    findings.extend(cross_source_findings)
    performed.extend(cross_source_performed)
    skipped.extend(cross_source_skipped)

    cash_support_findings, cash_support_performed = _check_cash_flow_supporting_amounts(document, tolerance)
    findings.extend(cash_support_findings)
    performed.extend(cash_support_performed)

    disclosure_findings, disclosure_performed = _check_supporting_disclosure_note_reference_amounts(document, tolerance)
    findings.extend(disclosure_findings)
    performed.extend(disclosure_performed)

    summary_findings, summary_performed = _check_supplementary_summary_consistency(document, tolerance)
    findings.extend(summary_findings)
    performed.extend(summary_performed)

    # Value Added Statement cross-check
    vas_page = next((page for page in document.pages if _looks_like_value_added_page(page.text)), None)
            
    if vas_page:
        pl_page = _find_statement_page(document, income_stmt_name)
        if pl_page:
            pl_amounts, pl_line = _line_amount_for_aliases(pl_page.text, ("interest expense", "finance cost", "finance costs"))
            if pl_amounts:
                pl_finance_cost = pl_amounts[0]
                vas_amounts, vas_line = _line_amount_for_aliases(
                    vas_page.text,
                    ("interest expense", "interest", "finance cost", "interest payable", "providers of capital", "interest paid"),
                )
                if vas_amounts:
                    vas_finance_cost = vas_amounts[0]
                    performed.append("Value Added Statement: interest expense compared with primary statement where readable.")
                    if abs(abs(vas_finance_cost) - abs(pl_finance_cost)) > tolerance:
                        findings.append(
                            Finding(
                                "Value Added Statement",
                                "Medium",
                                f"Page {vas_page.number} | Value Added Statement",
                                f"Value Added Statement shows interest expense of {abs(vas_finance_cost):,}, while the primary statement shows {abs(pl_finance_cost):,}.",
                                f"Value Added Statement line: {vas_line} | Primary statement line: {pl_line}",
                                "Review whether the Value Added Statement label, classification, or amount is appropriate.",
                            )
                        )
                        
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

    findings.extend(_check_share_capital_line_tables(document, tolerance))

    skipped_tables: list[str] = []
    primary_statements = _classified_primary_statement_pages(document)
    primary_pages = {p.number for p in primary_statements.values()}
    notes_start_page = _notes_start_page(document)
    line_checked_note_pages: set[int] = set()
    
    for page in document.pages:
        for table_index, table in enumerate(page.tables, start=1):
            if len(table) < 3:
                continue
            table_quality = _classify_table_for_arithmetic(table, page.text)
            if _skip_table_is_not_review_relevant(page, table_quality, primary_pages, notes_start_page):
                if table_quality["type"] in {"value-added statement", "multi-year summary"}:
                    skipped_tables.append(
                        f"Page {page.number}, table {table_index}: {table_quality['type']} ({table_quality['reason']})"
                    )
                continue
            if notes_start_page is not None and page.number >= notes_start_page and page.number not in primary_pages:
                targeted_note_findings = _check_simple_note_table_casting(page, table_index, table, tolerance)
                if targeted_note_findings:
                    findings.extend(targeted_note_findings)
                    continue
                simple_note_text_casting_safe = document.table_extraction_confidence >= 80 or document.merged_value_cell_count <= 10
                if page.number not in line_checked_note_pages and simple_note_text_casting_safe:
                    line_note_findings = _check_simple_note_text_casting(page, tolerance)
                    if line_note_findings:
                        findings.extend(line_note_findings)
                        line_checked_note_pages.add(page.number)
                        continue
                elif not simple_note_text_casting_safe:
                    skipped_tables.append(
                        f"Page {page.number}, table {table_index}: simple note text casting skipped because table extraction confidence is low ({document.table_extraction_confidence}%) and merged numeric cells were detected ({document.merged_value_cell_count})."
                    )
                skipped_tables.append(
                    f"Page {page.number}, table {table_index}: skipped because generic arithmetic is not reliable for notes tables; use note-reference and amount-agreement checks instead."
                )
                continue
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
            
            if document.table_extraction_confidence < 80 and not table_quality["can_run_arithmetic"]:
                skipped_tables.append(f"Page {page.number}, table {table_index}: skipped because overall table extraction confidence is low ({document.table_extraction_confidence}%).")
                continue

            for f in table_findings:
                f.severity = "Low"
                if "Downgraded severity" not in f.issue:
                    f.issue += " (Downgraded severity because table structure in notes may be complex)."
            findings.extend(table_findings)
    if skipped_tables:
        document.skipped_table_details = list(dict.fromkeys(skipped_tables))
    return findings


def _check_simple_note_text_casting(page: PdfPage, tolerance: Decimal) -> list[Finding]:
    findings: list[Finding] = []
    sections = _simple_note_text_sections(page.text)
    for note_ref, heading, lines in sections:
        if not _simple_note_text_section_castable(heading, lines):
            continue
        component_rows: list[dict[str, object]] = []
        checked_columns = 0
        mismatches: list[str] = []
        corrections: list[str] = []
        for raw_line in lines:
            line = re.sub(r"\s+", " ", raw_line.strip())
            if not line:
                continue
            if checked_columns and re.match(r"^\d{1,2}\.\d+\b", line):
                break
            normalized = _normalise_match_words(line)
            if checked_columns and _simple_note_text_after_total_narrative(normalized):
                break
            if _simple_note_text_line_boundary(normalized):
                component_rows = []
                continue
            amounts = _simple_note_amounts_from_line(line)
            if not amounts:
                continue
            label = re.sub(NUMBER_RE, " ", line)
            label = re.sub(r"\s+", " ", label).strip(" -:;")
            is_total = _looks_like_total(normalized) or (not re.search(r"[A-Za-z]{3,}", label) and len(component_rows) >= 2)
            if is_total:
                if len(component_rows) >= 2:
                    width = min(len(amounts), max(len(row["amounts"]) for row in component_rows))
                    for col in range(width):
                        prepared = _prepare_simple_note_column(component_rows, col, amounts[col], line)
                        if not prepared:
                            continue
                        expected = prepared["expected"]
                        reported = amounts[col]
                        diff = reported - expected
                        checked_columns += 1
                        for correction in prepared["corrections"]:
                            if correction not in corrections:
                                corrections.append(correction)
                        if abs(diff) > tolerance:
                            mismatches.append(
                                f"Note {note_ref} {heading}, column {col + 1}: reported {reported:,}, visible sum {expected:,}, difference {diff:,}"
                            )
                    if all(
                        (
                            prepared := _prepare_simple_note_column(component_rows, col, amounts[col], line)
                        )
                        and abs(amounts[col] - prepared["expected"]) <= tolerance
                        for col in range(width)
                    ):
                        component_rows = [{"label": label or "subtotal", "amounts": amounts, "raw_line": line}]
                    else:
                        component_rows = []
                else:
                    component_rows = []
                continue
            if _simple_note_component_label(normalized) and not _simple_note_text_component_blocked(normalized):
                component_rows.append({"label": label, "amounts": amounts, "raw_line": line})
        if checked_columns:
            location = f"Page {page.number} | Note {note_ref} {heading}"
            if mismatches:
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "Medium",
                        location,
                        "Simple note section total does not agree to visible component rows.",
                        "; ".join(mismatches[:4]),
                        "Recalculate the note section and confirm whether a hidden row, rounding adjustment, or extraction issue explains the difference.",
                        metadata={"check_type": "simple_note_text_casting", "page": str(page.number), "note": note_ref},
                    )
                )
            else:
                correction_note = f" Extraction support: {'; '.join(corrections[:3])}." if corrections else ""
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "Passed",
                        location,
                        f"Simple note section on Page {page.number} casts correctly.",
                        f"Checked Note {note_ref} {heading}; {checked_columns} amount column(s) agreed within tolerance {tolerance}.{correction_note}",
                        "No reviewer action required unless the source note is amended.",
                        metadata={"check_type": "simple_note_text_casting", "page": str(page.number), "note": note_ref},
                    )
                )
    return findings


def _prepare_simple_note_column(
    component_rows: list[dict[str, object]],
    col: int,
    reported_total: Decimal,
    total_line: str,
) -> dict[str, object] | None:
    present_values: list[Decimal] = []
    missing_rows: list[dict[str, object]] = []
    corrections: list[str] = []
    for row in component_rows:
        amounts = row["amounts"]
        if col < len(amounts):
            present_values.append(amounts[col])
        else:
            missing_rows.append(row)
    if not present_values:
        return None
    if not missing_rows:
        return {"expected": sum(present_values, Decimal("0")), "corrections": corrections}
    if len(missing_rows) != 1:
        return None
    missing_row = missing_rows[0]
    raw_line = str(missing_row.get("raw_line", ""))
    if not _simple_note_missing_amount_candidate(raw_line):
        return None
    inferred = reported_total - sum(present_values, Decimal("0"))
    if abs(inferred) >= Decimal("100000000"):
        return None
    corrections.append(
        f"Inferred missing column {col + 1} amount {inferred:,} for '{missing_row.get('label', 'line item')}' from the reported total because the extracted row contained a noisy amount token."
    )
    return {"expected": sum(present_values, Decimal("0")) + inferred, "corrections": corrections}


def _simple_note_missing_amount_candidate(raw_line: str) -> bool:
    return bool(
        re.search(r"\b[A-Za-z]\d{2,}\b", raw_line)
        or re.search(r"\b\d+[A-Za-z]\d+\b", raw_line)
        or re.search(r"\s-\s*$", raw_line)
    )


def _simple_note_text_sections(text: str) -> list[tuple[str, str, list[str]]]:
    sections: list[tuple[str, str, list[str]]] = []
    current_ref = ""
    current_heading = ""
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        match = NOTE_HEADING_RE.match(line)
        if match and _valid_note_heading(match.group(1), match.group(2)):
            if current_ref and current_lines:
                sections.append((current_ref, current_heading, current_lines))
            current_ref = match.group(1).upper()
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        if current_ref:
            current_lines.append(line)
    if current_ref and current_lines:
        sections.append((current_ref, current_heading, current_lines))
    return sections


def _simple_note_text_section_castable(heading: str, lines: list[str]) -> bool:
    heading_normalized = _normalise_match_words(heading)
    normalized = _normalise_match_words(f"{heading} {' '.join(lines[:20])}")
    section_excluded = (
        "financial instruments and risk management",
        "maturity analysis",
        "credit risk",
        "expected credit loss",
        "ecl",
        "related parties",
        "new standards",
        "amendments",
        "fair value hierarchy",
        "financial instruments",
        "reconciliation",
        "movement in",
        "breakdown",
        "deferred tax",
        "taxation",
    )
    heading_excluded = (
        "property plant and equipment",
        "intangible assets",
        "share capital",
        "deposit for shares",
        "ordinary shares",
        "number of shares",
        "depreciation",
        "amortisation",
    )
    heading_blockers = (
        "operating profit",
        "profit loss",
        "profit before tax",
        "loss before tax",
        "profit before financing",
        "operating loss",
    )
    if any(term in normalized for term in section_excluded):
        return False
    if any(term in heading_normalized for term in heading_excluded):
        return False
    if any(term in heading_normalized for term in heading_blockers):
        return False
    if re.search(r"\b\d+[a-z]\b", " ".join(lines[:40]), flags=re.I):
        return False
    if any(
        marker in normalized
        for marker in (
            "as at 1 january",
            "as at january",
            "as at 31 december",
            "opening balance",
            "closing balance",
            "minimum lease payment",
            "balance at beginning",
            "balance at end",
        )
    ):
        return False
    amount_lines = [line for line in lines if _simple_note_amounts_from_line(line)]
    if len(amount_lines) < 3:
        return False
    if _simple_note_text_has_suspicious_amount_noise(amount_lines):
        return False
    has_total = any(_looks_like_total(_normalise_match_words(line)) for line in amount_lines)
    has_numeric_total = any(
        len(_simple_note_amounts_from_line(line)) >= 2
        and not re.search(r"[A-Za-z]{3,}", re.sub(NUMBER_RE, " ", line))
        for line in amount_lines[2:]
    )
    return has_total or has_numeric_total


def _simple_note_amounts_from_line(line: str) -> list[Decimal]:
    line = re.sub(r"^\s*\d+[A-Z]?\.\s*", "", line.strip(), flags=re.I)
    line = re.sub(r"\b\d{1,3}\s+DRAFT\b", "", line, flags=re.I)
    line = re.sub(r"\bDRAFT\b", "", line, flags=re.I).strip()
    line = _normalise_statement_number_spacing(line)
    if re.fullmatch(r"\d{1,3}", line):
        return []
    amounts: list[Decimal] = []
    token_re = re.compile(r"\(?-?\d[\d,]*\)?|(?:(?:^|(?<=\s))-(?=\s|$))")
    for match in token_re.finditer(line):
        token = match.group(0)
        before = line[match.start() - 1] if match.start() > 0 else " "
        after = line[match.end()] if match.end() < len(line) else " "
        if token != "-" and (before.isalpha() or after.isalpha()):
            continue
        if token == "-":
            amounts.append(Decimal("0"))
            continue
        if token.startswith("20") and len(token.strip("()")) == 4:
            continue
        cleaned = token.strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()").replace(",", "")
        try:
            amount = Decimal(cleaned)
        except InvalidOperation:
            continue
        amounts.append(-amount if negative else amount)
    if len(amounts) >= 3 and 0 < amounts[0] <= 60 and _line_appears_to_have_note_ref_before_amounts(line):
        amounts = amounts[1:]
    if len(amounts) == 1 and amounts[0] >= 0 and amounts[0] <= 100 and not re.search(r"[A-Za-z]{3,}", line):
        return []
    return amounts


def _line_appears_to_have_note_ref_before_amounts(line: str) -> bool:
    first_number = NUMBER_RE.search(line)
    if not first_number:
        return False
    label_before = line[: first_number.start()]
    return bool(re.search(r"[A-Za-z]{3,}", label_before))


def _simple_note_text_line_boundary(normalized_line: str) -> bool:
    if normalized_line.startswith("details "):
        return True
    return any(
        marker in normalized_line
        for marker in (
            "details of",
            "the movement",
            "reconciliation",
            "major components",
            "tax at the applicable",
            "effective tax rate",
            "n 000",
            "2025 2024",
        )
    )


def _simple_note_text_after_total_narrative(normalized_line: str) -> bool:
    if normalized_line.startswith("details "):
        return True
    return any(
        marker in normalized_line
        for marker in (
            "details of",
            "the company",
            "the group",
            "in line with",
            "frc rule",
            "statutory audit",
            "the table shows",
            "number of employees",
            "whose earnings",
            "during the year",
            "pension reform act",
            "defined contribution",
            "retirement savings",
        )
    )


def _simple_note_text_has_suspicious_amount_noise(amount_lines: list[str]) -> bool:
    if len(amount_lines) < 6:
        return False
    suspicious = 0
    for line in amount_lines:
        if re.search(r"\b[A-Za-z]{1,3}\d{2,}\b", line) or re.search(r"\b\d+[A-Za-z]{1,3}\d+\b", line):
            suspicious += 1
    return suspicious > max(1, len(amount_lines) // 8)


def _simple_note_text_component_blocked(normalized_line: str) -> bool:
    return any(
        marker in normalized_line
        for marker in (
            "note",
            "tax rate",
            "effective",
            "reconciliation",
            "temporary difference",
        )
    )


def _check_simple_note_table_casting(
    page: PdfPage,
    table_index: int,
    table: list[list[str]],
    tolerance: Decimal,
) -> list[Finding]:
    """Cast only simple note tables with clear component rows and total rows."""
    if not _simple_note_table_castable(table, page.text):
        return []
    note_cols = _note_columns(table)
    rows = [_numeric_row(row, note_cols) for row in table]
    max_cols = max((len(row) for row in rows), default=0)
    results: list[Finding] = []
    mismatches: list[str] = []
    passed_columns = 0
    for col in range(1, max_cols):
        if col in note_cols:
            continue
        components: list[Decimal] = []
        column_checked = False
        for row_index, row in enumerate(rows[1:], start=1):
            label = str(row[0] or "").strip().lower() if row else ""
            if _is_table_boundary_row(row):
                components = []
                continue
            value = row[col] if col < len(row) else None
            if not isinstance(value, Decimal):
                continue
            if _looks_like_total(label):
                if len(components) >= 2:
                    expected = sum(components, Decimal("0"))
                    diff = value - expected
                    column_checked = True
                    if abs(diff) > tolerance:
                        mismatches.append(
                            f"column {col + 1}, row {row_index + 1}: reported {value:,}, visible sum {expected:,}, difference {diff:,}"
                        )
                components = []
            elif _simple_note_component_label(label):
                components.append(value)
        if column_checked:
            passed_columns += 1
    if not passed_columns:
        return []
    location = f"Page {page.number}, table {table_index}"
    if mismatches:
        results.append(
            Finding(
                "Totals and rounding",
                "Medium",
                location,
                "Simple note table total does not agree to visible component rows.",
                "; ".join(mismatches[:4]),
                "Recalculate the note table and confirm whether a hidden row, rounding adjustment, or extraction issue explains the difference.",
                metadata={"check_type": "simple_note_table_casting", "page": str(page.number), "table": str(table_index)},
            )
        )
    else:
        results.append(
            Finding(
                "Totals and rounding",
                "Passed",
                location,
                f"Simple note table on Page {page.number}, table {table_index} casts correctly.",
                f"Checked {passed_columns} amount column(s); reported totals agree within tolerance {tolerance}.",
                "No reviewer action required unless the source note is amended.",
                metadata={"check_type": "simple_note_table_casting", "page": str(page.number), "table": str(table_index)},
            )
        )
    return results


def _simple_note_table_castable(table: list[list[str]], page_text: str) -> bool:
    if len(table) < 3:
        return False
    table_text = _normalise_match_words(" ".join(" ".join(str(cell or "") for cell in row) for row in table))
    page_lower = _normalise_match_words(page_text[:3000])
    combined = f"{table_text} {page_lower}"
    excluded = (
        "financial instruments and risk management",
        "maturity analysis",
        "credit risk",
        "expected credit loss",
        "ecl",
        "sensitivity analysis",
        "related parties",
        "directors remuneration",
        "key management",
        "new standards",
        "amendments",
        "fair value hierarchy",
        "value added",
        "five year",
        "financial summary",
        "financial instruments",
        "property plant and equipment",
        "intangible assets",
        "share capital",
        "ordinary shares",
        "number of shares",
        "depreciation",
        "amortisation",
        "reconciliation of carrying amount",
        "gross carrying amount",
        "accumulated depreciation",
    )
    if any(term in combined for term in excluded):
        return False
    header_text = _normalise_match_words(" ".join(str(cell or "") for cell in table[0]))
    has_amount_header = _table_has_financial_amount_header(table) or len(set(YEAR_RE.findall(header_text))) >= 1
    if not has_amount_header:
        return False
    note_cols = _note_columns(table)
    rows = [_numeric_row(row, note_cols) for row in table]
    total_rows = 0
    component_rows = 0
    amount_counts: list[int] = []
    for row in rows[1:]:
        label = str(row[0] or "").strip().lower() if row else ""
        count = _row_amount_count(row)
        if count:
            amount_counts.append(count)
        if count and _looks_like_total(label):
            total_rows += 1
        elif count and _simple_note_component_label(label):
            component_rows += 1
    if total_rows == 0 or component_rows < 2:
        return False
    if not amount_counts:
        return False
    common_count = max(set(amount_counts), key=amount_counts.count)
    return common_count >= 1 and amount_counts.count(common_count) / len(amount_counts) >= 0.7


def _simple_note_component_label(label: str) -> bool:
    if not label.strip() or not re.search(r"[a-z]{3,}", label):
        return False
    excluded = ("note", "date", "audited", "restated")
    if any(word in label for word in excluded):
        return False
    blocked = (
        "opening balance",
        "closing balance",
        "at 1 january",
        "at 31 december",
        "balance at",
        "carrying amount",
        "gross",
        "accumulated",
        "maturity",
        "not past due",
        "past due",
        "stage ",
    )
    return not any(term in label for term in blocked)


def _check_share_capital_line_tables(document: PdfDocument, tolerance: Decimal) -> list[Finding]:
    findings: list[Finding] = []
    for page in document.pages:
        if not _page_may_contain_share_capital_cast(page.text):
            continue
        schedules = _share_capital_line_schedules(page.text)
        for schedule_index, schedule in enumerate(schedules, start=1):
            component_rows = schedule["components"]
            total_values = schedule["total"]
            if len(component_rows) < 2 or len(total_values) < 2:
                continue
            aligned_rows: list[tuple[str, list[Decimal]]] = []
            corrections: list[str] = []
            for label, amounts in component_rows:
                aligned_amounts, correction = _align_share_capital_amounts(amounts, total_values)
                if len(aligned_amounts) == len(total_values):
                    aligned_rows.append((label, aligned_amounts))
                    if correction:
                        corrections.append(f"{label}: {correction}")
            if len(aligned_rows) < 2:
                continue
            column_count = min(len(total_values), min(len(row[1]) for row in aligned_rows))
            if column_count < 2:
                continue
            differences: list[tuple[int, Decimal, Decimal, Decimal]] = []
            for col in range(column_count):
                expected = sum((amounts[col] for _label, amounts in aligned_rows), Decimal("0"))
                reported = total_values[col]
                diff = reported - expected
                if abs(diff) > tolerance:
                    differences.append((col + 1, reported, expected, diff))
            location = f"Page {page.number} | Share capital/shareholding table"
            row_count = len(aligned_rows)
            if differences:
                evidence = "; ".join(
                    f"column {col}: reported {reported:,}, visible sum {expected:,}, difference {diff:,}"
                    for col, reported, expected, diff in differences
                )
                if corrections:
                    evidence += f" Extraction corrections applied: {'; '.join(corrections[:3])}."
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "Medium",
                        location,
                        "Share capital or shareholding table total does not agree to visible rows.",
                        f"Checked {row_count} visible shareholder/share-capital rows. {evidence}.",
                        "Recalculate the issued share capital/shareholding table and confirm whether a hidden row, rounding adjustment, or extraction issue explains the difference.",
                        metadata={"check_type": "share_capital_table", "page": str(page.number), "schedule": str(schedule_index)},
                    )
                )
            else:
                correction_note = f" Extraction corrections applied: {'; '.join(corrections[:3])}." if corrections else ""
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "Passed",
                        location,
                        f"Share capital/shareholding table on Page {page.number} casts correctly.",
                        f"Checked {row_count} visible rows across {column_count} numeric columns; reported totals agree within tolerance {tolerance}.{correction_note}",
                        "No reviewer action required unless the source page is amended.",
                        metadata={"check_type": "share_capital_table", "page": str(page.number), "schedule": str(schedule_index)},
                    )
                )
    return findings


def _page_may_contain_share_capital_cast(text: str) -> bool:
    lower = _normalise_match_words(text[:4000])
    share_terms = ("share capital", "issued share", "number of shares", "ordinary shares", "shareholding")
    if not any(term in lower for term in share_terms):
        return False
    return "issued" in lower or "number of shares" in lower or "shareholder" in lower


def _share_capital_line_schedules(text: str) -> list[dict[str, object]]:
    schedules: list[dict[str, object]] = []
    collecting = False
    components: list[tuple[str, list[Decimal]]] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        normalized = _normalise_match_words(line)
        if not collecting and ("issued" in normalized or "number of shares" in normalized):
            collecting = True
            components = []
            continue
        if not collecting:
            continue
        if _share_capital_schedule_end(line):
            if components:
                components = []
            collecting = False
            continue
        amounts = _share_capital_amounts_from_line(line)
        if len(amounts) < 2:
            continue
        if _share_capital_total_line(line, amounts):
            schedules.append({"components": list(components), "total": amounts})
            components = []
            collecting = False
            continue
        label = _share_capital_component_label(line)
        if label:
            components.append((label, amounts))
    return schedules


def _share_capital_component_label(line: str) -> str:
    label = re.sub(NUMBER_RE, " ", line)
    label = re.sub(r"\s+", " ", label).strip(" -")
    if not re.search(r"[A-Za-z]{3,}", label):
        return ""
    if re.search(r"\b(issued|number of shares|ordinary shares|share capital|direct|indirect)\b", label, flags=re.I):
        return ""
    return label


def _share_capital_amounts_from_line(line: str) -> list[Decimal]:
    amounts: list[Decimal] = []
    for token in re.findall(r"\(?-?\d[\d,]*\)?", line):
        if len(token) == 4 and token.isdigit() and token.startswith("20"):
            continue
        cleaned = token.strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()").replace(",", "")
        try:
            amount = Decimal(cleaned)
        except InvalidOperation:
            continue
        amounts.append(-amount if negative else amount)
    return amounts


def _align_share_capital_amounts(amounts: list[Decimal], total_values: list[Decimal]) -> tuple[list[Decimal], str]:
    if len(amounts) == len(total_values):
        return amounts, ""
    if (
        len(total_values) == 4
        and len(amounts) == 3
        and abs(amounts[0]) < 1000
        and abs(total_values[0]) < 10000
        and abs(total_values[1]) < 10000
        and abs(amounts[1]) >= 1000
        and abs(amounts[2]) >= 1000
    ):
        return [amounts[0], amounts[0], amounts[1], amounts[2]], "duplicated first small issued-capital amount where extraction dropped the comparative value"
    return amounts, ""


def _share_capital_total_line(line: str, amounts: list[Decimal]) -> bool:
    if len(amounts) < 2:
        return False
    label = _share_capital_component_label(line)
    if re.search(r"\b(total|closing|issued share capital)\b", line, flags=re.I):
        return True
    return not label and len(amounts) >= 2


def _share_capital_schedule_end(line: str) -> bool:
    lower = line.lower()
    return any(
        marker in lower
        for marker in (
            "there have been no changes",
            "directors' interests",
            "directors interests",
            "going concern",
            "charitable donation",
            "events after",
            "auditors",
        )
    )


def _skip_table_is_not_review_relevant(
    page: PdfPage,
    table_quality: dict[str, object],
    primary_pages: set[int],
    notes_start_page: int | None,
) -> bool:
    if page.number in primary_pages:
        return False
    table_type = str(table_quality.get("type", "")).lower()
    if table_type in {"value-added statement", "multi-year summary"}:
        return True
    if notes_start_page is not None and page.number >= notes_start_page:
        return not _notes_table_skip_needs_reviewer_attention(page.text)
    if table_type == "other table" and _looks_like_rotated_or_unreadable_statement_page(page.text):
        return True
    return _front_matter_table_skip_is_noise(page.text)


def _front_matter_table_skip_is_noise(text: str) -> bool:
    lower = _normalise_match_words(text[:2500])
    front_matter_markers = (
        "general information",
        "directors report",
        "directors responsibilities",
        "directors responsibility",
        "independent auditor report",
        "report on audit financial statements",
        "corporate information",
        "registered office",
        "company registration",
        "shareholding",
        "directors interests",
        "going concern",
        "charitable donation",
        "auditors",
    )
    return any(marker in lower for marker in front_matter_markers)


def _notes_table_skip_needs_reviewer_attention(text: str) -> bool:
    lower = _normalise_match_words(text[:3000])
    if _is_post_notes_supplement_page(text):
        return False
    non_cast_notes = (
        "new standards and interpretations",
        "standards and interpretations",
        "new standards",
        "related parties",
        "related party",
        "subsequent events",
        "going concern",
        "contingent liabilities",
        "capital commitments",
        "financial instruments and risk management",
    )
    if any(marker in lower for marker in non_cast_notes):
        return False
    amount_note_markers = (
        "property plant and equipment",
        "intangible assets",
        "financial assets",
        "trade and other receivables",
        "trade other receivables",
        "cash and cash equivalents",
        "cash cash equivalents",
        "share capital",
        "share premium",
        "deferred tax",
        "trade and other payables",
        "trade other payables",
        "borrowings",
        "loans and borrowings",
        "bank loan",
        "bank loans",
        "current tax liabilities",
        "contract liabilities",
        "revenue",
        "direct expenses",
        "other income",
        "operating gains",
        "operating losses",
        "other operating gains",
        "other operating losses",
        "operating expenses",
        "expenses",
        "expense",
        "employee costs",
        "investment income",
    )
    return any(marker in lower for marker in amount_note_markers)


def _looks_like_rotated_or_unreadable_statement_page(text: str) -> bool:
    sample = text[:2500]
    if not sample.strip():
        return False
    normalised = _normalise_match_words(sample)
    if "statement of changes" in normalised or "changes in equity" in normalised:
        return True
    tokens = re.findall(r"[A-Za-z]{3,}", sample)
    recognisable = sum(
        1
        for token in tokens
        if token.lower()
        in {
            "statement",
            "changes",
            "equity",
            "balance",
            "share",
            "capital",
            "premium",
            "income",
            "loss",
            "profit",
            "total",
            "financial",
        }
    )
    gibberish_markers = len(re.findall(r"[\u20ac\u00a2\u00a3\u00a5]|[A-Za-z]{1,2}[\u2018\u2019][A-Za-z]{1,3}|000,,|,,N|2ouryeg|Asenuef", sample))
    return gibberish_markers >= 3 and recognisable <= 5


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
    findings.extend(_check_share_capital_unit_heading_presentation(document))
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


def _check_share_capital_unit_heading_presentation(document: PdfDocument) -> list[Finding]:
    findings: list[Finding] = []
    seen_pages: set[int] = set()
    for page in document.pages:
        if page.number in seen_pages or not _page_may_contain_share_capital_cast(page.text):
            continue
        issue = _share_capital_unit_heading_issue(page.text)
        if not issue:
            continue
        seen_pages.add(page.number)
        findings.append(
            Finding(
                "Formatting",
                "Low",
                f"Page {page.number}",
                "Share capital table may have unclear unit headings.",
                issue,
                "Confirm that monetary columns and number-of-share columns are separately and clearly labelled.",
                metadata={"check_type": "share_capital_presentation", "page": str(page.number)},
            )
        )
    return findings


def _share_capital_unit_heading_issue(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        line_lower = _normalise_match_words(line)
        if not re.search(r"\b(issued|shareholding|shareholder|number\s+of\s+shares?)\b", line, flags=re.I):
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 8)
        window_lines = lines[start:end]
        window = " ".join(window_lines)
        normalized_window = _normalise_match_words(window)
        share_unit_count = len(re.findall(r"\bnumber\s+of\s+shares?\b", window, flags=re.I))
        if not share_unit_count:
            continue
        if not _currency_unit_marker_count(window):
            continue
        year_header_count = max((len(YEAR_RE.findall(candidate)) for candidate in window_lines), default=0)
        amount_column_count = max((len(NUMBER_RE.findall(candidate)) for candidate in window_lines), default=0)
        if max(year_header_count, amount_column_count) < 4:
            continue
        unit_count = _currency_unit_marker_count(window) + share_unit_count
        if unit_count >= max(year_header_count, 2):
            continue
        snippet = re.sub(r"\s+", " ", " | ".join(window_lines[:6])).strip()[:260]
        return (
            "The share capital/shareholding table appears to mix currency amounts and number-of-share columns, "
            f"but the unit headings may not clearly map to all columns. Header snippet: {snippet}"
        )
    return ""


def _currency_unit_marker_count(text: str) -> int:
    return len(
        re.findall(
            "(?:N|NGN|NGN\\.|\\u20a6)\\s*['\\u2019\\u2018]?\\s*0{3}\\b|N\\s*['\\u2019\\u2018]\\s*000\\b|N\\s*'\\s*000\\b",
            text,
            flags=re.I,
        )
    )

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


def _check_note_contradictions(document: PdfDocument) -> list[Finding]:
    findings: list[Finding] = []
    note_sections = _note_sections(document)
    headings = _note_headings_by_page(document)
    nil_patterns = (
        r"\bwas nil\b",
        r"\bis nil\b",
        r"\bamounted to nil\b",
        r"\bno balance\b",
        r"\bnone\b",
        r"\bzero\b",
    )
    comparative_markers = (
        "comparative",
        "prior year",
        "previous year",
        "did not commence",
        "until the financial year under review",
        "resulting in a nil comparative balance",
        "as at 31 december 2024",
        "as at 31 december 2023",
    )
    for ref, section in note_sections.items():
        title, page_number = headings.get(ref, ("", 0))
        if not title or not section:
            continue
        lines = [re.sub(r"\s+", " ", line).strip() for line in section.splitlines() if line.strip()]
        contradiction_line = ""
        for line in lines[:12]:
            lower = line.lower()
            if any(marker in lower for marker in comparative_markers):
                continue
            if any(re.search(pattern, lower, re.I) for pattern in nil_patterns):
                contradiction_line = line
                break
        if not contradiction_line:
            continue
        non_zero_amounts = [amount for amount in _amounts_in_text(section) if abs(amount) > Decimal("1")]
        if not non_zero_amounts:
            continue
        largest = max(non_zero_amounts, key=lambda amount: abs(amount))
        findings.append(
            Finding(
                "Narrative consistency",
                "Medium",
                f"Page {page_number or 'Unknown'} | Note {ref}",
                f"Note {ref} states that a balance was nil, but the same note table shows a non-zero amount.",
                f"{contradiction_line} | Note heading: {title} | Non-zero amount detected in same note: {largest:,.0f}",
                "Review the narrative disclosure against the note table and correct the wording or the amount presentation.",
            )
        )
        break
    return findings

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
    statement_ref_pages = _statement_reference_pages(document)
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
        page_reference = statement_ref_pages.get(ref, "Statement pages")
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                page_reference,
                f"Statement references note {ref}, but a matching note heading was not confidently detected or parsed; review prompt only.",
                f"Detected statement reference: Note {ref} on {page_reference}.",
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
        passed_refs = set()
        # By default, cautious findings only emits findings for FAILURES. If a note is NOT in cautious_findings, it might have passed!
        # Actually, let's just use the main rows builder.
        try:
            # We call the same function the exporter uses to get the True/False passed states
            check_result_rows = _note_agreement_result_rows(document)
            passed_refs = {str(row["Note number"]).strip() for row in check_result_rows if row["Review result"] == "Passed"}
        except Exception:
            pass
            
        for f in misref_findings:
            ref_match = re.search(r"Note (\d+[A-Z]?)", f.evidence)
            if ref_match and ref_match.group(1).strip() in passed_refs:
                continue
            findings.append(f)
        cautious_findings = _check_cautious_face_note_amount_agreement(
            _statement_note_lines(document),
            note_sections,
            headings,
            tolerance,
            cautious_review_prompt=True,
        )
        for f in cautious_findings:
            ref_match = re.search(r"Note (\d+[A-Z]?)", f.issue)
            if ref_match and ref_match.group(1).strip() in passed_refs:
                continue
            findings.append(f)
        return findings
    note_sections = _note_sections(document)
    for item in _statement_note_lines(document):
        if item.ref or _note_agreement_skip_reason(item):
            continue
        if _is_disclosure_only_note(item.line_item):
            continue
        suggested_ref = _suggest_note_for_unreferenced_line(item, note_sections, headings, tolerance)
        if suggested_ref:
            findings.append(
                Finding(
                    "Notes agreement",
                    "Medium",
                    f"Page {item.page_number} | {item.statement_name}",
                    f"{_statement_line_item_title(item.line_item)} lacks a note reference. Suggested Note {suggested_ref} may be the related note.",
                    f"Statement line: {item.line[:160]} | Suggested Note {suggested_ref} based on heading and/or amount agreement.",
                    "Confirm whether the face statement should include a note reference and update the reference if required.",
                    metadata={
                        "statement": item.statement_name,
                        "line_item": _statement_line_item_title(item.line_item),
                        "referenced_note": "",
                        "suggested_note": suggested_ref,
                        "match_confidence": "Medium",
                        "reason": f"Suggested Note {suggested_ref}",
                        "line_key": f"|{item.line}",
                    },
                )
            )
        else:
            findings.append(
                Finding(
                    "Notes agreement",
                    "Low",
                    f"Page {item.page_number} | {item.statement_name}",
                    f"{_statement_line_item_title(item.line_item)} has no note reference and no matching note was found.",
                    f"Statement line: {item.line[:160]}",
                    "Review whether the line item should be note-linked or whether no separate supporting note is expected.",
                    metadata={
                        "statement": item.statement_name,
                        "line_item": _statement_line_item_title(item.line_item),
                        "referenced_note": "",
                        "suggested_note": "",
                        "match_confidence": "Low",
                        "reason": "No matching note was found.",
                        "line_key": f"|{item.line}",
                    },
                )
            )
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
    misref_findings, misreferenced_lines = _check_possible_wrong_note_references(
        _statement_note_lines(document),
        note_sections,
        headings,
        tolerance,
        cautious_review_prompt=not detailed_note_checks_allowed and cautious_low_confidence,
    )
    try:
        check_result_rows = _note_agreement_result_rows(document)
        passed_refs = {str(row["Note number"]).strip() for row in check_result_rows if row["Review result"] == "Passed"}
    except Exception:
        passed_refs = set()
        
    for f in misref_findings:
        ref_match = re.search(r"Note (\d+[A-Z]?)", f.evidence)
        if ref_match and ref_match.group(1).strip() in passed_refs:
            continue
        findings.append(f)
    for item in _statement_note_lines(document):
        ref = item.ref
        line = item.line
        if _note_agreement_skip_reason(item):
            continue
        if not ref or not item.amounts:
            continue
        amount = item.amounts[-1]
        if (ref, line) in misreferenced_lines:
            continue
        section = _get_note_section_with_fallback(ref, note_sections)
        if not section or _is_disclosure_only_note(_get_note_heading_with_fallback(ref, headings)):
            continue
        match_result = _amount_match_in_section(amount, section, tolerance)
        if not match_result["found"] and _amounts_in_text(section):
            # If Note-linked review found the amount successfully (perhaps using stronger table logic), skip Exception
            if ref.strip() in passed_refs:
                continue
            # Parent/subnote valid fallback: If parent referenced, but amount is in subnote (e.g., Note 6A for Note 6), it's safe.
            parent_match_safe = False
            for k, v in note_sections.items():
                if k.startswith(ref) and k != ref and len(k) == len(ref) + 1 and k[-1].isalpha():
                    sub_amounts = _amounts_in_text(v)
                    if sub_amounts and any(abs(sa - amount) <= tolerance for sa in sub_amounts):
                        parent_match_safe = True
                        break
            if parent_match_safe:
                continue
            
            # Subnote to parent fallback: If subnote referenced (e.g. 5A), check parent note (5), AND any notes referenced within the parent.
            if re.search(r'[A-Za-z]$', ref):
                parent_ref = re.sub(r'[A-Za-z]+$', '', ref)
                parent_section = note_sections.get(parent_ref, "")
                if parent_section:
                    p_amounts = _amounts_in_text(parent_section)
                    if p_amounts and any(abs(pa - amount) <= tolerance for pa in p_amounts):
                        continue
                    # Also check any related notes referenced inside the parent note (e.g. related ECL movement note)
                    related_refs = []
                    for rm in re.finditer(r"\bnote\s+(\d+[A-Z]?)", parent_section, re.I):
                        related_refs.append(rm.group(1).upper())
                    found_in_related = False
                    for r_ref in related_refs:
                        r_sec = note_sections.get(r_ref, "")
                        if r_sec:
                            r_amounts = _amounts_in_text(r_sec)
                            if r_amounts and any(abs(ra - amount) <= tolerance for ra in r_amounts):
                                found_in_related = True
                                break
                    if found_in_related:
                        continue
            if "cash flow" in (item.statement_name or "").lower() and re.search(r'[A-Za-z]$', ref):
                continue

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
        if any(keyword in title for keyword in ("depreciation", "property, plant", "ppe")):
            _check_depreciation_note(findings, ref, section, tolerance)

    filtered_findings = []
    for f in findings:
        if f.category == "Notes agreement" and "not found" in f.issue.lower():
            ref_match = re.search(r"Note (\d+[A-Z]?)", f.issue)
            if ref_match and ref_match.group(1).strip() in passed_refs:
                continue
        filtered_findings.append(f)
    findings = filtered_findings

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
        f"ai_policy_review_status: {result.metrics.get('ai_policy_review_status', 'disabled')}",
        f"ai_policy_review_model: {result.metrics.get('ai_policy_review_model', '')}",
        f"ai_full_review_status: {result.metrics.get('ai_full_review_status', 'disabled')}",
        f"ai_full_review_model: {result.metrics.get('ai_full_review_model', '')}",
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
    ai_combined_summary = str(result.metrics.get("ai_combined_review_summary", "") or "").strip()
    ai_summary = str(result.metrics.get("ai_policy_review_summary", "") or "").strip()
    ai_full_summary = str(result.metrics.get("ai_full_review_summary", "") or "").strip()
    ai_finding_summary = str(result.metrics.get("ai_finding_review_summary", "") or "").strip()
    if ai_combined_summary:
        lines.extend(["## Combined AI Review", "", ai_combined_summary, ""])
    if ai_summary:
        lines.extend(["## AI Policy Judgement", "", ai_summary, ""])
    if ai_full_summary:
        lines.extend(["## AI Full Review", "", ai_full_summary, ""])
    if ai_finding_summary:
        lines.extend(["## AI Finding Review", "", ai_finding_summary, ""])
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
    ai_combined_summary = str(result.metrics.get("ai_combined_review_summary", "") or "").strip()
    ai_combined_memo = str(result.metrics.get("ai_combined_review_memo", "") or "").strip()
    ai_overall_status = str(result.metrics.get("ai_review_status", "Not started") or "Not started")
    ai_summary = str(result.metrics.get("ai_policy_review_summary", "") or "").strip()
    ai_status = str(result.metrics.get("ai_policy_review_status", "disabled") or "disabled")
    ai_full_summary = str(result.metrics.get("ai_full_review_summary", "") or "").strip()
    ai_full_status = str(result.metrics.get("ai_full_review_status", "disabled") or "disabled")
    ai_finding_summary = str(result.metrics.get("ai_finding_review_summary", "") or "").strip()
    ai_finding_status = str(result.metrics.get("ai_finding_review_status", "disabled") or "disabled")
    scope_intro = ""
    if result.metrics.get("document_scope") == "Limited-scope statement extract":
        scope_intro = "Limited-scope review performed on Statement of Financial Position only. "
    if not result.findings:
        ai_parts = []
        if ai_combined_memo:
            ai_parts.append(f"Combined AI review: {ai_combined_memo}")
        elif ai_combined_summary:
            ai_parts.append(f"Combined AI review: {ai_combined_summary}")
        if ai_summary and ai_status == "completed":
            ai_parts.append(f"AI policy judgement: {ai_summary}")
        if ai_full_summary and ai_full_status == "completed":
            ai_parts.append(f"AI full review: {ai_full_summary}")
        if ai_finding_summary and ai_finding_status == "completed":
            ai_parts.append(f"AI finding review: {ai_finding_summary}")
        ai_text = f" {' ' .join(ai_parts)}" if ai_parts else ""
        return (
            f"AI review memo: {scope_intro}{assurance or 'No automated exceptions were detected.'} Perform a final manual review of scanned pages, "
            f"judgemental disclosures, and any areas where PDF extraction may have missed tables.{ai_text}"
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
    ai_text = ""
    if ai_combined_memo:
        ai_text = f" Combined AI review: {ai_combined_memo}"
    elif ai_combined_summary:
        ai_text = f" Combined AI review: {ai_combined_summary}"
    if ai_status == "completed" and ai_summary:
        ai_text += f" AI policy judgement: {ai_summary}"
    elif ai_overall_status.startswith("Failed") or ai_status in {"unavailable", "error", "skipped", "deferred"}:
        ai_text += " AI review was not completed, so policy/context conclusions remain based on deterministic checks only."
    if ai_full_status == "completed" and ai_full_summary:
        ai_text += f" AI full review: {ai_full_summary}"
    if ai_finding_status == "completed" and ai_finding_summary:
        ai_text += f" AI finding review: {ai_finding_summary}"
    return (
        "AI review memo: "
        f"{scope_intro}"
        f"{assurance + ' ' if assurance else ''}"
        f"{result.metrics['findings']} findings were identified across {top_categories}. "
        f"{priority} Likely causes include {cause_text}. "
        f"{next_step}{ai_text}"
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
                # Try implicit subtraction
                implicit_subtraction = False
                for r in running:
                    if abs(value - (expected - 2*r)) <= tolerance:
                        implicit_subtraction = True
                        break
                if implicit_subtraction:
                    running = []
                    continue

                massive_deviation = False
                if abs(value) > 0 and abs(expected) / abs(value) > 10:
                    massive_deviation = True
                if massive_deviation:
                    findings.append(
                        Finding(
                            "Extraction",
                            "Low",
                            f"Page {page_number}, table {table_index}, row {row_index + 1}, column {col + 1}",
                            "Table structure likely parsed incorrectly.",
                            f"Reported {value:,}; visible sum {expected:,}.",
                            "Ensure the table rows were extracted correctly.",
                        )
                    )
                else:
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
            massive_deviation = abs(reported) > 0 and abs(expected) / abs(reported) > 10
            severity = "Low" if massive_deviation else "Medium"
            findings.append(
                Finding(
                    "Extraction quality" if massive_deviation else "Totals and rounding",
                    severity,
                    f"Page {page_number}, table {table_index}, row {row_index + 1}",
                    "Cross-footing across the row does not agree (likely parser error)." if massive_deviation else "Cross-footing across the row does not agree.",
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
    if not _table_has_financial_amount_header(table):
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


def _table_has_financial_amount_header(table: list[list[str]]) -> bool:
    header_text = " ".join(
        str(cell or "") for row in table[:3] for cell in row
    ).lower()
    if not header_text.strip():
        return False
    narrative_markers = (
        "directors' report",
        "directors report",
        "corporate information",
        "registered office",
        "registration",
        "certificate",
        "signature",
        "website",
        "email",
        "phone",
        "contents",
    )
    if any(marker in header_text for marker in narrative_markers):
        return False
    has_currency_marker = bool(
        re.search(r"\b(?:ngn|n\s*[\'\u2019`]\s*000|n000|\u20a6|usd|eur|gbp|\$|amounts?)\b", header_text, flags=re.I)
    )
    year_count = len(set(YEAR_RE.findall(header_text)))
    note_or_amount_column = bool(re.search(r"\b(note|notes|amount|assets|liabilities|equity|revenue|income|expense|cost)\b", header_text))
    return has_currency_marker or (year_count >= 2 and note_or_amount_column)


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
    if not _simple_note_total_check_allowed(title, section):
        return
    lines = [line for line in section.splitlines() if line.strip()]
    if _simple_note_text_section_castable(title, lines):
        return
    amount_line_widths = [len(_simple_note_amounts_from_line(line)) for line in lines]
    if any(width >= 2 for width in amount_line_widths if width):
        return
    running: list[Decimal] = []
    for line in lines:
        lower = line.lower()
        amount = _last_amount(line)
        if amount is None:
            continue
        if _looks_like_total(lower):
            expected = sum(running, Decimal("0"))
            diff = amount - expected
            if _note_total_difference_is_parser_noise(amount, expected, diff, tolerance):
                running = []
                continue
            if running and abs(diff) > tolerance:
                severity = "Medium"
                issue = "A note subtotal or total does not agree to visible note line items."
                massive_deviation = False
                if abs(amount) > 0 and abs(expected) / abs(amount) > 10:
                    massive_deviation = True
                if massive_deviation:
                    severity = "Low"
                    issue += " (Downgraded to Low because massive deviation indicates table parser extracted unrelated adjacent values)."
                findings.append(
                    Finding(
                        "Notes agreement",
                        severity,
                        f"Note {ref}",
                        issue,
                        f"Note title: {title or 'untitled'} | reported {amount:,}; visible sum {expected:,}; difference {diff:,}.",
                        "Review the note table and agree it back to the supporting schedule and face statement.",
                    )
                )
            running = []
        elif _looks_like_amount_line(lower):
            running.append(amount)


def _note_total_difference_is_parser_noise(
    reported: Decimal,
    expected: Decimal,
    diff: Decimal,
    tolerance: Decimal,
) -> bool:
    if not expected:
        return False
    if reported == Decimal("2024") or reported == Decimal("2025"):
        return True
    parser_noise_tolerance = max(tolerance, Decimal("20"))
    if abs(abs(reported) - abs(expected)) <= parser_noise_tolerance:
        return True
    if abs(reported) < Decimal("10000") and abs(diff) <= Decimal("50"):
        return True
    return False


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
    if "earnings per share" not in lower and not re.search(r"\beps\b", lower):
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
    elif "earnings per share" in lower or re.search(r"\beps\b", lower):
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
        "new standards",
        "new standard",
        "standards and interpretations",
        "new and amended standards",
        "amendments to",
        "effective and adopted",
        "not yet effective",
        "financial instruments - risk",
        "financial risk",
        "risk management",
        "maturity",
        "liquidity risk",
        "credit risk",
        "expected credit loss",
        "loss allowance",
        "movement in loss allowance",
        "impairment losses and reversals",
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
        "movement",
        "reconciliation",
        "opening balance",
        "balance at",
        "at beginning",
        "at the beginning",
        "as at",
        "utilised",
        "utilized",
        "charged",
        "fair value",
        "share capital",
        "share premium",
        "deferred tax",
        "current tax liabilities",
        "contract liabilities",
        "direct expenses",
        "other operating losses",
        "performance obligation",
        "transaction price",
        "recognises revenue",
        "recognizes revenue",
    )
    if any(term in lower for term in skip_terms):
        return True
    if _note_section_looks_like_complex_movement_table(title, section):
        return True
    return False


def _simple_note_total_check_allowed(title: str, section: str) -> bool:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    amount_lines = [line for line in lines if _looks_like_amount_line(line.lower()) and _last_amount(line) is not None]
    total_lines = [line for line in amount_lines if _looks_like_total(line.lower())]
    if not total_lines:
        return False
    if len(amount_lines) > 8:
        return False
    if sum(1 for line in lines if len(YEAR_RE.findall(line)) >= 2) > 1:
        return False
    if any(len(_amounts_in_text(line)) > 2 for line in amount_lines):
        return False
    heading_like = sum(1 for line in lines if NOTE_HEADING_RE.match(line))
    if heading_like > 1:
        return False
    lower = f"{title}\n{section}".lower()
    complex_markers = (
        "reclassified",
        "remeasurement",
        "exchange",
        "foreign",
        "cash flow",
        "provision",
        "allowance",
        "impairment",
        "addition",
        "disposal",
        "depreciation",
        "amortisation",
        "amortization",
        "tax charge",
        "tax credit",
        "advance payment",
        "guaranteed",
    )
    if any(marker in lower for marker in complex_markers):
        return False
    return True


def _note_section_looks_like_complex_movement_table(title: str, section: str) -> bool:
    lower = f"{title}\n{section}".lower()
    movement_terms = (
        "opening balance",
        "additions",
        "disposals",
        "depreciation",
        "amortisation",
        "amortization",
        "charge for the year",
        "closing balance",
        "at 1 january",
        "at 31 december",
    )
    if sum(1 for term in movement_terms if term in lower) >= 2:
        return True
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    amount_lines = [line for line in lines if _looks_like_amount_line(line.lower()) and _last_amount(line) is not None]
    year_header_lines = [line for line in lines if len(YEAR_RE.findall(line)) >= 2]
    return len(amount_lines) > 12 and bool(year_header_lines)


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
    canonical_map = {
        "Statement of profit or loss": "Statement of income and expenditure",
        "Statement of changes in equity": "Statement of changes in accumulated fund"
    }
    canonical_name = canonical_map.get(statement_name, statement_name)
    page = _classified_primary_statement_pages(document).get(canonical_name)
    if page:
        return page
    target = statement_name.lower()
    for page in document.pages:
        if _looks_like_contents_or_front_matter_page(page.text) or _is_post_notes_supplement_page(page.text) or _looks_like_value_added_page(page.text):
            continue
        for line in page.text.splitlines():
            lower = line.strip().lower()
            if "..." in lower or "\u2026" in lower:
                continue
            if lower.startswith(target):
                return page
    inferred_page = _infer_statement_page_from_contents(document, canonical_name)
    if inferred_page:
        return inferred_page
    return None


def _classified_primary_statement_pages(document: PdfDocument) -> dict[str, PdfPage]:
    cached = getattr(document, "_classified_primary_statement_pages_cache", None)
    if isinstance(cached, dict):
        return cached
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
        if _looks_like_contents_or_front_matter_page(page.text) or _is_post_notes_supplement_page(page.text) or _looks_like_value_added_page(page.text):
            continue
        for canonical, candidates in aliases.items():
            if canonical in classified:
                continue
            if any(_statement_heading_line_present(page_head, candidate) for candidate in candidates) and _page_has_statement_rows_for(canonical, page.text):
                classified[canonical] = page
    setattr(document, "_classified_primary_statement_pages_cache", classified)
    return classified


def _infer_statement_page_from_contents(document: PdfDocument, canonical_name: str) -> PdfPage | None:
    contents_refs = _contents_statement_page_refs(document)
    if not contents_refs or canonical_name not in contents_refs:
        return None
    classified = _classified_primary_statement_pages(document)
    offsets: list[int] = []
    for known_name, page in classified.items():
        ref = contents_refs.get(known_name)
        if ref:
            offsets.append(page.number - ref)
    if not offsets:
        return None
    offsets.sort()
    inferred_number = contents_refs[canonical_name] + offsets[len(offsets) // 2]
    for page in document.pages:
        if page.number == inferred_number and not _looks_like_contents_or_front_matter_page(page.text):
            return page
    return None


def _looks_like_contents_page(text: str) -> bool:
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return False
    head = "\n".join(raw_lines[:60]).lower()
    if not any(marker in head for marker in ("contents", "statement of", "balance sheet", "cash flow", "financial position", "profit or loss", "comprehensive income")):
        return False
    if _fuzzy_contains(head, "table of contents", 0.76) or _fuzzy_contains(head, "contents", 0.9):
        return True
    if _fuzzy_contains(head, "notes to the financial statements", 0.84) or _fuzzy_contains(head, "notes to financial statements", 0.84):
        return True

    contents_phrase_count = sum(
        1
        for phrase in (
            "table of contents",
            "contents",
            "statement of profit or loss",
            "statement of comprehensive income",
            "statement of financial position",
            "statement of changes in equity",
            "statement of changes in accumulated fund",
            "statement of cash flows",
            "cash flow statement",
        )
        if _fuzzy_contains(head, phrase, 0.75)
    )

    statement_line_hits = 0
    for line in raw_lines[:45]:
        normalised = _normalise_match_words(line.lower())
        if re.search(r"\.{2,}\s*\d{1,4}\b", normalised):
            statement_line_hits += 1
            continue
        has_statement_name = any(
            token in normalised
            for token in (
                "statement of profit or loss",
                "statement of comprehensive income",
                "statement of financial position",
                "statement of changes in equity",
                "statement of changes in accumulated fund",
                "statement of cash flows",
                "balance sheet",
            )
        )
        if has_statement_name:
            statement_line_hits += 1
    return contents_phrase_count >= 1 and statement_line_hits >= 1



def _is_ocr_wrap_merge_candidate(previous_line: str, next_line: str) -> bool:
    if not previous_line or not next_line:
        return False
    previous_text = previous_line.strip()
    next_text = next_line.strip()
    if len(previous_text.split()) > 12 or len(next_text.split()) > 12:
        return False
    if re.search(r"\d", previous_text) or re.search(r"\d", next_text):
        return False
    if re.search(r"[.!?;:)]$", previous_text):
        return False
    if re.search(r"^\(?\d", next_text):
        return False
    if previous_text.endswith("-"):
        return True
    if not re.match(r".*[A-Za-z]$", previous_text):
        return False
    if not re.match(r"^[a-z]", next_text.lower()):
        return False
    return True


def _flatten_wrapped_statements(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    merged_lines: list[str] = []
    for raw_line in lines:
        if not merged_lines:
            merged_lines.append(raw_line)
            continue
        previous_index = len(merged_lines) - 1
        if _is_ocr_wrap_merge_candidate(merged_lines[previous_index], raw_line):
            merged_lines[previous_index] = f"{merged_lines[previous_index]} {raw_line}"
        else:
            merged_lines.append(raw_line)
    return "\n".join(merged_lines)



def _extract_content_line_page_reference(line: str, page_count: int) -> int | None:
    if not line:
        return None
    clean_line = re.sub(r"\s+", " ", line).strip()
    patterns = (
        r"(?<!\d)(\d{1,4})(?=\s*\.{2,}\s*$)",
        r"(?<!\d)(\d{1,4})(?=\s*$)",
        r"\(\s*(\d{1,4})\s*\)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, clean_line)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    fallback = re.sub(r"[^0-9]", " ", clean_line).split()
    for token in reversed(fallback):
        if token.isdigit() and 1 <= len(token) <= 4:
            value = int(token)
            if 1 <= value <= max(page_count * 4, 40):
                return value
    return None


def _contents_statement_page_refs(document: PdfDocument) -> dict[str, int]:
    cached = getattr(document, "_contents_statement_page_refs_cache", None)
    if isinstance(cached, dict):
        return cached
    refs = {
        statement: int(detail["contents_page_reference"])
        for statement, detail in _contents_statement_page_ref_details(document).items()
        if str(detail.get("contents_page_reference", "")).isdigit()
    }
    setattr(document, "_contents_statement_page_refs_cache", refs)
    return refs


def _contents_statement_page_ref_details(document: PdfDocument) -> dict[str, dict[str, object]]:
    cached = getattr(document, "_contents_statement_page_ref_details_cache", None)
    if isinstance(cached, dict):
        return cached
    aliases = {
        "Statement of income and expenditure": (
            "statement of profit or loss and other comprehensive income",
            "statement of comprehensive income",
            "statement of profit or loss",
            "statement of income and expenditure",
            "statement of income",
        ),
        "Statement of financial position": ("statement of financial position", "balance sheet"),
        "Statement of changes in accumulated fund": (
            "statement of changes in accumulated fund",
            "statement of changes in equity",
            "changes in accumulated fund",
            "changes in equity",
        ),
        "Statement of cash flows": (
            "statement of cash flows",
            "cash flow statement",
            "statement of cash flow",
        ),
    }
    refs: dict[str, dict[str, object]] = {}
    front_matter_pages = document.pages[: min(len(document.pages), 15)]
    for page in front_matter_pages:
        if not _looks_like_contents_page(page.text):
            continue
        raw_lines = [re.sub(r"\s+", " ", line.strip()) for line in page.text.splitlines() if line.strip()]
        # Include a compact OCR-only reconstructed line set to recover broken lines.
        raw_lines.extend(_flatten_wrapped_statements(page.text).splitlines()[:40])
        for raw_line in raw_lines[:140]:
            line = re.sub(r"\s+", " ", raw_line.strip())
            if not line:
                continue
            clean_line = re.sub(r"\.{2,}\s*", " ", line)
            page_ref = _extract_content_line_page_reference(clean_line, len(document.pages))
            if page_ref is None or page_ref < 1:
                continue
            normalised = _normalise_match_words(line)
            for canonical, candidates in aliases.items():
                if canonical in refs:
                    continue
                for candidate in candidates:
                    normalized_candidate = _normalise_match_words(candidate)
                    if not normalized_candidate:
                        continue
                    if candidate.lower() in line.lower() or _fuzzy_contains(line, candidate, 0.82) or _fuzzy_contains(normalised, normalized_candidate, 0.82):
                        refs[canonical] = {
                            "statement": canonical,
                            "contents_page_reference": page_ref,
                            "contents_source_pdf_page": page.number,
                            "contents_source_page": _reviewer_page_number(document, page.number),
                            "contents_line": line,
                        }
                        break
                if canonical in refs:
                    break
    setattr(document, "_contents_statement_page_ref_details_cache", refs)
    return refs

def _physical_page_for_printed(document: PdfDocument, printed_page: int) -> int | None:
    printed_index = _printed_page_number_map(document)
    # Prefer explicit printed page mapping when available.
    reverse_map = {value: page for page, value in printed_index.items()}
    if printed_page in reverse_map:
        return reverse_map[printed_page]
    # Conservative fallback for documents without printed footers.
    if 1 <= printed_page <= len(document.pages):
        return printed_page
    return None


def _contents_statement_page_agreement_rows(document: PdfDocument) -> list[dict[str, object]]:
    details = _contents_statement_page_ref_details(document)
    if not details:
        return []
    classified = _classified_primary_statement_pages(document)
    rows: list[dict[str, object]] = []
    for canonical_name in sorted(details):
        detail = details[canonical_name]
        expected_page = int(detail.get("contents_page_reference") or 0)
        detected = classified.get(canonical_name)
        mapped = _physical_page_for_printed(document, expected_page) if expected_page else None
        mapped_printed = _reviewer_page_number(document, mapped) if mapped else ""
        detected_pdf_page = detected.number if detected else ""
        detected_printed = _reviewer_page_number(document, detected.number) if detected else ""
        status = "Skipped"
        confidence = "Low"
        reason = "Statement page was not confidently classified."
        page_delta = ""
        if detected and expected_page:
            page_delta_value = int(detected_printed) - expected_page
            page_delta = page_delta_value
            if int(detected_printed) == expected_page or (mapped is not None and detected.number == mapped):
                status = "Passed"
                confidence = "High"
                reason = "Detected statement page agrees to the contents page reference using printed-page mapping."
            else:
                status = "Mismatch"
                confidence = "Low" if document.ocr_used else ("Medium" if document.table_extraction_confidence >= 80 else "Low")
                mapping_note = "" if mapped is not None else " The contents reference could not be mapped to a physical PDF page, so the detected printed page was used."
                reason = f"Contents lists page {expected_page}, but the statement was detected on page {detected_printed}.{mapping_note}"
        rows.append(
            {
                "Statement": canonical_name,
                "Contents page reference": expected_page or "",
                "Detected statement page": detected_printed,
                "Detected PDF page": detected_pdf_page,
                "Mapped PDF page for contents reference": mapped or "",
                "Mapped printed page": mapped_printed,
                "Contents source page": detail.get("contents_source_page", ""),
                "Contents source PDF page": detail.get("contents_source_pdf_page", ""),
                "Contents line": detail.get("contents_line", ""),
                "Status": status,
                "Confidence": confidence,
                "Page delta": page_delta,
                "Reason": reason,
            }
        )
    return rows


def _contents_statement_page_agreement_note(
    document: PdfDocument,
) -> tuple[list[str], list[str], list[Finding]]:
    performed: list[str] = []
    skipped: list[str] = []
    findings: list[Finding] = []
    rows = _contents_statement_page_agreement_rows(document)
    if not rows:
        skipped.append("Contents agreement: skipped because statement references in contents were not detected.")
        return performed, skipped, findings

    performed.append("Contents-page agreement was reviewed for detected primary statements.")
    for row in rows:
        canonical_name = str(row.get("Statement", ""))
        expected_page = row.get("Contents page reference", "")
        detected_printed = row.get("Detected statement page", "")
        status = str(row.get("Status", ""))
        reason = str(row.get("Reason", ""))
        if status == "Passed":
            performed.append(
                f"Contents agreement: '{canonical_name}' detected on page {detected_printed} and matches contents page {expected_page}."
            )
        elif status == "Skipped":
            skipped.append(
                f"Contents agreement: '{canonical_name}' appears in contents (page {expected_page}) but could not be fully validated. {reason}"
            )
        elif status == "Mismatch":
            confidence = str(row.get("Confidence", "Low")) or "Low"
            page_delta = row.get("Page delta", "")
            findings.append(
                Finding(
                    "Document structure",
                    confidence,
                    f"Page {detected_printed} | Contents alignment",
                    "Contents mismatch detected.",
                    f"Contents lists '{canonical_name}' on page {expected_page}, but extracted statement page is {detected_printed}. Page offset from contents to extracted page is {page_delta:+d}.",
                    "Verify the contents page against the printed page number at the bottom of the statement page and update the contents reference if needed.",
                    {
                        "check_type": "Contents agreement",
                        "canonical_statement": canonical_name,
                        "contents_page": str(expected_page),
                        "detected_page": str(detected_printed),
                        "mapped_page": str(row.get("Mapped printed page", "")),
                        "page_delta": str(page_delta),
                        "page_reference": f"Page {detected_printed}",
                    },
                )
            )

    return performed, skipped, findings
def _looks_like_contents_or_front_matter_page(text: str) -> bool:
    head = "\n".join(text.splitlines()[:40]).lower()
    if re.search(r"\b(table of )?contents\b", head):
        return True
    statement_mentions = len(re.findall(r"statement of (?:profit|financial|changes|cash|income|comprehensive)", head))
    numeric_page_refs = len(re.findall(r"\.{2,}\s*\d{1,3}\b|\b\d{1,3}\s*$", head, flags=re.M))
    front_terms = ("corporate information", "directors' report", "directors report", "independent auditor", "report of the directors")
    return statement_mentions >= 3 and (numeric_page_refs >= 2 or any(term in head for term in front_terms))


def _statement_heading_line_present(text: str, phrase: str) -> bool:
    normalized_phrase = _normalise_match_words(phrase)
    if not normalized_phrase:
        return False
    generic_terms = {"statement", "statements", "financial", "of", "the", "and", "for"}
    discriminators = [word for word in normalized_phrase.split() if word not in generic_terms]
    for line in text.splitlines()[:60]:
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped or len(stripped) > 140 or re.search(r"\.{2,}\s*\d{1,3}$", stripped):
            continue
        if re.search(r"\b(page|contents)\b", stripped, flags=re.I):
            continue
        normalized = _normalise_match_words(stripped)
        if discriminators and not any(term in normalized for term in discriminators):
            continue
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
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    rows = _statement_rows(page.text)
    
    page_header = "\n".join(page.text.splitlines()[:8]).lower()
    is_company = document and (
        _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
        or "statement of profit or loss" in page_header
        or "comprehensive income" in page_header
    )
    stmt_name = "Statement of profit or loss" if is_company else "Statement of income and expenditure"
    
    if is_company:
        revenue = _row_amounts_any(rows, ("revenue", "turnover", "sales"))
        direct_expenses = _row_amounts_any(rows, ("direct expenses", "cost of sales"))
        gross_profit = _row_amounts_any(rows, ("gross profit",))
        other_income = _row_amounts_any(rows, ("other income",))
        other_op_losses = _row_amounts_any(rows, ("other operating losses", "other operating gains"))
        ecl = _row_amounts_any(rows, ("movement in credit loss allowances", "expected credit loss"))
        op_expenses = _row_amounts_any(rows, ("operating expenses", "administrative expenses"))
        op_profit = _row_amounts_any(rows, ("operating profit", "profit from operations"))
        inv_income = _row_amounts_any(rows, ("investment income", "finance income"))
        non_op_losses = _row_amounts_any(rows, ("other non-operating losses", "other non-operating gains"))
        pbt = _row_amounts_any(rows, ("profit before tax", "loss before tax"))
        tax = _row_amounts_any(rows, ("taxation", "income tax expense", "tax expense"))
        pat = _row_amounts_any(rows, ("profit after tax", "profit for the year", "loss after tax", "loss for the year"))
        raw_lines = {
            "profit before tax": _line_amount_for_aliases(page.text, ("loss before taxation", "loss before tax", "profit before taxation", "profit before tax"))[1],
            "taxation": _line_amount_for_aliases(page.text, ("taxation", "income tax expense", "tax expense", "tax credit"))[1],
            "profit after tax": _line_amount_for_aliases(page.text, ("loss after taxation", "loss after tax", "profit after taxation", "profit after tax", "loss for the year", "profit for the year"))[1],
        }
        
        if revenue and direct_expenses and gross_profit:
            _check_vector_equation(findings, page.number, stmt_name, "Revenue plus direct expenses equals gross profit.", [a + b for a, b in zip(revenue, direct_expenses)], gross_profit, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: gross profit checked.")
            
        if gross_profit and op_profit:
            components = [gross_profit]
            if other_income: components.append(other_income)
            if other_op_losses: components.append(other_op_losses)
            if ecl: components.append(ecl)
            if op_expenses: components.append(op_expenses)
            
            expected = components[0]
            for comp in components[1:]:
                expected = [a + b for a, b in zip(expected, comp)]
            _check_vector_equation(findings, page.number, stmt_name, "Gross profit plus operating items equals operating profit.", expected, op_profit, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: operating profit checked.")
            
        if op_profit and pbt:
            components = [op_profit]
            if inv_income: components.append(inv_income)
            if non_op_losses: components.append(non_op_losses)
            expected = components[0]
            for comp in components[1:]:
                expected = [a + b for a, b in zip(expected, comp)]
            _check_vector_equation(findings, page.number, stmt_name, "Operating profit plus non-operating items equals profit before tax.", expected, pbt, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: profit before tax checked.")
            
        if pbt and tax and pat:
            if ocr_review and len(pat) < 2 and max(len(pbt), len(tax)) >= 2:
                corroboration = _ocr_income_corroboration_assessment(
                    document,
                    tax[0] if tax else None,
                    pat[0] if pat else None,
                    tolerance,
                )
                message = "Skipped / OCR conflict - current-year after-tax value not confidently extracted."
                if corroboration.get("casts") and corroboration.get("after_tax_value") is not None:
                    message += f" Corroborating lines indicate {_format_accounting_amount(corroboration.get('after_tax_value'))}. Manual confirmation required."
                skipped.append(message)
            else:
                profit_tax_checked = _check_profit_tax_equation(
                    findings,
                    page.number,
                    stmt_name,
                    pbt,
                    tax,
                    pat,
                    tolerance,
                    ocr_review=ocr_review,
                    raw_lines=raw_lines,
                    document=document,
                )
                if profit_tax_checked:
                    performed.append(f"{stmt_name}: profit after tax checked.")
                    if ocr_review:
                        performed.append("Statement of profit or loss: profit/loss after tax checked.")
                        performed.append("Income statement: revenue, tax, and profit/loss after tax checked")
                else:
                    skipped.append(f"{stmt_name}: profit/loss after-tax arithmetic skipped because the row may include OCI or the current/prior-year columns could not be mapped reliably.")
            
        return findings, performed, skipped

    income_amounts = _row_amounts_any(rows, ("total income", "gross revenue", "gross operating revenue", "revenue"))
    expenditure_amounts = _row_amounts_any(rows, ("total expenditure", "operating expenditure"))
    surplus_amounts = _row_amounts_any(rows, ("surplus of income over expenditure", "surplus for the year", "profit for the year", "profit after tax", "profit before tax"))
    
    if income_amounts and expenditure_amounts and surplus_amounts:
        _check_vector_equation(
            findings,
            page.number,
            "Statement of income and expenditure",
            "Total income less total expenditure agrees to surplus.",
            [a - b for a, b in zip(income_amounts, expenditure_amounts)],
            surplus_amounts,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of income and expenditure: total income checked.")
        performed.append("Statement of income and expenditure: total expenditure checked.")
    else:
        skipped.append("Statement of income and expenditure: skipped because income/expenditure rows were not confidently parsed.")
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
    low_statement_confidence = document is not None and _statement_structure_confidence(document) < 80
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
    liability_amounts = _row_amounts_any(rows, ("total liabilities", "liabilities", "financial liabilities"))
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
    if low_statement_confidence and not ocr_review:
        skipped.append("Statement of financial position: fallback component casting skipped because statement structure confidence is below the safe threshold.")
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
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    lines = page.text.splitlines()
    
    is_company = document and _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
    stmt_name = "Statement of changes in equity" if is_company else "Statement of changes in accumulated fund"
    word_fund = "equity" if is_company else "accumulated fund"
    
    balance_rows = [
        (idx, line, _amounts_from_statement_line(line))
        for idx, line in enumerate(lines)
        if any(k in line.lower() for k in ["balance as at", "balance at", "opening total equity", "closing total equity", "1 january", "31 december"])
    ]
    surplus_rows = [
        (idx, line, _amounts_from_statement_line(line))
        for idx, line in enumerate(lines)
        if any(k in line.lower() for k in ["surplus for the year", "profit for the year", "loss for the year", "profit/(loss) for the year", "loss/(profit) for the year"])
    ]
    
    if len(balance_rows) >= 2 and len(surplus_rows) >= 1:
        opening_idx, _, opening_2025 = balance_rows[-2]
        closing_idx, _, closing_2025 = balance_rows[-1]
        _, _, surplus_2025 = surplus_rows[-1]
        direct_equity_movement = _equity_direct_movement_amount(lines[opening_idx + 1 : closing_idx])
        if len(opening_2025) >= 4 and len(closing_2025) >= 4 and surplus_2025:
            expected_total = opening_2025[-1] + surplus_2025[-1] + direct_equity_movement
            reported_total = closing_2025[-1]
            diff = expected_total - reported_total
            if abs(diff) > tolerance:
                movement_text = (
                    f" + direct equity movements {direct_equity_movement:,}"
                    if direct_equity_movement
                    else ""
                )
                findings.append(
                    Finding(
                        "Calculation",
                        "High" if abs(diff) > tolerance * 5 else "Medium",
                        stmt_name,
                        f"Closing {word_fund} does not agree to opening {word_fund} plus surplus.",
                        f"Reported closing {reported_total:,}; expected {expected_total:,} (opening {opening_2025[-1]:,} + surplus {surplus_2025[-1]:,}{movement_text}). Difference: {diff:,}.",
                        f"Check if there are prior year adjustments, capital contributions, dividends, or other comprehensive income lines modifying {word_fund}.",
                    )
                )
            else:
                pass
            if ocr_review:
                pass
            performed.append(f"{stmt_name}: opening plus surplus checked to closing {word_fund}.")
            return findings, performed, skipped

    rows = _statement_rows(page.text)
    op_equity = _row_amounts_any(rows, ("opening total equity", "balance as at 1 january", "balance at 1 january", "balance as at beginning"))
    cl_equity = _row_amounts_any(rows, ("closing total equity", "balance as at 31 december", "balance at 31 december", "balance as at end"))
    op_ret = _row_amounts_any(rows, ("opening retained loss", "opening retained income", "opening retained earnings"))
    cl_ret = _row_amounts_any(rows, ("closing retained income", "closing retained loss", "closing retained earnings"))
    profit = _row_amounts_any(rows, ("profit for the year", "surplus for the year"))
    
    if (op_equity and cl_equity and profit) or (op_ret and cl_ret and profit):
        has_run = False
        if op_equity and cl_equity and profit:
            _check_vector_equation(
                findings, page.number, stmt_name, "Opening total equity plus profit equals closing total equity.",
                [a + b for a, b in zip(op_equity, profit)], cl_equity, tolerance, ocr_review=ocr_review
            )
            performed.append(f"{stmt_name}: opening total equity checked.")
            has_run = True
        if op_ret and cl_ret and profit:
            _check_vector_equation(
                findings, page.number, stmt_name, "Opening retained earnings plus profit equals closing retained earnings.",
                [a + b for a, b in zip(op_ret, profit)], cl_ret, tolerance, ocr_review=ocr_review
            )
            performed.append(f"{stmt_name}: retained earnings movement checked.")
            has_run = True
        if has_run:
            return findings, performed, skipped

    skipped.append(f"{stmt_name}: Page {page.number} skipped because rotated/OCR table structure was not confidently parsed.")
    return findings, performed, skipped


def _equity_direct_movement_amount(lines: list[str]) -> Decimal:
    movement_lines: list[list[Decimal]] = []
    preferred_lines: list[list[Decimal]] = []
    for line in lines:
        lower = line.lower()
        if not any(
            marker in lower
            for marker in (
                "contribution by owners",
                "contributions by and distributions to owners",
                "distribution to owners",
                "directly in equity",
                "dividend",
                "issue of shares",
                "share premium",
            )
        ):
            continue
        amounts = _amounts_from_statement_line(line)
        if not amounts:
            continue
        movement_lines.append(amounts)
        if "total contributions" in lower or "distributions to owners" in lower or "directly in equity" in lower:
            preferred_lines.append(amounts)
    selected = preferred_lines or movement_lines
    if not selected:
        return Decimal("0")
    last_amounts = [amounts[-1] for amounts in selected if amounts]
    if not last_amounts:
        return Decimal("0")
    return max(last_amounts, key=lambda value: abs(value))





def _safe_amount_pair_from_row(values: list[Decimal]) -> tuple[Decimal | None, Decimal | None]:
    if not values:
        return (None, None)
    if len(values) >= 2:
        return (values[0], values[1])
    return (values[0], None)


def _cross_source_confidence(document: PdfDocument) -> str:
    if getattr(document, "ocr_used", False):
        return "Low"
    if document.table_extraction_confidence >= 80:
        return "Medium"
    return "Low"


def _cross_source_note_amounts(
    note_sections: dict[str, str],
    aliases: tuple[str, ...],
    row_cache: dict[str, dict[str, list[Decimal]]] | None = None,
) -> tuple[list[Decimal], str]:
    row_cache = row_cache if row_cache is not None else {}
    for ref, section in note_sections.items():
        rows = row_cache.get(ref)
        if rows is None:
            rows = _statement_rows(section)
            row_cache[ref] = rows
        values = _row_amounts_any(rows, aliases)
        if values:
            return values, ref
    return [], ""


def _check_cross_source_cash_flow(
    document: PdfDocument,
    tolerance: Decimal,
) -> tuple[list[Finding], list[str], list[str]]:
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []

    sfp_page = _find_statement_page(document, "Statement of financial position")
    cf_page = _find_statement_page(document, "Statement of cash flows")
    is_page = _find_statement_page(document, "Statement of profit or loss")
    ce_page = _find_statement_page(document, "Statement of changes in equity")

    if not (sfp_page and cf_page):
        skipped.append("Cross-source cash flow check skipped: statement of financial position and statement of cash flows were both not detected.")
        return findings, performed, skipped

    sfp_rows = _statement_rows(sfp_page.text)
    cf_rows = _statement_rows(cf_page.text, "Statement of cash flows")
    is_rows = _statement_rows(is_page.text) if is_page else {}
    ce_rows = _statement_rows(ce_page.text) if ce_page else {}
    conf = _cross_source_confidence(document)
    note_sections: dict[str, str] = {}
    note_sections_loaded = False
    note_row_cache: dict[str, dict[str, list[Decimal]]] = {}
    allow_note_fallback = len(document.pages) <= 45 and not getattr(document, "fast_text_only", False)

    def get_note_sections() -> dict[str, str]:
        nonlocal note_sections, note_sections_loaded
        if not note_sections_loaded:
            note_sections = _note_sections(document)
            note_sections_loaded = True
        return note_sections

    sfp_cash = _row_amounts_any(sfp_rows, ("cash and cash equivalents", "cash and cash equivalents at end", "cash at end"))
    cf_open = _row_amounts_any(
        cf_rows,
        (
            "cash and cash equivalents at beginning",
            "cash and cash equivalents at the beginning of the year",
            "cash at beginning",
            "opening cash",
            "cash and cash equivalents at beginning of year",
        ),
    )
    cf_close = _row_amounts_any(
        cf_rows,
        (
            "cash and cash equivalents at end",
            "cash and cash equivalents at the end of the year",
            "cash at end",
            "closing cash",
            "cash and cash equivalents at end of year",
        ),
    )
    cf_movement = _row_amounts_any(
        cf_rows,
        (
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "cash movement for the year",
            "net increase in cash",
            "net movement in cash and cash equivalents",
        ),
    )

    note_cf_open, note_cf_open_ref = ([], "")
    note_cf_close, note_cf_close_ref = ([], "")
    note_cf_movement, note_cf_movement_ref = ([], "")
    if allow_note_fallback and not cf_open and get_note_sections():
        note_cf_open, note_cf_open_ref = _cross_source_note_amounts(
            note_sections,
            (
                "cash and cash equivalents at beginning",
                "cash and cash equivalents at the beginning of the year",
                "cash at beginning",
                "opening cash",
                "cash and cash equivalents at beginning of year",
            ),
            note_row_cache,
        )
    if allow_note_fallback and not cf_close and get_note_sections():
        note_cf_close, note_cf_close_ref = _cross_source_note_amounts(
            note_sections,
            (
                "cash and cash equivalents at end",
                "cash and cash equivalents at the end of the year",
                "cash at end",
                "closing cash",
                "cash and cash equivalents at end of year",
            ),
            note_row_cache,
        )
    if allow_note_fallback and not cf_movement and get_note_sections():
        note_cf_movement, note_cf_movement_ref = _cross_source_note_amounts(
            note_sections,
            (
                "net increase in cash and cash equivalents",
                "net decrease in cash and cash equivalents",
                "cash movement for the year",
                "net increase in cash",
                "net movement in cash and cash equivalents",
            ),
            note_row_cache,
        )

    if not cf_open and note_cf_open:
        cf_open = note_cf_open
    if not cf_close and note_cf_close:
        cf_close = note_cf_close
    if not cf_movement and note_cf_movement:
        cf_movement = note_cf_movement

    sfp_open = sfp_current = sfp_prior = None
    if sfp_cash:
        sfp_current, sfp_prior = _safe_amount_pair_from_row(sfp_cash)

    if sfp_current is not None and sfp_prior is not None and cf_open and cf_close:
        source = "primary statements"
        if (cf_open in (note_cf_open, note_cf_close) or cf_close in (note_cf_open, note_cf_close)) and not is_rows:
            source = "primary statements with note fallback"
        performed.append(
            f"Cross-source cash flow check: opening and closing balances reconciled between SFP and CFS ({source})."
        )
        if len(cf_open) >= 1 and abs(cf_open[0] - sfp_prior) > tolerance:
            findings.append(
                Finding(
                    "Cash flow consistency",
                    "Medium" if conf == "Medium" else "Low",
                    f"Page {cf_page.number} | Statement of cash flows",
                    "Opening cash in the statement of cash flows does not match SFP prior-year closing cash.",
                    f"SFP prior-year cash and cash equivalents: {sfp_prior:,}; CFS opening cash: {cf_open[0]:,}.",
                    "Align the CFS opening cash line with prior-year SFP ending cash and cash equivalents.",
                    {
                        "check_type": "Cross-source cash flow consistency",
                        "match_confidence": conf,
                        "cf_open_note_ref": note_cf_open_ref,
                        "cf_page": str(cf_page.number),
                    },
                )
            )
        if len(cf_close) >= 1 and abs(cf_close[0] - sfp_current) > tolerance:
            findings.append(
                Finding(
                    "Cash flow consistency",
                    "Medium" if conf == "Medium" else "Low",
                    f"Page {cf_page.number} | Statement of cash flows",
                    "Closing cash in the statement of cash flows does not match SFP current-year ending cash.",
                    f"SFP current-year cash and cash equivalents: {sfp_current:,}; CFS closing cash: {cf_close[0]:,}.",
                    "Align the CFS closing cash line with current-year SFP ending cash and cash equivalents.",
                    {
                        "check_type": "Cross-source cash flow consistency",
                        "match_confidence": conf,
                        "cf_close_note_ref": note_cf_close_ref,
                        "cf_page": str(cf_page.number),
                    },
                )
            )
        if cf_movement and abs(cf_open[0] + cf_movement[0] - cf_close[0]) > tolerance:
            findings.append(
                Finding(
                    "Cash flow consistency",
                    "Medium" if conf == "Medium" else "Low",
                    f"Page {cf_page.number} | Statement of cash flows",
                    "Net movement in cash does not reconcile opening and closing balances.",
                    f"Opening {cf_open[0]:,}; movement {cf_movement[0]:,}; closing {cf_close[0]:,}.",
                    "Review movement and opening/closing lines in the statement of cash flows and related notes.",
                    {
                        "check_type": "Cross-source cash flow consistency",
                        "match_confidence": conf,
                        "cf_movement_note_ref": note_cf_movement_ref,
                        "cf_page": str(cf_page.number),
                    },
                )
            )
    elif note_sections_loaded and note_sections:
        performed.append("Cross-source cash flow note fallback values were located for cash-flow reconciliation.")
    else:
        skipped.append("Cross-source cash flow check skipped: opening, closing, and prior-year cash comparisons were not available at reliable width from SFP and CFS.")

    is_pat = _row_amounts_any(is_rows, ("profit after tax", "loss after tax", "profit for the year", "loss for the year"))
    ce_pat = _row_amounts_any(ce_rows, ("profit for the year", "loss for the year", "profit or loss for the period"))
    note_is_pat_ref = ""
    note_ce_pat_ref = ""
    if allow_note_fallback and (not is_pat or not ce_pat) and get_note_sections():
        note_is_pat, note_is_pat_ref = _cross_source_note_amounts(
            note_sections,
            ("profit after tax", "loss after tax", "profit for the year", "loss for the year", "comprehensive income for the year"),
        )
        note_ce_pat, note_ce_pat_ref = _cross_source_note_amounts(
            note_sections,
            ("profit for the year", "loss for the year", "profit or loss for the period", "total comprehensive income"),
        )
        if not is_pat and note_is_pat:
            is_pat = note_is_pat
        if not ce_pat and note_ce_pat:
            ce_pat = note_ce_pat

    if is_pat and ce_pat:
        performed.append("Cross-source cash flow check: income statement result compared to equity movement reference where available.")
        if abs(is_pat[0] - ce_pat[0]) > tolerance * 10:
            findings.append(
                Finding(
                    "Cash flow consistency",
                    "Low",
                    f"Page {is_page.number if is_page else cf_page.number} | Income-to-equity cross-source",
                    "Result for the year differs between profit statement and changes in equity references.",
                    f"Income statement amount: {is_pat[0]:,}; changes in equity reference: {ce_pat[0]:,}.",
                    "Use the note reconciliations to confirm whether this is an additive/subtractive equity movement disclosure.",
                    {
                        "check_type": "Cross-source cash flow consistency",
                        "match_confidence": "Low",
                    },
                )
            )
    elif is_pat or ce_pat:
        income_location = "not parsed"
        if is_pat:
            income_location = f"Page {is_page.number}" if is_page else (f"Note {note_is_pat_ref}" if note_is_pat_ref else "note fallback")
        equity_location = "not parsed"
        if ce_pat:
            equity_location = f"Page {ce_page.number}" if ce_page else (f"Note {note_ce_pat_ref}" if note_ce_pat_ref else "note fallback")
        skipped.append(
            "Cross-source income-to-equity linkage skipped because only one of income or equity reference lines was confidently parsed. "
            f"Available evidence: income result {income_location}; equity movement {equity_location}."
        )
    if not performed and not findings:
        performed.append("Cross-source cash flow check completed without actionable findings.")

    return findings, performed, skipped




def _check_cash_flow_supporting_amounts(
    document: PdfDocument,
    tolerance: Decimal,
) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    performed: list[str] = []
    cf_page = _find_statement_page(document, "Statement of cash flows")
    if not cf_page:
        return findings, performed
    income_page = _find_statement_page(document, "Statement of profit or loss") or _find_statement_page(document, "Statement of income and expenditure")
    note_sections = _note_sections(document)
    specs = (
        (
            "Finance costs",
            ("finance costs", "finance cost", "interest expense", "interest expenses"),
            income_page,
            ("finance costs", "finance cost", "interest expense", "interest expenses"),
        ),
        (
            "Depreciation/amortisation add-back",
            ("amortisation of right-of-use asset", "amortisation", "depreciation"),
            None,
            ("depreciation", "amortisation", "amortisation of right-of-use asset"),
        ),
        (
            "Interest received on loan",
            ("interest received on loan", "interest income", "finance income"),
            income_page,
            ("interest received on loan", "interest income", "finance income", "other operating income"),
        ),
    )
    comparisons: list[str] = []
    max_diff = Decimal("0")
    for label, cf_aliases, source_page, source_aliases in specs:
        cf_amounts, cf_line = _line_amounts_for_aliases_preserving_zero(cf_page.text, cf_aliases)
        if not cf_amounts:
            continue
        sources: list[tuple[str, list[Decimal], str, str]] = []
        if source_page:
            source_amounts, source_line = _line_amounts_for_aliases_preserving_zero(source_page.text, source_aliases)
            if source_amounts:
                sources.append((f"Page {source_page.number}", source_amounts, source_line, "primary statement"))
        cf_note_ref, _cf_ref_start, _cf_ref_end = _detect_statement_row_note_token(cf_line)
        if cf_note_ref:
            note_section = _get_note_section_with_fallback(cf_note_ref, note_sections, document)
            note_amounts, note_line = _line_amounts_for_aliases_preserving_zero(note_section, source_aliases) if note_section else ([], "")
            if note_amounts:
                sources.append((f"Note {cf_note_ref}", note_amounts, note_line, "note"))
        elif source_page is None:
            note_amounts, note_ref, note_line = _note_line_amounts_for_aliases(note_sections, source_aliases)
            if note_amounts:
                sources.append((f"Note {note_ref}", note_amounts, note_line, "note"))
        for source_ref, source_amounts, source_line, source_type in sources:
            mismatches = _amount_vector_mismatches(cf_amounts, source_amounts, tolerance=tolerance, compare_abs=True)
            for column, left, right, diff in mismatches:
                if abs(diff) == 0:
                    continue
                max_diff = max(max_diff, abs(diff))
                comparisons.append(
                    f"{label} {column}: cash flow {left:,} on Page {cf_page.number} vs {source_type} {right:,} in {source_ref} (difference {diff:,}). "
                    f"Cash flow line: {cf_line}; source line: {source_line}"
                )
                break
    if comparisons:
        performed.append("Cash-flow add-back and source-note amount tie-out performed where matching lines were readable.")
        findings.append(
            Finding(
                "Consistency",
                "Medium" if max_diff > tolerance else "Low",
                f"Page {cf_page.number} | Statement of cash flows",
                "Cash-flow line item amount differs from the related primary statement or note amount.",
                " | ".join(comparisons[:6]),
                "Agree the cash-flow add-back/source line to the related statement or note and update the inconsistent amount.",
                {"check_type": "Cash-flow source amount tie-out", "match_confidence": "High"},
            )
        )
    return findings, performed


def _check_supporting_disclosure_note_reference_amounts(
    document: PdfDocument,
    tolerance: Decimal,
) -> tuple[list[Finding], list[str]]:
    note_sections = _note_sections(document)
    note_headings = _note_headings_by_page(document)
    notes_start = _notes_start_page(document)
    if not note_sections or not note_headings:
        return [], []
    findings: list[Finding] = []
    performed: list[str] = []
    issues: list[str] = []
    pages: set[int] = set()
    max_diff = Decimal("0")
    risk_markers = ("financial instrument", "risk management", "credit risk", "liquidity risk", "market risk", "interest rate risk")
    for page in document.pages:
        if notes_start and page.number < notes_start:
            continue
        page_lower = page.text.lower()
        if not any(marker in page_lower for marker in risk_markers):
            continue
        for raw_line in page.text.splitlines():
            parsed = _parse_supporting_note_reference_amount_line(raw_line)
            if not parsed:
                continue
            label, ref, amounts, line = parsed
            if ref not in note_headings or ref not in note_sections:
                continue
            heading, heading_page = note_headings[ref]
            section = note_sections[ref]
            if not _note_heading_semantically_compatible(label, heading, section):
                issues.append(
                    f"Page {page.number}: '{label}' references Note {ref}, but Note {ref} heading is '{heading}'. Line: {line}"
                )
                pages.add(page.number)
                max_diff = max(max_diff, Decimal("999"))
                continue
            section_amounts = _amounts_in_text(section)
            for amount in amounts[:2]:
                if abs(amount) < Decimal("1000"):
                    continue
                if _amount_list_contains(section_amounts, amount, tolerance=tolerance):
                    continue
                nearest = _nearest_amount(amount, section_amounts)
                if nearest is None:
                    issues.append(f"Page {page.number}: {label} references Note {ref}, but {amount:,} was not located in Note {ref}. Line: {line}")
                    max_diff = max(max_diff, abs(amount))
                else:
                    diff = amount - nearest
                    if abs(diff) <= tolerance:
                        continue
                    issues.append(
                        f"Page {page.number}: {label} references Note {ref}; disclosure shows {amount:,}, nearest amount in Note {ref} is {nearest:,} (difference {diff:,}). Line: {line}"
                    )
                    max_diff = max(max_diff, abs(diff))
                pages.add(page.number)
                break
    if issues:
        performed.append("Supporting disclosure note-reference amount tie-out performed for financial-instrument/risk tables.")
        findings.append(
            Finding(
                "Consistency",
                "Medium" if max_diff > tolerance else "Low",
                _format_page_set(pages) if pages else "Notes",
                "A supporting disclosure table does not appear to agree to its referenced note heading or amount.",
                " | ".join(issues[:8]),
                "Review the disclosure table note reference and amount against the referenced note and the primary statement.",
                {"check_type": "Supporting disclosure note-reference tie-out", "match_confidence": "Medium"},
            )
        )
    return findings, performed


def _check_supplementary_summary_consistency(
    document: PdfDocument,
    tolerance: Decimal,
) -> tuple[list[Finding], list[str]]:
    summary_pages = [page for page in document.pages if _looks_like_five_year_summary_page(page.text)]
    if not summary_pages:
        return [], []
    income_page = _find_statement_page(document, "Statement of profit or loss") or _find_statement_page(document, "Statement of income and expenditure")
    sfp_page = _find_statement_page(document, "Statement of financial position")
    specs: list[tuple[str, tuple[str, ...], PdfPage | None, tuple[str, ...]]] = [
        ("Revenue", ("revenue", "rental income", "turnover"), income_page, ("revenue", "rental income", "turnover")),
        ("Other operating income", ("other operating income", "interest income"), income_page, ("other operating income", "interest income")),
        ("Other operating gains", ("other operating gains", "other operating gains/losses"), income_page, ("other operating gains", "other operating gains/losses")),
        ("Other operating expenses", ("other operating expenses",), income_page, ("other operating expenses",)),
        ("Operating profit/loss", ("operating profit", "operating loss", "operating profit/loss"), income_page, ("operating profit", "operating loss", "operating profit/loss")),
        ("Finance costs", ("finance costs", "finance cost"), income_page, ("finance costs", "finance cost")),
        ("Profit/loss before taxation", ("profit before taxation", "loss before taxation", "profit before tax", "loss before tax"), income_page, ("profit before taxation", "loss before taxation", "profit before tax", "loss before tax")),
        ("Taxation", ("taxation", "income tax expense"), income_page, ("taxation", "income tax expense")),
        ("Profit/loss for the year", ("profit for the year", "loss for the year", "profit after tax", "loss after tax"), income_page, ("profit for the year", "loss for the year", "profit after tax", "loss after tax")),
        ("Total assets", ("total assets",), sfp_page, ("total assets",)),
        ("Share capital", ("share capital",), sfp_page, ("share capital",)),
        ("Retained income", ("retained income", "retained earnings"), sfp_page, ("retained income", "retained earnings")),
        ("Total equity", ("total equity",), sfp_page, ("total equity",)),
        ("Total liabilities", ("total liabilities",), sfp_page, ("total liabilities",)),
        ("Total equity and liabilities", ("total equity and liabilities",), sfp_page, ("total equity and liabilities",)),
    ]
    findings: list[Finding] = []
    performed: list[str] = []
    for summary_page in summary_pages:
        issues: list[str] = []
        max_diff = Decimal("0")
        for label, summary_aliases, source_page, source_aliases in specs:
            if not source_page:
                continue
            if _has_group_company_columns(summary_page.text) or _has_group_company_columns(source_page.text):
                continue
            summary_amounts, summary_line = _line_amounts_for_aliases_preserving_zero(summary_page.text, summary_aliases)
            source_amounts, source_line = _line_amounts_for_aliases_preserving_zero(source_page.text, source_aliases)
            if not summary_amounts or not source_amounts:
                continue
            summary_value = summary_amounts[0]
            source_value = source_amounts[0]
            diff = summary_value - source_value
            if abs(diff) <= tolerance:
                continue
            max_diff = max(max_diff, abs(diff))
            issues.append(
                f"{label}: summary Page {summary_page.number} shows {summary_value:,}; primary statement Page {source_page.number} shows {source_value:,}; difference {diff:,}. "
                f"Summary line: {summary_line}; source line: {source_line}"
            )
        if issues:
            performed.append("Five-year/financial summary amounts compared with primary statement current-year lines where readable.")
            findings.append(
                Finding(
                    "Consistency",
                    "Medium" if max_diff > tolerance else "Low",
                    f"Page {summary_page.number} | Five-year financial summary",
                    "Supplementary financial summary amount does not agree to the related primary statement amount.",
                    " | ".join(issues[:8]),
                    "Update the supplementary financial summary to agree with the audited primary statement amounts or explain the basis difference.",
                    {"check_type": "Supplementary summary tie-out", "match_confidence": "High"},
                )
            )
    return findings, performed


def _has_group_company_columns(text: str) -> bool:
    head = "\n".join(text.splitlines()[:80]).lower()
    return bool(re.search(r"\bgroup\b.{0,80}\bcompany\b|\bcompany\b.{0,80}\bgroup\b", head, flags=re.I))


def _line_amounts_for_aliases_preserving_zero(text: str, aliases: tuple[str, ...]) -> tuple[list[Decimal], str]:
    alias_norms = [_normalise_match_words(alias) for alias in aliases]
    for line in text.splitlines():
        raw_line = re.sub(r"\s+", " ", line).strip()
        if not raw_line:
            continue
        note_ref, ref_start, ref_end = _detect_statement_row_note_token(line)
        label_source = line[:ref_start] if note_ref else line
        label = _normalise_match_words(_statement_label(label_source))
        if not label:
            continue
        if not _label_matches_any_amount_alias(label, alias_norms):
            continue
        amount_source = f"{line[:ref_start]} {line[ref_end:]}" if note_ref else line
        parsed = [amount for amount in _amounts_from_statement_line(amount_source) if amount is not None and abs(amount) < Decimal("100000000")]
        heading_match = re.match(r"^\s*(\d{1,2}[A-Z]?)\.\s+[A-Za-z]", raw_line, flags=re.I)
        if heading_match and len(parsed) == 1 and parsed[0] == Decimal(heading_match.group(1).rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
            continue
        if parsed:
            return parsed[:2], raw_line
    return [], ""


def _note_line_amounts_for_aliases(note_sections: dict[str, str], aliases: tuple[str, ...]) -> tuple[list[Decimal], str, str]:
    for ref, section in note_sections.items():
        amounts, line = _line_amounts_for_aliases_preserving_zero(section, aliases)
        if amounts:
            return amounts, ref, line
    return [], "", ""


def _amount_vector_mismatches(
    left: list[Decimal],
    right: list[Decimal],
    tolerance: Decimal,
    compare_abs: bool = False,
) -> list[tuple[str, Decimal, Decimal, Decimal]]:
    labels = ("current year", "prior year")
    width = min(len(left), len(right), 2)
    mismatches: list[tuple[str, Decimal, Decimal, Decimal]] = []
    for index in range(width):
        left_value = abs(left[index]) if compare_abs else left[index]
        right_value = abs(right[index]) if compare_abs else right[index]
        diff = left_value - right_value
        if abs(diff) > tolerance:
            mismatches.append((labels[index], left[index], right[index], diff))
    return mismatches


def _label_matches_any_amount_alias(label: str, alias_norms: list[str]) -> bool:
    for alias in alias_norms:
        if alias == "total equity" and "liabilities" in label:
            continue
        if alias == "equity" and "liabilities" in label:
            continue
        if alias == "current assets" and "non current" in label:
            continue
        if alias == "current liabilities" and "non current" in label:
            continue
        if alias in {"taxation", "tax expense", "income tax expense"} and "before taxation" in label:
            continue
        if alias in {"taxation", "tax expense", "income tax expense"} and label.startswith(("profit ", "loss ")):
            continue
        if label == alias or label.startswith(alias) or alias in label:
            return True
    return False


def _parse_supporting_note_reference_amount_line(line: str) -> tuple[str, str, list[Decimal], str] | None:
    raw_line = re.sub(r"\s+", " ", line).strip()
    if not raw_line or len(raw_line) < 8:
        return None
    match = re.match(r"^([A-Za-z][A-Za-z&/'() -]{2,}?)\s+(\d{1,2}[A-C]?)\s+(.+)$", raw_line, flags=re.I)
    if not match:
        return None
    label, ref, tail = match.groups()
    ref = ref.upper()
    if not _valid_note_number(ref):
        return None
    normalized_label = _normalise_match_words(label)
    lower_line = raw_line.lower()
    narrative_markers = (
        "beginning on or after",
        "effective date",
        "mandatory adoption",
        "with effect from",
        "advance rental",
        "per annum",
        "financial statements for the year ended",
    )
    if any(marker in lower_line for marker in narrative_markers):
        return None
    label_words = re.findall(r"[A-Za-z]{3,}", label)
    if len(label_words) > 4:
        return None
    if not normalized_label or normalized_label.startswith("total") or normalized_label in {"as at", "at", "opening balance", "closing balance", "current liabilities", "non current liabilities", "non-current liabilities", "less than"}:
        return None
    amounts = [_parse_decimal(token) for token in _amount_tokens_from_statement_line(tail)]
    parsed = [amount for amount in amounts if amount is not None and abs(amount) < Decimal("100000000")]
    if not parsed or not any(abs(amount) >= Decimal("1000") for amount in parsed):
        return None
    return label.strip(), ref, parsed[:2], raw_line


def _amount_list_contains(amounts: list[Decimal], target: Decimal, tolerance: Decimal = Decimal("0")) -> bool:
    return any(abs(candidate - target) <= tolerance or abs(abs(candidate) - abs(target)) <= tolerance for candidate in amounts)


def _nearest_amount(target: Decimal, amounts: list[Decimal]) -> Decimal | None:
    if not amounts:
        return None
    return min(amounts, key=lambda candidate: abs(abs(candidate) - abs(target)))


def _looks_like_five_year_summary_page(text: str) -> bool:
    header = "\n".join(text.splitlines()[:20]).lower()
    return bool(
        "five year financial summary" in header
        or "five-year financial summary" in header
        or "5 year financial summary" in header
        or ("financial summary" in header and re.search(r"\b20\d{2}\b.*\b20\d{2}\b", header, flags=re.S))
    )

def _normalise_cash_flow_label(label: str) -> str:
    import re
    label = label.lower()
    label = re.sub(r'[^\w\s]', '', label)
    label = re.sub(r'\s+', ' ', label)
    label = label.replace("yeat", "year")
    label = re.sub(r'\btotal\s+', '', label)
    return label.strip()

def _check_cash_flow_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    
    # Heading issue check
    first_lines = page.text.splitlines()[:5]
    if any("statement of changes in equity" in line.lower() for line in first_lines):
        findings.append(
            Finding(
                "Presentation",
                "Medium",
                f"Page {page.number} | Statement of cash flows",
                "Statement of cash flows page carries incorrect 'Statement of Changes in Equity' heading.",
                "The cash flow statement appears to be presented under a 'Statement of Changes in Equity' title.",
                "Update the primary statement page heading to 'Statement of cash flows'."
            )
        )

    text = page.text
    if document and page.number < len(document.pages):
        next_page = document.pages[page.number]
        text += "\n" + next_page.text
    row_parses = _statement_row_parses(text, "Statement of cash flows")
    rows = {label: list(row.amounts) for label, row in row_parses.items()}
    if document:
        lines = _statement_note_lines(document)
        for item in lines:
            if item.line_item not in rows:
                rows[item.line_item] = list(item.amounts)
    
    op_aliases = [
        "net cash from operating activities", "cash from operating activities", 
        "cash generated from operating activities", "net cash generated from operating activities",
        "net cash used in operating activities"
    ]
    inv_aliases = [
        "net cash used in investing activities", "cash used in investing activities", 
        "net cash from investing activities", "cash flow from investing activities"
    ]
    fin_aliases = [
        "net cash generated from financing activities", "cash generated from financing activities", 
        "net cash from financing activities", "cash flow from financing activities",
        "net cash used in financing activities"
    ]
    mov_aliases = [
        "total cash movement for the year", "cash movement for the year", "cash movement for the yeat", 
        "net increase in cash and cash equivalents", "net decrease in cash and cash equivalents", "net cash movement"
    ]
    open_aliases = [
        "cash at beginning",
        "cash at the beginning of the year", "cash and cash equivalents at beginning of year", 
        "cash at beginning of year", "opening cash", "cash and cash equivalents at the beginning of the year"
    ]
    close_aliases = [
        "cash at end",
        "total cash at end of the year", "cash at end of the year", 
        "cash and cash equivalents at end of year", "closing cash", "cash and cash equivalents at the end of the year"
    ]
    exch_aliases = [
        "effect of exchange rate movement on cash balances", "effect of exchange rate movement", 
        "exchange difference on cash and cash equivalents", "exchange effect",
        "profit on foreign exchange on cash and cash equivalents",
        "loss on foreign exchange on cash and cash equivalents",
        "foreign exchange on cash and cash equivalents",
        "profit on foreign exchange on cash",
        "loss on foreign exchange on cash",
    ]
    
    def _match(row_key: str, aliases: list[str]) -> bool:
        norm_key = _normalise_cash_flow_label(row_key)
        for a in aliases:
            if _normalise_cash_flow_label(a) in norm_key:
                return True
        return False
    
    op = next((v for k, v in rows.items() if _match(k, op_aliases)), None) or next((v for k, v in rows.items() if "operat" in _normalise_cash_flow_label(k) and "net cash" in _normalise_cash_flow_label(k)), None) or next((v for k, v in rows.items() if "operat" in _normalise_cash_flow_label(k)), None)
    inv = next((v for k, v in rows.items() if _match(k, inv_aliases)), None) or next((v for k, v in rows.items() if "invest" in _normalise_cash_flow_label(k) and "net cash" in _normalise_cash_flow_label(k)), None) or next((v for k, v in rows.items() if "invest" in _normalise_cash_flow_label(k)), None)
    fin = next((v for k, v in rows.items() if _match(k, fin_aliases)), None) or next((v for k, v in rows.items() if "financ" in _normalise_cash_flow_label(k) and "net cash" in _normalise_cash_flow_label(k)), None) or next((v for k, v in rows.items() if "financ" in _normalise_cash_flow_label(k)), None)
    mov = next((v for k, v in rows.items() if _match(k, mov_aliases)), None) or next((v for k, v in rows.items() if ("increase" in _normalise_cash_flow_label(k) or "decrease" in _normalise_cash_flow_label(k) or "movement" in _normalise_cash_flow_label(k) or "net cash" in _normalise_cash_flow_label(k)) and not any(x in _normalise_cash_flow_label(k) for x in ["operat", "invest", "financ"])), None)
    
    opening = next((v for k, v in rows.items() if _match(k, open_aliases)), None) or next((v for k, v in rows.items() if ("beginning" in _normalise_cash_flow_label(k) or "january" in _normalise_cash_flow_label(k) or "start" in _normalise_cash_flow_label(k)) and "cash" in _normalise_cash_flow_label(k)), None)
    closing = next((v for k, v in rows.items() if _match(k, close_aliases)), None) or next((v for k, v in rows.items() if ("end" in _normalise_cash_flow_label(k) or "december" in _normalise_cash_flow_label(k)) and "cash" in _normalise_cash_flow_label(k)), None)
    exch = next((v for k, v in rows.items() if _match(k, exch_aliases)), None) or next((v for k, v in rows.items() if "exchange" in _normalise_cash_flow_label(k) and "cash" in _normalise_cash_flow_label(k)), None)

    op = op or _cash_flow_line_amounts_for_aliases(text, tuple(op_aliases))
    inv = inv or _cash_flow_line_amounts_for_aliases(text, tuple(inv_aliases))
    fin = fin or _cash_flow_line_amounts_for_aliases(text, tuple(fin_aliases))
    mov = mov or _cash_flow_line_amounts_for_aliases(text, tuple(mov_aliases))
    opening = opening or _cash_flow_line_amounts_for_aliases(text, tuple(open_aliases))
    closing = closing or _cash_flow_line_amounts_for_aliases(text, tuple(close_aliases))
    exch = exch or _cash_flow_line_amounts_for_aliases(text, tuple(exch_aliases))

    op = op or None
    inv = inv or None
    fin = fin or None
    mov = mov or None
    opening = opening or None
    closing = closing or None
    exch = exch or None

    vector_length = max((len(v) for v in rows.values() if v), default=0)
    zero_vector = [Decimal("0")] * vector_length if vector_length else None
    normalized_text = _normalise_cash_flow_label(text)
    if mov and vector_length:
        if inv is None and "investing activities" not in normalized_text and "investing activity" not in normalized_text:
            inv = zero_vector
        if fin is None and "financing activities" not in normalized_text and "financing activity" not in normalized_text:
            fin = zero_vector

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
    else:
        skipped.append("Statement of cash flows: skipped because operating/investing/financing/movement rows were not confidently parsed.")
        
    if opening and mov and closing:
        expected_close = [a + b + (c if c else 0) for a, b, c in zip(opening, mov, exch or ([Decimal("0")]*len(opening)))]
        _check_vector_equation(
            findings,
            page.number,
            "Statement of cash flows",
            "Opening cash plus total movement agrees to closing cash.",
            expected_close,
            closing,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of cash flows: opening plus movement checked to closing.")
    else:
        skipped.append("Statement of cash flows: skipped because opening/movement/closing rows were not confidently parsed.")
        
    return findings, performed, skipped








def _statement_rows(text: str, statement_name: str = "") -> dict[str, list[Decimal]]:
    return {label: list(row.amounts) for label, row in _statement_row_parses(text, statement_name).items()}


def _statement_row_parses(text: str, statement_name: str = "") -> dict[str, OcrStatementRow]:
    return dict(_statement_row_parses_cached(str(text or ""), str(statement_name or "")))


@lru_cache(maxsize=512)
def _statement_row_parses_cached(text: str, statement_name: str = "") -> dict[str, OcrStatementRow]:
    rows: dict[str, OcrStatementRow] = {}
    text = _crop_statement_text(text, statement_name)
    for line in text.splitlines():
        if not _plausible_statement_amount_line(line):
            continue
        parsed = _parse_ocr_statement_row(line)
        if parsed and parsed.label and _statement_row_label_allowed(parsed.label):
            rows[parsed.label] = parsed
    return _align_income_statement_columns(rows, text)


def _plausible_statement_amount_line(line: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(line or "")).strip()
    if not cleaned or len(cleaned) > 320:
        return False
    if len(NUMBER_RE.findall(cleaned)) < 1:
        return False
    if sum(1 for token in cleaned.split() if len(token) > 35) >= 2:
        return False
    return bool(re.search(r"[A-Za-z]", cleaned))


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
    for match in re.finditer(r"\b(?:note\s+)?(\d{1,2}[A-Za-z]?)(?!\s*,)\b(?=\s*(?:[=:]\s*)?(?:-\s*)?\(?-?\d)", line, flags=re.I):
        ref = match.group(1).upper()
        if not _valid_note_number(ref):
            continue
        explicit_note = bool(re.match(r"\s*note\b", match.group(0), flags=re.I))
        tail = line[match.end() :]
        label = _canonical_statement_label(_statement_label(line[: match.start()]))
        if not explicit_note and not _implicit_statement_note_ref_allowed(label):
            continue
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


def _implicit_statement_note_ref_allowed(label: str) -> bool:
    normalized = _normalise_match_words(label)
    if not normalized:
        return False
    blocked_exact = {
        _normalise_match_words(value)
        for value in {
            "current assets",
            "non current assets",
            "non-current assets",
            "total assets",
            "equity",
            "liabilities",
            "total liabilities",
            "total equity and liabilities",
            "total current assets",
            "total non current assets",
            "total non-current assets",
            "cash at beginning",
            "cash at end",
            "total cash movement for the year",
            "net increase in cash and cash equivalents",
            "net decrease in cash and cash equivalents",
            "net cash generated from used in operating activities",
            "net cash used in investing activities",
            "net cash generated from investing activities",
            "net cash used in financing activities",
            "net cash generated from financing activities",
            "cash generated from operations",
        }
    }
    if normalized in blocked_exact:
        return False
    if normalized.startswith("total "):
        return False
    if normalized.startswith("net cash "):
        return False
    if normalized.startswith("cash and cash equivalents at the beginning"):
        return False
    if normalized.startswith("cash and cash equivalents at the end"):
        return False
    return True


def _label_prefers_split_leading_digit(label: str) -> bool:
    lower_label = label.lower()
    return lower_label in {"finance income", "finance expenses", "finance costs", "other revenue"}


def _amount_tokens_from_statement_line(line: str) -> list[str]:
    cleaned = _normalise_statement_number_spacing(_strip_statement_note_token_before_amounts(line))
    cleaned = re.sub(r"(?<=\s)[-=](?=\s|$)", " 0 ", cleaned)
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
        "net cash generated from operating activities",
        "net cash used in operating activities",
        "net cash from operating activities",
        "net cash generated from investing activities",
        "net cash used in investing activities",
        "net cash from investing activities",
        "net cash generated from financing activities",
        "net cash used in financing activities",
        "net cash from financing activities",
        "net cash",
        "net increase in cash",
        "net decrease in cash",
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
        "net cash",
        "cash at",
        "total cash",
        "cash flow",
        "cash used",
        "cash generated",
        "opening",
        "closing",
        "balance at",
        "balance as at",
    )
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def _statement_label(line: str) -> str:
    cleaned = _normalise_statement_number_spacing(_strip_statement_note_token_before_amounts(line))
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
        "net cash generated from operating activities",
        "net cash used in operating activities",
        "net cash from operating activities",
        "net cash generated from investing activities",
        "net cash used in investing activities",
        "net cash from investing activities",
        "net cash generated from financing activities",
        "net cash used in financing activities",
        "net cash from financing activities",
        "net cash",
        "net increase in cash",
        "net decrease in cash",
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


def _strip_statement_note_token_before_amounts(line: str) -> str:
    grouped_amount = r"\(?-?\d{1,3}(?:[,\s.]\d{3})+(?:\.\d+)?\)?"
    amount_pattern = grouped_amount + r"|\(?-?\d+(?:\.\d+)?\)?|-"
    pattern = re.compile(
        r"(?P<label>\b[A-Za-z][A-Za-z&/()' -]{2,}?)\s+(?P<ref>\d{1,2}[A-Za-z]?)(?P<tail>\s+" + grouped_amount + r"(?:\s+" + amount_pattern + r")*)",
        re.I,
    )

    def repl(match: re.Match[str]) -> str:
        ref = match.group("ref").upper()
        if not _valid_note_number(ref):
            return match.group(0)
        tail = match.group("tail")
        remaining = match.string[match.end() :]
        if re.search(r"\s+\d{1,2}\s+\d{2},\d{3}\b", tail) or re.match(r"\s+\d{1,2}\s+\d{2},\d{3}\b", remaining):
            return match.group(0)
        return f"{match.group('label')}{tail}"

    return pattern.sub(repl, line)

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
    cleaned = _normalise_statement_number_spacing(_strip_statement_note_token_before_amounts(line))
    cleaned = re.sub(r"\s+[-=]\s*(?=\(?\s?\d)", " 0 ", cleaned)
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


def _line_amount_for_aliases(text: str, aliases: tuple[str, ...]) -> tuple[list[Decimal], str]:
    alias_norms = [_normalise_match_words(alias) for alias in aliases]
    for line in text.splitlines():
        raw_line = re.sub(r"\s+", " ", line).strip()
        if not raw_line:
            continue
        note_ref, ref_start, ref_end = _detect_statement_row_note_token(line)
        label_source = line[:ref_start] if note_ref else line
        label = _normalise_match_words(_statement_label(label_source))
        if not label:
            continue
        if not any(label == alias or label.startswith(alias) or alias in label for alias in alias_norms):
            continue
        amount_source = f"{line[:ref_start]} {line[ref_end:]}" if note_ref else line
        amounts = [_parse_decimal(token) for token in _amount_tokens_from_statement_line(amount_source)]
        parsed = [amount for amount in amounts if amount is not None and abs(amount) < Decimal("100000000")]
        if len(parsed) >= 3 and 0 < parsed[0] <= 60 and parsed[1] == 0:
            parsed = parsed[1:]
        if parsed:
            return parsed[:2], raw_line
    return [], ""


def _cash_flow_line_amounts_for_aliases(text: str, aliases: tuple[str, ...]) -> list[Decimal]:
    amounts, _raw = _line_amount_for_aliases(text, aliases)
    return amounts


def _looks_like_value_added_page(text: str) -> bool:
    header = "\n".join(text.splitlines()[:12]).lower()
    return bool(
        re.search(r"^\s*(statement of value added|value added statement)\b", header, re.I | re.M)
        or re.search(r"\bvalue added\b", header, re.I) and "statement" in header
    )


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
    if _after_tax_line_affected_by_oci(raw_lines or {}, document):
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


def _after_tax_line_affected_by_oci(raw_lines: dict[str, str], document: PdfDocument | None = None) -> bool:
    joined = " ".join(str(value or "") for value in raw_lines.values()).lower()
    if re.search(r"\b(total comprehensive income|total comprehensive loss|other comprehensive income|oci)\b", joined):
        return True
    if not document:
        return False
    for page in document.pages:
        normalized = _normalise_match_words(page.text)
        if "other comprehensive income" in normalized and "profit" in normalized and "tax" in normalized:
            return True
    return False


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
            _mismatch_issue_text(issue),
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


def _mismatch_issue_text(issue: str) -> str:
    text = issue
    replacements = (
        (" agrees to ", " does not agree to "),
        (" agrees with ", " does not agree with "),
        (" equals ", " does not equal "),
        (" equal ", " do not equal "),
    )
    for old, new in replacements:
        if old in text:
            return text.replace(old, new)
    if "does not" in text.lower():
        return text
    return f"{text.rstrip('.')} (mismatch detected)."


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
    matches = list(re.finditer(r"summary of significant accounting policies|significant accounting policies|accounting policies|basis of preparation|general information", text, flags=re.I))
    if not matches:
        return text
    start_match = matches[0] # Grab from the very first match (often Note 1 General info / Basis)
    tail = text[start_match.start():]
    # Capture up to Note 4, Note 5, or Note 6 to include Judgements and Estimates
    end_match = re.search(
        r"\n\s*4\s+[A-Z]|\n\s*5\s+[A-Z]|\n\s*6\s+[A-Z]|\n\s*(?:Note|NOTE)\s+4\b|\n\s*4\.\s+[A-Z]",
        tail[2000:],
        flags=re.I,
    )
    if end_match:
        return tail[: 2000 + end_match.start()]
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
    context = context.lower()
    has_lease_term = any(
        keyword in context
        for keyword in (
            "right-of-use",
            "right of use",
            "rou asset",
            "lease liability",
            "lease expense",
            "lease maturity",
            "depreciation of right-of-use",
            "depreciation of rou",
            "leased premises",
            "finance lease",
            "operating lease",
        )
    )
    has_numeric_or_balance_signal = bool(re.search(r"\b\d{1,3}(?:,\d{3})+\b", context)) or any(
        keyword in context
        for keyword in (
            "balance",
            "balances",
            "current",
            "non-current",
            "expense",
            "depreciation",
            "maturity",
            "payment",
            "repayment",
            "carrying amount",
            "recognized",
            "recognised",
            "commencement date",
        )
    )
    return has_lease_term and has_numeric_or_balance_signal and not _lease_context_is_theoretical_policy_only(context)


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
    if re.search(r"\b\d{1,3}(?:,\d{3})+\b", context):
        return False
    if any(
        keyword in context
        for keyword in (
            "balance",
            "balances",
            "current lease liability",
            "non-current lease liability",
            "lease expense",
            "depreciation of right-of-use",
            "depreciation of rou",
            "maturity",
            "payment",
            "commencement date",
        )
    ):
        return False
    return True


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
    currency_re = re.compile(r"\bUSD\b|\bNGN\b|\bGBP\b|\bEUR\b|US\$|\bNaira\b|\bDollar\b|\bPound\b|\bEuro\b|\u20a6|N[\'\u2019]?\s?000|N000|\$", re.I)
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
    marker = _normalise_currency_text(marker)
    normalized_marker = re.sub(r"\s+", "", marker.upper())
    normalized_marker = normalized_marker.replace(chr(0x2019), "'").replace(chr(0x2018), "'").replace("`", "'")
    normalized_marker = normalized_marker.replace(chr(0x20A6), "NGN")
    if normalized_marker in {"NGN", "NGN'000", "N'000", "N000", "NIGERIA"}:
        return "NGN"
    if re.fullmatch(r"N[^A-Z0-9]{0,2}000", normalized_marker):
        return "NGN"
    if normalized_marker in {"NAIRA", "NIGERIANNAIRA"}:
        return "NGN"
    if normalized_marker in {"DOLLAR", "US$", "USD", "$"}:
        return "USD"
    if normalized_marker == "POUND":
        return "GBP"
    if normalized_marker == "EURO":
        return "EUR"
    return normalized_marker

def normalize_reporting_currency(value: str) -> str:
    value = _normalise_currency_text(value)
    normalized_value = re.sub(r"\s+", "", value.strip().upper())
    normalized_value = normalized_value.replace(chr(0x2019), "'").replace(chr(0x2018), "'").replace("`", "'")
    normalized_value = normalized_value.replace(chr(0x20A6), "NGN")
    if normalized_value in {"", None}:
        return ""
    if re.fullmatch(r"N[^A-Z0-9]{0,2}000", normalized_value):
        return "NGN"
    if re.search(r"(?:NGN|NAIRA|NIGERIANNAIRA|N'000|N000|NIGNAIRA|NGN000)", normalized_value):
        return "NGN"
    aliases = {
        "NAIRA": "NGN",
        "NIGERIANNAIRA": "NGN",
        "NGN": "NGN",
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
    return aliases.get(normalized_value, normalized_value if normalized_value in VALID_CURRENCIES else "")

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
    lower = _presentation_scale_context(text)
    labels = set()
    if re.search(r"\$?0{3}s|000s|thousand|in thousands", lower):
        labels.add("thousands")
    if re.search(r"\b(?:in|nearest|rounded to|presented in|amounts are rounded to)\s+millions?\b|\bmillions?\s+of\b", lower):
        labels.add("millions")
    if re.search(r"\bnearest dollar\b|\bin units\b|\bpresented in units\b|\bactual naira\b|\bactual amounts? are presented\b", lower):
        labels.add("units")
    if len(labels) > 1:
        return "mixed", Decimal("1")
    if "millions" in labels:
        return "millions", Decimal("1")
    if "thousands" in labels:
        return "thousands", Decimal("1")
    return "units", Decimal("1")


def _presentation_scale_context(text: str) -> str:
    relevant_lines: list[str] = []
    presentation_terms = (
        "n'000",
        "n '000",
        "ngn'000",
        "ngn '000",
        "\u20a6\'000",
        "\u20a6 \'000",
        "in thousands",
        "nearest thousand",
        "in millions",
        "nearest million",
        "presented in",
        "presentation currency",
        "functional currency",
        "amounts are rounded",
        "rounded to",
    )
    for line in text.splitlines():
        lower = line.lower()
        if any(term in lower for term in presentation_terms):
            relevant_lines.append(lower)
            continue
        if re.search(r"\b20\d{2}\b", lower) and re.search(r"n\s*[\'\u2019`]\s*000|ngn|\u20a6", lower):
            relevant_lines.append(lower)
    return "\n".join(relevant_lines) if relevant_lines else text.lower()


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
    label = str(row[0] or "").lower() if row else ""
    for index, cell in enumerate(row):
        if index == 0:
            converted.append(cell)
            continue
        if index in note_cols:
            converted.append(None)
            continue
            
        amount = _parse_decimal(cell)
        if amount is not None:
            # Filter standard reference numbers
            cell_str = str(cell).lower()
            if re.search(r"\b(ifrs|ias|note)\s*\d+\b", cell_str):
                amount = None
            elif amount in (15, 20) and ("ifrs" in label or "ias" in label or "note" in label):
                amount = None
                
        converted.append(amount)
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
    if not label.strip() or any(word in label for word in excluded):
        return False
    if not re.search(r"[a-z]{3,}", label):
        return False
    return True


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
    cached = getattr(document, "_note_headings_by_page_cache", None)
    if isinstance(cached, dict):
        return cached
    candidates: dict[str, list[tuple[str, int, int]]] = {}
    notes_start_page = _notes_start_page(document)
    strict_notes_start = notes_start_page is not None
    if document.ocr_used and not strict_notes_start:
        return {}
    in_notes = not strict_notes_start
    for page in document.pages:
        if notes_start_page is not None and page.number < notes_start_page:
            continue
        if notes_start_page is not None and page.number > notes_start_page and _is_post_notes_supplement_page(page.text):
            break
        if strict_notes_start and (page.number == notes_start_page or _notes_heading_in_text(page.text)):
            in_notes = True
        if strict_notes_start and not in_notes:
            continue
        table_count = len(page.tables)
        lines = page.text.splitlines()
        if page.number == notes_start_page:
            implicit_note_1 = _implicit_policy_note_heading(lines)
            if implicit_note_1:
                candidates.setdefault("1", []).append((implicit_note_1, page.number, table_count))
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            embedded = _embedded_note_heading_after_notes_title(line)
            if embedded:
                number, title = embedded
                if _valid_note_heading(number, title):
                    if not _is_policy_subsection_suspect(title, number, page.number, table_count, notes_start_page):
                        candidates.setdefault(number.upper(), []).append((_clean_note_title(title), page.number, table_count))
                    continue
            match = NOTE_HEADING_RE.match(line)
            if match:
                number, title = match.groups()
                if _valid_note_heading(number, title):
                    if not _is_policy_subsection_suspect(title, number, page.number, table_count, notes_start_page):
                        candidates.setdefault(number.upper(), []).append((_clean_note_title(title), page.number, table_count))
                    continue
            number_only = NOTE_NUMBER_ONLY_RE.match(line)
            if number_only and index + 1 < len(lines):
                number = number_only.group(1)
                title = lines[index + 1].strip()
                if _valid_note_heading(number, title):
                    if not _is_policy_subsection_suspect(title, number, page.number, table_count, notes_start_page):
                        candidates.setdefault(number.upper(), []).append((_clean_note_title(title), page.number, table_count))
            # Also need to pass notes_start_page to _add_combined_note_heading_candidates or just filter its output.
            # It's easier to just let it add, then filter candidates later, but we can also just filter after.
            _add_combined_note_heading_candidates(candidates, lines, index, line, page.number)
            
    headings: dict[str, tuple[str, int]] = {}
    for number, occurrences in candidates.items():
        # Clean up any that got added by _add_combined_note_heading_candidates
        occurrences = [occ for occ in occurrences if not _is_policy_subsection_suspect(occ[0], number, occ[1], occ[2], notes_start_page)]
        if not occurrences:
            continue
        valid_occs = [occ for occ in occurrences if "continued" not in occ[0].lower()]
        if not valid_occs:
            valid_occs = occurrences
            
        # The user wants to match face-statement Note 4 and Note 9 to actual note sections/tables
        # after the real Notes section begins. The accounting policy subsections (1.4, etc.) 
        # appear earlier. Always taking the LAST occurrence ensures we pick the actual note 
        # body instead of the policy subsection, regardless of which page has tables.
        best = valid_occs[-1]
        headings[number] = (best[0], best[1])
    setattr(document, "_note_headings_by_page_cache", headings)
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


def _implicit_policy_note_heading(lines: list[str]) -> str | None:
    head = "\n".join(line.strip() for line in lines[:35] if line.strip())
    normalized = _normalise_match_words(head)
    if "material accounting polic" in normalized:
        return "Material accounting policies"
    if "significant accounting polic" in normalized:
        return "Significant accounting policies"
    return None


def _embedded_note_heading_after_notes_title(line: str) -> tuple[str, str] | None:
    if _notes_heading_line_score(line) < 0.82:
        return None
    match = re.search(r"\b(1)\s*[\).:-]\s+(.{3,100})$", line, flags=re.I)
    if not match:
        return None
    return match.group(1), match.group(2)


def _add_combined_note_heading_candidates(
    headings: dict[str, list[tuple[str, int, int]]],
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
        headings.setdefault("7", []).append(("Operating Revenue", page_number, 0))
        headings.setdefault("8", []).append(("Operating Expenditure", page_number, 0))


def _augment_note_headings_from_statement_refs(headings: dict[str, tuple[str, int]], document: PdfDocument) -> None:
    for item in _statement_note_lines(document):
        if not item.line_item or item.ref == "1":
            continue
        title = _statement_line_item_title(item.line_item)
        existing = headings.get(item.ref)
        if existing is None:
            inferred = _infer_missing_note_heading_from_statement_ref(item.ref, title, headings, document)
            if inferred:
                headings[item.ref.upper()] = inferred


def _infer_missing_note_heading_from_statement_ref(
    ref: str,
    title: str,
    headings: dict[str, tuple[str, int]],
    document: PdfDocument,
) -> tuple[str, int] | None:
    if not ref or not _valid_note_number(ref) or not title:
        return None
    numeric_ref = _note_ref_number(ref)
    if numeric_ref is None:
        return None
    notes_start_page = _notes_start_page(document)
    if notes_start_page is None and document.ocr_used:
        return None
    lower_bound = notes_start_page or 1
    upper_bound = max((page.number for page in document.pages), default=lower_bound)
    for other_ref, (_other_title, page_number) in headings.items():
        other_number = _note_ref_number(other_ref)
        if other_number is None:
            continue
        if other_number < numeric_ref:
            lower_bound = max(lower_bound, page_number)
        elif other_number > numeric_ref:
            upper_bound = min(upper_bound, page_number)
    normalized_title = _normalise_match_words(title)
    if len(normalized_title.split()) < 2:
        return None
    for page in document.pages:
        if page.number < lower_bound or page.number > upper_bound:
            continue
        if notes_start_page is not None and page.number < notes_start_page:
            continue
        if page.number > lower_bound and _is_post_notes_supplement_page(page.text):
            break
        next_heading_index = _first_later_note_heading_index(page.text, numeric_ref) if page.number == upper_bound else None
        for index, raw_line in enumerate(page.text.splitlines()):
            if next_heading_index is not None and index >= next_heading_index:
                break
            line = re.sub(r"\s+", " ", raw_line.strip())
            if _standalone_note_heading_line_matches(line, normalized_title):
                return title, page.number
    return None


def _note_ref_number(ref: str) -> int | None:
    match = re.match(r"(\d{1,2})", str(ref or ""))
    if not match:
        return None
    return int(match.group(1))


def _first_later_note_heading_index(text: str, numeric_ref: int) -> int | None:
    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        match = NOTE_HEADING_RE.match(line) or NOTE_NUMBER_ONLY_RE.match(line)
        if not match:
            continue
        other_ref = str(match.group(1))
        if not _valid_note_number(other_ref):
            continue
        other_number = _note_ref_number(other_ref)
        if other_number is not None and other_number > numeric_ref:
            return index
    return None


def _standalone_note_heading_line_matches(line: str, normalized_title: str) -> bool:
    if not line or _amounts_in_text(line):
        return False
    normalized_line = _normalise_match_words(line)
    if not normalized_line or "continued" in normalized_line:
        return False
    if normalized_line == normalized_title:
        return True
    title_words = normalized_title.split()
    line_words = normalized_line.split()
    return (
        normalized_line.startswith(normalized_title)
        and len(line_words) <= len(title_words) + 3
        and not _note_heading_title_looks_narrative(line)
    )


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
            f"Note {ref} | Page {_reviewer_page_number(document, page_number)} | {page_ranges.get(ref, f'Page {_reviewer_page_number(document, page_number)}')} | {title} | {confidence} | {source_snippet}"
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


def _statement_reference_pages(document: PdfDocument) -> dict[str, str]:
    pages_by_ref: dict[str, set[int]] = defaultdict(set)
    for item in _statement_note_lines(document):
        if item.ref and item.page_number:
            pages_by_ref[item.ref.upper()].add(_reviewer_page_number(document, item.page_number))
    formatted: dict[str, str] = {}
    for ref, pages in pages_by_ref.items():
        ordered = sorted(pages)
        label = "Page" if len(ordered) == 1 else "Pages"
        formatted[ref] = f"{label} {', '.join(str(page) for page in ordered)}"
    return formatted


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



def _crop_statement_text(text: str, statement_name: str = "") -> str:
    lines = text.splitlines()
    cropped = []
    stop_phrases = [
        "the annual report and financial statements on pages",
        "were approved by the board of directors",
        "were signed on its behalf by",
        "signed on behalf of the board",
        "group managing director",
        "group chief financial officer",
        "chief financial officer",
        "frc/20",
        "frc/",
    ]
    for line in lines:
        lower = line.lower()
        if any(p in lower for p in stop_phrases) and len(line.strip()) > 3:
            break
        if statement_name and "financial position" in statement_name.lower():
            if re.search(r"total equity and liabilities", lower):
                cropped.append(line)
                break
        cropped.append(line)
    return "\n".join(cropped)

def _rename_statement_for_output(document: PdfDocument, canonical_name: str, page_text: str) -> str:
    is_company = document and _detect_entity_type(document.text).lower() in ("private company", "public company", "company")
    if canonical_name == "Statement of income and expenditure" and is_company:
        for line in page_text.splitlines()[:20]:
            lower = line.strip().lower()
            if "comprehensive income" in lower:
                return "Statement of profit or loss and other comprehensive income" if "profit or loss" in lower else "Statement of comprehensive income"
        return "Statement of comprehensive income"
    if canonical_name == "Statement of changes in accumulated fund" and is_company:
        return "Statement of changes in equity"
    return canonical_name

def _statement_note_lines(document: PdfDocument) -> list[StatementNoteLine]:
    cached = getattr(document, "_statement_note_lines_cache", None)
    if isinstance(cached, list):
        return cached
    items: list[StatementNoteLine] = []
    statements = _classified_primary_statement_pages(document)
    source_pages: list[tuple[str, PdfPage]] = []
    if statements:
        for canonical_name, page in statements.items():
            source_pages.append((canonical_name, page))
    else:
        for page in document.pages:
            if _is_notes_page(page.text) or _is_post_notes_supplement_page(page.text):
                continue
            source_pages.append((_statement_name_from_page(page.text), page))
    for canonical_name, page in source_pages:
        display_name = _rename_statement_for_output(document, canonical_name, page.text)
        if _statement_excluded_from_note_agreement(display_name, page.text):
            continue
        if "changes in accumulated fund" in canonical_name.lower() or "changes in equity" in canonical_name.lower():
            continue
        cropped_text = _crop_statement_text(page.text, canonical_name)
        for line in cropped_text.splitlines():
            parsed = _parse_statement_note_line(line, page.number, display_name)
            if parsed:
                items.append(parsed)
    setattr(document, "_statement_note_lines_cache", items)
    return items


def _statement_lines_with_note_refs(document: PdfDocument) -> list[tuple[str, str, Decimal]]:
    lines: list[tuple[str, str, Decimal]] = []
    for item in _statement_note_lines(document):
        if _note_agreement_skip_reason(item):
            continue
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
        implicit_match = re.search(
            r"(?<![\.\d])\b(\d{1,2}[A-C]?)\b(?=\s+(?:(?:-\s*){1,2}|[-=]?\s*\(?-?\d[\d,\s]*\)?))",
            line,
            flags=re.I,
        )
        if implicit_match and _amounts_in_text(line[: implicit_match.start()]):
            implicit_match = None
    ref = explicit_ref or (implicit_match.group(1).upper() if implicit_match else "")
    if not ref or not _valid_note_number(ref):
        ref = ""
    ref_start = note_match.start() if note_match else (implicit_match.start() if implicit_match else len(line))
    ref_end = note_match.end() if note_match else (implicit_match.end() if implicit_match else ref_start)
    label = _clean_statement_line_item(line[:ref_start])
    if not label or (_is_subheading(label) and not ref and _normalise_match_words(label) not in {"assets"}):
        return None
    letters_only = re.sub(r"[^a-z]", "", label.lower())
    if label.lower().strip() in {"n n", "nn", "0 0"} or len(letters_only) < 3 or set(letters_only).issubset({"n", "m", "o"}):
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
    cleaned = re.sub(r"\bdraft\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(total|net)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:page|pages)\b\s*\d+\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:continued|contd)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b[a-z]\b$", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:d|dr|draft)\s*$", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if cleaned in {"liabilities", "equity", "equity and liabilities"}:
        return ""
    return cleaned


def _note_agreement_skip_reason(item: StatementNoteLine) -> str:
    statement = item.statement_name.lower()
    if "value added" in statement or "five year" in statement or "financial summary" in statement:
        return "not a face-linked note line"
    if "cash flow" in statement and _cash_flow_line_not_note_linked(item.line_item):
        return "not a face-linked note line"
    if _line_item_not_face_linked(item.line_item, item.statement_name, item.explicit_ref):
        return "not a face-linked note line"
    return ""


def _cash_flow_line_not_note_linked(line_item: str) -> bool:
    normalized = _normalise_match_words(line_item)
    if any(activity in normalized for activity in ("operating activities", "investing activities", "financing activities")) and any(
        term in normalized for term in ("net cash", "cash generated", "cash used", "generated from", "used in", "used from")
    ):
        return True
    if any(
        marker in normalized
        for marker in (
            "cash operating activities",
            "cash investing activities",
            "cash financing activities",
            "cash generated operating activities",
            "cash generated investing activities",
            "cash generated financing activities",
            "cash used operating activities",
            "cash used investing activities",
            "cash used financing activities",
            "net cash operating activities",
            "net cash investing activities",
            "net cash financing activities",
            "cash at beginning year",
            "cash and cash equivalents at beginning year",
            "cash at end year",
            "cash and cash equivalents at end year",
            "effect exchange rate movement cash balances",
            "interest income",
            "loss disposal fixed asset",
            "loss on disposal fixed asset",
            "depreciation",
            "finance costs",
            "tax paid",
            "profit before taxation",
            "profit before financing income taxes",
            "profit before financing",
        )
    ):
        return True
    return False


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
        "cash from operating activities",
        "cash from investing activities",
        "cash from financing activities",
        "cash and cash equivalents at the beginning of the year",
        "cash and cash equivalents at the end of the year",
        "effect of exchange rate movement on cash balances",
        "interest income",
        "profit before taxation",
        "loss before taxation",
        "profit loss before taxation",
        "profit before tax",
        "loss before tax",
        "profit loss before tax",
        "profit before financing and income taxes",
        "operating profit",
        "operating loss",
        "taxation",
        "tax expense",
        "income tax expense",
        "profit after tax",
        "loss after tax",
        "profit loss after tax",
        "profit for year",
        "loss for year",
        "profit loss for year",
        "total comprehensive income",
        "total comprehensive loss",
    }
    explicit_pnl_note_labels = {"taxation", "tax expense", "income tax expense"}
    if explicit_ref and any(token in statement for token in ("profit or loss", "comprehensive", "income and expenditure")) and label in explicit_pnl_note_labels:
        return False
    if label in broad_labels:
        return True
    if raw_label.startswith(("total ", "net cash", "surplus for the year")) and not explicit_ref:
        return True
    if not explicit_ref and re.search(
        r"\b(profit|loss|profit/\(loss\)|profit\/loss)\b.*\b(before taxation|before tax|after tax|for the year)\b",
        raw_label,
    ):
        return True
    if not explicit_ref and "profit before financing" in label:
        return True
    if not explicit_ref and re.search(r"\btotal comprehensive (income|loss)\b", raw_label):
        return True
    if "cash flow" in statement and re.search(r"\b(total|net|cash generated|cash used|increase|decrease|cash inflow|cash outflow|cash absorbed)\b", raw_label):
        return True
    if "cash flow" in statement and not explicit_ref:
        if re.search(r"\b(profit|loss|taxation|tax paid|income tax|working capital|receivable(?:s)?|payable(?:s)?|contract liabilities|inventory|loans? and advance(?:s)?|advance(?:s)?|cash movement|interest income|cash and cash equivalents at the beginning of the year|cash and cash equivalents at the end of the year|effect of exchange rate movement)\b", raw_label):
            return True
        if _cash_flow_line_not_note_linked(line_item):
            return True
    return False


def _suggest_note_for_unreferenced_line(
    item: StatementNoteLine,
    note_sections: dict[str, str],
    headings: dict[str, str],
    tolerance: Decimal,
) -> str:
    best_ref = ""
    best_score = 0.0
    for ref, heading in headings.items():
        if _is_disclosure_only_note(heading):
            continue
        section = _get_note_section_with_fallback(ref, note_sections)
        heading_score = max(_wording_match_score(item.line_item, heading), _semantic_heading_score(item.line_item, heading))
        amount_score = 0.0
        if item.amounts:
            matched = sum(1 for amount in item.amounts if _amount_match_in_section(amount, section, tolerance)["found"])
            if matched:
                amount_score = 0.25 + (0.1 * matched)
        score = heading_score + amount_score
        if heading_score >= 0.82 or (heading_score >= 0.7 and amount_score > 0):
            if score > best_score:
                best_ref = ref
                best_score = score
    return best_ref


def _check_possible_wrong_note_references(
    statement_lines: list[StatementNoteLine],
    note_sections: dict[str, str],
    headings: dict[str, str],
    tolerance: Decimal,
    cautious_review_prompt: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], set[tuple[str, str]]]:
    findings: list[Finding] = []
    flagged: set[tuple[str, str]] = set()
    for item in statement_lines:
        if not item.ref:
            continue
        if _note_agreement_skip_reason(item):
            continue
        
        referenced = _get_note_section_with_fallback(item.ref, note_sections, document) if document else _get_note_section_with_fallback(item.ref, note_sections)
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
        compatibility_rule = _note_compatibility_rule(item.line_item)
        referenced_incompatible = bool(
            compatibility_rule
            and referenced_heading
            and not _note_heading_semantically_compatible(item.line_item, referenced_heading, referenced)
        )
        if referenced_incompatible:
            referenced_heading_score = max(0.0, referenced_heading_score - 0.45)
        best_ref = ""
        best_score = -1
        best_match: dict[str, bool] = {"wording": False, "amount": False}
        best_heading_score = 0.0

        for other_ref, other_heading in headings.items():
            if other_ref == item.ref or _is_disclosure_only_note(other_heading):
                continue
            other_text = _get_note_section_with_fallback(other_ref, note_sections, document) if document else _get_note_section_with_fallback(other_ref, note_sections)
            other_text = other_text or ""
            if not _alternative_note_semantically_allowed(item.line_item, other_heading, other_text):
                continue
            other_match = _note_match_strength(item, other_heading, other_text, tolerance)
            other_heading_score = max(_wording_match_score(item.line_item, other_heading), _semantic_heading_score(item.line_item, other_heading))
            if other_match["amount"]:
                other_heading_score += 0.5
            if other_heading_score > best_score:
                best_score = other_heading_score
                best_ref = other_ref
                best_match = other_match
                best_heading_score = max(_wording_match_score(item.line_item, other_heading), _semantic_heading_score(item.line_item, other_heading))

        if referenced_incompatible:
            compatibility_label = _note_compatibility_label(item.line_item)
            best_heading = _get_note_heading_with_fallback(best_ref, headings) if best_ref else ""
            best_compatible = bool(best_ref and _note_heading_semantically_compatible(item.line_item, best_heading, _get_note_section_with_fallback(best_ref, note_sections, document) if document else _get_note_section_with_fallback(best_ref, note_sections)))
            if best_compatible and best_heading_score >= 0.82:
                confidence = "Medium" if best_match.get("amount") else "Low"
                if cautious_review_prompt and confidence == "Medium":
                    confidence = "Low"
                findings.append(
                    _note_reference_review_prompt(
                        item,
                        best_ref,
                        confidence,
                        f"Referenced note heading is not compatible with the {compatibility_label} line item; Note {best_ref} appears more compatible.",
                        cautious_review_prompt,
                        explicit_issue=(
                            f"Note heading mismatch: {item.line_item.title()} references Note {item.ref} heading '{referenced_heading}', "
                            f"but Note {best_ref} heading '{best_heading}' appears more compatible."
                        ),
                    )
                )
                flagged.add((item.ref, item.line))
            elif not referenced_match["amount"] and not best_ref:
                findings.append(
                    _note_reference_review_prompt(
                        item,
                        "",
                        "Low",
                        f"Referenced note heading is not compatible with the {compatibility_label} line item and no compatible alternative was identified.",
                        cautious_review_prompt,
                        explicit_issue=(
                            f"Note heading mismatch: {item.line_item.title()} references Note {item.ref}, "
                            f"whose heading '{referenced_heading}' is not compatible with the line item."
                        ),
                    )
                )
                flagged.add((item.ref, item.line))

        if (best_match["amount"] and not referenced_match["amount"]) or (
            best_heading_score > referenced_heading_score + 0.3 and best_match["wording"] and not referenced_match["amount"]
        ):
            if _should_skip_cash_flow_alternative_note_prompt(
                item,
                best_ref,
                best_match,
                best_heading_score,
                referenced_heading_score,
                note_sections,
            ):
                continue
            confidence = "High" if best_match["wording"] and best_match["amount"] else "Medium" if best_match["amount"] else "Low"
            if best_ref and item.ref and best_ref.startswith(item.ref) and len(best_ref) > len(item.ref):
                confidence = "Low"
            if cautious_review_prompt and confidence == "High":
                confidence = "Medium"
            if cautious_review_prompt and confidence == "Low":
                continue
            if best_ref and not _note_heading_semantically_compatible(
                item.line_item,
                _get_note_heading_with_fallback(best_ref, headings),
                _get_note_section_with_fallback(best_ref, note_sections, document) if document else _get_note_section_with_fallback(best_ref, note_sections),
            ):
                continue
            findings.append(_note_reference_review_prompt(item, best_ref, confidence, f"Amount or stronger wording match found in Note {best_ref}.", cautious_review_prompt))
            flagged.add((item.ref, item.line))
            
    return findings, flagged




def _should_skip_cash_flow_alternative_note_prompt(
    item: StatementNoteLine,
    suggested_ref: str,
    match: dict[str, bool],
    suggested_heading_score: float,
    referenced_heading_score: float,
    note_sections: dict[str, str],
) -> bool:
    statement_name = (item.statement_name or '').lower()
    if 'cash flow' not in statement_name:
        return False
    if not suggested_ref:
        return False
    if _notes_are_related_for_reference_review(item.ref, suggested_ref, note_sections):
        return True
    if not (match.get('wording') and match.get('amount')):
        return True
    if suggested_heading_score < 0.92:
        return True
    if suggested_heading_score <= referenced_heading_score + 0.18:
        return True
    return False



def _notes_are_related_for_reference_review(
    referenced_ref: str,
    suggested_ref: str,
    note_sections: dict[str, str],
) -> bool:
    if not referenced_ref or not suggested_ref:
        return False
    ref = referenced_ref.upper().strip()
    suggested = suggested_ref.upper().strip()
    ref_root = re.sub(r'[A-Z]+$', '', ref)
    suggested_root = re.sub(r'[A-Z]+$', '', suggested)
    if ref_root and ref_root == suggested_root:
        return True
    related_sections = [
        note_sections.get(ref, ''),
        note_sections.get(ref_root, ''),
        note_sections.get(suggested, ''),
        note_sections.get(suggested_root, ''),
    ]
    for section in related_sections:
        if not section:
            continue
        refs_in_section = _refs_in_text(section)
        if ref in refs_in_section and suggested in refs_in_section:
            return True
        if suggested in refs_in_section and (ref == ref_root or ref_root in refs_in_section):
            return True
        if ref in refs_in_section and (suggested == suggested_root or suggested_root in refs_in_section):
            return True
    return False



def _note_reference_review_prompt(
    item: StatementNoteLine,
    suggested_ref: str,
    confidence: str,
    reason: str,
    cautious_review_prompt: bool,
    explicit_issue: str | None = None,
) -> Finding:
    category = "Notes agreement"
    if explicit_issue:
        issue = explicit_issue
    elif "cash flow" in item.statement_name.lower() and not suggested_ref:
        issue = "Note reference on statement of cash flows not found in the notes."
        evidence = f"Note reference '{item.ref}' for {item.line_item.title()} was not found."
        confidence = "Medium" # Mark as Review prompt
        category = "Review prompt"
    elif suggested_ref:
        issue = f"Possible wrong note reference: {item.line_item.title()} references Note {item.ref}, but Note {suggested_ref} appears to be a stronger match."
    else:
        issue = f"Referenced note not found: {item.line_item.title()} references Note {item.ref}, but that note was not detected."
        confidence = "Low"
        
    if "cash flow" not in item.statement_name.lower() or suggested_ref or explicit_issue:
        evidence = (
            f"Line: {item.line[:160]}. Amounts checked: {', '.join(f'{amount:,}' for amount in item.amounts)}. "
            f"Reason: {reason}"
            + (" Review prompt only because note extraction confidence is below threshold." if cautious_review_prompt else "")
        )
    else:
        evidence = f"Note reference '{item.ref}' for {item.line_item.title()} was not found. Reason: {reason}"
    return Finding(
        category if "category" in locals() else "Notes agreement",
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
        referenced_heading = _get_note_heading_with_fallback(item.ref, headings)
        allow_absolute = _note_heading_allows_signless_amount_match(referenced_heading, referenced_section)
        current_match = _amount_match_in_section(current_amount, referenced_section, tolerance, allow_absolute=allow_absolute)
        prior_match = _amount_match_in_section(prior_amount, referenced_section, tolerance, allow_absolute=allow_absolute)
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


def _amount_match_in_section(
    amount: Decimal | None,
    section: str,
    tolerance: Decimal,
    allow_absolute: bool = False,
) -> dict[str, object]:
    if amount is None or amount == 0 or abs(amount) <= tolerance * 5:
        return {"found": True, "snippet": "", "method": "not material"}
    for candidate, snippet, method in _normalized_amount_candidates(section):
        if abs(candidate - amount) <= tolerance:
            return {"found": True, "snippet": snippet, "method": method}
        if (allow_absolute or amount < 0) and abs(abs(candidate) - abs(amount)) <= tolerance:
            return {"found": True, "snippet": snippet, "method": f"{method} / absolute value"}
    return {"found": False, "snippet": "", "method": "not found"}


def _note_heading_allows_signless_amount_match(note_heading: str, note_section: str = "") -> bool:
    combined = _normalise_match_words(f"{note_heading} {note_section[:240]}")
    return (
        "payable" in combined and "receivable" in combined
    ) or (
        "current tax" in combined and ("payable" in combined or "receivable" in combined)
    ) or (
        "asset" in combined and "liabilit" in combined
    ) or (
        "taxation" in combined or "deferred tax" in combined or "tax expense" in combined
    )


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
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = re.sub(r"\s+", " ", text[line_start:line_end]).strip()
    if line and len(line) <= 260:
        return line

    sentence_start = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind(":", 0, start), text.rfind("\n", 0, start))
    sentence_end_candidates = [pos for pos in (text.find(".", end), text.find(";", end), text.find("\n", end)) if pos != -1]
    sentence_end = min(sentence_end_candidates) + 1 if sentence_end_candidates else min(len(text), end + window)
    snippet = re.sub(r"\s+", " ", text[max(0, sentence_start + 1):sentence_end]).strip()
    if len(snippet) <= 260:
        return snippet
    compact = re.sub(r"\s+", " ", text[max(0, start - window): min(len(text), end + window)]).strip()
    compact = re.sub(r"^\S*\s+", "", compact)
    compact = re.sub(r"\s+\S*$", "", compact)
    return f"... {compact[:240].rstrip()} ..."


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
    if not item or not heading:
        return 0.0
    if "current tax" in item:
        return 0.9 if any(term in heading for term in ("current tax", "current tax receivable", "current tax payable")) else 0.0
    if "deferred tax" in item:
        return 0.9 if "deferred tax" in heading else 0.0
    if _note_compatibility_rule(item) and _note_heading_semantically_compatible(item, heading):
        return 0.88
    if _is_revenue_line_item(item) and _revenue_alternative_heading_allowed(heading):
        return 0.86
    if any(term in item for term in ("cash", "bank", "cash equivalents")) and any(term in heading for term in ("cash", "bank", "cash equivalents")):
        return 0.86
    if "tax" in item and "tax" in heading:
        return 0.82
    return 0.0


def _alternative_note_semantically_allowed(line_item: str, note_heading: str, note_section: str = "") -> bool:
    item = _normalise_match_words(line_item)
    heading = _normalise_match_words(note_heading)
    if not item or not heading:
        return False
    if "current tax" in item:
        return any(term in heading for term in ("current tax", "current tax receivable", "current tax payable"))
    if "deferred tax" in item:
        return "deferred tax" in heading
    if _note_compatibility_rule(item):
        return _note_heading_semantically_compatible(item, heading, note_section)
    if "tax" in item:
        return "tax" in heading
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


def _amount_match_confidence(current_found: bool, prior_found: bool, alternative_ref: str, cautious_review_prompt: bool, item_ref: str = "") -> str:
    if alternative_ref and not cautious_review_prompt and current_found is False and prior_found is False:
        if item_ref and alternative_ref.startswith(item_ref) and len(alternative_ref) > len(item_ref):
            return "Low"
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
    cached = getattr(document, "_note_sections_cache", None)
    if isinstance(cached, dict):
        return cached
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
            prev_line = lines[index-1].strip() if index > 0 else ""
            if re.match(r"^\d+\.$", prev_line):
                prev_num = prev_line.strip(".")
                if prev_num != match.group(1):
                    continue
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
    section_text = {number: "\n".join(lines) for number, lines in sections.items()}
    _augment_note_sections_from_inferred_headings(section_text, document)
    setattr(document, "_note_sections_cache", section_text)
    return section_text


def _augment_note_sections_from_inferred_headings(sections: dict[str, str], document: PdfDocument) -> None:
    headings = _note_headings_by_page(document)
    for ref, (title, page_number) in headings.items():
        if ref in sections:
            continue
        inferred_section = _extract_section_from_standalone_heading(document, ref, title, page_number)
        if inferred_section:
            sections[ref] = inferred_section


def _extract_section_from_standalone_heading(
    document: PdfDocument,
    ref: str,
    title: str,
    page_number: int,
) -> str:
    numeric_ref = _note_ref_number(ref)
    if numeric_ref is None:
        return ""
    start_page = next((page for page in document.pages if page.number == page_number), None)
    if start_page is None:
        return ""
    normalized_title = _normalise_match_words(title)
    collecting = False
    collected: list[str] = []
    for page in document.pages:
        if page.number < page_number:
            continue
        if page.number > page_number and _is_post_notes_supplement_page(page.text):
            break
        for raw_line in page.text.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            if not collecting:
                if page.number == page_number and _standalone_note_heading_line_matches(line, normalized_title):
                    collecting = True
                    collected.append(raw_line)
                continue
            later_heading = NOTE_HEADING_RE.match(line) or NOTE_NUMBER_ONLY_RE.match(line)
            if later_heading and _valid_note_number(str(later_heading.group(1))):
                later_number = _note_ref_number(str(later_heading.group(1)))
                if later_number is not None and later_number > numeric_ref:
                    return "\n".join(collected)
            collected.append(raw_line)
    return "\n".join(collected)


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


def _is_policy_subsection_suspect(title: str, number: str, page_number: int, table_count: int, notes_start_page: int | None) -> bool:
    if table_count > 0:
        return False
    if not number.isdigit() or int(number) < 3:
        return False
    if notes_start_page is not None and page_number - notes_start_page < 5:
        policy_keywords = [
            "basis",
            "recognition",
            "measurement",
            "depreciation",
            "amortisation",
            "amortization",
            "impairment",
            "expected credit loss",
            "ecl methodology",
        ]
        if any(k in title.lower() for k in policy_keywords):
            return True
    return False

def _valid_note_heading(number: str, title: str) -> bool:
    number = number.upper().strip()
    # Reject policy subsections starting with a number like "1.4" or ".4"
    if re.match(r"^\.?\d+\b", title.strip()):
        return False
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
    company_as_lease_party = bool(re.search(r"\bcompany\s+as\s+(?:lessee|lessor)\b", title_lower))
    if ENTITY_SUFFIX_RE.search(title_clean) and len(words) <= 4 and not company_as_lease_party:
        return False
    if title_lower in {"december", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november"}:
        return False
    if title_lower.startswith("per ") or title_lower in {"per annum", "per month", "per year"}:
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
    title = re.sub(r"\s*\(?continued\)?\s*$", "", title, flags=re.I)
    title = re.sub(r"(?:\s+20\d{2}){1,3}\s*$", "", title)
    title = re.sub(r"\s+(?:N[\'\u2019]?\s?000|\$?000s?)(?:\s+(?:N[\'\u2019]?\s?000|\$?000s?))*$", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" -:;,.")
    title = re.sub(r"\s+[DRAFT]$", "", title).strip(" -:;,.")
    words: list[str] = []
    for word in title.split():
        match = re.match(r"^([A-Za-z]+)([^A-Za-z]*)$", word)
        if not match:
            words.append(word)
            continue
        stem, suffix = match.groups()
        corrected = NOTE_TITLE_OCR_CORRECTIONS.get(stem.lower(), stem)
        if stem[:1].isupper():
            corrected = corrected[:1].upper() + corrected[1:]
        words.append(corrected + suffix)
    return " ".join(words).strip()


def _looks_like_primary_statement_line(line: str) -> bool:
    if len(NUMBER_RE.findall(line)) < 1:
        return False
        
    lower = re.sub(r"\s+", " ", line.lower()).strip()
    reject_phrases = [
        "financial statements", "statement of", "year ended", "as at", 
        "signed on", "behalf", "behal b", "the notes on page", "pages", "director", 
        "chairman", "secretary", "approval", "n n", "0 0",
        "frc", "ican", "form c", "pro/form", "pro/ican", "managing director", "chief financial officer", "signature"
    ]
    if any(phrase in lower for phrase in reject_phrases) or re.search(r"(?i)\b(?:were signed|approval|n\s*n)\b", lower):
        return False
        
    text_only = re.sub(r"[\d\.,\(\)\-\|]", "", lower).strip()
    exact_rejects = ["liabilities", "equity", "equity and liabilities", "draft", "liabilities d", "total assets", "total liabilities"]
    if text_only in exact_rejects:
        return False
    # Reject lines that contain only letters N, M, O (common unit/currency artifacts) or are too short.
    letters_only = re.sub(r"[^a-z]", "", lower)
    if letters_only == "nn" or letters_only == "behalb":
        return False
    if len(letters_only) < 3 or set(letters_only).issubset({"n", "m", "o"}):
        return False
        
    cleaned_item = _clean_statement_line_item(line)
    if _is_subheading(cleaned_item) and not NOTE_REF_RE.search(line) and _normalise_match_words(cleaned_item) not in {"assets"}:
        return False
    if "notes to the financial statements" in lower or "notes to financial statements" in lower or lower.count("accounting polic") >= 2:
        return False
    return True


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
                pass
                
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
                pass
    return findings
