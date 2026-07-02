from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_policy_review import (
    AiProviderError,
    MalformedAiResponseError,
    _call_openai,
    _friendly_ai_error_message,
    _normalize_confidence,
    _normalize_severity,
    _parse_response_json,
    _repair_response_json,
)
from models import DEFAULT_AI_MODEL, CompanyProfile, Finding, PdfDocument

COMBINED_AI_OUTPUT_TOKENS = max(1200, min(int(os.getenv("OPENAI_COMBINED_REVIEW_OUTPUT_TOKENS", "3500")), 6000))
COMBINED_AI_REPAIR_SHAPE = (
    '{"summary":"short conclusion","executive_review_memo":"short memo",'
    '"overall_signoff_conclusion":"ready|not ready|manual review required with reason",'
    '"immediate_action_points":["action 1","action 2"],'
    '"cash_flow_conclusion":"short note on cash flow correctness",'
    '"regulatory_reference_conclusion":"short note on regulatory-reference issues",'
    '"casting_cross_casting_conclusion":"short note on casting and cross-casting issues",'
    '"review_comment_rows":[{"section_or_statement_or_note":"...","page_number":"Page X",'
    '"account_or_line_item":"...","current_wording_amount_reference":"...",'
    '"issue_identified":"...","expected_correction_recommendation":"...",'
    '"category":"Spelling / Grammar|Regulatory Reference|Note Cross-reference|Casting|Cross-casting|Cash Flow|Disclosure|Presentation|Internal Consistency",'
    '"priority":"High|Medium|Low","status":"Open","reviewer_comments":"..."}],'
    '"policy_review_findings":[{"title":"...","severity":"High|Medium|Low","confidence":"High|Medium|Low",'
    '"status":"exception|review_prompt|ok","page_reference":"Page X","note_reference":"Note X","issue":"...",'
    '"evidence_snippet":"...","recommendation":"...","rationale":"..."}],'
    '"missed_review_findings":[{"title":"...","category":"Formatting|Grammar|Disclosure|Presentation|Notes agreement|Cash Flow|Regulatory Reference|Internal Consistency",'
    '"severity":"High|Medium|Low","confidence":"High|Medium|Low","status":"exception|review_prompt|ok",'
    '"page_reference":"Page X","note_reference":"Note X","issue":"...","evidence_snippet":"...",'
    '"recommendation":"...","rationale":"..."}],'
    '"finding_adjudications":[{"finding_id":"F1","decision":"keep|downgrade|suppress|rewrite",'
    '"revised_severity":"High|Medium|Low","confidence":"High|Medium|Low","status":"confirmed_exception|review_prompt|likely_false_positive|insufficient_evidence",'
    '"reason":"...","recommended_action":"...","rewrite":"..."}]}'
)

STATEMENT_PATTERNS = (
    "statement of financial position",
    "statement of profit or loss",
    "statement of comprehensive income",
    "statement of income and expenditure",
    "statement of changes in equity",
    "statement of changes in accumulated fund",
    "statement of cash flows",
)

MODE_LIMITS = {
    "quick": {
        "primary_chars": 4500,
        "notes_chars": 2500,
        "contents_chars": 1600,
        "key_pages": 4,
        "key_page_chars": 700,
        "findings": 25,
        "skipped": 12,
    },
    "standard": {
        "primary_chars": 8500,
        "notes_chars": 5500,
        "contents_chars": 2500,
        "key_pages": 8,
        "key_page_chars": 1000,
        "findings": 45,
        "skipped": 20,
    },
    "deep": {
        "primary_chars": 14000,
        "notes_chars": 10000,
        "contents_chars": 4000,
        "key_pages": 14,
        "key_page_chars": 1300,
        "findings": 70,
        "skipped": 35,
    },
}


@dataclass
class CombinedAiReviewResult:
    findings: list[Finding]
    policy_export: list[dict[str, str]] = field(default_factory=list)
    full_export: list[dict[str, str]] = field(default_factory=list)
    finding_export: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    executive_memo: str = ""
    status: str = "disabled"
    model: str = ""
    message: str = ""
    suppressed_count: int = 0
    suppressed_rows: list[dict[str, str]] = field(default_factory=list)
    reviewed_count: int = 0
    evidence_rows: list[dict[str, str]] = field(default_factory=list)
    error_rows: list[dict[str, str]] = field(default_factory=list)
    review_comment_rows: list[dict[str, str]] = field(default_factory=list)
    summary_fields: dict[str, str] = field(default_factory=dict)
    review_mode: str = "standard"



def run_combined_ai_review(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool],
    findings: list[Finding],
    checks_skipped: list[str],
    model: str = DEFAULT_AI_MODEL,
    review_mode: str = "Standard review",
) -> CombinedAiReviewResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    normalized_mode = _normalize_review_mode(review_mode)
    if not api_key:
        return CombinedAiReviewResult(
            findings=list(findings),
            status="unavailable",
            model=model,
            message="AI review skipped because OPENAI_API_KEY is not configured.",
            review_mode=normalized_mode,
        )

    attempts = _model_attempts(model)
    error_rows: list[dict[str, str]] = []
    for attempt_index, attempt_model in enumerate(attempts, start=1):
        attempt_mode = normalized_mode if attempt_index == 1 else "quick"
        package = _build_compact_review_package(
            document,
            profile,
            note_sections,
            policy_map,
            findings,
            checks_skipped,
            attempt_mode,
        )
        payload = _build_payload(attempt_model, package)
        evidence_row = _evidence_pack_row(package, attempt_model, attempt_mode, attempt_index)
        try:
            response_json = _call_openai(api_key, payload)
            try:
                parsed = _parse_response_json(response_json)
            except MalformedAiResponseError as parse_exc:
                parsed = _repair_response_json(api_key, attempt_model, parse_exc.text, COMBINED_AI_REPAIR_SHAPE)
            return _parsed_to_result(parsed, findings, attempt_model, attempt_mode, evidence_row, error_rows)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            row = _error_row_from_exception(exc, attempt_model, attempt_mode, attempt_index, package)
            error_rows.append(row)
            if _should_try_fallback(exc) and attempt_index < len(attempts):
                continue
            return CombinedAiReviewResult(
                findings=list(findings),
                status="deferred" if _is_retryable_or_capacity_error(exc) else "error",
                model=attempt_model,
                message=_friendly_ai_error_message(exc),
                evidence_rows=[evidence_row],
                error_rows=error_rows,
                review_mode=attempt_mode,
            )

    return CombinedAiReviewResult(
        findings=list(findings),
        status="error",
        model=model,
        message="AI review could not be completed after model fallback attempts.",
        error_rows=error_rows,
        review_mode=normalized_mode,
    )



def _build_payload(model: str, package: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "max_output_tokens": COMBINED_AI_OUTPUT_TOKENS,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial-statement quality-control reviewer. Use only the supplied compact evidence package. "
                    "The deterministic engine remains authoritative for arithmetic unless the supplied evidence proves otherwise. "
                    "Return one valid JSON object only; no markdown, no comments, no trailing commas."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(package),
            },
        ],
    }



def _build_prompt(package: dict[str, Any]) -> str:
    return (
        "Perform a detailed quality-control review of this compact draft financial-statement evidence package.\n\n"
        "Scope to consider where evidence is supplied:\n"
        "- Directors' Report, Directors' Responsibility Statement, management certifications, ICFR report, independent auditor's report, primary statements, notes, value-added statement, and five-year financial summary.\n"
        "- Spelling, grammar, typographical, formatting, wording, duplicate caption, terminology, and presentation issues.\n"
        "- Improper, outdated, inconsistent, or incomplete regulatory references, including CAMA 2020, FRC Nigeria requirements, FRC ICFR guidance, Investments and Securities Act, tax laws, IFRS/IAS, and other applicable Nigerian reporting references.\n"
        "- Note references on the face of profit or loss/OCI, financial position, changes in equity, and cash flows; check note numbering, page references, note descriptions, and line-item compatibility.\n"
        "- Casting and cross-casting: vertical/horizontal casts, subtotals, grand totals, movement tables, comparatives, Group versus Company columns, and ties between primary statements and notes.\n"
        "- IAS 7 cash flow correctness: operating/investing/financing subtotals, net movement, opening/closing cash, overdrafts, interest, tax, non-cash items, fair value/impairment adjustments, PPE/investment-property movements, financial asset/liability movements, subsidiaries/associates.\n\n"
        "Tasks:\n"
        "1. Assess policy relevance, standards context, disclosure completeness, industry fit, and regulatory-reference quality.\n"
        "2. Identify missed presentation, spelling, grammar, note-reference, narrative contradiction, disclosure, cash-flow, casting, and internal-consistency issues.\n"
        "3. Review deterministic draft findings and recommend keep, downgrade, suppress, or rewrite where evidence supports it.\n"
        "4. Identify false positives caused by OCR, watermarks, signatures, repeated headers, five-year summaries, value-added statements, or table parsing drift.\n"
        "5. Populate review_comment_rows using the audit-style categories and High/Medium/Low priorities from the JSON schema.\n\n"
        "Rules:\n"
        "- Use only the supplied compact evidence package; do not invent missing pages or note content.\n"
        "- Do not suppress arithmetic findings unless supplied amount evidence supports suppression.\n"
        "- For amount findings, include page number, statement/note name, reported amount, expected amount, difference, and evidence in the rationale/recommendation fields where available.\n"
        "- For cross-reference issues, show both the incorrect reference and the correct reference where evidence supports it.\n"
        "- For regulatory-reference issues, quote or paraphrase the current wording and recommend the revised wording.\n"
        "- High priority means material arithmetic, regulatory issue, cash-flow error, incorrect primary-statement tie-in, or audit sign-off issue. Medium means disclosure inconsistency, note-reference error, significant wording, or presentation correction. Low means minor spelling, grammar, formatting, or style.\n"
        "- Low-confidence issues should be review_prompt, not confirmed exceptions.\n\n"
        "Return JSON with this shape:\n"
        f"{COMBINED_AI_REPAIR_SHAPE}\n\n"
        "Compact evidence package:\n"
        f"{json.dumps(package, ensure_ascii=False, indent=2)}"
    )


def _build_compact_review_package(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool],
    findings: list[Finding],
    checks_skipped: list[str],
    review_mode: str,
) -> dict[str, Any]:
    limits = MODE_LIMITS.get(review_mode, MODE_LIMITS["standard"])
    package = {
        "review_mode": review_mode,
        "company_profile": {
            "company_name": profile.company_name or "Not specified",
            "industry": profile.industry or "Not specified",
            "reporting_currency": profile.reporting_currency or "Not specified",
            "framework": profile.presentation_standard or "IFRS",
            "expected_policies": list(profile.expected_policies),
            "significant_transactions": list(profile.significant_transactions),
            "forced_checklist_areas": list(profile.checklist_areas),
        },
        "extraction_profile": {
            "pages": len(document.pages),
            "ocr_used": document.ocr_used,
            "ocr_pages": document.ocr_pages,
            "text_coverage": f"{document.extraction_coverage:.0%}",
            "extraction_confidence": document.extraction_confidence,
            "table_extraction_confidence": document.table_extraction_confidence,
        },
        "detected_policy_topics": [key for key, value in sorted(policy_map.items()) if value],
        "contents_or_section_map": _contents_context(document, limits["contents_chars"]),
        "primary_statements": _primary_statement_context(document, limits["primary_chars"]),
        "notes_1_and_2": _notes_1_and_2_context(document, note_sections, limits["notes_chars"]),
        "note_heading_map": _note_heading_map(note_sections),
        "front_matter_and_other_sections": _section_context(document, limits),
        "cash_flow_context": _cash_flow_context(document, limits["primary_chars"]),
        "regulatory_reference_snippets": _regulatory_reference_context(document, limits["key_pages"], limits["key_page_chars"]),
        "draft_exception_register": _draft_findings(findings, limits["findings"]),
        "checks_skipped": [str(item)[:700] for item in checks_skipped[: limits["skipped"]]],
        "key_pages_or_snippets": _key_page_snippets(document, findings, limits["key_pages"], limits["key_page_chars"]),
    }
    package["package_character_count"] = len(json.dumps(package, ensure_ascii=False))
    package["input_token_estimate"] = max(1, int(package["package_character_count"] / 4))
    return package



def _contents_context(document: PdfDocument, char_limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used = 0
    for page in document.pages:
        lower = str(page.text or "").lower()[:2500]
        if "contents" not in lower and "table of contents" not in lower:
            continue
        snippet = _compact(page.text, min(1200, char_limit - used))
        if snippet:
            rows.append({"page": str(page.number), "snippet": snippet})
            used += len(snippet)
        if used >= char_limit:
            break
    return rows



def _primary_statement_context(document: PdfDocument, char_limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used = 0
    for page in document.pages:
        lower = str(page.text or "").lower()[:3500]
        matched = [pattern for pattern in STATEMENT_PATTERNS if pattern in lower]
        if not matched:
            continue
        snippet = _compact(page.text, min(1800, char_limit - used))
        if snippet:
            rows.append({"page": str(page.number), "statement_type": ", ".join(matched[:2]), "snippet": snippet})
            used += len(snippet)
        if used >= char_limit:
            break
    return rows



def _notes_1_and_2_context(document: PdfDocument, note_sections: dict[str, str], char_limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used = 0
    for ref in ("1", "2"):
        section = str(note_sections.get(ref, "") or "")
        if not section:
            continue
        snippet = _compact(section, min(2800, char_limit - used))
        if snippet:
            rows.append({"note": ref, "snippet": snippet})
            used += len(snippet)
        if used >= char_limit:
            return rows
    if rows:
        return rows
    for page in document.pages:
        lower = str(page.text or "").lower()
        if "significant accounting policies" not in lower and "basis of preparation" not in lower:
            continue
        snippet = _compact(page.text, min(2400, char_limit - used))
        if snippet:
            rows.append({"note": "policy-page", "page": str(page.number), "snippet": snippet})
            used += len(snippet)
        if used >= char_limit:
            break
    return rows




def _section_context(document: PdfDocument, limits: dict[str, int]) -> list[dict[str, str]]:
    section_keywords = {
        "directors_report": ("directors' report", "directors report", "report of the directors"),
        "directors_responsibility": ("directors' responsibility", "directors responsibility", "statement of directors"),
        "management_certification": ("management certification", "certification", "chief executive officer", "chief financial officer"),
        "icfr": ("internal control over financial reporting", "icfr", "internal control"),
        "auditor_report": ("independent auditor", "auditor's report", "independent auditors"),
        "value_added_statement": ("value added statement", "statement of value added"),
        "five_year_summary": ("five-year financial summary", "five year financial summary", "five-year summary", "five year summary"),
    }
    max_rows = max(4, min(int(limits.get("key_pages", 8)), 14))
    chars_per_row = max(500, min(int(limits.get("key_page_chars", 900)), 1400))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for page in document.pages:
        lower = str(page.text or "").lower()
        for section, keywords in section_keywords.items():
            if not any(keyword in lower for keyword in keywords):
                continue
            key = (section, page.number)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"section": section, "page": str(page.number), "snippet": _compact(page.text, chars_per_row)})
            if len(rows) >= max_rows:
                return rows
    return rows


def _cash_flow_context(document: PdfDocument, char_limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used = 0
    cash_terms = (
        "statement of cash flows",
        "cash flows",
        "cash and cash equivalents",
        "net cash",
        "operating activities",
        "investing activities",
        "financing activities",
    )
    for page in document.pages:
        lower = str(page.text or "").lower()
        if not any(term in lower for term in cash_terms):
            continue
        snippet = _compact(page.text, min(1800, char_limit - used))
        if snippet:
            rows.append({"page": str(page.number), "snippet": snippet})
            used += len(snippet)
        if used >= char_limit:
            break
    return rows


def _regulatory_reference_context(document: PdfDocument, max_pages: int, chars_per_page: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    patterns = (
        "cama", "companies and allied matters", "financial reporting council", "frc", "icfr",
        "investments and securities", "securities and exchange", "companies income tax", "finance act",
        "ifrs", "ias ", "isa ", "nigerian", "corporate governance",
    )
    for page in document.pages:
        lower = str(page.text or "").lower()
        if not any(pattern in lower for pattern in patterns):
            continue
        rows.append({"page": str(page.number), "snippet": _compact(page.text, chars_per_page)})
        if len(rows) >= max_pages:
            break
    return rows



def _note_heading_map(note_sections: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ref, section in list(note_sections.items())[:45]:
        heading = _compact(str(section or "").split("\n", 1)[0], 120)
        rows.append({"note": str(ref), "heading_or_start": heading})
    return rows



def _draft_findings(findings: list[Finding], limit: int) -> list[dict[str, str]]:
    priority = {"High": 0, "Medium": 1, "Low": 2}
    ordered = sorted(enumerate(findings, start=1), key=lambda item: (priority.get(item[1].severity, 3), item[0]))
    rows: list[dict[str, str]] = []
    for original_index, finding in ordered[:limit]:
        metadata = finding.metadata or {}
        rows.append(
            {
                "finding_id": f"F{original_index}",
                "category": finding.category,
                "severity": finding.severity,
                "location": finding.location,
                "page_reference": str(metadata.get("page_reference", "") or ""),
                "note_reference": str(metadata.get("note_reference", "") or ""),
                "issue": finding.issue[:900],
                "evidence": finding.evidence[:1200],
                "recommendation": finding.recommendation[:700],
                "statement": str(metadata.get("statement", "") or ""),
                "line_item": str(metadata.get("line_item", "") or ""),
                "reported_amount": str(metadata.get("reported_amount", "") or ""),
                "expected_amount": str(metadata.get("expected_amount", "") or ""),
                "difference": str(metadata.get("difference", "") or ""),
            }
        )
    return rows



def _key_page_snippets(document: PdfDocument, findings: list[Finding], max_pages: int, chars_per_page: int) -> list[dict[str, str]]:
    pages: list[int] = []
    for finding in findings:
        text = "\n".join(str(part or "") for part in (finding.location, finding.evidence, finding.issue))
        for match in re.findall(r"\bPage\s+(\d+)\b", text, flags=re.I):
            value = int(match)
            if value not in pages:
                pages.append(value)
        if len(pages) >= max_pages:
            break
    rows: list[dict[str, str]] = []
    for page_number in pages[:max_pages]:
        page = next((page for page in document.pages if page.number == page_number), None)
        if page and page.text.strip():
            rows.append({"page": str(page.number), "snippet": _compact(page.text, chars_per_page)})
    return rows



def _parsed_to_result(
    parsed: dict[str, Any],
    original_findings: list[Finding],
    model: str,
    review_mode: str,
    evidence_row: dict[str, str],
    error_rows: list[dict[str, str]],
) -> CombinedAiReviewResult:
    findings_after_adjudication, finding_export, suppressed_rows, reviewed_count = _apply_combined_adjudications(
        original_findings,
        parsed.get("finding_adjudications", []) or [],
    )
    policy_findings, policy_rows = _observations_to_outputs(parsed.get("policy_review_findings", []) or parsed.get("policy_findings", []), "AI policy judgement")
    missed_findings, missed_rows = _observations_to_outputs(parsed.get("missed_review_findings", []) or parsed.get("missed_findings", []), "AI full review")
    final_findings = findings_after_adjudication + policy_findings + missed_findings
    review_comment_rows = _review_comment_rows_from_parsed(parsed, policy_rows + missed_rows)
    return CombinedAiReviewResult(
        findings=final_findings,
        policy_export=policy_rows,
        full_export=missed_rows,
        finding_export=finding_export,
        summary=str(parsed.get("summary", "") or "").strip(),
        executive_memo=str(parsed.get("executive_review_memo", "") or "").strip(),
        status="completed",
        model=model,
        suppressed_count=len(suppressed_rows),
        suppressed_rows=suppressed_rows,
        reviewed_count=reviewed_count,
        evidence_rows=[evidence_row],
        error_rows=error_rows,
        review_comment_rows=review_comment_rows,
        summary_fields=_summary_fields_from_parsed(parsed),
        review_mode=review_mode,
    )



def _observations_to_outputs(observations: list[dict[str, Any]], category: str) -> tuple[list[Finding], list[dict[str, str]]]:
    findings: list[Finding] = []
    rows: list[dict[str, str]] = []
    for observation in observations[:30]:
        status = str(observation.get("status", "") or "").strip().lower()
        if status == "ok":
            continue
        severity = _normalize_severity(observation.get("severity", "Low"))
        confidence = _normalize_confidence(observation.get("confidence", "Low"))
        title = str(observation.get("title", "") or "AI review observation").strip()
        page_reference = str(observation.get("page_reference", "") or "").strip()
        note_reference = str(observation.get("note_reference", "") or "").strip()
        issue = str(observation.get("issue", "") or title).strip()
        evidence = str(observation.get("evidence_snippet", "") or observation.get("rationale", "") or title).strip()
        recommendation = str(observation.get("recommendation", "") or "Review the referenced source page/note and confirm the AI observation.").strip()
        metadata = {
            "check_type": category,
            "ai_review_status": "confirmed_exception" if status == "exception" else "review_prompt",
            "ai_review_confidence": confidence,
            "ai_review_reason": str(observation.get("rationale", "") or ""),
            "match_confidence": confidence,
            "page_reference": page_reference,
            "note_reference": note_reference,
            "line_item": title,
        }
        findings.append(Finding(category, severity, page_reference or note_reference or "Document-wide", issue, evidence, recommendation, metadata))
        rows.append(
            {
                "Title": title,
                "Status": status or "review_prompt",
                "Severity": severity,
                "Confidence": confidence,
                "Page reference": page_reference,
                "Note reference": note_reference,
                "Issue": issue,
                "Evidence": evidence,
                "Recommendation": recommendation,
                "Rationale": str(observation.get("rationale", "") or ""),
            }
        )
    return findings, rows




def _summary_fields_from_parsed(parsed: dict[str, Any]) -> dict[str, str]:
    action_points = parsed.get("immediate_action_points", [])
    if isinstance(action_points, list):
        action_text = "\n".join(str(item).strip() for item in action_points if str(item).strip())
    else:
        action_text = str(action_points or "").strip()
    return {
        "Overall sign-off conclusion": str(parsed.get("overall_signoff_conclusion", "") or "").strip(),
        "Recommended immediate action points": action_text,
        "Cash flow correctness note": str(parsed.get("cash_flow_conclusion", "") or "").strip(),
        "Regulatory-reference note": str(parsed.get("regulatory_reference_conclusion", "") or "").strip(),
        "Casting and cross-casting note": str(parsed.get("casting_cross_casting_conclusion", "") or "").strip(),
    }


def _review_comment_rows_from_parsed(parsed: dict[str, Any], fallback_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    raw_rows = parsed.get("review_comment_rows", [])
    rows: list[dict[str, str]] = []
    if isinstance(raw_rows, list):
        for index, row in enumerate(raw_rows, start=1):
            if not isinstance(row, dict):
                continue
            normalized = _normalize_review_comment_row(row, index)
            if normalized["Issue identified"]:
                rows.append(normalized)
    if rows:
        return rows
    for index, row in enumerate(fallback_rows, start=1):
        rows.append(
            {
                "S/N": str(index),
                "Section / Statement / Note": str(row.get("Title", "") or row.get("Note reference", "") or "AI review"),
                "Page number": str(row.get("Page reference", "") or ""),
                "Account / line item": str(row.get("Title", "") or ""),
                "Current wording / amount / reference": str(row.get("Evidence", "") or ""),
                "Issue identified": str(row.get("Issue", "") or ""),
                "Expected correction / recommendation": str(row.get("Recommendation", "") or ""),
                "Category": _normalize_review_comment_category(row.get("Title", "") or row.get("Issue", "")),
                "Priority": _normalize_severity(row.get("Severity", "Low")),
                "Status": "Open",
                "Reviewer comments": str(row.get("Rationale", "") or ""),
            }
        )
    return rows


def _normalize_review_comment_row(row: dict[str, Any], index: int) -> dict[str, str]:
    return {
        "S/N": str(row.get("S/N") or row.get("s_n") or row.get("sn") or index),
        "Section / Statement / Note": str(row.get("Section / Statement / Note") or row.get("section_or_statement_or_note") or row.get("section") or "").strip(),
        "Page number": str(row.get("Page number") or row.get("page_number") or row.get("page_reference") or "").strip(),
        "Account / line item": str(row.get("Account / line item") or row.get("account_or_line_item") or row.get("line_item") or "").strip(),
        "Current wording / amount / reference": str(row.get("Current wording / amount / reference") or row.get("current_wording_amount_reference") or row.get("current_reference") or row.get("evidence") or "").strip(),
        "Issue identified": str(row.get("Issue identified") or row.get("issue_identified") or row.get("issue") or "").strip(),
        "Expected correction / recommendation": str(row.get("Expected correction / recommendation") or row.get("expected_correction_recommendation") or row.get("recommendation") or "").strip(),
        "Category": _normalize_review_comment_category(row.get("Category") or row.get("category") or row.get("issue_identified") or ""),
        "Priority": _normalize_severity(row.get("Priority") or row.get("priority") or row.get("severity") or "Low"),
        "Status": str(row.get("Status") or row.get("status") or "Open").strip() or "Open",
        "Reviewer comments": str(row.get("Reviewer comments") or row.get("reviewer_comments") or row.get("rationale") or "").strip(),
    }


def _normalize_review_comment_category(value: object) -> str:
    lower = str(value or "").lower()
    if "spell" in lower or "grammar" in lower or "typograph" in lower or "wording" in lower:
        return "Spelling / Grammar"
    if "regulat" in lower or "cama" in lower or "frc" in lower or "icfr" in lower or "securities" in lower or "tax law" in lower:
        return "Regulatory Reference"
    if "note" in lower and ("reference" in lower or "cross" in lower):
        return "Note Cross-reference"
    if "cross" in lower and ("cast" in lower or "tie" in lower or "agreement" in lower):
        return "Cross-casting"
    if "cash" in lower or "ias 7" in lower:
        return "Cash Flow"
    if "cast" in lower or "total" in lower or "subtotal" in lower or "arithmetic" in lower:
        return "Casting"
    if "disclosure" in lower or "missing" in lower:
        return "Disclosure"
    if "present" in lower or "format" in lower or "caption" in lower:
        return "Presentation"
    return "Internal Consistency"



def _apply_combined_adjudications(
    original_findings: list[Finding],
    adjudications: list[dict[str, Any]],
) -> tuple[list[Finding], list[dict[str, str]], list[dict[str, str]], int]:
    finding_map = {f"F{index}": index - 1 for index in range(1, len(original_findings) + 1)}
    decisions = {str(item.get("finding_id", "") or "").strip(): item for item in adjudications if isinstance(item, dict)}
    final_findings: list[Finding] = []
    export_rows: list[dict[str, str]] = []
    suppressed_rows: list[dict[str, str]] = []
    reviewed_count = 0
    for index, finding in enumerate(original_findings, start=1):
        cloned = _clone_finding(finding)
        decision = decisions.get(f"F{index}") or decisions.get(f"EX-{index:03d}")
        if not decision:
            final_findings.append(cloned)
            continue
        reviewed_count += 1
        action = str(decision.get("decision", "keep") or "keep").strip().lower()
        confidence = _normalize_confidence(decision.get("confidence", "Low"))
        reason = str(decision.get("reason", "") or "").strip()
        revised_severity = _normalize_severity(decision.get("revised_severity", cloned.severity))
        status = str(decision.get("status", "") or "").strip()
        row = {
            "Finding ID": f"F{index}",
            "Category": cloned.category,
            "Original severity": cloned.severity,
            "Decision": action.title(),
            "Revised severity": revised_severity,
            "AI status": status,
            "AI confidence": confidence,
            "Issue": cloned.issue,
            "Reason": reason,
            "Recommended action": str(decision.get("recommended_action", "") or ""),
        }
        export_rows.append(row)
        if action == "suppress":
            suppressed_rows.append(row)
            continue
        if action in {"downgrade", "rewrite"}:
            cloned.severity = _lower_or_same_severity(cloned.severity, revised_severity)
            metadata = dict(cloned.metadata or {})
            metadata.update({"ai_review_status": status, "ai_review_confidence": confidence, "ai_review_reason": reason})
            cloned.metadata = metadata
            if action == "rewrite" and str(decision.get("rewrite", "") or "").strip():
                cloned.issue = str(decision.get("rewrite", "") or "").strip()
            if reason:
                cloned.evidence = f"{cloned.evidence}\nAI review: {reason}".strip()
        final_findings.append(cloned)
    return final_findings, export_rows, suppressed_rows, reviewed_count



def _lower_or_same_severity(current: str, revised: str) -> str:
    order = {"High": 3, "Medium": 2, "Low": 1}
    return revised if order.get(revised, 0) <= order.get(current, 0) else current



def _clone_finding(finding: Finding) -> Finding:
    return Finding(
        finding.category,
        finding.severity,
        finding.location,
        finding.issue,
        finding.evidence,
        finding.recommendation,
        dict(finding.metadata or {}),
    )



def _model_attempts(model: str) -> list[str]:
    preferred = model or DEFAULT_AI_MODEL
    fallback = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    if fallback == preferred:
        fallback = os.getenv("OPENAI_SECONDARY_FALLBACK_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return [preferred, fallback]



def _should_try_fallback(exc: Exception) -> bool:
    category = _error_category(exc)
    return category in {"rate_limit", "timeout", "temporary_service_error", "payload_too_large", "busy", "other"}



def _is_retryable_or_capacity_error(exc: Exception) -> bool:
    return _error_category(exc) in {"rate_limit", "timeout", "temporary_service_error", "payload_too_large", "busy", "insufficient_quota"}



def _error_category(exc: Exception) -> str:
    if isinstance(exc, AiProviderError):
        return str(exc.diagnostics.get("error_category", "") or "other")
    text = str(exc or "").lower()
    if "quota" in text or "billing" in text or "credit" in text:
        return "insufficient_quota"
    if "token" in text or "context" in text or "too large" in text:
        return "payload_too_large"
    if "429" in text or "rate" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "busy" in text:
        return "busy"
    return "other"



def _error_row_from_exception(
    exc: Exception,
    model: str,
    review_mode: str,
    attempt_index: int,
    package: dict[str, Any],
) -> dict[str, str]:
    diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
    return {
        "Attempt": str(attempt_index),
        "Model": model,
        "Review mode": review_mode,
        "Exception type": diagnostics.get("error_type", type(exc).__name__),
        "Status code": str(diagnostics.get("status_code", "")),
        "Error category": diagnostics.get("error_category", _error_category(exc)),
        "Error message": str(diagnostics.get("error_message", str(exc)))[:1500],
        "Input token estimate": str(diagnostics.get("input_token_estimate", package.get("input_token_estimate", ""))),
        "Output token limit": str(diagnostics.get("max_output_tokens", COMBINED_AI_OUTPUT_TOKENS)),
        "Retry count": str(diagnostics.get("retry_count", "")),
        "Retry wait time": json.dumps(diagnostics.get("retry_wait_seconds", [])),
        "Package character count": str(package.get("package_character_count", "")),
    }



def _evidence_pack_row(package: dict[str, Any], model: str, review_mode: str, attempt_index: int) -> dict[str, str]:
    return {
        "Evidence type": "Combined AI compact review package",
        "AI role": "Policy judgement, missed review findings, and deterministic finding cleanup in one request.",
        "Model": model,
        "Review mode": review_mode,
        "Attempt": str(attempt_index),
        "Input token estimate": str(package.get("input_token_estimate", "")),
        "Package character count": str(package.get("package_character_count", "")),
        "Primary statements included": str(len(package.get("primary_statements", []))),
        "Notes 1 and 2 included": str(len(package.get("notes_1_and_2", []))),
        "Draft findings included": str(len(package.get("draft_exception_register", []))),
    }



def _normalize_review_mode(value: str) -> str:
    lower = str(value or "standard").strip().lower()
    if lower.startswith("quick"):
        return "quick"
    if lower.startswith("deep"):
        return "deep"
    return "standard"



def _compact(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[: max(0, limit)]
