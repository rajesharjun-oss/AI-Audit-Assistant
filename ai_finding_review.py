from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from ai_policy_review import MalformedAiResponseError, _call_openai, _friendly_ai_error_message, _normalize_confidence, _normalize_severity, _parse_response_json, _repair_response_json
from models import DEFAULT_AI_MODEL, CompanyProfile, Finding, PdfDocument


REVIEWABLE_CATEGORIES = {
    "Totals and rounding",
    "Notes agreement",
    "Formatting",
    "Name consistency",
    "Date consistency",
    "Grammar review",
    "Value Added Statement",
    "Presentation",
    "Narrative consistency",
    "Consistency",
}
SKIPPED_CATEGORIES = {"Extraction quality", "Document scope", "AI policy judgement"}
MAX_REVIEW_FINDINGS = 40
AI_SOURCE_PDF_MAX_BYTES = int(os.getenv("AI_SOURCE_PDF_MAX_BYTES", str(12 * 1024 * 1024)))
_AI_FINDING_REVIEW_REPAIR_SHAPE = (
    '{"summary":"short reviewer summary","adjudications":[{"finding_id":"F1","decision":"keep|downgrade|suppress|rewrite",'
    '"revised_severity":"High|Medium|Low","status":"confirmed_exception|review_prompt|likely_false_positive|insufficient_evidence",'
    '"confidence":"High|Medium|Low","reason":"...","recommended_action":"...","rewrite":"...",'
    '"amount_evidence":{"page_number":"Page X","statement_or_note_name":"...","reported_amount":"...","expected_amount":"...","difference":"...","evidence":"..."},'
    '"judgement_evidence":{"page_number":"Page X","issue":"...","evidence":"...","recommendation":"...","severity":"High|Medium|Low","confidence":"High|Medium|Low"}}]}'
)
NOTE_HEADING_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)(?:\s*\([a-z]\))?\s*[\).:-]?\s*(.{3,100})$", re.I)


@dataclass(frozen=True)
class AiFindingReviewResult:
    findings: list[Finding]
    export_rows: list[dict[str, str]]
    summary: str
    status: str
    model: str
    message: str = ""
    reviewed_count: int = 0
    suppressed_count: int = 0
    evidence_rows: list[dict[str, str]] | None = None
    suppressed_rows: list[dict[str, str]] | None = None


def run_ai_finding_review(
    document: PdfDocument,
    profile: CompanyProfile,
    findings: list[Finding],
    model: str = DEFAULT_AI_MODEL,
    pdf_path: str | Path | None = None,
) -> AiFindingReviewResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AiFindingReviewResult(
            findings=findings,
            export_rows=[],
            summary="",
            status="unavailable",
            model=model,
            message="AI finding review skipped because OPENAI_API_KEY is not configured.",
            evidence_rows=[],
        )

    candidates = _review_candidates(document, findings)
    if not candidates:
        return AiFindingReviewResult(
            findings=findings,
            export_rows=[],
            summary="",
            status="skipped",
            model=model,
            message="AI finding review skipped because no ambiguous findings were eligible for adjudication.",
            evidence_rows=[],
        )

    user_content, source_pdf_evidence_row = _build_user_content(document, profile, candidates, pdf_path)
    payload = {
        "model": model,
        "max_output_tokens": 2200,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are assisting a deterministic financial-statement review engine. "
                    "You must be conservative, evidence-bound, and page-grounded. "
                    "Do not invent evidence. Only use the attached PDF when provided, extracted statement/table context, page snippets, note snippets, issue text, and evidence text supplied. "
                    "Treat deterministic arithmetic and structure findings as primary evidence, but suppress likely false positives from OCR noise, split digits, duplicated headers, five-year-summary contamination, value-added contamination, weak note linkage, or layout extraction drift. "
                    "Do not suppress a finding simply because it is inconvenient. "
                    "Return one valid JSON object only; no markdown, no comments, and no trailing commas."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }

    try:
        response_json = _call_openai(api_key, payload)
        try:
            parsed = _parse_response_json(response_json)
        except MalformedAiResponseError as parse_exc:
            parsed = _repair_response_json(
                api_key,
                model,
                parse_exc.text,
                _AI_FINDING_REVIEW_REPAIR_SHAPE,
            )
        return _apply_adjudications(findings, candidates, parsed, model, source_pdf_evidence_row)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return AiFindingReviewResult(
            findings=findings,
            export_rows=[],
            summary="",
            status="deferred",
            model=model,
            message=_friendly_ai_error_message(exc),
            evidence_rows=([source_pdf_evidence_row] if 'source_pdf_evidence_row' in locals() and source_pdf_evidence_row else []) + (_candidate_evidence_rows(candidates) if 'candidates' in locals() else []),
        )


def _review_candidates(document: PdfDocument, findings: list[Finding]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if finding.category in SKIPPED_CATEGORIES:
            continue
        if REVIEWABLE_CATEGORIES and finding.category not in REVIEWABLE_CATEGORIES:
            continue
        if not _finding_is_ai_reviewable(document, finding):
            continue
        metadata = dict(finding.metadata or {})
        page_reference = _finding_page_reference(document, finding, metadata)
        note_reference = _finding_note_reference(finding, metadata)
        page_numbers = _candidate_page_numbers(document, page_reference)
        line_item = metadata.get("line_item", "")
        statement = metadata.get("statement", "")
        keywords = _candidate_keywords(finding, line_item, statement)
        page_snippet = _page_snippet(document, page_numbers, keywords, note_reference)
        note_snippet = _note_snippet(document, note_reference, keywords)
        candidates.append(
            {
                "finding_id": f"F{index + 1}",
                "index": index,
                "category": finding.category,
                "severity": finding.severity,
                "location": finding.location,
                "issue": finding.issue,
                "evidence": finding.evidence,
                "recommendation": finding.recommendation,
                "page_reference": page_reference,
                "note_reference": note_reference,
                "statement": statement,
                "line_item": line_item,
                "reason": metadata.get("reason", ""),
                "check_type": metadata.get("check_type", ""),
                "document_scope": _document_scope(document),
                "ocr_used": document.ocr_used,
                "extraction_confidence": document.extraction_confidence,
                "table_extraction_confidence": document.table_extraction_confidence,
                "page_snippet": page_snippet,
                "note_snippet": note_snippet,
                "amount_context": _finding_amount_context(finding, metadata),
                "extracted_statement_tables": _statement_table_context(document, page_numbers),
                "draft_register_row": _draft_register_row(finding, metadata, page_reference, note_reference),
            }
        )
        if len(candidates) >= MAX_REVIEW_FINDINGS:
            break
    return candidates




def _build_user_content(
    document: PdfDocument,
    profile: CompanyProfile,
    candidates: list[dict[str, Any]],
    pdf_path: str | Path | None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    source_pdf_item, source_pdf_row = _source_pdf_input_item(pdf_path)
    content: list[dict[str, str]] = [{"type": "input_text", "text": _build_prompt(document, profile, candidates, source_pdf_row)}]
    if source_pdf_item:
        content.append(source_pdf_item)
    return content, source_pdf_row


def _source_pdf_input_item(pdf_path: str | Path | None) -> tuple[dict[str, str] | None, dict[str, str]]:
    row = {
        "Evidence type": "Original PDF",
        "AI role": "Source PDF supplied to AI reviewer when available and below size cap.",
        "Status": "Not attached",
        "File name": "",
        "File size bytes": "",
        "Reason": "No source PDF path was supplied to the AI reviewer.",
    }
    if not pdf_path:
        return None, row
    path = Path(pdf_path)
    row["File name"] = path.name
    try:
        if not path.exists():
            row["Reason"] = "Source PDF path was not found on disk."
            return None, row
        size = path.stat().st_size
        row["File size bytes"] = str(size)
        if size > AI_SOURCE_PDF_MAX_BYTES:
            row["Reason"] = f"Source PDF is larger than AI_SOURCE_PDF_MAX_BYTES ({AI_SOURCE_PDF_MAX_BYTES})."
            return None, row
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        row["Reason"] = f"Source PDF could not be read: {exc}"
        return None, row
    row["Status"] = "Attached"
    row["Reason"] = "Original PDF attached as input_file for page-faithful AI review."
    return {
        "type": "input_file",
        "filename": path.name,
        "file_data": f"data:application/pdf;base64,{encoded}",
    }, row


def _draft_register_row(finding: Finding, metadata: dict[str, Any], page_reference: str, note_reference: str) -> dict[str, str]:
    return {
        "Category": finding.category,
        "Severity": finding.severity,
        "Page reference": page_reference,
        "Note reference": note_reference,
        "Location": finding.location,
        "Issue": finding.issue,
        "Evidence": finding.evidence,
        "Recommendation": finding.recommendation,
        "Statement": str(metadata.get("statement", "") or ""),
        "Line item": str(metadata.get("line_item", "") or ""),
        "Check type": str(metadata.get("check_type", finding.category) or finding.category),
    }


def _finding_amount_context(finding: Finding, metadata: dict[str, Any]) -> dict[str, str]:
    keys = (
        "reported_amount",
        "expected_amount",
        "difference",
        "current_year_amount",
        "prior_year_amount",
        "amount_found_in_note",
        "amount_match_confidence",
    )
    context = {key: str(metadata.get(key, "") or "") for key in keys if str(metadata.get(key, "") or "").strip()}
    amounts = re.findall(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?", f"{finding.issue} {finding.evidence}")
    if amounts:
        context["amounts_visible_in_finding"] = ", ".join(amounts[:8])
    return context


def _statement_table_context(document: PdfDocument, page_numbers: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pages = page_numbers or _primary_statement_pages(document)
    for page_number in pages[:4]:
        if page_number < 1 or page_number > len(document.pages):
            continue
        page = document.pages[page_number - 1]
        page_text = _compact_text(page.text, 900)
        if page_text:
            rows.append({"Page": str(page.number), "Context type": "page_text", "Extracted content": page_text})
        for table_index, table in enumerate(page.tables[:3], start=1):
            table_lines = []
            for table_row in table[:8]:
                table_lines.append(" | ".join(str(cell or "").strip() for cell in table_row[:8]))
            if table_lines:
                rows.append({
                    "Page": str(page.number),
                    "Context type": f"table_{table_index}",
                    "Extracted content": "\n".join(table_lines),
                })
    return rows[:10]


def _primary_statement_pages(document: PdfDocument) -> list[int]:
    patterns = (
        "statement of financial position",
        "statement of profit or loss",
        "statement of comprehensive income",
        "statement of income and expenditure",
        "statement of changes in equity",
        "statement of changes in accumulated fund",
        "statement of cash flows",
    )
    pages: list[int] = []
    for page in document.pages:
        lower = str(page.text or "").lower()[:3000]
        if any(pattern in lower for pattern in patterns):
            pages.append(page.number)
    return pages[:8]


def _compact_text(text: str, limit: int = 1200) -> str:
    return " ".join(str(text or "").split())[:limit]

def _candidate_evidence_rows(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        amount_context = candidate.get("amount_context") if isinstance(candidate.get("amount_context"), dict) else {}
        draft_row = candidate.get("draft_register_row") if isinstance(candidate.get("draft_register_row"), dict) else {}
        table_context = candidate.get("extracted_statement_tables") if isinstance(candidate.get("extracted_statement_tables"), list) else []
        rows.append(
            {
                "Evidence type": "Engine finding adjudication pack",
                "Finding ID": str(candidate.get("finding_id", "")),
                "Category": str(candidate.get("category", "")),
                "Severity": str(candidate.get("severity", "")),
                "Page reference": str(candidate.get("page_reference", "")),
                "Note reference": str(candidate.get("note_reference", "")),
                "Statement": str(candidate.get("statement", "")),
                "Line item": str(candidate.get("line_item", "")),
                "Issue": str(candidate.get("issue", "")),
                "Evidence": str(candidate.get("evidence", "")),
                "Page snippet": str(candidate.get("page_snippet", "")),
                "Note snippet": str(candidate.get("note_snippet", "")),
                "Amount context": json.dumps(amount_context, ensure_ascii=True),
                "Draft register row": json.dumps(draft_row, ensure_ascii=True),
                "Extracted statement table context": json.dumps(table_context, ensure_ascii=True),
                "AI role": "Classify as keep, downgrade, rewrite, or likely false positive using supplied evidence only",
            }
        )
    return rows

def _finding_is_ai_reviewable(document: PdfDocument, finding: Finding) -> bool:
    metadata = finding.metadata or {}
    if finding.severity in {"Low", "Medium"}:
        return True
    if finding.severity != "High":
        return False
    if metadata.get("ocr_review"):
        return True
    if document.ocr_used:
        return True
    if document.extraction_confidence < 85 or document.table_extraction_confidence < 85:
        return True
    return False


def _build_prompt(
    document: PdfDocument,
    profile: CompanyProfile,
    candidates: list[dict[str, Any]],
    source_pdf_row: dict[str, str] | None = None,
) -> str:
    company_name = profile.company_name or "Not specified"
    industry = profile.industry or "Auto-detect from context"
    source_pdf_status = source_pdf_row or {}
    return (
        "Review the deterministic engine's draft exception register for a financial-statement quality-control review.\n"
        "The deterministic engine remains responsible for exact arithmetic and structured checks. Your role is a second reviewer: validate evidence, identify false positives, and add judgement findings where the supplied context supports them.\n\n"
        "Inputs supplied in this message include draft findings, extracted page/table context, page/note snippets, and draft exception-register rows. If an original PDF is attached in the message, use it only to verify page-faithful evidence.\n\n"
        "Return JSON with this exact shape:\n"
        "{"
        '"summary":"short reviewer summary",'
        '"adjudications":[{'
        '"finding_id":"F1",'
        '"decision":"keep|downgrade|suppress|rewrite",'
        '"revised_severity":"High|Medium|Low",'
        '"status":"confirmed_exception|review_prompt|likely_false_positive|insufficient_evidence",'
        '"confidence":"High|Medium|Low",'
        '"reason":"...",'
        '"recommended_action":"...",'
        '"rewrite":"...",'
        '"amount_evidence":{"page_number":"Page X","statement_or_note_name":"...","reported_amount":"...","expected_amount":"...","difference":"...","evidence":"..."},'
        '"judgement_evidence":{"page_number":"Page X","issue":"...","evidence":"...","recommendation":"...","severity":"High|Medium|Low","confidence":"High|Medium|Low"}'
        "}]}\n\n"
        "Rules:\n"
        "1. Be conservative.\n"
        "2. Identify real presentation, spelling, grammar, disclosure, narrative contradiction, and note-reference issues when the evidence supports them.\n"
        "3. Identify likely false positives caused by OCR, watermarks, signatures, page headers, duplicated headings, five-year-summary contamination, value-added-statement contamination, or table parsing drift.\n"
        "4. Recommend keep, downgrade, suppress, or rewrite. Use suppress only when the supplied evidence supports a likely false positive.\n"
        "5. Do not promote severity above the original severity.\n"
        "6. Do not change or suppress amount/arithmetic findings without amount_evidence containing page number, statement/note name, reported amount, expected amount, difference, and evidence.\n"
        "7. For judgement findings or judgement-based changes, provide judgement_evidence containing page number, issue, evidence, recommendation, severity, and confidence.\n"
        "8. Do not invent evidence. Use the supplied source PDF, extracted statement tables, draft rows, page snippets, note snippets, issue text, and evidence text.\n"
        "9. Return valid JSON only. No markdown, no comments, no trailing commas.\n\n"
        f"Document context:\n- Company: {company_name}\n- Industry: {industry}\n"
        f"- OCR used: {'Yes' if document.ocr_used else 'No'}\n"
        f"- Extraction confidence: {document.extraction_confidence}%\n"
        f"- Table extraction confidence: {document.table_extraction_confidence}%\n"
        f"- Document scope: {_document_scope(document)}\n"
        f"- Original PDF status: {source_pdf_status.get('Status', 'Not attached')} ({source_pdf_status.get('Reason', '')})\n\n"
        "Candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=True, indent=2)}"
    )

def _apply_adjudications(
    findings: list[Finding],
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    model: str,
    source_pdf_evidence_row: dict[str, str] | None = None,
) -> AiFindingReviewResult:
    adjudications = parsed.get("adjudications", [])
    if not isinstance(adjudications, list):
        adjudications = []
    by_id = {
        str(item.get("finding_id", "")).strip(): item
        for item in adjudications
        if isinstance(item, dict) and str(item.get("finding_id", "")).strip()
    }
    export_rows: list[dict[str, str]] = []
    suppressed_rows: list[dict[str, str]] = []
    updated_findings = list(findings)
    suppressed_indexes: set[int] = set()

    for candidate in candidates:
        finding_id = candidate["finding_id"]
        adjudication = by_id.get(finding_id, {})
        decision = _normalize_ai_decision(adjudication.get("decision"))
        revised_severity = _cap_revised_severity(
            str(candidate.get("severity", "Low") or "Low"),
            _normalize_severity(adjudication.get("revised_severity") or candidate.get("severity", "Low")),
        )
        confidence = _normalize_confidence(adjudication.get("confidence") or candidate.get("severity", "Low"))
        status = str(adjudication.get("status", "") or "review_prompt").strip() or "review_prompt"
        reason = str(adjudication.get("reason", "") or "").strip()
        action = str(adjudication.get("recommended_action", "") or "").strip()
        rewrite = str(adjudication.get("rewrite", "") or "").strip()
        original = updated_findings[candidate["index"]]
        metadata = dict(original.metadata or {})
        unsupported_by_ai = _ai_reason_indicates_unsupported(reason, status)

        if _has_strong_narrative_contradiction_evidence(original):
            decision = "keep"
            status = "confirmed_exception"
            confidence = "High" if confidence != "Low" else confidence
            if unsupported_by_ai or not reason:
                reason = "The finding is supported by an explicit nil-style narrative and a non-zero amount in the same note evidence."
        elif unsupported_by_ai:
            if _ai_reason_indicates_likely_false_positive(reason, status):
                decision = "suppress"
                status = "likely_false_positive"
                confidence = "High"
            else:
                decision = "downgrade" if original.severity in {"High", "Medium"} else "suppress"
                status = "insufficient_evidence"
                confidence = "Medium" if confidence == "High" else confidence
            if not action:
                action = "Treat as a low-confidence AI review prompt unless the reviewer confirms the source evidence."

        if decision == "suppress" and not unsupported_by_ai and _suppression_conflicts_with_note_evidence(original, candidate, reason):
            decision = "downgrade" if original.severity == "High" else "keep"
            status = "confirmed_exception" if original.severity in {"High", "Medium"} else "review_prompt"
            confidence = "Medium" if confidence == "Low" else confidence
            reason = (
                reason
                or "The AI response describes a note/reference mismatch, so the finding is retained for reviewer follow-up."
            )

        change_requested = decision in {"downgrade", "suppress", "rewrite"} or revised_severity != original.severity
        guardrail = ""
        if change_requested:
            if _candidate_is_amount_related(candidate) and not _adjudication_has_amount_evidence(adjudication):
                decision = "keep"
                revised_severity = original.severity
                status = "confirmed_exception" if original.severity in {"High", "Medium"} else "review_prompt"
                guardrail = "AI reviewer requested a change to an amount-related finding without the required amount evidence. The deterministic finding was retained."
            elif not _candidate_is_amount_related(candidate) and not _adjudication_has_judgement_evidence(adjudication):
                decision = "keep"
                revised_severity = original.severity
                status = "confirmed_exception" if original.severity in {"High", "Medium"} else "review_prompt"
                guardrail = "AI reviewer requested a judgement-based change without the required judgement evidence. The deterministic finding was retained."
            if guardrail:
                reason = _append_reason(reason, guardrail)
                action = action or original.recommendation

        if decision == "suppress" and not _can_suppress_finding(original, confidence):
            decision = "downgrade" if original.severity in {"High", "Medium"} else "keep"
            status = "review_prompt"
            revised_severity = _downgraded_severity(original.severity) if decision == "downgrade" else original.severity
            reason = _append_reason(
                reason,
                "Suppression was not applied because the finding severity/confidence did not meet the suppression guardrail.",
            )

        amount_evidence = adjudication.get("amount_evidence") if isinstance(adjudication.get("amount_evidence"), dict) else {}
        judgement_evidence = adjudication.get("judgement_evidence") if isinstance(adjudication.get("judgement_evidence"), dict) else {}
        metadata.update(
            {
                "ai_review_status": status,
                "ai_review_confidence": confidence,
                "ai_review_reason": reason,
                "ai_review_decision": decision,
                "ai_review_model": model,
            }
        )
        if guardrail:
            metadata["ai_review_guardrail"] = guardrail
        if decision == "rewrite" and rewrite:
            metadata["ai_review_original_issue"] = original.issue

        export_row = {
            "Finding ID": finding_id,
            "Category": str(candidate.get("category", "")),
            "Original severity": str(candidate.get("severity", "")),
            "Decision": decision.title(),
            "Revised severity": revised_severity,
            "AI status": status,
            "AI confidence": confidence,
            "Page reference": str(candidate.get("page_reference", "")),
            "Note reference": str(candidate.get("note_reference", "")),
            "Issue": str(candidate.get("issue", "")),
            "Reason": reason,
            "Recommended action": action,
            "Rewrite": rewrite,
            "Guardrail applied": guardrail,
            "Amount evidence page": _evidence_value(amount_evidence, "page_number"),
            "Amount evidence statement/note": _evidence_value(amount_evidence, "statement_or_note_name"),
            "Reported amount": _evidence_value(amount_evidence, "reported_amount"),
            "Expected amount": _evidence_value(amount_evidence, "expected_amount"),
            "Difference": _evidence_value(amount_evidence, "difference"),
            "Amount evidence": _evidence_value(amount_evidence, "evidence"),
            "Judgement evidence page": _evidence_value(judgement_evidence, "page_number"),
            "Judgement issue": _evidence_value(judgement_evidence, "issue"),
            "Judgement evidence": _evidence_value(judgement_evidence, "evidence"),
            "Judgement recommendation": _evidence_value(judgement_evidence, "recommendation"),
            "Judgement severity": _evidence_value(judgement_evidence, "severity"),
            "Judgement confidence": _evidence_value(judgement_evidence, "confidence"),
            "Page snippet": str(candidate.get("page_snippet", "")),
            "Note snippet": str(candidate.get("note_snippet", "")),
        }
        export_rows.append(export_row)

        if decision == "suppress" and _can_suppress_finding(original, confidence):
            suppressed_indexes.add(candidate["index"])
            suppressed_rows.append(export_row)
            continue

        if decision == "downgrade":
            target_severity = _downgraded_severity(original.severity, revised_severity)
        elif decision == "rewrite":
            target_severity = _cap_revised_severity(original.severity, revised_severity)
        else:
            target_severity = original.severity

        updated_findings[candidate["index"]] = Finding(
            category=original.category,
            severity=target_severity,
            location=original.location,
            issue=rewrite if decision == "rewrite" and rewrite else original.issue,
            evidence=original.evidence,
            recommendation=action or original.recommendation,
            metadata=metadata,
        )

    final_findings = [finding for idx, finding in enumerate(updated_findings) if idx not in suppressed_indexes]
    summary = str(parsed.get("summary", "") or "").strip()
    evidence_rows = []
    if source_pdf_evidence_row:
        evidence_rows.append(source_pdf_evidence_row)
    evidence_rows.extend(_candidate_evidence_rows(candidates))
    return AiFindingReviewResult(
        findings=final_findings,
        export_rows=export_rows,
        summary=summary,
        status="completed",
        model=model,
        reviewed_count=len(candidates),
        suppressed_count=len(suppressed_indexes),
        evidence_rows=evidence_rows,
        suppressed_rows=suppressed_rows,
    )


def _normalize_ai_decision(value: object) -> str:
    decision = str(value or "keep").strip().lower()
    if decision == "remove":
        return "suppress"
    if decision in {"keep", "downgrade", "suppress", "rewrite"}:
        return decision
    return "keep"


def _severity_rank(value: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(_normalize_severity(value), 1)


def _cap_revised_severity(original: str, revised: str) -> str:
    original_norm = _normalize_severity(original)
    revised_norm = _normalize_severity(revised)
    return original_norm if _severity_rank(revised_norm) > _severity_rank(original_norm) else revised_norm


def _downgraded_severity(original: str, revised: str | None = None) -> str:
    original_norm = _normalize_severity(original)
    if revised:
        revised_norm = _cap_revised_severity(original_norm, revised)
        if _severity_rank(revised_norm) < _severity_rank(original_norm):
            return revised_norm
    if original_norm == "High":
        return "Medium"
    if original_norm == "Medium":
        return "Low"
    return "Low"


def _append_reason(reason: str, addition: str) -> str:
    if not addition:
        return reason
    if not reason:
        return addition
    if addition in reason:
        return reason
    return f"{reason} {addition}"


def _candidate_is_amount_related(candidate: dict[str, Any]) -> bool:
    category = str(candidate.get("category", "") or "").lower()
    if category in {"totals and rounding", "key amount consistency", "value added statement"}:
        return True
    combined = " ".join(
        str(candidate.get(key, "") or "")
        for key in ("issue", "evidence", "recommendation", "reason", "check_type", "line_item", "statement")
    ).lower()
    amount_context = candidate.get("amount_context")
    if isinstance(amount_context, dict) and amount_context:
        return True
    return any(
        term in combined
        for term in (
            "amount",
            "reported",
            "expected",
            "difference",
            "visible sum",
            "subtotal",
            "total",
            "casting",
            "cast",
            "current-year",
            "current year",
            "prior-year",
            "prior year",
            "cash flow",
            "statement of cash flows",
            "does not agree",
            "not located in referenced note",
        )
    )


def _adjudication_has_amount_evidence(adjudication: dict[str, Any]) -> bool:
    evidence = adjudication.get("amount_evidence") if isinstance(adjudication, dict) else None
    if not isinstance(evidence, dict):
        return False
    return all(
        _is_meaningful_evidence_value(evidence.get(key))
        for key in ("page_number", "statement_or_note_name", "reported_amount", "expected_amount", "difference", "evidence")
    )


def _adjudication_has_judgement_evidence(adjudication: dict[str, Any]) -> bool:
    evidence = adjudication.get("judgement_evidence") if isinstance(adjudication, dict) else None
    if not isinstance(evidence, dict):
        return False
    return all(
        _is_meaningful_evidence_value(evidence.get(key))
        for key in ("page_number", "issue", "evidence", "recommendation", "severity", "confidence")
    )


def _is_meaningful_evidence_value(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"n/a", "na", "none", "unknown", "not applicable", "not available"}


def _evidence_value(evidence: dict[str, Any], key: str) -> str:
    return str(evidence.get(key, "") or "").strip() if isinstance(evidence, dict) else ""

def _ai_reason_indicates_likely_false_positive(reason: str, status: str = "") -> bool:
    combined = f"{status} {reason}".lower()
    return any(
        phrase in combined
        for phrase in (
            "likely false positive",
            "false positive",
            "extraction drift",
            "not supported by robust evidence",
            "not supported by the supplied evidence",
        )
    )


def _ai_reason_indicates_unsupported(reason: str, status: str = "") -> bool:
    combined = f"{status} {reason}".lower()
    if _ai_reason_indicates_likely_false_positive(reason, status):
        return True
    return any(
        phrase in combined
        for phrase in (
            "insufficient evidence",
            "no evidence",
            "not visible",
            "not found in the supplied",
            "not found in the provided",
            "not found in the page",
            "not found in the note",
            "no repeated word",
            "no deterministic table",
            "cannot confirm",
            "ambiguous",
            "reviewer attention",
            "manual confirmation",
            "confirm if this is",
            "apparent contradiction",
        )
    )


def _has_strong_narrative_contradiction_evidence(finding: Finding) -> bool:
    if finding.category != "Narrative consistency":
        return False
    evidence = str(finding.evidence or "").lower()
    issue = str(finding.issue or "").lower()
    return (
        "nil" in issue
        and "non-zero" in issue
        and "nil" in evidence
        and "non-zero amount detected in same note" in evidence
    )


def _can_suppress_finding(finding: Finding, confidence: str) -> bool:
    if _has_strong_narrative_contradiction_evidence(finding):
        return False
    if finding.severity == "Low":
        return True
    if finding.severity == "Medium" and confidence == "High":
        return True
    if finding.severity == "High" and confidence == "High":
        metadata = finding.metadata or {}
        return bool(metadata.get("ocr_review"))
    return False




def _suppression_conflicts_with_note_evidence(finding: Finding, candidate: dict[str, Any], reason: str) -> bool:
    if finding.category != "Notes agreement":
        return False
    combined = " ".join(
        str(value or "")
        for value in (
            finding.issue,
            finding.evidence,
            finding.recommendation,
            candidate.get("issue"),
            candidate.get("note_reference"),
            candidate.get("note_snippet"),
            reason,
        )
    ).lower()
    note_context = any(term in combined for term in ("note", "referenced", "reference", "heading"))
    mismatch_context = any(
        term in combined
        for term in (
            "not found",
            "not located",
            "does not contain",
            "doesn't contain",
            "no amount",
            "no mention",
            "only contains",
            "wrong note",
            "mismatch",
            "stronger match",
            "appears in another note",
            "linking issue",
            "reference issue",
        )
    )
    extraction_noise = any(
        term in combined
        for term in (
            "ocr drift",
            "split-digit",
            "split digit",
            "layout noise",
            "extraction noise",
            "table extraction",
            "merged cell",
        )
    )
    return note_context and mismatch_context and not extraction_noise


def _finding_page_reference(document: PdfDocument, finding: Finding, metadata: dict[str, Any]) -> str:
    for key in ("page_reference", "page", "pages"):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return _normalize_page_reference(document, value)
    match = re.search(r"Pages?\s+[0-9,\- ]+", finding.location)
    if match:
        return _normalize_page_reference(document, match.group(0).strip())
    return _normalize_page_reference(document, finding.location) if "Page" in finding.location else ""


def _finding_note_reference(finding: Finding, metadata: dict[str, Any]) -> str:
    for key in ("note_reference", "referenced_note", "note"):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return value
    match = re.search(r"Note\s+(\d+[A-Za-z]?)", f"{finding.location} {finding.issue}", re.I)
    return f"Note {match.group(1)}" if match else ""


def _candidate_page_numbers(document: PdfDocument, page_reference: str) -> list[int]:
    reviewer_pages = _numbers_from_page_reference(page_reference)
    if not reviewer_pages:
        return []
    mapping = _reviewer_to_physical_page_map(document)
    pages: list[int] = []
    for reviewer_page in reviewer_pages:
        physical = mapping.get(reviewer_page)
        if physical is not None:
            pages.append(physical)
        elif 1 <= reviewer_page <= len(document.pages):
            pages.append(reviewer_page)
    return sorted(dict.fromkeys(pages))


def _numbers_from_page_reference(text: str) -> list[int]:
    if not text:
        return []
    values: list[int] = []
    for start, end in re.findall(r"(\d+)\s*-\s*(\d+)", text):
        a, b = int(start), int(end)
        if a <= b:
            values.extend(range(a, b + 1))
    singles = [int(token) for token in re.findall(r"\b\d+\b", text)]
    if values:
        return sorted(dict.fromkeys(values))
    return sorted(dict.fromkeys(singles))


def _reviewer_to_physical_page_map(document: PdfDocument) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for page in document.pages:
        printed = _printed_footer_page_number(page.text)
        if printed is not None and printed not in mapping:
            mapping[printed] = page.number
    return mapping


def _physical_to_reviewer_page_map(document: PdfDocument) -> dict[int, int]:
    return {physical: reviewer for reviewer, physical in _reviewer_to_physical_page_map(document).items()}


def _normalize_page_reference(document: PdfDocument, text: str) -> str:
    if not text:
        return ""
    raw = str(text).strip()
    physical_to_reviewer = _physical_to_reviewer_page_map(document)
    reviewer_to_physical = _reviewer_to_physical_page_map(document)

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        body = match.group(2)
        translated: list[str] = []
        for token in re.split(r"(\D+)", body):
            if token.isdigit():
                page_number = int(token)
                if page_number in physical_to_reviewer:
                    translated.append(str(physical_to_reviewer[page_number]))
                elif page_number in reviewer_to_physical:
                    translated.append(str(page_number))
                else:
                    translated.append(str(page_number))
            else:
                translated.append(token)
        return prefix + "".join(translated)

    return re.sub(r"\b(Pages?\s+)([\d,\-\sand]+)", repl, raw, flags=re.I)


def _printed_footer_page_number(text: str) -> int | None:
    if not text:
        return None
    non_empty_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return None
    for line in reversed(non_empty_lines[-12:]):
        cleaned = line.strip(" -")
        if re.fullmatch(r"\d{1,3}", cleaned):
            return int(cleaned)
    return None


def _candidate_keywords(finding: Finding, line_item: str, statement: str) -> list[str]:
    seeds = [line_item, statement, finding.issue, finding.location]
    keywords: list[str] = []
    for seed in seeds:
        if not seed:
            continue
        normalized = re.sub(r"[^A-Za-z0-9 ]+", " ", seed).strip().lower()
        for token in normalized.split():
            if len(token) >= 4 and token not in {"page", "note", "does", "from", "with", "that", "this", "column", "review", "possible", "detected"}:
                keywords.append(token)
    return list(dict.fromkeys(keywords))[:8]


def _page_snippet(document: PdfDocument, page_numbers: list[int], keywords: list[str], note_reference: str) -> str:
    if not page_numbers:
        return ""
    collected: list[str] = []
    note_token = note_reference.lower().replace("note ", "") if note_reference else ""
    for number in page_numbers[:2]:
        page = next((p for p in document.pages if p.number == number), None)
        if not page or not page.text.strip():
            continue
        lines = [re.sub(r"\s+", " ", line).strip() for line in page.text.splitlines() if line.strip()]
        matched = []
        for line in lines:
            lower = line.lower()
            if note_token and re.search(rf"\b{re.escape(note_token)}\b", lower):
                matched.append(line)
                continue
            if any(keyword in lower for keyword in keywords):
                matched.append(line)
        snippet_lines = matched[:6] if matched else lines[:6]
        snippet = " ".join(snippet_lines)
        if snippet:
            label = _physical_to_reviewer_page_map(document).get(number, number)
            collected.append(f"Page {label}: {snippet[:700]}")
    return " | ".join(collected)


def _note_snippet(document: PdfDocument, note_reference: str, keywords: list[str]) -> str:
    if not note_reference:
        return ""
    ref = note_reference.lower().replace("note ", "").strip()
    for page in document.pages:
        lines = [re.sub(r"\s+", " ", line).strip() for line in page.text.splitlines() if line.strip()]
        capture = False
        snippet_lines: list[str] = []
        for line in lines:
            heading = NOTE_HEADING_RE.match(line)
            if heading:
                current_ref = heading.group(1).lower()
                if current_ref == ref:
                    capture = True
                    snippet_lines = [line]
                    continue
                if capture:
                    break
            if capture:
                snippet_lines.append(line)
                if len(" ".join(snippet_lines)) > 700:
                    break
        if snippet_lines:
            text = " ".join(snippet_lines)
            return text[:700]
    for page in document.pages:
        text = re.sub(r"\s+", " ", page.text)
        if re.search(rf"\bnote\s+{re.escape(ref)}\b|\b{re.escape(ref)}\b", text, re.I):
            return text[:700]
    return ""


def _document_scope(document: PdfDocument) -> str:
    if not document.pages:
        return "empty"
    lower = document.text.lower()
    if "limited-scope statement extract" in lower:
        return "Limited-scope statement extract"
    if len(document.pages) == 1:
        return "Single-page upload"
    return "Full financial statement or mixed upload"
