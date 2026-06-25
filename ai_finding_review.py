from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from ai_policy_review import _call_openai, _friendly_ai_error_message, _normalize_confidence, _normalize_severity, _parse_response_json
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
MAX_REVIEW_FINDINGS = 24
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


def run_ai_finding_review(
    document: PdfDocument,
    profile: CompanyProfile,
    findings: list[Finding],
    model: str = DEFAULT_AI_MODEL,
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

    payload = {
        "model": model,
        "max_output_tokens": 2200,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are assisting a deterministic financial-statement review engine. "
                    "You must be conservative, evidence-bound, and page-grounded. "
                    "Do not invent evidence. Only use the page snippets, note snippets, issue text, and evidence text supplied. "
                    "Treat deterministic arithmetic and structure findings as primary evidence, but suppress likely false positives from OCR noise, split digits, duplicated headers, five-year-summary contamination, value-added contamination, weak note linkage, or layout extraction drift. "
                    "Do not suppress a finding simply because it is inconvenient. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(document, profile, candidates),
            },
        ],
    }

    try:
        response_json = _call_openai(api_key, payload)
        parsed = _parse_response_json(response_json)
        return _apply_adjudications(findings, candidates, parsed, model)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return AiFindingReviewResult(
            findings=findings,
            export_rows=[],
            summary="",
            status="deferred",
            model=model,
            message=_friendly_ai_error_message(exc),
            evidence_rows=_candidate_evidence_rows(candidates) if 'candidates' in locals() else [],
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
            }
        )
        if len(candidates) >= MAX_REVIEW_FINDINGS:
            break
    return candidates



def _candidate_evidence_rows(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidates:
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
                "AI role": "Classify as keep, downgrade, or likely false positive using supplied evidence only",
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


def _build_prompt(document: PdfDocument, profile: CompanyProfile, candidates: list[dict[str, Any]]) -> str:
    company_name = profile.company_name or "Not specified"
    industry = profile.industry or "Auto-detect from context"
    return (
        "Review these ambiguous engine findings and decide whether each should stay, be downgraded to a review prompt, or be suppressed as a likely false positive.\n\n"
        "Return JSON with this exact shape:\n"
        "{"
        '"summary":"short reviewer summary",'
        '"adjudications":[{'
        '"finding_id":"F1",'
        '"decision":"keep|downgrade|suppress",'
        '"revised_severity":"High|Medium|Low",'
        '"status":"confirmed_exception|review_prompt|likely_false_positive|insufficient_evidence",'
        '"confidence":"High|Medium|Low",'
        '"reason":"...",'
        '"recommended_action":"..."'
        "}]}\n\n"
        "Rules:\n"
        "1. Be conservative.\n"
        "2. Suppress only when the evidence strongly suggests extraction/layout noise or a weak contextual mismatch.\n"
        "3. If deterministic evidence looks structurally sound, keep the finding.\n"
        "4. Prefer downgrade when the finding may still matter but needs reviewer confirmation.\n"
        "5. Do not promote severity above the original severity.\n"
        "6. Use the supplied page and note snippets; do not infer missing facts.\n\n"
        f"Document context:\n- Company: {company_name}\n- Industry: {industry}\n"
        f"- OCR used: {'Yes' if document.ocr_used else 'No'}\n"
        f"- Extraction confidence: {document.extraction_confidence}%\n"
        f"- Table extraction confidence: {document.table_extraction_confidence}%\n"
        f"- Document scope: {_document_scope(document)}\n\n"
        "Candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=True, indent=2)}"
    )


def _apply_adjudications(
    findings: list[Finding],
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
    model: str,
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
    updated_findings = list(findings)
    suppressed_indexes: set[int] = set()

    for candidate in candidates:
        finding_id = candidate["finding_id"]
        adjudication = by_id.get(finding_id, {})
        decision = str(adjudication.get("decision", "") or "keep").strip().lower()
        revised_severity = _normalize_severity(adjudication.get("revised_severity") or candidate["severity"])
        confidence = _normalize_confidence(adjudication.get("confidence") or candidate["severity"])
        status = str(adjudication.get("status", "") or "review_prompt").strip() or "review_prompt"
        reason = str(adjudication.get("reason", "") or "").strip()
        action = str(adjudication.get("recommended_action", "") or "").strip()
        original = updated_findings[candidate["index"]]
        metadata = dict(original.metadata or {})
        if _has_strong_narrative_contradiction_evidence(original):
            decision = "keep"
            status = "confirmed_exception"
            confidence = "High" if confidence != "Low" else confidence
            if not reason:
                reason = "The finding is supported by an explicit nil-style narrative and a non-zero amount in the same note evidence."
        metadata.update(
            {
                "ai_review_status": status,
                "ai_review_confidence": confidence,
                "ai_review_reason": reason,
                "ai_review_decision": decision,
                "ai_review_model": model,
            }
        )
        export_rows.append(
            {
                "Finding ID": finding_id,
                "Category": candidate["category"],
                "Original severity": candidate["severity"],
                "Decision": decision.title(),
                "Revised severity": revised_severity,
                "AI status": status,
                "AI confidence": confidence,
                "Page reference": candidate["page_reference"],
                "Note reference": candidate["note_reference"],
                "Issue": candidate["issue"],
                "Reason": reason,
                "Recommended action": action,
                "Page snippet": candidate["page_snippet"],
                "Note snippet": candidate["note_snippet"],
            }
        )

        if decision == "suppress" and _can_suppress_finding(original, confidence):
            suppressed_indexes.add(candidate["index"])
            continue

        if decision in {"downgrade", "suppress"}:
            downgraded_severity = "Low" if original.severity == "Medium" else revised_severity
            if original.severity == "High":
                downgraded_severity = "Medium" if revised_severity == "High" else revised_severity
            updated_findings[candidate["index"]] = Finding(
                category=original.category,
                severity=downgraded_severity,
                location=original.location,
                issue=original.issue,
                evidence=original.evidence,
                recommendation=action or original.recommendation,
                metadata=metadata,
            )
            continue

        updated_findings[candidate["index"]] = Finding(
            category=original.category,
            severity=original.severity,
            location=original.location,
            issue=original.issue,
            evidence=original.evidence,
            recommendation=action or original.recommendation,
            metadata=metadata,
        )

    final_findings = [finding for idx, finding in enumerate(updated_findings) if idx not in suppressed_indexes]
    summary = str(parsed.get("summary", "") or "").strip()
    return AiFindingReviewResult(
        findings=final_findings,
        export_rows=export_rows,
        summary=summary,
        status="completed",
        model=model,
        reviewed_count=len(candidates),
        suppressed_count=len(suppressed_indexes),
        evidence_rows=_candidate_evidence_rows(candidates),
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
