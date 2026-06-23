from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from ai_policy_review import _call_openai, _friendly_ai_error_message, _normalize_confidence, _normalize_severity, _parse_response_json
from models import CompanyProfile, Finding, PdfDocument


REVIEWABLE_CATEGORIES = {
    "Totals and rounding",
    "Notes agreement",
    "Formatting",
    "Name consistency",
    "Date consistency",
    "Grammar review",
    "Value Added Statement",
}
SKIPPED_CATEGORIES = {"Extraction quality", "Document scope", "AI policy judgement"}
MAX_REVIEW_FINDINGS = 12


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


def run_ai_finding_review(
    document: PdfDocument,
    profile: CompanyProfile,
    findings: list[Finding],
    model: str = "gpt-5-mini",
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
        )

    candidates = _review_candidates(findings)
    if not candidates:
        return AiFindingReviewResult(
            findings=findings,
            export_rows=[],
            summary="",
            status="skipped",
            model=model,
            message="AI finding review skipped because no weak deterministic findings were eligible for adjudication.",
        )

    payload = {
        "model": model,
        "max_output_tokens": 1400,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are assisting a deterministic financial-statement review engine. "
                    "You must be conservative. Do not invent evidence. "
                    "Only suppress a finding when it looks like a likely false positive from extraction, "
                    "layout noise, duplicated context, or weak contextual linkage. "
                    "Prefer keep or downgrade when evidence is mixed. "
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
        )


def _review_candidates(findings: list[Finding]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if finding.severity not in {"Low", "Medium"}:
            continue
        if finding.category in SKIPPED_CATEGORIES:
            continue
        if REVIEWABLE_CATEGORIES and finding.category not in REVIEWABLE_CATEGORIES:
            continue
        metadata = finding.metadata or {}
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
                "page_reference": metadata.get("page_reference", ""),
                "note_reference": metadata.get("note_reference", metadata.get("referenced_note", "")),
                "statement": metadata.get("statement", ""),
                "line_item": metadata.get("line_item", ""),
                "reason": metadata.get("reason", ""),
            }
        )
        if len(candidates) >= MAX_REVIEW_FINDINGS:
            break
    return candidates


def _build_prompt(document: PdfDocument, profile: CompanyProfile, candidates: list[dict[str, Any]]) -> str:
    company_name = profile.company_name or "Not specified"
    industry = profile.industry or "Auto-detect from context"
    return (
        "Review these weak deterministic findings and decide whether each should stay, be downgraded to a review prompt, "
        "or be suppressed as a likely false positive.\n\n"
        "Return JSON with this shape:\n"
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
        "}]}.\n\n"
        "Rules:\n"
        "1. Be conservative.\n"
        "2. Suppress only when the finding is likely a false positive.\n"
        "3. Medium findings should normally stay as Medium or be downgraded to Low review prompts; do not promote severity.\n"
        "4. Low findings may be suppressed when evidence is weak or context is clearly noisy.\n"
        "5. Ignore categories not provided in the candidate list.\n\n"
        f"Document context:\n- Company: {company_name}\n- Industry: {industry}\n"
        f"- OCR used: {'Yes' if document.ocr_used else 'No'}\n"
        f"- Extraction confidence: {document.extraction_confidence}%\n"
        f"- Table extraction confidence: {document.table_extraction_confidence}%\n\n"
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
    candidate_by_id = {candidate["finding_id"]: candidate for candidate in candidates}
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
                "Page reference": candidate["location"],
                "Note reference": candidate["note_reference"],
                "Issue": candidate["issue"],
                "Reason": reason,
                "Recommended action": action,
            }
        )

        if decision == "suppress" and candidate["severity"] == "Low":
            suppressed_indexes.add(candidate["index"])
            continue

        if decision in {"downgrade", "suppress"}:
            downgraded = Finding(
                category=original.category,
                severity="Low" if candidate["severity"] == "Medium" else revised_severity,
                location=original.location,
                issue=original.issue,
                evidence=original.evidence,
                recommendation=action or original.recommendation,
                metadata=metadata,
            )
            updated_findings[candidate["index"]] = downgraded
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
    )
