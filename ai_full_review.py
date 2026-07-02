from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_policy_review import (
    _call_openai,
    _friendly_ai_error_message,
    _normalize_confidence,
    _normalize_severity,
    _parse_response_json,
    _repair_response_json,
)
from models import DEFAULT_AI_MODEL, CompanyProfile, Finding, PdfDocument


@dataclass(frozen=True)
class AiFullReviewResult:
    findings: list[Finding]
    export_rows: list[dict[str, str]]
    summary: str
    status: str
    model: str
    message: str = ""
    evidence_rows: list[dict[str, str]] | None = None


_FULL_REVIEW_OUTPUT_TOKENS = 2800
_FULL_REVIEW_REPAIR_SHAPE = (
    '{"summary":"short reviewer summary","observations":[{"title":"...","section_or_statement":"...",'
    '"dimension":"policy_relevance | disclosure_completeness | standard_context | industry_fit | grammar | presentation | notes_and_reconciliation",'
    '"standard_or_topic":"...","severity":"High|Medium|Low","confidence":"High|Medium|Low",'
    '"status":"exception|review_prompt|ok","issue":"...","rationale":"...",'
    '"expected_fix":"...","recommendation":"...","page_reference":"Page X",'
    '"note_reference":"Note X","evidence_snippet":"..."}]}'
)


def run_ai_full_review(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool] | None = None,
    model: str = DEFAULT_AI_MODEL,
) -> AiFullReviewResult:
    api_key = __import__("os").environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AiFullReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="unavailable",
            model=model,
            message="AI full review was skipped because OPENAI_API_KEY is not configured.",
            evidence_rows=[],
        )

    statement_text = _compact(document.text)
    if not statement_text:
        return AiFullReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="skipped",
            model=model,
            message="AI full review was skipped because no readable PDF text was extracted.",
            evidence_rows=[],
        )

    note_context = _note_context_lines(note_sections)
    policy_context = _policy_context_lines(policy_map or {})
    payload = {
        "model": model,
        "max_output_tokens": _FULL_REVIEW_OUTPUT_TOKENS,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial-statement auditor assistant reviewing a drafted financial statement for consistency and quality. "
                    "Use only the supplied context. Be conservative and include findings only when evidence is explicit in the provided text. "
                    "If confidence is weak, use review_prompt and never use status=ok as an exception. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(document, profile, statement_text, note_context, policy_context),
            },
        ],
    }

    try:
        response_json = _call_openai(api_key, payload)
        try:
            parsed = _parse_response_json(response_json)
        except Exception as parse_exc:
            parsed = _repair_response_json(api_key, model, str(parse_exc), _FULL_REVIEW_REPAIR_SHAPE)
        findings, export_rows = _rows_to_outputs(parsed.get("observations", []))
        summary = str(parsed.get("summary", "") or "").strip()
        return AiFullReviewResult(
            findings=findings,
            export_rows=export_rows,
            summary=summary,
            status="completed",
            model=model,
            evidence_rows=[
                {
                    "Evidence type": "Full AI review prompt",
                    "Source page": "Document extracted pages",
                    "Model": model,
                    "Context snippet": statement_text[:350],
                }
            ],
        )
    except Exception as exc:
        return AiFullReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="deferred" if _is_rate_limit_related(exc) else "error",
            model=model,
            message=_friendly_ai_error_message(exc),
            evidence_rows=[],
        )


def _build_prompt(document: PdfDocument, profile: CompanyProfile, statement_text: str, note_context: str, policy_context: str) -> str:
    company = profile.company_name or "Not specified"
    industry = profile.industry or "Not specified"
    currency = profile.reporting_currency or "Not specified"
    framework = profile.presentation_standard or "IFRS"
    expected = ", ".join(profile.expected_policies) if profile.expected_policies else "Not explicitly provided"
    significant_transactions = ", ".join(profile.significant_transactions) if profile.significant_transactions else "Not explicitly provided"

    return (
        f"You are reviewing a financial-statement audit draft for company '{company}' (industry: {industry}, currency marker: {currency}, framework: {framework}).\n\n"
        f"Company setup context: expected policies={expected}; expected significant transactions={significant_transactions}.\n"
        f"Detected policy topics from explicit disclosures: {policy_context or 'None identified'}\n"
        "Scope:\n"
        "1) Check note-reference consistency between face statements and note disclosures.\n"
        "2) Check major presentation issues, spelling/grammar defects, wording, and drafting consistency.\n"
        "3) Review disclosure completeness relative to obvious amounts/flows/standards context in evidence.\n"
        "4) Report likely mismatches in sections like accounting policies, standards references, and reconciliation narratives.\n\n"
        "Expected output JSON shape:\n"
        f"{_FULL_REVIEW_REPAIR_SHAPE}\n\n"
        "Observations should use status values as follows:\n"
        "- exception: confirmed issue to report as a finding\n"
        "- review_prompt: likely issue needing reviewer confirmation\n"
        "- ok: no action required\n\n"
        "Return concise, specific, and page-referenced items.\n\n"
        "Document sections/notes available:\n"
        f"{note_context or 'No clear note headings detected in parsed context.'}\n\n"
        "Extracted statement/context:\n"
        f"{statement_text}\n"
    )


def _rows_to_outputs(observations: list[dict[str, Any]]) -> tuple[list[Finding], list[dict[str, str]]]:
    findings: list[Finding] = []
    export_rows: list[dict[str, str]] = []

    for observation in observations[:30]:
        status = str(observation.get("status", "") or "").strip().lower()
        if status == "ok":
            continue
        severity = _normalize_severity(observation.get("severity", "low"))
        confidence = _normalize_confidence(observation.get("confidence", "Low"))
        dimension = str(observation.get("dimension", "") or "").strip()
        title = str(observation.get("title", "") or "Potential issue")
        issue = str(observation.get("issue", "") or "").strip() or title
        section_or_statement = str(observation.get("section_or_statement", "") or "").strip()
        rationale = str(observation.get("rationale", "") or "").strip()
        recommendation = str(observation.get("recommendation", "") or "").strip()
        expected_fix = str(observation.get("expected_fix", "") or "").strip()
        page_reference = str(observation.get("page_reference", "") or "").strip()
        note_reference = str(observation.get("note_reference", "") or "").strip()
        evidence_snippet = str(observation.get("evidence_snippet", "") or "").strip()
        standard_or_topic = str(observation.get("standard_or_topic", "") or "").strip()

        findings.append(
            Finding(
                category="AI full review",
                severity=severity,
                location=section_or_statement or page_reference or "Document-wide",
                issue=issue,
                evidence=(f"Dimension: {dimension}. {rationale}" if rationale else title).strip(),
                recommendation=recommendation or expected_fix or "Review the source section and confirm whether the observation is consistent with the note text.",
                metadata={
                    "check_type": "full_review",
                    "ai_review_status": "confirmed_exception" if status == "exception" else "review_prompt" if status == "review_prompt" else "",
                    "match_confidence": confidence,
                    "page_reference": page_reference,
                    "note_reference": note_reference,
                    "standard_or_topic": standard_or_topic,
                    "dimension": dimension,
                    "section_or_statement": section_or_statement,
                    "rationale": rationale,
                    "expected_fix": expected_fix,
                    "evidence_snippet": evidence_snippet,
                    "ai_review_confidence": confidence,
                    "ai_review_reason": rationale,
                    "line_item": title,
                },
            )
        )

        export_rows.append(
            {
                "Title": title,
                "Section": section_or_statement,
                "Dimension": dimension,
                "Status": status or "exception",
                "Severity": severity,
                "Confidence": confidence,
                "Page reference": page_reference,
                "Note reference": note_reference,
                "Standard/topic": standard_or_topic,
                "Issue": issue,
                "Recommendation": recommendation,
                "Rationale": rationale,
                "Evidence": evidence_snippet,
            }
        )
    return findings, export_rows


def _note_context_lines(note_sections: dict[str, str]) -> str:
    if not note_sections:
        return ""
    pairs = list(note_sections.items())[:20]
    return "\n".join(f"Note {note}: {heading}" for note, heading in pairs)


def _policy_context_lines(policy_map: dict[str, bool]) -> str:
    detected = [key for key, value in sorted(policy_map.items()) if value]
    return ", ".join(detected)


def _compact(text: str, limit: int = 12000) -> str:
    compacted = " ".join(str(text or "").split())
    return compacted[:limit]


def _is_rate_limit_related(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate-limit",
            "rate exceeded",
            "too many requests",
            "busy",
            "service busy",
            "temporarily busy",
            "cooldown",
            "timed out",
            "timeout",
        )
    )
