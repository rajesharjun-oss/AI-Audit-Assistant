from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from models import CompanyProfile, Finding, PdfDocument


NOTES_HEADING_PATTERNS = (
    "notes to the financial statements",
    "notes to financial statements",
    "notes forming part of the financial statements",
    "notes to the accounts",
)

POLICY_KEYWORDS = (
    "significant accounting policies",
    "material accounting policies",
    "basis of preparation",
    "revenue from contracts with customers",
    "financial instruments",
    "taxation",
    "property, plant and equipment",
    "intangible assets",
    "lease liability",
    "right-of-use",
    "related party",
    "events after the reporting period",
)


@dataclass(frozen=True)
class AiPolicyReviewResult:
    findings: list[Finding]
    export_rows: list[dict[str, str]]
    summary: str
    status: str
    model: str
    message: str = ""


def run_ai_policy_review(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool] | None = None,
    model: str = "gpt-5-mini",
) -> AiPolicyReviewResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AiPolicyReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="unavailable",
            model=model,
            message="AI policy and standards judgement skipped because OPENAI_API_KEY is not configured.",
        )

    note_context = _policy_context(note_sections, document)
    if not note_context.strip():
        return AiPolicyReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="skipped",
            model=model,
            message="AI policy and standards judgement skipped because no policy/disclosure note context was detected.",
        )

    payload = {
        "model": model,
        "max_output_tokens": 1200,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are reviewing a financial statement for policy relevance, disclosure completeness, "
                    "standards context, and industry fit. Return strict JSON only. Be conservative. "
                    "Do not invent missing evidence. If evidence is weak, use Low confidence and prefer review prompts. "
                    "Ignore generic standards/amendments text unless the report clearly applies the standard in current accounting policies or balances."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(document, profile, note_context, policy_map or {}),
            },
        ],
    }

    try:
        response_json = _call_openai(api_key, payload)
        parsed = _parse_response_json(response_json)
        findings, export_rows = _rows_to_outputs(parsed.get("observations", []))
        summary = str(parsed.get("summary", "") or "").strip()
        return AiPolicyReviewResult(
            findings=findings,
            export_rows=export_rows,
            summary=summary,
            status="completed",
            model=model,
        )
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        friendly_message = _friendly_ai_error_message(exc)
        return AiPolicyReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="deferred" if _is_rate_limit_error(exc) else "error",
            model=model,
            message=friendly_message,
        )


def _build_prompt(
    document: PdfDocument,
    profile: CompanyProfile,
    note_context: str,
    policy_map: dict[str, bool],
) -> str:
    doc_text = document.text
    detected_indicators = []
    for key, present in sorted(policy_map.items()):
        if present:
            detected_indicators.append(key)
    company_name = profile.company_name or "Not specified"
    industry = profile.industry or "Auto-detect from context"
    currency = profile.reporting_currency or "Not specified"
    framework = profile.presentation_standard or "IFRS"
    expected_policies = ", ".join(profile.expected_policies) if profile.expected_policies else "None provided"
    significant_transactions = ", ".join(profile.significant_transactions) if profile.significant_transactions else "None provided"
    checklist_areas = ", ".join(profile.checklist_areas) if profile.checklist_areas else "None forced"
    disclosure_context = _keyword_evidence(doc_text)
    return (
        "Review the accounting policies and related disclosures.\n\n"
        "Return JSON with this shape:\n"
        "{"
        '"summary":"short reviewer summary",'
        '"observations":[{'
        '"title":"...",'
        '"dimension":"policy_relevance|disclosure_completeness|standard_context|industry_fit",'
        '"standard_or_topic":"...",'
        '"severity":"High|Medium|Low",'
        '"confidence":"High|Medium|Low",'
        '"status":"exception|review_prompt|ok",'
        '"issue":"...",'
        '"rationale":"...",'
        '"recommendation":"...",'
        '"page_reference":"Page X or Pages X-Y if visible",'
        '"note_reference":"Note X if visible",'
        '"evidence_snippet":"short quote/paraphrase from provided text"'
        "}]}.\n\n"
        "Rules:\n"
        "1. Use High only when the wording clearly conflicts with the entity, transaction, or current standard context.\n"
        "2. Use Medium or Low for judgement prompts.\n"
        "3. If a policy is present and appears tailored, you may return status=ok.\n"
        "4. If disclosure wording is generic but not clearly wrong, prefer review_prompt.\n"
        "5. Consider industry fit when judging whether a paragraph represents the entity's business.\n"
        "6. Limit output to at most 6 observations.\n\n"
        f"Company profile:\n- Company name: {company_name}\n- Industry: {industry}\n- Reporting currency: {currency}\n"
        f"- Framework: {framework}\n- Expected policies: {expected_policies}\n- Significant transactions: {significant_transactions}\n"
        f"- Forced checklist areas: {checklist_areas}\n- Deterministically detected policy areas: {', '.join(detected_indicators) or 'None'}\n\n"
        f"Disclosure evidence snippets:\n{disclosure_context}\n\n"
        f"Policies and note context:\n{note_context}"
    )


def _call_openai(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/responses"
    last_error: Exception | None = None
    for attempt in range(3):
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < 2:
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After", ""))
                time.sleep(retry_after if retry_after is not None else (2 + attempt * 3))
                last_error = RuntimeError(f"OpenAI API error {exc.code}: {body[:240]}")
                continue
            raise RuntimeError(f"OpenAI API error {exc.code}: {body[:240]}") from exc
        except Exception as exc:
            last_error = exc
            break
    if last_error:
        raise last_error
    raise RuntimeError("OpenAI API call failed without a specific error.")


def _parse_response_json(response_json: dict[str, Any]) -> dict[str, Any]:
    text = _extract_response_text(response_json).strip()
    if not text:
        raise RuntimeError("OpenAI response did not contain text output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise RuntimeError("OpenAI response was not valid JSON.")
        return json.loads(match.group(0))


def _extract_response_text(response_json: dict[str, Any]) -> str:
    collected: list[str] = []
    _append_text_fragments(collected, response_json.get("output_text"))
    for item in response_json.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        _append_text_fragments(collected, item.get("content"))
    if not collected:
        for key in ("text", "content", "output"):
            _append_text_fragments(collected, response_json.get(key))
    return "".join(fragment for fragment in collected if fragment)


def _append_text_fragments(collected: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        collected.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _append_text_fragments(collected, item)
        return
    if not isinstance(value, dict):
        return

    content_type = str(value.get("type", "") or "").strip().lower()
    if content_type in {"output_text", "text", "input_text"}:
        text_value = value.get("text")
        if isinstance(text_value, str):
            collected.append(text_value)
            return
        if isinstance(text_value, dict):
            nested = text_value.get("value")
            if isinstance(nested, str):
                collected.append(nested)
                return
    if "value" in value and isinstance(value.get("value"), str):
        collected.append(str(value["value"]))
    for key in ("text", "content", "output", "value"):
        nested = value.get(key)
        if nested is not None and nested is not value:
            _append_text_fragments(collected, nested)


def _rows_to_outputs(observations: list[dict[str, Any]]) -> tuple[list[Finding], list[dict[str, str]]]:
    findings: list[Finding] = []
    export_rows: list[dict[str, str]] = []
    for observation in observations[:6]:
        if not isinstance(observation, dict):
            continue
        title = str(observation.get("title", "") or "").strip()
        dimension = str(observation.get("dimension", "") or "").strip() or "policy_relevance"
        standard_or_topic = str(observation.get("standard_or_topic", "") or "").strip()
        severity = _normalize_severity(observation.get("severity"))
        confidence = _normalize_confidence(observation.get("confidence"))
        status = str(observation.get("status", "") or "").strip() or "review_prompt"
        issue = str(observation.get("issue", "") or "").strip()
        rationale = str(observation.get("rationale", "") or "").strip()
        recommendation = str(observation.get("recommendation", "") or "").strip()
        page_reference = str(observation.get("page_reference", "") or "").strip() or "Document-wide"
        note_reference = str(observation.get("note_reference", "") or "").strip()
        evidence_snippet = str(observation.get("evidence_snippet", "") or "").strip()
        if not issue and not title:
            continue
        export_rows.append(
            {
                "Title": title or issue[:80],
                "Dimension": dimension,
                "Standard / topic": standard_or_topic,
                "Severity": severity,
                "Confidence": confidence,
                "Status": status,
                "Page reference": page_reference,
                "Note reference": note_reference,
                "Issue": issue,
                "Rationale": rationale,
                "Evidence snippet": evidence_snippet,
                "Recommendation": recommendation,
            }
        )
        if status == "ok":
            continue
        findings.append(
            Finding(
                category="AI policy judgement",
                severity=severity,
                location=page_reference,
                issue=issue or title,
                evidence=evidence_snippet or rationale or "AI judgement did not return a supporting snippet.",
                recommendation=recommendation or "Review the referenced policy/disclosure manually.",
                metadata={
                    "dimension": dimension,
                    "standard_or_topic": standard_or_topic,
                    "match_confidence": confidence,
                    "note_reference": note_reference,
                    "reason": rationale,
                    "check_type": "AI policy judgement",
                },
            )
        )
    return findings, export_rows


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().title()
    return text if text in {"High", "Medium", "Low"} else "Low"


def _normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().title()
    return text if text in {"High", "Medium", "Low"} else "Low"


def _policy_context(note_sections: dict[str, str], document: PdfDocument) -> str:
    note_lines: list[str] = []
    for ref, text in sorted(note_sections.items(), key=lambda item: _sort_note_key(item[0])):
        clean = str(text or "").strip()
        if not clean:
            continue
        heading = clean.splitlines()[0].strip()
        lower = clean.lower()
        if any(marker in lower[:300] for marker in ("value added statement", "five year financial summary")):
            continue
        if ref in {"1", "2"} or any(keyword in lower for keyword in POLICY_KEYWORDS):
            note_lines.append(f"Note {ref}\n{clean[:2200]}")
    if note_lines:
        return "\n\n".join(note_lines[:6])
    fallback_pages = []
    for page in document.pages:
        lower = page.text.lower()
        if any(pattern in lower for pattern in NOTES_HEADING_PATTERNS) or "significant accounting policies" in lower:
            fallback_pages.append(f"Page {page.number}\n{page.text[:2200]}")
    return "\n\n".join(fallback_pages[:3])


def _keyword_evidence(text: str) -> str:
    snippets: list[str] = []
    lower = text.lower()
    for keyword in POLICY_KEYWORDS:
        index = lower.find(keyword)
        if index == -1:
            continue
        start = max(0, index - 80)
        end = min(len(text), index + 180)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        snippets.append(f"- {keyword}: {snippet}")
        if len(snippets) >= 8:
            break
    return "\n".join(snippets) or "- No focused policy snippets were extracted."


def _retry_after_seconds(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return max(1, min(int(text), 12))
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "429" in text or "rate limit" in text or "rate exceeded" in text or "too many requests" in text


def _friendly_ai_error_message(exc: Exception) -> str:
    if _is_rate_limit_error(exc):
        return (
            "AI policy and standards judgement was deferred because the AI service is busy right now. "
            "The core deterministic review still completed. Please retry in a minute, ideally one file at a time."
        )
    return f"AI policy and standards judgement could not be completed: {exc}"


def _sort_note_key(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", str(value))
    if not match:
        return (999, str(value))
    return (int(match.group(1)), str(value))
