from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from models import DEFAULT_AI_MODEL, CompanyProfile, Finding, PdfDocument


NOTES_HEADING_PATTERNS = (
    "notes to the financial statements",
    "notes to financial statements",
    "notes forming part of the financial statements",
    "notes to the accounts",
)

def _parse_retry_backoff_seconds() -> tuple[int, ...]:
    raw = os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "5,10,20,40,60")
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(float(text))
        except ValueError:
            continue
        if value > 0:
            values.append(max(1, min(value, 120)))
    return tuple(values or [5, 10, 20, 40])


AI_RETRY_BACKOFF_SECONDS = _parse_retry_backoff_seconds()
AI_RATE_LIMIT_COOLDOWN_SECONDS = max(5, min(int(os.getenv("OPENAI_RATE_LIMIT_COOLDOWN_SECONDS", "20")), 120))
AI_REQUEST_TIMEOUT_SECONDS = max(5, min(int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "15")), 45))
AI_MAX_ATTEMPTS = max(1, min(int(os.getenv("OPENAI_MAX_ATTEMPTS", str(min(5, len(AI_RETRY_BACKOFF_SECONDS) + 1)))), 5))
AI_REQUEST_LOCK_TIMEOUT_SECONDS = max(1.0, min(float(os.getenv("OPENAI_REQUEST_LOCK_TIMEOUT_SECONDS", "180")), 300.0))
_AI_RATE_LIMIT_UNTIL: float = 0.0
_AI_RATE_LIMIT_LOCK = threading.Lock()
_AI_REQUEST_LOCK = threading.Lock()

AI_POLICY_OBSERVATION_LIMIT = 30
AI_MAX_OUTPUT_TOKENS = 2500
AI_REPAIR_OUTPUT_TOKENS = 2800

STANDARD_AI_REVIEW_QUERY = (
    "You are an expert financial-statement quality-control reviewer. Apply the same standard to all audited statements, "
    "including the report front matter, directors report, management statements, independent auditor sections, and all notes.\n\n"
    "Scope:\n"
    "1. Review spelling, grammar, wording, typographical and presentation defects in the supplied extraction text.\n"
    "2. Review note references shown on the face of primary statements (statement of profit or loss and OCI, statement of "
    "financial position, statement of changes in equity, statement of cash flows).\n"
    "3. Verify consistency of regulation references (CAMA, FRC, IFRS/IAS, tax laws, and related Nigerian requirements).\n"
    "4. Verify line-item note-links are correct, with compatible note headings and accurate note numbers.\n"
    "5. Perform casting/cross-casting checks where possible: subtotals, grand totals, prior-year comparatives, grouped "
    "movement tables, and agreement between primary statements and notes.\n"
    "6. Perform IAS 7 statement-of-cash-flows checks: operating/investing/financing subtotals, net movement, opening/closing "
    "reconciliation to cash balances, and non-cash treatment consistency.\n"
    "7. Identify missing disclosures, wrong section notes, outdated wording, unsupported amounts, and drafting-quality issues.\n\n"
    "Output requirements:\n"
    "Return one valid JSON object only, with no markdown. Use conservative confidence and label uncertain items as review prompts.\n"
    "You may flag only issues with explicit evidence in the extracted context.\n\n"
    "Expected JSON object shape:\n"
    "{\n"
    '  "summary": "short overall conclusion",\n'
    '  "observations": [\n'
    "    {\n"
    '      "title": "brief issue title",\n'
    '      "section_or_statement": "statement/page/section where issue appears",\n'
    '      "dimension": "policy_relevance | disclosure_completeness | standard_context | industry_fit",\n'
    '      "category": "optional audit category (e.g., Note Cross-reference or Spelling/Grammar)",\n'
    '      "standard_or_topic": "regulatory standard/topic",\n'
    '      "severity": "High|Medium|Low",\n'
    '      "priority": "High|Medium|Low",\n'
    '      "confidence": "High|Medium|Low",\n'
    '      "status": "exception|review_prompt|ok",\n'
    '      "issue": "what is wrong",\n'
    '      "rationale": "why this issue is identified",\n'
    '      "expected_fix": "what corrective wording/structure is expected",\n'
    '      "recommendation": "next reviewer step",\n'
    '      "page_reference": "Page X",\n'
    '      "note_reference": "Note X",\n'
    '      "evidence_snippet": "short quote or paraphrase from supplied text"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Use \"status\": \"ok\" only where evidence supports no further action.\n"
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
    evidence_rows: list[dict[str, str]] | None = None


class MalformedAiResponseError(RuntimeError):
    def __init__(self, message: str, text: str):
        super().__init__(message)
        self.text = text


class AiProviderError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def run_ai_policy_review(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool] | None = None,
    model: str = DEFAULT_AI_MODEL,
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
            evidence_rows=[],
        )

    note_context = _policy_context(note_sections, document)
    evidence_rows = _policy_evidence_rows(note_sections, document, policy_map or {})
    if not note_context.strip():
        return AiPolicyReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="skipped",
            model=model,
            message="AI policy and standards judgement skipped because no policy/disclosure note context was detected.",
            evidence_rows=[],
        )

    payload = {
        "model": model,
        "max_output_tokens": AI_MAX_OUTPUT_TOKENS,
        "input": [
            {
                "role": "system",
                "content": (
                    f"{STANDARD_AI_REVIEW_QUERY}\n"
                    "Do not invent missing evidence. If evidence is weak, use Medium/Low confidence and prefer review_prompt."
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
        try:
            parsed = _parse_response_json(response_json)
        except MalformedAiResponseError as parse_exc:
            parsed = _repair_response_json(
                api_key,
                model,
                parse_exc.text,
                '{"summary":"short reviewer summary","observations":[{"title":"...","severity":"High|Medium|Low","confidence":"High|Medium|Low","status":"exception|review_prompt|ok","issue":"...","rationale":"...","recommendation":"...","page_reference":"Page X","note_reference":"Note X","evidence_snippet":"..."}]}',
            )
        findings, export_rows = _rows_to_outputs(parsed.get("observations", []))
        summary = str(parsed.get("summary", "") or "").strip()
        return AiPolicyReviewResult(
            findings=findings,
            export_rows=export_rows,
            summary=summary,
            status="completed",
            model=model,
            evidence_rows=evidence_rows,
        )
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        friendly_message = _friendly_ai_error_message(exc)
        return AiPolicyReviewResult(
            findings=[],
            export_rows=[],
            summary="",
            status="deferred" if _is_rate_limit_error(exc) or isinstance(exc, MalformedAiResponseError) else "error",
            model=model,
            message=friendly_message,
            evidence_rows=evidence_rows if 'evidence_rows' in locals() else [],
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
    json_shape = (
        "{\n"
        '  "summary": "short reviewer summary",\n'
        '  "observations": [\n'
        '    {\n'
        '      "title": "...",\n'
        '      "section_or_statement": "statement/page/section where issue appears",\n'
        '      "dimension": "policy_relevance | disclosure_completeness | standard_context | industry_fit",\n'
        '      "category": "optional audit category",\n'
        '      "standard_or_topic": "...",\n'
        '      "severity": "High|Medium|Low",\n'
        '      "priority": "High|Medium|Low",\n'
        '      "confidence": "High|Medium|Low",\n'
        '      "status": "exception|review_prompt|ok",\n'
        '      "issue": "...",\n'
        '      "rationale": "...",\n'
        '      "expected_fix": "proposed correction or expected wording, if applicable",\n'
        '      "recommendation": "...",\n'
        '      "page_reference": "Page X or Pages X-Y if visible",\n'
        '      "note_reference": "Note X if visible",\n'
        '      "evidence_snippet": "short quote/paraphrase from provided text"\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    return (
        f"{STANDARD_AI_REVIEW_QUERY}\n\n"
        "Return JSON with this shape:\n"
        f"{json_shape}\n"
        "Rules:\n"
        "1. Use High only when the wording clearly conflicts with the entity, transaction, or current standard context.\n"
        "2. Use Medium or Low for judgement prompts.\n"
        "3. If a policy is present and appears tailored, you may return status=ok.\n"
        "4. If disclosure wording is generic but not clearly wrong, prefer review_prompt.\n"
        "5. Consider industry fit when judging whether a paragraph represents the entity's business.\n"
        f"6. Do not return more than {AI_POLICY_OBSERVATION_LIMIT} observations unless evidence strongly requires it.\n\n"
        f"Company profile:\n- Company name: {company_name}\n- Industry: {industry}\n- Reporting currency: {currency}\n"
        f"- Framework: {framework}\n- Expected policies: {expected_policies}\n- Significant transactions: {significant_transactions}\n"
        f"- Forced checklist areas: {checklist_areas}\n- Deterministically detected policy areas: {', '.join(detected_indicators) or 'None'}\n\n"
        f"Disclosure evidence snippets:\n{disclosure_context}\n\n"
        f"Policies and note context:\n{note_context}"
    )



def _policy_evidence_rows(
    note_sections: dict[str, str],
    document: PdfDocument,
    policy_map: dict[str, bool],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ref, text in sorted(note_sections.items(), key=lambda item: _sort_note_key(item[0])):
        clean = _clean_ai_snippet(text)
        if not clean:
            continue
        lower = clean.lower()
        topics = _policy_topics_for_text(clean, policy_map)
        if ref in {"1", "2"} or topics or any(keyword in lower for keyword in POLICY_KEYWORDS):
            rows.append(
                {
                    "Evidence type": "Policy/note context",
                    "Page reference": _note_page_reference(document, ref, clean),
                    "Note reference": f"Note {ref}",
                    "Detected topics": ", ".join(topics) or "Policy/disclosure context",
                    "Snippet": clean[:700],
                    "AI role": "Policy relevance, disclosure completeness, standards context, and industry fit judgement",
                }
            )
        if len(rows) >= 8:
            break
    if rows:
        return rows
    for page in document.pages:
        lower = page.text.lower()
        if any(pattern in lower for pattern in NOTES_HEADING_PATTERNS) or "significant accounting policies" in lower:
            clean = _clean_ai_snippet(page.text)
            rows.append(
                {
                    "Evidence type": "Fallback page context",
                    "Page reference": f"Page {page.number}",
                    "Note reference": "",
                    "Detected topics": ", ".join(_policy_topics_for_text(clean, policy_map)) or "Policy/disclosure context",
                    "Snippet": clean[:700],
                    "AI role": "Fallback policy/disclosure judgement context",
                }
            )
        if len(rows) >= 3:
            break
    return rows



POLICY_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "contracts with customers", "ifrs 15", "turnover", "rental income"),
    "financial instruments": ("financial instrument", "expected credit", "ecl", "amortised cost", "derecognition"),
    "tax": ("taxation", "income tax", "deferred tax", "current tax"),
    "ppe": ("property, plant and equipment", "ppe", "depreciation", "capital work in progress"),
    "intangibles": ("intangible", "software", "amortisation", "amortization"),
    "inventory": ("inventor", "stock", "net realisable", "net realizable"),
    "leases": ("right-of-use", "right of use", "lease liability", "lease expense", "lease maturity"),
    "related parties": ("related part", "key management", "director", "sister compan", "shareholder"),
    "consolidation": ("consolidated", "subsidiar", "non-controlling", "investment in subsidiar", "group financial"),
    "foreign currency": ("foreign currenc", "exchange difference", "functional currency"),
    "employee benefits": ("employee benefit", "pension", "defined contribution", "retirement benefit"),
    "events after reporting period": ("events after", "subsequent event", "after the reporting period"),
}


def _clean_ai_snippet(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)\bD\s+R\s+A\s+F\s+T\b", " ", text)
    text = re.sub(r"(?i)\bT\s+F\s+A\s+R\s+D\b", " ", text)
    text = re.sub(r"(?m)^\s*[DRAFT]{1}\s*$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _policy_topics_for_text(text: str, policy_map: dict[str, bool]) -> list[str]:
    lower = str(text or "").lower()
    topics: list[str] = []
    for topic, keywords in POLICY_TOPIC_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            topics.append(topic)
    for key, present in sorted(policy_map.items()):
        normalized = str(key or "").strip().lower()
        if present and normalized and normalized in lower and normalized not in topics:
            topics.append(normalized)
    return topics


def _note_page_reference(document: PdfDocument, ref: str, clean_text: str) -> str:
    note_pattern = re.compile(rf"(?<!\d)(?:note\s+)?{re.escape(str(ref))}\s*[\).:-]?\s+", re.I)
    seed = _clean_ai_snippet(clean_text)[:220]
    seed_words = [word for word in re.findall(r"[A-Za-z]{4,}", seed.lower()) if word not in {"octerra", "capital", "limited", "financial", "statements", "company"}][:6]
    for page in document.pages:
        page_clean = _clean_ai_snippet(page.text)
        page_lower = page_clean.lower()
        note_match = note_pattern.search(page_clean)
        if note_match and seed_words and sum(1 for word in seed_words if word in page_lower) >= min(3, len(seed_words)):
            return f"Page {page.number}"
        if str(ref).upper() == "1" and note_match and any(term in page_lower for term in ("accounting polic", "basis of preparation", "reporting entity")):
            return f"Page {page.number}"
    return "Page not isolated"

def _call_openai(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _base_ai_diagnostics(payload)
    acquired = _AI_REQUEST_LOCK.acquire(timeout=AI_REQUEST_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        diagnostics.update(
            {
                "error_type": "AIRequestQueueTimeout",
                "error_category": "busy",
                "error_message": "Another AI review request did not finish before the queue wait timeout.",
                "retry_count": 0,
            }
        )
        raise AiProviderError(
            "AI service busy: another AI review request did not finish before the queue wait timeout.",
            diagnostics,
        )

    try:
        last_error: Exception | None = None
        endpoint_styles = _openai_endpoint_styles()
        for attempt in range(AI_MAX_ATTEMPTS):
            diagnostics["retry_count"] = attempt
            cooldown_remaining = _rate_limit_wait_seconds()
            if cooldown_remaining > 0:
                diagnostics.setdefault("retry_wait_seconds", []).append(cooldown_remaining)
                _sleep_before_ai_retry(cooldown_remaining)

            for style_index, endpoint_style in enumerate(endpoint_styles):
                endpoint = _openai_endpoint_for_style(endpoint_style)
                request_payload = _payload_for_endpoint_style(payload, endpoint_style)
                diagnostics.update({"endpoint_style": endpoint_style, "endpoint": endpoint})
                req = request.Request(
                    endpoint,
                    data=json.dumps(request_payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=AI_REQUEST_TIMEOUT_SECONDS) as response:
                        diagnostics["status_code"] = getattr(response, "status", 200)
                        raw_body = response.read().decode("utf-8", errors="replace")
                        diagnostics["response_preview"] = raw_body[:500]
                        try:
                            return json.loads(raw_body)
                        except json.JSONDecodeError as exc:
                            provider_error = _invalid_provider_response_error(raw_body, endpoint_style, endpoint, diagnostics)
                            last_error = provider_error
                            if _can_try_next_endpoint_style(endpoint_styles, style_index, provider_error):
                                continue
                            raise provider_error from exc
                except error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    retryable = exc.code == 429 or exc.code in {500, 502, 503, 504}
                    diagnostics.update(
                        {
                            "status_code": exc.code,
                            "error_type": type(exc).__name__,
                            "error_message": body[:1200],
                            "response_preview": body[:500],
                            "error_category": _classify_ai_error(exc.code, body),
                        }
                    )
                    last_error = AiProviderError(f"OpenAI API error {exc.code}: {body[:240]}", dict(diagnostics))
                    if _can_try_next_endpoint_style(endpoint_styles, style_index, last_error):
                        continue
                    if not retryable:
                        raise last_error from exc
                    wait_seconds = _retry_after_seconds(exc.headers.get("Retry-After", ""))
                    if wait_seconds is None:
                        wait_seconds = _retry_wait_seconds(attempt)
                    diagnostics.setdefault("retry_wait_seconds", []).append(wait_seconds)
                    if exc.code == 429:
                        _set_rate_limit_block(wait_seconds)
                    if attempt + 1 < AI_MAX_ATTEMPTS:
                        _sleep_before_ai_retry(wait_seconds)
                        break
                    diagnostics["retry_count"] = attempt + 1
                    raise AiProviderError(
                        f"AI review failed after {AI_MAX_ATTEMPTS} automatic retry attempt(s): OpenAI API error {exc.code}.",
                        dict(diagnostics),
                    ) from exc
                except AiProviderError as exc:
                    last_error = exc
                    if _is_retryable_ai_error(exc) and attempt + 1 < AI_MAX_ATTEMPTS:
                        wait_seconds = _retry_wait_seconds(attempt)
                        diagnostics.setdefault("retry_wait_seconds", []).append(wait_seconds)
                        _sleep_before_ai_retry(wait_seconds)
                        break
                    if _is_retryable_ai_error(exc):
                        diagnostics["retry_count"] = attempt + 1
                        raise AiProviderError(
                            f"AI review failed after {AI_MAX_ATTEMPTS} automatic retry attempt(s): {exc}",
                            dict(getattr(exc, "diagnostics", diagnostics)),
                        ) from exc
                    raise exc
                except Exception as exc:
                    diagnostics.update(
                        {
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:1200],
                            "error_category": _classify_ai_error(None, str(exc)),
                        }
                    )
                    last_error = exc
                    if _is_retryable_ai_error(exc) and attempt + 1 < AI_MAX_ATTEMPTS:
                        wait_seconds = _retry_wait_seconds(attempt)
                        diagnostics.setdefault("retry_wait_seconds", []).append(wait_seconds)
                        _sleep_before_ai_retry(wait_seconds)
                        break
                    if _is_retryable_ai_error(exc):
                        diagnostics["retry_count"] = attempt + 1
                        raise AiProviderError(
                            f"AI review failed after {AI_MAX_ATTEMPTS} automatic retry attempt(s): {exc}",
                            dict(diagnostics),
                        ) from exc
                    raise AiProviderError(f"AI review could not be completed: {exc}", dict(diagnostics)) from exc
            else:
                continue
            continue
        if last_error:
            raise AiProviderError(
                f"AI review failed after {AI_MAX_ATTEMPTS} automatic retry attempt(s): {last_error}",
                dict(getattr(last_error, "diagnostics", diagnostics)),
            ) from last_error
        raise AiProviderError("OpenAI API call failed without a specific error.", dict(diagnostics))
    finally:
        _AI_REQUEST_LOCK.release()


def _openai_endpoint_styles() -> list[str]:
    style = str(os.getenv("OPENAI_API_STYLE", "auto") or "auto").strip().lower().replace("-", "_")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/").lower()
    if style in {"chat", "chat_completion", "chat_completions"}:
        return ["chat_completions"]
    if style in {"response", "responses"}:
        return ["responses"]
    if base.endswith("/chat/completions"):
        return ["chat_completions"]
    if base.endswith("/responses"):
        return ["responses"]
    return ["responses", "chat_completions"]


def _openai_endpoint_for_style(style: str) -> str:
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    lower = base.lower()
    if lower.endswith("/responses") or lower.endswith("/chat/completions"):
        return base
    if style == "chat_completions":
        return base + "/chat/completions"
    return base + "/responses"


def _payload_for_endpoint_style(payload: dict[str, Any], style: str) -> dict[str, Any]:
    if style != "chat_completions":
        return payload
    chat_payload: dict[str, Any] = {"model": payload.get("model", "")}
    token_limit = payload.get("max_output_tokens")
    if token_limit:
        token_field = str(os.getenv("OPENAI_CHAT_TOKEN_FIELD", "max_tokens") or "max_tokens").strip() or "max_tokens"
        chat_payload[token_field] = token_limit
    messages: list[dict[str, str]] = []
    for item in payload.get("input", []) or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user") or "user").strip() or "user"
        if role not in {"system", "user", "assistant", "developer"}:
            role = "user"
        if role == "developer":
            role = "system"
        content = _prompt_content_to_text(item.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        messages.append({"role": "user", "content": _prompt_content_to_text(payload.get("input"))})
    chat_payload["messages"] = messages
    return chat_payload


def _prompt_content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_prompt_content_to_text(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("text", "content", "input_text", "output_text"):
            if key in value:
                text = _prompt_content_to_text(value.get(key))
                if text:
                    return text
        return " ".join(part for part in (_prompt_content_to_text(v) for v in value.values()) if part)
    return str(value)


def _invalid_provider_response_error(raw_body: str, endpoint_style: str, endpoint: str, diagnostics: dict[str, Any]) -> AiProviderError:
    preview = str(raw_body or "")[:500]
    message = "Provider returned an empty response." if not str(raw_body or "").strip() else "Provider returned a non-JSON response."
    error_diagnostics = dict(diagnostics)
    error_diagnostics.update(
        {
            "error_type": "InvalidProviderResponse",
            "error_category": "invalid_provider_response",
            "error_message": f"{message} endpoint_style={endpoint_style}; endpoint={endpoint}; response_preview={preview}",
            "response_preview": preview,
        }
    )
    return AiProviderError(error_diagnostics["error_message"], error_diagnostics)


def _can_try_next_endpoint_style(styles: list[str], style_index: int, exc: Exception) -> bool:
    if style_index + 1 >= len(styles):
        return False
    category = ""
    status_code = ""
    message = str(exc or "").lower()
    if isinstance(exc, AiProviderError):
        diagnostics = getattr(exc, "diagnostics", {}) or {}
        category = str(diagnostics.get("error_category", "") or "").lower()
        status_code = str(diagnostics.get("status_code", "") or "").strip()
        message = f"{message} {diagnostics.get('error_message', '')}".lower()
    if category in {"invalid_provider_response", "unsupported_structured_output"}:
        return True
    if status_code in {"400", "404", "405", "501"} and any(marker in message for marker in ("responses", "endpoint", "not found", "unsupported", "not supported")):
        return True
    return False


def _parse_response_json(response_json: dict[str, Any]) -> dict[str, Any]:
    text = _extract_response_text(response_json).strip()
    if not text:
        raise RuntimeError("OpenAI response did not contain text output.")

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_text_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    detail = f"AI response was not valid JSON: {last_error}" if last_error else "AI response was not valid JSON."
    raise MalformedAiResponseError(detail, text)


def _json_text_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(stripped)
    balanced = _balanced_json_object(stripped)
    if balanced:
        candidates.append(balanced)
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1].strip()
    return ""


def _repair_response_json(api_key: str, model: str, malformed_text: str, expected_shape: str) -> dict[str, Any]:
    repair_payload = {
        "model": model,
        "max_output_tokens": AI_REPAIR_OUTPUT_TOKENS,
        "input": [
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON produced by another model. Return valid JSON only. "
                    "Do not add markdown. Do not change the factual content, only fix JSON syntax. "
                    "If a field is unclear, keep the nearest valid string value."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Expected JSON shape:\n{expected_shape}\n\n"
                    "Malformed JSON to repair:\n"
                    f"{malformed_text[:12000]}"
                ),
            },
        ],
    }
    return _parse_response_json(_call_openai(api_key, repair_payload))


def _extract_response_text(response_json: dict[str, Any]) -> str:
    collected: list[str] = []
    _append_text_fragments(collected, response_json.get("output_text"))
    for item in response_json.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        _append_text_fragments(collected, item.get("content"))
    for choice in response_json.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            _append_text_fragments(collected, message.get("content"))
        _append_text_fragments(collected, choice.get("text"))
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
    for observation in observations[:AI_POLICY_OBSERVATION_LIMIT]:
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
        category = str(observation.get("category", "") or "").strip() or "General"
        section_or_statement = str(observation.get("section_or_statement", "") or "").strip() or "Document-wide"
        priority = str(observation.get("priority", "") or "").strip() or "Medium"
        expected_fix = str(observation.get("expected_fix", "") or "").strip()

        if not issue and not title:
            continue
        severity, confidence, status = _calibrate_policy_observation(
            severity=severity,
            confidence=confidence,
            status=status,
            standard_or_topic=standard_or_topic,
            issue=issue,
            rationale=rationale,
            evidence_snippet=evidence_snippet,
        )
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
                "Section / statement": section_or_statement,
                "Expected fix": expected_fix,
                "Priority": priority,
                "Category detail": category,
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
                    "category_detail": category,
                    "expected_fix": expected_fix,
                    "section_or_statement": section_or_statement,
                    "review_priority": priority,
                },
            )
        )
    return findings, export_rows




def _calibrate_policy_observation(
    *,
    severity: str,
    confidence: str,
    status: str,
    standard_or_topic: str,
    issue: str,
    rationale: str,
    evidence_snippet: str,
) -> tuple[str, str, str]:
    combined = " ".join([standard_or_topic, issue, rationale, evidence_snippet]).lower()
    if _weak_consolidation_observation(combined) or _weak_lease_observation(combined):
        return "Low", "Low", "review_prompt" if status != "ok" else status
    if severity == "High" and confidence != "High":
        return "Medium", confidence, status
    if severity == "High" and status != "exception":
        return "Medium", confidence, status
    return severity, confidence, status


def _weak_consolidation_observation(text: str) -> bool:
    consolidation_terms = ("consolidation", "consolidated", "subsidiary", "investment entity", "group structure")
    if not any(term in text for term in consolidation_terms):
        return False
    strong_terms = ("investment in subsidiary", "non-controlling", "consolidated statement", "consolidated financial", "parent company")
    if any(term in text for term in strong_terms):
        return False
    weak_terms = ("related part", "sister compan", "common control", "shareholder", "director")
    return any(term in text for term in weak_terms)


def _weak_lease_observation(text: str) -> bool:
    lease_terms = ("ifrs 16", "lease", "right-of-use", "right of use")
    if not any(term in text for term in lease_terms):
        return False
    strong_disclosure_terms = (
        "lease maturity",
        "lease expense",
        "rou depreciation",
        "finance lease",
        "operating lease balance",
        "actual lease arrangement",
        "lease commitment",
    )
    if any(term in text for term in strong_disclosure_terms):
        return False
    balance_context = any(term in text for term in ("lease liability balance", "right-of-use asset balance", "right of use asset balance", "carrying amount", "recognised amount", "recognized amount"))
    amount_context = bool(re.search(r"\b\d{1,3}(?:,\d{3})+\b", text))
    generic_sections = ("new standard", "amendment", "deferred tax", "theoretical", "accounting policy")
    if any(term in text for term in generic_sections) and not amount_context:
        return True
    if ("lease liability" in text or "right-of-use asset" in text or "right of use asset" in text) and (balance_context or amount_context):
        return False
    return not (balance_context or amount_context)


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


def _base_ai_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(payload.get("model", "") or ""),
        "input_token_estimate": _estimate_payload_tokens(payload),
        "max_output_tokens": payload.get("max_output_tokens", ""),
        "retry_count": 0,
        "retry_wait_seconds": [],
        "status_code": "",
        "error_type": "",
        "error_category": "",
        "error_message": "",
    }


def _estimate_payload_tokens(payload: dict[str, Any]) -> int:
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)
    return max(1, int(len(text) / 4))


def _classify_ai_error(status_code: int | None, message: str) -> str:
    lower = str(message or "").lower()
    if _looks_like_dns_error(lower):
        return "network_dns"
    if status_code == 429 or "rate limit" in lower or "too many requests" in lower:
        if "insufficient_quota" in lower or "quota" in lower or "billing" in lower:
            return "insufficient_quota"
        return "rate_limit"
    if "invalidproviderresponse" in lower or "non-json response" in lower or "empty response" in lower:
        return "invalid_provider_response"
    if status_code in {408, 504} or "timeout" in lower or "timed out" in lower:
        return "timeout"
    if status_code == 400 and any(marker in lower for marker in ("json_schema", "response_format", "text.format", "schema", "structured output", "structured outputs")):
        return "unsupported_structured_output"
    if status_code == 400 and any(marker in lower for marker in ("model", "does not exist", "not found", "not supported", "unsupported model", "no access")):
        return "unsupported_model"
    if status_code == 400 and any(marker in lower for marker in ("context", "token", "too large", "maximum", "payload")):
        return "payload_too_large"
    if status_code == 413 or "payload too large" in lower or "request too large" in lower:
        return "payload_too_large"
    if "insufficient_quota" in lower or "quota" in lower or "billing" in lower or "credit" in lower:
        return "insufficient_quota"
    if status_code in {500, 502, 503} or "temporarily unavailable" in lower or "service busy" in lower:
        return "temporary_service_error"
    if "invalid" in lower and "api key" in lower:
        return "authentication"
    return "other"


def _looks_like_dns_error(message: str) -> bool:
    lower = str(message or "").lower()
    return any(
        marker in lower
        for marker in (
            "temporary failure in name resolution",
            "name resolution",
            "getaddrinfo",
            "failed to resolve",
            "no address associated",
            "nodename nor servname",
            "dns",
            "[errno -3]",
            "[errno 11001]",
        )
    )


def _retry_wait_seconds(attempt: int) -> float:
    if not AI_RETRY_BACKOFF_SECONDS:
        base = AI_RATE_LIMIT_COOLDOWN_SECONDS
    else:
        index = min(max(attempt, 0), len(AI_RETRY_BACKOFF_SECONDS) - 1)
        base = AI_RETRY_BACKOFF_SECONDS[index]
    jitter = random.uniform(0, max(0.5, min(5.0, base * 0.25)))
    return min(120.0, max(1.0, base + jitter))


def _sleep_before_ai_retry(wait_seconds: int | float) -> None:
    wait = max(0.0, min(float(wait_seconds or 0), 120.0))
    if wait > 0:
        time.sleep(wait)


def _retry_after_seconds(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return max(1, min(int(text), 120))
    return None


def _rate_limit_wait_seconds() -> int:
    now = time.time()
    with _AI_RATE_LIMIT_LOCK:
        if _AI_RATE_LIMIT_UNTIL <= now:
            return 0
        return max(1, int((_AI_RATE_LIMIT_UNTIL - now) + 0.5))


def _set_rate_limit_block(wait_seconds: int) -> None:
    wait = max(1, min(int(wait_seconds or AI_RATE_LIMIT_COOLDOWN_SECONDS), 120))
    with _AI_RATE_LIMIT_LOCK:
        global _AI_RATE_LIMIT_UNTIL
        _AI_RATE_LIMIT_UNTIL = max(_AI_RATE_LIMIT_UNTIL, time.time() + wait)


def _is_rate_limit_blocked() -> bool:
    return _rate_limit_wait_seconds() > 0


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate-limit" in text
        or "rate exceeded" in text
        or "too many requests" in text
        or "ai service busy" in text
        or "service busy" in text
        or "temporarily busy" in text
        or "cooldown" in text
        or "timed out" in text
        or "timeout" in text
    )


def _is_retryable_ai_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        _is_rate_limit_error(exc)
        or isinstance(exc, TimeoutError)
        or _looks_like_dns_error(text)
        or "temporarily unavailable" in text
        or "connection reset" in text
        or "remote end closed" in text
        or "503" in text
        or "502" in text
        or "500" in text
        or "504" in text
    )


def _friendly_ai_error_message(exc: Exception) -> str:
    if isinstance(exc, AiProviderError):
        category = str(exc.diagnostics.get("error_category", "") or "").lower()
        if category == "insufficient_quota":
            return "AI review was not completed because the OpenAI API quota or billing credit appears unavailable. The deterministic review and exports were still completed."
        if category == "payload_too_large":
            return "AI review was not completed because the AI evidence package was too large. Retry with Quick AI review mode. The deterministic review and exports were still completed."
        if category == "authentication":
            return "AI review was not completed because the OpenAI API key could not be authenticated. The deterministic review and exports were still completed."
        if category == "unsupported_model":
            return "AI review was not completed because the configured AI model is not available to this API key. The deterministic review and exports were still completed; see AI debug details for the model name."
        if category == "unsupported_structured_output":
            return "AI review was not completed because the provider rejected the structured-output request format. The deterministic review and exports were still completed; see AI debug details."
        if category == "invalid_provider_response":
            return "AI review was not completed because the AI provider returned an empty or non-JSON response. Check OPENAI_BASE_URL and set OPENAI_API_STYLE=chat_completions if your router does not support the Responses API. The deterministic review and exports were still completed."
        if category == "network_dns":
            return "AI review was not completed because the AI provider host could not be resolved. Check OPENAI_BASE_URL, provider DNS, and deployment network egress. The deterministic review and exports were still completed."
        if category in {"rate_limit", "timeout", "temporary_service_error", "busy"}:
            return "AI review was not completed after automatic retry attempts because the AI service remained busy or rate-limited. The deterministic review and exports were still completed. Use Retry AI Review to run only the AI layer again."
        return "AI review was not completed. The deterministic review and exports were still completed; see AI debug details for the provider error."
    if _is_rate_limit_error(exc):
        return (
            "AI review was not completed after automatic retry attempts because the AI service remained temporarily busy; "
            "the deterministic review and exports were still completed. Use Retry AI Review to run only the AI layer again."
        )
    if isinstance(exc, MalformedAiResponseError):
        return (
            "AI review was deferred because the AI returned malformed structured output; "
            "the deterministic review and exports were still completed. Please rerun the AI layer."
        )
    return f"AI review could not be completed: {exc}"


def _sort_note_key(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", str(value))
    if not match:
        return (999, str(value))
    return (int(match.group(1)), str(value))
