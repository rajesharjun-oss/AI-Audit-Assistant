from __future__ import annotations

import json
import os
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

AI_RATE_LIMIT_COOLDOWN_SECONDS = max(5, min(int(os.getenv("OPENAI_RATE_LIMIT_COOLDOWN_SECONDS", "20")), 20))
AI_REQUEST_TIMEOUT_SECONDS = max(5, min(int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "15")), 45))
AI_MAX_ATTEMPTS = max(1, min(int(os.getenv("OPENAI_MAX_ATTEMPTS", "1")), 2))
AI_REQUEST_LOCK_TIMEOUT_SECONDS = max(0.1, min(float(os.getenv("OPENAI_REQUEST_LOCK_TIMEOUT_SECONDS", "1.5")), 5.0))
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
    blocked_remaining = _rate_limit_wait_seconds()
    if blocked_remaining > 0:
        raise RuntimeError(f"OpenAI rate limit cooldown active for {blocked_remaining} second(s).")

    acquired = _AI_REQUEST_LOCK.acquire(timeout=AI_REQUEST_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        raise RuntimeError("AI service busy: another AI review request is already running.")

    try:
        blocked_remaining = _rate_limit_wait_seconds()
        if blocked_remaining > 0:
            raise RuntimeError(f"OpenAI rate limit cooldown active for {blocked_remaining} second(s).")

        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/responses"
        last_error: Exception | None = None
        for attempt in range(AI_MAX_ATTEMPTS):
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
                with request.urlopen(req, timeout=AI_REQUEST_TIMEOUT_SECONDS) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    block_seconds = _retry_after_seconds(exc.headers.get("Retry-After", ""))
                    if block_seconds is None:
                        block_seconds = AI_RATE_LIMIT_COOLDOWN_SECONDS
                    _set_rate_limit_block(block_seconds)
                    raise RuntimeError(f"OpenAI API error {exc.code}: {body[:240]}") from exc
                last_error = RuntimeError(f"OpenAI API error {exc.code}: {body[:240]}")
                if attempt + 1 >= AI_MAX_ATTEMPTS:
                    raise last_error from exc
            except Exception as exc:
                last_error = exc
                break
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI API call failed without a specific error.")
    finally:
        _AI_REQUEST_LOCK.release()


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


def _retry_after_seconds(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return max(1, min(int(text), 8))
    return None


def _rate_limit_wait_seconds() -> int:
    now = time.time()
    with _AI_RATE_LIMIT_LOCK:
        if _AI_RATE_LIMIT_UNTIL <= now:
            return 0
        return max(1, int((_AI_RATE_LIMIT_UNTIL - now) + 0.5))


def _set_rate_limit_block(wait_seconds: int) -> None:
    wait = max(1, min(int(wait_seconds or AI_RATE_LIMIT_COOLDOWN_SECONDS), AI_RATE_LIMIT_COOLDOWN_SECONDS))
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
        or "rate exceeded" in text
        or "too many requests" in text
        or "ai service busy" in text
        or "timed out" in text
        or "timeout" in text
    )


def _friendly_ai_error_message(exc: Exception) -> str:
    if _is_rate_limit_error(exc):
        return (
            "AI review was deferred because the AI service is temporarily busy; "
            "the deterministic review and exports were still completed."
            f" Try the AI layer again in about {AI_RATE_LIMIT_COOLDOWN_SECONDS} second(s)."
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




