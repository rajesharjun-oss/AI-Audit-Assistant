from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ai_combined_review import run_combined_ai_review
from ai_finding_review import run_ai_finding_review
from models import DEFAULT_AI_MODEL, CompanyProfile, Finding, PdfDocument

AI_PIPELINE_LOCK_TIMEOUT_SECONDS = max(
    1.0,
    min(float(os.getenv("OPENAI_AI_PIPELINE_LOCK_TIMEOUT_SECONDS", "30")), 120.0),
)
_AI_PIPELINE_LOCK = threading.Lock()
FAILED_AI_STATUSES = {"deferred", "error", "unavailable"}


@dataclass(frozen=True)
class AiReviewContext:
    document: PdfDocument
    profile: CompanyProfile
    note_sections: dict[str, str]
    policy_map: dict[str, bool]
    findings: list[Finding]
    model: str
    pdf_path: str | Path | None = None
    use_policy_review: bool = False
    use_full_review: bool = False
    checks_skipped: list[str] | None = None
    review_mode: str = "Standard review"


@dataclass
class AiReviewPipelineResult:
    findings: list[Finding]
    checks_performed: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)
    policy_export: list[dict[str, str]] = field(default_factory=list)
    policy_summary: str = ""
    policy_status: str = "disabled"
    policy_model: str = ""
    policy_message: str = ""
    full_export: list[dict[str, str]] = field(default_factory=list)
    full_summary: str = ""
    full_status: str = "disabled"
    full_model: str = ""
    full_message: str = ""
    finding_export: list[dict[str, str]] = field(default_factory=list)
    finding_summary: str = ""
    finding_status: str = "disabled"
    finding_model: str = ""
    finding_message: str = ""
    finding_suppressed: int = 0
    finding_suppressed_rows: list[dict[str, str]] = field(default_factory=list)
    finding_reviewed: int = 0
    evidence_pack_rows: list[dict[str, str]] = field(default_factory=list)
    error_rows: list[dict[str, str]] = field(default_factory=list)
    review_comment_rows: list[dict[str, str]] = field(default_factory=list)
    summary_fields: dict[str, str] = field(default_factory=dict)
    review_mode: str = "standard"
    combined_summary: str = ""
    combined_memo: str = ""



def run_ai_review_pipeline(review_context: AiReviewContext) -> AiReviewPipelineResult:
    """Run the optional AI layer through one queued combined review pipeline."""
    result = AiReviewPipelineResult(
        findings=list(review_context.findings),
        policy_model=review_context.model or DEFAULT_AI_MODEL,
        full_model=review_context.model or DEFAULT_AI_MODEL,
        finding_model=review_context.model or DEFAULT_AI_MODEL,
        review_mode=_display_review_mode(review_context.review_mode),
    )
    if not review_context.use_policy_review and not review_context.use_full_review:
        return result

    acquired = _AI_PIPELINE_LOCK.acquire(timeout=AI_PIPELINE_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        _mark_pipeline_queue_timeout(result, review_context)
        return result

    try:
        combined = run_combined_ai_review(
            review_context.document,
            review_context.profile,
            review_context.note_sections,
            review_context.policy_map,
            review_context.findings,
            review_context.checks_skipped or [],
            model=review_context.model,
            review_mode=review_context.review_mode,
        )
        _merge_combined_result(result, combined, review_context)
        if combined.status == "completed":
            return result

        # Combined review failed. Run the smaller cleanup path only for parser/format failures;
        # capacity/configuration failures should not trigger a second AI call.
        if _should_run_finding_cleanup_after_combined_failure(combined):
            _run_finding_cleanup_step(review_context, result, pdf_path=None)
        return result
    finally:
        _AI_PIPELINE_LOCK.release()



def _merge_combined_result(result: AiReviewPipelineResult, combined, review_context: AiReviewContext) -> None:
    result.findings = combined.findings
    result.policy_export = combined.policy_export
    summary_fields = dict(getattr(combined, "summary_fields", {}) or {})
    mode_label = _display_review_mode(getattr(combined, "review_mode", "") or review_context.review_mode).title()
    result.policy_summary = summary_fields.get("AI policy judgement summary") or _stage_summary_from_rows(
        "AI policy judgement",
        result.policy_export,
        f"No AI policy exceptions were added in {mode_label} mode from the supplied evidence.",
    )
    result.policy_status = _stage_status_from_combined(combined.status, review_context.use_policy_review)
    result.policy_model = combined.model or review_context.model or DEFAULT_AI_MODEL
    result.policy_message = combined.message
    result.full_export = combined.full_export
    result.full_summary = summary_fields.get("AI full review summary") or _stage_summary_from_rows(
        "AI full review",
        result.full_export or list(getattr(combined, "review_comment_rows", []) or []),
        f"No additional AI full-review exceptions were added in {mode_label} mode from the supplied evidence.",
    )
    result.full_status = _stage_status_from_combined(combined.status, review_context.use_full_review)
    result.full_model = combined.model or review_context.model or DEFAULT_AI_MODEL
    result.full_message = combined.message
    result.finding_export = combined.finding_export
    result.finding_summary = summary_fields.get("AI finding review summary") or _finding_summary_from_rows(result.finding_export, mode_label)
    result.finding_status = "completed" if combined.status == "completed" else combined.status
    result.finding_model = combined.model or review_context.model or DEFAULT_AI_MODEL
    result.finding_message = combined.message
    result.finding_suppressed = combined.suppressed_count
    result.finding_suppressed_rows = combined.suppressed_rows
    result.finding_reviewed = combined.reviewed_count
    result.evidence_pack_rows.extend(combined.evidence_rows)
    result.error_rows.extend(combined.error_rows)
    result.review_comment_rows = list(getattr(combined, "review_comment_rows", []) or [])
    result.summary_fields = summary_fields
    result.review_mode = combined.review_mode
    result.combined_summary = summary_fields.get("Combined AI review summary") or combined.summary
    result.combined_memo = combined.executive_memo if combined.executive_memo != combined.summary else result.combined_summary
    if combined.status == "completed":
        result.checks_performed.append(
            f"Combined AI review completed using {combined.model} in {combined.review_mode.title()} mode; "
            f"{combined.reviewed_count} deterministic finding(s) reviewed and {combined.suppressed_count} suppressed."
        )
    elif combined.message:
        result.checks_skipped.append(combined.message)


def _stage_summary_from_rows(section_name: str, rows: list[dict[str, str]], default: str) -> str:
    supported_rows = [row for row in list(rows or []) if _row_has_page_or_note(row) and _row_status_is_reviewable(row)]
    if not supported_rows:
        return default
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for row in supported_rows:
        severity = _normalize_row_severity(row)
        counts[severity] = counts.get(severity, 0) + 1
    count_text = ", ".join(f"{count} {severity}" for severity, count in counts.items() if count)
    locations = _row_locations(supported_rows)
    location_text = f" Locations: {locations}." if locations else ""
    return f"{section_name} added {count_text or len(supported_rows)} evidence-backed review item(s).{location_text}"


def _finding_summary_from_rows(rows: list[dict[str, str]], mode_label: str) -> str:
    if not rows:
        return f"AI finding review completed in {mode_label} mode; no deterministic findings required suppression, downgrade, or rewrite."
    decisions: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("Decision", "") or "Reviewed").strip().title()
        decisions[decision] = decisions.get(decision, 0) + 1
    decision_text = ", ".join(f"{count} {decision.lower()}" for decision, count in sorted(decisions.items()))
    return f"AI finding review completed in {mode_label} mode; {decision_text} deterministic finding(s)."


def _row_has_page_or_note(row: dict[str, str]) -> bool:
    page = str(row.get("Page reference") or row.get("Page number") or row.get("page_reference") or row.get("page_number") or "").strip()
    note = str(row.get("Note reference") or row.get("note_reference") or "").strip()
    return bool(page or note)


def _row_status_is_reviewable(row: dict[str, str]) -> bool:
    status = str(row.get("Status") or row.get("status") or "").strip().lower()
    return status not in {"", "ok", "passed", "pass", "skipped", "not elevated", "internal note"}


def _normalize_row_severity(row: dict[str, str]) -> str:
    text = str(row.get("Severity") or row.get("Priority") or row.get("priority") or "Low").strip().title()
    return text if text in {"High", "Medium", "Low"} else "Low"


def _row_locations(rows: list[dict[str, str]], limit: int = 6) -> str:
    locations: list[str] = []
    for row in rows:
        page = str(row.get("Page reference") or row.get("Page number") or row.get("page_reference") or row.get("page_number") or "").strip()
        note = str(row.get("Note reference") or row.get("note_reference") or "").strip()
        location = " | ".join(part for part in (page, note) if part)
        if location and location not in locations:
            locations.append(location)
    shown = ", ".join(locations[:limit])
    if len(locations) > limit:
        shown += f", plus {len(locations) - limit} more"
    return shown


def _stage_status_from_combined(status: str, enabled: bool) -> str:
    if not enabled:
        return "disabled"
    return "completed" if status == "completed" else status



def _should_run_finding_cleanup_after_combined_failure(combined) -> bool:
    if getattr(combined, "status", "") in {"completed", "unavailable"}:
        return False
    category = _combined_failure_category(combined)
    capacity_or_configuration_failures = {
        "timeout",
        "rate_limit",
        "busy",
        "insufficient_quota",
        "unsupported_model",
        "payload_too_large",
        "network_dns",
        "invalid_api_endpoint",
    }
    return category not in capacity_or_configuration_failures


def _combined_failure_category(combined) -> str:
    rows = list(getattr(combined, "error_rows", None) or [])
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        category = str(row.get("Error category", "") or row.get("error_category", "") or "").strip().lower()
        if category:
            return category
    return ""


def _finding_cleanup_model(review_context: AiReviewContext, result: AiReviewPipelineResult) -> str:
    for candidate in (result.finding_model, result.full_model, result.policy_model, review_context.model, DEFAULT_AI_MODEL):
        text = str(candidate or "").strip()
        if text:
            return text
    return DEFAULT_AI_MODEL


def _run_finding_cleanup_step(review_context: AiReviewContext, result: AiReviewPipelineResult, pdf_path: str | Path | None | object = Ellipsis) -> None:
    finding_review = run_ai_finding_review(
        review_context.document,
        review_context.profile,
        result.findings,
        model=_finding_cleanup_model(review_context, result),
        pdf_path=review_context.pdf_path if pdf_path is Ellipsis else pdf_path,
    )
    result.findings = finding_review.findings
    result.finding_export = finding_review.export_rows
    result.finding_summary = finding_review.summary
    result.finding_status = finding_review.status
    result.finding_model = finding_review.model
    result.finding_message = finding_review.message
    result.evidence_pack_rows.extend(getattr(finding_review, "evidence_rows", None) or [])
    result.finding_suppressed = finding_review.suppressed_count
    result.finding_suppressed_rows = getattr(finding_review, "suppressed_rows", None) or []
    result.finding_reviewed = finding_review.reviewed_count
    if finding_review.status == "completed":
        result.checks_performed.append(
            f"AI finding review completed using {finding_review.model} on {finding_review.reviewed_count} weak finding(s); "
            f"{finding_review.suppressed_count} low-confidence finding(s) were suppressed."
        )
    elif finding_review.message:
        result.checks_skipped.append(finding_review.message)




def _display_review_mode(value: str) -> str:
    lower = str(value or "standard").strip().lower()
    if lower.startswith("quick"):
        return "quick"
    if lower.startswith("deep"):
        return "deep"
    return "standard"


def _mark_pipeline_queue_timeout(result: AiReviewPipelineResult, review_context: AiReviewContext) -> None:
    message = (
        "AI review pipeline was not completed because another AI review did not finish before the queue wait timeout. "
        "The deterministic review and exports were still completed. Use Retry AI Review to run the AI layer again."
    )
    if review_context.use_policy_review:
        result.policy_status = "deferred"
        result.policy_message = message
    if review_context.use_full_review:
        result.full_status = "deferred"
        result.full_message = message
    result.finding_status = "deferred"
    result.finding_message = message
    result.checks_skipped.append(message)
