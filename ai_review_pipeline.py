from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ai_finding_review import run_ai_finding_review
from ai_full_review import run_ai_full_review
from ai_policy_review import run_ai_policy_review
from models import CompanyProfile, Finding, PdfDocument

AI_PIPELINE_LOCK_TIMEOUT_SECONDS = max(
    1.0,
    min(float(os.getenv("OPENAI_AI_PIPELINE_LOCK_TIMEOUT_SECONDS", "300")), 600.0),
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



def run_ai_review_pipeline(review_context: AiReviewContext) -> AiReviewPipelineResult:
    """Run the optional AI layer as one queued, sequential pipeline.

    Individual OpenAI requests still use the shared retry/backoff request helper.
    This pipeline lock prevents two whole AI review sequences from interleaving.
    """
    result = AiReviewPipelineResult(
        findings=list(review_context.findings),
        policy_model=review_context.model,
        full_model=review_context.model,
        finding_model=review_context.model,
    )
    if not review_context.use_policy_review and not review_context.use_full_review:
        return result

    acquired = _AI_PIPELINE_LOCK.acquire(timeout=AI_PIPELINE_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        _mark_pipeline_queue_timeout(result, review_context)
        return result

    try:
        prior_step_failed = False
        if review_context.use_policy_review:
            prior_step_failed = _run_policy_step(review_context, result)

        if review_context.use_full_review:
            if prior_step_failed:
                result.full_status = "deferred"
                result.full_message = (
                    "AI full review was not run because AI policy judgement did not complete after automatic retries. "
                    f"{result.policy_message}"
                ).strip()
                result.checks_skipped.append(result.full_message)
            else:
                prior_step_failed = _run_full_step(review_context, result)

        if prior_step_failed:
            result.finding_status = "deferred"
            result.finding_message = "AI finding review was not run because an earlier AI review step did not complete after automatic retries."
            result.checks_skipped.append(result.finding_message)
        else:
            _run_finding_cleanup_step(review_context, result)
        return result
    finally:
        _AI_PIPELINE_LOCK.release()



def _run_policy_step(review_context: AiReviewContext, result: AiReviewPipelineResult) -> bool:
    policy_review = run_ai_policy_review(
        review_context.document,
        review_context.profile,
        review_context.note_sections,
        policy_map=review_context.policy_map,
        model=review_context.model,
    )
    result.policy_status = policy_review.status
    result.policy_model = policy_review.model
    result.policy_summary = policy_review.summary
    result.policy_export = policy_review.export_rows
    result.policy_message = policy_review.message
    result.evidence_pack_rows.extend(getattr(policy_review, "evidence_rows", None) or [])
    if policy_review.status == "completed":
        result.findings.extend(policy_review.findings)
        result.checks_performed.append(f"AI policy and standards judgement completed using {policy_review.model}.")
        return False
    if policy_review.message:
        result.checks_skipped.append(policy_review.message)
    return policy_review.status in FAILED_AI_STATUSES



def _run_full_step(review_context: AiReviewContext, result: AiReviewPipelineResult) -> bool:
    full_review = run_ai_full_review(
        review_context.document,
        review_context.profile,
        review_context.note_sections,
        policy_map=review_context.policy_map,
        model=review_context.model,
    )
    result.full_status = full_review.status
    result.full_model = full_review.model
    result.full_summary = full_review.summary
    result.full_export = full_review.export_rows
    result.full_message = full_review.message
    result.evidence_pack_rows.extend(getattr(full_review, "evidence_rows", None) or [])
    if full_review.status == "completed":
        result.findings.extend(full_review.findings)
        result.checks_performed.append(f"AI full financial statement review completed using {full_review.model}.")
        return False
    if full_review.message:
        result.checks_skipped.append(full_review.message)
    return full_review.status in FAILED_AI_STATUSES



def _run_finding_cleanup_step(review_context: AiReviewContext, result: AiReviewPipelineResult) -> None:
    finding_review = run_ai_finding_review(
        review_context.document,
        review_context.profile,
        result.findings,
        model=review_context.model,
        pdf_path=review_context.pdf_path,
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
