from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import CompanyProfile, ReviewOptions, ReviewResult
from report_exports import build_excel_export, exported_file_stem
from reviewer import findings_to_markdown, review_pdf


def review_summary(result: ReviewResult, pdf_path: Path) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "pdf_path": str(pdf_path),
        "company_name": str(metrics.get("detected_profile", {}).get("Company name", "")) if isinstance(metrics.get("detected_profile"), dict) else "",
        "file_stem": exported_file_stem(result),
        "findings": metrics.get("findings", len(result.findings)),
        "high_findings": metrics.get("high", 0),
        "medium_findings": metrics.get("medium", 0),
        "low_findings": metrics.get("low", 0),
        "checks_performed": metrics.get("checks_performed_count", 0),
        "checks_passed": metrics.get("checks_passed_count", 0),
        "checks_skipped": metrics.get("checks_skipped_count", 0),
        "document_scope": metrics.get("detected_profile", {}).get("Document scope", "") if isinstance(metrics.get("detected_profile"), dict) else "",
        "entity_type": metrics.get("detected_profile", {}).get("Entity type", "") if isinstance(metrics.get("detected_profile"), dict) else "",
        "ai_policy_review_status": metrics.get("ai_policy_review_status", "disabled"),
        "ai_policy_review_message": metrics.get("ai_policy_review_message", ""),
        "ai_policy_review_summary": metrics.get("ai_policy_review_summary", ""),
        "ai_full_review_status": metrics.get("ai_full_review_status", "disabled"),
        "ai_full_review_message": metrics.get("ai_full_review_message", ""),
        "ai_full_review_summary": metrics.get("ai_full_review_summary", ""),
        "ai_finding_review_status": metrics.get("ai_finding_review_status", "disabled"),
        "ai_finding_review_message": metrics.get("ai_finding_review_message", ""),
        "ai_finding_review_summary": metrics.get("ai_finding_review_summary", ""),
        "ai_finding_reviewed": metrics.get("ai_finding_reviewed", 0),
        "ai_finding_suppressed": metrics.get("ai_finding_suppressed", 0),
    }


def write_review_outputs(
    pdf_path: Path,
    profile: CompanyProfile,
    options: ReviewOptions,
    output_dir: Path,
    *,
    write_excel: bool = True,
    write_markdown: bool = True,
    write_json_summary: bool = True,
) -> tuple[ReviewResult, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = review_pdf(pdf_path, profile, options)
    stem = exported_file_stem(result)
    files: dict[str, str] = {}

    if write_excel:
        excel_path = output_dir / f"{stem}_exception_register.xlsx"
        excel_path.write_bytes(build_excel_export(result))
        files["excel"] = str(excel_path)

    if write_markdown:
        markdown_path = output_dir / f"{stem}_review_debug.md"
        markdown_path.write_text(findings_to_markdown(result), encoding="utf-8")
        files["markdown"] = str(markdown_path)

    summary = review_summary(result, pdf_path)
    summary["output_files"] = files

    if write_json_summary:
        json_path = output_dir / f"{stem}_run_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        files["json_summary"] = str(json_path)

    return result, summary


def ai_verification_errors(
    result: ReviewResult,
    *,
    require_ai_available: bool = False,
    require_ai_policy_completed: bool = False,
    require_ai_finding_available: bool = False,
) -> list[str]:
    metrics = result.metrics
    errors: list[str] = []
    policy_status = str(metrics.get("ai_policy_review_status", "disabled") or "disabled")
    full_status = str(metrics.get("ai_full_review_status", "disabled") or "disabled")
    finding_status = str(metrics.get("ai_finding_review_status", "disabled") or "disabled")

    if require_ai_available:
        if policy_status in {"unavailable", "error", "deferred"}:
            errors.append(f"AI policy review status is {policy_status}.")
        if full_status in {"unavailable", "error", "deferred"}:
            errors.append(f"AI full review status is {full_status}.")
        if finding_status in {"unavailable", "error", "deferred"}:
            errors.append(f"AI finding review status is {finding_status}.")

    if require_ai_policy_completed and policy_status != "completed":
        errors.append(f"AI policy review status is {policy_status}, expected completed.")

    if require_ai_finding_available and finding_status in {"unavailable", "error", "deferred"}:
        errors.append(f"AI finding review status is {finding_status}.")

    return errors
