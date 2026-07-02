from __future__ import annotations

import argparse
from pathlib import Path

from job_runner import ai_verification_errors, write_review_outputs
from models import CompanyProfile, DEFAULT_AI_MODEL, ReviewOptions
from reviewer import findings_to_markdown, review_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Review a prepared financial statement PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF report")
    parser.add_argument("--company-name", default="", help="Expected reporting entity name")
    parser.add_argument("--industry", default="", help="Company industry")
    parser.add_argument("--currency", default="", help="Expected reporting currency marker, such as NGN")
    parser.add_argument("--presentation-standard", default="IFRS", help="Presentation framework, default IFRS")
    parser.add_argument("--expected-policy", action="append", default=[], help="Policy area expected to apply. Can be supplied more than once.")
    parser.add_argument("--significant-transaction", action="append", default=[], help="Significant transaction or balance expected to have tailored accounting policy coverage.")
    parser.add_argument("--checklist-area", action="append", default=[], help="Standard or disclosure area to force into the standards checklist review.")
    parser.add_argument("--ocr", action="store_true", help="Run local OCR if the PDF has low embedded text coverage")
    parser.add_argument("--ocr-max-pages", type=int, default=60, help="Maximum scanned pages to OCR")
    parser.add_argument("--ocr-dpi", type=int, default=200, help="OCR render DPI, usually 150-300")
    parser.add_argument("--ai-policy-review", action="store_true", help="Run optional AI judgement for policy relevance, disclosure completeness, and industry fit")
    parser.add_argument("--ai-full-review", action="store_true", help="Run optional full AI quality-control review over extracted statement text and note context")
    parser.add_argument("--ai-model", default=DEFAULT_AI_MODEL, help="Model to use for optional AI review")
    parser.add_argument("--output", type=Path, help="Optional markdown report path")
    parser.add_argument("--export-dir", type=Path, help="Optional directory for standard backend artifacts (Excel, markdown, JSON summary).")
    parser.add_argument("--skip-excel-export", action="store_true", help="When using --export-dir, skip the Excel exception register export.")
    parser.add_argument("--skip-json-summary", action="store_true", help="When using --export-dir, skip the machine-readable JSON run summary.")
    parser.add_argument("--require-ai-available", action="store_true", help="Fail if AI policy/finding review is unavailable, deferred, or errors.")
    parser.add_argument("--require-ai-policy-completed", action="store_true", help="Fail unless AI policy review status is completed.")
    parser.add_argument("--require-ai-finding-available", action="store_true", help="Fail if AI finding review is unavailable, deferred, or errors.")
    args = parser.parse_args()

    profile = CompanyProfile(
        company_name=args.company_name,
        industry=args.industry,
        reporting_currency=args.currency,
        expected_policies=tuple(args.expected_policy),
        significant_transactions=tuple(args.significant_transaction),
        presentation_standard=args.presentation_standard,
        checklist_areas=tuple(args.checklist_area),
    )
    options = ReviewOptions(
        use_ocr=args.ocr,
        ocr_max_pages=args.ocr_max_pages,
        ocr_dpi=args.ocr_dpi,
        use_ai_policy_review=args.ai_policy_review,
        use_ai_full_review=args.ai_full_review,
        ai_model=args.ai_model,
    )

    summary = None
    if args.export_dir:
        result, summary = write_review_outputs(
            args.pdf,
            profile,
            options,
            args.export_dir,
            write_excel=not args.skip_excel_export,
            write_markdown=True,
            write_json_summary=not args.skip_json_summary,
        )
        print(f"Exported review artifacts to {args.export_dir}")
        for key, value in (summary.get("output_files") or {}).items():
            print(f"{key}: {value}")
        print(
            "AI status:",
            f"policy={summary.get('ai_policy_review_status', 'disabled')},",
            f"full={summary.get('ai_full_review_status', 'disabled')},",
            f"finding={summary.get('ai_finding_review_status', 'disabled')}",
        )
    else:
        result = review_pdf(args.pdf, profile, options)

    report = findings_to_markdown(result)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    elif not args.export_dir:
        print(report)

    ai_errors = ai_verification_errors(
        result,
        require_ai_available=args.require_ai_available,
        require_ai_policy_completed=args.require_ai_policy_completed,
        require_ai_finding_available=args.require_ai_finding_available,
    )
    if ai_errors:
        for error in ai_errors:
            print(f"AI verification failed: {error}")
        return 2

    return 1 if result.metrics["high"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
