from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from job_runner import write_review_outputs
from models import CompanyProfile, ReviewOptions


@dataclass
class BatchGate:
    fail_on_high: bool = True
    max_medium: int | None = None
    max_findings: int | None = None
    forbidden_issue_text: tuple[str, ...] = ()


def _safe_folder_name(path: Path) -> str:
    stem = path.stem.strip() or "financial_statement"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    cleaned = "_".join(part for part in cleaned.split("_") if part) or "financial_statement"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:36].rstrip('_')}_{digest}"


def _iter_pdfs(input_dir: Path, pattern: str, limit: int | None = None) -> list[Path]:
    pdfs = sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".pdf")
    if limit is not None and limit > 0:
        return pdfs[:limit]
    return pdfs


def _finding_texts(result) -> list[str]:
    return [
        " | ".join(
            str(value or "")
            for value in (
                finding.severity,
                finding.category,
                finding.location,
                finding.issue,
                finding.evidence,
            )
        )
        for finding in result.findings
    ]


def _gate_failures(result, gate: BatchGate) -> list[str]:
    failures: list[str] = []
    metrics = result.metrics
    high = int(metrics.get("high", 0) or 0)
    medium = int(metrics.get("medium", 0) or 0)
    findings = int(metrics.get("findings", len(result.findings)) or 0)
    if gate.fail_on_high and high > 0:
        failures.append(f"high findings present: {high}")
    if gate.max_medium is not None and medium > gate.max_medium:
        failures.append(f"medium findings {medium} exceed limit {gate.max_medium}")
    if gate.max_findings is not None and findings > gate.max_findings:
        failures.append(f"findings {findings} exceed limit {gate.max_findings}")
    combined = "\n".join(_finding_texts(result)).lower()
    for snippet in gate.forbidden_issue_text:
        if snippet.lower() in combined:
            failures.append(f"forbidden finding text found: {snippet}")
    return failures


def run_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.pdf",
    options: ReviewOptions | None = None,
    profile: CompanyProfile | None = None,
    gate: BatchGate | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, object]], bool]:
    options = options or ReviewOptions()
    profile = profile or CompanyProfile()
    gate = gate or BatchGate()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for pdf_path in _iter_pdfs(input_dir, pattern, limit):
        run_dir = output_dir / _safe_folder_name(pdf_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, object] = {
            "pdf": str(pdf_path),
            "output_dir": str(run_dir),
            "status": "error",
            "gate_passed": False,
            "error": "",
        }
        try:
            result, summary = write_review_outputs(pdf_path, profile, options, run_dir)
            failures = _gate_failures(result, gate)
            row.update(
                {
                    "status": "completed",
                    "company_name": summary.get("company_name", ""),
                    "findings": summary.get("findings", 0),
                    "high_findings": summary.get("high_findings", 0),
                    "medium_findings": summary.get("medium_findings", 0),
                    "low_findings": summary.get("low_findings", 0),
                    "checks_performed": summary.get("checks_performed", 0),
                    "checks_passed": summary.get("checks_passed", 0),
                    "checks_skipped": summary.get("checks_skipped", 0),
                    "excel": (summary.get("output_files", {}) or {}).get("excel", ""),
                    "markdown": (summary.get("output_files", {}) or {}).get("markdown", ""),
                    "json_summary": (summary.get("output_files", {}) or {}).get("json_summary", ""),
                    "gate_passed": not failures,
                    "gate_failures": "; ".join(failures),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI use; kept defensive for batch runs.
            row.update({"error": f"{type(exc).__name__}: {exc}", "gate_failures": "processing error"})
        rows.append(row)

    _write_batch_outputs(output_dir, rows)
    all_passed = bool(rows) and all(bool(row.get("gate_passed")) for row in rows)
    return rows, all_passed


def _write_batch_outputs(output_dir: Path, rows: list[dict[str, object]]) -> None:
    json_path = output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_dir / "batch_summary.csv"
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Iterable[dict[str, object]], all_passed: bool) -> None:
    rows = list(rows)
    print(f"Batch status: {'PASSED' if all_passed else 'FAILED'}")
    print(f"Files tested: {len(rows)}")
    for row in rows:
        print(
            " - {name}: {status}, findings={findings}, high={high}, medium={medium}, skipped={skipped}, gate={gate}{failures}".format(
                name=Path(str(row.get("pdf", ""))).name,
                status=row.get("status", ""),
                findings=row.get("findings", ""),
                high=row.get("high_findings", ""),
                medium=row.get("medium_findings", ""),
                skipped=row.get("checks_skipped", ""),
                gate="PASS" if row.get("gate_passed") else "FAIL",
                failures=(f" ({row.get('gate_failures')})" if row.get("gate_failures") else ""),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real financial statement review pipeline against a local folder of PDFs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing local PDF test files.")
    parser.add_argument("--output-dir", type=Path, default=Path("scratch") / "batch_verify", help="Folder for generated Excel/Markdown/JSON outputs.")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern to select PDFs from input-dir.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of PDFs to process.")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR fallback for low-text PDFs.")
    parser.add_argument("--ocr-max-pages", type=int, default=300)
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--allow-high", action="store_true", help="Do not fail the batch solely because high findings are present.")
    parser.add_argument("--max-medium", type=int, default=None, help="Fail if a file has more than this many medium findings.")
    parser.add_argument("--max-findings", type=int, default=None, help="Fail if a file has more than this many total findings.")
    parser.add_argument("--forbid-finding", action="append", default=[], help="Fail if finding text contains this snippet. Can be supplied more than once.")
    args = parser.parse_args()

    options = ReviewOptions(use_ocr=args.ocr, ocr_max_pages=args.ocr_max_pages, ocr_dpi=args.ocr_dpi)
    gate = BatchGate(
        fail_on_high=not args.allow_high,
        max_medium=args.max_medium,
        max_findings=args.max_findings,
        forbidden_issue_text=tuple(args.forbid_finding),
    )
    rows, all_passed = run_batch(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        options=options,
        gate=gate,
        limit=args.limit,
    )
    _print_summary(rows, all_passed)
    print(f"Summary files: {args.output_dir / 'batch_summary.json'} and {args.output_dir / 'batch_summary.csv'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
