from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_checks import run_canonical_checks
from canonical_extraction import extract_statement_facts
from canonical_workbook import write_canonical_review_workbook


def _extract_document(pdf_path: Path, *, use_ocr: bool = False, ocr_max_pages: int = 60, ocr_dpi: int = 300):
    # Prefer the existing high-level extractor so this command benefits from the
    # repo's extraction confidence and OCR routing. Fall back to extraction.py if
    # reviewer.py is unavailable during isolated tests.
    try:
        from models import ReviewOptions
        from reviewer import extract_review_document

        return extract_review_document(pdf_path, ReviewOptions(use_ocr=use_ocr, ocr_max_pages=ocr_max_pages, ocr_dpi=ocr_dpi))
    except Exception:
        from extraction import extract_pdf, extract_pdf_with_ocr
        from models import ReviewOptions

        base = extract_pdf(pdf_path)
        if use_ocr:
            return extract_pdf_with_ocr(pdf_path, base, ReviewOptions(use_ocr=True, ocr_max_pages=ocr_max_pages, ocr_dpi=ocr_dpi))
        return base


def run_canonical_qc(
    pdf_path: str | Path,
    *,
    output_path: str | Path,
    template_path: str | Path | None = None,
    use_ocr: bool = False,
    ocr_max_pages: int = 60,
    ocr_dpi: int = 300,
    debug_json_path: str | Path | None = None,
) -> Path:
    pdf = Path(pdf_path)
    document = _extract_document(pdf, use_ocr=use_ocr, ocr_max_pages=ocr_max_pages, ocr_dpi=ocr_dpi)
    facts = extract_statement_facts(document)
    findings, check_results, audit_rows = run_canonical_checks(document, facts)
    output = write_canonical_review_workbook(
        findings=findings,
        check_results=check_results,
        audit_rows=audit_rows,
        output_path=output_path,
        template_path=template_path,
    )
    if debug_json_path:
        payload = {
            "pdf_path": str(pdf),
            "output_path": str(output),
            "facts": [fact.__dict__ | {"amount": str(fact.amount)} for fact in facts],
            "checks": [result.to_row() for result in check_results],
            "findings": [finding.__dict__ for finding in findings],
        }
        Path(debug_json_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical financial-statement QC checks and write an Excel review workbook.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF financial statements.")
    parser.add_argument("--template", type=Path, help="Optional existing Excel review-comment workbook to append to.")
    parser.add_argument("--output", type=Path, required=True, help="Output Excel workbook path.")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR fallback for low-text PDFs.")
    parser.add_argument("--ocr-max-pages", type=int, default=60)
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--debug-json", type=Path, help="Optional JSON dump of facts, checks and findings.")
    args = parser.parse_args()

    output = run_canonical_qc(
        args.pdf,
        output_path=args.output,
        template_path=args.template,
        use_ocr=args.ocr,
        ocr_max_pages=args.ocr_max_pages,
        ocr_dpi=args.ocr_dpi,
        debug_json_path=args.debug_json,
    )
    print(f"Canonical QC workbook written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
