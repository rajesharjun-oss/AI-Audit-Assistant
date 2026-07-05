# Canonical financial-statement QC layer

This package adds a validation-first deterministic layer beside the existing AI Audit Assistant pipeline.

## Run

```powershell
python canonical_qc.py "C:\path\to\financial_statements.pdf" `
  --template "C:\path\to\Review Comments.xlsx" `
  --output "C:\path\to\Updated Review Comments.xlsx" `
  --debug-json "C:\path\to\canonical_debug.json"
```

For scanned PDFs:

```powershell
python canonical_qc.py "C:\path\to\financial_statements.pdf" --ocr --ocr-max-pages 300 --ocr-dpi 300 --output canonical_qc.xlsx
```

## Output sheets

- `Review Comments`: appended findings in the standard audit review format.
- `Summary`: total findings, priority/category counts, cash-flow conclusion and casting/cross-casting conclusion.
- `Recalculation Checks`: deterministic pass/fail checks with reported amount, expected amount, difference, formula and source rows.
- `Extraction Audit`: parsed facts by page, statement, entity, year, line item, note reference and amount.

## Key design

- `amount_parser.py` prevents note numbers, page numbers, years, percentages and registration identifiers from becoming amounts.
- `canonical_extraction.py` maps statement rows to canonical facts by entity/year, including Group/Company and 2025/2024 columns.
- `canonical_checks.py` performs deterministic SFP, P&L, cash-flow and note-reference compatibility checks.
- `canonical_workbook.py` appends findings to an existing review template or creates a new workbook.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_amount_parser.py tests/test_canonical_extraction.py tests/test_canonical_checks.py -q
```
