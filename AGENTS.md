# AI Audit Assistant contributor instructions

Use the canonical financial-statement QC layer for deterministic audit-quality checks. The existing heuristic reviewer and AI review remain useful, but arithmetic findings should be generated from canonical facts where possible.

Project rules:

1. Use code, not AI, for arithmetic checks.
2. Every deterministic finding must show page/source rows, reported amount, expected amount, difference and formula.
3. Every parsed amount should pass through `amount_parser.parse_amount_cell` or `amount_parser.extract_amount_cells`.
4. Do not treat note numbers, reporting years, page numbers, percentages or registration identifiers as monetary amounts.
5. Low-confidence parses should be visible in `Extraction Audit`, not elevated as high-confidence findings.
6. When updating an existing review workbook, append findings and do not delete existing comments.

Run the focused test suite before merging canonical QC changes:

```bash
python -m pytest tests/test_amount_parser.py tests/test_canonical_extraction.py tests/test_canonical_checks.py -q
```
