---
name: financial-statement-canonical-qc
description: Run canonical deterministic QC checks on IFRS financial statement PDFs and produce an Excel workbook with Review Comments, Summary, Recalculation Checks and Extraction Audit sheets.
---

When this skill is invoked:

1. Locate the financial-statement PDF and optional Excel review template.
2. Run `python canonical_qc.py <pdf> --template <template.xlsx> --output <output.xlsx>`.
3. Use `--ocr` only when the PDF has poor embedded text.
4. Review the `Extraction Audit` sheet before relying on deterministic findings.
5. Confirm that high-priority findings show reported amount, expected amount, difference, formula and source rows.
6. Do not delete the user's existing review comments.
7. Return the output workbook path and a short summary of high/medium/low findings.
