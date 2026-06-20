Local regression fixtures

Drop JSON case manifests into `local_test_fixtures/cases/`.

These files are intentionally ignored by git so you can point them at local PDFs without committing client documents or workstation-specific paths.

Run:

```powershell
python -m pytest
```

Case manifest fields:

```json
{
  "name": "sample_case",
  "pdf_path": "C:/absolute/path/to/sample.pdf",
  "profile": {
    "presentation_standard": "IFRS",
    "reporting_currency": "NGN"
  },
  "options": {
    "use_ocr": true,
    "ocr_max_pages": 60,
    "ocr_dpi": 300,
    "run_cautious_note_agreement": true,
    "use_ai_policy_review": false
  },
  "assertions": {
    "required_sheets": [
      "Detected profile",
      "Summary",
      "Findings summary",
      "Exception register",
      "Checks performed",
      "Checks skipped"
    ],
    "expected_summary": {
      "High findings": 0
    },
    "max_summary": {
      "Medium findings": 3
    },
    "min_summary": {
      "Checks performed": 4
    },
    "required_checks_performed": [
      "Statement of cash flows: opening plus movement checked to closing."
    ],
    "forbidden_checks_skipped": [
      "operating/investing/financing/movement rows were not confidently parsed"
    ],
    "required_findings": [
      "Statement of cash flows page carries incorrect"
    ],
    "forbidden_findings": [
      "Current tax receivable references Note 22"
    ]
  }
}
```

The integration test:

- runs `review_pdf(...)`
- builds the normal Excel exception register
- saves it to a temp output path
- asserts sheets, summary metrics, findings, and skipped checks from the generated workbook
