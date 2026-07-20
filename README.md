# Financial Statement Reviewer

This is a local AI-style audit assistant for prepared PDF financial statements. It extracts PDF text and tables, then reviews four dimensions:

- Totals and rounding: totals, subtotals, cross-footings, duplicate totals, and consistent rounding labels such as `$000s`, thousands, or millions.
- Formatting: inconsistent number formats, missing brackets for negatives, mixed currency symbols, missing comparative periods, and IFRS statement heading checks.
- Notes agreement: cross-references, note totals, face statement amounts, segment totals, EPS calculations, tax notes, and depreciation charge agreement.
- Accounting policies: irrelevant policies for the company context, boilerplate wording, missing policies for significant transactions, and references to superseded standards.
- Standards checklist: triggered IFRS disclosure checks for IAS 1, IFRS 15, IFRS 16, IFRS 7 / IFRS 9, IAS 12, IAS 16, IAS 38, IAS 36, IAS 33, IFRS 8, IAS 24, IAS 10, and IAS 37.

The tool is designed as a review assistant. It does not replace professional judgement, source working papers, or a disclosure checklist.

## Run the app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI API setup

For direct OpenAI API usage, configure these environment variables. Do not commit API keys to Git. See `.env.example` for a safe placeholder template.

Local PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-proj-your-openai-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_API_STYLE="responses"
$env:OPENAI_STRUCTURED_OUTPUTS="1"
$env:OPENAI_QUICK_REVIEW_MODEL="gpt-5-mini"
$env:OPENAI_STANDARD_REVIEW_MODEL="gpt-5.1"
$env:OPENAI_DEEP_REVIEW_MODEL="gpt-5.1"
$env:OPENAI_SAFE_FALLBACK_MODEL="gpt-4o-mini"
$env:OPENAI_MAX_ATTEMPTS="2"
$env:OPENAI_RETRY_BACKOFF_SECONDS="4,8"
$env:OPENAI_REQUEST_TIMEOUT_SECONDS="12"
$env:OPENAI_REQUEST_LOCK_TIMEOUT_SECONDS="30"
$env:OPENAI_AI_PIPELINE_LOCK_TIMEOUT_SECONDS="30"
$env:OPENAI_MODEL_FALLBACK_LIMIT="2"
```

Deployment environment variables:

```env
OPENAI_API_KEY=sk-proj-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_STYLE=responses
OPENAI_STRUCTURED_OUTPUTS=1
OPENAI_QUICK_REVIEW_MODEL=gpt-5-mini
OPENAI_STANDARD_REVIEW_MODEL=gpt-5.1
OPENAI_DEEP_REVIEW_MODEL=gpt-5.1
OPENAI_SAFE_FALLBACK_MODEL=gpt-4o-mini
OPENAI_MAX_ATTEMPTS=2
OPENAI_RETRY_BACKOFF_SECONDS=4,8
OPENAI_REQUEST_TIMEOUT_SECONDS=12
OPENAI_REQUEST_LOCK_TIMEOUT_SECONDS=30
OPENAI_AI_PIPELINE_LOCK_TIMEOUT_SECONDS=30
OPENAI_MODEL_FALLBACK_LIMIT=2
```

If you change providers later, keep the provider base URL in `OPENAI_BASE_URL`; for OpenAI direct usage it must be `https://api.openai.com/v1`, not a dashboard URL and not a router URL.

The automatic Quick AI review is intentionally bounded by the timeout and fallback settings above so deterministic exports are not delayed for several minutes. Increase these values only for deliberate Deep/partner-style testing.

## Run from the command line

Basic markdown run:

```powershell
python cli.py "C:\path\to\financial_statement.pdf" --company-name "Example Plc" --industry "Technology" --currency NGN --significant-transaction leases --checklist-area "IFRS 16" --output review.md
```

For scanned or signed image-based PDFs, enable local OCR:

```powershell
python cli.py "C:\path\to\signed_afs.pdf" --ocr --ocr-max-pages 60 --ocr-dpi 200 --output review.md
```

For backend verification without the Streamlit uploader, export the standard artifacts directly from the real review pipeline:

```powershell
python cli.py "C:\path\to\financial_statement.pdf" `
  --ocr `
  --ai-policy-review `
  --export-dir .\scratch\verification_run `
  --require-ai-available
```

That command writes:

- Excel exception register
- Markdown debug report
- JSON run summary with AI/hybrid statuses

The JSON summary is useful for automated checks because it records:

- findings counts
- checks performed / passed / skipped
- AI policy review status
- AI finding review status
- generated output file paths

If you want stricter AI verification, you can require a completed AI policy pass:

```powershell
python cli.py "C:\path\to\financial_statement.pdf" --ai-policy-review --export-dir .\scratch\ai_check --require-ai-policy-completed
```

OCR uses local Tesseract. The app renders pages in memory and does not save page images or OCR text unless you explicitly download a review report.

## Review logic

The checks are intentionally transparent and conservative:

- table totals and row cross-footings are compared against visible component rows using a one-unit presentation tolerance;
- note references are matched with note headings found in the extracted text;
- note sections are searched for internal total agreement and targeted EPS, tax, depreciation, and segment indicators;
- policy relevance is checked using keyword sets for common IFRS policy areas, company industry, expected policy areas, and significant transactions.
- checklist items are activated by detected balances/transactions or by forced checklist areas supplied by the reviewer.
- low-text PDFs are routed through an OCR fallback when enabled; otherwise the app raises an extraction-quality finding instead of producing misleading accounting exceptions.
- OCR table reconstruction uses word positions and line-level numeric patterns to rebuild candidate rows such as `Revenue | 10,000 | 8,000`, allowing totals and cross-footing checks to run on scanned statements where OCR quality is sufficient.
- extraction confidence scores combine text coverage, OCR status, unreadable placeholders, and suspicious merged numeric cells. Low-confidence extraction stops deterministic audit checks and raises extraction-quality findings instead of producing unreliable exceptions.
- findings are evidence-gated before export: weak review prompts remain visible in `Review prompts not elevated`, but they do not inflate the Exception register unless page/note evidence and confidence are sufficient.
- optional AI review is used as an adjudicator over evidence packs for policy/disclosure judgement and likely false-positive review; it should not be treated as the source of truth without deterministic evidence.

## Regression tests

Run the reusable safety net before pushing changes:

```powershell
python -m pytest tests\test_pipeline_regression.py tests\test_synthetic_regression_fixtures.py tests\test_quality_gates.py tests\test_job_runner.py -q
```

or use:

```powershell
.\run_regression_tests.ps1
```

The tests cover local PDF cases when available, generic synthetic financial-statement patterns, workbook export creation, and evidence-gating rules that prevent low-confidence prompts from becoming exception-register findings.

For a practical folder-level smoke test against real local PDFs, use the batch verifier. It runs the same backend pipeline as the UI/CLI, writes each normal workbook output, and creates `batch_summary.json` plus `batch_summary.csv`:

```powershell
python batch_verify.py `
  --input-dir "C:\Users\ionawoga\Downloads\audit assistant test files" `
  --output-dir .\scratch\batch_verify `
  --ocr `
  --ocr-max-pages 300 `
  --ocr-dpi 300 `
  --max-medium 3 `
  --forbid-finding "For:Kreston"
```

Use `--limit 1` for a quick single-file smoke test. By default the batch fails if any file produces High findings; use `--allow-high` only when you intentionally want to inspect High findings without failing the batch.

## Extraction confidence

The tool classifies uploaded PDFs as `text-based`, `partially scanned`, `image-only`, `ocr-assisted`, or `empty`.

It flags extraction quality issues when:

- text coverage is too low;
- OCR fails or only covers part of the report;
- values are unreadable or contain placeholders such as `#####`;
- replacement characters or blanked-out value markers appear;
- table cells appear to contain merged numeric values.

If extraction confidence is below the reliability threshold, the app stops before arithmetic, note-agreement, policy, and checklist checks. This prevents the exception register from presenting poor extraction as audit evidence.

The bundled checklist is a starter engine, not a full licensed IFRS disclosure checklist. Expand `STANDARD_CHECKLIST` in `reviewer.py` with your firm's detailed disclosure requirements and evidence phrases.

## Code structure

- `models.py` contains shared dataclasses for documents, findings, profile inputs, and review options.
- `extraction.py` handles PDF text extraction, OCR fallback, and OCR table reconstruction.
- `reviewer.py` coordinates deterministic review checks and report text.
- `job_runner.py` writes standard backend artifacts and AI verification summaries without the UI.
- `app.py` provides the Streamlit interface.
- `cli.py` provides batch review from the command line.

## OCR prerequisites

Install Tesseract OCR locally and ensure `tesseract.exe` is on PATH. On this workstation the app also checks the standard Windows install path:

```text
C:\Users\<user>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
```

PDF extraction quality depends on the source PDF. Scanned image-only reports need OCR before this tool can review them.
