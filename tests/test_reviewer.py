from decimal import Decimal

from extraction import _line_to_table_row, _reconstruct_ocr_tables
from models import CompanyProfile, PdfDocument, PdfPage, ReviewOptions
from reviewer import (
    check_formatting,
    check_extraction_quality,
    check_notes_agreement,
    check_policy_relevance,
    check_rounding_and_casting,
    check_standard_checklist,
    review_pdf,
)


def test_rounding_check_flags_bad_visible_total():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [
                        ["Description", "2025"],
                        ["Revenue", "100"],
                        ["Other income", "50"],
                        ["Total income", "160"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert findings
    assert findings[0].category == "Totals and rounding"


def test_note_column_is_not_treated_as_amount_column():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [
                        ["Description", "Note", "2025", "2024"],
                        ["Revenue", "5", "100", "90"],
                        ["Other income", "6", "50", "40"],
                        ["Total income", "99", "150", "130"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert findings == []


def test_notes_check_flags_missing_note_heading():
    document = PdfDocument([PdfPage(1, "Revenue Note 7 100\nProfit for the year 40", [])])

    findings = check_notes_agreement(document)

    assert any("note 7" in finding.issue.lower() for finding in findings)


def test_policy_check_flags_boilerplate_lease_policy():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Accounting policies\nIFRS 16 is applied to lease contracts at commencement date.\nCash and cash equivalents 500",
                [],
            )
        ]
    )

    findings = check_policy_relevance(document, CompanyProfile())

    assert any("leases policy" in finding.issue.lower() for finding in findings)


def test_policy_check_flags_industry_mismatch_and_superseded_standard():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Accounting policies\nBiological assets are measured under IAS 41.\nLeases are accounted for under IAS 17.",
                [],
            )
        ]
    )

    findings = check_policy_relevance(document, CompanyProfile(industry="Technology"))

    assert any("inconsistent with the stated industry" in finding.issue.lower() for finding in findings)
    assert any("superseded" in finding.issue.lower() for finding in findings)


def test_formatting_flags_missing_comparative_period():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position 2025\nRevenue 100\nTotal assets 500",
                [],
            )
        ]
    )

    findings = check_formatting(document, CompanyProfile())

    assert any("only one reporting year" in finding.issue.lower() for finding in findings)


def test_notes_check_flags_eps_formula_difference():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nProfit for the year Note 9 100\nNotes to the financial statements\n9 Earnings per share\nProfit attributable 100\nWeighted average shares 20\nBasic EPS 8",
                [],
            )
        ]
    )

    findings = check_notes_agreement(document)

    assert any("eps calculation" in finding.issue.lower() for finding in findings)


def test_standard_checklist_flags_revenue_disclosure_gap():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 500\nNotes to the financial statements\nRevenue is recognised when control transfers.",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile())

    assert any(finding.category == "Standards checklist" and "IFRS 15" in finding.location for finding in findings)


def test_standard_checklist_can_be_forced_by_area():
    document = PdfDocument([PdfPage(1, "Notes to the financial statements\nLease expense 20", [])])

    findings = check_standard_checklist(document, CompanyProfile(checklist_areas=("IFRS 16",)))

    assert any("IFRS 16" in finding.location for finding in findings)


def test_standard_checklist_disabled_for_local_gaap():
    document = PdfDocument([PdfPage(1, "Revenue 500", [])])

    findings = check_standard_checklist(document, CompanyProfile(presentation_standard="Local GAAP"))

    assert findings == []


def test_extraction_quality_flags_scanned_pdf_like_document():
    document = PdfDocument([PdfPage(1, "", []), PdfPage(2, "", [])])

    findings = check_extraction_quality(document)

    assert findings
    assert findings[0].category == "Extraction quality"


def test_review_options_enable_ocr_without_changing_defaults():
    options = ReviewOptions(use_ocr=True, ocr_max_pages=10, ocr_dpi=150)

    assert options.use_ocr is True
    assert options.ocr_max_pages == 10
    assert options.ocr_dpi == 150


def test_ocr_line_to_table_row_splits_label_and_amounts():
    row = _line_to_table_row("Revenue from contracts 12,500 10,250")

    assert row == ["Revenue from contracts", "12,500", "10,250"]


def test_ocr_line_to_table_row_drops_note_reference_column():
    row = _line_to_table_row("Cash and cash equivalents 24 11,343,889 5,102,400")

    assert row == ["Cash and cash equivalents", "11,343,889", "5,102,400"]


def test_ocr_line_to_table_row_ignores_docusign_header():
    row = _line_to_table_row("DocuSign Envelope ID 4895-2047-947")

    assert row is None


def test_ocr_table_reconstruction_builds_candidate_table():
    lines = [
        {"text": "Revenue 100 90"},
        {"text": "Cost of sales (60) (50)"},
        {"text": "Total profit 40 40"},
    ]

    tables = _reconstruct_ocr_tables(lines)

    assert tables == [[["Revenue", "100", "90"], ["Cost of sales", "(60)", "(50)"], ["Total profit", "40", "40"]]]


def test_metrics_include_ocr_table_count():
    document = PdfDocument(
        [PdfPage(1, "Revenue 100\n" * 120, [[["Revenue", "100"], ["Total revenue", "100"]]])],
        ocr_used=True,
        ocr_pages=1,
        ocr_tables=1,
    )

    findings = check_extraction_quality(document)

    assert findings[0].severity == "Low"
    assert "reconstructed 1 table" in findings[0].evidence
    assert document.ocr_tables == 1
