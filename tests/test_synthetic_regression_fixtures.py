from __future__ import annotations

from models import PdfDocument, PdfPage
from reviewer import check_rounding_and_casting, infer_detected_profile


def _primary_statement_page() -> PdfPage:
    return PdfPage(
        1,
        "\n".join(
            [
                "Statement of financial position",
                "Cash and cash equivalents 160 120",
                "Trade and other receivables 150 130",
                "Total assets 310 250",
                "Trade and other payables 70 55",
                "Borrowings 80 70",
                "Total liabilities 150 125",
                "Share capital 100 100",
                "Retained earnings 60 25",
                "Total equity 160 125",
                "Total equity and liabilities 310 250",
            ]
        ),
        [],
    )


def _poor_grid_note_page(page_number: int, heading: str, lines: list[str]) -> PdfPage:
    return PdfPage(
        page_number,
        "\n".join(["Notes to the financial statements", heading, *lines]),
        [[["Financial Statements for the year ended", "31", "2025"], ["N '", "000", "000"], ["Header fragment", "", ""]]],
    )


def _clean_grid_note_page(page_number: int, heading: str, rows: list[list[str]]) -> PdfPage:
    return PdfPage(page_number, f"Notes to the financial statements\n{heading}", [rows])


def test_generic_private_company_simple_notes_cast_from_mixed_table_quality():
    document = PdfDocument(
        [
            _primary_statement_page(),
            _clean_grid_note_page(
                10,
                "3. Revenue",
                [["Revenue", "2025", "2024"], ["Service income", "700", "600"], ["Product income", "300", "250"], ["Total", "1,000", "850"]],
            ),
            _poor_grid_note_page(
                11,
                "4. Cash and cash equivalents",
                ["Bank balances 110 80", "Short-term deposits 50 40", "Total 160 120"],
            ),
            _poor_grid_note_page(
                12,
                "5. Trade and other receivables",
                ["Trade receivables 100 90", "Prepayments 50 40", "Total 150 130"],
            ),
            _poor_grid_note_page(
                13,
                "6. Trade and other payables",
                ["Accruals 40 30", "Statutory payables 30 25", "Total 70 55"],
            ),
            _poor_grid_note_page(
                14,
                "7. Borrowings",
                ["Bank loan 60 50", "Other loan 20 20", "Total 80 70"],
            ),
        ]
    )

    findings = check_rounding_and_casting(document)
    failed = [finding for finding in findings if finding.severity != "Passed"]
    passed = [finding.issue for finding in findings if finding.severity == "Passed"]

    assert not failed
    assert any("Simple note table on Page 10" in issue for issue in passed)
    assert sum("Simple note section" in issue for issue in passed) == 4


def test_generic_simple_note_wrong_total_becomes_reviewable_exception():
    document = PdfDocument(
        [
            _primary_statement_page(),
            _poor_grid_note_page(
                10,
                "8. Administrative expenses",
                ["Audit fees 70 60", "Professional fees 30 25", "Total 105 85"],
            ),
        ]
    )

    findings = check_rounding_and_casting(document)

    assert any(
        finding.severity == "Medium"
        and "Simple note section total does not agree" in finding.issue
        and "Page 10" in finding.location
        for finding in findings
    )


def test_generic_simple_note_text_handles_dash_zero_note_refs_and_rolling_subtotals():
    document = PdfDocument(
        [
            _primary_statement_page(),
            _poor_grid_note_page(
                10,
                "8. Direct expenses",
                [
                    "Agents commission 2,247,050 2,005,494",
                    "Draw wins 1,948,705 1,921,200",
                    "Guaranteed cashback 1,919,434 1,677,470",
                    "Regulatory fees - 57,636",
                    "Web hosting and domains 193,277 232,415",
                    "6,308,466 5,894,215",
                ],
            ),
            _poor_grid_note_page(
                11,
                "9. Other operating gains and losses",
                [
                    "(Losses)/gains on disposals",
                    "Property, plant and equipment 3 (396) 1,110",
                    "Foreign exchange losses",
                    "Net foreign exchange loss (2,627) (35,826)",
                    "Total other operating gains (losses) (3,023) (34,716)",
                ],
            ),
            _poor_grid_note_page(
                12,
                "10. Employee costs",
                [
                    "Salaries and wages 498,828 362,682",
                    "Leave allowance 14,400 14,400",
                    "Pension 26,952 19,461",
                    "540,180 396,543",
                    "ITF expenses 9,466 3,538",
                    "NSITF expenses 4,900 3,538",
                    "554,546 403,619",
                    "The table shows the number of employees whose earnings fell within the ranges shown below:",
                    "N0 - N1,000,000 8 11",
                    "71 58",
                ],
            ),
        ]
    )

    findings = check_rounding_and_casting(document)
    failed = [finding for finding in findings if finding.severity != "Passed"]
    passed = [finding.issue for finding in findings if finding.severity == "Passed"]

    assert not failed
    assert sum("Simple note section" in issue for issue in passed) == 3


def test_generic_complex_notes_remain_outside_targeted_casting():
    document = PdfDocument(
        [
            _primary_statement_page(),
            _poor_grid_note_page(
                10,
                "9. Financial instruments and risk management",
                ["Loans receivable 100 100", "Cash and cash equivalents 50 50", "Total credit exposure 150 150"],
            ),
            _poor_grid_note_page(
                11,
                "10. Property, plant and equipment",
                ["Cost 500 450", "Accumulated depreciation (100) (80)", "Carrying amount 400 370"],
            ),
            _poor_grid_note_page(
                12,
                "11. Share capital",
                ["400,000 ordinary shares of N0.50 each 200 200", "Total 200 200"],
            ),
        ]
    )

    findings = check_rounding_and_casting(document)

    assert not any("Simple note section" in finding.issue for finding in findings)
    assert not any("Simple note table" in finding.issue for finding in findings)


def test_generic_professional_body_fixture_is_detected_without_name_hardcoding():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "SYNTHETIC INSTITUTE OF ACCOUNTABILITY",
                        "Statement of income and expenditure",
                        "Subscriptions 1,200 1,100",
                        "Members' fund 5,000 4,700",
                        "Accumulated fund 5,000 4,700",
                        "Principal activities are membership services, training, certification and professional development for fellows and associates.",
                    ]
                ),
                [],
            )
        ]
    )

    profile = infer_detected_profile(document)

    assert profile["Entity type"] == "Non-profit / professional body"
    assert "Professional membership body" in profile["Principal activities"]


def test_generic_scanned_private_company_fixture_keeps_ocr_arithmetic_cautious():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "SYNTHETIC PROPERTY LIMITED",
                        "Statement of financial position",
                        "Investment property 4,000 3,900",
                        "Cash and cash equivalents 100 90",
                        "Total assets 4,100 3,990",
                        "Share capital 1,000 1,000",
                        "Financial liabilities 3,100 2,990",
                        "Total equity and liabilities 4,100 3,990",
                    ]
                ),
                [[["OCR", "table", "candidate"]]],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    profile = infer_detected_profile(document)
    findings = check_rounding_and_casting(document)

    assert "Private company" in profile["Entity type"]
    assert not any(finding.severity == "High" for finding in findings)
