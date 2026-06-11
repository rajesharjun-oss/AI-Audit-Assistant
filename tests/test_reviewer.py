from decimal import Decimal
from pathlib import Path

import extraction
import reviewer
from cross_page_consistency import _names_look_like_spelling_variants, check_cross_page_consistency
from extraction import _line_to_table_row, _reconstruct_ocr_tables, extract_pdf_with_ocr
from models import CompanyProfile, PdfDocument, PdfPage, ReviewOptions
from reviewer import (
    _note_headings,
    _note_headings_by_page,
    _amounts_from_statement_line,
    check_formatting,
    check_extraction_quality,
    check_notes_agreement,
    check_primary_statement_consistency,
    check_policy_relevance,
    check_rounding_and_casting,
    check_standard_checklist,
    build_ai_review_memo,
    infer_detected_profile,
    normalize_reporting_currency,
    review_pdf,
)


def test_name_consistency_only_flags_typo_like_variants_not_joined_names():
    assert _names_look_like_spelling_variants("Mzer Michael Terungwa", "Mzer Micheal Terungwa")
    assert _names_look_like_spelling_variants("Lai Labode", "Lait Labode")

    assert not _names_look_like_spelling_variants("Aliyu Ibrahim Bala", "Aliyu Ibrahim Bala Edeh")
    assert not _names_look_like_spelling_variants("Edeh Anthony Uzodinma", "Anthony Uzodinma")
    assert not _names_look_like_spelling_variants("Adeoye Simileoluwa", "Simileoluwa Adeoye")
    assert not _names_look_like_spelling_variants("Lai Labode", "Lai Labode Stanley Emurotu")


def test_name_consistency_suppresses_single_page_ocr_artifact_when_canonical_exists_same_page():
    document = PdfDocument(
        [
            PdfPage(2, "Directors Lai Labode", []),
            PdfPage(5, "Directors\nLait Labode Chief Executive Officer\nShareholding\nLai Labode 1,027,442", []),
            PdfPage(41, "Related parties\nLai Labode, a major shareholder and director", []),
        ]
    )

    findings, export = check_cross_page_consistency(document)

    assert not any("Lait Labode" in finding.issue for finding in findings)
    assert not any("Lait Labode" in str(row) for row in export["names"])


def test_excel_name_export_guard_suppresses_one_page_ocr_artifact():
    from export_utils import clean_name_consistency_rows

    rows = [
        {
            "Name variant 1": "Lai Labode",
            "Page 1": "41, 2, 5, 6",
            "Name variant 2": "Lait Labode",
            "Page 2": "5",
            "Suggested standard spelling": "Lait Labode",
        },
        {
            "Name variant 1": "Mzer Michael Terungwa",
            "Page 1": "5",
            "Name variant 2": "Mzer Micheal Terungwa",
            "Page 2": "41",
            "Suggested standard spelling": "Mzer Michael Terungwa",
        },
    ]

    cleaned = clean_name_consistency_rows(rows)

    assert len(cleaned) == 1
    assert cleaned[0]["Name variant 1"] == "Mzer Michael Terungwa"


def test_key_amount_consistency_summarizes_consistent_rows_without_long_context():
    document = PdfDocument(
        [
            PdfPage(4, "Revenue 8,398,634 7,336,635", []),
            PdfPage(15, "Revenue 13 8,398,634 7,336,635", []),
            PdfPage(44, "Revenue 8,398,634 7,336,635", []),
        ]
    )

    findings, export = check_cross_page_consistency(document)

    assert findings == []
    assert export["key_amounts"] == [
        {
            "Metric": "Revenue",
            "Amount": "8,398,634",
            "Pages checked": "Pages 4, 15, 44",
            "Context": "Consistent across detected occurrences.",
            "Issue": "Consistent",
        }
    ]


def test_amount_match_snippet_returns_complete_extracted_row():
    text = "7 Operating Revenue\nSubscriptions 2 ,783,064\nPrior year 2\n,029,846"
    start = text.index("2 ,783,064")
    snippet = reviewer._amount_snippet_around(text, start, start + len("2 ,783,064"))

    assert snippet == "Subscriptions 2 ,783,064"


def test_page_reference_extracts_page_lists_from_inconsistency_evidence():
    document = PdfDocument(
        [
            PdfPage(5, "Directors' report\nMzer Michael Terungwa", []),
            PdfPage(41, "Directors' report\nMzer Micheal Terungwa", []),
        ]
    )

    findings, _export = check_cross_page_consistency(document)
    name_finding = next(finding for finding in findings if "Name spelt differently" in finding.issue)

    assert name_finding.location == "Pages 5, 41"


def test_excel_export_page_reference_uses_detected_note_pages():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert "_page_reference_for_finding" in app_source
    assert "_note_page_reference_map" in app_source


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


def test_rounding_scale_ignores_narrative_million_amounts_when_presentation_is_thousands():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Presentation currency N'000\nA deferred tax liability of N2 million arose during the year.",
                [],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert not any("Mixed rounding or scaling" in finding.issue for finding in findings)


def test_column_consistency_ignores_narrative_directors_report_tables():
    document = PdfDocument(
        [
            PdfPage(
                4,
                "Directors' report",
                [
                    [
                        ["Financial statements for the year ended 31 December 2025"],
                        ["Revenue", "2025", "2024"],
                        ["Profit", "100", "90"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert not any("only one comparative period" in finding.issue.lower() for finding in findings)


def test_review_memo_zero_high_uses_extraction_quality_next_step():
    from models import Finding, ReviewResult

    result = ReviewResult(
        findings=[
            Finding(
                "Extraction quality",
                "Medium",
                "Table extraction",
                "Some extracted table cells appear to contain multiple merged values.",
                "Detected merged cells.",
                "Review extraction quality.",
            )
        ],
        metrics={
            "findings": 1,
            "high": 0,
            "positive_assurance": "No exceptions noted from line-based checks on primary statements.",
        },
    )

    memo = build_ai_review_memo(result)

    assert "clear high-severity items first" not in memo
    assert "No high-severity exceptions were identified" in memo
    assert "rerun detailed note agreement after table extraction confidence improves" in memo


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


def test_statement_line_parser_handles_split_amounts_and_note_columns():
    assert _amounts_from_statement_line("Other Revenue 9 307,482 1 89,751")[-2:] == [
        Decimal("307482"),
        Decimal("189751"),
    ]
    assert _amounts_from_statement_line("Office accommodation costs 10 3 9,772 1 7,996")[-2:] == [
        Decimal("39772"),
        Decimal("17996"),
    ]
    assert _amounts_from_statement_line("Total Liabilities 1 41,411 1 54,819")[-2:] == [
        Decimal("141411"),
        Decimal("154819"),
    ]
    assert _amounts_from_statement_line("Net cash inflow from financing activities - ( 16,128)")[-2:] == [
        Decimal("0"),
        Decimal("-16128"),
    ]


def test_ocr_statement_row_parser_separates_note_number_from_amounts(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Share capital 8 10,000 10,000",
                        "Cash and cash equivalents 7 39,387 193,627",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    rows = reviewer._statement_row_parses(document.pages[0].text)
    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    debug_rows = result.metrics["ocr_statement_rows"]

    assert rows["share capital"].note_ref == "8"
    assert rows["share capital"].amounts == (Decimal("10000"), Decimal("10000"))
    assert rows["cash and cash equivalents"].note_ref == "7"
    assert rows["cash and cash equivalents"].amounts == (Decimal("39387"), Decimal("193627"))
    assert rows["cash and cash equivalents"].correction_applied == "No"
    assert "share capital | 8 | 10,000 | 10,000" in debug_rows
    assert "cash and cash equivalents | 7 | 39,387 | 193,627" in debug_rows


def test_ocr_statement_row_parser_extracts_sfp_note_numbers_without_merging_amounts():
    rows = reviewer._statement_row_parses(
        "\n".join(
            [
                "Statement of financial position",
                "Investment property 3 4,072,229 4,112,524",
                "Trade and other receivables 5 1,910,631 131,254",
                "Financial liabilities 9 5,356,392 4,555,742",
            ]
        )
    )

    assert rows["investment property"].note_ref == "3"
    assert rows["investment property"].amounts == (Decimal("4072229"), Decimal("4112524"))
    assert rows["trade and other receivables"].note_ref == "5"
    assert rows["trade and other receivables"].amounts == (Decimal("1910631"), Decimal("131254"))
    assert rows["financial liabilities"].note_ref == "9"
    assert rows["financial liabilities"].amounts == (Decimal("5356392"), Decimal("4555742"))


def test_ocr_statement_row_parser_keeps_split_bracketed_current_year_amount():
    rows = reviewer._statement_row_parses(
        "\n".join(
            [
                "Statement of profit or loss",
                "Loss before taxation (221,494) (173,516)",
                "Taxation (226,684) 53,127",
                "Loss for the year (448, 178) (120,389)",
            ]
        )
    )

    assert rows["profit after tax"].amounts == (Decimal("-448178"), Decimal("-120389"))


def test_ocr_statement_amount_parser_normalizes_decimal_thousands_separator():
    rows = reviewer._statement_row_parses(
        "\n".join(
            [
                "Statement of profit or loss",
                "Loss before taxation (173,516) (681,559)",
                "Taxation 53.127 253.124",
            ]
        )
    )

    assert rows["taxation"].amounts == (Decimal("53127"), Decimal("253124"))


def test_ocr_income_alignment_infers_missing_current_year_after_tax_from_prior_match():
    rows = reviewer._statement_row_parses(
        "\n".join(
            [
                "Statement of profit or loss",
                "2022 2021",
                "Loss before taxation (221,494) (173,516)",
                "Taxation (226,684) 53,127",
                "Loss for the year (120,389)",
            ]
        )
    )

    assert rows["profit after tax"].amounts == (Decimal("-448178"), Decimal("-120389"))
    assert rows["profit after tax"].confidence == "Low-Medium"


def test_ocr_income_alignment_does_not_infer_single_amount_without_prior_cast_match():
    rows = reviewer._statement_row_parses(
        "\n".join(
            [
                "Statement of profit or loss",
                "2022 2021",
                "Loss before taxation (221,494) (173,516)",
                "Taxation (226,684) 53,127",
                "Loss for the year (999,999)",
            ]
        )
    )

    assert rows["profit after tax"].amounts == (Decimal("-999999"),)


def test_ocr_statement_row_parser_does_not_concatenate_two_digit_note_with_revenue():
    rows = reviewer._statement_row_parses("Statement of profit or loss\nRevenue 13 707,189 297,041")

    assert rows["revenue"].note_ref == "13"
    assert rows["revenue"].amounts == (Decimal("707189"), Decimal("297041"))
    assert rows["revenue"].correction_applied == "No"


def test_primary_statement_checks_run_from_line_text_when_tables_are_low_confidence():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of income and expenditure",
                        "Subscriptions 242,511 120,424",
                        "Registrations 75,392 81,050",
                        "Operating Revenue 7 2 ,783,064 2,029,846",
                        "Gross Operating Revenue 3,100,967 2,231,320",
                        "Operating Expenditure 8 (1,269,506) (986,920)",
                        "Gross Revenue 1,831,461 1,244,400",
                        "Other Revenue 9 307,482 1 89,751",
                        "Finance Income 5 2,200 40,216",
                        "TOTAL INCOME 2,191,143 1 ,474,367",
                        "Office accommodation costs 10 3 9,772 1 7,996",
                        "Personnel costs 633,447 441,731",
                        "Administrative costs 871,713 589,872",
                        "Finance Expenses 1 1,013 14,599",
                        "TOTAL EXPENDITURE 1,555,945 1 ,064,198",
                        "SURPLUS OF INCOME OVER EXPENDITURE 635,198 410,169",
                        "TOTAL COMPREHENSIVE INCOME 635,198 410,169",
                    ]
                ),
                [],
            )
        ]
    )

    findings, performed, skipped = check_primary_statement_consistency(document)

    assert findings == []
    assert any("total income checked" in item for item in performed)
    assert any("total expenditure checked" in item for item in performed)


def test_single_page_sfp_extract_runs_limited_scope_checks_without_full_afs_highs(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Funtierra Limited",
                        "Statement of Financial Position",
                        "As at 31 December 2025",
                        "N’000",
                        "Non-current assets 4,072,229 4,112,524",
                        "Current assets 2,160,355 324,881",
                        "Total assets 6,232,584 4,437,405",
                        "Equity 876,192 (118,337)",
                        "Liabilities 5,356,392 4,555,742",
                        "Total equity and liabilities 6,232,584 4,437,405",
                        "Statement extract detail " * 60,
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    memo = build_ai_review_memo(result)

    assert result.metrics["document_scope"] == "Limited-scope statement extract"
    assert result.metrics["detected_profile"]["Document scope"] == "Limited-scope statement extract"
    assert result.metrics["detected_profile"]["Currency"] == "NGN / N'000"
    assert result.metrics["high"] == 0
    assert not any("IAS 1" in finding.location for finding in result.findings)
    assert any(finding.category == "Document scope" for finding in result.findings)
    assert "Statement of financial position: total assets checked from line-extracted rows." in result.metrics["checks_performed"]
    assert "Limited-scope review performed on Statement of Financial Position only." in memo
    assert "Full financial statement completeness" in result.metrics["checks_skipped"]


def test_detected_company_name_removes_short_ocr_prefix():
    document = PdfDocument([PdfPage(1, "Fl Funtierra Limited\nFinancial statements", [])])

    assert infer_detected_profile(document)["Company name"] == "Funtierra Limited"


def test_ocr_profit_or_loss_rows_are_parsed_with_fuzzy_labels():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenve 1,000 900",
                        "Loss before taxation (200) (100)",
                        "Taxation (50) (30)",
                        "Loss after tax (250) (130)",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings, performed, skipped = check_primary_statement_consistency(document)

    assert not findings
    assert any("profit/loss after tax checked" in item for item in performed)


def test_ocr_sfp_requested_rows_are_parsed_with_fuzzy_labels():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Non current assets 600 500",
                        "Current assets 400 300",
                        "Total assets 1,000 800",
                        "Equity 700 550",
                        "Liabilities 300 250",
                        "Total liabilities and equity 1,000 800",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings, performed, skipped = check_primary_statement_consistency(document)

    assert not findings
    assert any("total assets checked from line-extracted rows" in item for item in performed)
    assert any("equity and liabilities equation checked" in item for item in performed)


def test_ocr_cash_beginning_and_end_rows_are_parsed():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of cash flows",
                        "Net increase in cash and cash equivalents 100 80",
                        "Cash and cash equivalents at beginning of year 250 170",
                        "Cash and cash equivalents at end of year 350 250",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings, performed, skipped = check_primary_statement_consistency(document)

    assert not findings
    assert any("cash at beginning/end checked" in item for item in performed)


def test_arithmetic_skips_merged_numeric_cells():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [[["Description", "2025", "2024"], ["Revenue", "100 90", ""], ["Total revenue", "100", "90"]]],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("skipped" in finding.issue.lower() for finding in findings)
    assert not any("does not agree" in finding.issue.lower() for finding in findings)


def test_arithmetic_skips_five_year_summary():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [
                        ["Five year financial summary", "2025", "2024", "2023"],
                        ["Revenue", "100", "90", "80"],
                        ["Total assets", "300", "280", "250"],
                        ["Total equity", "200", "190", "170"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("multi-year summary" in finding.evidence.lower() for finding in findings)
    assert not any("does not agree" in finding.issue.lower() for finding in findings)


def test_arithmetic_skips_value_added_statement():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [
                        ["Value added statement", "2025", "2024"],
                        ["Revenue", "100", "90"],
                        ["Bought in materials", "40", "35"],
                        ["Total value added", "60", "55"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("value-added statement" in finding.evidence.lower() for finding in findings)
    assert not any("does not agree" in finding.issue.lower() for finding in findings)


def test_low_confidence_table_skips_are_grouped():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [["Value added statement", "2025", "2024"], ["Revenue", "100", "90"], ["Total", "100", "90"]],
                    [["Five year financial summary", "2025", "2024"], ["Assets", "300", "280"], ["Equity", "200", "190"]],
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))
    grouped = [finding for finding in findings if "table(s) skipped" in finding.issue.lower()]

    assert len(grouped) == 1
    assert "Page 1, table 1" in grouped[0].evidence
    assert "Page 1, table 2" in grouped[0].evidence


def test_generic_arithmetic_skips_ocr_reconstructed_tables():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nRevenue 100 90\nTotal revenue 999 999",
                [[["Description", "2025", "2024"], ["Revenue", "100", "90"], ["Total revenue", "999", "999"]]],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
        ocr_tables=1,
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("generic arithmetic checks were skipped" in finding.issue.lower() for finding in findings)
    assert not any("does not agree" in finding.issue.lower() for finding in findings)


def test_arithmetic_respects_repeated_table_headers_in_note_blocks():
    document = PdfDocument(
        [
            PdfPage(
                20,
                "Notes to the financial statements\nN'000",
                [
                    [
                        ["December", "31", "2025"],
                        ["N'", "000", "000"],
                        ["Trade and other receivable", "288,706", "288,706"],
                        ["Cash", "875,869", "875,869"],
                        ["Note (s) N'", "000", "000"],
                        ["Trade and other payables", "141,411", "141,411"],
                        ["Note (s) N'", "000", "000"],
                        ["Trade and other payables", "154,819", "154,819"],
                        ["Note (s) N'", "000", "000"],
                        ["Trade and other payables", "141,411", "1", "54,819"],
                        ["Total Liabilities", "141,411", "154,819"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert not any("does not agree" in finding.issue.lower() for finding in findings)


def test_arithmetic_stops_when_new_note_heading_starts():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "",
                [
                    [
                        ["Description", "2025"],
                        ["Other payables", "127,575"],
                        ["Withholding tax - State", "7,968"],
                        ["Accrued fees", "5,869"],
                        ["Total", "141,411"],
                        ["20 Receivables & Advances", ""],
                        ["Receivables", "2,000,000"],
                        ["Total", "2,000,000"],
                    ]
                ],
            )
        ]
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert not any("visible sum 2,141,411" in finding.evidence for finding in findings)


def test_notes_check_flags_missing_note_heading():
    document = PdfDocument([PdfPage(1, "Revenue Note 7 100\nProfit for the year 40", [])])

    findings = check_notes_agreement(document)

    assert any("note 7" in finding.issue.lower() for finding in findings)


def test_note_reference_detection_rejects_years_and_large_numbers():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 2025 4001 3001\nOther Revenue 9 307,482 189,751",
                [],
            ),
            PdfPage(2, "Notes\n9 Other Revenue\nFair value gain 307,482", []),
        ]
    )

    findings = check_notes_agreement(document)

    assert not any("note 2025" in finding.issue.lower() for finding in findings)
    assert not any("note 4001" in finding.issue.lower() for finding in findings)
    assert not any("note 3001" in finding.issue.lower() for finding in findings)
    assert not any("note 9" in finding.issue.lower() and "not found" in finding.issue.lower() for finding in findings)


def test_note_heading_detection_handles_multiline_header_with_year_columns():
    text = "NOTE 9 2025 2024\nOther Revenue N'000 N'000\nFair Value Gain 100 90"

    headings = _note_headings(text)

    assert headings["9"] == "Other Revenue"


def test_note_heading_detection_handles_combined_and_split_table_headings():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of income and expenditure",
                        "Personnel costs 11 633,447 441,731",
                    ]
                ),
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "NOTES TO THE FINANCIAL STATEMENTS",
                        "Note - 2025 7 8",
                        "Cost of Generating",
                        "Revenue Heads Operating Revenue Operating Revenue Net Surplus",
                        "N'000 N'000 N'000",
                        "2,783,064 1,269,506 1,513,130",
                        "10 Office accommodation N'000 N'000",
                        "Repairs 39,772 17,996",
                        "11(a) Staff costs N'000 N'000",
                        "Salaries 633,447 441,731",
                        "16 Inventories 2025 2024",
                        "N'000 N'000",
                        "Study packs 7,601 15,969",
                    ]
                ),
                [],
            )
        ]
    )

    headings = _note_headings_by_page(document)

    assert headings["7"][0] == "Operating Revenue"
    assert headings["8"][0] == "Operating Expenditure"
    assert headings["10"][0] == "Office accommodation"
    assert headings["11"][0] == "Staff costs"
    assert headings["16"][0] == "Inventories"


def test_disclosure_only_notes_are_not_flagged_as_unreferenced():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nCash Note 18 100",
                [],
            ),
            PdfPage(
                2,
                "18 Cash and cash equivalents\nCash 100\n25 Capital commitments\nThere were no capital commitments.\n26 Contingent liabilities\nNone.\n27 Subsequent events disclosure\nThere were no subsequent events.\n28 Related party transactions\nNo transactions.",
                [],
            ),
        ]
    )

    findings = check_notes_agreement(document)

    assert not any("note 25 exists but was not referenced" in finding.issue.lower() for finding in findings)
    assert not any("note 26 exists but was not referenced" in finding.issue.lower() for finding in findings)
    assert not any("note 27 exists but was not referenced" in finding.issue.lower() for finding in findings)
    assert not any("note 28 exists but was not referenced" in finding.issue.lower() for finding in findings)


def test_notes_check_flags_possible_wrong_note_reference_when_item_is_in_another_note():
    filler = "Revenue 100 90\n" * 80
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of income and expenditure\nOther Revenue 7 307,482 189,751\n" + filler,
                [],
            ),
            PdfPage(
                2,
                "Notes to the financial statements\n7 Operating revenue\nSubscriptions 242,511 120,424\nRegistrations 75,392 81,050\n9 Other Revenue\nFair value gain 307,482 189,751\n",
                [],
            ),
        ]
    )

    findings = check_notes_agreement(document)

    wrong_ref = [finding for finding in findings if "possible wrong note reference" in finding.issue.lower()]
    assert wrong_ref
    assert wrong_ref[0].severity == "High"
    assert "Other Revenue references Note 7" in wrong_ref[0].issue
    assert "Note 9 appears to be a stronger match" in wrong_ref[0].issue
    assert wrong_ref[0].metadata["statement"] == "Statement of income and expenditure"
    assert wrong_ref[0].metadata["line_item"] == "Other Revenue"
    assert wrong_ref[0].metadata["referenced_note"] == "7"
    assert wrong_ref[0].metadata["suggested_note"] == "9"


def test_wrong_note_reference_check_respects_low_confidence_gate():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOther Revenue 7 307,482 189,751", []),
            PdfPage(
                2,
                "Notes to the financial statements\n7 Operating revenue\nSubscriptions 242,511\n9 Other Revenue\nFair value gain 307,482",
                [],
            ),
        ]
    )

    normal_findings = check_notes_agreement(document)
    cautious_findings = check_notes_agreement(document, cautious_low_confidence=True)

    assert not any("possible wrong note reference" in finding.issue.lower() for finding in normal_findings)
    assert not any("possible wrong note reference" in finding.issue.lower() for finding in cautious_findings)


def test_cautious_note_reference_validation_uses_detected_headings_when_sections_are_weak():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOther Revenue 7 307,482 189,751", []),
            PdfPage(
                2,
                "Notes to the financial statements\n7 Operating Revenue\nRevenue schedule text without matching amount\n9 Other Revenue\nNarrative only",
                [],
            ),
        ]
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)
    wrong_ref = [finding for finding in findings if "possible wrong note reference" in finding.issue.lower()]

    assert wrong_ref == []


def test_cautious_note_reference_validation_flags_explicit_missing_note_as_review_prompt():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash Note 18 875,869 605,645", []),
            PdfPage(2, "Notes to the financial statements\n19 Trade and other payables\nOther payables 141,411", []),
        ]
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)
    missing = [finding for finding in findings if "referenced note not found" in finding.issue.lower()]

    assert missing
    assert missing[0].severity == "Low"
    assert missing[0].metadata["line_item"] == "Cash"
    assert missing[0].metadata["referenced_note"] == "18"


def test_cautious_face_to_note_amount_agreement_flags_amount_found_elsewhere():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash Note 18 875,869 605,645", []),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Notes to the financial statements",
                        "18 Cash and cash equivalents",
                        "Bank balances 100,000 90,000",
                        "19 Cash balances",
                        "Bank balances 875,869 605,645",
                    ]
                ),
                [],
            ),
        ]
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)
    amount_prompts = [finding for finding in findings if "amount appears in another note" in finding.issue.lower()]

    assert amount_prompts
    assert amount_prompts[0].severity == "Medium"
    assert amount_prompts[0].metadata["statement"] == "Statement of financial position"
    assert amount_prompts[0].metadata["line_item"] == "Cash"
    assert amount_prompts[0].metadata["referenced_note"] == "18"
    assert amount_prompts[0].metadata["alternative_note_found"] == "19"
    assert amount_prompts[0].metadata["current_year_amount_found"] == "No"
    assert amount_prompts[0].metadata["prior_year_amount_found"] == "No"
    assert amount_prompts[0].metadata["amount_match_confidence"] == "Medium"


def test_cautious_face_to_note_amount_agreement_keeps_low_amount_not_located_out_of_findings(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOther Revenue Note 9 307,482 189,751", []),
            PdfPage(
                2,
                "Notes to the financial statements\n9 Other Revenue\nFair value gain 100,000 90,000\n10 Office accommodation\nRent 50,000 40,000",
                [],
            ),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    findings = check_notes_agreement(document, cautious_low_confidence=True)
    amount_prompts = [finding for finding in findings if "amount not located in referenced note" in finding.issue.lower()]
    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    rows = result.metrics["note_agreement_results"]

    assert not amount_prompts
    assert any(
        row["Line item"] == "Other Revenue"
        and row["Result"] == "Review prompt"
        and row["Reason"] == "Amount not located in referenced note."
        for row in rows
    )


def test_cautious_face_to_note_amount_agreement_does_not_flag_when_amounts_are_in_referenced_note():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOther Revenue Note 9 307,482 189,751", []),
            PdfPage(2, "Notes to the financial statements\n9 Other Revenue\nFair value gain 307,482 189,751", []),
        ]
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)

    assert not any("amount not located" in finding.issue.lower() for finding in findings)
    assert not any("amount appears in another note" in finding.issue.lower() for finding in findings)


def test_cautious_face_to_note_amount_agreement_matches_normalized_visible_amounts(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOperating Revenue Note 7 2,783,064 2,029,846", []),
            PdfPage(2, "Notes to the financial statements\n7 Operating Revenue\nSubscriptions 2 ,783,064\nPrior year 2\n,029,846", []),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Operating Revenue")

    assert row["Result"] == "Passed"
    assert row["Current year amount found in referenced note?"] == "Yes"
    assert row["Prior year amount found in referenced note?"] == "Yes"
    assert "normalized amount" in row["Matching method"]
    assert "2 ,783,064" in row["Matched text snippet from referenced note"]


def test_cautious_face_to_note_amount_agreement_does_not_suggest_amount_only_alternative(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of income and expenditure\nOperating Revenue Note 7 2,783,064 2,029,846", []),
            PdfPage(
                2,
                "Notes to the financial statements\n7 Operating Revenue\nSubscriptions 100,000 90,000\n8 Operating Expenditure\nExpenses 2,783,064 2,029,846",
                [],
            ),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Operating Revenue")

    assert row["Alternative note found"] == ""
    assert row["Result"] == "Review prompt"
    assert not any("Operating Revenue amount appears in Note 8" in finding.issue for finding in result.findings)


def test_cash_note_reference_does_not_suggest_non_cash_alternative(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash At End Of The Year Note 7 100 90", []),
            PdfPage(
                2,
                "Notes to the financial statements\n3 Investment property\nCash At End Of The Year 100 90\n7 Directors' interests in shares\nNarrative only\n"
                + ("Additional OCR note text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert not any(
        finding.metadata
        and finding.metadata.get("line_item") == "Cash At End Of The Year"
        and finding.metadata.get("suggested_note") == "3"
        for finding in result.findings
    )


def test_revenue_note_reference_does_not_suggest_investment_property_without_revenue_context(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue Note 13 707,189 297,041", []),
            PdfPage(
                2,
                "Notes to the financial statements\n3 Investment property\nFair value movement 707,189 297,041\n13 Revenue\nNarrative only\n"
                + ("Additional OCR note text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert not any(
        finding.metadata
        and finding.metadata.get("line_item") == "Revenue"
        and finding.metadata.get("suggested_note") == "3"
        for finding in result.findings
    )


def test_ocr_revenue_heading_prompt_stays_in_note_results_not_exception_register(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue Note 13 707,189 297,041", []),
            PdfPage(
                2,
                "Notes to the financial statements\n3 Rental income\nRental income 707,189 297,041\n13 Investment property\nNarrative only\n"
                + ("Additional OCR note text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Revenue")

    assert row["Alternative note found"] == "3"
    assert row["Match confidence"] == "Low"
    assert row["Result"] == "Skipped"
    assert "debug" in row["Reason"].lower()
    assert not any(finding.metadata and finding.metadata.get("line_item") == "Revenue" for finding in result.findings)


def test_ocr_revenue_heading_prompt_runs_for_direct_revenue_heading(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nOther Revenue Note 7 307,482 189,751", []),
            PdfPage(
                2,
                "Notes to the financial statements\n7 Operating revenue\nRevenue 100 90\n9 Other Revenue\nOther Revenue 307,482 189,751\n"
                + ("Additional OCR note text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Other Revenue")

    assert row["Alternative note found"] == "9"
    assert row["Result"] == "Review prompt"
    assert result.metrics["note_reference_findings"] == 0


def test_ocr_revenue_note_reference_does_not_suggest_weak_tenant_note(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue Note 13 707,189 297,041", []),
            PdfPage(
                2,
                "Notes to the financial statements\n10B Tenant deposits\nTenant balances and rent deposits 707,189 297,041\n13 Investment property\nNarrative only\n"
                + ("Additional OCR note text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Revenue")

    assert row["Alternative note found"] == ""
    assert row["Result"] == "Skipped"
    assert not any(finding.metadata and finding.metadata.get("suggested_note") == "10B" for finding in result.findings)


def test_revenue_amount_agreement_does_not_suggest_tenant_note_without_approved_heading(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue Note 13 707,189 297,041", []),
            PdfPage(
                2,
                "Notes to the financial statements\n10B Tenant balances\nTenant rental balances 707,189 297,041\n13 Revenue\nNarrative only",
                [],
            ),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Revenue")

    assert row["Alternative note found"] == ""
    assert not any(finding.metadata and finding.metadata.get("suggested_note") == "10B" for finding in result.findings)


def test_cautious_face_to_note_amount_agreement_skips_non_face_linked_lines(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCurrent liabilities Note 19 141,411 154,819", []),
            PdfPage(2, "Notes to the financial statements\n19 Trade and other payables\nOther payables 141,411 154,819", []),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Current Liabilities")

    assert row["Result"] == "Skipped"
    assert row["Reason"] == "Skipped - not a face-linked note line"


def test_note_agreement_excludes_cash_flow_subtotals_without_explicit_note(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of cash flows",
                        "Net cash inflow from operating activities a 141,411 154,819",
                        "Net cash absorbed in investing activities b (20,000) (15,000)",
                        "Net increase in cash and cash equivalents a+b 121,411 139,819",
                    ]
                ),
                [],
            ),
            PdfPage(2, "Notes to the financial statements\n18 Cash and cash equivalents\nCash 121,411 139,819", []),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["note_agreement_results"] == []
    assert not any("cash inflow" in finding.issue.lower() for finding in result.findings)


def test_note_agreement_keeps_cash_flow_rows_with_explicit_note(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of cash flows\nCash and cash equivalents at end of year Note 18 875,869 605,645", []),
            PdfPage(2, "Notes to the financial statements\n18 Cash and cash equivalents\nCash 875,869 605,645", []),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert any(row["Line item"] == "Cash And Cash Equivalents At End Of Year" and row["Result"] == "Passed" for row in result.metrics["note_agreement_results"])


def test_note_agreement_excludes_value_added_and_five_year_summary(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of value added\nSurplus for the year 9 307,482 189,751", []),
            PdfPage(2, "Five year financial summary\nRevenue 7 2,783,064 2,029,846", []),
            PdfPage(3, "Notes to the financial statements\n7 Revenue\nRevenue 2,783,064 2,029,846\n9 Other Revenue\nOther revenue 307,482 189,751", []),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["note_agreement_results"] == []


def test_cautious_face_to_note_amount_agreement_does_not_treat_amount_digits_as_note_refs():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nTotal Current Assets 1,172,176 811,598", []),
            PdfPage(2, "Notes to the financial statements\n8 Operating expenditure\nExpenses 811,598", []),
        ]
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)

    assert not any("current assets references note 8" in finding.issue.lower() for finding in findings)


def test_note_agreement_results_skip_rows_where_only_note_number_was_detected_as_amount(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nOther Financial Assets Note 6 6", []),
            PdfPage(2, "Notes to the financial statements\n6 Other financial assets\nNarrative only", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Other Financial Assets")

    assert row["Referenced note"] == "6"
    assert row["Current year amount"] == ""
    assert row["Result"] == "Skipped"
    assert "no reliable statement amount" in row["Reason"].lower()


def test_review_pdf_reports_cautious_note_reference_override_as_performed(monkeypatch):
    noisy_table = [["Description", "2025", "2024"]]
    noisy_table.extend([["Line item", "100 90", ""] for _ in range(40)])
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of income and expenditure\nRevenue 100 90\n" * 80,
                [noisy_table],
            )
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Cautious face-to-note amount agreement performed in review-prompt mode." in result.metrics["checks_performed"]
    assert "Cautious note-reference validation performed in review-prompt mode; no possible wrong note references detected." in result.metrics["checks_performed"]
    assert "Cautious note-reference validation skipped" not in result.metrics["checks_skipped"]
    assert result.metrics["cautious_note_validation_enabled"] is True
    assert result.metrics["note_validation_mode"] == "review_prompt"
    assert result.metrics["note_reference_rows_detected"] == 0
    assert result.metrics["note_headings_detected"] == 0
    assert result.metrics["note_reference_findings"] == 0


def test_note_agreement_results_include_passed_and_review_prompt_rows(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Cash Note 18 875,869 605,645",
                        "Trade payables Note 19 141,411 154,819",
                    ]
                ),
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Notes to the financial statements",
                        "18 Cash and cash equivalents",
                        "Cash at bank 875,869 605,645",
                        "19 Trade and other payables",
                        "Other payables 141,411 100,000",
                    ]
                ),
                [],
            ),
        ]
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    rows = result.metrics["note_agreement_results"]

    assert any(row["Line item"] == "Cash" and row["Result"] == "Passed" for row in rows)
    assert any(row["Line item"] == "Trade Payables" and row["Result"] == "Review prompt" for row in rows)
    assert any(row["Prior year amount found in referenced note?"] == "No" for row in rows)


def test_notes_agreement_is_conservative_for_ocr_documents():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nRevenue Note 7 100\nNotes to the financial statements\n8 Revenue",
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_notes_agreement(document)

    assert len(findings) == 1
    assert findings[0].category == "Extraction quality"
    assert "skipped for an ocr-assisted document" in findings[0].issue.lower()


def test_note_heading_detection_rejects_years_and_ocr_noise():
    text = "2021 Statement of changes\n2022 Revenue\n89B OCR garbage\n4 Revenue\n5 Trade and other receivables"

    headings = _note_headings(text)

    assert "2021" not in headings
    assert "2022" not in headings
    assert "89B" not in headings
    assert headings["4"] == "Revenue"


def test_note_heading_detection_rejects_report_furniture_and_entity_names():
    text = "\n".join(
        [
            "1 Significant accounting policies",
            "1S Example Holdings Limited",
            "4 Example Holdings Limited",
            "5 Directors",
            "7 Financial Statements for the year ended December 31, 2021",
            "9 _ Financial liabilities",
            "10 Trade and other payables",
        ]
    )

    headings = _note_headings(text)

    assert headings["1"] == "Significant accounting policies"
    assert "1S" not in headings
    assert "4" not in headings
    assert "5" not in headings
    assert "7" not in headings
    assert headings["9"] == "_ Financial liabilities"
    assert headings["10"] == "Trade and other payables"


def test_note_heading_detection_rejects_narrative_suffix_heading():
    text = "\n".join(
        [
            "Notes to the financial statements",
            "10A Advances from tenants",
            "10B This represents the advance payment made by tenants for future rental periods",
            "11 Trade and other payables",
        ]
    )

    headings = _note_headings(text)

    assert headings["10A"] == "Advances from tenants"
    assert "10B" not in headings
    assert headings["11"] == "Trade and other payables"


def test_note_heading_detection_accepts_clear_continued_heading_when_original_split():
    document = PdfDocument(
        [
            PdfPage(32, "Notes to the financial statements\nIntangible assets\nCost 100 90", []),
            PdfPage(33, "4. Intangible assets (continued)\nAccumulated amortisation 20 10", []),
        ]
    )

    headings = _note_headings_by_page(document)

    assert headings["4"] == ("Intangible assets", 33)


def test_missing_statement_note_reference_includes_statement_page(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                14,
                "Statement of financial position\nDeferred tax Note 9 100 90\nTotal assets 100 90\nTotal equity and liabilities 100 90\n"
                + ("Primary statement context.\n" * 70),
                [],
            ),
            PdfPage(35, "Notes to the financial statements\n10 Deferred tax\nDeferred tax 100 90\n" + ("Note context.\n" * 70), []),
        ]
    )
    monkeypatch.setattr(PdfDocument, "table_extraction_confidence", property(lambda self: 100))

    findings = check_notes_agreement(document)

    missing = next(finding for finding in findings if "Statement references note 9" in finding.issue)
    assert missing.location == "Page 14"
    assert "Page 14" in missing.evidence


def test_note_internal_total_skips_complex_movement_schedule_noise(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                20,
                "\n".join(
                    [
                        "Notes to the financial statements",
                        "6 Trade and other receivables",
                        "Movement in loss allowance",
                        "Opening balance 100",
                        "Charge for the year 20",
                        "Utilised during the year (10)",
                        "Closing balance 130",
                        "Total 999",
                    ]
                )
                + "\n"
                + ("Note context.\n" * 70),
                [],
            )
        ]
    )
    monkeypatch.setattr(PdfDocument, "table_extraction_confidence", property(lambda self: 100))

    findings = check_notes_agreement(document)

    assert not any("subtotal or total does not agree" in finding.issue.lower() for finding in findings)


def test_skipped_table_summary_groups_notes_arithmetic_skips(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90\n"
                + ("Primary context.\n" * 70),
                [],
            ),
            PdfPage(10, "Notes to the financial statements\n3 Revenue\nRevenue 100 90", [[["Revenue", "2025", "2024"], ["Fees", "100", "90"], ["Total", "100", "90"]]]),
            PdfPage(11, "4 Expenses\nExpenses 60 50", [[["Expenses", "2025", "2024"], ["Admin", "60", "50"], ["Total", "60", "50"]]]),
        ]
    )
    monkeypatch.setattr(PdfDocument, "table_extraction_confidence", property(lambda self: 100))
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions())
    summary = result.metrics["skipped_table_summary"]

    notes_group = next(row for row in summary if row["Skipped check group"] == "Notes tables - generic arithmetic skipped")
    assert notes_group["Pages affected"] == "Pages 10-11"
    assert notes_group["Tables affected"] == "2"
    assert notes_group["Can automated check be fixed?"] == "Partially"
    assert "may merge" in notes_group["Why reviewer should review"].lower()


def test_note_heading_detection_starts_after_notes_heading_when_present():
    document = PdfDocument(
        [
            PdfPage(1, "Directors' report\n7 Directors interests in shares\n8 Employment and employees", []),
            PdfPage(2, "Notes to the financial statements\n7 Revenue\nRevenue 100 90\n8 Cost of sales\nCost 50 40", []),
        ]
    )

    headings = _note_headings_by_page(document)

    assert headings["7"] == ("Revenue", 2)
    assert headings["8"] == ("Cost of sales", 2)
    assert "Directors interests" not in headings["7"][0]


def test_ocr_note_heading_detection_ignores_directors_report_before_notes_section():
    document = PdfDocument(
        [
            PdfPage(1, "Directors' report\n7 Directors' interests in shares\nEmployment and employees\n", []),
            PdfPage(2, "Notes to the financial statements\n1 Accounting policies\n2 Investment property\n7 Trade and other payables\n", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    headings = _note_headings_by_page(document)

    assert headings["7"] == ("Trade and other payables", 2)
    assert all("directors" not in title.lower() for title, _page in headings.values())


def test_ocr_note_headings_never_include_pages_before_notes_start_page():
    document = PdfDocument(
        [
            PdfPage(5, "Directors' report\n7 Directors' interests in shares\n11 Employment and employees", []),
            PdfPage(14, "Notes to the financial statements\n1 Accounting policies\n3 Investment property\n13 Revenue\n", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    headings = _note_headings_by_page(document)

    assert headings["13"] == ("Revenue", 14)
    assert "7" not in headings
    assert "11" not in headings


def test_note_headings_are_not_overwritten_from_statement_references():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue Note 13 707,189 297,041", []),
            PdfPage(14, "Notes to the financial statements\n3 Investment property\n13 Revenue from contracts with customers", []),
        ]
    )

    headings = _note_headings_by_page(document)

    assert headings["3"] == ("Investment property", 14)
    assert headings["13"] == ("Revenue from contracts with customers", 14)


def test_notes_start_page_requires_actual_notes_heading_not_front_matter_phrase(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(2, "Independent auditor's report\nThe notes to the financial statements are part of our audit.", []),
            PdfPage(14, "Notes to the financial statements\n1 Accounting policies\n13 Revenue", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 14


def test_ocr_note_validation_skips_when_notes_section_start_is_not_detected():
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash Note 7 100 90", []),
            PdfPage(2, "Directors' report\n7 Directors' interests in shares", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    findings = check_notes_agreement(document, cautious_low_confidence=True)

    assert any("notes section start was not detected" in finding.issue.lower() for finding in findings)
    assert not any("possible wrong note reference" in finding.issue.lower() for finding in findings)


def test_review_pdf_does_not_report_ocr_note_validation_performed_without_notes_boundary(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash Note 7 100 90", []),
            PdfPage(2, "Directors' report\n7 Directors' interests in shares", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Heading-based note-reference validation performed" not in result.metrics["checks_performed"]
    assert "OCR note-reference validation skipped because the notes section start was not detected." in result.metrics["checks_skipped"]
    assert result.metrics["note_validation_mode"] == "skipped"
    assert result.metrics["note_headings_detected"] == 0


def test_ocr_note_validation_status_stays_out_of_exception_register(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nCash Note 7 100 90\n" + ("OCR statement context.\n" * 40), []),
            PdfPage(2, "Notes to the financial statements\n7 Cash and cash equivalents\nCash 100 90\n" + ("OCR note context.\n" * 40), []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Heading-based note-reference validation performed in OCR review-prompt mode." in result.metrics["checks_performed"]
    assert not any("Heading-based note-reference validation was run" in finding.issue for finding in result.findings)


def test_ocr_notes_start_page_accepts_fuzzy_notes_heading_variants():
    document = PdfDocument(
        [
            PdfPage(2, "Independent auditor's report\nThe notes to the financial statements are part of our audit.", []),
            PdfPage(14, "NOTES FORMING PART OF THE FINANCIAL STATEMENTS\n1 Accounting policies\n13 Revenue", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    assert reviewer._notes_start_page(document) == 14
    assert "NOTES FORMING PART" in reviewer._format_notes_heading_snippet(document)


def test_ocr_notes_start_page_uses_strong_heading_candidate_with_note_heading_on_same_line():
    document = PdfDocument(
        [
            PdfPage(2, "Independent auditor's report\nThe notes to the financial statements are part of our audit.", []),
            PdfPage(14, "Notes to the Financial Statements 1. Significant accounting policies\n2 Revenue\nRevenue 100 90", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    assert reviewer._notes_start_page(document) == 14
    snippet = reviewer._format_notes_heading_snippet(document)
    assert "Confidence" in snippet
    assert "Significant accounting policies" in snippet
    headings = _note_headings_by_page(document)
    assert headings["1"] == ("Significant accounting policies", 14)
    debug = reviewer._format_note_heading_debug(document)
    assert "Confidence:" in debug
    assert "Source:" in debug


def test_ocr_note_heading_rejects_amount_table_rows_after_notes_start():
    document = PdfDocument(
        [
            PdfPage(
                14,
                "\n".join(
                    [
                        "Notes to the Financial Statements",
                        "1. Significant accounting policies",
                        "1 Taxation (226,684) 53,127",
                        "2 Revenue",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    headings = _note_headings_by_page(document)

    assert headings["1"] == ("Significant accounting policies", 14)
    assert headings["2"] == ("Revenue", 14)


def test_ocr_note_headings_stop_before_value_added_and_five_year_summary_pages():
    document = PdfDocument(
        [
            PdfPage(14, "Notes to the Financial Statements\n1. Significant accounting policies\n7 Cash and cash equivalents", []),
            PdfPage(31, "Statement of value added\n1 Depreciation 10,000 9,000\n7 Notes to the Financial Statements", []),
            PdfPage(32, "Five-year financial summary\n8 Assets 13233 12000", []),
        ],
        ocr_used=True,
        ocr_pages=3,
    )

    headings = _note_headings_by_page(document)

    assert headings["1"] == ("Significant accounting policies", 14)
    assert headings["7"] == ("Cash and cash equivalents", 14)
    assert "8" not in headings


def test_note_heading_rejects_repeated_notes_header_as_title():
    document = PdfDocument([PdfPage(14, "Notes to the Financial Statements\n7 Notes to the Financial Statements\n7 Cash and cash equivalents", [])])

    headings = _note_headings_by_page(document)

    assert headings["7"] == ("Cash and cash equivalents", 14)


def test_ocr_note_headings_keep_clear_note_lines_and_reject_noise():
    document = PdfDocument(
        [
            PdfPage(
                14,
                "\n".join(
                    [
                        "Notes to the Financial Statements",
                        "1 Significant accounting policies",
                        "The financial statements are prepared under IFRS",
                        "3 Investment property",
                        "Investment property is measured at fair value",
                        "5 Trade and other receivables",
                        "5 Trade and other receivables 1,910,631 131,254",
                        "6 Other financial assets",
                        "7 Notes to the Financial Statements",
                        "7 Cash and cash equivalents",
                        "8 Share capital",
                        "9 Financial liabilities",
                        "10 Trade and other payables",
                        "12 Taxation",
                        "13 Revenue",
                        "14 Direct costs",
                        "15 Other operating income",
                        "16 Other operating gains",
                        "17 Administrative expenses",
                        "18 Finance cost",
                        "20 Related parties",
                        "21 Going concern",
                        "23 Financial instruments and risk management",
                        "23 Credit risk",
                        "23 Liquidity risk",
                        "24 This represents the advance payment made by customers",
                        "25 Financial instruments are accounted for at amortised cost",
                    ]
                ),
                [],
            ),
            PdfPage(31, "Statement of value added\n26 Depreciation 10,000 9,000", []),
            PdfPage(32, "Five-year financial summary\n27 Revenue 13233 12000", []),
        ],
        ocr_used=True,
        ocr_pages=3,
    )

    headings = _note_headings_by_page(document)

    expected = {
        "1": "Significant accounting policies",
        "3": "Investment property",
        "5": "Trade and other receivables",
        "6": "Other financial assets",
        "7": "Cash and cash equivalents",
        "8": "Share capital",
        "9": "Financial liabilities",
        "10": "Trade and other payables",
        "12": "Taxation",
        "13": "Revenue",
        "14": "Direct costs",
        "15": "Other operating income",
        "16": "Other operating gains",
        "17": "Administrative expenses",
        "18": "Finance cost",
        "20": "Related parties",
        "21": "Going concern",
        "23": "Financial instruments and risk management",
    }
    assert {ref: title for ref, (title, _page) in headings.items()} == expected


def test_ocr_note_headings_reject_financial_instrument_subheadings_without_replacing_main_heading():
    document = PdfDocument(
        [
            PdfPage(
                14,
                "\n".join(
                    [
                        "Notes to the Financial Statements",
                        "23 Financial instruments and risk management",
                        "23 Credit risk",
                        "23 Liquidity risk",
                        "23 Market risk",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    headings = _note_headings_by_page(document)

    assert headings == {"23": ("Financial instruments and risk management", 14)}


def test_ocr_notes_heading_accepts_repeating_report_header_with_numbered_policy(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(3, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(
                14,
                "Example Limited\nFinancial statements for the year ended 31 December 2025\n"
                "Directors' and independent auditor's reports\n"
                "Notes to the Financial Statements\n"
                "1. Significant accounting policies\n2 Revenue\nRevenue 100 90",
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 14
    assert result.metrics["note_structure_confidence"] != "0%"
    assert any(row["Accepted"] == "Yes" and row["Page"] == "14" for row in result.metrics["notes_heading_candidates"])
    assert _note_headings_by_page(document)["1"] == ("Significant accounting policies", 14)


def test_ocr_notes_start_infers_note_1_from_material_accounting_policies():
    document = PdfDocument(
        [
            PdfPage(3, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(
                14,
                "Financial statements for the year ended 31 December 2025\n"
                "Notes to the Financial Statements\n"
                "Material accounting policies\n"
                "The material accounting policies applied in preparing these financial statements are set out below.\n"
                "1.1 Basis of preparation\n"
                "2 New Standards and Interpretations",
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    headings = _note_headings_by_page(document)

    assert headings["1"] == ("Material accounting policies", 14)


def test_ocr_notes_heading_candidates_reject_contents_page():
    document = PdfDocument(
        [
            PdfPage(
                2,
                "Statement of financial position 12\n"
                "Statement of comprehensive income 13\n"
                "Statement of cash flows 14\n"
                "Notes to the Financial Statements 15\n"
                "Value Added Statement 39\n"
                "Five Year Financial Summary 40",
                [],
            ),
            PdfPage(
                16,
                "Notes to the Financial Statements\n"
                "1. Material accounting policies\n"
                "2 New Standards and Interpretations",
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )

    candidates = reviewer._notes_heading_candidate_rows(document)

    contents_row = next(row for row in candidates if row["Page"] == "2")
    assert contents_row["Accepted"] == "No"


def test_ocr_notes_start_page_searches_raw_page_text_with_broken_lines(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(3, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(
                14,
                "NOtes to\n the Financial\n Statements\n1 Significant accounting policies\n2 Revenue\nRevenue 100 90",
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 14
    candidates = result.metrics["notes_heading_candidates"]
    assert any(row["Accepted"] == "Yes" and row["Page"] == "14" for row in candidates)
    assert any(row["Normalized snippet"] for row in candidates)


def test_ocr_notes_start_page_accepts_notes_to_accounts_variant_and_records_candidate_rows(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(12, "Notes to the accounts\n1. Significant accounting policies\n2 Revenue\nRevenue 100 90", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 12
    candidates = result.metrics["notes_heading_candidates"]
    assert any(row["Accepted"] == "Yes" and row["Page"] == "12" for row in candidates)


def test_ocr_notes_candidate_scan_not_blocked_by_later_false_statement_page(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(3, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(14, "Notes to the Financial Statements\n1. Significant accounting policies\n2 Revenue", []),
            PdfPage(30, "Statement of financial position five year summary\nTotal assets 100 90", []),
        ],
        ocr_used=True,
        ocr_pages=3,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 14


def test_ocr_notes_start_page_falls_back_to_significant_accounting_policies_after_statements(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(3, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
            PdfPage(14, "1. Significant accounting policies\n2 Revenue\nRevenue 100 90", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["notes_section_start_page"] == 14
    assert any("Significant accounting policies" in row["Raw OCR snippet"] and row["Accepted"] == "Yes" for row in result.metrics["notes_heading_candidates"])
    assert any(row["Normalized snippet"] for row in result.metrics["notes_heading_candidates"])


def test_notes_heading_candidates_include_rejected_diagnostic_rows(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of financial position\nTotal assets 100 90\nTotal equity and liabilities 100 90\n" + ("OCR filler\n" * 80), []),
            PdfPage(10, "Notes and financial statement extracts\nNarrative only\n" + ("OCR filler\n" * 80), []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert any(row["Accepted"] == "No" for row in result.metrics["notes_heading_candidates"])


def test_excel_export_wires_notes_heading_candidates_sheet():
    app_source = Path("app.py").read_text(encoding="utf-8")

    assert 'sheet_name="Notes heading candidates"' in app_source
    assert 'writer.book["Notes heading candidates"]' in app_source


def test_ocr_heading_based_note_reference_validation_runs_as_review_prompt(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, ("Statement of profit or loss\nOther Revenue Note 7 307,482 189,751\n" * 20), []),
            PdfPage(2, "Notes to the financial statements\n7 Operating revenue\nRevenue 100 90\n9 Other Revenue\nOther revenue 307,482 189,751", []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["note_validation_mode"] == "review_prompt"
    row = next(row for row in result.metrics["note_agreement_results"] if row["Line item"] == "Other Revenue")
    assert row["Alternative note found"] == "9"
    assert row["Result"] == "Review prompt"
    assert not any("possible wrong note reference" in finding.issue.lower() for finding in result.findings)


def test_ifrs_16_checklist_triggers_from_actual_lease_balance_disclosure():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Accounting policies\nIFRS 16 is applied to lease liability balances at commencement date.\nCash and cash equivalents 500",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile(checklist_areas=("IFRS 16",)))

    assert any("IFRS 16" in finding.location for finding in findings)


def test_policy_check_does_not_trigger_lease_policy_from_generic_contract_wording():
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

    assert not any("leases policy" in finding.issue.lower() for finding in findings)


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


def test_currency_check_ignores_generic_policy_currency_words():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Accounting policies\nForeign currency transactions may be denominated in Dollar or Euro as examples.\nStatement of financial position\nPresented in N'000",
                [],
            )
        ]
    )

    findings = check_formatting(document, CompanyProfile())

    assert not any("currency marker" in finding.issue.lower() for finding in findings)


def test_ocr_formatting_ignores_standalone_fragments_in_unstructured_summary():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Five-year financial summary\nRevenue growth history\n13233\nRevenue 13233 12000\n2021 2022 2023\n",
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_formatting(document, CompanyProfile(reporting_currency="NGN"))

    assert not any("thousands separators" in finding.issue.lower() for finding in findings)


def test_ocr_formatting_ignores_financial_review_fragments_without_reliable_table():
    document = PdfDocument(
        [
            PdfPage(
                32,
                "5 year financial review\nRevenue 13233 12000\nAssets 55555 44444\n",
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_formatting(document, CompanyProfile(reporting_currency="NGN"))

    assert not any("thousands separators" in finding.issue.lower() for finding in findings)


def test_ocr_formatting_ignores_low_confidence_summary_table_fragments():
    document = PdfDocument(
        [
            PdfPage(
                32,
                "Five-year financial summary",
                [
                    [
                        ["Revenue", "13233 12000"],
                        ["Assets", "55555 44444"],
                    ]
                ],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_formatting(document, CompanyProfile(reporting_currency="NGN"))

    assert not any("thousands separators" in finding.issue.lower() for finding in findings)


def test_currency_normalization_accepts_naira_aliases_and_rejects_invalid_code():
    assert normalize_reporting_currency("NGN") == "NGN"
    assert normalize_reporting_currency("Naira") == "NGN"
    assert normalize_reporting_currency("₦") == "NGN"
    assert normalize_reporting_currency("N’000") == "NGN"
    assert normalize_reporting_currency("N '000") == "NGN"
    assert normalize_reporting_currency("N ‘000") == "NGN"
    assert normalize_reporting_currency("₦’000") == "NGN"
    assert normalize_reporting_currency("NGN’000") == "NGN"
    assert normalize_reporting_currency("NGN / N'000") == "NGN"
    assert normalize_reporting_currency("N000") == "NGN"
    assert normalize_reporting_currency("NGB") == ""


def test_detected_profile_infers_upload_only_context():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "NATIONAL INSTITUTE OF PROFESSIONAL ADMINISTRATORS OF NIGERIA\nFinancial Statements\nYear ended December 31, 2025\nPrepared in accordance with IFRS and presented in N'000.\nPrincipal activities are professional membership services, professional development, training and certification for members and associates. This paragraph continues with extracted report boilerplate that should not be pasted in full.\nSubscriptions 200\nMembers fund 300\nCash and cash equivalents 100\nTrade and other receivables 50",
                [],
            )
        ]
    )

    profile = infer_detected_profile(document)

    assert profile["Company name"] == "National Institute of Professional Administrators of Nigeria"
    assert profile["Year end"] == "December 31, 2025"
    assert profile["Currency"] == "NGN / N'000"
    assert profile["Framework"] == "IFRS"
    assert "professional body" in profile["Entity type"].lower()
    assert profile["Principal activities"] == "Professional membership body, including member services, professional development, training, and certification."


def test_detected_profile_classifies_limited_property_company_before_professional_body_terms():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Example Property Investments Limited",
                        "Directors' report",
                        "The company has share capital and directors.",
                        "The principal activity is property investment.",
                        "Investment property 1,000 900",
                        "Professional advisers are listed below.",
                    ]
                ),
                [],
            )
        ]
    )

    profile = infer_detected_profile(document)

    assert profile["Entity type"] == "Private company / property investment company"


def test_generic_professional_body_pattern_detects_membership_entity_without_company_name_hardcode():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "SYNTHETIC COUNCIL OF REGISTERED PROFESSIONALS",
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

    assert profile["Company name"] == "Synthetic Council of Registered Professionals"
    assert profile["Entity type"] == "Non-profit / professional body"
    assert "Professional membership body" in profile["Principal activities"]


def test_generic_scanned_private_company_pattern_detects_company_and_runs_sfp_check(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Example Property Investments Limited",
                        "Directors' report",
                        "The company has ordinary shares and share capital.",
                    ]
                ),
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Non-current assets 600 500",
                        "Current assets 400 300",
                        "Total assets 1,000 800",
                        "Equity 700 550",
                        "Liabilities 300 250",
                        "Total equity and liabilities 1,000 800",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    profile = infer_detected_profile(document)
    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert profile["Entity type"] == "Private company / property investment company"
    assert "Statement of financial position: total assets checked" in result.metrics["checks_performed"]
    assert any(row["Result"] == "Passed" and "total assets" in row["Check"].lower() for row in result.metrics["check_results"])


def test_ocr_statement_rows_include_row_confidence(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nCash and cash equivalents 39,387 193,627\nTotal assets 39,387 193,627\nTotal equity and liabilities 39,387 193,627\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "cash and cash equivalents" in result.metrics["ocr_statement_rows"]
    assert "Low-Medium" in result.metrics["ocr_statement_rows"]


def test_text_confidence_separates_table_confidence():
    noisy_table = [["Description", "2025", "2024"], ["Revenue", "100 90", ""], ["Total", "100", "90"]]
    document = PdfDocument(
        [PdfPage(1, "Statement of financial position\n" + ("Revenue 100 90\n" * 120), [noisy_table])]
    )

    assert document.extraction_confidence >= 90
    assert document.table_extraction_confidence < document.extraction_confidence


def test_formatting_ignores_registration_numbers_and_nonfinancial_negatives():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "FRC registration number 816300\nCertificate number 375217\nAddress 12345 Lagos\nThis text has extraction artefact -12345",
                [],
            )
        ]
    )

    findings = check_formatting(document, CompanyProfile())

    assert not any("thousands separators" in finding.issue.lower() for finding in findings)
    assert not any("negative" in finding.issue.lower() for finding in findings)


def test_superseded_standard_context_classifies_current_policy_as_high_and_transition_as_low():
    current_policy = PdfDocument(
        [PdfPage(1, "Accounting policies\nFinancial instruments are accounted for under IAS 39.", [])]
    )
    transition_note = PdfDocument(
        [PdfPage(2, "New and amended standards\nIAS 39 was replaced by IFRS 9 and is discussed as transition history.", [])]
    )

    current_findings = check_policy_relevance(current_policy, CompanyProfile())
    transition_findings = check_policy_relevance(transition_note, CompanyProfile())

    assert any(finding.severity == "High" and "Page 1" in finding.location for finding in current_findings)
    assert any(finding.severity == "Low" and "Page 2" in finding.location for finding in transition_findings)
    assert all("Context:" in finding.evidence for finding in current_findings if finding.category == "Accounting policies")


def test_consolidation_policy_is_not_triggered_by_generic_group_wording():
    document = PdfDocument(
        [PdfPage(1, "Accounting policies\nThe group of standards issued by the IASB is reviewed annually.", [])]
    )

    findings = check_policy_relevance(document, CompanyProfile())

    assert not any("consolidation-related" in finding.issue.lower() for finding in findings)


def test_revenue_policy_recognises_revenue_from_contracts_with_customers():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 1,000 900\nNotes to the financial statements\nAccounting policies\nRevenue from contracts with customers is recognised when control transfers.",
                [],
            )
        ]
    )

    findings = check_policy_relevance(document, CompanyProfile())

    assert not any("revenue-related balances" in finding.issue.lower() for finding in findings)


def test_revenue_policy_recognises_ocr_variant_revenue_from_contract_with_customer():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 1,000 900\nNotes to the financial statements\nAccounting policies\nRevenue from contract with customer is recognised when control transfers.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("revenue-related balances" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 15" in finding.location for finding in checklist_findings)


def test_revenue_policy_recognises_broken_ocr_revenue_contracts_customers_phrase():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 1,000 900\nNotes to financial statements\nAccounting policies\nRevenue contracts customers recognised when control transfers.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())

    assert not any("revenue-related balances" in finding.issue.lower() for finding in policy_findings)


def test_revenue_policy_recognises_split_ocr_revenue_from_contracts_with_customers_phrase():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of profit or loss\nRevenue 1,000 900\nNotes to financial statements\n"
                "Accounting policies\nRevenue from\ncontracts with\ncustorners is recognised when control transfers.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("revenue-related balances" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 15" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_generic_standards_amendments_text():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew standards and amendments\nIFRS 16 Leases amendments are effective for annual periods beginning after the reporting date.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_right_of_use_wording_inside_amendments_section():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew standards and amendments\nAmendments mention right-of-use asset and lease liability examples effective for annual periods beginning after the reporting date.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_generic_deferred_tax_single_transaction_text():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew standards and amendments\n"
                "Deferred tax related to assets and liabilities arising from a single transaction includes examples for right-of-use asset and lease liability.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_generic_new_and_amended_standards_text():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew and amended standards issued but not effective\n"
                "The amendments to IFRS 16 discuss lease liability and right-of-use asset measurement examples.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_theoretical_lease_asset_liability_wording_even_if_forced():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew standards and deferred tax amendments\n"
                "The recognition of a lease asset and lease liability may arise from theoretical IFRS 16 examples.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile(checklist_areas=("IFRS 16",)))

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_triggers_from_actual_lease_liability_amount():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nLease liabilities\nCurrent lease liability 125\nNon-current lease liability 430",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile())

    assert any("IFRS 16" in finding.location for finding in findings)


def test_ifrs_16_triggers_from_actual_lease_arrangement_disclosure():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nThe company leases office premises under a cancellable lease arrangement.",
                [],
            )
        ]
    )

    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert any("IFRS 16" in finding.location for finding in checklist_findings)


def test_ifrs_16_not_triggered_by_generic_lease_contract_policy_wording():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Accounting policies\nIFRS 16 is applied to lease contracts at commencement date.",
                [],
            )
        ]
    )

    policy_findings = check_policy_relevance(document, CompanyProfile())
    checklist_findings = check_standard_checklist(document, CompanyProfile())

    assert not any("leases policy" in finding.issue.lower() for finding in policy_findings)
    assert not any("IFRS 16" in finding.location for finding in checklist_findings)


def test_detected_profile_does_not_suggest_ifrs_16_from_generic_new_standards_text():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Notes to the financial statements\nNew standards and amendments\nIFRS 16 right-of-use asset and lease liability examples are discussed as amendments.",
                [],
            )
        ]
    )

    profile = infer_detected_profile(document)

    assert "IFRS 16" not in profile["Suggested checklist areas"]


def test_ocr_sfp_statement_specific_checks_run_only_on_confident_rows():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\n" + ("Non-current assets current assets total assets equity liabilities\n" * 80),
                [
                    [
                        ["Description", "2025"],
                        ["Non-current assets", "100"],
                        ["Current assets", "50"],
                        ["Total assets", "153"],
                        ["Equity", "60"],
                        ["Liabilities", "90"],
                        ["Total equity and liabilities", "150"],
                    ]
                ],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("Non-current assets + current assets" in finding.issue for finding in findings)


def test_ocr_sfp_line_based_checks_run_when_tables_are_not_structured():
    document = PdfDocument(
        [
            PdfPage(
                1,
                ("Statement of financial position\n" * 40)
                + "\n".join(
                    [
                        "Non-current assets 100 90",
                        "Current assets 50 40",
                        "Total assets 153 130",
                        "Equity 60 50",
                        "Liabilities 90 80",
                        "Total equity and liabilities 150 130",
                    ]
                ),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("Non-current assets + current assets" in finding.issue for finding in findings)


def test_confidence_metrics_separate_ocr_text_from_statement_structure(monkeypatch):
    document = PdfDocument(
        [PdfPage(1, "OCR text without parseable statement rows\n" * 80, [])],
        ocr_used=True,
        ocr_pages=1,
        ocr_tables=3,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert result.metrics["ocr_text_coverage"] == "100%"
    assert result.metrics["statement_structure_confidence"] == "0%"
    assert result.metrics["table_arithmetic_confidence"] == "0%"
    assert result.metrics["ocr_tables"] == 3


def test_statement_structure_confidence_reflects_checkable_rows_not_headings_only(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue text without amounts\n" + ("OCR filler\n" * 80), []),
            PdfPage(2, "Statement of financial position\nTotal assets 500 400\n" + ("OCR filler\n" * 80), []),
            PdfPage(3, "Statement of changes in equity\nMovement narrative only\n" + ("OCR filler\n" * 80), []),
            PdfPage(4, "Statement of cash flows\nCash flow narrative only\n" + ("OCR filler\n" * 80), []),
        ],
        ocr_used=True,
        ocr_pages=4,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert int(result.metrics["statement_structure_confidence"].rstrip("%")) < 40
    assert "Statement of financial position: equity and liabilities equation checked" not in result.metrics["checks_performed"]


def test_fuzzy_ocr_statement_page_detection_classifies_distorted_headings(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statememt of financiaI position\nTotal assets 100 90\nTotal equity and liabilities 100 90\n" * 20, []),
            PdfPage(2, "Statment of profit or Ioss\nRevenue 100 90\nProfit 10 8\n" * 20, []),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Statement of financial position | Page 1" in result.metrics["primary_statement_pages"]
    assert "Statement of income and expenditure | Page 2" in result.metrics["primary_statement_pages"]
    assert result.metrics["statement_structure_confidence"] != "0%"


def test_review_report_includes_ocr_statement_rows_debug(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Non current assets 600 500",
                        "Current assets 400 300",
                        "Total assets 1,000 800",
                        "Equity 700 550",
                        "Liabilities 300 250",
                        "Total equity and liability 1,000 800",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "non-current assets" in result.metrics["ocr_statement_rows"]
    assert "total equity and liabilities" in result.metrics["ocr_statement_rows"]
    assert "Statement of financial position: equity and liabilities equation checked" in result.metrics["checks_performed"]


def test_ocr_statement_rows_ignore_title_and_date_lines(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Example Property Investments Limited financial statements for the year ended 31 December 2021 2020",
                        "Statement of profit or loss and other comprehensive income",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (173,516) (681,559)",
                        "Taxation 253,124",
                        "Loss after taxation (120,389) (428,435)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Statement of financial position as at 31 December 2021 2020",
                        "Non-current assets 1,200,000 1,100,000",
                        "Current assets 300,000 250,000",
                        "Total assets 1,500,000 1,350,000",
                        "Equity 900,000 850,000",
                        "Liabilities 600,000 500,000",
                        "Trade and other receivables 1,910,631 131,254",
                        "Cash and cash equivalents 739,387 193,627",
                        "Financial liabilities 5,356,392 4,555,742",
                        "Total equity and liabilities 1,500,000 1,350,000",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    rows = result.metrics["ocr_statement_rows"]

    assert "revenue |  | 707,189 | 297,041" in rows
    assert "profit before tax |  | -173,516 | -681,559" in rows
    assert "taxation |  | 253,124" in rows
    assert "profit after tax |  | -120,389 | -428,435" in rows
    assert "non-current assets |  | 1,200,000 | 1,100,000" in rows
    assert "trade and other receivables |  | 1,910,631 | 131,254" in rows
    assert "cash and cash equivalents |  | 739,387 | 193,627" in rows
    assert "financial liabilities |  | 5,356,392 | 4,555,742" in rows
    assert "total equity and liabilities |  | 1,500,000 | 1,350,000" in rows
    assert "financial statements for the year ended" not in rows.lower()
    assert "Statement of financial position: equity and liabilities equation checked" in result.metrics["checks_performed"]


def test_ocr_statement_rows_ignore_table_of_contents_pages(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                2,
                "\n".join(
                    [
                        "Contents",
                        "Statement of profit or loss ........ 10",
                        "Statement of financial position .... 11",
                        "Statement of changes in equity ..... 12",
                        "Statement of cash flows ............ 13",
                        "Equity 11",
                    ]
                ),
                [],
            ),
            PdfPage(
                11,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Investment property 4,900,000 4,850,000",
                        "Trade and other receivables 1,910,631 131,254",
                        "Cash and cash equivalents 739,387 193,627",
                        "Total assets 7,550,018 5,174,881",
                        "Equity 2,193,626 619,139",
                        "Financial liabilities 5,356,392 4,555,742",
                        "Total equity and liabilities 7,550,018 5,174,881",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    rows = result.metrics["ocr_statement_rows"]

    assert "equity | 11" not in rows.lower()
    assert "Statement of financial position | Page 11" in rows
    assert "financial liabilities |  | 5,356,392 | 4,555,742" in rows


def test_ocr_statement_rows_drive_partial_statement_checks(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (173,516) (681,559)",
                        "Taxation 53,127 253,124",
                        "Loss after taxation (120,389) (428,435)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Investment property 4,900,000 4,850,000",
                        "Total non-current assets 4,900,000 4,850,000",
                        "Trade and other receivables 1,910,631 131,254",
                        "Cash and cash equivalents 739,387 193,627",
                        "Total current assets 2,650,018 324,881",
                        "Total assets 7,550,018 5,174,881",
                        "Equity 2,193,626 619,139",
                        "Financial liabilities 5,356,392 4,555,742",
                        "Total equity and liabilities 7,550,018 5,174,881",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Income statement: revenue, tax, and profit/loss after tax checked" in result.metrics["checks_performed"]
    assert "Statement of financial position: total assets checked from extracted asset rows." in result.metrics["checks_performed"]
    assert "Statement of financial position: equity and liabilities equation checked" in result.metrics["checks_performed"]
    assert "Statement-specific OCR checks were skipped" not in "\n".join(finding.issue for finding in result.findings)
    assert result.metrics["checks_passed_count"] >= 2
    check_results = result.metrics["check_results"]
    assert any(
        row["Result"] == "Passed" and "total assets" in row["Check"].lower()
        for row in check_results
    )


def test_ocr_income_statement_runs_current_year_only_when_tax_row_has_one_amount(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (173,516) (681,559)",
                        "Taxation 253,124",
                        "Loss after taxation (120,389) (428,435)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Income statement: revenue, tax, and profit/loss after tax checked" in result.metrics["checks_performed"]
    tax_findings = [finding for finding in result.findings if "Profit/loss after tax" in finding.issue]
    assert tax_findings
    assert all(finding.severity in {"Medium", "Low"} for finding in tax_findings)
    assert all("Possible mismatch from OCR line extraction" in finding.issue for finding in tax_findings)
    assert "Loss/profit before taxation raw line" in tax_findings[0].evidence
    assert "Taxation raw line" in tax_findings[0].evidence
    assert "Extracted values" in tax_findings[0].evidence


def test_ocr_income_statement_prompt_includes_corroborating_report_values(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Directors' report\nThe loss before taxation was (173,516). Tax credit was 53,127. Loss after taxation was (120,389).",
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (473,516) (681,559)",
                        "Taxation 53,127",
                        "Loss after taxation (120,389) (428,435)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
            PdfPage(3, "Statement of cash flows\nLoss before taxation (173,516)\nTaxation 53,127", []),
        ],
        ocr_used=True,
        ocr_pages=3,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    tax_findings = [finding for finding in result.findings if "Profit/loss after tax" in finding.issue]

    assert tax_findings
    assert tax_findings[0].severity == "Low"
    assert tax_findings[0].issue.startswith("OCR conflict - manual confirmation required")
    assert "Corroborating OCR values" in tax_findings[0].evidence
    assert "Corroboration assessment" in tax_findings[0].evidence
    assert "before tax=+1" not in tax_findings[0].evidence
    assert "before tax=(173,516)" in tax_findings[0].evidence
    assert "Page 1" in tax_findings[0].evidence
    assert "Page 3" in tax_findings[0].evidence


def test_ocr_income_incomplete_current_year_after_tax_uses_clear_manual_confirmation_wording(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Directors' report\nThe loss before taxation was (221,494). Taxation was (226,684). Loss after taxation was (448,178).",
                [],
            ),
            PdfPage(
                2,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (221,494) (173,516)",
                        "Taxation (226,684) 53,127",
                        "Loss for the year (999,999)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            ),
        ],
        ocr_used=True,
        ocr_pages=2,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    tax_findings = [finding for finding in result.findings if "Profit/loss after tax" in finding.issue]

    assert not tax_findings
    assert "current-year after-tax value not confidently extracted" in result.metrics["checks_skipped"]
    assert "Corroborating lines indicate (448,178)" in result.metrics["checks_skipped"]
    assert "Manual confirmation required" in result.metrics["checks_skipped"]


def test_ocr_income_incomplete_after_tax_skips_arithmetic_instead_of_mixing_years(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of profit or loss",
                        "Revenue 707,189 297,041",
                        "Loss before taxation (221,494) (173,516)",
                        "Taxation (226,684) 53,127",
                        "Loss for the year (999,999)",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))

    assert "Skipped / OCR conflict - current-year after-tax value not confidently extracted." in result.metrics["checks_skipped"]
    assert not any("Expected 5,190" in finding.evidence for finding in result.findings)
    assert not any("reported -120,389" in finding.evidence for finding in result.findings)


def test_ocr_statement_mismatches_are_review_prompts_not_high(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(
                1,
                "\n".join(
                    [
                        "Statement of financial position",
                        "Total assets 100 90",
                        "Total equity and liabilities 80 70",
                    ]
                )
                + "\n"
                + ("Additional OCR statement text for coverage.\n" * 80),
                [],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )
    monkeypatch.setattr(reviewer, "extract_pdf", lambda _path: document)

    result = review_pdf("unused.pdf", options=ReviewOptions(run_cautious_note_agreement=True))
    total_findings = [finding for finding in result.findings if finding.category == "Totals and rounding"]

    assert total_findings
    assert all(finding.severity in {"Medium", "Low"} for finding in total_findings)
    assert all("Possible mismatch from OCR line extraction" in finding.issue for finding in total_findings)


def test_ocr_sfp_statement_specific_checks_skip_low_confidence_rows():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\nTotal assets ########",
                [[["Description", "2025"], ["Non-current assets", "100 50"], ["Total assets", "150"]]],
            )
        ],
        ocr_used=True,
        ocr_pages=1,
    )

    findings = check_rounding_and_casting(document, tolerance=Decimal("1"))

    assert any("statement-specific ocr checks were skipped" in finding.issue.lower() for finding in findings)
    assert not any(finding.category == "Totals and rounding" for finding in findings)


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
                "Statement of profit or loss\nProfit for the year Note 9 100\n"
                + ("ordinary shares earnings per share disclosure\n" * 80)
                + "Notes to the financial statements\n9 Earnings per share\nProfit attributable 100\nWeighted average shares 20\nBasic EPS 8",
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
                "Statement of profit or loss\nContract asset 500\nNotes to the financial statements\nRevenue is recognised when control transfers.",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile())

    assert any(finding.category == "Standards checklist" and "IFRS 15" in finding.location for finding in findings)


def test_standard_checklist_specific_triggers_avoid_irrelevant_eps_and_segments():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Customer segment credit risk ageing\nSegment by customer type\nOrdinary members attended training",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile())

    assert not any("IAS 33" in finding.location for finding in findings)
    assert not any("IFRS 8" in finding.location for finding in findings)


def test_standard_checklist_suppresses_tax_exempt_and_no_subsequent_events():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "The entity is tax-exempt and not subject to income tax.\nSubsequent events\nThere were no subsequent events after the reporting period.",
                [],
            )
        ]
    )

    findings = check_standard_checklist(document, CompanyProfile())

    assert not any("IAS 12" in finding.location for finding in findings)
    assert not any("IAS 10" in finding.location for finding in findings)


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


def test_extraction_quality_flags_placeholder_values():
    document = PdfDocument([PdfPage(1, "Revenue #####\nTotal income #####", [])])

    findings = check_extraction_quality(document)

    assert any("Unreadable" in finding.issue for finding in findings)
    assert any("too low" in finding.issue for finding in findings)


def test_extraction_quality_flags_merged_numeric_cells():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Revenue 100 90\n" * 120,
                [[["Description", "2025", "2024"], ["Revenue", "100 90", ""], ["Total", "100", "90"]]],
            )
        ]
    )

    findings = check_extraction_quality(document)

    assert any("merged values" in finding.issue.lower() for finding in findings)


def test_clean_text_document_has_high_extraction_confidence():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "Statement of financial position\n" + ("Revenue 100 90\n" * 120),
                [[["Description", "2025", "2024"], ["Revenue", "100", "90"], ["Total revenue", "100", "90"]]],
            )
        ]
    )

    assert document.extraction_profile == "text-based"
    assert document.extraction_confidence >= 90


def test_text_based_document_with_layout_noise_is_not_blocked():
    noisy_table = [["Description", "2025", "2024"]]
    noisy_table.extend([["Line item", "100 90", ""] for _ in range(20)])
    document = PdfDocument(
        [PdfPage(1, "Statement of financial position\n" + ("Revenue 100 90\n" * 120), [noisy_table])]
    )

    findings = check_extraction_quality(document)

    assert document.extraction_confidence >= 50
    assert not any("too low" in finding.issue for finding in findings)


def test_ocr_fallback_does_not_crash_when_tesseract_missing(monkeypatch):
    monkeypatch.setattr(extraction, "_resolve_tesseract", lambda: "")
    base_document = PdfDocument([PdfPage(1, "", [])])

    document = extract_pdf_with_ocr("missing.pdf", base_document, ReviewOptions(use_ocr=True))

    assert document.ocr_error
    assert document.pages == base_document.pages


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


def test_notes_agreement_flags_missing_note_reference_on_face_statement(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nRevenue 10,000 8,000", []),
            PdfPage(2, "Notes to the financial statements\n1. Revenue\nSales of goods 10,000", []),
        ]
    )
    monkeypatch.setattr(PdfDocument, "table_extraction_confidence", property(lambda self: 100))
    findings = check_notes_agreement(document)
    finding = next((f for f in findings if "lacks a note reference" in f.issue), None)
    assert finding is not None
    assert finding.severity == "Medium"
    assert "Revenue" in finding.issue
    assert "Note 1" in finding.issue
    assert "Suggested Note 1" in finding.evidence


def test_notes_agreement_flags_missing_note_reference_when_no_note_found(monkeypatch):
    document = PdfDocument(
        [
            PdfPage(1, "Statement of profit or loss\nAssets 5,000 4,000", []),
            PdfPage(2, "Notes to the financial statements\n1. Revenue\nSales of goods 10,000", []),
        ]
    )
    monkeypatch.setattr(PdfDocument, "table_extraction_confidence", property(lambda self: 100))
    findings = check_notes_agreement(document)
    finding = next((f for f in findings if "has no note reference" in f.issue), None)
    assert finding is not None
    assert finding.severity == "Low"
    assert "Assets" in finding.issue
    assert "no matching note was found" in finding.issue
