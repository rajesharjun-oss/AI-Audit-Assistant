from decimal import Decimal

import extraction
import reviewer
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
    assert headings["11"][0] == "Personnel costs"
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
    assert any("possible wrong note reference" in finding.issue.lower() for finding in cautious_findings)
    assert all(finding.severity in {"Low", "Medium"} for finding in cautious_findings if "possible wrong note reference" in finding.issue.lower())


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

    assert wrong_ref
    assert wrong_ref[0].severity == "Low"
    assert wrong_ref[0].metadata["referenced_note"] == "7"
    assert wrong_ref[0].metadata["suggested_note"] == "9"
    assert wrong_ref[0].metadata["reason"]


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

    assert "Cautious note-reference validation performed in review-prompt mode; no possible wrong note references detected." in result.metrics["checks_performed"]
    assert "Cautious note-reference validation skipped" not in result.metrics["checks_skipped"]
    assert result.metrics["cautious_note_validation_enabled"] is True
    assert result.metrics["note_validation_mode"] == "review_prompt"
    assert result.metrics["note_reference_rows_detected"] == 0
    assert result.metrics["note_headings_detected"] == 0
    assert result.metrics["note_reference_findings"] == 0


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
            "1S Funtierra Limited",
            "4 Funtierra Limited",
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


def test_currency_normalization_accepts_naira_aliases_and_rejects_invalid_code():
    assert normalize_reporting_currency("NGN") == "NGN"
    assert normalize_reporting_currency("Naira") == "NGN"
    assert normalize_reporting_currency("₦") == "NGN"
    assert normalize_reporting_currency("N’000") == "NGN"
    assert normalize_reporting_currency("N000") == "NGN"
    assert normalize_reporting_currency("NGB") == ""


def test_detected_profile_infers_upload_only_context():
    document = PdfDocument(
        [
            PdfPage(
                1,
                "CHARTERED INSTITUTE OF PERSONNEL MANAGEMENT OF NIGERIA\nFinancial Statements\nYear ended December 31, 2025\nPrepared in accordance with IFRS and presented in N'000.\nPrincipal activities are professional membership services, professional development, training and certification for personnel management practitioners. This paragraph continues with extracted report boilerplate that should not be pasted in full.\nCash and cash equivalents 100\nTrade and other receivables 50",
                [],
            )
        ]
    )

    profile = infer_detected_profile(document)

    assert profile["Company name"] == "Chartered Institute of Personnel Management of Nigeria"
    assert profile["Year end"] == "December 31, 2025"
    assert profile["Currency"] == "NGN"
    assert profile["Framework"] == "IFRS"
    assert "professional body" in profile["Entity type"].lower()
    assert profile["Principal activities"] == "Professional membership body for personnel management, including member services, professional development, training, and certification."


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
