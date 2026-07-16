from canonical_checks import run_canonical_checks
from models import PdfDocument, PdfPage


def test_cash_flow_movement_difference_is_flagged():
    text = '''Example Limited
Statement of Cash Flows
2025 2024
N'000 N'000
Net cash used in operating activities (100) (200)
Net cash used in investing activities (50) (50)
Net cash generated from financing activities 20 20
Total cash movement for the year (120) (230)
Cash at the beginning of the year 500 730
Effect of exchange rate movement on cash balances - -
Total cash at end of the year 370 500
'''
    doc = PdfDocument([PdfPage(1, text, [])])
    findings, checks, audit = run_canonical_checks(doc)
    failed = [check for check in checks if check.status == "Fail"]
    assert any(check.check_name == "Cash flow net movement cast" for check in failed)
    assert findings


def test_changes_in_equity_closing_balance_cast_passes():
    pl_text = """Example Limited
Statement of profit or loss and other comprehensive income
2025 2024
Profit for the year 40 30
"""
    equity_text = """Example Limited
Statement of changes in equity
2025 2024
Balance at 1 January 100 70
Profit for the year 40 30
Dividends (10) -
Balance at 31 December 130 100
"""
    doc = PdfDocument([PdfPage(1, pl_text, []), PdfPage(2, equity_text, [])])
    findings, checks, audit = run_canonical_checks(doc)
    assert any(check.check_name == "Changes in equity closing balance cast" and check.status == "Pass" for check in checks)
    assert not [finding for finding in findings if finding.category == "Equity movement"]


def test_note_reference_compatibility_flags_wrong_heading_generically():
    doc = PdfDocument([
        PdfPage(1, """Example Limited
Statement of financial position
2025 2024
N'000 N'000
Intangible assets 4 100 90
Total assets 100 90
Total equity and liabilities 100 90
""", []),
        PdfPage(2, "Notes to the Financial Statements\n1. Significant accounting policies\n4 Property, plant and equipment", []),
    ])
    findings, checks, audit = run_canonical_checks(doc)
    assert any(check.check_name == "Face statement note reference compatibility" and check.status == "Fail" for check in checks)
    assert any("Intangible assets" in finding.evidence for finding in findings)


def test_note_reference_compatibility_accepts_matching_heading_generically():
    doc = PdfDocument([
        PdfPage(1, """Example Limited
Statement of financial position
2025 2024
N'000 N'000
Intangible assets 4 100 90
Total assets 100 90
Total equity and liabilities 100 90
""", []),
        PdfPage(2, "Notes to the Financial Statements\n1. Significant accounting policies\n4 Intangible assets", []),
    ])
    findings, checks, audit = run_canonical_checks(doc)
    assert not any(check.check_name == "Face statement note reference compatibility" and check.status == "Fail" for check in checks)
