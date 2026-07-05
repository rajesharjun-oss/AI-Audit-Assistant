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
