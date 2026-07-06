from decimal import Decimal

from canonical_extraction import extract_statement_facts, facts_for, note_heading_map
from models import PdfDocument, PdfPage


def _doc(text):
    return PdfDocument([PdfPage(1, text, [])])


def test_group_company_columns_and_note_reference_are_mapped():
    text = '''Example Limited
Consolidated and Separate Statements of Financial Position
Notes 2025 2024 2025 2024
N'000 N'000 N'000 N'000
Assets
Cash and cash equivalents 4 1,759,784 168,203 206,603 66,358
Bank overdraft 4 2,955,300 2,106,407 2,955,300 2,106,407
Total assets 33,264,220 17,882,380 28,059,917 15,380,685
Total liabilities and equity 33,264,220 17,882,380 28,059,917 15,380,685
'''
    facts = extract_statement_facts(_doc(text))
    group_cash = facts_for(facts, canonical="cash and cash equivalents", entity="Group", year=2025)[0]
    company_cash = facts_for(facts, canonical="cash and cash equivalents", entity="Company", year=2025)[0]
    assert group_cash.amount == Decimal("1759784")
    assert company_cash.amount == Decimal("206603")
    assert group_cash.note_ref == "4"


def test_note_heading_map_supports_numbered_notes():
    doc = PdfDocument([PdfPage(1, "Notes to the Financial Statements\n4. Cash and cash equivalents\n5 Interest income", [])])
    headings = note_heading_map(doc)
    assert headings["4"] == "Cash and cash equivalents"
    assert headings["5"] == "Interest income"


def test_note_heading_map_ignores_numbered_front_matter_before_notes_section():
    doc = PdfDocument([
        PdfPage(1, "Directors' report\n7 Directors' interests in shares\n11 Employment and employees", []),
        PdfPage(2, "Statement of financial position\n2025 2024\nTotal assets 100 90\nTotal equity and liabilities 100 90", []),
        PdfPage(3, "Notes to the Financial Statements\n1. Significant accounting policies\n3 Investment property\nThis represents the fair value movement", []),
    ])
    headings = note_heading_map(doc)
    assert headings["1"] == "Significant accounting policies"
    assert headings["3"] == "Investment property"
    assert "7" not in headings
    assert "11" not in headings
