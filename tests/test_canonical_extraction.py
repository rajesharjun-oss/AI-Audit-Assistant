from decimal import Decimal

from canonical_extraction import detect_statement_columns, document_section_map, extract_statement_facts, facts_for, note_heading_map, table_classification_rows
from models import PdfDocument, PdfPage


def _doc(text):
    return PdfDocument([PdfPage(1, text, [])])


def test_stacked_comparative_years_not_collapsed_to_single_year_soce():
    # Comparative years split across separate header lines leave header_lines
    # empty, but the SoCE must not be treated as single-year (which would keep
    # only the rightmost column and mislabel it with the max year).
    text = """Statement of changes in equity
For the year ended 31 December 2025
2025
2024
Balance at 1 January 100 90
Balance at 31 December 130 110
"""
    columns = detect_statement_columns(text, "Statement of changes in equity")
    # Either two comparative columns are detected, or none are, but never a
    # single fabricated column spanning both years.
    assert len(columns) != 1
    years = {column.year for column in columns}
    assert not (len(columns) == 1 and years == {2025})


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


def test_document_section_map_and_table_classification_are_generic():
    doc = PdfDocument([
        PdfPage(1, "Contents\nStatement of financial position 5\nNotes to the financial statements 12", []),
        PdfPage(2, "Directors' report\n7 Directors' interests in shares", []),
        PdfPage(
            5,
            "Example Limited\nStatement of financial position\n2025 2024\nNon-current assets 60 50\nCurrent assets 40 40\nTotal assets 100 90",
            [["", "2025", "2024"], ["Non-current assets", "60", "50"], ["Current assets", "40", "40"], ["Total assets", "100", "90"]],
        ),
        PdfPage(12, "Notes to the Financial Statements\n1. Significant accounting policies\n3 Property, plant and equipment", []),
        PdfPage(30, "Five-year financial summary\nRevenue 1 2 3 4 5", [["Revenue", "1", "2", "3", "4", "5"]]),
    ])
    section_rows = document_section_map(doc)
    assert section_rows[0]["Section"] == "Contents"
    assert section_rows[1]["Section"] == "Front matter / reports"
    assert section_rows[2]["Section"] == "Primary statement"
    assert section_rows[3]["Section"] == "Notes to the financial statements"
    assert section_rows[4]["Section"] == "Supplementary schedule"

    table_rows = table_classification_rows(doc)
    assert any(row["Table type"] == "Primary statement table" for row in table_rows)
    assert any(row["Table type"] == "Supplementary schedule" for row in table_rows)
