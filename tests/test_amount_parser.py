from decimal import Decimal

from amount_parser import extract_amount_cells, parse_amount_cell


def test_bracketed_negative_parses_as_negative():
    cell = parse_amount_cell("(1,234)")
    assert cell.value == Decimal("-1234")
    assert cell.is_negative


def test_dash_can_be_zero_or_blank():
    assert parse_amount_cell("-").value == Decimal("0")
    assert parse_amount_cell("-", allow_dash_zero=False).value is None


def test_years_and_percentages_do_not_become_amounts_by_default():
    assert parse_amount_cell("2025").kind == "year"
    assert parse_amount_cell("30%").kind == "percentage"


def test_extract_amount_cells_keeps_rightmost_amounts_when_note_ref_present():
    cells = extract_amount_cells("Cash and cash equivalents 4 1,759,784 168,203 206,603 66,358", expected_amounts=4)
    assert [cell.value for cell in cells] == [Decimal("1759784"), Decimal("168203"), Decimal("206603"), Decimal("66358")]
