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


def test_round_thousand_amounts_are_not_truncated():
    # A grouped thousand ends in "000"; the units-suffix strip must not eat it.
    assert parse_amount_cell("5,000").value == Decimal("5000")
    assert parse_amount_cell("2,000").value == Decimal("2000")
    assert parse_amount_cell("100,000").value == Decimal("100000")
    assert parse_amount_cell("1,000,000").value == Decimal("1000000")
    assert parse_amount_cell("(2,000)").value == Decimal("-2000")


def test_grouped_value_matching_a_year_is_still_an_amount():
    # "2,000" / "(2,000)" are amounts, not the reporting year 2000.
    assert parse_amount_cell("2,000").kind == "amount"
    assert parse_amount_cell("(2,050)").value == Decimal("-2050")
    # A bare four-digit token with no grouping is still treated as a year,
    # including signed/bracketed forms of a bare year.
    assert parse_amount_cell("2025").kind == "year"
    assert parse_amount_cell("(2024)").kind == "year"
    assert parse_amount_cell("-2025").kind == "year"


def test_explicit_scale_suffix_still_stripped():
    # The legitimate "in 000s" / "N'000" units suffix must still be removed.
    assert parse_amount_cell("5000s").value == Decimal("5")
    assert parse_amount_cell("5N'000").value == Decimal("5")


def test_calendar_day_in_date_caption_is_not_an_amount():
    # The day number in a statement date caption must not become a phantom amount.
    assert extract_amount_cells("As at 31 December 2025") == []
    # A real balance row keeps its amounts but drops the calendar day.
    assert [c.value for c in extract_amount_cells("Balance at 1 January 2025 5,000")] == [Decimal("5000")]
    assert [c.value for c in extract_amount_cells("Balance at 31 December 2025 6,100 5,300")] == [
        Decimal("6100"),
        Decimal("5300"),
    ]
    # A grouped value that merely starts with a day-like number is still an amount.
    assert [c.value for c in extract_amount_cells("Profit 31,000 28,000")] == [Decimal("31000"), Decimal("28000")]
