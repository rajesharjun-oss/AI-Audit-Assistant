from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

AmountKind = Literal["amount", "blank", "dash_zero", "percentage", "year", "note_ref", "page_ref", "identifier", "unreadable", "invalid"]

GROUPED_NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?\)?")
PLAIN_NUMBER_RE = re.compile(r"\(?-?\d+(?:\.\d+)?\)?")
AMOUNT_TOKEN_RE = re.compile(
    r"\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?%?|\(?-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?\)?%?"
    r"|\(?-?\d+(?:\.\d+)?\)?%?"
    r"|(?:(?<=\s)|^)[-–—](?=\s|$)"
)
UNREADABLE_RE = re.compile(r"(#{3,}|\uFFFD|\u25A1)")


@dataclass(frozen=True)
class AmountCell:
    raw: str
    value: Decimal | None
    kind: AmountKind
    is_negative: bool = False
    is_zero: bool = False
    scale: Decimal = Decimal("1")
    confidence: Literal["High", "Medium", "Low"] = "High"
    reason: str = ""


def parse_amount_cell(
    raw: object,
    *,
    scale: Decimal | int | str = Decimal("1"),
    allow_dash_zero: bool = True,
    allow_percentages: bool = False,
    reject_years: bool = True,
    reject_small_note_refs: bool = False,
    reject_page_refs: bool = False,
    context: str = "",
) -> AmountCell:
    """Parse one PDF/table token into a typed financial amount.

    The parser is context-aware: years, note references, percentages, page
    numbers and registration-style identifiers should not silently become
    monetary amounts.
    """
    token = _normalise_token(raw)
    scale_decimal = Decimal(str(scale))
    if not token:
        return AmountCell(str(raw or ""), None, "blank", scale=scale_decimal, reason="Empty token.")
    if UNREADABLE_RE.search(token):
        return AmountCell(token, None, "unreadable", scale=scale_decimal, confidence="Low", reason="Token contains unreadable placeholder characters.")
    if token in {"-", "–", "—", "--"}:
        if allow_dash_zero:
            return AmountCell(token, Decimal("0"), "dash_zero", is_zero=True, scale=scale_decimal, reason="Dash treated as nil in a financial statement amount column.")
        return AmountCell(token, None, "blank", scale=scale_decimal, reason="Dash treated as blank in this context.")

    candidate = _strip_currency_and_units(token)
    if _looks_like_identifier(candidate):
        return AmountCell(token, None, "identifier", scale=scale_decimal, confidence="High", reason="Token resembles an identifier rather than an amount.")

    is_percentage = candidate.endswith("%")
    if is_percentage:
        if not allow_percentages:
            return AmountCell(token, None, "percentage", scale=scale_decimal, reason="Percentage token excluded from monetary amount parsing.")
        candidate = candidate[:-1].strip()

    negative = False
    if candidate.startswith("(") and candidate.endswith(")"):
        negative = True
        candidate = candidate[1:-1].strip()
    if candidate.startswith("-"):
        negative = True
        candidate = candidate[1:].strip()

    normalized = _normalize_numeric_text(candidate)
    if not normalized:
        return AmountCell(token, None, "invalid", scale=scale_decimal, confidence="Low", reason="No numeric content remained after normalization.")

    # A thousands-grouped value ("2,000") or a bracketed/negative value
    # ("(2,000)") is unambiguously a monetary amount, never a reporting year, so
    # it must not be discarded by the four-digit year guard.
    grouped_or_signed = bool(re.search(r"\d[,.\s]\d{3}", candidate)) or negative
    if reject_years and not grouped_or_signed and _looks_like_year(normalized, context):
        return AmountCell(token, None, "year", scale=scale_decimal, reason="Four-digit reporting year excluded from amount parsing.")
    if reject_page_refs and _looks_like_page_reference(normalized, context):
        return AmountCell(token, None, "page_ref", scale=scale_decimal, reason="Page reference excluded from amount parsing.")
    if reject_small_note_refs and _looks_like_note_reference(normalized):
        return AmountCell(token, None, "note_ref", scale=scale_decimal, reason="Small integer note reference excluded from amount parsing.")

    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return AmountCell(token, None, "invalid", scale=scale_decimal, confidence="Low", reason="Token could not be converted to Decimal.")
    if negative and value != 0:
        value = -value
    value *= scale_decimal
    return AmountCell(
        token,
        value,
        "percentage" if is_percentage else "amount",
        is_negative=value < 0,
        is_zero=value == 0,
        scale=scale_decimal,
        confidence="High",
        reason="Parsed successfully.",
    )


def extract_amount_cells(
    text: str,
    *,
    expected_amounts: int | None = None,
    reject_years: bool = True,
    reject_small_note_refs: bool = True,
    context: str = "",
) -> list[AmountCell]:
    """Extract typed numeric cells from a row of financial-statement text."""
    cells: list[AmountCell] = []
    for match in AMOUNT_TOKEN_RE.finditer(str(text or "")):
        token = match.group(0)
        before = text[match.start() - 1] if match.start() > 0 else " "
        after = text[match.end()] if match.end() < len(text) else " "
        if token not in {"-", "–", "—"} and (before.isalpha() or after.isalpha()):
            continue
        cell = parse_amount_cell(
            token,
            reject_years=reject_years,
            reject_small_note_refs=reject_small_note_refs,
            context=context or text,
        )
        if cell.value is not None:
            cells.append(cell)
    if expected_amounts and len(cells) > expected_amounts:
        cells = cells[-expected_amounts:]
    return cells


def _normalise_token(raw: object) -> str:
    text = str(raw or "").strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_currency_and_units(token: str) -> str:
    value = token.strip()
    value = re.sub(r"^(?:NGN|N|₦|US\$|USD|GBP|EUR)\s*", "", value, flags=re.I)
    # Strip an explicit scale/units suffix such as "N'000" or a standalone
    # "000s" token. The bare "000" form is deliberately NOT matched here: the
    # trailing group of any grouped thousand (e.g. "5,000", "1,000,000") ends in
    # "000", and stripping it would silently truncate real amounts to a fraction
    # of their value. Require either a currency marker or the literal "s".
    value = re.sub(r"\s*(?:N'?000|₦'?000|NGN'?000|000s)$", "", value, flags=re.I)
    return value.strip()


def _normalize_numeric_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\s*[,]+\s*", ",", text)
    text = re.sub(r"(?<=\d)[\s.](?=\d{3}(?:\D|$))", "", text)
    text = text.replace(",", "")
    text = text.replace(" ", "")
    return text


def _looks_like_year(value: str, context: str = "") -> bool:
    return bool(re.fullmatch(r"20\d{2}", value))


def _looks_like_note_reference(value: str) -> bool:
    try:
        number = int(Decimal(value))
    except Exception:
        return False
    return Decimal(value) == Decimal(number) and 1 <= number <= 99


def _looks_like_page_reference(value: str, context: str = "") -> bool:
    if not re.fullmatch(r"\d{1,3}", value):
        return False
    return bool(re.search(r"\bpage\s+" + re.escape(value) + r"\b", context, flags=re.I))


def _looks_like_identifier(value: str) -> bool:
    if re.search(r"[A-Za-z]{2,}/\d", value) or re.search(r"\d/[A-Za-z]{2,}", value):
        return True
    if re.search(r"\b[A-Z]{2,6}/\d", value, flags=re.I):
        return True
    return False
