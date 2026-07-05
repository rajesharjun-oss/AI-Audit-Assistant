from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

YEAR_RE = re.compile(r"\b20\d{2}\b")
_AMOUNT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?|-",
    re.I,
)
_SCALE_RE = re.compile(
    r"(?:\bN\s*['’`]?\s*000\b|₦\s*['’`]?\s*000\b|\bNGN\s*['’`]?\s*000\b|\bUSD\s*['’`]?\s*000\b|\$\s*['’`]?\s*000\b|\b000s\b|\bthousands\b)",
    re.I,
)
_PERCENT_RE = re.compile(r"%\s*$")


@dataclass(frozen=True)
class AmountToken:
    raw: str
    value: Decimal | None
    start: int
    end: int
    is_blank: bool = False
    is_zero_dash: bool = False
    is_percentage: bool = False
    is_year: bool = False
    is_note_like: bool = False
    confidence: str = "High"
    reason: str = ""


def detect_scale(text: str) -> int:
    """Return presentation scale for amount labels such as N'000.

    The parser stores values exactly as presented in the financial statements.
    This scale is exposed for downstream exports that want to disclose that
    parsed amounts are in thousands/millions rather than units.
    """
    if not text:
        return 1
    if _SCALE_RE.search(text):
        return 1000
    if re.search(r"\b(millions?|mn)\b", text, flags=re.I):
        return 1_000_000
    return 1


def is_year_token(raw: str) -> bool:
    return bool(YEAR_RE.fullmatch(str(raw or "").strip("() ")))


def is_note_like_token(raw: str) -> bool:
    try:
        value = Decimal(str(raw or "").strip().strip("()").replace(",", ""))
    except InvalidOperation:
        return False
    return value == value.to_integral_value() and Decimal("1") <= value <= Decimal("99")


def parse_amount(raw: object, *, dash_is_zero: bool = True) -> AmountToken:
    text = str(raw or "").strip()
    if not text:
        return AmountToken("", None, 0, 0, is_blank=True, confidence="Low", reason="blank")
    if text in {"-", "–", "—", "--"}:
        return AmountToken(text, Decimal("0") if dash_is_zero else None, 0, len(text), is_blank=not dash_is_zero, is_zero_dash=dash_is_zero, confidence="High" if dash_is_zero else "Medium", reason="dash")

    match = _AMOUNT_TOKEN_RE.search(text)
    if not match or match.group(0) in {"-", "–", "—"}:
        return AmountToken(text, None, 0, len(text), is_blank=True, confidence="Low", reason="no amount token")

    token = match.group(0).strip()
    is_percentage = bool(_PERCENT_RE.search(token))
    negative = token.startswith("(") and token.endswith(")")
    cleaned = token.replace("%", "").strip("()").replace(",", "")
    cleaned = re.sub(r"(?<=\d)\s+(?=\d{3}(?:\D|$))", "", cleaned)
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return AmountToken(token, None, match.start(), match.end(), confidence="Low", reason="invalid decimal")
    if negative:
        value = -value
    return AmountToken(token, value, match.start(), match.end(), is_percentage=is_percentage, is_year=is_year_token(token), is_note_like=is_note_like_token(token) and not is_percentage)


def amount_tokens(text: str, *, include_dashes: bool = True) -> list[AmountToken]:
    tokens: list[AmountToken] = []
    for match in _AMOUNT_TOKEN_RE.finditer(str(text or "")):
        raw = match.group(0).strip()
        if raw in {"-", "–", "—"} and not include_dashes:
            continue
        parsed = parse_amount(raw)
        tokens.append(
            AmountToken(
                raw=raw,
                value=parsed.value,
                start=match.start(),
                end=match.end(),
                is_blank=parsed.is_blank,
                is_zero_dash=parsed.is_zero_dash,
                is_percentage=parsed.is_percentage,
                is_year=parsed.is_year,
                is_note_like=parsed.is_note_like,
                confidence=parsed.confidence,
                reason=parsed.reason,
            )
        )
    return tokens


def filter_financial_amount_tokens(
    tokens: Iterable[AmountToken],
    *,
    expected_count: int | None = None,
    drop_note_like_prefix: bool = True,
    drop_years: bool = True,
) -> tuple[str | None, list[AmountToken], list[str]]:
    """Remove likely note/year tokens from a row's numeric tokens.

    Returns ``(note_ref, amount_tokens, reasons)``.  It does not require a fixed
    column count, but when ``expected_count`` is supplied it uses it to avoid
    keeping note numbers or statement years as amounts.
    """
    work = list(tokens)
    reasons: list[str] = []
    note_ref: str | None = None

    if drop_years:
        before = len(work)
        # Drop 20xx tokens when enough non-year values remain for the expected amount columns.
        non_year = [token for token in work if not token.is_year]
        if expected_count is None or len(non_year) >= min(expected_count, len(work)):
            work = non_year
            if len(work) != before:
                reasons.append("dropped year tokens")

    if drop_note_like_prefix and work:
        first = work[0]
        remaining = work[1:]
        if first.is_note_like and remaining:
            if expected_count is None or len(remaining) >= min(expected_count, len(remaining)):
                note_ref = first.raw.strip().upper()
                work = remaining
                reasons.append(f"dropped note-like prefix {note_ref}")

    if expected_count is not None and len(work) > expected_count:
        # Keep the trailing amount columns; row labels often contain dates or note numbers before them.
        dropped = work[:-expected_count]
        work = work[-expected_count:]
        if dropped:
            reasons.append("kept trailing expected amount columns")

    return note_ref, [token for token in work if token.value is not None], reasons


def decimal_values(tokens: Iterable[AmountToken]) -> list[Decimal]:
    return [token.value for token in tokens if token.value is not None]
