from __future__ import annotations

import re
from difflib import SequenceMatcher

from amount_parser import extract_amount_cells
from canonical_models import CanonicalStatementTable, StatementColumn, StatementFact
from models import PdfDocument, PdfPage

STATEMENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Statement of financial position", ("statement of financial position", "statements of financial position", "balance sheet")),
    ("Statement of profit or loss and other comprehensive income", ("statement of profit or loss", "statements of profit or loss", "statement of comprehensive income")),
    ("Statement of changes in equity", ("statement of changes in equity", "statements of changes in equity", "statement of changes in accumulated fund")),
    ("Statement of cash flows", ("statement of cash flows", "statements of cash flows", "statement of cash flow")),
    ("Value added statement", ("value added statement", "statement of value added")),
    ("Five-year financial summary", ("five year financial summary", "five-year financial summary", "5 year financial summary")),
)

LINE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cash and cash equivalents", ("cash and cash equivalent", "cash and cash equivalents", "total cash at end of the year", "cash at end")),
    ("cash at beginning", ("cash at beginning", "cash at the beginning of the year", "cash and cash equivalents at the beginning of the year")),
    ("bank overdraft", ("bank overdraft", "overdraft")),
    ("property, plant and equipment", ("property plant and equipment", "property, plant and equipment", "ppe")),
    ("current assets", ("current assets", "total current assets")),
    ("non-current assets", ("non current assets", "non-current assets", "total non current assets", "total non-current assets")),
    ("total assets", ("total assets",)),
    ("current liabilities", ("current liabilities", "total current liabilities")),
    ("non-current liabilities", ("non current liabilities", "non-current liabilities", "total non current liabilities", "total non-current liabilities")),
    ("total liabilities", ("total liabilities",)),
    ("total equity", ("total equity", "equity", "capital and reserves", "share capital and reserves", "shareholders equity")),
    ("total equity and liabilities", ("total equity and liabilities", "total liabilities and equity", "total equity and liability")),
    ("share capital", ("share capital", "issued share capital")),
    ("deposit for shares", ("deposit for shares",)),
    ("retained earnings", ("retained earnings", "retained losses", "retained income", "accumulated losses")),
    ("revenue", ("revenue", "interest income", "income", "turnover", "gross earnings")),
    ("project costs", ("project costs", "project cost", "cost of sales", "interest expense", "direct costs")),
    ("gross profit", ("gross profit", "net interest loss", "net interest income")),
    ("operating profit", ("operating profit", "operating loss", "profit before finance cost and taxation")),
    ("finance costs", ("finance costs", "finance cost")),
    ("profit before tax", ("profit before tax", "profit before taxation", "loss before tax", "loss before taxation")),
    ("taxation", ("taxation", "tax expense", "income tax expense", "tax credit")),
    ("profit after tax", ("profit after tax", "profit after taxation", "loss after tax", "loss after taxation", "profit for the year", "loss for the year")),
    ("total comprehensive income", ("total comprehensive income", "total comprehensive loss", "total comprehensive income for the year", "total comprehensive loss for the year")),
    ("net cash from operating activities", ("net cash from operating activities", "net cash generated from operating activities", "net cash used in operating activities", "net cash flows used in operating activities")),
    ("net cash from investing activities", ("net cash from investing activities", "net cash used in investing activities", "net cash generated from investing activities", "net cash flows used in investing activities")),
    ("net cash from financing activities", ("net cash from financing activities", "net cash used in financing activities", "net cash generated from financing activities", "net cash flows used in financing activities")),
    ("net movement in cash", ("net movement in cash and cash equivalents", "total cash movement for the year", "net cash movement", "net increase in cash and cash equivalents", "net decrease in cash and cash equivalents")),
    ("effect of exchange rate movement", ("effect of exchange rate movement on cash balances", "exchange rate movement", "effect of foreign exchange")),
    ("purchase of property, plant and equipment", ("purchase of property plant and equipment", "purchase of property, plant and equipment")),
)

SECTION_LINES = {
    "assets", "non-current assets", "non current assets", "current assets", "equity and liabilities", "equity",
    "liabilities", "non-current liabilities", "non current liabilities", "current liabilities", "adjustments for",
    "changes in working capital", "cash flows from operating activities", "cash flows from investing activities",
    "cash flows from financing activities", "group", "company", "other comprehensive income",
}

YEAR_RE = re.compile(r"\b20\d{2}\b")
NOTE_REF_RE = re.compile(r"\b(?:note\(?s?\)?|notes?)\s*(\d+[A-Za-z]?)\b", re.I)
AMOUNT_START_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+|\(?-?\d+(?:\.\d+)?\)?|(?<![A-Za-z])[-–—](?![A-Za-z])")


def extract_canonical_tables(document: PdfDocument) -> list[CanonicalStatementTable]:
    tables: list[CanonicalStatementTable] = []
    for page in document.pages:
        statement = classify_statement_page(page)
        if not statement:
            continue
        columns = detect_statement_columns(page.text, statement)
        if not columns:
            tables.append(CanonicalStatementTable(statement, page.number, [], [], "Low", "Could not detect year/entity columns."))
            continue
        facts = _facts_from_page(page, statement, columns)
        confidence = "High" if facts and all(col.year for col in columns) else "Medium" if facts else "Low"
        tables.append(CanonicalStatementTable(statement, page.number, columns, facts, confidence, f"Detected {len(facts)} fact(s)."))
    return tables


def extract_statement_facts(document: PdfDocument) -> list[StatementFact]:
    facts: list[StatementFact] = []
    for table in extract_canonical_tables(document):
        facts.extend(table.facts)
    return facts


def extraction_audit_rows(document: PdfDocument, facts: list[StatementFact] | None = None) -> list[dict[str, object]]:
    facts = facts if facts is not None else extract_statement_facts(document)
    rows = [
        {
            "Page": fact.source_page,
            "Statement / note": fact.statement,
            "Entity": fact.entity,
            "Year": fact.year,
            "Raw row": fact.source_line,
            "Line item": fact.line_item,
            "Canonical line item": fact.canonical_line_item,
            "Note reference": fact.note_ref,
            "Parsed amount": fact.amount,
            "Column index": fact.column_index,
            "Confidence": fact.confidence,
            "Reason": fact.reason,
        }
        for fact in facts
    ]
    if rows:
        return rows
    for page in document.pages:
        statement = classify_statement_page(page)
        if statement:
            rows.append({"Page": page.number, "Statement / note": statement, "Reason": "Statement page detected, but no canonical facts parsed."})
    return rows


def note_heading_map(document: PdfDocument) -> dict[str, str]:
    headings: dict[str, str] = {}
    start_page = _notes_section_start_page(document)
    if start_page is None:
        return headings
    for page in document.pages:
        if page.number < start_page:
            continue
        for raw_line in page.text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if _line_ends_notes_section(line):
                return headings
            if _line_is_repeated_notes_header(line):
                continue
            match = re.match(r"^(?:note\s+)?(\d{1,2}[A-Za-z]?)(?:[.)]\s*|\s+)(.{3,100})$", line, flags=re.I)
            if not match:
                continue
            ref, title = match.groups()
            title = re.sub(r"\s+", " ", title).strip(" -:.")
            if not _valid_note_ref(ref) or not _title_looks_like_note_heading(title):
                continue
            if ref.upper() not in headings:
                headings[ref.upper()] = title[:100]
    return headings


def _notes_section_start_page(document: PdfDocument) -> int | None:
    primary_pages = [page.number for page in document.pages if classify_statement_page(page)]
    first_allowed = min(primary_pages) if primary_pages else 1
    best_fallback: int | None = None
    for page in document.pages:
        if page.number < first_allowed:
            continue
        normalized = _normalise_words(page.text)
        has_notes_heading = (
            "notes financial statements" in normalized
            or "notes financial statement" in normalized
            or "notes accounts" in normalized
            or "notes forming part financial statements" in normalized
        )
        has_policy_heading = bool(re.search(r"(?im)^\s*1\s*[.)]?\s*significant accounting policies\b", page.text))
        if has_notes_heading and has_policy_heading:
            return page.number
        if has_notes_heading and best_fallback is None:
            best_fallback = page.number
        elif has_policy_heading and best_fallback is None and primary_pages and page.number > max(primary_pages):
            best_fallback = page.number
    return best_fallback


def _line_is_repeated_notes_header(line: str) -> bool:
    normalized = _normalise_words(line)
    return normalized in {
        "notes financial statements",
        "notes financial statement",
        "notes accounts",
        "notes forming part financial statements",
    } or "financial statements year ended" in normalized


def _line_ends_notes_section(line: str) -> bool:
    normalized = _normalise_words(line)
    return any(
        marker in normalized
        for marker in (
            "value added statement",
            "statement value added",
            "five year financial summary",
            "5 year financial summary",
            "detailed income statement",
        )
    )


def _title_looks_like_note_heading(title: str) -> bool:
    if not title or _line_looks_like_amount_row(title):
        return False
    normalized = _normalise_words(title)
    if not normalized or _line_is_repeated_notes_header(title) or _line_ends_notes_section(title):
        return False
    narrative_starts = (
        "this represents",
        "this relates",
        "the company",
        "the group",
        "these comprise",
        "which represents",
        "during year",
    )
    if any(normalized.startswith(start) for start in narrative_starts):
        return False
    if len(normalized.split()) > 12 and not any(keyword in normalized for keyword in ("accounting policies", "financial instruments", "risk management")):
        return False
    return True


def classify_statement_page(page: PdfPage) -> str:
    header = "\n".join(page.text.splitlines()[:35]).lower()
    for statement, aliases in STATEMENT_PATTERNS:
        if any(alias in header for alias in aliases):
            return statement
    return ""


def detect_statement_columns(text: str, statement: str = "") -> list[StatementColumn]:
    header_lines = [line for line in text.splitlines()[:30] if len(YEAR_RE.findall(line)) >= 2]
    context_header = " ".join(text.splitlines()[:18])
    header = " ".join(header_lines or text.splitlines()[:18])
    header_context = f"{context_header} {header}"
    years = [int(value) for value in YEAR_RE.findall(header)]
    if len(years) >= 4 and _has_group_company_header(header_context):
        years = years[-4:]
        entities = ["Group", "Group", "Company", "Company"]
    elif _has_group_company_header(header_context) and len(years) >= 2:
        years = years[-2:]
        entities = ["Group", "Group", "Company", "Company"]
        years = [years[0], years[1], years[0], years[1]]
    elif len(years) >= 5 and "five" in statement.lower():
        years = years[-5:]
        entities = [_summary_entity(text)] * len(years)
    elif len(years) >= 2:
        years = years[-2:]
        entities = [_default_entity(text)] * len(years)
    else:
        all_years = sorted({int(value) for value in YEAR_RE.findall(text)}, reverse=True)
        if len(all_years) < 2:
            return []
        years = all_years[:2]
        entities = [_default_entity(text)] * len(years)
    return [StatementColumn(index + 1, entities[index], year, raw_header=header[:220]) for index, year in enumerate(years)]


def _facts_from_page(page: PdfPage, statement: str, columns: list[StatementColumn]) -> list[StatementFact]:
    facts: list[StatementFact] = []
    expected = len(columns)
    stop_after_statement = False
    for raw_line in page.text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if _line_is_signature_or_footer(line):
            stop_after_statement = True
        if stop_after_statement or _line_is_header_or_section(line):
            continue
        note_ref, label_part, amount_part = _split_note_ref(line, expected)
        label = clean_line_label(label_part or line)
        if not label:
            continue
        cells = extract_amount_cells(amount_part if amount_part else line, expected_amounts=expected, reject_years=True, reject_small_note_refs=False, context=line)
        if not cells:
            continue
        mapped_columns = columns[-len(cells):]
        confidence = "High" if len(cells) == expected else "Low"
        reason = "Mapped visible amount cells to detected statement columns." if len(cells) == expected else f"Expected {expected} column(s), parsed {len(cells)}."
        canonical = canonical_line_item(label)
        for column, cell in zip(mapped_columns, cells):
            if cell.value is None:
                continue
            facts.append(
                StatementFact(
                    statement=statement,
                    entity=column.entity,
                    year=column.year,
                    line_item=label,
                    canonical_line_item=canonical,
                    amount=cell.value,
                    source_page=page.number,
                    source_line=line,
                    note_ref=note_ref,
                    column_index=column.index,
                    confidence=confidence,  # type: ignore[arg-type]
                    reason=reason,
                )
            )
    return facts


def _split_note_ref(line: str, expected_amounts: int = 0) -> tuple[str, str, str]:
    explicit = NOTE_REF_RE.search(line)
    if explicit:
        return explicit.group(1).upper(), line[: explicit.start()].strip(), line[explicit.end() :].strip()
    match = re.match(r"^([A-Za-z][A-Za-z&/'() .,-]{2,}?)\s+(\d{1,2}[A-Za-z]?)\s+(.+)$", line)
    if match:
        label, ref, tail = match.groups()
        if _valid_note_ref(ref) and AMOUNT_START_RE.search(tail[:12]):
            tail_cells = extract_amount_cells(tail, reject_years=True, reject_small_note_refs=False, context=line)
            if tail_cells and (not expected_amounts or len(tail_cells) >= expected_amounts):
                return ref.upper(), label.strip(), tail.strip()
    return "", line, line


def clean_line_label(text: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(text or ""))
    value = re.sub(r"\b(?:note\(?s?\)?|notes?|n'?000|ngn'?000|draft)\b", " ", value, flags=re.I)
    value = re.sub(r"\b20\d{2}\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -:.;")
    amount_match = AMOUNT_START_RE.search(value)
    if amount_match:
        value = value[: amount_match.start()].strip(" -:.;")
    return value


def canonical_line_item(label: str) -> str:
    normalized = _normalise_words(label)
    if not normalized:
        return ""
    for canonical, aliases in LINE_ALIASES:
        for alias in sorted(aliases, key=len, reverse=True):
            if _labels_match(normalized, _normalise_words(alias)):
                return canonical
    return normalized


def fact_index(facts: list[StatementFact]) -> dict[tuple[str, str, int, str], StatementFact]:
    index: dict[tuple[str, str, int, str], StatementFact] = {}
    for fact in facts:
        index.setdefault(fact.key, fact)
    return index


def facts_for(facts: list[StatementFact], *, statement_contains: str = "", canonical: str = "", entity: str = "", year: int | None = None) -> list[StatementFact]:
    rows = facts
    if statement_contains:
        rows = [fact for fact in rows if statement_contains.lower() in fact.statement.lower()]
    if canonical:
        rows = [fact for fact in rows if fact.canonical_line_item == canonical]
    if entity:
        rows = [fact for fact in rows if fact.entity == entity]
    if year is not None:
        rows = [fact for fact in rows if fact.year == year]
    return rows


def _labels_match(label_norm: str, alias_norm: str) -> bool:
    if not label_norm or not alias_norm:
        return False
    if alias_norm in {"equity", "total equity"} and "liabilities" in label_norm:
        return False
    if alias_norm in {"current assets", "current liabilities"} and "non current" in label_norm:
        return False
    if label_norm == alias_norm or label_norm.startswith(alias_norm) or alias_norm in label_norm:
        return True
    return SequenceMatcher(None, label_norm, alias_norm).ratio() >= 0.94


def _normalise_words(text: str) -> str:
    value = text.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    value = re.sub(r"\b(?:the|and|of|to|from|for)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _has_group_company_header(header: str) -> bool:
    lower = header.lower()
    return ("group" in lower and "company" in lower) or ("consolidated" in lower and "separate" in lower) or "consolidated and separate" in lower


def _default_entity(text: str) -> str:
    header = " ".join(text.splitlines()[:20]).lower()
    if "consolidated" in header and "separate" in header:
        return "Group"
    if "group" in header and "company" not in header:
        return "Group"
    return "Company"


def _summary_entity(text: str) -> str:
    header = " ".join(text.splitlines()[:12]).lower()
    return "Group" if "group" in header else "Company"


def _line_is_header_or_section(line: str) -> bool:
    lower = re.sub(r"\s+", " ", line.lower()).strip(" -:;")
    if not lower or lower in SECTION_LINES:
        return True
    if any(marker in lower for marker in ("financial statements for the year ended", "together with", "note(s)", "n'000", "frc/", "signed on", "approved by")):
        return True
    if any(alias in lower for _statement, aliases in STATEMENT_PATTERNS for alias in aliases):
        return True
    if lower in {"2025 2024", "2025 2024 2025 2024", "group company"}:
        return True
    return bool(re.fullmatch(r"[\d\s%'-]+", lower))


def _line_is_signature_or_footer(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in ("the annual report and financial statements", "were approved by", "signed on behalf", "managing director", "chief financial officer", "company secretary"))


def _line_looks_like_amount_row(text: str) -> bool:
    return len(extract_amount_cells(text, reject_years=True, reject_small_note_refs=False, context=text)) >= 2


def _valid_note_ref(ref: str) -> bool:
    match = re.fullmatch(r"(\d{1,2})([A-Za-z]?)", str(ref or "").strip())
    return bool(match and 1 <= int(match.group(1)) <= 99)
