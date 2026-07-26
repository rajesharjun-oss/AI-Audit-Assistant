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
    ("cash at beginning", ("cash at beginning", "cash at the beginning of the year", "cash and cash equivalents at the beginning of the year", "cash cash equivalents at beginning year")),
    ("net movement in cash", ("net movement in cash and cash equivalents", "total cash movement for the year", "net cash movement", "net increase in cash and cash equivalents", "net decrease in cash and cash equivalents")),
    ("cash and cash equivalents", ("cash and cash equivalent", "cash and cash equivalents", "total cash at end of the year", "cash at end", "cash cash equivalents at end year")),
    ("bank overdraft", ("bank overdraft", "overdraft")),
    ("other operating losses", ("losses gains on disposal", "losses on disposal", "loss on disposal", "gains on disposal", "gain on disposal", "losses on foreign exchange", "gains on foreign exchange", "other operating losses", "other non-operating losses", "other non-operating gains")),
    ("property, plant and equipment", ("property plant and equipment", "property, plant and equipment", "ppe")),
    ("intangible assets", ("intangible assets", "intangible asset", "software", "computer software")),
    ("investment property", ("investment property", "investment properties")),
    ("inventories", ("inventories", "inventory", "stock")),
    ("trade and other receivables", ("trade and other receivables", "trade receivables", "other receivables", "receivables")),
    ("other financial assets", ("other financial assets", "financial assets", "investment securities")),
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
    ("opening equity", ("balance at 1 january", "balance at beginning", "opening balance", "at 1 january", "as at 1 january")),
    ("closing equity", ("balance at 31 december", "balance at end", "closing balance", "at 31 december", "as at 31 december")),
    ("dividends", ("dividend", "dividends", "dividend paid", "dividends paid")),
    ("issue of shares", ("issue of shares", "shares issued", "share issue", "proceeds from issue of shares")),
    ("other comprehensive income", ("other comprehensive income", "other comprehensive loss")),
    ("investment income", ("investment income", "finance income")),
    ("other income", ("other income", "other operating income", "miscellaneous income")),
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
    ("effect of exchange rate movement", ("effect of exchange rate movement on cash balances", "exchange rate movement", "effect of foreign exchange", "loss on foreign exchange on cash and cash equivalents", "profit on foreign exchange on cash and cash equivalents", "foreign exchange on cash and cash equivalents", "exchange differences on cash and cash equivalents")),
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
NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:,\d{3})+|\(?-?\d+(?:\.\d+)?\)?")
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


def document_section_map(document: PdfDocument) -> list[dict[str, object]]:
    """Classify every page into a reusable financial-statement section map."""
    notes_start = _notes_section_start_page(document)
    primary_pages = {page.number for page in document.pages if classify_statement_page(page)}
    rows: list[dict[str, object]] = []
    notes_ended = False
    for page in document.pages:
        statement = classify_statement_page(page)
        normalized = _normalise_words(page.text)
        snippet = re.sub(r"\s+", " ", page.text or "").strip()[:260]
        section = "Unknown / narrative"
        confidence = "Low"
        reason = "No strong section heading detected."
        if any(marker in normalized for marker in ("contents", "table contents")) and page.number <= 5:
            section = "Contents"
            confidence = "High"
            reason = "Contents heading appears in front matter."
        elif statement:
            section = "Primary statement" if statement not in {"Value added statement", "Five-year financial summary"} else "Supplementary schedule"
            confidence = "High"
            reason = f"Detected {statement} heading."
        elif notes_start and page.number >= notes_start and not notes_ended:
            section = "Notes to the financial statements"
            confidence = "Medium"
            reason = f"Page is at or after detected notes start page {notes_start}."
            if any(_line_ends_notes_section(line) for line in page.text.splitlines()):
                notes_ended = True
        elif page.number < (notes_start or 10**9) and any(marker in normalized for marker in ("directors report", "corporate information", "independent auditor", "audit report")):
            section = "Front matter / reports"
            confidence = "High"
            reason = "Directors, corporate information, or auditor report wording detected before notes."
        elif any(marker in normalized for marker in ("value added statement", "five year financial summary", "5 year financial summary")):
            section = "Supplementary schedule"
            confidence = "High"
            reason = "Value-added or five-year summary wording detected."
        elif page.number in primary_pages:
            section = "Primary statement"
            confidence = "Medium"
            reason = "Page previously classified as a primary statement."
        rows.append(
            {
                "Page": page.number,
                "Section": section,
                "Statement": statement,
                "Confidence": confidence,
                "Reason": reason,
                "Text snippet": snippet,
            }
        )
    return rows


def table_classification_rows(document: PdfDocument) -> list[dict[str, object]]:
    """Classify extracted table candidates before arithmetic checks are applied."""
    notes_start = _notes_section_start_page(document)
    rows: list[dict[str, object]] = []
    for page in document.pages:
        statement = classify_statement_page(page)
        for index, table in enumerate(page.tables, start=1):
            row_count = len(table or [])
            col_count = max((len(row) for row in table), default=0)
            flattened = " ".join(str(cell or "") for row in table for cell in row)
            combined_norm = _normalise_words(f"{page.text[:1000]} {flattened}")
            amount_cells = extract_amount_cells(flattened, reject_years=True, reject_small_note_refs=True, context=flattened)
            merged_numeric_cells = sum(
                1
                for row in table
                for cell in row[1:]
                if len(NUMBER_RE.findall(str(cell or ""))) > 1
            )
            table_type = "Low-confidence / non-standard table"
            confidence = "Low"
            action = "Skip arithmetic"
            reason = "Table does not have enough reliable amount cells for deterministic arithmetic."
            if statement and statement not in {"Value added statement", "Five-year financial summary"} and amount_cells:
                table_type = "Primary statement table"
                confidence = "High" if merged_numeric_cells == 0 and row_count >= 4 else "Medium"
                action = "Use for statement-specific checks only"
                reason = f"Page classified as {statement}; table has {len(amount_cells)} amount cell(s)."
            elif any(marker in combined_norm for marker in ("value added statement", "five year financial summary", "5 year financial summary")):
                table_type = "Supplementary schedule"
                confidence = "Medium"
                action = "Exclude from normal casting"
                reason = "Value-added or five-year summary tables have different presentation rules."
            elif any(marker in combined_norm for marker in ("credit risk", "liquidity risk", "maturity", "ageing", "expected credit loss", "ecl")):
                table_type = "Risk / maturity disclosure table"
                confidence = "Medium"
                action = "Exclude from normal casting unless a specific disclosure check applies"
                reason = "Risk disclosure tables contain buckets and maturity bands, not simple subtotals."
            elif notes_start and page.number >= notes_start and amount_cells:
                table_type = "Note amount table"
                confidence = "Medium" if merged_numeric_cells == 0 else "Low"
                action = "Use cautiously for note agreement"
                reason = f"Table appears inside notes section from page {notes_start}."
            elif amount_cells:
                table_type = "Narrative amount table"
                confidence = "Low"
                action = "Manual review"
                reason = "Amounts detected outside primary statements/notes; avoid automatic casting."
            rows.append(
                {
                    "Page": page.number,
                    "Table index": index,
                    "Table type": table_type,
                    "Statement": statement,
                    "Confidence": confidence,
                    "Recommended action": action,
                    "Reason": reason,
                    "Rows": row_count,
                    "Columns": col_count,
                    "Amount cells": len(amount_cells),
                    "Merged numeric cells": merged_numeric_cells,
                }
            )
    if rows:
        return rows
    return [{"Page": "None", "Table type": "No extracted table candidates", "Recommended action": "Use line-based extraction", "Reason": "No PDF table grids were available."}]


def note_heading_map(document: PdfDocument) -> dict[str, str]:
    headings: dict[str, str] = {}
    start_page = _notes_section_start_page(document)
    if start_page is None:
        return headings
    for page in document.pages:
        if page.number < start_page:
            continue
        if page.number == start_page and "1" not in headings:
            implicit_note_1 = _implicit_note_1_heading(page.text)
            if implicit_note_1:
                headings["1"] = implicit_note_1
        for raw_line in page.text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if _line_ends_notes_section(line):
                return headings
            if _line_is_repeated_notes_header(line):
                continue
            if re.match(r"^\s*\d{1,2}\.\d{1,2}\b", line):
                continue
            match = re.match(r"^(?:note\s+)?(\d{1,2}[A-Za-z]?)(?:[.)]\s*|\s+)(.{3,100})$", line, flags=re.I)
            if not match:
                continue
            ref, title = match.groups()
            title = _clean_note_heading_title(title)
            if not _valid_note_ref(ref) or not _title_looks_like_note_heading(title):
                continue
            if _looks_like_collapsed_policy_subsection(ref, title, headings, page.number, start_page):
                continue
            if ref.upper() not in headings:
                headings[ref.upper()] = title[:100]
    return headings


def _notes_section_start_page(document: PdfDocument) -> int | None:
    primary_statement_names = {
        "Statement of financial position",
        "Statement of profit or loss and other comprehensive income",
        "Statement of changes in equity",
        "Statement of cash flows",
    }
    primary_pages = [page.number for page in document.pages if classify_statement_page(page) in primary_statement_names]
    first_allowed = (max(primary_pages) + 1) if primary_pages else 1
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
        has_policy_heading = bool(
            re.search(r"(?im)^\s*(?:note\s+)?1\s*(?:[.)]|:)?\s*(?:reporting entity|significant accounting polic|material accounting polic|basis of prep\w*)\b", page.text)
            or re.search(r"(?im)^\s*1\.1\s+(?:basis of prep\w*|material accounting polic|significant accounting polic)\b", page.text)
        )
        if has_notes_heading and has_policy_heading:
            return page.number
        if has_notes_heading and best_fallback is None:
            best_fallback = page.number
        elif has_policy_heading and best_fallback is None and page.number >= first_allowed:
            best_fallback = page.number
    return best_fallback


def _implicit_note_1_heading(text: str) -> str:
    if re.search(r"(?im)^\s*1\s*(?:[.)]|:)?\s*reporting entity\b", text):
        return "Reporting entity"
    if re.search(r"(?im)^\s*1\s*(?:[.)]|:)?\s*material accounting polic", text):
        return "Material accounting policies"
    if re.search(r"(?im)^\s*1\s*(?:[.)]|:)?\s*significant accounting polic", text):
        return "Significant accounting policies"
    normalized = _normalise_words(text)
    if "material accounting policies" in normalized:
        return "Material accounting policies"
    if "significant accounting policies" in normalized:
        return "Significant accounting policies"
    return ""


def _clean_note_heading_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip(" -:.")
    cleaned = re.sub(r"\s*\((?:continued|cont\.?)\)\s*$", "", cleaned, flags=re.I).strip(" -:.")
    return cleaned


def _looks_like_collapsed_policy_subsection(ref: str, title: str, headings: dict[str, str], page_number: int, start_page: int) -> bool:
    numeric = "".join(ch for ch in ref if ch.isdigit())
    if not numeric or "2" in headings:
        return False
    seen_real_later_note = any(
        existing_ref.isdigit() and 2 <= int(existing_ref) <= 9
        for existing_ref in headings
    )
    if seen_real_later_note:
        return False
    if page_number - start_page > 10:
        return False
    if not numeric.startswith("1") or len(numeric) != 2:
        return False
    title_norm = _normalise_words(title)
    policy_topics = (
        "property plant equipment",
        "intangible assets",
        "financial instruments",
        "financial assets",
        "financial liabilities",
        "revenue",
        "leases",
        "inventories",
        "tax",
        "impairment",
        "foreign currency",
        "employee benefits",
        "cash cash equivalents",
        "trade other receivables",
        "trade other payables",
    )
    return any(topic in title_norm for topic in policy_topics)


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
        "for details",
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
    lines = [line.strip() for line in page.text.splitlines()[:35] if line.strip()]
    header = "\n".join(lines).lower()
    if page.number <= 5 and ("contents" in header or "table of contents" in header) and (
        "...." in header or header.count("statement of") >= 2
    ):
        return ""
    notes_header_values = {
        "notes financial statements",
        "notes financial statement",
        "notes accounts",
        "notes forming part financial statements",
    }
    if any(_normalise_words(line) in notes_header_values for line in lines[:8]):
        return ""
    supplementary_patterns = {
        "Value added statement": ("value added statement", "statement of value added"),
        "Five-year financial summary": ("five year financial summary", "five-year financial summary", "5 year financial summary"),
    }
    for statement, aliases in supplementary_patterns.items():
        if any(_line_matches_statement_heading(line, aliases) for line in lines):
            return statement
    for statement, aliases in STATEMENT_PATTERNS:
        if statement in supplementary_patterns:
            continue
        if any(_line_matches_statement_heading(line, aliases) for line in lines):
            return statement
    return ""


def _line_matches_statement_heading(line: str, aliases: tuple[str, ...]) -> bool:
    normalized = _normalise_words(line)
    if not normalized:
        return False
    if normalized.startswith(("we have audited", "which comprise", "comprise", "including", "notes")):
        return False
    allowed_heading_prefixes = {"consolidated", "separate", "company", "group", "parent"}
    for alias in aliases:
        alias_norm = _normalise_words(alias)
        if normalized == alias_norm or normalized.startswith(f"{alias_norm} "):
            return True
        index = normalized.find(alias_norm)
        if index > 0:
            prefix = normalized[:index].strip()
            if prefix and all(word in allowed_heading_prefixes for word in prefix.split()):
                return True
    return False
def detect_statement_columns(text: str, statement: str = "") -> list[StatementColumn]:
    header_lines = [line for line in text.splitlines()[:30] if len(YEAR_RE.findall(line)) >= 2]
    if "changes in equity" in statement.lower() and not header_lines:
        # Single-year statement of changes in equity: no comparative-year header
        # row (only "for the year ended 31 December 20XX"). Map each movement row
        # to that single reporting year using the rightmost (total-equity) amount
        # so the roll-forward check can still run. Multi-year statements have a
        # "20XX 20XX" header line and take the general path below.
        return _single_year_columns(text)
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


def _single_year_columns(text: str) -> list[StatementColumn]:
    """Single reporting-year column for a statement with no comparative header."""
    years = [int(value) for value in YEAR_RE.findall(text)]
    if not years:
        return []
    header = " ".join(text.splitlines()[:18])
    return [StatementColumn(1, _default_entity(text), max(years), raw_header=header[:220])]


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
        trailing_row = _split_trailing_label_row(line, expected)
        if trailing_row:
            note_ref, label_part, amount_part = trailing_row
        else:
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


def _split_trailing_label_row(line: str, expected_amounts: int = 0) -> tuple[str, str, str] | None:
    """Handle rows extracted as note/amount columns first and the label last."""
    if expected_amounts <= 0:
        return None
    matches = list(AMOUNT_START_RE.finditer(line))
    if len(matches) < expected_amounts:
        return None
    amount_matches = matches[-expected_amounts:]
    trailing_label = line[amount_matches[-1].end() :].strip(" -:.;")
    if not trailing_label or not re.search(r"[A-Za-z]", trailing_label):
        return None
    prefix = line[: amount_matches[0].start()].strip()
    if prefix and not _valid_note_ref(prefix):
        return None
    # Avoid treating ordinary narrative lines with trailing text as statement rows.
    if len(trailing_label.split()) < 3:
        return None
    amount_part = line[amount_matches[0].start() : amount_matches[-1].end()].strip()
    return prefix.upper(), trailing_label, amount_part


def _split_note_ref(line: str, expected_amounts: int = 0) -> tuple[str, str, str]:
    explicit = NOTE_REF_RE.search(line)
    if explicit:
        return explicit.group(1).upper(), line[: explicit.start()].strip(), line[explicit.end() :].strip()
    match = re.match(r"^([A-Za-z][A-Za-z&/'() .,-]{2,}?)\s+(\d{1,2}[A-Za-z]?)\s+(.+)$", line)
    if match:
        label, ref, tail = match.groups()
        if _valid_note_ref(ref) and AMOUNT_START_RE.match(tail.strip()):
            tail_cells = extract_amount_cells(tail, reject_years=True, reject_small_note_refs=False, context=line)
            if tail_cells and (not expected_amounts or len(tail_cells) >= expected_amounts):
                return ref.upper(), label.strip(), tail.strip()
    return "", line, line


def clean_line_label(text: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(text or ""))
    value = re.sub(r"\b(?:note\(?s?\)?|notes?|n'?000|ngn'?000|draft)\b", " ", value, flags=re.I)
    value = re.sub(r"\b20\d{2}\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -:.;")
    amount_match = _first_amount_start_for_label(value)
    if amount_match:
        value = value[: amount_match.start()].strip(" -:.;")
    return value


def _first_amount_start_for_label(value: str) -> re.Match[str] | None:
    for match in AMOUNT_START_RE.finditer(value):
        following = value[match.end() :].lstrip().lower()
        if following.startswith(("january", "december")):
            continue
        return match
    return None


def canonical_line_item(label: str) -> str:
    normalized = _normalise_words(label)
    if not normalized:
        return ""
    if "total comprehensive" in normalized:
        return "total comprehensive income"
    if "other comprehensive" in normalized:
        return "other comprehensive income"
    if re.search(r"\b(profit|loss)\b.*\bafter\s+tax(?:ation)?\b", normalized):
        return "profit after tax"
    if re.search(r"\b(profit|loss)\b.*\bbefore\s+tax(?:ation)?\b", normalized):
        return "profit before tax"
    if any(term in normalized for term in ("net movement in cash", "net increase in cash", "net decrease in cash", "net cash movement", "total cash movement")):
        return "net movement in cash"
    if "cash" in normalized and any(term in normalized for term in ("beginning", "opening", "start")):
        return "cash at beginning"
    if "cash" in normalized and any(term in normalized for term in ("foreign exchange", "exchange difference", "exchange differences", "exchange rate movement")):
        return "effect of exchange rate movement"
    if "cash" in normalized and any(term in normalized for term in (" end", "ending", "closing")):
        return "cash and cash equivalents"
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
    if "cash equivalent" in alias_norm and any(term in label_norm for term in ("beginning", "opening", "start")):
        return False
    if "cash equivalent" in alias_norm and any(term in label_norm for term in ("net movement", "net increase", "net decrease", "cash movement")):
        return False
    if "cash equivalent" in alias_norm and any(term in label_norm for term in ("foreign exchange", "exchange difference", "exchange differences", "exchange rate movement")):
        return False
    if "cash at beginning" in alias_norm and any(term in label_norm for term in (" end", "closing")):
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
