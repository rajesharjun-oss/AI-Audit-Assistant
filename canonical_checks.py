from __future__ import annotations

import re
from decimal import Decimal

from canonical_extraction import extraction_audit_rows, extract_statement_facts, note_heading_map
from canonical_models import ReconciliationCheckResult, StatementFact
from models import Finding, PdfDocument

TOLERANCE = Decimal("1")

NOTE_HEADING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cash and cash equivalents": ("cash", "bank", "cash equivalent"),
    "bank overdraft": ("cash", "bank", "overdraft"),
    "property, plant and equipment": ("property", "plant", "equipment", "ppe", "fixed asset"),
    "intangible assets": ("intangible", "software", "amortisation", "amortization"),
    "investment property": ("investment property", "investment properties", "property"),
    "inventories": ("inventor", "stock"),
    "trade and other receivables": ("receivable", "debtor", "trade and other receivables", "loan and advance", "loans and advances"),
    "other financial assets": ("financial asset", "finacial asset", "investment", "treasury", "securities"),
    "current assets": ("asset",),
    "non-current assets": ("asset",),
    "total assets": ("asset",),
    "current liabilities": ("liabilit", "payable", "borrow", "loan"),
    "non-current liabilities": ("liabilit", "borrow", "loan"),
    "total liabilities": ("liabilit", "borrow", "loan", "payable"),
    "trade and other payables": ("payable", "creditor", "trade and other payables"),
    "current tax payable": ("current tax", "tax payable", "tax"),
    "share capital": ("share capital", "ordinary share", "issued capital"),
    "deposit for shares": ("deposit for shares", "share deposit", "deposit"),
    "retained earnings": ("retained", "earning", "loss", "reserve"),
    "opening equity": ("equity", "reserve", "accumulated fund", "retained"),
    "closing equity": ("equity", "reserve", "accumulated fund", "retained"),
    "dividends": ("dividend",),
    "issue of shares": ("share", "capital"),
    "investment income": ("investment income", "finance income", "interest income"),
    "other income": ("other income", "other operating income", "miscellaneous income"),
    "other operating losses": ("other operating losses", "other operating gains", "other non-operating losses", "other non-operating gains", "fair value", "foreign exchange", "disposal"),
    "revenue": ("revenue", "rental income", "operating income", "turnover", "income from property", "other operating income", "other income", "interest income"),
    "project costs": ("project cost", "cost", "direct cost", "cost of sales", "interest expense"),
    "gross profit": ("gross profit", "net interest", "margin"),
    "operating profit": ("operating profit", "operating loss"),
    "finance costs": ("finance cost", "interest expense", "borrowing cost"),
    "profit before tax": ("profit before tax", "profit before taxation", "loss before tax", "loss before taxation"),
    "taxation": ("tax", "taxation", "income tax"),
    "profit after tax": ("profit after", "loss after", "profit for the year", "loss for the year", "result for the year"),
    "total comprehensive income": ("comprehensive", "profit", "loss"),
}


def run_canonical_checks(document: PdfDocument, facts: list[StatementFact] | None = None) -> tuple[list[Finding], list[ReconciliationCheckResult], list[dict[str, object]]]:
    facts = facts if facts is not None else extract_statement_facts(document)
    results: list[ReconciliationCheckResult] = []
    results.extend(check_statement_of_financial_position(facts))
    results.extend(check_profit_or_loss(facts))
    results.extend(check_changes_in_equity(facts))
    results.extend(check_cash_flow(facts))
    results.extend(check_note_references(document, facts))
    findings = [_finding_from_result(result) for result in results if result.status == "Fail"]
    return findings, results, extraction_audit_rows(document, facts)


def check_statement_of_financial_position(facts: list[StatementFact]) -> list[ReconciliationCheckResult]:
    sfp_facts = [fact for fact in facts if "financial position" in fact.statement.lower()]
    pairs = sorted({(fact.entity, fact.year) for fact in sfp_facts})
    results: list[ReconciliationCheckResult] = []
    for entity, year in pairs:
        non_current = _one(sfp_facts, "non-current assets", entity, year)
        current = _one(sfp_facts, "current assets", entity, year)
        total_assets = _one(sfp_facts, "total assets", entity, year)
        results.append(_equation_result("SFP total assets cast", "Statement of financial position", entity, year, [non_current, current], total_assets, "non-current assets + current assets", "Casting", "High"))
        non_current_liab = _one(sfp_facts, "non-current liabilities", entity, year)
        current_liab = _one(sfp_facts, "current liabilities", entity, year)
        total_liab = _one(sfp_facts, "total liabilities", entity, year)
        if non_current_liab or current_liab:
            results.append(_equation_result("SFP total liabilities cast", "Statement of financial position", entity, year, [non_current_liab, current_liab], total_liab, "non-current liabilities + current liabilities", "Casting", "High"))
        equity = _one(sfp_facts, "total equity", entity, year)
        total_equity_liab = _one(sfp_facts, "total equity and liabilities", entity, year)
        results.append(_equation_result("SFP equity plus liabilities cast", "Statement of financial position", entity, year, [equity, total_liab], total_equity_liab, "total equity + total liabilities", "Casting", "High"))
        results.append(_comparison_result("SFP total assets equals total equity and liabilities", "Statement of financial position", entity, year, total_assets, total_equity_liab, "total assets = total equity and liabilities", "Cross-casting", "High"))
    return [result for result in results if result.status != "Not tested"]


def check_profit_or_loss(facts: list[StatementFact]) -> list[ReconciliationCheckResult]:
    pl_facts = [fact for fact in facts if "profit or loss" in fact.statement.lower() or "comprehensive income" in fact.statement.lower()]
    pairs = sorted({(fact.entity, fact.year) for fact in pl_facts})
    results: list[ReconciliationCheckResult] = []
    for entity, year in pairs:
        revenue = _one(pl_facts, "revenue", entity, year)
        project_costs = _one(pl_facts, "project costs", entity, year)
        gross_profit = _one(pl_facts, "gross profit", entity, year)
        if revenue and project_costs and gross_profit:
            results.append(_equation_result("P&L gross profit cast", "Statement of profit or loss and OCI", entity, year, [revenue, project_costs], gross_profit, "revenue + project costs/direct costs", "Casting", "High"))
        pbt = _one(pl_facts, "profit before tax", entity, year)
        tax = _one(pl_facts, "taxation", entity, year)
        pat = _one(pl_facts, "profit after tax", entity, year)
        results.append(_equation_result("P&L after-tax result cast", "Statement of profit or loss and OCI", entity, year, [pbt, tax], pat, "profit/loss before tax + taxation", "Casting", "High"))
    return [result for result in results if result.status != "Not tested"]


def check_changes_in_equity(facts: list[StatementFact]) -> list[ReconciliationCheckResult]:
    equity_facts = [fact for fact in facts if "changes in equity" in fact.statement.lower() or "accumulated fund" in fact.statement.lower()]
    if not equity_facts:
        return []
    pl_facts = [fact for fact in facts if "profit or loss" in fact.statement.lower() or "comprehensive income" in fact.statement.lower()]
    pairs = sorted({(fact.entity, fact.year) for fact in equity_facts})
    results: list[ReconciliationCheckResult] = []
    for entity, year in pairs:
        opening = _one(equity_facts, "opening equity", entity, year)
        closing = _one(equity_facts, "closing equity", entity, year) or _one(equity_facts, "total equity", entity, year)
        result_for_year = _one(equity_facts, "profit after tax", entity, year) or _one(pl_facts, "profit after tax", entity, year)
        oci = _one(equity_facts, "other comprehensive income", entity, year) or _one(equity_facts, "total comprehensive income", entity, year)
        dividends = _one(equity_facts, "dividends", entity, year)
        share_issue = _one(equity_facts, "issue of shares", entity, year)
        movements = [fact for fact in (result_for_year, oci, dividends, share_issue) if fact is not None]
        if not opening or not closing or not movements:
            continue
        expected = opening.amount + sum((fact.amount for fact in movements), Decimal("0"))
        labels = ["opening equity", *(fact.canonical_line_item for fact in movements)]
        results.append(
            _calculated_result(
                "Changes in equity closing balance cast",
                "Statement of changes in equity",
                entity,
                year,
                closing,
                expected,
                " + ".join(labels),
                [opening, *movements, closing],
                "Equity movement",
                "Medium",
            )
        )
    return results


def check_cash_flow(facts: list[StatementFact]) -> list[ReconciliationCheckResult]:
    cf_facts = [fact for fact in facts if "cash flow" in fact.statement.lower()]
    sfp_facts = [fact for fact in facts if "financial position" in fact.statement.lower()]
    pairs = sorted({(fact.entity, fact.year) for fact in cf_facts})
    results: list[ReconciliationCheckResult] = []
    for entity, year in pairs:
        operating = _one(cf_facts, "net cash from operating activities", entity, year)
        investing = _one(cf_facts, "net cash from investing activities", entity, year)
        financing = _one(cf_facts, "net cash from financing activities", entity, year)
        movement = _one(cf_facts, "net movement in cash", entity, year)
        results.append(_equation_result("Cash flow net movement cast", "Statement of cash flows", entity, year, [operating, investing, financing], movement, "net operating cash flow + net investing cash flow + net financing cash flow", "Cash Flow", "High"))
        opening = _one(cf_facts, "cash at beginning", entity, year)
        fx = _one(cf_facts, "effect of exchange rate movement", entity, year)
        closing = _one(cf_facts, "cash and cash equivalents", entity, year)
        results.append(_equation_result("Cash flow opening-to-closing cash cast", "Statement of cash flows", entity, year, [opening, movement, fx], closing, "opening cash + net movement + effect of exchange rate movement", "Cash Flow", "High"))
        sfp_cash = _one(sfp_facts, "cash and cash equivalents", entity, year)
        if closing and sfp_cash:
            results.append(_comparison_result("Cash flow closing cash agrees to SFP cash", "Statement of cash flows", entity, year, closing, sfp_cash, "closing cash per cash flow = SFP cash and cash equivalents", "Cash Flow", "Medium"))
    return [result for result in results if result.status != "Not tested"]


def check_note_references(document: PdfDocument, facts: list[StatementFact]) -> list[ReconciliationCheckResult]:
    headings = note_heading_map(document)
    primary = [fact for fact in facts if any(marker in fact.statement.lower() for marker in ("financial position", "profit or loss", "comprehensive income", "cash flow"))]
    results: list[ReconciliationCheckResult] = []
    seen: set[tuple[int, str, str, str]] = set()
    for fact in primary:
        if not fact.note_ref:
            continue
        if _cash_flow_subtotal_without_note_detail(fact):
            continue
        key = (fact.source_page, fact.statement, fact.line_item, fact.note_ref)
        if key in seen:
            continue
        seen.add(key)
        ref = fact.note_ref.upper()
        parent_ref = "".join(ch for ch in ref if ch.isdigit())
        heading = headings.get(ref) or headings.get(parent_ref, "")
        if not heading:
            results.append(ReconciliationCheckResult("Face statement note reference exists", fact.statement, fact.entity, fact.year, "Fail", source_pages=f"Page {fact.source_page}", source_rows=fact.source_line, category="Note Cross-reference", priority="Medium", confidence="Medium", formula=f"Note {ref} should exist in notes heading map.", recommendation=f"Add Note {ref} or correct the note reference for '{fact.line_item}'."))
            continue
        if not _heading_compatible(fact.canonical_line_item, heading):
            results.append(ReconciliationCheckResult("Face statement note reference compatibility", fact.statement, fact.entity, fact.year, "Fail", source_pages=f"Page {fact.source_page}", source_rows=f"{fact.source_line} | Note {ref} heading detected as '{heading}'.", category="Note Cross-reference", priority="Medium", confidence="High", formula=f"'{fact.line_item}' should point to a compatible note heading.", recommendation=f"Review Note {ref}. The heading '{heading}' does not appear compatible with '{fact.line_item}'."))
    return results


def _one(facts: list[StatementFact], canonical: str, entity: str, year: int) -> StatementFact | None:
    exact = [fact for fact in facts if fact.entity == entity and fact.year == year and fact.canonical_line_item == canonical]
    if canonical == "profit after tax":
        exact = [
            fact
            for fact in exact
            if not re.search(r"\b(total comprehensive|other comprehensive|oci)\b", fact.source_line, flags=re.I)
        ]
    if canonical == "profit before tax":
        exact = [
            fact
            for fact in exact
            if not re.search(r"\bstated after charging\b|\bafter charging\b", fact.source_line, flags=re.I)
        ]
    return exact[0] if exact else None


def _equation_result(name: str, statement: str, entity: str, year: int, components: list[StatementFact | None], reported: StatementFact | None, formula: str, category: str, priority: str) -> ReconciliationCheckResult:
    present = [component for component in components if component is not None]
    if reported is None or len(present) != len(components):
        return ReconciliationCheckResult(name, statement, entity, year, "Not tested", formula=formula, category=category, priority=priority, confidence="Low", recommendation="Required source rows were not all parsed with sufficient confidence.")
    expected = sum((component.amount for component in present), Decimal("0"))
    return _calculated_result(name, statement, entity, year, reported, expected, formula, [*present, reported], category, priority)


def _comparison_result(name: str, statement: str, entity: str, year: int, left: StatementFact | None, right: StatementFact | None, formula: str, category: str, priority: str) -> ReconciliationCheckResult:
    if left is None or right is None:
        return ReconciliationCheckResult(name, statement, entity, year, "Not tested", formula=formula, category=category, priority=priority, confidence="Low", recommendation="Required source rows were not all parsed with sufficient confidence.")
    return _calculated_result(name, statement, entity, year, left, right.amount, formula, [left, right], category, priority)


def _calculated_result(name: str, statement: str, entity: str, year: int, reported: StatementFact, expected: Decimal, formula: str, source_facts: list[StatementFact], category: str, priority: str) -> ReconciliationCheckResult:
    difference = reported.amount - expected
    status = "Pass" if abs(difference) <= TOLERANCE else "Fail"
    pages = sorted({fact.source_page for fact in source_facts})
    return ReconciliationCheckResult(name, statement, entity, year, status, reported_amount=reported.amount, expected_amount=expected, difference=difference, formula=formula, source_pages="Pages " + ", ".join(str(page) for page in pages), source_rows=" | ".join(dict.fromkeys(fact.source_line for fact in source_facts))[:1200], category=category, priority=priority if status == "Fail" else "Low", confidence="High" if all(fact.confidence == "High" for fact in source_facts) else "Medium", recommendation="No action required." if status == "Pass" else "Recalculate the affected line item and update the primary statement or supporting note.")


def _finding_from_result(result: ReconciliationCheckResult) -> Finding:
    issue = f"{result.check_name} failed for {result.entity} {result.year}."
    evidence = f"Reported {result.reported_amount:,.0f} vs expected {result.expected_amount:,.0f}; difference {result.difference:,.0f}. Formula: {result.formula}. {result.source_rows}" if result.reported_amount is not None and result.expected_amount is not None and result.difference is not None else f"Formula: {result.formula}. {result.source_rows}"
    return Finding(result.category, result.priority, result.source_pages or result.statement, issue, evidence, result.recommendation, metadata={"check_type": result.check_name, "statement": result.statement, "entity": str(result.entity), "year": str(result.year), "reported_amount": str(result.reported_amount or ""), "expected_amount": str(result.expected_amount or ""), "difference": str(result.difference or ""), "formula": result.formula, "match_confidence": result.confidence})


def _heading_compatible(canonical_line_item: str, heading: str) -> bool:
    keywords = NOTE_HEADING_KEYWORDS.get(canonical_line_item)
    if not keywords:
        return True
    heading_norm = re.sub(r"\s+", " ", heading.lower().replace("-", " ")).strip()
    line_norm = re.sub(r"\s+", " ", canonical_line_item.lower().replace("-", " ")).strip()
    if line_norm and (line_norm == heading_norm or line_norm in heading_norm or heading_norm in line_norm):
        return True
    if line_norm.startswith("movement in "):
        base = line_norm.removeprefix("movement in ").strip()
        if base and (base in heading_norm or heading_norm in base):
            return True
    return any(keyword.lower().replace("-", " ") in heading_norm for keyword in keywords)

def _cash_flow_subtotal_without_note_detail(fact: StatementFact) -> bool:
    if "cash flow" not in fact.statement.lower():
        return False
    label = fact.canonical_line_item or fact.line_item.lower()
    return label in {
        "net cash from operating activities",
        "net cash from investing activities",
        "net cash from financing activities",
        "net movement in cash",
        "cash at beginning",
        "effect of exchange rate movement",
    }
