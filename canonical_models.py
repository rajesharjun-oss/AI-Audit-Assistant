from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

CheckStatus = Literal["Pass", "Fail", "Not tested", "Manual review required"]
Confidence = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class StatementColumn:
    index: int
    entity: str
    year: int
    raw_header: str = ""


@dataclass(frozen=True)
class StatementFact:
    statement: str
    entity: str
    year: int
    line_item: str
    canonical_line_item: str
    amount: Decimal
    source_page: int
    source_line: str
    note_ref: str = ""
    column_index: int = 0
    confidence: Confidence = "Medium"
    reason: str = ""

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.statement, self.entity, self.year, self.canonical_line_item)


@dataclass
class CanonicalStatementTable:
    statement: str
    source_page: int
    columns: list[StatementColumn] = field(default_factory=list)
    facts: list[StatementFact] = field(default_factory=list)
    confidence: Confidence = "Medium"
    reason: str = ""


@dataclass
class ReconciliationCheckResult:
    check_name: str
    statement: str
    entity: str
    year: int | str
    status: CheckStatus
    reported_amount: Decimal | None = None
    expected_amount: Decimal | None = None
    difference: Decimal | None = None
    formula: str = ""
    source_pages: str = ""
    source_rows: str = ""
    category: str = "Casting"
    priority: str = "Medium"
    confidence: Confidence = "Medium"
    recommendation: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "Check": self.check_name,
            "Statement": self.statement,
            "Entity": self.entity,
            "Year": self.year,
            "Status": self.status,
            "Reported amount": _decimal_to_display(self.reported_amount),
            "Expected amount": _decimal_to_display(self.expected_amount),
            "Difference": _decimal_to_display(self.difference),
            "Formula": self.formula,
            "Source pages": self.source_pages,
            "Source rows": self.source_rows,
            "Category": self.category,
            "Priority": self.priority,
            "Confidence": self.confidence,
            "Recommendation": self.recommendation,
        }


def _decimal_to_display(value: Decimal | None) -> str:
    if value is None:
        return ""
    return f"{value:,.0f}" if value == value.to_integral_value() else f"{value:,}"
