from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?")
UNREADABLE_RE = re.compile(r"(#{3,}|�|□|_{3,}|\*{3,})")


@dataclass
class PdfPage:
    number: int
    text: str
    tables: list[list[list[str]]]


@dataclass
class PdfDocument:
    pages: list[PdfPage]
    ocr_used: bool = False
    ocr_pages: int = 0
    ocr_tables: int = 0
    ocr_error: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def text_pages(self) -> int:
        return sum(1 for page in self.pages if page.text.strip())

    @property
    def text_chars(self) -> int:
        return sum(len(page.text.strip()) for page in self.pages)

    @property
    def extraction_coverage(self) -> float:
        return self.text_pages / len(self.pages) if self.pages else 0.0

    @property
    def extraction_profile(self) -> str:
        if not self.pages:
            return "empty"
        if self.extraction_coverage == 0:
            return "image-only"
        if self.extraction_coverage < 0.75:
            return "partially scanned"
        if self.ocr_used:
            return "ocr-assisted"
        return "text-based"

    @property
    def unreadable_value_count(self) -> int:
        count = len(UNREADABLE_RE.findall(self.text))
        for page in self.pages:
            for table in page.tables:
                for row in table:
                    count += sum(1 for cell in row if UNREADABLE_RE.search(str(cell or "")))
        return count

    @property
    def merged_value_cell_count(self) -> int:
        count = 0
        for page in self.pages:
            for table in page.tables:
                for row in table:
                    for cell in row[1:]:
                        if len(NUMBER_RE.findall(str(cell or ""))) > 1:
                            count += 1
        return count

    @property
    def extraction_confidence(self) -> int:
        if not self.pages:
            return 0
        score = int(self.extraction_coverage * 100)
        if self.text_chars < 1000:
            score = min(score, 35)
        score -= min(40, self.unreadable_value_count * 3)
        if self.ocr_error:
            score = min(score, 30)
        return max(0, min(100, score))

    @property
    def table_extraction_confidence(self) -> int:
        if not self.pages:
            return 0
        score = self.extraction_confidence
        if self.merged_value_cell_count <= 10:
            merged_penalty = self.merged_value_cell_count
        else:
            merged_penalty = 10 + int((self.merged_value_cell_count - 10) * 0.5)
        score -= min(25, merged_penalty)
        return max(0, min(100, score))


@dataclass
class Finding:
    category: str
    severity: str
    location: str
    issue: str
    evidence: str
    recommendation: str
    metadata: dict[str, str] | None = None


@dataclass
class ReviewResult:
    findings: list[Finding]
    metrics: dict[str, Any]


@dataclass
class CompanyProfile:
    company_name: str = ""
    industry: str = ""
    reporting_currency: str = ""
    expected_policies: tuple[str, ...] = ()
    significant_transactions: tuple[str, ...] = ()
    presentation_standard: str = "IFRS"
    checklist_areas: tuple[str, ...] = ()


@dataclass
class ReviewOptions:
    use_ocr: bool = False
    ocr_max_pages: int | None = 60
    ocr_dpi: int = 300
    run_cautious_note_agreement: bool = False


@dataclass(frozen=True)
class ChecklistItem:
    standard: str
    area: str
    requirement: str
    applies_when: tuple[str, ...]
    evidence_keywords: tuple[str, ...]
    severity: str = "Medium"
