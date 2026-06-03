from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class Finding:
    category: str
    severity: str
    location: str
    issue: str
    evidence: str
    recommendation: str


@dataclass
class ReviewResult:
    findings: list[Finding]
    metrics: dict[str, int | str]


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
    ocr_dpi: int = 200


@dataclass(frozen=True)
class ChecklistItem:
    standard: str
    area: str
    requirement: str
    applies_when: tuple[str, ...]
    evidence_keywords: tuple[str, ...]
    severity: str = "Medium"
