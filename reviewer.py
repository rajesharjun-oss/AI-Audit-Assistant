from __future__ import annotations

import re
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pypdf import PdfReader


NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?")
YEAR_RE = re.compile(r"\b20\d{2}\b")
NOTE_REF_RE = re.compile(r"\bnote\s+(\d+[A-Za-z]?)\b|\bnotes?\s+(\d+[A-Za-z]?)\b", re.I)
NOTE_HEADING_RE = re.compile(r"^\s*(?:note\s+)?(\d+[A-Za-z]?)\s*[\).:-]?\s+(.{3,100})$", re.I)


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


POLICY_RULES = {
    "revenue": {
        "policy": ("revenue recognition", "ifrs 15", "revenue from contracts"),
        "evidence": ("revenue", "turnover", "sales", "contract asset", "contract liability"),
    },
    "inventory": {
        "policy": ("inventories", "inventory", "ias 2", "net realisable value"),
        "evidence": ("inventory", "inventories", "stock", "work in progress", "raw materials"),
    },
    "leases": {
        "policy": ("leases", "ifrs 16", "right-of-use", "right of use"),
        "evidence": ("lease liability", "right-of-use", "right of use", "leased asset"),
    },
    "ppe": {
        "policy": ("property, plant and equipment", "ias 16", "depreciation"),
        "evidence": ("property, plant and equipment", "ppe", "depreciation", "plant and machinery"),
    },
    "intangibles": {
        "policy": ("intangible", "ias 38", "amortisation", "amortization"),
        "evidence": ("intangible", "goodwill", "software", "amortisation", "amortization"),
    },
    "financial instruments": {
        "policy": ("financial instruments", "ifrs 9", "expected credit loss"),
        "evidence": ("trade receivables", "borrowings", "cash and cash equivalents", "loans", "ecl"),
    },
    "tax": {
        "policy": ("income tax", "deferred tax", "ias 12"),
        "evidence": ("tax expense", "current tax", "deferred tax", "tax payable"),
    },
    "foreign currency": {
        "policy": ("foreign currency", "exchange differences", "functional currency"),
        "evidence": ("foreign exchange", "exchange gain", "exchange loss", "translation reserve"),
    },
    "employee benefits": {
        "policy": ("employee benefits", "defined benefit", "pension", "ias 19"),
        "evidence": ("staff costs", "retirement benefit", "pension", "gratuity"),
    },
    "consolidation": {
        "policy": ("consolidated financial statements", "subsidiaries", "ifrs 10"),
        "evidence": ("non-controlling interest", "subsidiary", "group", "consolidated"),
    },
    "biological assets": {
        "policy": ("biological assets", "ias 41", "agricultural produce"),
        "evidence": ("biological assets", "livestock", "plantation", "agricultural produce"),
    },
    "investment property": {
        "policy": ("investment property", "ias 40", "fair value model"),
        "evidence": ("investment property", "rental income", "fair value gain"),
    },
}

INDUSTRY_POLICY_MISMATCHES = {
    "technology": ("biological assets", "mineral resources", "insurance contracts"),
    "software": ("biological assets", "mineral resources", "insurance contracts"),
    "fintech": ("biological assets", "mineral resources"),
    "manufacturing": ("insurance contracts", "biological assets"),
    "bank": ("biological assets", "construction contracts"),
    "financial services": ("biological assets", "construction contracts"),
}

SUPERSEDED_REFERENCES = {
    "ias 17": "IFRS 16 replaced IAS 17 for leases.",
    "ias 39": "IFRS 9 replaced IAS 39 for most financial instrument accounting.",
    "ifrs 4": "IFRS 17 replaced IFRS 4 for insurance contracts.",
    "sic-15": "SIC-15 was superseded by IFRS 16 lease guidance.",
    "ifric 4": "IFRIC 4 was superseded by IFRS 16 lease guidance.",
}

GENERIC_POLICY_PHRASES = (
    "the company has adopted all standards",
    "where applicable",
    "in the normal course of business",
    "management believes",
    "no material impact",
    "not applicable to the company",
)

STANDARD_CHECKLIST = (
    ChecklistItem(
        "IAS 1",
        "presentation",
        "Financial statements should include a complete set of primary statements and notes.",
        (),
        (
            "statement of financial position",
            "statement of profit or loss",
            "statement of changes in equity",
            "statement of cash flows",
            "notes to the financial statements",
        ),
        "High",
    ),
    ChecklistItem(
        "IAS 1",
        "going concern",
        "The report should disclose the going concern basis or material uncertainty where relevant.",
        (),
        ("going concern", "material uncertainty", "continue as a going concern"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 1",
        "judgements and estimates",
        "Significant judgements and key estimation uncertainty should be disclosed.",
        (),
        ("significant judgement", "critical accounting judgement", "key source of estimation", "estimation uncertainty"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 15",
        "revenue",
        "Revenue disclosures should explain performance obligations, timing, and disaggregation where revenue is significant.",
        ("revenue", "turnover", "sales", "contract asset", "contract liability"),
        ("performance obligation", "disaggregated revenue", "contract balance", "contract asset", "contract liability"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 16",
        "leases",
        "Lease disclosures should identify right-of-use assets, lease liabilities, depreciation, interest, and maturity information where leases exist.",
        ("lease", "right-of-use", "right of use", "lease liability"),
        ("right-of-use", "right of use", "lease liability", "lease maturity", "interest on lease", "depreciation of right"),
        "Medium",
    ),
    ChecklistItem(
        "IFRS 7 / IFRS 9",
        "financial instruments",
        "Financial instrument disclosures should cover risk exposure, credit risk, liquidity risk, fair value, and impairment methodology.",
        ("trade receivables", "borrowings", "loans", "cash and cash equivalents", "financial instruments"),
        ("credit risk", "liquidity risk", "market risk", "expected credit loss", "fair value", "maturity analysis"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 12",
        "tax",
        "Tax disclosures should reconcile current tax, deferred tax, and the effective tax relationship where tax is material.",
        ("tax expense", "income tax", "deferred tax", "current tax"),
        ("current tax", "deferred tax", "effective tax", "tax reconciliation", "tax rate reconciliation"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 16",
        "ppe",
        "Property, plant and equipment disclosures should include depreciation policy and carrying amount reconciliation.",
        ("property, plant and equipment", "ppe", "depreciation", "plant and machinery"),
        ("depreciation rate", "depreciation method", "carrying amount", "cost", "accumulated depreciation", "additions"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 38",
        "intangibles",
        "Intangible asset disclosures should include amortisation policy and carrying amount reconciliation.",
        ("intangible", "software", "goodwill", "amortisation", "amortization"),
        ("amortisation", "amortization", "useful life", "carrying amount", "impairment"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 36",
        "impairment",
        "Impairment disclosures should describe impairment losses, reversals, or key assumptions where impairment indicators or goodwill exist.",
        ("impairment", "goodwill", "cash-generating unit", "cgu"),
        ("impairment loss", "recoverable amount", "value in use", "cash-generating unit", "key assumption"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 33",
        "eps",
        "Entities presenting EPS should disclose basic and diluted EPS inputs.",
        ("earnings per share", "eps", "ordinary shares"),
        ("basic earnings per share", "diluted earnings per share", "weighted average", "ordinary shares"),
        "High",
    ),
    ChecklistItem(
        "IFRS 8",
        "segments",
        "Operating segment disclosures should reconcile segment revenue, profit, assets, and liabilities to the financial statements where segments are presented.",
        ("segment", "operating segment"),
        ("segment revenue", "segment profit", "segment assets", "reconciliation", "chief operating decision maker"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 24",
        "related parties",
        "Related party disclosures should identify relationships, transactions, balances, and key management compensation.",
        ("related party", "director", "key management", "shareholder"),
        ("related party", "key management compensation", "transactions", "outstanding balances"),
        "Medium",
    ),
    ChecklistItem(
        "IAS 10",
        "events after reporting period",
        "Events after the reporting period should be disclosed or explicitly addressed.",
        (),
        ("events after the reporting period", "subsequent events", "after the reporting date"),
        "Low",
    ),
    ChecklistItem(
        "IAS 37",
        "provisions and contingencies",
        "Provisions and contingent liabilities should disclose nature, uncertainty, movements, and possible obligations where relevant.",
        ("provision", "contingent", "litigation", "claim", "legal"),
        ("provision", "contingent liability", "movement", "uncertainty", "possible obligation"),
        "Medium",
    ),
)


def extract_pdf(path: str | Path) -> PdfDocument:
    path = Path(path)
    fast_pages = _extract_text_fast(path)
    if fast_pages is not None and not any(page.text.strip() for page in fast_pages):
        return PdfDocument(fast_pages)

    pages: list[PdfPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            fast_text = fast_pages[index - 1].text if fast_pages and index - 1 < len(fast_pages) else ""
            if not fast_text.strip():
                pages.append(PdfPage(index, "", []))
                continue
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or fast_text
            tables = page.extract_tables() if text.strip() else []
            cleaned_tables = [
                [[_clean_cell(cell) for cell in row] for row in table if any(row)]
                for table in (tables or [])
            ]
            pages.append(PdfPage(index, text, cleaned_tables))
    return PdfDocument(pages)


def _extract_text_fast(path: Path) -> list[PdfPage] | None:
    try:
        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            if not _page_has_font_resources(page):
                pages.append(PdfPage(index, "", []))
                continue
            pages.append(PdfPage(index, page.extract_text() or "", []))
        return pages
    except Exception:
        return None


def _page_has_font_resources(page: object) -> bool:
    try:
        resources = page.get("/Resources") or {}
        if hasattr(resources, "get_object"):
            resources = resources.get_object()
        return bool(resources.get("/Font"))
    except Exception:
        return True


def review_pdf(
    path: str | Path,
    profile: CompanyProfile | None = None,
    options: ReviewOptions | None = None,
) -> ReviewResult:
    options = options or ReviewOptions()
    document = extract_pdf(path)
    if _requires_ocr(document) and options.use_ocr:
        document = extract_pdf_with_ocr(path, document, options)
    profile = profile or CompanyProfile()
    findings: list[Finding] = []
    findings.extend(check_extraction_quality(document))
    if _requires_ocr(document):
        return _build_result(document, findings)
    findings.extend(check_totals_and_rounding(document))
    findings.extend(check_formatting(document, profile))
    findings.extend(check_notes_agreement(document))
    findings.extend(check_policy_relevance(document, profile))
    findings.extend(check_standard_checklist(document, profile))
    return _build_result(document, findings)


def _build_result(document: PdfDocument, findings: list[Finding]) -> ReviewResult:
    metrics = {
        "pages": len(document.pages),
        "text_pages": document.text_pages,
        "text_chars": document.text_chars,
        "extraction_coverage": f"{document.extraction_coverage:.0%}",
        "ocr_used": "Yes" if document.ocr_used else "No",
        "ocr_pages": document.ocr_pages,
        "ocr_tables": document.ocr_tables,
        "tables": sum(len(page.tables) for page in document.pages),
        "findings": len(findings),
        "high": sum(1 for item in findings if item.severity == "High"),
        "medium": sum(1 for item in findings if item.severity == "Medium"),
        "low": sum(1 for item in findings if item.severity == "Low"),
    }
    return ReviewResult(findings=findings, metrics=metrics)


def check_extraction_quality(document: PdfDocument) -> list[Finding]:
    findings: list[Finding] = []
    if document.ocr_error:
        findings.append(
            Finding(
                "Extraction quality",
                "High",
                "OCR pipeline",
                "OCR could not be completed for this PDF.",
                document.ocr_error,
                "Confirm Tesseract OCR is installed and accessible, then retry. You can also upload a text-selectable PDF.",
            )
        )
    if document.ocr_used and not _requires_ocr(document):
        findings.append(
            Finding(
                "Extraction quality",
                "Low",
                "OCR pipeline",
                "OCR was used to recover text from a scanned or image-based PDF.",
                f"OCR processed {document.ocr_pages} page(s), reconstructed {document.ocr_tables} table candidate(s), and text coverage is now {document.extraction_coverage:.0%}.",
                "Review extracted findings carefully because OCR can misread figures, punctuation, and note references in signed/scanned financial statements.",
            )
        )
        return findings
    if not _requires_ocr(document):
        return findings
    if document.text_chars == 0:
        evidence = f"0 extractable text pages out of {len(document.pages)} pages."
    else:
        evidence = (
            f"{document.text_pages} extractable text page(s) out of {len(document.pages)} pages; "
            f"{document.text_chars} extracted characters."
        )
    recommendation = (
        "Enable OCR in the app, or upload a text-selectable/exported PDF. Re-run the review after OCR so totals, "
        "notes, policies, and standards checklist checks can inspect the statement content."
    )
    if document.ocr_used:
        recommendation = (
            "OCR ran but did not recover enough text for reliable automated review. Try a higher-quality scan, "
            "a text-selectable PDF, or manual OCR settings."
        )
    findings.append(
        Finding(
            "Extraction quality",
            "High",
            "PDF extraction",
            "The PDF appears to be scanned or image-based, so automated audit checks cannot run reliably.",
            evidence,
            recommendation,
        )
    )
    return findings


def _requires_ocr(document: PdfDocument) -> bool:
    if not document.pages:
        return True
    return document.text_chars < 1000 or document.extraction_coverage < 0.25


def extract_pdf_with_ocr(
    path: str | Path,
    base_document: PdfDocument | None = None,
    options: ReviewOptions | None = None,
) -> PdfDocument:
    options = options or ReviewOptions(use_ocr=True)
    base_document = base_document or extract_pdf(path)
    executable = _resolve_tesseract()
    if not executable:
        return PdfDocument(
            base_document.pages,
            ocr_used=False,
            ocr_pages=0,
            ocr_tables=0,
            ocr_error="Tesseract OCR executable was not found on PATH or in the standard local install folder.",
        )
    pytesseract.pytesseract.tesseract_cmd = executable

    pages: list[PdfPage] = []
    ocr_pages = 0
    max_pages = options.ocr_max_pages if options.ocr_max_pages and options.ocr_max_pages > 0 else None
    try:
        source = fitz.open(str(path))
        try:
            for index, page in enumerate(source, start=1):
                existing = base_document.pages[index - 1] if index - 1 < len(base_document.pages) else PdfPage(index, "", [])
                if existing.text.strip() and len(existing.text.strip()) >= 100:
                    pages.append(existing)
                    continue
                if max_pages is not None and ocr_pages >= max_pages:
                    pages.append(existing)
                    continue
                text, tables = _ocr_page(page, options.ocr_dpi)
                pages.append(PdfPage(index, text, tables))
                ocr_pages += 1
        finally:
            source.close()
    except Exception as exc:
        return PdfDocument(
            base_document.pages,
            ocr_used=True,
            ocr_pages=ocr_pages,
            ocr_tables=sum(len(page.tables) for page in pages),
            ocr_error=f"OCR failed after {ocr_pages} page(s): {exc}",
        )
    return PdfDocument(
        pages,
        ocr_used=ocr_pages > 0,
        ocr_pages=ocr_pages,
        ocr_tables=sum(len(page.tables) for page in pages),
    )


def _ocr_page(page: object, dpi: int) -> tuple[str, list[list[list[str]]]]:
    dpi = max(120, min(dpi, 300))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
        image = image.convert("L")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pytesseract.image_to_data(image, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)
    lines = _ocr_lines_from_data(data)
    text = "\n".join(line["text"] for line in lines)
    tables = _reconstruct_ocr_tables(lines)
    return text, tables


def _ocr_lines_from_data(data: dict[str, list[object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    total = len(data.get("text", []))
    for index in range(total):
        token = str(data["text"][index] or "").strip()
        if not token:
            continue
        try:
            confidence = float(data.get("conf", ["-1"])[index])
        except ValueError:
            confidence = -1
        if confidence < 20:
            continue
        word = {
            "text": token,
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index]),
        }
        key = (
            int(data.get("block_num", [0])[index]),
            int(data.get("par_num", [0])[index]),
            int(data.get("line_num", [0])[index]),
        )
        grouped[key].append(word)

    lines: list[dict[str, object]] = []
    for words in grouped.values():
        words.sort(key=lambda item: int(item["left"]))
        line_text = " ".join(str(word["text"]) for word in words)
        lines.append(
            {
                "text": line_text,
                "top": min(int(word["top"]) for word in words),
                "left": min(int(word["left"]) for word in words),
                "bottom": max(int(word["top"]) + int(word["height"]) for word in words),
                "words": words,
            }
        )
    lines.sort(key=lambda item: (int(item["top"]), int(item["left"])))
    return lines


def _reconstruct_ocr_tables(lines: list[dict[str, object]]) -> list[list[list[str]]]:
    rows: list[list[str]] = []
    for line in lines:
        row = _line_to_table_row(str(line["text"]))
        if row:
            rows.append(row)
    if sum(1 for row in rows if len(row) >= 2) < 3:
        return []
    return [rows]


def _line_to_table_row(line: str) -> list[str] | None:
    cleaned = re.sub(r"\s+", " ", line).strip()
    if not cleaned:
        return None
    lower = cleaned.lower()
    if "docusign envelope" in lower or "envelope id" in lower:
        return None
    matches = list(NUMBER_RE.finditer(cleaned))
    if len(matches) < 2:
        return None
    first = matches[0]
    label = cleaned[: first.start()].strip(" .:-")
    if not label:
        return None
    amounts = [match.group(0).strip() for match in matches]
    if len(amounts) >= 3 and _looks_like_note_column(amounts[0]):
        amounts = amounts[1:]
    return [label, *amounts]


def _looks_like_note_column(value: str) -> bool:
    parsed = _parse_decimal(value)
    return parsed is not None and parsed == parsed.to_integral_value() and Decimal("1") <= parsed <= Decimal("99")


def _resolve_tesseract() -> str:
    executable = shutil.which("tesseract")
    if executable:
        return executable
    candidate = Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"
    return str(candidate) if candidate.exists() else ""


def check_standard_checklist(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    if profile.presentation_standard.upper() != "IFRS":
        return []
    text = document.text.lower()
    requested_areas = {area.strip().lower() for area in profile.checklist_areas if area.strip()}
    expected = {item.strip().lower() for item in profile.expected_policies if item.strip()}
    significant = {item.strip().lower() for item in profile.significant_transactions if item.strip()}
    context = " ".join(sorted(requested_areas | expected | significant | {profile.industry.lower()}))
    findings: list[Finding] = []

    for item in STANDARD_CHECKLIST:
        active = _checklist_item_applies(item, text, context, requested_areas)
        if not active:
            continue
        hits = [keyword for keyword in item.evidence_keywords if keyword in text]
        required_hits = len(item.evidence_keywords) if not item.applies_when else min(2, len(item.evidence_keywords))
        if len(hits) < required_hits:
            findings.append(
                Finding(
                    "Standards checklist",
                    item.severity,
                    item.standard,
                    f"Potential missing or incomplete {item.standard} disclosure: {item.area}.",
                    f"Checklist expectation: {item.requirement} Detected evidence: {', '.join(hits) if hits else 'none'}.",
                    "Review the disclosure against the applicable standard and add the missing policy, note, reconciliation, or judgement disclosure if applicable.",
                )
            )
    return findings


def check_rounding_and_casting(document: PdfDocument, tolerance: Decimal = Decimal("1")) -> list[Finding]:
    return check_totals_and_rounding(document, tolerance)


def check_totals_and_rounding(document: PdfDocument, tolerance: Decimal | None = None) -> list[Finding]:
    findings: list[Finding] = []
    scale_label, scale_tolerance = _detect_rounding_scale(document.text)
    tolerance = tolerance if tolerance is not None else scale_tolerance

    if scale_label == "mixed":
        findings.append(
            Finding(
                "Totals and rounding",
                "Medium",
                "Document-wide",
                "Mixed rounding or scaling labels were detected.",
                "The report refers to more than one presentation scale, such as units, thousands, or millions.",
                "Use one presentation basis consistently, or clearly label exceptions in the affected note or statement.",
            )
        )

    for page in document.pages:
        for table_index, table in enumerate(page.tables, start=1):
            if len(table) < 3:
                continue
            rows = [_numeric_row(row) for row in table]
            max_cols = max((len(row) for row in rows), default=0)
            for col in range(1, max_cols):
                _check_vertical_totals(findings, page.number, table_index, rows, col, tolerance)
            _check_cross_footings(findings, page.number, table_index, rows, tolerance)
            _check_column_consistency(findings, page.number, table_index, table)
    return findings


def check_formatting(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    text = document.text
    findings: list[Finding] = []
    currency_symbols = re.findall(r"\bUSD\b|\bNGN\b|\bGBP\b|\bEUR\b|US\$|Naira|Dollar|Pound|Euro|\$", text, flags=re.I)
    if profile.reporting_currency:
        expected = profile.reporting_currency.upper()
        unexpected = [symbol for symbol in currency_symbols if symbol.upper() != expected]
        if unexpected:
            findings.append(
                Finding(
                    "Formatting",
                    "Medium",
                    "Document-wide",
                    "Multiple or unexpected currency markers appear in the report.",
                    f"Expected {expected}; observed {dict(Counter(symbol.upper() for symbol in currency_symbols))}.",
                    "Confirm the presentation currency and standardise currency labels in statements, notes, and headers.",
                )
            )
    elif len(set(symbol.upper() for symbol in currency_symbols)) > 1:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "The report appears to use more than one currency marker.",
                f"Observed {dict(Counter(symbol.upper() for symbol in currency_symbols))}.",
                "Confirm whether mixed currencies are intentional and clearly labelled.",
            )
        )

    parenthesis_negatives = bool(re.search(r"\(\s?\d[\d,]*(?:\.\d+)?\s?\)", text))
    minus_negatives = bool(re.search(r"(?<!\w)-\d[\d,]*(?:\.\d+)?", text))
    if parenthesis_negatives and minus_negatives:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "Negative amounts use mixed styles.",
                "Both bracketed negatives and leading-minus negatives were detected.",
                "Use one negative-number convention consistently across the statements and notes.",
            )
        )
    elif minus_negatives and not parenthesis_negatives:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                "Document-wide",
                "Negative amounts appear to use leading minus signs instead of brackets.",
                "At least one negative amount was detected with a leading minus sign.",
                "If the reporting format requires brackets for negatives, update the affected statements and notes.",
            )
        )

    _check_comparatives(findings, document)
    _check_required_statement_names(findings, document, profile)
    for page in document.pages:
        bad_separators = [
            token for token in re.findall(r"\b\d{4,}(?:\.\d+)?\b", page.text)
            if _looks_like_unformatted_amount(token)
        ]
        if bad_separators:
            findings.append(
                Finding(
                    "Formatting",
                    "Low",
                    f"Page {page.number}",
                    "Some large numbers may be missing thousands separators.",
                    ", ".join(bad_separators[:5]),
                    "Review numeric formatting and apply the report's standard separator convention.",
                )
            )
    return findings


def _looks_like_unformatted_amount(token: str) -> bool:
    if YEAR_RE.fullmatch(token):
        return False
    digits = token.split(".")[0]
    if len(digits) < 5:
        return False
    if digits.startswith("0"):
        return False
    return True


def check_notes_agreement(document: PdfDocument, tolerance: Decimal = Decimal("1")) -> list[Finding]:
    text = document.text
    findings: list[Finding] = []
    headings = _note_headings(text)
    statement_refs = _statement_note_references(document)
    heading_refs = set(headings)
    for ref in sorted(statement_refs - heading_refs, key=_note_sort_key):
        findings.append(
            Finding(
                "Notes agreement",
                "High",
                "Primary statements",
                f"Statement references note {ref}, but a matching note heading was not found.",
                f"Detected statement reference: Note {ref}.",
                "Add the missing note or correct the note reference in the primary statement.",
            )
        )
    for ref in sorted(heading_refs - statement_refs, key=_note_sort_key):
        if ref.isdigit() and int(ref) <= 3:
            continue
        findings.append(
            Finding(
                "Notes agreement",
                "Low",
                f"Note {ref}",
                f"Note {ref} exists but was not referenced from the extracted primary statements.",
                headings[ref][:90],
                "Confirm whether this is a required disclosure-only note or whether a statement reference is missing.",
            )
        )

    note_sections = _note_sections(text)
    for ref, line, amount in _statement_lines_with_note_refs(document):
        section = note_sections.get(ref, "")
        if not section:
            continue
        note_amounts = _amounts_in_text(section)
        if note_amounts and not any(abs(note_amount - amount) <= tolerance for note_amount in note_amounts):
            findings.append(
                Finding(
                    "Notes agreement",
                    "Medium",
                    f"Note {ref}",
                    "The amount on the statement was not found in the related note text.",
                    f"Statement line: {line[:140]} | amount {amount:,}.",
                    "Confirm the note table totals agree to the face of the financial statement.",
                )
            )

    for ref, section in note_sections.items():
        title = headings.get(ref, "").lower()
        _check_note_internal_total(findings, ref, title, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("segment", "operating segment")):
            _check_segment_note(findings, ref, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("earnings per share", "eps")):
            _check_eps_note(findings, ref, section)
        if any(keyword in title or keyword in section.lower() for keyword in ("tax", "income tax", "deferred tax")):
            _check_tax_note(findings, ref, section, tolerance)
        if any(keyword in title or keyword in section.lower() for keyword in ("depreciation", "property, plant", "ppe")):
            _check_depreciation_note(findings, ref, section, tolerance)
    return findings


def check_policy_relevance(document: PdfDocument, profile: CompanyProfile) -> list[Finding]:
    text = document.text.lower()
    findings: list[Finding] = []
    expected = {item.strip().lower() for item in profile.expected_policies if item.strip()}
    significant = {item.strip().lower() for item in profile.significant_transactions if item.strip()}

    for policy_name, rule in POLICY_RULES.items():
        policy_present = any(keyword in text for keyword in rule["policy"])
        evidence_present = any(keyword in text for keyword in rule["evidence"])
        explicitly_expected = policy_name in expected or policy_name in significant
        if policy_present and not evidence_present and not explicitly_expected:
            findings.append(
                Finding(
                    "Accounting policies",
                    "Medium",
                    "Accounting policies note",
                    f"The {policy_name} policy is disclosed, but matching balances or activity were not detected.",
                    f"Policy indicators: {', '.join(rule['policy'][:3])}.",
                    "Remove boilerplate policy wording if it does not apply, or add the missing related disclosure if it does apply.",
                )
            )
        if (evidence_present or explicitly_expected) and not policy_present:
            findings.append(
                Finding(
                    "Accounting policies",
                    "Medium",
                    "Accounting policies note",
                    f"The report contains {policy_name}-related balances or expected transactions, but the matching accounting policy was not detected.",
                    f"Evidence indicators: {', '.join(rule['evidence'][:4])}.",
                    "Add or cross-reference the applicable accounting policy.",
                )
            )

    _check_industry_policy_fit(findings, text, profile)
    _check_superseded_standards(findings, text)
    _check_boilerplate_policy_language(findings, text)
    if profile.company_name and profile.company_name.lower() not in text:
        findings.append(
            Finding(
                "Formatting",
                "High",
                "Document-wide",
                "The configured company name was not detected in the extracted PDF text.",
                profile.company_name,
                "Confirm the correct report was uploaded and that headers/cover pages identify the reporting entity.",
            )
        )
    return findings


def findings_to_markdown(result: ReviewResult) -> str:
    lines = [
        "# AI Audit Assistant Review",
        "",
        build_ai_review_memo(result),
        "",
        f"Pages reviewed: {result.metrics['pages']}",
        f"Tables reviewed: {result.metrics['tables']}",
        f"Findings: {result.metrics['findings']} "
        f"(High {result.metrics['high']}, Medium {result.metrics['medium']}, Low {result.metrics['low']})",
        "",
        "## Review dimensions",
        "",
        "- Totals and rounding: totals, subtotals, cross-footings, and scaling labels.",
        "- Formatting: number formats, negative amounts, currency labels, comparatives, and statement presentation.",
        "- Notes agreement: note cross-references and reconciliation of note figures to face statements.",
        "- Accounting policies: relevance, missing policies, boilerplate wording, and superseded standards.",
        "- Standards checklist: triggered IFRS disclosure checks for presentation, policies, and transaction-specific notes.",
        "",
    ]
    if not result.findings:
        lines.append("No issues were detected by the automated checks.")
        return "\n".join(lines)
    for finding in result.findings:
        lines.extend(
            [
                f"## {finding.severity}: {finding.category}",
                f"Location: {finding.location}",
                f"Issue: {finding.issue}",
                f"Evidence: {finding.evidence}",
                f"Recommendation: {finding.recommendation}",
                "",
            ]
        )
    return "\n".join(lines)


def build_ai_review_memo(result: ReviewResult) -> str:
    if not result.findings:
        return (
            "AI review memo: No automated exceptions were detected. Perform a final manual review of scanned pages, "
            "judgemental disclosures, and any areas where PDF extraction may have missed tables."
        )
    by_category = Counter(finding.category for finding in result.findings)
    high_risk = [finding for finding in result.findings if finding.severity == "High"]
    top_categories = ", ".join(f"{category} ({count})" for category, count in by_category.most_common())
    if high_risk:
        first_priority = high_risk[0]
        priority = (
            f"Priority: start with {first_priority.category.lower()} at {first_priority.location}. "
            f"{first_priority.issue}"
        )
    else:
        first_priority = result.findings[0]
        priority = (
            f"Priority: no high-severity issue was detected; start with {first_priority.category.lower()} "
            f"at {first_priority.location}."
        )
    likely_causes = []
    categories = set(by_category)
    if "Totals and rounding" in categories:
        likely_causes.append("formula, casting, hidden-line, or rounding carry-forward differences")
    if "Formatting" in categories:
        likely_causes.append("inconsistent report template or late-stage manual formatting edits")
    if "Notes agreement" in categories:
        likely_causes.append("note schedules not updated after face statement changes")
    if "Accounting policies" in categories:
        likely_causes.append("boilerplate policy wording not tailored to the entity")
    cause_text = "; ".join(likely_causes) if likely_causes else "presentation or extraction exceptions"
    return (
        "AI review memo: "
        f"{result.metrics['findings']} findings were identified across {top_categories}. "
        f"{priority} Likely causes include {cause_text}. "
        "Recommended next step: clear high-severity items first, then re-run the review on the final PDF."
    )


def _checklist_item_applies(
    item: ChecklistItem,
    text: str,
    context: str,
    requested_areas: set[str],
) -> bool:
    if item.area in requested_areas or item.standard.lower() in requested_areas:
        return True
    if not item.applies_when:
        return True
    trigger_text = f"{text} {context}"
    return any(trigger in trigger_text for trigger in item.applies_when) or item.area in context


def _check_vertical_totals(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    rows: list[list[str | Decimal | None]],
    col: int,
    tolerance: Decimal,
) -> None:
    subtotal_rows: list[tuple[int, Decimal]] = []
    running: list[Decimal] = []
    for row_index, row in enumerate(rows):
        label = str(row[0]).lower() if row else ""
        value = row[col] if col < len(row) else None
        if not isinstance(value, Decimal):
            continue
        if _looks_like_total(label):
            subtotal_rows.append((row_index, value))
            expected = sum(running, Decimal("0"))
            diff = value - expected
            if running and abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Totals and rounding",
                        "High" if abs(diff) > tolerance * 5 else "Medium",
                        f"Page {page_number}, table {table_index}, row {row_index + 1}, column {col + 1}",
                        "Total or subtotal does not agree with the visible component rows.",
                        f"Reported {value:,}; visible sum {expected:,}; difference {diff:,}.",
                        "Trace the source schedule and confirm whether a hidden line, rounding adjustment, or formula error explains the variance.",
                    )
                )
            running = []
        elif _looks_like_amount_line(label):
            running.append(value)
    _check_adjacent_totals(findings, page_number, table_index, col, subtotal_rows, tolerance)


def _check_cross_footings(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    rows: list[list[str | Decimal | None]],
    tolerance: Decimal,
) -> None:
    for row_index, row in enumerate(rows):
        label = str(row[0]).lower() if row else ""
        values = [item for item in row[1:] if isinstance(item, Decimal)]
        if len(values) < 3 or not any(keyword in label for keyword in ("total", "segment", "tax", "depreciation")):
            continue
        expected = sum(values[:-1], Decimal("0"))
        reported = values[-1]
        diff = reported - expected
        if abs(diff) > tolerance:
            findings.append(
                Finding(
                    "Totals and rounding",
                    "Medium",
                    f"Page {page_number}, table {table_index}, row {row_index + 1}",
                    "Cross-footing across the row does not agree.",
                    f"Visible row sum {expected:,}; reported final column {reported:,}; difference {diff:,}.",
                    "Check whether the row total, segment total, or final column has been carried across correctly.",
                )
            )


def _check_column_consistency(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    table: list[list[str]],
) -> None:
    if not table:
        return
    header = " ".join(table[0]).lower()
    years = set(YEAR_RE.findall(header))
    if len(years) == 1 and any(YEAR_RE.search(" ".join(row)) for row in table[1:]):
        findings.append(
            Finding(
                "Formatting",
                "Low",
                f"Page {page_number}, table {table_index}",
                "A table appears to have only one comparative period in the header.",
                f"Header text: {header[:120]}.",
                "Confirm that the current and comparative reporting periods are both presented where required.",
            )
        )
    decimal_usage = Counter("decimal" if re.search(r"\d+\.\d+", cell) else "whole" for row in table for cell in row if NUMBER_RE.search(cell))
    if decimal_usage["decimal"] and decimal_usage["whole"]:
        findings.append(
            Finding(
                "Formatting",
                "Low",
                f"Page {page_number}, table {table_index}",
                "The table mixes whole-number and decimal amount formats.",
                f"Detected amount styles: {dict(decimal_usage)}.",
                "Standardise decimals according to the report's rounding basis.",
            )
        )


def _check_comparatives(findings: list[Finding], document: PdfDocument) -> None:
    primary_text = "\n".join(page.text for page in document.pages if not _is_notes_page(page.text))
    years = sorted(set(YEAR_RE.findall(primary_text)))
    if years and len(years) < 2:
        findings.append(
            Finding(
                "Formatting",
                "Medium",
                "Primary statements",
                "Only one reporting year was detected in the primary statements.",
                f"Detected years: {', '.join(years)}.",
                "Confirm whether comparative information is required and whether the comparative column is missing from extraction or presentation.",
            )
        )


def _check_required_statement_names(findings: list[Finding], document: PdfDocument, profile: CompanyProfile) -> None:
    if profile.presentation_standard.upper() != "IFRS":
        return
    text = document.text.lower()
    required = {
        "statement of financial position": ("statement of financial position", "balance sheet"),
        "statement of profit or loss": ("statement of profit or loss", "statement of comprehensive income", "income statement"),
        "statement of changes in equity": ("statement of changes in equity",),
        "statement of cash flows": ("statement of cash flows", "cash flow statement"),
        "notes to the financial statements": ("notes to the financial statements",),
    }
    missing = [name for name, aliases in required.items() if not any(alias in text for alias in aliases)]
    if missing:
        findings.append(
            Finding(
                "Formatting",
                "Medium",
                "Document-wide",
                "One or more standard IFRS statement headings were not detected.",
                f"Missing detected headings: {', '.join(missing)}.",
                "Confirm whether the report uses acceptable alternate headings or whether a required primary statement is missing.",
            )
        )


def _check_note_internal_total(findings: list[Finding], ref: str, title: str, section: str, tolerance: Decimal) -> None:
    lines = [line for line in section.splitlines() if line.strip()]
    running: list[Decimal] = []
    for line in lines:
        lower = line.lower()
        amount = _last_amount(line)
        if amount is None:
            continue
        if _looks_like_total(lower):
            expected = sum(running, Decimal("0"))
            diff = amount - expected
            if running and abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Notes agreement",
                        "Medium",
                        f"Note {ref}",
                        "A note subtotal or total does not agree to visible note line items.",
                        f"Note title: {title or 'untitled'} | reported {amount:,}; visible sum {expected:,}; difference {diff:,}.",
                        "Review the note table and agree it back to the supporting schedule and face statement.",
                    )
                )
            running = []
        elif _looks_like_amount_line(lower):
            running.append(amount)


def _check_segment_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    for line in section.splitlines():
        lower = line.lower()
        if "total" not in lower:
            continue
        values = _amounts_in_text(line)
        if len(values) >= 3:
            expected = sum(values[:-1], Decimal("0"))
            diff = values[-1] - expected
            if abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Notes agreement",
                        "High",
                        f"Note {ref}",
                        "Segment totals do not cross-foot to the reported total.",
                        f"Line: {line[:140]} | visible segment sum {expected:,}; reported total {values[-1]:,}.",
                        "Reconcile segment totals to the face statement and management reporting schedule.",
                    )
                )


def _check_eps_note(findings: list[Finding], ref: str, section: str) -> None:
    lower = section.lower()
    if "earnings per share" not in lower and "eps" not in lower:
        return
    earnings = _amount_near(section, ("profit attributable", "earnings", "profit for the year"))
    shares = _amount_near(section, ("weighted average", "ordinary shares", "shares"))
    eps = _amount_near(section, ("earnings per share", "basic eps", "diluted eps"))
    if earnings and shares and eps and shares != 0:
        calculated = earnings / shares
        if abs(calculated - eps) > Decimal("0.01"):
            findings.append(
                Finding(
                    "Notes agreement",
                    "High",
                    f"Note {ref}",
                    "EPS calculation does not agree to earnings divided by weighted average shares.",
                    f"Calculated EPS {calculated:.4f}; reported EPS {eps}.",
                    "Recalculate basic and diluted EPS using the final attributable earnings and weighted average share count.",
                )
            )
    elif "earnings per share" in lower or "eps" in lower:
        findings.append(
            Finding(
                "Notes agreement",
                "Low",
                f"Note {ref}",
                "EPS note was detected but the automated check could not identify all inputs.",
                "Expected inputs include attributable earnings, weighted average shares, and reported EPS.",
                "Review the EPS note manually and ensure both basic and diluted EPS are supported.",
            )
        )


def _check_tax_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    current_tax = _amount_near(section, ("current tax", "current income tax"))
    deferred_tax = _amount_near(section, ("deferred tax",))
    total_tax = _amount_near(section, ("tax expense", "income tax expense", "total tax"))
    if current_tax is not None and deferred_tax is not None and total_tax is not None:
        expected = current_tax + deferred_tax
        if abs(total_tax - expected) > tolerance:
            findings.append(
                Finding(
                    "Notes agreement",
                    "High",
                    f"Note {ref}",
                    "Tax expense does not agree to current tax plus deferred tax.",
                    f"Current tax {current_tax:,}; deferred tax {deferred_tax:,}; total tax {total_tax:,}.",
                    "Reconcile the tax note to the statement of profit or loss and deferred tax movement schedule.",
                )
            )


def _check_depreciation_note(findings: list[Finding], ref: str, section: str, tolerance: Decimal) -> None:
    depreciation = _amount_near(section, ("depreciation charge", "charge for the year", "depreciation"))
    profit_or_loss = _amount_near(section, ("profit or loss", "administrative expenses", "cost of sales"))
    if depreciation is not None and profit_or_loss is not None and abs(depreciation - profit_or_loss) > tolerance:
        findings.append(
            Finding(
                "Notes agreement",
                "Medium",
                f"Note {ref}",
                "Depreciation charge in the PPE note may not agree to the expense disclosure.",
                f"PPE depreciation {depreciation:,}; expense disclosure {profit_or_loss:,}.",
                "Agree depreciation charges to expense classification notes and the cash flow add-back.",
            )
        )


def _check_industry_policy_fit(findings: list[Finding], text: str, profile: CompanyProfile) -> None:
    industry = profile.industry.lower().strip()
    if not industry:
        return
    mismatches: tuple[str, ...] = ()
    for key, policies in INDUSTRY_POLICY_MISMATCHES.items():
        if key in industry:
            mismatches = policies
            break
    for policy in mismatches:
        if policy in text:
            findings.append(
                Finding(
                    "Accounting policies",
                    "High",
                    "Accounting policies note",
                    f"The {policy} policy appears inconsistent with the stated industry.",
                    f"Industry: {profile.industry}; detected policy phrase: {policy}.",
                    "Tailor the policy note to the entity and remove industry-irrelevant boilerplate unless the company actually has this activity.",
                )
            )


def _check_superseded_standards(findings: list[Finding], text: str) -> None:
    for reference, message in SUPERSEDED_REFERENCES.items():
        if reference in text:
            findings.append(
                Finding(
                    "Accounting policies",
                    "High",
                    "Accounting policies note",
                    f"Reference to superseded accounting guidance detected: {reference.upper()}.",
                    message,
                    "Update the accounting policy wording to current applicable standards and confirm transition disclosures where relevant.",
                )
            )


def _check_boilerplate_policy_language(findings: list[Finding], text: str) -> None:
    hits = [phrase for phrase in GENERIC_POLICY_PHRASES if phrase in text]
    if len(hits) >= 2:
        findings.append(
            Finding(
                "Accounting policies",
                "Low",
                "Accounting policies note",
                "Policy wording appears generic or boilerplate.",
                f"Detected generic phrases: {', '.join(hits[:4])}.",
                "Tailor the policy wording to the entity's actual transactions, estimates, judgements, and measurement bases.",
            )
        )


def _detect_rounding_scale(text: str) -> tuple[str, Decimal]:
    lower = text.lower()
    labels = set()
    if re.search(r"\$?0{3}s|000s|thousand|in thousands", lower):
        labels.add("thousands")
    if re.search(r"million|in millions", lower):
        labels.add("millions")
    if re.search(r"nearest dollar|actual amount|in units", lower):
        labels.add("units")
    if len(labels) > 1:
        return "mixed", Decimal("1")
    if "millions" in labels:
        return "millions", Decimal("1")
    if "thousands" in labels:
        return "thousands", Decimal("1")
    return "units", Decimal("1")


def _clean_cell(cell: object) -> str:
    return str(cell or "").replace("\n", " ").strip()


def _numeric_row(row: list[str]) -> list[str | Decimal | None]:
    converted: list[str | Decimal | None] = []
    for index, cell in enumerate(row):
        if index == 0:
            converted.append(cell)
            continue
        converted.append(_parse_decimal(cell))
    return converted


def _parse_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw in {"-", "--"}:
        return None
    match = NUMBER_RE.search(raw)
    if not match:
        return None
    token = match.group(0)
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "")
    try:
        amount = Decimal(token)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _amounts_in_text(text: str) -> list[Decimal]:
    amounts = [_parse_decimal(match.group(0)) for match in NUMBER_RE.finditer(text)]
    return [amount for amount in amounts if amount is not None]


def _amount_near(text: str, labels: tuple[str, ...]) -> Decimal | None:
    for line in text.splitlines():
        lower = line.lower()
        if any(label in lower for label in labels):
            amount = _last_amount(line)
            if amount is not None:
                return amount
    return None


def _looks_like_total(label: str) -> bool:
    keywords = ("total", "subtotal", "closing balance", "closing")
    return any(keyword in label for keyword in keywords)


def _looks_like_amount_line(label: str) -> bool:
    excluded = ("note", "year", "date", "audited", "restated")
    return bool(label.strip()) and not any(word in label for word in excluded)


def _check_adjacent_totals(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    col: int,
    subtotal_rows: list[tuple[int, Decimal]],
    tolerance: Decimal,
) -> None:
    for (first_row, first_value), (second_row, second_value) in zip(subtotal_rows, subtotal_rows[1:]):
        if abs(first_value - second_value) <= tolerance and second_row == first_row + 1:
            findings.append(
                Finding(
                    "Totals and rounding",
                    "Low",
                    f"Page {page_number}, table {table_index}, column {col + 1}",
                    "Adjacent subtotal or total rows show the same amount.",
                    f"Rows {first_row + 1} and {second_row + 1} both report approximately {first_value:,}.",
                    "Check whether a subtotal has been duplicated or whether one of the line descriptions should be revised.",
                )
            )


def _note_headings(text: str) -> dict[str, str]:
    headings: dict[str, str] = {}
    for line in text.splitlines():
        match = NOTE_HEADING_RE.match(line.strip())
        if match:
            number, title = match.groups()
            if not title.lower().startswith(("to the", "are", "and")):
                headings[number.upper()] = title.strip()
    return headings


def _statement_note_references(document: PdfDocument) -> set[str]:
    refs: set[str] = set()
    for page in document.pages:
        if _is_notes_page(page.text):
            continue
        refs.update(_refs_in_text(page.text))
    return refs


def _statement_lines_with_note_refs(document: PdfDocument) -> list[tuple[str, str, Decimal]]:
    lines: list[tuple[str, str, Decimal]] = []
    for page in document.pages:
        if _is_notes_page(page.text):
            continue
        for line in page.text.splitlines():
            refs = _refs_in_text(line)
            amount = _last_amount(line)
            if refs and amount is not None:
                for ref in refs:
                    lines.append((ref, line, amount))
    return lines


def _refs_in_text(text: str) -> set[str]:
    refs = set()
    for match in NOTE_REF_RE.finditer(text):
        refs.add((match.group(1) or match.group(2)).upper())
    return refs


def _note_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in text.splitlines():
        match = NOTE_HEADING_RE.match(line.strip())
        if match:
            current = match.group(1).upper()
        if current:
            sections[current].append(line)
    return {number: "\n".join(lines) for number, lines in sections.items()}


def _is_notes_page(text: str) -> bool:
    lower = text.lower()
    return "notes to the financial statements" in lower or lower.count("accounting polic") >= 2


def _last_amount(line: str) -> Decimal | None:
    amounts = _amounts_in_text(line)
    return amounts[-1] if amounts else None


def _note_sort_key(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)([A-Z]?)", value)
    if not match:
        return (9999, value)
    return (int(match.group(1)), match.group(2))
