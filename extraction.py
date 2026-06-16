from __future__ import annotations

import re
import shutil
import warnings
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pypdf import PdfReader

from models import PdfDocument, PdfPage, ReviewOptions


NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?")
FINANCIAL_OCR_WORDS = {
    "statement",
    "changes",
    "equity",
    "capital",
    "premium",
    "reserve",
    "accumulated",
    "loss",
    "profit",
    "income",
    "balance",
    "january",
    "december",
    "share",
    "total",
    "company",
    "financial",
}


def extract_pdf(path: str | Path) -> PdfDocument:
    path = Path(path)
    fast_pages = _extract_text_fast(path)

    pages: list[PdfPage] = []
    rotated_page_details: list[dict[str, object]] = []
    executable = _resolve_tesseract()
    fitz_source = fitz.open(str(path)) if executable else None
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    try:
        with pdfplumber.open(str(path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                fast_text = fast_pages[index - 1].text if fast_pages and index - 1 < len(fast_pages) else ""
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or fast_text
                if not text.strip():
                    pages.append(PdfPage(index, "", []))
                    continue
                tables = page.extract_tables() or []
                cleaned_tables = [
                    [[_clean_cell(cell) for cell in row] for row in table if any(row)]
                    for table in tables
                ]

                if executable and fitz_source and _should_retry_with_rotated_ocr(text):
                    ocr_text, ocr_tables, rotation = _ocr_page_best_rotation(fitz_source[index - 1], 200)
                    if _ocr_text_quality_score(ocr_text) >= _ocr_text_quality_score(text) + 8:
                        text = ocr_text
                        cleaned_tables = ocr_tables
                        rotated_page_details.append({"page": index, "rotation": rotation})

                # If the extracted tables don't have good column structure, fallback to text lines
                has_good_structure = False
                for table in cleaned_tables:
                    if sum(1 for row in table if len(row) >= 2) >= 3:
                        has_good_structure = True
                        break

                if not cleaned_tables or not has_good_structure:
                    cleaned_tables = _tables_from_text_lines(text)
                pages.append(PdfPage(index, text, cleaned_tables))
    finally:
        if fitz_source is not None:
            fitz_source.close()
    if pages:
        document = PdfDocument(pages)
        setattr(document, "rotated_page_details", rotated_page_details)
        return document
    document = PdfDocument(fast_pages or [])
    setattr(document, "rotated_page_details", rotated_page_details)
    return document


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
    result = PdfDocument(
        pages,
        ocr_used=ocr_pages > 0,
        ocr_pages=ocr_pages,
        ocr_tables=sum(len(page.tables) for page in pages),
    )
    setattr(result, "rotated_page_details", getattr(base_document, "rotated_page_details", []))
    return result


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


def _ocr_page_best_rotation(page: object, dpi: int) -> tuple[str, list[list[list[str]]], int]:
    dpi = max(120, min(dpi, 300))
    best_text = ""
    best_tables: list[list[list[str]]] = []
    best_score = float("-inf")
    best_rotation = 0
    for rotation in (0, 90, 270, 180):
        matrix = fitz.Matrix(dpi / 72, dpi / 72).prerotate(rotation)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        with Image.open(BytesIO(pixmap.tobytes("png"))) as image:
            image = image.convert("L")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = pytesseract.image_to_data(image, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)
        lines = _ocr_lines_from_data(data)
        text = "\n".join(line["text"] for line in lines)
        score = _ocr_text_quality_score(text)
        if score > best_score:
            best_score = score
            best_text = text
            best_tables = _reconstruct_ocr_tables(lines)
            best_rotation = rotation
    return best_text, best_tables, best_rotation


def _should_retry_with_rotated_ocr(text: str) -> bool:
    sample = text[:2500]
    if not sample.strip():
        return False
    if "â‚¬" in sample or "â€" in sample or "€" in sample:
        return True
    tokens = re.findall(r"[A-Za-z]{3,}", sample)
    if len(tokens) < 20:
        return False
    recognisable = sum(1 for token in tokens if token.lower() in FINANCIAL_OCR_WORDS)
    odd_markers = len(re.findall(r"[â‚¬Â¢Â£Â¥]|[A-Za-z]{1,2}[â€˜â€™][A-Za-z]{1,3}|000,,|,,N", sample))
    return odd_markers >= 3 and recognisable <= 6


def _ocr_text_quality_score(text: str) -> int:
    if not text.strip():
        return 0
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    numbers = NUMBER_RE.findall(text)
    recognisable = sum(1 for token in tokens if token.lower() in FINANCIAL_OCR_WORDS)
    odd_markers = len(re.findall(r"[â‚¬Â¢Â£Â¥]|[â€˜â€™]|�", text))
    score = min(len(tokens), 80) + min(len(numbers) * 2, 60) + recognisable * 8 - odd_markers * 10
    if "statement of changes in equity" in text.lower():
        score += 50
    return score


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
    if not _table_rows_are_structured(rows):
        return []
    return [rows]


def _tables_from_text_lines(text: str) -> list[list[list[str]]]:
    rows = []
    for line in text.splitlines():
        row = _line_to_table_row(line)
        if row:
            rows.append(row)
    if len(rows) < 3:
        return []
    if not _table_rows_are_structured(rows):
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


def _table_rows_are_structured(rows: list[list[str]]) -> bool:
    amount_counts = [len(row) - 1 for row in rows if len(row) > 1]
    if len(amount_counts) < 3:
        return False
    most_common_count = max(set(amount_counts), key=amount_counts.count)
    consistency = amount_counts.count(most_common_count) / len(amount_counts)
    return most_common_count >= 2 and consistency >= 0.6


def _looks_like_note_column(value: str) -> bool:
    parsed = _parse_decimal(value)
    return parsed is not None and parsed == parsed.to_integral_value() and Decimal("1") <= parsed <= Decimal("99")


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


def _clean_cell(cell: object) -> str:
    return str(cell or "").replace("\n", " ").strip()


def _resolve_tesseract() -> str:
    executable = shutil.which("tesseract")
    if executable:
        return executable
    candidate = Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"
    return str(candidate) if candidate.exists() else ""
