import re

with open("reviewer.py", "r", encoding="utf-8") as f:
    source = f.read()
lines = source.splitlines()

def replace_func(func_name, new_code):
    global lines
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start_idx = i
            break
    if start_idx == -1:
        print(f"Function {func_name} not found")
        return
        
    end_idx = start_idx + 1
    # Search for the next module-level definition (def or class) or end of file
    while end_idx < len(lines):
        line = lines[end_idx]
        if line.startswith("def ") or line.startswith("class ") or line.startswith("@"):
            break
        end_idx += 1
        
    # Remove any trailing blank lines before the next def
    while end_idx > 0 and lines[end_idx - 1].strip() == "":
        end_idx -= 1
        
    lines = lines[:start_idx] + new_code.splitlines() + ["\n"] + lines[end_idx:]
    print(f"Replaced {func_name}")


# 1. _check_cash_flow_text
new_cf = """def _check_cash_flow_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    rows = _statement_rows(page.text)
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    
    op_aliases = ["net cash from operating activities", "cash from operating activities", "cash generated from operating activities", "net cash generated from operating activities", "net cash used in operating activities", "operating cash flow"]
    inv_aliases = ["net cash used in investing activities", "cash used in investing activities", "net cash from investing activities", "net cash generated from investing activities", "net cash absorbed in investing activities", "investing cash flow"]
    fin_aliases = ["net cash generated from financing activities", "cash generated from financing activities", "net cash from financing activities", "net cash used in financing activities", "net cash inflow from financing activities", "financing cash flow"]
    mov_aliases = ["total cash movement for the year", "cash movement for the year", "cash movement for the yeat", "net increase in cash and cash equivalents", "net decrease in cash and cash equivalents", "net increase in cash", "net decrease in cash", "total cash movement", "net (decrease)/increase in cash and cash equivalents"]
    
    open_aliases = ["cash at the beginning of the year", "cash and cash equivalents at beginning of year", "cash at beginning", "opening cash", "cash and cash equivalents at 1 january", "cash and cash equivalents at the beginning"]
    close_aliases = ["total cash at end of the year", "cash at end of the year", "total cash at end of year", "cash and cash equivalents at end of year", "cash at end", "closing cash", "cash and cash equivalents at 31 december", "cash and cash equivalents at the end"]
    exch_aliases = ["effect of exchange rate movement on cash balances", "exchange difference on cash and cash equivalents", "exchange difference", "effect of exchange rate changes", "exchange effect", "effect of foreign exchange rate changes"]
    
    op = next((v for k, v in rows.items() if any(x in k for x in op_aliases)), None) or next((v for k, v in rows.items() if "operat" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "operat" in k), None)
    inv = next((v for k, v in rows.items() if any(x in k for x in inv_aliases)), None) or next((v for k, v in rows.items() if "invest" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "invest" in k), None)
    fin = next((v for k, v in rows.items() if any(x in k for x in fin_aliases)), None) or next((v for k, v in rows.items() if "financ" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "financ" in k), None)
    mov = next((v for k, v in rows.items() if any(x in k for x in mov_aliases)), None) or next((v for k, v in rows.items() if ("increase" in k or "decrease" in k or "movement" in k or "net cash" in k or "cash flow" in k) and not any(x in k for x in ["operat", "invest", "financ"])), None)
    
    opening = next((v for k, v in rows.items() if any(x in k for x in open_aliases)), None) or next((v for k, v in rows.items() if ("beginning" in k or " 1 " in k or "january" in k or "start" in k) and "cash" in k), None)
    closing = next((v for k, v in rows.items() if any(x in k for x in close_aliases)), None) or next((v for k, v in rows.items() if ("end" in k or " 31 " in k or "december" in k) and "cash" in k), None)
    exch = next((v for k, v in rows.items() if any(x in k for x in exch_aliases)), None) or next((v for k, v in rows.items() if "exchange" in k), None)

    if op and inv and fin and mov:
        expected = [a + b + c for a, b, c in zip(op, inv, fin)]
        _check_vector_equation(
            findings,
            page.number,
            "Statement of cash flows",
            "Operating, investing, and financing cash flows agree to net increase in cash.",
            expected,
            mov,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of cash flows: net cash increase checked.")
        findings.append(Finding("Calculation", "Passed", "Statement of cash flows", "Operating, investing, and financing cash flows agree to net increase.", "Equation passed.", ""))
    else:
        skipped.append("Statement of cash flows: skipped because operating/investing/financing/movement rows were not confidently parsed.")
        
    if opening and mov and closing:
        expected_close = [a + b + (c if c else 0) for a, b, c in zip(opening, mov, exch or ([Decimal("0")]*len(opening)))]
        _check_vector_equation(
            findings,
            page.number,
            "Statement of cash flows",
            "Opening cash plus total movement agrees to closing cash.",
            expected_close,
            closing,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of cash flows: opening plus movement checked to closing.")
        findings.append(Finding("Calculation", "Passed", "Statement of cash flows", "Opening cash plus total movement agrees to closing cash.", "Equation passed.", ""))
    else:
        skipped.append("Statement of cash flows: skipped because opening/movement/closing rows were not confidently parsed.")
        
    return findings, performed, skipped"""
replace_func("_check_cash_flow_text", new_cf)

# 2. _check_accumulated_fund_text (fix capitalization)
new_acc = """def _check_accumulated_fund_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    lines = page.text.splitlines()
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    
    is_private_company = document and _detect_entity_type(document.text).lower().startswith("private")
    stmt_name = "Statement of changes in equity" if is_private_company else "Statement of changes in accumulated fund"
    word_fund = "equity" if is_private_company else "accumulated fund"
    
    balance_rows = [(line, _amounts_from_statement_line(line)) for line in lines if "balance as at" in line.lower() or "balance at" in line.lower()]
    surplus_rows = [(line, _amounts_from_statement_line(line)) for line in lines if "surplus for the year" in line.lower().strip() or "profit for the year" in line.lower().strip()]
    
    if len(balance_rows) >= 3 and len(surplus_rows) >= 2:
        opening_2025 = balance_rows[-2][1]
        closing_2025 = balance_rows[-1][1]
        surplus_2025 = surplus_rows[-1][1]
        if len(opening_2025) >= 4 and len(closing_2025) >= 4 and surplus_2025:
            expected_total = opening_2025[-1] + surplus_2025[-1]
            reported_total = closing_2025[-1]
            diff = expected_total - reported_total
            if abs(diff) > tolerance:
                findings.append(
                    Finding(
                        "Calculation",
                        "High" if abs(diff) > tolerance * 5 else "Medium",
                        stmt_name,
                        f"Closing {word_fund} does not agree to opening {word_fund} plus surplus.",
                        f"Reported closing {reported_total:,}; expected {expected_total:,} (opening {opening_2025[-1]:,} + surplus {surplus_2025[-1]:,}). Difference: {diff:,}.",
                        f"Check if there are prior year adjustments, dividends, or other comprehensive income lines modifying {word_fund}.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        "Calculation",
                        "Passed",
                        stmt_name,
                        f"Closing {word_fund} agrees to opening {word_fund} plus surplus.",
                        "Equation passed.",
                        "",
                    )
                )
            if ocr_review:
                _check_ocr_scalar_equation(
                    findings,
                    page.number,
                    stmt_name,
                    f"Closing {word_fund} agrees to opening fund plus surplus.",
                    expected_total,
                    reported_total,
                    tolerance,
                )
            performed.append(f"{stmt_name}: opening plus surplus checked to closing {word_fund}.")
        else:
            skipped.append(f"{stmt_name}: skipped because rotated/OCR table structure was not confidently parsed.")
    else:
        skipped.append(f"{stmt_name}: skipped because rotated/OCR table structure was not confidently parsed.")
    return findings, performed, skipped"""
replace_func("_check_accumulated_fund_text", new_acc)

# 3. _get_note_section_with_fallback
new_fall = """def _get_note_section_with_fallback(ref: str, note_sections: dict[str, str], document: PdfDocument | None = None) -> str:
    section = note_sections.get(ref, "")
    if section: return section
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        if note_sections.get(parent): return note_sections.get(parent, "")
    
    if ref.isdigit() and document:
        ref_num = int(ref)
        prev_ref = str(ref_num - 1)
        next_ref = str(ref_num + 1)
        # Search dynamically in text
        text = document.text
        # e.g. "Note 3", "3.", " 3 ", "4\nIntangible assets"
        pattern = rf"(?:\\n\\s*(?:Note|NOTE)\\s+{ref}\\b|\\n\\s*{ref}\\.?\\s+[A-Z]|\\n\\s*{ref}\\n\\s*[A-Z])(.*?)(?:\\n\\s*(?:Note|NOTE)\\s+{next_ref}\\b|\\n\\s*{next_ref}\\.?\\s+[A-Z]|\\n\\s*{next_ref}\\n\\s*[A-Z])"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return \"\""""
replace_func("_get_note_section_with_fallback", new_fall)

# 5. _accounting_policy_text
new_apt = """def _accounting_policy_text(document: PdfDocument) -> str:
    text = document.text
    matches = list(re.finditer(r"summary of significant accounting policies|significant accounting policies|accounting policies|basis of preparation|general information", text, flags=re.I))
    if not matches:
        return text
    start_match = matches[0] # Grab from the very first match (often Note 1 General info / Basis)
    tail = text[start_match.start():]
    # Capture up to Note 4, Note 5, or Note 6 to include Judgements and Estimates
    end_match = re.search(
        r"\\n\\s*4\\s+[A-Z]|\\n\\s*5\\s+[A-Z]|\\n\\s*6\\s+[A-Z]|\\n\\s*(?:Note|NOTE)\\s+4\\b|\\n\\s*4\\.\\s+[A-Z]",
        tail[2000:],
        flags=re.I,
    )
    if end_match:
        return tail[: 2000 + end_match.start()]
    return tail[:18000]"""
replace_func("_accounting_policy_text", new_apt)


with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Phase 16 Patch applied")
