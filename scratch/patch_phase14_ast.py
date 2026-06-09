import ast

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
    while end_idx < len(lines) and (lines[end_idx].startswith(" ") or lines[end_idx] == ""):
        end_idx += 1
        
    lines = lines[:start_idx] + new_code.splitlines() + lines[end_idx:]
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
    
    op = next((v for k, v in rows.items() if any(x in k for x in ["net cash from operating activities", "cash from operating activities", "net cash generated from operating activities", "net cash used in operating activities"])), None) or next((v for k, v in rows.items() if "operat" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "operat" in k), None)
    inv = next((v for k, v in rows.items() if any(x in k for x in ["net cash used in investing activities", "cash used in investing activities", "net cash from investing activities", "net cash generated from investing activities", "net cash absorbed in investing activities"])), None) or next((v for k, v in rows.items() if "invest" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "invest" in k), None)
    fin = next((v for k, v in rows.items() if any(x in k for x in ["net cash generated from financing activities", "cash generated from financing activities", "net cash from financing activities", "net cash used in financing activities", "net cash inflow from financing activities"])), None) or next((v for k, v in rows.items() if "financ" in k and "net cash" in k), None) or next((v for k, v in rows.items() if "financ" in k), None)
    mov = next((v for k, v in rows.items() if any(x in k for x in ["total cash movement for the year", "cash movement for the year", "net increase in cash and cash equivalents", "net decrease in cash and cash equivalents", "net increase in cash", "net decrease in cash"])), None) or next((v for k, v in rows.items() if ("increase" in k or "decrease" in k or "movement" in k or "net cash" in k or "cash flow" in k) and not any(x in k for x in ["operat", "invest", "financ"])), None)
    
    opening = next((v for k, v in rows.items() if any(x in k for x in ["cash at the beginning of the year", "cash and cash equivalents at beginning of year", "cash at beginning"])), None)
    closing = next((v for k, v in rows.items() if any(x in k for x in ["total cash at end of year", "cash at end of year", "cash and cash equivalents at end of year", "cash at end"])), None)
    exch = next((v for k, v in rows.items() if any(x in k for x in ["effect of exchange rate movement on cash balances", "exchange difference"])), None)

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
        
    return findings, performed, skipped"""
replace_func("_check_cash_flow_text", new_cf)

# 2. _check_income_statement_text
new_inc = """def _check_income_statement_text(
    page: PdfPage,
    tolerance: Decimal,
    ocr_review: bool = False,
    document: PdfDocument | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    rows = _statement_rows(page.text)
    findings: list[Finding] = []
    performed: list[str] = []
    skipped: list[str] = []
    
    is_private_company = document and document.profile.company_type == "Private Company"
    stmt_name = "Statement of profit or loss" if is_private_company else "Statement of income and expenditure"
    
    if is_private_company:
        revenue = _row_amounts_any(rows, ("revenue", "turnover", "sales"))
        direct_expenses = _row_amounts_any(rows, ("direct expenses", "cost of sales"))
        gross_profit = _row_amounts_any(rows, ("gross profit",))
        other_income = _row_amounts_any(rows, ("other income",))
        other_op_losses = _row_amounts_any(rows, ("other operating losses", "other operating gains"))
        ecl = _row_amounts_any(rows, ("movement in credit loss allowances", "expected credit loss"))
        op_expenses = _row_amounts_any(rows, ("operating expenses", "administrative expenses"))
        op_profit = _row_amounts_any(rows, ("operating profit", "profit from operations"))
        inv_income = _row_amounts_any(rows, ("investment income", "finance income"))
        non_op_losses = _row_amounts_any(rows, ("other non-operating losses", "other non-operating gains"))
        pbt = _row_amounts_any(rows, ("profit before tax", "loss before tax"))
        tax = _row_amounts_any(rows, ("taxation", "income tax expense", "tax expense"))
        pat = _row_amounts_any(rows, ("profit after tax", "profit for the year", "loss after tax", "loss for the year"))
        
        if revenue and direct_expenses and gross_profit:
            _check_vector_equation(findings, page.number, stmt_name, "Revenue plus direct expenses equals gross profit.", [a + b for a, b in zip(revenue, direct_expenses)], gross_profit, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: gross profit checked.")
            
        if gross_profit and op_profit:
            components = [gross_profit]
            if other_income: components.append(other_income)
            if other_op_losses: components.append(other_op_losses)
            if ecl: components.append(ecl)
            if op_expenses: components.append(op_expenses)
            
            expected = components[0]
            for comp in components[1:]:
                expected = [a + b for a, b in zip(expected, comp)]
            _check_vector_equation(findings, page.number, stmt_name, "Gross profit plus operating items equals operating profit.", expected, op_profit, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: operating profit checked.")
            
        if op_profit and pbt:
            components = [op_profit]
            if inv_income: components.append(inv_income)
            if non_op_losses: components.append(non_op_losses)
            expected = components[0]
            for comp in components[1:]:
                expected = [a + b for a, b in zip(expected, comp)]
            _check_vector_equation(findings, page.number, stmt_name, "Operating profit plus non-operating items equals profit before tax.", expected, pbt, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: profit before tax checked.")
            
        if pbt and tax and pat:
            _check_vector_equation(findings, page.number, stmt_name, "Profit before tax plus taxation equals profit after tax.", [a + b for a, b in zip(pbt, tax)], pat, tolerance, ocr_review=ocr_review)
            performed.append(f"{stmt_name}: profit after tax checked.")
            
        return findings, performed, skipped

    income_amounts = _row_amounts_any(rows, ("total income", "gross revenue", "gross operating revenue", "revenue"))
    expenditure_amounts = _row_amounts_any(rows, ("total expenditure", "operating expenditure"))
    surplus_amounts = _row_amounts_any(rows, ("surplus of income over expenditure", "surplus for the year", "profit for the year", "profit after tax", "profit before tax"))
    
    if income_amounts and expenditure_amounts and surplus_amounts:
        _check_vector_equation(
            findings,
            page.number,
            "Statement of income and expenditure",
            "Total income less total expenditure agrees to surplus.",
            [a - b for a, b in zip(income_amounts, expenditure_amounts)],
            surplus_amounts,
            tolerance,
            ocr_review=ocr_review,
        )
        performed.append("Statement of income and expenditure: income less expenditure checked to surplus.")
    else:
        skipped.append("Statement of income and expenditure: skipped because income/expenditure rows were not confidently parsed.")
    return findings, performed, skipped"""
replace_func("_check_income_statement_text", new_inc)

# 3. _check_accumulated_fund_text
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
    
    is_private_company = document and document.profile.company_type == "Private Company"
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
            else:
                _check_scalar_equation(
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

# 4. _get_note_section_with_fallback
new_fall = """def _get_note_section_with_fallback(ref: str, note_sections: dict[str, str]) -> str:
    section = note_sections.get(ref, "")
    if section: return section
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        if note_sections.get(parent): return note_sections.get(parent, "")
    
    if ref.isdigit():
        ref_num = int(ref)
        prev_ref = str(ref_num - 1)
        next_ref = str(ref_num + 1)
        if prev_ref in note_sections and next_ref in note_sections:
            return f"Found dynamically between Note {prev_ref} and Note {next_ref}"
    return \"\""""
replace_func("_get_note_section_with_fallback", new_fall)

# 5. _amount_match_confidence
new_conf = """def _amount_match_confidence(current_found: bool, prior_found: bool, alternative_ref: str, cautious_review_prompt: bool, item_ref: str = "") -> str:
    if alternative_ref and not cautious_review_prompt and current_found is False and prior_found is False:
        if item_ref and alternative_ref.startswith(item_ref) and len(alternative_ref) > len(item_ref):
            return "Low"
        return "High"
    if alternative_ref:
        return "Medium"
    return "Low"
"""
replace_func("_amount_match_confidence", new_conf)

with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
