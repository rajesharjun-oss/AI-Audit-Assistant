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


# _check_vertical_totals
new_vert = """def _check_vertical_totals(
    findings: list[Finding],
    page_number: int,
    table_index: int,
    rows: list[list[str | Decimal | None]],
    col: int,
    tolerance: Decimal,
) -> None:
    subtotal_rows: list[tuple[int, Decimal]] = []
    running: list[Decimal] = []
    expected_amount_count = _common_amount_count(rows)
    for row_index, row in enumerate(rows[1:], start=1):
        label = str(row[0]).lower() if row else ""
        if _is_table_boundary_row(row):
            running = []
            continue
        if expected_amount_count and _row_amount_count(row) not in {0, expected_amount_count}:
            running = []
            continue
        value = row[col] if col < len(row) else None
        if not isinstance(value, Decimal):
            continue
        if _looks_like_total(label):
            subtotal_rows.append((row_index, value))
            expected = sum(running, Decimal("0"))
            diff = value - expected
            if running and abs(diff) > tolerance:
                massive_deviation = False
                if abs(value) > 0 and abs(expected) / abs(value) > 10:
                    massive_deviation = True
                if massive_deviation:
                    findings.append(
                        Finding(
                            "Extraction",
                            "Low",
                            f"Page {page_number}, table {table_index}, row {row_index + 1}, column {col + 1}",
                            "Table structure likely parsed incorrectly.",
                            f"Reported {value:,}; visible sum {expected:,}.",
                            "Ensure the table rows were extracted correctly.",
                        )
                    )
                else:
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
"""
replace_func("_check_vertical_totals", new_vert)

with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
