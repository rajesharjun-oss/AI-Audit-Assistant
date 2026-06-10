import re

with open("reviewer.py", "r", encoding="utf-8") as f:
    source = f.read()

source = source.replace('passed_refs = {row["Note number"] for row in check_result_rows if row["Review result"] == "Passed"}', 'passed_refs = {str(row["Note number"]).strip() for row in check_result_rows if row["Review result"] == "Passed"}')

# Fix cautious findings comparison
source = source.replace('if ref_match and ref_match.group(1) in passed_refs:', 'if ref_match and ref_match.group(1).strip() in passed_refs:')

with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write(source)
