import re

with open("reviewer.py", "r", encoding="utf-8") as f:
    source = f.read()

source = source.replace('row["Note reference"] for row in check_result_rows if row["Result"] == "Passed"', 'row["Note number"] for row in check_result_rows if row["Review result"] == "Passed"')

with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write(source)
