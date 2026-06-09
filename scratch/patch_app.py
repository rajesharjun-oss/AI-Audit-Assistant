import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Insert Unreferenced notes sheet
old_code = """        pd.DataFrame(policy_rows).to_excel(writer, sheet_name="Notes 1 and 2 policy review", index=False)
        
        cross_export = result.metrics.get("cross_page_export", {})"""
new_code = """        pd.DataFrame(policy_rows).to_excel(writer, sheet_name="Notes 1 and 2 policy review", index=False)
        
        unref_rows = result.metrics.get("unreferenced_notes", []) or [{"Note": "None", "Heading": "None found", "Comment": "All notes referenced or filtered"}]
        pd.DataFrame(unref_rows).to_excel(writer, sheet_name="Unreferenced notes", index=False)
        
        cross_export = result.metrics.get("cross_page_export", {})"""
text = text.replace(old_code, new_code)

old_format = """        _format_excel_table_sheet(writer.book["Notes 1 and 2 policy review"], "PolicyReview")"""
new_format = """        _format_excel_table_sheet(writer.book["Notes 1 and 2 policy review"], "PolicyReview")
        _format_excel_table_sheet(writer.book["Unreferenced notes"], "UnreferencedNotes")"""
text = text.replace(old_format, new_format)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
