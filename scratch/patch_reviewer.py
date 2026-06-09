import re

with open("reviewer.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. False high note findings
text = re.sub(
    r'Finding\(\s*"Notes agreement",\s*"High",\s*"Primary statements",\s*f"Statement references note \{ref\}, but a matching note heading was not found.",\s*f"Detected statement reference: Note \{ref\}\.",\s*"Add the missing note or correct the note reference in the primary statement.",\s*\)',
    r'''Finding(
                "Extraction quality",
                "Low",
                "Notes agreement",
                f"Statement references note {ref}, but a matching note heading was not confidently detected or parsed; review prompt only.",
                f"Detected statement reference: Note {ref}.",
                "Confirm if the note exists manually. (Downgraded to Low to avoid false positives from OCR/heading extraction misses).",
            )''',
    text
)

# 2. Match note subsections
helpers = """def _get_note_section_with_fallback(ref: str, note_sections: dict[str, str]) -> str:
    section = note_sections.get(ref, "")
    if section: return section
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        return note_sections.get(parent, "")
    return ""

def _get_note_heading_with_fallback(ref: str, headings: dict[str, str]) -> str:
    heading = headings.get(ref, "")
    if heading: return heading
    if re.search(r'[A-Za-z]$', ref):
        parent = re.sub(r'[A-Za-z]+$', '', ref)
        return headings.get(parent, "")
    return ""

def _"""

if "def _get_note_section_with_fallback" not in text:
    text = re.sub(r'def _', helpers, text, count=1)
    text = text.replace('note_sections.get(item.ref, "")', '_get_note_section_with_fallback(item.ref, note_sections)')
    text = text.replace('note_sections.get(ref, "")', '_get_note_section_with_fallback(ref, note_sections)')
    text = text.replace('note_sections.get(candidate_ref, "")', '_get_note_section_with_fallback(candidate_ref, note_sections)')
    
    text = text.replace('headings.get(item.ref, "")', '_get_note_heading_with_fallback(item.ref, headings)')
    text = text.replace('headings.get(ref, "")', '_get_note_heading_with_fallback(ref, headings)')
    text = text.replace('headings.get(candidate_ref, "")', '_get_note_heading_with_fallback(candidate_ref, headings)')

# 3. Fix Note 4 and Note 9
text = text.replace(
"    if NUMBER_RE.search(title_clean):\n        return False",
"    if len(NUMBER_RE.findall(title_clean)) > 1:\n        return False"
)

# 4. Remove OCR fragments from line items
old_ocr = """    if any(phrase in lower for phrase in reject_phrases):
        return False
        
    text_only = re.sub(r"[\\d\\.,\\(\\)\\-\\|]", "", lower).strip()
    # Reject lines that contain only letters N, M, O (common unit/currency artifacts) or are too short.
    if len(text_only) < 3 or not re.sub(r"[nmo\\s]", "", text_only):
        return False"""
new_ocr = """    if any(phrase in lower for phrase in reject_phrases) or re.search(r"(?i)\\b(?:were signed|approval|n\\s*n)\\b", lower):
        return False
        
    text_only = re.sub(r"[\\d\\.,\\(\\)\\-\\|]", "", lower).strip()
    # Reject lines that contain only letters N, M, O (common unit/currency artifacts) or are too short.
    if len(re.sub(r"[nmo\\s]", "", text_only)) < 3:
        return False"""
text = text.replace(old_ocr, new_ocr)

# 5. Cash Flow Checks
cf_old = """    mov = next((v for k, v in rows.items() if ("increase" in k or "decrease" in k) and "cash" in k and "equivalent" in k), None)"""
cf_new = """    mov = next((v for k, v in rows.items() if ("increase" in k or "decrease" in k or "movement" in k) and "cash" in k), None)"""
text = text.replace(cf_old, cf_new)

# 7. Generic Table Arithmetic
text = re.sub(
    r'severity = "High" if is_primary else "Medium"',
    r'severity = "Low" if document.table_extraction_confidence < 90 else ("High" if is_primary else "Medium")',
    text
)

old_table_check = """    if not detailed_table_checks_allowed and not cautious_low_confidence:
        return findings"""
new_table_check = """    if document.table_extraction_confidence < 90:
        cautious_low_confidence = True
        
    if not detailed_table_checks_allowed and not cautious_low_confidence:
        return findings"""
text = text.replace(old_table_check, new_table_check)

# 8. Unreferenced Notes
unref_old = """        for ref in sorted(heading_refs - statement_refs, key=_note_sort_key):
            if ref.isdigit() and int(ref) <= 3:
                continue
            if _is_disclosure_only_note(headings[ref]):
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
            )"""
unref_new = """        for ref in sorted(heading_refs - statement_refs, key=_note_sort_key):
            if ref.isdigit() and int(ref) <= 3:
                continue
            if _is_disclosure_only_note(headings[ref]):
                continue
            document.unreferenced_notes = getattr(document, "unreferenced_notes", [])
            document.unreferenced_notes.append({
                "Note": ref,
                "Heading": headings[ref],
                "Comment": "Note exists but was not referenced from the extracted primary statements."
            })"""
text = text.replace(unref_old, unref_new)

# Add to review_pdf metrics
metrics_old = """    metrics["checks_performed_count"] = len(checks_performed)"""
metrics_new = """    metrics["checks_performed_count"] = len(checks_performed)
    metrics["unreferenced_notes"] = getattr(document, "unreferenced_notes", [])"""
text = text.replace(metrics_old, metrics_new)

with open("reviewer.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
