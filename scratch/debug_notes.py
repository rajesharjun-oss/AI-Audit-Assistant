from reviewer import _notes_start_page, _looks_like_front_matter_page, _notes_heading_in_text
from extraction import extract_pdf

path = r"C:\Users\ionawoga\Downloads\FS 2025 - CeLD Innovations Limited.pdf"
doc = extract_pdf(path)

for page in doc.pages:
    is_front = _looks_like_front_matter_page(page.text)
    is_heading = _notes_heading_in_text(page.text)
    if is_heading:
        print(f"Page {page.number} has heading. is_front={is_front}")
