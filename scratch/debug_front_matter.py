from reviewer import _looks_like_front_matter_page, _normalise_match_words
from extraction import extract_pdf

path = r"C:\Users\ionawoga\Downloads\FS 2025 - CeLD Innovations Limited.pdf"
doc = extract_pdf(path)

for page in doc.pages:
    text = page.text
    lines = [line.strip() for line in text.splitlines()[:40] if line.strip()]
    head = "\n".join(lines).lower()
    front_terms = (
        "independent auditor",
        "auditor's report",
        "auditors report",
        "directors' report",
        "directors report",
        "corporate information",
        "report of the directors",
        "contents",
    )
    for term in front_terms:
        if term in head:
            print(f"Page {page.number} matched front_term: {term}")
            break
