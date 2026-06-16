import re
from models import PdfDocument, Finding, CompanyProfile

STANDARD_TOPICS = {
    "IFRS 15": ["revenue", "performance obligation", "contract asset", "contract liability", "timing", "customer contract", "goods or services"],
    "IFRS 9": ["financial instrument", "classification", "measurement", "impairment", "ecl", "credit risk", "amortised cost", "fvoci", "fvtpl", "expected credit loss"],
    "IAS 12": ["current tax", "deferred tax", "temporary difference", "tax expense", "tax reconciliation", "taxation"],
    "IAS 16": ["ppe", "property, plant", "depreciation", "useful li", "residual value", "impairment", "disposal"],
    "IAS 38": ["intangible", "amortisation", "useful li", "impairment", "development cost"],
    "IAS 1": ["basis of preparation", "going concern", "presentation", "material accounting polic", "judgement", "estimate"],
    "IAS 36": ["impairment indicator", "recoverable amount", "value in use", "fair value less cost", "impairment reversal"],
    "IFRS 16": ["lease", "right-of-use", "lessee", "lessor"],
    "IAS 2": ["inventory", "inventories", "net realisable value", "fifo", "weighted average"],
    "IAS 19": ["employee benefit", "defined contribution", "defined benefit", "post-employment", "pension"],
    "IAS 24": ["related party", "key management personnel"]
}

NOTES_HEADING_PATTERNS = (
    "notes to the financial statements",
    "notes to financial statements",
    "notes forming part of the financial statements",
    "notes to the accounts",
)

def _extract_nature_of_business(document: PdfDocument, note_1_2_text: str) -> str:
    """Attempts to extract the nature of business from Note 1/2 or General Info."""
    text_lower = note_1_2_text.lower()
    
    def _clean_match(text: str) -> str:
        paragraphs = re.split(r'\n\s*\n', text)
        for p in paragraphs:
            if re.search(r"(?:nature of business|principal activities?|principal business)", p):
                clean = re.split(r"(?i)\b(?:by order of the board|frc/|director|secretary|dated|signed on its behalf|behalf of the board|report|financial statements)\b", p)[0]
                return re.sub(r'\s+', ' ', clean).strip()
        match = re.search(r"(?:nature of business|principal activities?|principal business)(?:[\s\S]{1,200})", text)
        if match:
            clean = re.split(r"(?i)\b(?:by order of the board|frc/|director|secretary|dated|signed on its behalf|behalf of the board|report|financial statements)\b", match.group(0))[0]
            return re.sub(r'\s+', ' ', clean).strip()
        return ""

    found = _clean_match(text_lower)
    if found:
        return found
    
    doc_text_lower = "\n".join(p.text.lower() for p in document.pages[:15])
    found_doc = _clean_match(doc_text_lower)
    if found_doc:
        return found_doc
        
    if "loyalty" in text_lower or "rewards" in text_lower:
        return "consumer loyalty and rewards / cash reward service"
        
    return ""

def _infer_expected_policies(nature_text: str, document_text: str = "") -> list[str]:
    policies = []
    nature_text = nature_text.lower()
    if any(w in nature_text for w in ["sell", "sale", "retail", "wholesale", "goods", "trading", "consumer"]):
        policies.extend(["revenue from customer contracts", "trade receivables and ECL"])
        if "inventor" in nature_text or "inventor" in document_text.lower() or "stock" in nature_text:
            policies.append("inventory")
    if any(w in nature_text for w in ["software", "technology", "platform", "app ", "digital", "loyalty", "reward"]):
        policies.extend(["revenue from customer contracts", "contract liabilities", "ECL/trade receivables", "financial instruments", "intangible/software", "tax", "PPE", "cash and cash equivalents"])
    if any(w in nature_text for w in ["manufactur", "production", "plant", "factory"]):
        policies.extend(["property, plant and equipment", "revenue from customer contracts"])
        if "inventor" in nature_text or "inventor" in document_text.lower() or "stock" in nature_text:
            policies.append("inventory")
    if any(w in nature_text for w in ["bank", "financ", "lend", "loan", "credit", "invest"]):
        policies.extend(["financial instruments", "ECL", "fair value measurement"])
    return list(set(policies))


def _policy_heading_from_note_text(note_text: str) -> str:
    for line in str(note_text or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip(" -:.")
        if clean and len(clean) > 5:
            return clean
    return ""


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _policy_paragraph_usable(paragraph: str) -> bool:
    normalized = _normalise_text(paragraph)
    if not normalized or len(normalized) < 40:
        return False
    header_markers = (
        "directors report",
        "independent auditor",
        "statement of financial position",
        "statement of profit or loss",
        "statement of comprehensive income",
        "statement of cash flows",
        "statement of changes in equity",
        "together with the directors",
        "financial statements for the year ended",
        "notes to the financial statements",
        "value added statement",
        "five year financial summary",
    )
    if any(marker in normalized[:220] for marker in header_markers):
        return False
    digit_count = sum(character.isdigit() for character in paragraph)
    letter_count = sum(character.isalpha() for character in paragraph)
    non_ascii_count = sum(1 for character in paragraph if ord(character) > 127)
    if non_ascii_count > max(12, len(paragraph) // 10):
        return False
    if digit_count > max(18, int(letter_count * 0.4)):
        return False
    if normalized.count("n '000") >= 2 or normalized.count("n'000") >= 2:
        return False
    return True


def _policy_chunks(note_text: str, prefix: str) -> list[str]:
    if not note_text.strip():
        return []
    pattern = rf"(?=^\s*{re.escape(prefix)}\.\d+\b)"
    parts = re.split(pattern, note_text, flags=re.M)
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks if chunks else [note_text]


def _policy_match_score(paragraph_lower: str, keywords: list[str]) -> int:
    score = sum(1 for keyword in keywords if keyword in paragraph_lower)
    head = paragraph_lower[:180]
    score += sum(2 for keyword in keywords if keyword in head)
    return score


def _preview_around_keywords(paragraph: str, keywords: list[str], width: int = 200) -> str:
    lower = paragraph.lower()
    for keyword in keywords:
        index = lower.find(keyword)
        if index == -1:
            continue
        start = max(0, index - 40)
        end = min(len(paragraph), index + width)
        snippet = paragraph[start:end].strip()
        return ("... " if start > 0 else "") + snippet + ("..." if end < len(paragraph) else "")
    return paragraph[:200] + ("..." if len(paragraph) > 200 else "")


def _notes_1_and_2_text(document: PdfDocument, note_sections: dict[str, str]) -> tuple[str, str]:
    note_1 = str(note_sections.get("1", "") or "")
    note_2 = str(note_sections.get("2", "") or "")
    if not note_1 and not note_2:
        return "", ""

    pages = document.pages
    note_2_heading = _policy_heading_from_note_text(note_2)
    notes_start_page = None
    fallback_notes_page = None
    note_2_page = None

    for page in pages:
        page_lower = page.text.lower()
        if any(marker in page_lower for marker in NOTES_HEADING_PATTERNS):
            if fallback_notes_page is None:
                fallback_notes_page = page.number
            if notes_start_page is None and any(
                marker in page_lower
                for marker in ("material accounting policies", "significant accounting policies", "basis of preparation", "1.1", "1.2")
            ):
                notes_start_page = page.number
    notes_start_page = notes_start_page or fallback_notes_page

    for page in pages:
        page_lower = page.text.lower()
        if notes_start_page is None or page.number < notes_start_page:
            continue
        normalized_page = _normalise_text(page.text)
        if note_2_page is None and (
            re.search(r"(?:^|\n)\s*2[\).]?\s+(?:new\s+standards|standards\s+and\s+interpretations)\b", page.text, re.I)
            or (note_2_heading and _normalise_text(note_2_heading[:60])[:35] in normalized_page)
        ):
            note_2_page = page.number

    if note_1 and notes_start_page and note_2_page and note_2_page > notes_start_page:
        middle_pages = [
            page.text
            for page in pages
            if notes_start_page < page.number < note_2_page
        ]
        if middle_pages:
            note_1 = "\n\n".join([note_1, *middle_pages]).strip()

    return note_1, note_2

TOPIC_POLICY_MAP = {
    "Basis of preparation": None,
    "Significant judgements and estimates": None,
    "PPE": "ppe",
    "Intangible assets": "intangibles",
    "Financial instruments": "financial instruments",
    "Receivables / ECL": "financial instruments",
    "Tax": "tax",
    "Revenue": "revenue",
    "Contract liabilities": "revenue",
    "Cash and cash equivalents": "financial instruments",
    "Standards and interpretations": None,
}


def _row_status(comment: str, suggested: str) -> str:
    if "[missing policy]" in comment.lower():
        return "Not elevated"
    if suggested:
        return "Review"
    return "Observed"


def _topic_expected(topic: str, expected_policies: list[str], document_text: str, nature_text: str) -> bool:
    lower_expected = " | ".join(expected_policies).lower()
    lower_doc = document_text.lower()
    lower_nature = nature_text.lower()
    if topic in {"Basis of preparation", "Significant judgements and estimates"}:
        return True
    if topic == "Revenue":
        return any(term in lower_expected or term in lower_doc or term in lower_nature for term in ("revenue", "contract", "customer", "turnover", "income"))
    if topic == "Contract liabilities":
        return any(term in lower_doc for term in ("contract liability", "deferred revenue", "unearned revenue", "advance from customers"))
    if topic == "Tax":
        return any(term in lower_doc for term in ("tax", "taxation", "deferred tax", "current tax"))
    if topic == "Cash and cash equivalents":
        return any(term in lower_doc for term in ("cash and cash equivalents", "bank balance", "short-term deposit"))
    if topic == "Receivables / ECL":
        return any(term in lower_doc for term in ("receivable", "expected credit loss", "ecl", "loss allowance", "trade receivable"))
    if topic == "Financial instruments":
        return any(term in lower_doc for term in ("financial asset", "financial liability", "fvtpl", "fvoci", "amortised cost", "financial instrument"))
    if topic == "PPE":
        return any(term in lower_doc for term in ("property, plant", "ppe", "depreciation"))
    if topic == "Intangible assets":
        return any(term in lower_doc for term in ("intangible", "software", "amortisation", "amortization"))
    return False


def review_notes_1_and_2(
    document: PdfDocument,
    profile: CompanyProfile,
    note_sections: dict[str, str],
    policy_map: dict[str, bool] | None = None,
) -> tuple[list[Finding], list[dict[str, str]]]:
    findings = []
    export_rows = []
    policy_map = policy_map or {}
    
    note_1 = note_sections.get("1", "")
    note_2 = note_sections.get("2", "")
    note_1, note_2 = _notes_1_and_2_text(document, note_sections)
    
    # Also include any sub-notes like 1.1, 2.1, 2.2 etc.
    subnotes_1 = [v for k, v in note_sections.items() if str(k).startswith("1.") or str(k).startswith("1A") or str(k).startswith("1B")]
    subnotes_2 = [v for k, v in note_sections.items() if str(k).startswith("2.") or str(k).startswith("2A") or str(k).startswith("2B")]
    
    combined_text = (note_1 + "\n\n" + "\n\n".join(subnotes_1) + "\n\n" + note_2 + "\n\n" + "\n\n".join(subnotes_2)).strip()
    if not combined_text:
        return findings, export_rows
        
    nature_of_business = _extract_nature_of_business(document, combined_text)
    
    if nature_of_business and nature_of_business in combined_text:
        combined_text = combined_text.replace(nature_of_business, "")
        
    doc_text = "\n".join(page.text for page in document.pages)
    expected_policies = _infer_expected_policies(nature_of_business, doc_text)
    industry_context_str = f"Nature of business mentions {nature_of_business[:60]}..." if nature_of_business else "Nature of business not clearly detected."
    
    # Generate an industry comment if we found one
    if expected_policies:
        export_rows.append({
            "Paragraph reviewed": "[Industry Summary]",
            "Standard mentioned": "N/A",
            "Expected standard topic": ", ".join(expected_policies),
            "Industry alignment": industry_context_str,
            "Comment": f"Based on nature of business, expected policy areas include: {', '.join(expected_policies)}.",
            "Suggested correction if needed": "",
            "Review status": "Observed",
        })

    # Map the 10 specific policy topics!
    REQUIRED_TOPICS = {
        "Basis of preparation": ["basis of preparation", "statement of compliance", "going concern", "historical cost", "accounting convention"],
        "Significant judgements and estimates": ["judgement", "judgment", "estimate", "assumption", "uncertaint"],
        "PPE": ["property, plant", "ppe", "depreciation", "useful li", "residual value"],
        "Intangible assets": ["intangible", "amortisation", "software", "development cost", "goodwill"],
        "Financial instruments": ["financial instrument", "financial asset", "financial liabilit", "amortised cost", "fvtpl", "fvoci"],
        "Receivables / ECL": ["receivable", "expected credit loss", "ecl", "impairment", "credit risk", "provision matrix", "loss allowance"],
        "Tax": ["taxation", "income tax", "deferred tax", "current tax", "tax expense"],
        "Revenue": ["revenue", "performance obligation", "contract with customer", "sale of goods", "rendering of services"],
        "Contract liabilities": ["contract liabilit", "deferred revenue", "advance", "unearned"],
        "Cash and cash equivalents": ["cash and cash", "cash equivalent", "bank balance", "short-term deposit"],
        "Standards and interpretations": ["new standards", "standards and interpretations", "not yet effective", "effective and adopted", "amendments to", "interpretations effective"],
    }
    
    # Check paragraphs against topics
    combined_text = "\n\n".join(part for part in (note_1, note_2) if part.strip()).strip() or combined_text
    policy_candidates = [
        *(_policy_chunks(note_1, "1") if note_1 else []),
        *(_policy_chunks(note_2, "2") if note_2 else []),
    ]
    if not policy_candidates:
        policy_candidates = [p.strip() for p in re.split(r'\n\s*\n', combined_text) if p.strip()]
    paragraphs = [p for p in policy_candidates if _policy_paragraph_usable(p)]
    found_topics = set()
    topic_matches: dict[str, tuple[int, str, str]] = {}
    
    for para in paragraphs:
        if len(para) < 30:
            continue
            
        para_lower = para.lower()
        matched_topics = []
        for topic_name, keywords in REQUIRED_TOPICS.items():
            if any(k in para_lower for k in keywords):
                matched_topics.append(topic_name)
                found_topics.add(topic_name)
                
        mentioned_standards = []
        matches = re.finditer(r"\b(IFRS|IAS)\s+(\d+[A-Z]?)\b", para, flags=re.I)
        for m in matches:
            mentioned_standards.append(m.group(0).upper())
            
        mentioned_standards = list(dict.fromkeys(mentioned_standards))
        standards_str = ", ".join(mentioned_standards) if mentioned_standards else "None specifically cited"
        
        for topic in matched_topics:
            keyword_score = _policy_match_score(para_lower, REQUIRED_TOPICS[topic])
            existing = topic_matches.get(topic)
            preview = _preview_around_keywords(para, REQUIRED_TOPICS[topic])
            if existing is None or keyword_score > existing[0]:
                topic_matches[topic] = (keyword_score, preview, standards_str)

    for topic in REQUIRED_TOPICS:
        match = topic_matches.get(topic)
        if not match:
            continue
        _score, preview, standards_str = match
        export_rows.append({
            "Paragraph reviewed": preview,
            "Standard mentioned": standards_str,
            "Expected standard topic": topic,
            "Industry alignment": "Appears relevant",
            "Comment": f"Paragraph addresses {topic}.",
            "Suggested correction if needed": "",
            "Review status": "Observed",
        })

    # Output missing core topics
    for topic in REQUIRED_TOPICS:
        mapped_policy = TOPIC_POLICY_MAP.get(topic)
        if mapped_policy and policy_map.get(mapped_policy):
            continue
        if topic in found_topics:
            continue
        if not _topic_expected(topic, expected_policies, doc_text, nature_of_business):
            continue
        suggested = f"Consider adding a policy paragraph for {topic} if applicable."
        comment = f"[Missing policy] The core policy '{topic}' was not clearly detected in Note 1 or 2."
        export_rows.append({
            "Paragraph reviewed": "[MISSING POLICY]",
            "Standard mentioned": "N/A",
            "Expected standard topic": topic,
            "Industry alignment": "Missing expected policy",
            "Comment": comment,
            "Suggested correction if needed": suggested,
            "Review status": _row_status(comment, suggested),
        })
            
    return findings, export_rows
