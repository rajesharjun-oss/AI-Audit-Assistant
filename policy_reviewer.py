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

def review_notes_1_and_2(document: PdfDocument, profile: CompanyProfile, note_sections: dict[str, str]) -> tuple[list[Finding], list[dict[str, str]]]:
    findings = []
    export_rows = []
    
    note_1 = note_sections.get("1", "")
    note_2 = note_sections.get("2", "")
    
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
            "Suggested correction if needed": ""
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
        "Cash and cash equivalents": ["cash and cash", "cash equivalent", "bank balance", "short-term deposit"]
    }
    
    # Check paragraphs against topics
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', combined_text) if p.strip()]
    found_topics = set()
    
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
        
        if not matched_topics:
            export_rows.append({
                "Paragraph reviewed": para[:200] + ("..." if len(para) > 200 else ""),
                "Standard mentioned": standards_str,
                "Expected standard topic": "N/A",
                "Industry alignment": "Possible boilerplate / other topic",
                "Comment": "Reviewed but did not map strongly to the core 10 policy areas.",
                "Suggested correction if needed": ""
            })
            continue
            
        for topic in matched_topics:
            export_rows.append({
                "Paragraph reviewed": para[:200] + ("..." if len(para) > 200 else ""),
                "Standard mentioned": standards_str,
                "Expected standard topic": topic,
                "Industry alignment": "Appears relevant",
                "Comment": f"Paragraph addresses {topic}.",
                "Suggested correction if needed": ""
            })

    # Output missing core topics
    for topic in REQUIRED_TOPICS:
        if topic not in found_topics:
            export_rows.append({
                "Paragraph reviewed": "[MISSING POLICY]",
                "Standard mentioned": "N/A",
                "Expected standard topic": topic,
                "Industry alignment": "Missing core policy",
                "Comment": f"The core policy '{topic}' was not clearly detected in Note 1 or 2.",
                "Suggested correction if needed": f"Consider adding a policy paragraph for {topic} if applicable."
            })
            
    return findings, export_rows
