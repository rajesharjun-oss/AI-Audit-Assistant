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
    match = re.search(r"(?:nature of business|principal activities?|principal business)(?:[\s\S]{1,300})", text_lower)
    if match:
        return match.group(0).strip()
    
    # Check whole document first 15 pages
    doc_text_lower = "\n".join(p.text.lower() for p in document.pages[:15])
    match = re.search(r"(?:nature of business|principal activities?)(?:[\s\S]{1,300})", doc_text_lower)
    if match:
        return match.group(0).strip()
        
    return ""

def _infer_expected_policies(nature_text: str) -> list[str]:
    policies = []
    nature_text = nature_text.lower()
    if any(w in nature_text for w in ["sell", "sale", "retail", "wholesale", "goods", "trading", "consumer"]):
        policies.extend(["revenue from customer contracts", "inventory", "trade receivables and ECL"])
    if any(w in nature_text for w in ["software", "technology", "platform", "app ", "digital", "loyalty"]):
        policies.extend(["revenue from customer contracts", "contract liabilities", "intangible assets/software"])
    if any(w in nature_text for w in ["manufactur", "production", "plant", "factory"]):
        policies.extend(["property, plant and equipment", "inventory", "revenue from customer contracts"])
    if any(w in nature_text for w in ["bank", "financ", "lend", "loan", "credit", "invest"]):
        policies.extend(["financial instruments", "ECL", "fair value measurement"])
    return list(set(policies))

def review_notes_1_and_2(document: PdfDocument, profile: CompanyProfile, note_sections: dict[str, str]) -> tuple[list[Finding], list[dict[str, str]]]:
    findings = []
    export_rows = []
    
    note_1 = note_sections.get("1", "")
    note_2 = note_sections.get("2", "")
    
    combined_text = (note_1 + "\n\n" + note_2).strip()
    if not combined_text:
        return findings, export_rows
        
    nature_of_business = _extract_nature_of_business(document, combined_text)
    expected_policies = _infer_expected_policies(nature_of_business)
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

    # Split into rough paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', combined_text) if p.strip()]
    
    for para in paragraphs:
        if len(para) < 30:
            continue
            
        mentioned_standards = []
        matches = re.finditer(r"\b(IFRS|IAS)\s+(\d+[A-Z]?)\b", para, flags=re.I)
        for m in matches:
            mentioned_standards.append(m.group(0).upper())
            
        mentioned_standards = list(dict.fromkeys(mentioned_standards))
        
        for std in mentioned_standards:
            expected_topics = STANDARD_TOPICS.get(std, [])
            if not expected_topics:
                # Standard mentioned but we don't have expected topics mapped
                export_rows.append({
                    "Paragraph reviewed": para[:200] + ("..." if len(para) > 200 else ""),
                    "Standard mentioned": std,
                    "Expected standard topic": "N/A",
                    "Industry alignment": "N/A",
                    "Comment": f"{std} mentioned, no strict topic alignment enforced.",
                    "Suggested correction if needed": ""
                })
                continue
                
            para_lower = para.lower()
            aligned = any(topic in para_lower for topic in expected_topics)
            
            if aligned:
                comment = f"Paragraph aligns with {std} expected topics."
                correction = ""
            else:
                comment = f"Paragraph mentions {std} but does not discuss expected topics like {expected_topics[0]}."
                correction = f"Align the policy wording with {std} by discussing {', '.join(expected_topics[:3])}."
                
                findings.append(Finding(
                    "Accounting policies",
                    "Medium",
                    f"Note 1/2 Policy: {std}",
                    f"Policy mentions {std} but lacks expected specific topics.",
                    para[:150] + "...",
                    correction
                ))
                
            export_rows.append({
                "Paragraph reviewed": para[:200] + ("..." if len(para) > 200 else ""),
                "Standard mentioned": std,
                "Expected standard topic": ", ".join(expected_topics),
                "Industry alignment": "Appears relevant" if aligned else "Possible boilerplate",
                "Comment": comment,
                "Suggested correction if needed": correction
            })
            
    return findings, export_rows
