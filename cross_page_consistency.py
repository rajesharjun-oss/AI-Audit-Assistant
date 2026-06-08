import re
import difflib
from collections import defaultdict
from decimal import Decimal
from models import PdfDocument, Finding

# 1. Key Amounts Consistency
KEY_METRICS = {
    "Revenue": re.compile(r"^(?:Revenue|Turnover|Gross Earnings)\b(?!.*contract liabilit)", re.I),
    "Profit before tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?\s+before\s+tax(?:ation)?\b", re.I),
    "Taxation": re.compile(r"^(?:Taxation|Income tax expense)\b", re.I),
    "Profit after tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?(?:\s+after\s+tax(?:ation)?)?(?:\s+for\s+the\s+(?:year|period))?\b(?!.*loss allowance)(?!.*loss on foreign)(?!.*loss carried forward)", re.I),
    "Total comprehensive income": re.compile(r"^Total\s+comprehensive\s+(?:income|loss)\b", re.I)
}

# 2. Dates
DATE_FORMAT_1_RE = re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b")
DATE_FORMAT_2_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b")

def check_cross_page_consistency(document: PdfDocument) -> tuple[list[Finding], dict[str, list[dict[str, str]]]]:
    findings = []
    export_data = {
        "key_amounts": [],
        "names": [],
        "dates": []
    }
    
    amount_occurrences = defaultdict(list)
    date_occurrences = defaultdict(list)
    name_candidates = []
    
    # Simple NER for names (Capitalized words, 2-4 words)
    NAME_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b")
    
    for page in document.pages:
        # Avoid extracting dates/names from purely legal/standard texts if possible
        text = page.text
        
        # Extract dates
        for match in DATE_FORMAT_1_RE.finditer(text):
            date_occurrences[match.group(0)].append(page.number)
        for match in DATE_FORMAT_2_RE.finditer(text):
            date_occurrences[match.group(0)].append(page.number)
            
        # Extract potential names in signature blocks or directors lists
        if any(kw in text.lower() for kw in ("director", "secretary", "chief executive", "officer", "auditor")):
            for match in NAME_RE.finditer(text):
                name = match.group(0)
                stop_words = [
                    "annual report", "financial statement", "statement of", "notes to", "cash flow", "value added", 
                    "the company", "limited", "bank", "plc", "kpmg", "pwc", "deloitte", "ernst", "kreston", "pedabo", 
                    "audit", "services", "ifrs", "ias", "board of", "directors", "report of", "committee", "chairman", 
                    "secretary", "executive", "officer", "accounting", "policy", "policies", "standards", "international", 
                    "reporting", "corporate", "governance", "independent", "opinion", "basis for", "key audit", "matters", 
                    "other information", "responsibilities of", "consolidated", "separate", "comprehensive income", 
                    "financial position", "changes in equity", "general information", "address", "registered office", 
                    "principal place", "business", "nature of", "for the year", "ended", "december", "january"
                ]
                if not any(stop in name.lower() for stop in stop_words):
                    name_candidates.append((name, page.number))

        for line in text.splitlines():
            line = line.strip()
            # Try to match key metrics
            for metric_name, pattern in KEY_METRICS.items():
                if pattern.match(line):
                    raw_amounts = re.findall(r"\(?-?\d[\d,\.]*\)?", line)
                    amounts = []
                    for a in raw_amounts:
                        clean = a.replace(",", "").replace("(", "").replace(")", "").replace("-", "")
                        if not clean: continue
                        if "." in clean and float(clean) < 100: continue
                        if len(clean) <= 2 and clean != "0": continue
                        amounts.append(a)
                    if amounts:
                        amt_str = amounts[0]
                        clean_amt = amt_str.replace(",", "").replace("(", "-").replace(")", "")
                        try:
                            val = Decimal(clean_amt)
                            amount_occurrences[metric_name].append((val, page.number, line))
                        except Exception:
                            pass

    # Process amounts
    for metric_name, occurrences in amount_occurrences.items():
        if not occurrences:
            continue
        val_map = defaultdict(list)
        for val, page_num, line in occurrences:
            val_map[val].append((page_num, line))
        
        # If multiple different values found for the same metric
        if len(val_map) > 1:
            desc = []
            for val, locs in val_map.items():
                pages = [str(p) for p, l in locs]
                desc.append(f"Amount {val} found on pages: {', '.join(pages)}")
                for p, l in locs:
                    export_data["key_amounts"].append({
                        "Metric": metric_name,
                        "Amount": str(val),
                        "Page": str(p),
                        "Context": l,
                        "Issue": "Discrepancy"
                    })
            
            findings.append(
                Finding(
                    "Consistency",
                    "Medium",
                    f"Inconsistent {metric_name}",
                    f"The amount for {metric_name} varies across pages.",
                    " | ".join(desc),
                    "Verify the correct amount across the directors' report, primary statements, and notes."
                )
            )
        else:
            # Consistent
            val, locs = list(val_map.items())[0]
            for p, l in locs:
                export_data["key_amounts"].append({
                    "Metric": metric_name,
                    "Amount": str(val),
                    "Page": str(p),
                    "Context": l,
                    "Issue": "Consistent"
                })

    # Process dates
    expected_format = "31 December 2025 (DD Month YYYY)"
    preferred_re = DATE_FORMAT_1_RE

    for date_str, pages in date_occurrences.items():
        # Only flag date formatting if the text around it indicates reporting or signing context
        context_text = " ".join([page.text for p in document.pages if p.number in pages]).lower()
        if any(kw in context_text for kw in ("ended", "signed", "dated", "approved", "as at")):
            if not preferred_re.match(date_str):
                export_data["dates"].append({
                    "Date found": date_str,
                    "Page": ", ".join(map(str, set(pages))),
                    "Expected format": expected_format,
                    "Comment": "Inconsistent date format."
                })
                findings.append(
                    Finding(
                        "Formatting",
                        "Low",
                        "Inconsistent Date Format",
                        f"Date '{date_str}' does not match the preferred format.",
                        f"Found on pages: {', '.join(map(str, set(pages)))}",
                        f"Update to match the predominant format ({expected_format})."
                    )
                )

    # Process names using sequence matcher
    unique_names = defaultdict(list)
    for name, page in name_candidates:
        unique_names[name].append(page)
        
    names_list = list(unique_names.keys())
    flagged_pairs = set()
    
    for i, name1 in enumerate(names_list):
        for j in range(i + 1, len(names_list)):
            name2 = names_list[j]
            # Must be similar but not exact, and share at least one word
            # Check ratio
            ratio = difflib.SequenceMatcher(None, name1, name2).ratio()
            # Also check if it's the same words reversed e.g. "Taiwo Olasore" vs "Olasore Taiwo"
            set1 = set(name1.split())
            set2 = set(name2.split())
            common_words = set1.intersection(set2)
            
            if (ratio > 0.85 or (len(common_words) >= 2 and len(set1) == len(set2))) and name1 != name2:
                pair_key = tuple(sorted([name1, name2]))
                if pair_key not in flagged_pairs:
                    flagged_pairs.add(pair_key)
                    pages1 = ", ".join(map(str, set(unique_names[name1])))
                    pages2 = ", ".join(map(str, set(unique_names[name2])))
                    export_data["names"].append({
                        "Name variant 1": name1,
                        "Page 1": pages1,
                        "Name variant 2": name2,
                        "Page 2": pages2,
                        "Suggested standard spelling": max(name1, name2, key=len),
                        "Comment": "Names appear to refer to the same person but are spelt differently."
                    })
                    findings.append(
                        Finding(
                            "Consistency",
                            "Low",
                            "Inconsistent Name Spelling",
                            f"Name spelt differently across pages: '{name1}' vs '{name2}'",
                            f"Variant 1 on pages {pages1}, Variant 2 on pages {pages2}",
                            "Standardize the spelling of the name across all reports and signatures."
                        )
                    )

    return findings, export_data
