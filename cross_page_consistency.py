import re
import difflib
from collections import defaultdict
from decimal import Decimal
from models import PdfDocument, Finding

# 1. Key Amounts Consistency
KEY_METRICS = {
    "Revenue": re.compile(r"^(?:Revenue|Turnover|Gross Earnings)\b", re.I),
    "Profit before tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?\s+before\s+tax(?:ation)?\b", re.I),
    "Taxation": re.compile(r"^(?:Taxation|Income tax expense)\b", re.I),
    "Profit after tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?(?:\s+after\s+tax(?:ation)?|\s+for\s+the\s+(?:year|period))\b", re.I),
    "Total comprehensive income": re.compile(r"^Total\s+comprehensive\s+(?:income|loss)\b", re.I)
}

# 2. Dates
DATE_FORMAT_1_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I)
DATE_FORMAT_2_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+20\d{2}\b", re.I)
DATE_FORMAT_3_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I)


def _date_format_context_requires_standardisation(line: str) -> bool:
    lower = line.lower()
    excluded = (
        "incorporated",
        "commenced",
        "legal framework",
        "pending legal",
        "litigation",
        "contingenc",
        "tax rate",
        "effective",
        "adopt",
        "amendment",
        "standard",
        "ifrs",
        "ias",
    )
    if any(term in lower for term in excluded):
        return False
    required = (
        "year ended",
        "financial statements for the year ended",
        "approved",
        "signed",
        "dated",
        "date of approval",
        "as at",
        "reporting period",
    )
    return any(term in lower for term in required)


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
        
        # Extract dates with line-level context. Page-level context creates false positives because most pages
        # carry a repeating "year ended" header.
        for line in text.splitlines():
            for match in DATE_FORMAT_1_RE.finditer(line):
                date_occurrences[match.group(0)].append((page.number, line.strip()))
            for match in DATE_FORMAT_2_RE.finditer(line):
                date_occurrences[match.group(0)].append((page.number, line.strip()))
            for match in DATE_FORMAT_3_RE.finditer(line):
                # Only add if it wasn't already matched as part of FORMAT 1 or 2
                if not any(match.group(0) in d for d in date_occurrences.keys()):
                    date_occurrences[match.group(0)].append((page.number, line.strip()))
            
        # Extract potential names in signature blocks or directors lists
        page_lower = text.lower()
        target_page_keywords = [
            "corporate information", "general information", "directors' report", "directors report",
            "directors' responsibility", "directors responsibility", "management certification",
            "independent auditor", "shareholding", "directors' interest", "directors interest"
        ]
        is_target_page = any(kw in page_lower for kw in target_page_keywords)
        
        if is_target_page:
            chunks = re.split(r'\n|\s{2,}', text)
            for chunk in chunks:
                for match in NAME_RE.finditer(chunk):
                    raw_name = match.group(0)

                    exclude_words = [
                        "financial", "financials", "group", "instruments", "instrument", "statement", "summary", "years", "year",
                        "nigeria", "appointed", "resigned", "monday", "company", "limited", "plc",
                        "bank", "administrator", "administrators", "admistrator", "standard", "sacks", "sack", "property",
                        "revenue", "income", "expense", "equity", "assets", "liabilities", "note", "pension", "fund",
                        "tax", "ifrs", "ias", "frc", "audit", "services", "report", "accounting",
                        "policy", "policies", "standards", "international", "reporting", "corporate",
                        "governance", "independent", "opinion", "basis", "key", "matters", "other",
                        "consolidated", "separate", "comprehensive", "position", "changes", "december",
                        "january", "street", "road", "cost", "accumulated", "carrying", "pay",
                        "employees", "government", "tuesday", "wednesday", "thursday", "friday",
                        "saturday", "sunday",
                        "opening", "additions", "depreciation", "total", "value", "distributed",
                        "balance", "at", "as", "for", "the", "ended", "loss", "profit",
                        "net", "gross", "operating", "cash", "flows", "financing", "investing", "activities"
                    ]
                    if any(re.search(fr"\b{ex}\b", raw_name, re.I) for ex in exclude_words):
                        continue

                    remove_titles = [
                        "group managing director", "chief financial officer", "managing director",
                        "non-executive director", "executive director", "signing partner",
                        "non-executive", "executive", "chairman", "director", "directors",
                        "secretary", "chief", "officer", "managing", "manager", "committee",
                        "board", "mr", "mrs", "dr", "sir", "non", "appointed", "resigned",
                        "nigeria", "monday", "frc", "pro", "ican", "form"
                    ]

                    clean_name = raw_name
                    for title in remove_titles:
                        clean_name = re.sub(fr"(?i)\b{title}\b", " ", clean_name)

                    clean_name = re.sub(r"\s+", " ", clean_name).strip()

                    tokens = clean_name.split()
                    # Reject if more than 4 tokens (multi-person strings)
                    if len(tokens) > 4:
                        continue
                    if 2 <= len(tokens) <= 4 and all(t[0].isupper() for t in tokens if t.isalpha()):
                        name_candidates.append((clean_name, page.number))

        for line in text.splitlines():
            line = line.strip()
            # Try to match key metrics
            for metric_name, pattern in KEY_METRICS.items():
                if pattern.match(line):
                    lower = line.lower()
                    if any(sw in lower for sw in ["loss allowance", "loss on foreign", "loss carried forward", "revenue contract", "contract liabilit"]):
                        continue
                    raw_amounts = re.findall(r"\(?-?\d[\d,\.]*\)?", line)
                    amounts = []
                    for a in raw_amounts:
                        clean = a.replace(",", "").replace("(", "").replace(")", "").replace("-", "")
                        if not clean: continue
                        if "." in clean and float(clean) < 100: continue
                        if len(clean) <= 2 and clean != "0": continue # Exclude note numbers
                        if len(clean) <= 4 and clean.startswith("20"): continue # Exclude years
                        amounts.append(a)
                    if amounts:
                        amt_str = amounts[0]
                        clean_amt = amt_str.replace(",", "").replace("(", "-").replace(")", "")
                        
                        prior_str = amounts[1] if len(amounts) >= 2 else None
                        prior_amt = None
                        if prior_str:
                            try:
                                prior_amt = Decimal(prior_str.replace(",", "").replace("(", "-").replace(")", ""))
                            except Exception:
                                pass
                        
                        try:
                            val = Decimal(clean_amt)
                            amount_occurrences[metric_name].append((val, prior_amt, page.number, line))
                        except Exception:
                            pass

    # Process amounts
    for metric_name, occurrences in amount_occurrences.items():
        if not occurrences:
            continue
        val_map = defaultdict(list)
        all_prior_amts = set()
        for val, prior_amt, page_num, line in occurrences:
            val_map[val].append((page_num, line))
            if prior_amt is not None:
                all_prior_amts.add(prior_amt)
        
        # If multiple different values found for the same metric
        val_keys = [k for k in val_map.keys() if k not in all_prior_amts]
        if len(val_keys) > 1:
            desc = []
            issue_pages = set()
            for val, locs in val_map.items():
                pages = [str(p) for p, _line in locs]
                issue_pages.update(p for p, _line in locs)
                desc.append(f"Amount {val:,.0f} found on pages: {', '.join(sorted(set(pages), key=int))}")
                for p, l in locs:
                    export_data["key_amounts"].append({
                        "Metric": metric_name,
                        "Amount": f"{val:,.0f}",
                        "Page": str(p),
                        "Context": _short_context(l),
                        "Issue": "Discrepancy"
                    })
            
            findings.append(
                Finding(
                    "Consistency",
                    "Medium",
                    _format_page_location(issue_pages),
                    f"The amount for {metric_name} varies across pages.",
                    " | ".join(desc),
                    "Verify the correct amount across the directors' report, primary statements, and notes."
                )
            )
        else:
            # Consistent
            val, locs = list(val_map.items())[0]
            pages = sorted({p for p, _line in locs})
            export_data["key_amounts"].append({
                "Metric": metric_name,
                "Amount": f"{val:,.0f}",
                "Pages checked": _format_page_location(pages),
                "Context": "Consistent across detected occurrences.",
                "Issue": "Consistent"
            })

    # Process dates
    expected_format = "31 December 2025 (DD Month YYYY)"
    preferred_re = re.compile(r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I)

    for date_str, occurrences in date_occurrences.items():
        pages = [page for page, _line in occurrences]
        relevant_contexts = [
            (page, line)
            for page, line in occurrences
            if _date_format_context_requires_standardisation(line)
        ]
        if relevant_contexts:
            if not preferred_re.match(date_str):
                relevant_pages = [page for page, _line in relevant_contexts]
                export_data["dates"].append({
                    "Date found": date_str,
                    "Page": ", ".join(map(str, sorted(set(relevant_pages)))),
                    "Expected format": expected_format,
                    "Comment": "Inconsistent date format."
                })
                findings.append(
                    Finding(
                        "Formatting",
                        "Low",
                        _format_page_location(relevant_pages),
                        f"Date '{date_str}' does not match the preferred format.",
                        f"Found on pages: {', '.join(map(str, sorted(set(relevant_pages))))}",
                        f"Update to match the predominant format ({expected_format})."
                    )
                )
            else:
                export_data["dates"].append({
                    "Date found": date_str,
                    "Page": ", ".join(map(str, sorted(set(pages)))),
                    "Expected format": expected_format,
                    "Comment": "Consistent."
                })

    # Process names using sequence matcher
    unique_names = defaultdict(list)
    for name, page in name_candidates:
        unique_names[name].append(page)
        
    names_list = list(unique_names.keys())
    flagged_pairs = set()
    
    for i, name1 in enumerate(names_list):
        for j in range(i + 1, len(names_list)):
            name2 = names_list[j]
            is_match = _names_look_like_spelling_variants(name1, name2)
                
            if is_match and name1 != name2:
                pair_key = tuple(sorted([name1, name2]))
                if pair_key not in flagged_pairs:
                    pages1_set = set(unique_names[name1])
                    pages2_set = set(unique_names[name2])
                    if _likely_ocr_name_artifact(name1, pages1_set, name2, pages2_set):
                        continue
                    flagged_pairs.add(pair_key)
                    pages1 = ", ".join(map(str, sorted(pages1_set)))
                    pages2 = ", ".join(map(str, sorted(pages2_set)))
                    standard = _suggest_standard_name(name1, pages1_set, name2, pages2_set)
                    export_data["names"].append({
                        "Name variant 1": name1,
                        "Page 1": pages1,
                        "Name variant 2": name2,
                        "Page 2": pages2,
                        "Suggested standard spelling": standard,
                        "Reason": "Names appear to refer to the same person but are spelt differently.",
                        "Confidence": "High"
                    })
                    findings.append(
                        Finding(
                            "Consistency",
                            "Low",
                            _format_page_location(set(unique_names[name1]) | set(unique_names[name2])),
                            f"Name spelt differently across pages: '{name1}' vs '{name2}'",
                            f"Variant 1 on pages {pages1}, Variant 2 on pages {pages2}",
                            "Standardize the spelling of the name across all reports and signatures."
                        )
                    )

    return findings, export_data


def _short_context(line: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(line or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _likely_ocr_name_artifact(name1: str, pages1: set[int], name2: str, pages2: set[int]) -> bool:
    if pages1 == pages2 and len(pages1) == 1:
        return True
    if _single_page_token_artifact(name1, pages1, name2, pages2):
        return True
    count1 = len(pages1)
    count2 = len(pages2)
    if count1 == count2:
        return False
    common_pages = pages1 & pages2
    if not common_pages:
        return False
    rare_name, rare_pages, common_name, common_pages_set = (
        (name1, pages1, name2, pages2) if count1 < count2 else (name2, pages2, name1, pages1)
    )
    if len(rare_pages) > 1 or len(common_pages_set) < 2:
        return False
    return _names_look_like_spelling_variants(rare_name, common_name)


def _single_page_token_artifact(name1: str, pages1: set[int], name2: str, pages2: set[int]) -> bool:
    if not (pages1 & pages2):
        return False
    rare_name, rare_pages, common_name, common_pages = (
        (name1, pages1, name2, pages2) if len(pages1) < len(pages2) else (name2, pages2, name1, pages1)
    )
    if len(rare_pages) != 1 or len(common_pages) < 2:
        return False
    rare_tokens = _normalise_name_tokens(rare_name)
    common_tokens = _normalise_name_tokens(common_name)
    if len(rare_tokens) != len(common_tokens) or len(rare_tokens) < 2:
        return False
    differing_pairs = [(left, right) for left, right in zip(rare_tokens, common_tokens) if left != right]
    if len(differing_pairs) != 1:
        return False
    left, right = differing_pairs[0]
    if left[:1] != right[:1]:
        return False
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return len(longer) - len(shorter) == 1 and longer.startswith(shorter)


def _suggest_standard_name(name1: str, pages1: set[int], name2: str, pages2: set[int]) -> str:
    if len(pages1) != len(pages2):
        return name1 if len(pages1) > len(pages2) else name2
    return min((name1, name2), key=lambda name: (len(name), name))


def _names_look_like_spelling_variants(name1: str, name2: str) -> bool:
    tokens1 = _normalise_name_tokens(name1)
    tokens2 = _normalise_name_tokens(name2)
    if len(tokens1) != len(tokens2) or len(tokens1) < 2:
        return False
    if tokens1 == tokens2:
        return False
    if set(tokens1) == set(tokens2):
        return False

    exact = 0
    fuzzy = 0
    for left, right in zip(tokens1, tokens2):
        if left == right:
            exact += 1
            continue
        if left[:1] == right[:1] and difflib.SequenceMatcher(None, left, right).ratio() >= 0.84:
            fuzzy += 1
            continue
        return False
    return exact >= 1 and fuzzy == 1


def _normalise_name_tokens(name: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+", name)]


def _format_page_location(pages) -> str:
    clean_pages = sorted({int(page) for page in pages if str(page).isdigit() or isinstance(page, int)})
    if not clean_pages:
        return "Document-wide"
    if len(clean_pages) == 1:
        return f"Page {clean_pages[0]}"
    return "Pages " + ", ".join(str(page) for page in clean_pages)
