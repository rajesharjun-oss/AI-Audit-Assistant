import re
import difflib
from collections import defaultdict
from decimal import Decimal
from models import PdfDocument, Finding

# 1. Key Amounts Consistency
KEY_METRICS = {
    "Revenue": re.compile(r"^(?:['‘’\"]\s*)?(?:\d+[A-Za-z]?[.)]?\s+)?(?:Revenue|Turnover|Gross Earnings)\b", re.I),
    "Profit before tax": re.compile(r"^(?:['‘’\"]\s*)?(?:\d+[A-Za-z]?[.)]?\s+)?(?:Profit|Loss)(?:\/\(loss\))?\s+before\s+tax(?:ation)?\b", re.I),
    "Taxation": re.compile(r"^(?:['‘’\"]\s*)?(?:\d+[A-Za-z]?[.)]?\s+)?(?:Taxation|Income tax expense)\b", re.I),
    "Profit after tax": re.compile(r"^(?:['‘’\"]\s*)?(?:\d+[A-Za-z]?[.)]?\s+)?(?:Profit|Loss)(?:\/\(loss\))?(?:\s+after\s+tax(?:ation)?|\s+for\s+the\s+(?:year|period))\b", re.I),
    "Total comprehensive income": re.compile(r"^(?:['‘’\"]\s*)?(?:\d+[A-Za-z]?[.)]?\s+)?Total\s+comprehensive\s+(?:income|loss)\b", re.I)
}

# 2. Dates
DATE_FORMAT_1_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I)
DATE_FORMAT_2_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+20\d{2}\b", re.I)
DATE_FORMAT_3_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b", re.I)
REPEATED_WORD_RE = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I)
MISSING_SPACE_AFTER_PUNCT_RE = re.compile(r"(?<=[A-Za-z])([,;:])(?=[A-Za-z])|(?<=[A-Za-z])\.(?=[A-Za-z][a-z])")
COMMON_SPELLING_CORRECTIONS = {
    "teh": "the",
    "statment": "statement",
    "statments": "statements",
    "finacial": "financial",
    "managment": "management",
    "goverance": "governance",
    "occurence": "occurrence",
    "seperate": "separate",
    "comittee": "committee",
    "subsiduary": "subsidiary",
    "equiptment": "equipment",
    "deffered": "deferred",
    "depreciaton": "depreciation",
    "ammortisation": "amortisation",
    "ammortization": "amortization",
    "intengible": "intangible",
    "recievable": "receivable",
    "recievables": "receivables",
    "liabilty": "liability",
    "liabilties": "liabilities",
    "reconcilliation": "reconciliation",
    "remuneraton": "remuneration",
    "busines": "business",
    "orgnisation": "organisation",
    "orgnaisation": "organisation",
    "acheive": "achieve",
    "goverment": "government",
    "contrat": "contract",
    "relevent": "relevant",
}


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


def _normalise_preferred_date_format(date_str: str) -> str:
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", date_str.strip(), flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned.replace(",", " ")).strip()
    month_first = re.match(r"^([A-Za-z]+)\s+(\d{1,2})\s+(20\d{2})$", cleaned)
    if month_first:
        month, day, year = month_first.groups()
        return f"{month} {int(day)} {year}"
    day_first = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})$", cleaned)
    if day_first:
        day, month, year = day_first.groups()
        return f"{month} {int(day)} {year}"
    return cleaned


def _looks_like_grammar_review_line(line: str) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    lower = clean.lower()
    if len(clean) < 20:
        return False
    if len(re.findall(r"\d", clean)) > 3:
        return False
    excluded = (
        "statement of",
        "notes to the financial statements",
        "n'000",
        "note(s)",
        "year ended",
        "as at",
        "page ",
        "draft",
    )
    if any(marker in lower for marker in excluded):
        return False
    if _looks_like_signature_or_firm_line(clean):
        return False
    words = re.findall(r"[A-Za-z']+", clean)
    if _looks_like_table_header_fragment(clean, words):
        return False
    return len(words) >= 5


def _looks_like_table_header_fragment(line: str, words: list[str]) -> bool:
    lower = line.lower()
    financial_terms = {
        "accumulated", "fund", "funds", "equity", "assets", "liabilities", "liability",
        "land", "building", "buildings", "property", "plant", "equipment", "motor",
        "vehicles", "generators", "library", "books", "cash", "between", "over",
        "less", "total", "cost", "depreciation", "carrying", "value", "note",
    }
    if not words or any(char in line for char in ".,;:"):
        return False
    lowered_words = [word.lower() for word in words]
    if len(lowered_words) >= 4 and sum(1 for word in lowered_words if word in financial_terms) >= 3:
        return True
    if REPEATED_WORD_RE.search(line) and sum(1 for word in lowered_words if word in financial_terms) >= 2:
        return True
    return False


def _looks_like_signature_or_firm_line(line: str) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    lower = clean.lower()
    if re.match(r"^(?:for|per)\s*:", lower) and any(term in lower for term in ("audit", "auditor", "accountant", "services")):
        return True
    if re.match(r"^(?:for|per)\s*:", lower) and len(re.findall(r"[A-Za-z]+", clean)) <= 6:
        return True
    if lower in {"for the board", "on behalf of the board"}:
        return True
    return False


def _grammar_issue_for_line(line: str) -> str:
    if REPEATED_WORD_RE.search(line):
        return "Repeated word detected."
    if MISSING_SPACE_AFTER_PUNCT_RE.search(line):
        return "Possible missing space after punctuation."
    return ""


def _spelling_issue_for_line(line: str) -> str:
    tokens = re.findall(r"[A-Za-z']+", line)
    for index, token in enumerate(tokens):
        normalized = token.lower()
        if len(normalized) < 4:
            continue
        if normalized in {"ifrs", "ias", "naira", "ngn", "usd", "eur", "gbp"}:
            continue
        if index > 0 and token[:1].isupper():
            # likely a name/proper noun in narrative text
            continue
        correction = COMMON_SPELLING_CORRECTIONS.get(normalized)
        if correction:
            return f"Possible spelling error: '{token}' -> '{correction}'."
    return ""


def _metric_page_is_excluded(page_text: str) -> bool:
    lower = page_text.lower()
    excluded_markers = (
        "value added statement",
        "five year financial summary",
        "five-year financial summary",
        "5 year financial summary",
        "financial summary",
    )
    return any(marker in lower for marker in excluded_markers)


def _metric_page_context_not_comparable(metric_name: str, page_text: str) -> bool:
    lower = page_text.lower()
    if metric_name in {"Profit after tax", "Total comprehensive income"} and "statement of changes" in lower:
        return True
    return False


def _metric_line_is_comparable(metric_name: str, line: str) -> bool:
    lower = re.sub(r"\s+", " ", line.lower()).strip()
    if metric_name == "Revenue":
        if re.search(r"[A-Za-z]\d{1,3},\d{3}", line):
            return False
        if "revenue from contracts with customers" in lower and not re.search(r"\brevenue\b\s+\(?-?\d", lower):
            return False
        if any(marker in lower for marker in ("director", "chairman", "management report", "financial highlights", "five year", "five-year")):
            return False
        prefix = re.split(r"\(?-?\d[\d,\.]*\)?", lower, maxsplit=1)[0]
        prefix = re.sub(r"[^a-z ]", " ", prefix)
        prefix_words = [word for word in prefix.split() if word not in {"note", "notes"}]
        if prefix_words[:1] == ["revenue"] and len(prefix_words) <= 2:
            return True
        if prefix_words[:2] == ["total", "revenue"]:
            return True
        if prefix_words[:1] == ["turnover"]:
            return True
        if prefix_words[:2] == ["gross", "earnings"]:
            return True
        if "from contracts with customers" in lower:
            return False
    return True


def _metric_context_priority(metric_name: str, page_text: str, line: str) -> int:
    lower_line = re.sub(r"\s+", " ", line.lower()).strip()
    lower_page = page_text.lower()
    if metric_name == "Revenue":
        if "statement of profit or loss" in lower_page or "statement of income and expenditure" in lower_page:
            return 3
        if "notes to the financial statements" in lower_page:
            return 2
        if any(marker in lower_page for marker in ("directors' report", "management report", "five year", "five-year", "value added")):
            return 0
    return 1


def check_cross_page_consistency(document: PdfDocument) -> tuple[list[Finding], dict[str, list[dict[str, str]]]]:
    findings = []
    export_data = {
        "key_amounts": [],
        "names": [],
        "dates": [],
        "grammar": [],
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
            if _looks_like_grammar_review_line(line):
                grammar_issue = _grammar_issue_for_line(line)
                spelling_issue = _spelling_issue_for_line(line)
                if grammar_issue:
                    export_data["grammar"].append(
                        {
                            "Page": str(page.number),
                            "Issue": grammar_issue,
                            "Context": _short_context(line),
                            "Comment": "Review wording and punctuation.",
                        }
                    )
                    findings.append(
                        Finding(
                            "Formatting",
                            "Low",
                            f"Page {page.number}",
                            "Possible grammatical or drafting issue detected.",
                            f"{grammar_issue} Context: {_short_context(line, 180)}",
                            "Review the sentence for grammar, punctuation, and drafting clarity.",
                        )
                    )
                if spelling_issue:
                    export_data["grammar"].append(
                        {
                            "Page": str(page.number),
                            "Issue": spelling_issue,
                            "Context": _short_context(line),
                            "Comment": "Review spelling and standardise the wording.",
                        }
                    )
                    findings.append(
                        Finding(
                            "Formatting",
                            "Low",
                            f"Page {page.number}",
                            "Possible spelling issue detected.",
                            f"{spelling_issue} Context: {_short_context(line, 180)}",
                            "Review the word choice and correct the spelling if needed.",
                        )
                    )
            
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
                        "net", "gross", "operating", "cash", "flows", "financing", "investing", "activities",
                        "internal", "control", "controls", "sponsoring", "organisation", "organizations",
                        "organization", "committee", "framework", "environment", "social", "governance"
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

        lines = text.splitlines()
        seen_metric_rows: set[tuple[str, int, str]] = set()
        for index, line in enumerate(lines):
            line = line.strip()
            # Try to match key metrics
            for metric_name, pattern in KEY_METRICS.items():
                if pattern.match(line):
                    if _metric_page_is_excluded(text):
                        continue
                    if _metric_page_context_not_comparable(metric_name, text):
                        continue
                    if not _metric_line_is_comparable(metric_name, line):
                        continue
                    if metric_name == "Taxation" and "income tax expense" in line.lower() and not re.search(r"\d", line):
                        continue
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
                            row_key = (metric_name, page.number, f"{val}|{prior_amt}")
                            if row_key not in seen_metric_rows:
                                seen_metric_rows.add(row_key)
                                amount_occurrences[metric_name].append((val, prior_amt, page.number, line, _metric_context_priority(metric_name, text, line)))
                        except Exception:
                            pass
                    else:
                        follow_on = _follow_on_metric_amounts(metric_name, lines, index)
                        if follow_on:
                            val, prior_amt, context_line = follow_on
                            row_key = (metric_name, page.number, f"{val}|{prior_amt}")
                            if row_key not in seen_metric_rows:
                                seen_metric_rows.add(row_key)
                            amount_occurrences[metric_name].append((val, prior_amt, page.number, context_line, _metric_context_priority(metric_name, text, context_line)))

    # Process amounts
    for metric_name, occurrences in amount_occurrences.items():
        if not occurrences:
            continue
        val_map = defaultdict(list)
        all_prior_amts = set()
        for val, prior_amt, page_num, line, priority in occurrences:
            compare_val = _consistency_compare_value(metric_name, val)
            val_map[compare_val].append((page_num, line, val, priority))
            if prior_amt is not None:
                all_prior_amts.add(_consistency_compare_value(metric_name, prior_amt))
        
        val_map = _merge_equivalent_key_amounts(val_map)
        # If multiple different values found for the same metric
        val_keys = [k for k in val_map.keys() if k not in all_prior_amts]
        if len(val_keys) > 1:
            preferred_keys = [
                key for key, locs in val_map.items()
                if any(priority >= 2 for _page, _line, _raw, priority in locs)
            ]
            if preferred_keys:
                val_keys = preferred_keys
        if len(val_keys) > 1:
            desc = []
            issue_pages = set()
            for compare_val, locs in val_map.items():
                if compare_val not in val_keys:
                    continue
                pages = [str(p) for p, _line, _raw, _priority in locs]
                issue_pages.update(p for p, _line, _raw, _priority in locs)
                desc.append(f"Amount {compare_val:,.0f} found on pages: {', '.join(sorted(set(pages), key=int))}")
                for p, l, raw_val, _priority in locs:
                    export_data["key_amounts"].append({
                        "Metric": metric_name,
                        "Amount": f"{raw_val:,.0f}",
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
            selected_key = val_keys[0] if val_keys else next(iter(val_map))
            locs = val_map[selected_key]
            pages = sorted({p for p, _line, _raw, _priority in locs})
            display_val = next((raw for _page, _line, raw, _priority in locs), selected_key)
            simple_direct_context = all(
                KEY_METRICS[metric_name].search(line.strip()) and "->" not in line
                for _page, line, _raw, _priority in locs
            )
            export_data["key_amounts"].append({
                "Metric": metric_name,
                "Amount": f"{display_val:,.0f}",
                "Pages checked": _format_page_location(pages),
                "Context": (
                    "Consistent across detected occurrences."
                    if simple_direct_context
                    else "; ".join(f"Page {page}: {_short_context(line)}" for page, line, _raw, _priority in locs[:4])
                ),
                "Issue": "Consistent"
            })

    # Process dates
    expected_format = "December 31 2025 (Month DD YYYY)"
    preferred_re = re.compile(r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+20\d{2}$", re.I)

    for date_str, occurrences in date_occurrences.items():
        if re.fullmatch(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}", date_str, re.I):
            continue
        pages = [page for page, _line in occurrences]
        relevant_contexts = [
            (page, line)
            for page, line in occurrences
            if _date_format_context_requires_standardisation(line)
        ]
        if relevant_contexts:
            normalized_date = _normalise_preferred_date_format(date_str)
            if not preferred_re.match(normalized_date):
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
                    "Date found": normalized_date,
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


def _merge_equivalent_key_amounts(val_map: dict[Decimal, list[tuple[int, str, Decimal, int]]]) -> dict[Decimal, list[tuple[int, str, Decimal, int]]]:
    if len(val_map) <= 1:
        return val_map
    sorted_keys = sorted(val_map)
    merged: dict[Decimal, list[tuple[int, str, Decimal, int]]] = {}
    consumed: set[Decimal] = set()
    tolerance = Decimal("1")
    for key in sorted_keys:
        if key in consumed:
            continue
        group = [other for other in sorted_keys if other not in consumed and abs(other - key) <= tolerance]
        canonical = max(group, key=lambda item: len(val_map[item]))
        rows: list[tuple[int, str, Decimal, int]] = []
        for item in group:
            rows.extend(val_map[item])
            consumed.add(item)
        merged[canonical] = rows
    return merged


def _short_context(line: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(line or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _consistency_compare_value(metric_name: str, value: Decimal) -> Decimal:
    if metric_name == "Taxation":
        return abs(value)
    return value


def _follow_on_metric_amounts(
    metric_name: str,
    lines: list[str],
    start_index: int,
) -> tuple[Decimal, Decimal | None, str] | None:
    if metric_name not in {"Taxation", "Revenue", "Profit before tax", "Profit after tax", "Total comprehensive income"}:
        return None
    candidates: list[tuple[Decimal, Decimal | None, str]] = []
    for offset in range(1, 20):
        if start_index + offset >= len(lines):
            break
        candidate = re.sub(r"\s+", " ", lines[start_index + offset].strip())
        if not candidate:
            continue
        candidate_lower = candidate.lower()
        if metric_name == "Taxation" and any(marker in candidate_lower for marker in ("reconciliation", "accounting profit", "tax effect of adjustments")):
            break
        if any(pattern.match(candidate) for pattern in KEY_METRICS.values()):
            break
        if re.match(r"^\d{1,2}\.\s", candidate):
            break
        amounts = re.findall(r"\(?-?\d[\d,]*\)?", candidate)
        parsed: list[Decimal] = []
        for amount in amounts:
            clean = amount.replace(",", "").replace("(", "-").replace(")", "")
            if len(clean.strip("-")) <= 2:
                continue
            if len(clean.strip("-")) == 4 and clean.strip("-").startswith("20"):
                continue
            try:
                parsed.append(Decimal(clean))
            except Exception:
                continue
        if len(parsed) >= 1 and not re.search(r"[A-Za-z]{3,}", re.sub(r"\(?-?\d[\d,]*\)?", " ", candidate)):
            val = parsed[0]
            prior = parsed[1] if len(parsed) >= 2 else None
            candidates.append((val, prior, f"{lines[start_index].strip()} -> {candidate}"))
    if not candidates:
        return None
    if metric_name == "Taxation":
        return candidates[-1]
    return candidates[0]


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
