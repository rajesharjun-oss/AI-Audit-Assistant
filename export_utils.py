from __future__ import annotations

import re
from difflib import SequenceMatcher


def clean_name_consistency_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name1 = str(row.get("Name variant 1", "")).strip()
        name2 = str(row.get("Name variant 2", "")).strip()
        pages1 = _parse_page_set(row.get("Page 1", ""))
        pages2 = _parse_page_set(row.get("Page 2", ""))
        if _looks_like_one_page_name_artifact(name1, pages1, name2, pages2):
            continue
        if name1 and name2:
            standard = name1 if len(pages1) >= len(pages2) else name2
            row = dict(row)
            row["Suggested standard spelling"] = standard
            row["Page 1"] = ", ".join(str(page) for page in sorted(pages1)) if pages1 else str(row.get("Page 1", ""))
            row["Page 2"] = ", ".join(str(page) for page in sorted(pages2)) if pages2 else str(row.get("Page 2", ""))
        cleaned.append(row)
    return cleaned


def _parse_page_set(value: object) -> set[int]:
    return {int(match) for match in re.findall(r"\d+", str(value or ""))}


def _looks_like_one_page_name_artifact(name1: str, pages1: set[int], name2: str, pages2: set[int]) -> bool:
    if not name1 or not name2:
        return False
    if not (pages1 & pages2):
        return False
    if _single_page_token_artifact(name1, pages1, name2, pages2):
        return True
    if len(pages1) == len(pages2):
        return len(pages1) == 1 and _name_similarity(name1, name2) >= 0.9
    rare_pages, common_pages = (pages1, pages2) if len(pages1) < len(pages2) else (pages2, pages1)
    return len(rare_pages) == 1 and len(common_pages) >= 2 and _name_similarity(name1, name2) >= 0.9


def _name_similarity(name1: str, name2: str) -> float:
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()


def _single_page_token_artifact(name1: str, pages1: set[int], name2: str, pages2: set[int]) -> bool:
    rare_name, rare_pages, common_name, common_pages = (
        (name1, pages1, name2, pages2) if len(pages1) < len(pages2) else (name2, pages2, name1, pages1)
    )
    if len(rare_pages) != 1 or len(common_pages) < 2:
        return False
    rare_tokens = re.findall(r"[A-Za-z]+", rare_name.lower())
    common_tokens = re.findall(r"[A-Za-z]+", common_name.lower())
    if len(rare_tokens) != len(common_tokens) or len(rare_tokens) < 2:
        return False
    diffs = [(left, right) for left, right in zip(rare_tokens, common_tokens) if left != right]
    if len(diffs) != 1:
        return False
    left, right = diffs[0]
    if left[:1] != right[:1]:
        return False
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return len(longer) - len(shorter) == 1 and longer.startswith(shorter)
