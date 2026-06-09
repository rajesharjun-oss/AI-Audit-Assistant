import re

with open("cross_page_consistency.py", "r", encoding="utf-8") as f:
    text = f.read()

# 6. Key Amount Consistency
old_pat = r'"Profit after tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?(?:\s+after\s+tax(?:ation)?)?(?:\s+for\s+the\s+(?:year|period))?\b", re.I)'
new_pat = r'"Profit after tax": re.compile(r"^(?:Profit|Loss)(?:\/\(loss\))?(?:\s+after\s+tax(?:ation)?|\s+for\s+the\s+(?:year|period))\b", re.I)'
text = text.replace(old_pat, new_pat)

# 10. Name Consistency
old_name = """                name = re.sub(r"(?i)\s+(?:board|director|directors|manager|officer|table|notes|chief|executive|committee|chairman)$", "", match.group(0))"""
new_name = """                name = re.sub(r"(?i)\\b(?:chief|mr|mrs|dr|sir|board|director|directors|manager|officer|table|notes|executive|committee|chairman)\\b", " ", match.group(0))
                name = re.sub(r"\\s+", " ", name).strip()
                if len(name.split()) >= 4:
                    parts = name.split()
                    name_candidates.append((" ".join(parts[:2]), page.number))
                    name_candidates.append((" ".join(parts[2:]), page.number))
                    continue"""
text = text.replace(old_name, new_name)

with open("cross_page_consistency.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
