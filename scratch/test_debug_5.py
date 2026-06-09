import sys
import os
from reviewer import review_pdf
from models import ReviewOptions

try:
    result = review_pdf("C:/Users/ionawoga/Downloads/CeLD.pdf", options=ReviewOptions(use_ocr=False))

    print("Findings:", result.metrics.get("findings"))
    print("High findings:", result.metrics.get("high"))
    print("Medium findings:", result.metrics.get("medium"))
    print("Low findings:", result.metrics.get("low"))

    for r in result.metrics.get("policy_export", []):
        if r["Paragraph reviewed"] == "[Industry Summary]":
            print("Nature of business:", r["Industry alignment"])
            print("Policies:", r["Expected standard topic"])
except Exception as e:
    print(f"Error: {e}")
