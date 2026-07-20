from __future__ import annotations

from pathlib import Path

from apply_ai_timeout_fix_v2 import main as apply_patch


ROOT = Path(__file__).resolve().parents[1]


def already_applied() -> bool:
    policy = (ROOT / "ai_policy_review.py").read_text(encoding="utf-8")
    combined = (ROOT / "ai_combined_review.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    return all(
        marker in source
        for source, marker in (
            (policy, 'OPENAI_REQUEST_TIMEOUT_SECONDS", "45"'),
            (policy, "def _is_timeout_error"),
            (combined, 'defaults = {"quick": 1200'),
            (app, "def _ai_failure_status_message"),
            (app, "and not ai_overall_failed"),
        )
    )


if __name__ == "__main__":
    if already_applied():
        print("AI timeout patch is already applied; running validation only.")
    else:
        apply_patch()
