from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = _read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one literal match in {path}, found {count}: {old[:120]!r}")
    _write(path, text.replace(old, new, 1))
    print(f"patched {path}: {old[:70]!r}")


def replace_all(path: str, old: str, new: str) -> None:
    text = _read(path)
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"Expected at least one literal match in {path}: {old!r}")
    _write(path, text.replace(old, new))
    print(f"patched {path}: {count} occurrence(s) of {old!r}")


def replace_regex_once(path: str, pattern: str, replacement: str) -> None:
    text = _read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Expected exactly one regex match in {path}, found {count}: {pattern!r}")
    _write(path, updated)
    print(f"patched {path}: regex {pattern[:90]!r}")


def main() -> None:
    replace_once(
        "ai_policy_review.py",
        'AI_REQUEST_TIMEOUT_SECONDS = max(5, min(int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "30")), 45))',
        'AI_REQUEST_TIMEOUT_SECONDS = max(5, min(int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")), 45))',
    )
    replace_once(
        "ai_policy_review.py",
        '            status="deferred" if _is_rate_limit_error(exc) or isinstance(exc, MalformedAiResponseError) else "error",',
        '            status="deferred" if _is_retryable_ai_error(exc) or isinstance(exc, MalformedAiResponseError) else "error",',
    )

    new_error_helpers = '''def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if isinstance(exc, AiProviderError):
        diagnostics = getattr(exc, "diagnostics", {}) or {}
        category = str(diagnostics.get("error_category", "") or "").lower()
        if category in {"rate_limit", "busy"}:
            return True
        text = f"{text} {diagnostics.get('error_message', '')}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate-limit",
            "rate exceeded",
            "too many requests",
            "ai service busy",
            "service busy",
            "temporarily busy",
            "cooldown",
        )
    )



def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if isinstance(exc, AiProviderError):
        diagnostics = getattr(exc, "diagnostics", {}) or {}
        category = str(diagnostics.get("error_category", "") or "").lower()
        if category == "timeout":
            return True
        text = f"{text} {diagnostics.get('error_message', '')}".lower()
    return isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text



def _is_retryable_ai_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        _is_rate_limit_error(exc)
        or _is_timeout_error(exc)
        or _looks_like_dns_error(text)
        or "temporarily unavailable" in text
        or "connection reset" in text
        or "remote end closed" in text
        or "503" in text
        or "502" in text
        or "500" in text
        or "504" in text
    )


'''
    replace_regex_once(
        "ai_policy_review.py",
        r"def _is_rate_limit_error\(exc: Exception\) -> bool:\n.*?(?=def _friendly_ai_error_message\(exc: Exception\) -> str:)",
        new_error_helpers,
    )
    replace_once(
        "ai_policy_review.py",
        '''        if category in {"rate_limit", "timeout", "temporary_service_error", "busy"}:
            return "AI review was not completed after automatic retry attempts because the AI service remained busy or rate-limited. The deterministic review and exports were still completed. Use Retry AI Review to run only the AI layer again."
''',
        '''        if category == "timeout":
            return "AI review timed out before the provider returned a complete response. The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        if category in {"rate_limit", "busy"}:
            return "AI review was not completed because the provider remained busy or rate-limited after the fallback attempt. The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        if category == "temporary_service_error":
            return "AI review was not completed because the provider returned a temporary service error after the fallback attempt. The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
''',
    )
    replace_regex_once(
        "ai_policy_review.py",
        r"    if _is_rate_limit_error\(exc\):\n.*?(?=    if isinstance\(exc, MalformedAiResponseError\):)",
        '''    if _is_timeout_error(exc):
        return (
            "AI review timed out before the provider returned a complete response. "
            "The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        )
    if _is_rate_limit_error(exc):
        return (
            "AI review was not completed because the provider remained temporarily busy or rate-limited. "
            "The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        )
''',
    )

    replace_regex_once(
        "ai_combined_review.py",
        r'    "quick": \{\n.*?    \},\n    "standard":',
        '''    "quick": {
        "primary_chars": 3600,
        "notes_chars": 1800,
        "contents_chars": 900,
        "key_pages": 3,
        "key_page_chars": 550,
        "findings": 18,
        "skipped": 8,
    },
    "standard":''',
    )
    replace_once(
        "ai_combined_review.py",
        '    defaults = {"quick": 1600, "standard": COMBINED_AI_OUTPUT_TOKENS, "deep": 4200}',
        '    defaults = {"quick": 1200, "standard": COMBINED_AI_OUTPUT_TOKENS, "deep": 4200}',
    )

    replace_once(
        "ai_review_pipeline.py",
        '    mode_label = _display_review_mode(getattr(combined, "review_mode", "") or review_context.review_mode).title()\n',
        '    mode_label = _display_review_mode(getattr(combined, "review_mode", "") or review_context.review_mode).title()\n    combined_message = _combined_user_message(combined)\n',
    )
    replace_once("ai_review_pipeline.py", "result.policy_message = combined.message", "result.policy_message = combined_message")
    replace_once("ai_review_pipeline.py", "result.full_message = combined.message", "result.full_message = combined_message")
    replace_once("ai_review_pipeline.py", "result.finding_message = combined.message", "result.finding_message = combined_message")
    replace_regex_once(
        "ai_review_pipeline.py",
        r"    elif combined\.message:\n        result\.checks_skipped\.append\(combined\.message\)\n\n+(?=def _stage_summary_from_rows)",
        '''    elif combined_message:
        result.checks_skipped.append(combined_message)



def _combined_user_message(combined) -> str:
    category = _combined_failure_category(combined)
    if category == "timeout":
        return (
            "AI review timed out before the provider returned a complete response. "
            "The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        )
    if category in {"rate_limit", "busy"}:
        return (
            "AI review was not completed because the provider remained busy or rate-limited after the fallback attempt. "
            "The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        )
    if category == "temporary_service_error":
        return (
            "AI review was not completed because the provider returned a temporary service error after the fallback attempt. "
            "The deterministic review and exports were still completed. Use Retry AI Review to rerun only the AI layer."
        )
    return str(getattr(combined, "message", "") or "").strip()


''',
    )

    replace_once(
        "app.py",
        '''    if any(marker in lower for marker in ("429", "rate limit", "rate exceeded", "too many requests", "ai service busy", "timed out", "timeout")):
        return (
            "The review worker or AI service is temporarily busy, so this upload could not finish cleanly. "
            "Please wait a moment and try again. If deterministic output was already produced, use Retry AI Review rather than re-uploading the PDF."
        )
''',
        '''    if "timed out" in lower or "timeout" in lower:
        return (
            "The AI request timed out before a complete response was returned. "
            "If deterministic output was already produced, use Retry AI Review rather than re-uploading the PDF."
        )
    if any(marker in lower for marker in ("429", "rate limit", "rate exceeded", "too many requests", "ai service busy")):
        return (
            "The AI provider is temporarily busy or rate-limited, so the AI layer could not finish cleanly. "
            "If deterministic output was already produced, use Retry AI Review rather than re-uploading the PDF."
        )
''',
    )
    replace_regex_once(
        "app.py",
        r'(    return " Last AI error \(debug\): " \+ "; "\.join\(pieces\) \+ "\." if pieces else ""\n)\n*(?=def _metric_lines)',
        '''\1

def _ai_failure_status_message(ai_error_rows: object) -> str:
    last = None
    if isinstance(ai_error_rows, list):
        last = next((row for row in reversed(ai_error_rows) if isinstance(row, dict)), None)
    category = str((last or {}).get("Error category", "") or "").strip().lower()
    if category == "timeout":
        return (
            "AI review timed out before the provider returned a complete response. "
            "The deterministic review and exports are still available; use Retry AI Review to rerun only the AI layer."
        )
    if category in {"rate_limit", "busy"}:
        return (
            "AI review could not complete because the provider remained busy or rate-limited after the fallback attempt. "
            "The deterministic review and exports are still available; use Retry AI Review to rerun only the AI layer."
        )
    if category == "temporary_service_error":
        return (
            "AI review could not complete because the provider returned a temporary service error after the fallback attempt. "
            "The deterministic review and exports are still available; use Retry AI Review to rerun only the AI layer."
        )
    return (
        "AI review could not be completed. The deterministic review and exports are still available; "
        "use Retry AI Review to rerun only the AI layer."
    )


''',
    )
    replace_once(
        "app.py",
        'ai_overall_status = str(result.metrics.get("ai_review_status", "Not started") or "Not started")\n',
        'ai_overall_status = str(result.metrics.get("ai_review_status", "Not started") or "Not started")\nai_overall_failed = ai_overall_status.startswith("Failed")\n',
    )
    replace_regex_once(
        "app.py",
        r'elif ai_overall_status\.startswith\("Failed"\):\n    st\.warning\(\n.*?\n    \)\n',
        '''elif ai_overall_failed:
    st.warning(_ai_failure_status_message(ai_error_rows))
''',
    )
    replace_once("app.py", "elif use_ai_policy_review:\n", "elif use_ai_policy_review and not ai_overall_failed:\n")
    replace_once("app.py", "elif use_ai_full_review:\n", "elif use_ai_full_review and not ai_overall_failed:\n")
    replace_once(
        "app.py",
        "elif use_ai_policy_review or use_ai_full_review:\n",
        "elif (use_ai_policy_review or use_ai_full_review) and not ai_overall_failed:\n",
    )

    replace_all(".env.example", "OPENAI_QUICK_REVIEW_OUTPUT_TOKENS=1600", "OPENAI_QUICK_REVIEW_OUTPUT_TOKENS=1200")
    replace_all(".env.example", "OPENAI_REQUEST_TIMEOUT_SECONDS=30", "OPENAI_REQUEST_TIMEOUT_SECONDS=45")
    replace_all("README.md", 'OPENAI_QUICK_REVIEW_OUTPUT_TOKENS="1600"', 'OPENAI_QUICK_REVIEW_OUTPUT_TOKENS="1200"')
    replace_all("README.md", "OPENAI_QUICK_REVIEW_OUTPUT_TOKENS=1600", "OPENAI_QUICK_REVIEW_OUTPUT_TOKENS=1200")
    replace_all("README.md", 'OPENAI_REQUEST_TIMEOUT_SECONDS="30"', 'OPENAI_REQUEST_TIMEOUT_SECONDS="45"')
    replace_all("README.md", "OPENAI_REQUEST_TIMEOUT_SECONDS=30", "OPENAI_REQUEST_TIMEOUT_SECONDS=45")

    replace_once(
        "tests/test_reviewer.py",
        '    assert payload["max_output_tokens"] == 1600',
        '    assert payload["max_output_tokens"] == 1200',
    )

    _write(
        "tests/test_ai_timeout_regressions.py",
        '''from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ai_combined_review
import ai_policy_review
import ai_review_pipeline


def _timeout_error() -> ai_policy_review.AiProviderError:
    return ai_policy_review.AiProviderError(
        "The read operation timed out.",
        {"error_category": "timeout", "error_message": "The read operation timed out."},
    )


def test_timeout_is_not_misclassified_as_rate_limit():
    exc = _timeout_error()
    assert ai_policy_review._is_timeout_error(exc)
    assert not ai_policy_review._is_rate_limit_error(exc)
    assert ai_policy_review._is_retryable_ai_error(exc)


def test_timeout_message_is_specific_and_actionable():
    message = ai_policy_review._friendly_ai_error_message(_timeout_error())
    assert "timed out" in message.lower()
    assert "rate-limited" not in message.lower()
    assert "Retry AI Review" in message
    assert "deterministic review and exports" in message


def test_combined_pipeline_preserves_timeout_specific_message():
    combined = SimpleNamespace(error_rows=[{"Error category": "timeout"}], message="Generic provider failure.")
    message = ai_review_pipeline._combined_user_message(combined)
    assert "timed out" in message.lower()
    assert "rate-limited" not in message.lower()


def test_quick_review_uses_smaller_bounded_package(monkeypatch):
    monkeypatch.delenv("OPENAI_QUICK_REVIEW_OUTPUT_TOKENS", raising=False)
    assert ai_combined_review._output_tokens_for_package({"review_mode": "quick"}) == 1200
    assert ai_combined_review.MODE_LIMITS["quick"]["primary_chars"] == 3600
    assert ai_combined_review.MODE_LIMITS["quick"]["findings"] == 18


def test_failed_combined_review_is_not_repeated_in_all_three_stage_cards():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "elif use_ai_policy_review and not ai_overall_failed:" in source
    assert "elif use_ai_full_review and not ai_overall_failed:" in source
    assert "elif (use_ai_policy_review or use_ai_full_review) and not ai_overall_failed:" in source
    assert "st.warning(_ai_failure_status_message(ai_error_rows))" in source
''',
    )
    print("wrote tests/test_ai_timeout_regressions.py")


if __name__ == "__main__":
    main()
