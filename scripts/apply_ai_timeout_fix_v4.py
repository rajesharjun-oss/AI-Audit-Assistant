from __future__ import annotations

from pathlib import Path

from apply_ai_timeout_fix_v2 import main as apply_patch
from apply_ai_timeout_fix_v3 import already_applied


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    if not already_applied():
        apply_patch()
    app_path = ROOT / "app.py"
    app_text = app_path.read_text(encoding="utf-8")
    if "\x01" in app_text:
        app_text = app_text.replace(
            "\x01",
            '    return " Last AI error (debug): " + "; ".join(pieces) + "." if pieces else ""',
            1,
        )
        app_path.write_text(app_text, encoding="utf-8")
        print("repaired app.py debug-summary line")
    else:
        print("app.py debug-summary line did not require repair")
