from pathlib import Path

from batch_verify import BatchGate, _safe_folder_name, run_batch
from models import Finding, ReviewResult


def _fake_result(high=0, medium=0, low=0, issue=""):
    findings = []
    if high:
        findings.append(Finding("Totals", "High", "Page 1", issue or "High issue", "Evidence", "Fix it"))
    if medium:
        findings.append(Finding("Consistency", "Medium", "Page 2", issue or "Medium issue", "Evidence", "Fix it"))
    if low:
        findings.append(Finding("Formatting", "Low", "Page 3", issue or "Low issue", "Evidence", "Fix it"))
    return ReviewResult(
        findings=findings,
        metrics={
            "findings": len(findings),
            "high": high,
            "medium": medium,
            "low": low,
            "checks_performed_count": 5,
            "checks_passed_count": 5,
            "checks_skipped_count": 1,
        },
    )




def test_batch_verify_uses_short_hashed_output_folder_names():
    name = _safe_folder_name(Path("Very Long Financial Statement Name With Many Words And Symbols 2025 Final Draft.pdf"))

    assert len(name) <= 27
    assert name.endswith(name[-8:])

def test_batch_verify_writes_summary_and_passes_clean_file(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    pdf_path = input_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_write_review_outputs(path, profile, options, run_dir):
        result = _fake_result()
        summary = {
            "company_name": "Sample Limited",
            "findings": 0,
            "high_findings": 0,
            "medium_findings": 0,
            "low_findings": 0,
            "checks_performed": 5,
            "checks_passed": 5,
            "checks_skipped": 1,
            "output_files": {"excel": str(run_dir / "sample_exception_register.xlsx")},
        }
        (run_dir / "sample_exception_register.xlsx").write_bytes(b"xlsx")
        return result, summary

    monkeypatch.setattr("batch_verify.write_review_outputs", fake_write_review_outputs)

    rows, all_passed = run_batch(input_dir, output_dir, gate=BatchGate())

    assert all_passed is True
    assert rows[0]["gate_passed"] is True
    assert (output_dir / "batch_summary.json").exists()
    assert (output_dir / "batch_summary.csv").exists()


def test_batch_verify_fails_on_high_or_forbidden_text(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")

    def fake_write_review_outputs(path, profile, options, run_dir):
        result = _fake_result(high=1, issue="False positive signature block")
        summary = {
            "company_name": "Sample Limited",
            "findings": 1,
            "high_findings": 1,
            "medium_findings": 0,
            "low_findings": 0,
            "checks_performed": 5,
            "checks_passed": 4,
            "checks_skipped": 1,
            "output_files": {},
        }
        return result, summary

    monkeypatch.setattr("batch_verify.write_review_outputs", fake_write_review_outputs)

    rows, all_passed = run_batch(
        input_dir,
        output_dir,
        gate=BatchGate(forbidden_issue_text=("signature block",)),
    )

    assert all_passed is False
    assert rows[0]["gate_passed"] is False
    assert "high findings present" in rows[0]["gate_failures"]
    assert "forbidden finding text" in rows[0]["gate_failures"]
