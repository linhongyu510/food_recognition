"""Tests for scripts/audit_attention_doc.py.

An auditor that silently passes everything is worse than no auditor, so these
tests confirm it actually bites: it must fail when a number drifts, when the
claim-separation guard-rails are violated, and when the audit JSON contradicts
itself. The real document is also audited, so CI fails if prose and data part
company.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_attention_doc.py"
BENCH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
DOC = Path(__file__).resolve().parents[1] / "docs" / "attention_variance.md"


def _load():
    spec = importlib.util.spec_from_file_location("audit_attention_doc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = _load()


def test_real_document_passes_its_own_audit() -> None:
    """The shipped document must trace to its sources."""
    result = auditor.audit(DOC.read_text(), BENCH)
    assert result.failures == [], result.failures
    assert result.passed > 30


def test_audit_script_runs_clean_via_main() -> None:
    rc = auditor.main(["--doc", str(DOC), "--bench", str(BENCH), "--quiet"])
    assert rc == 0


def test_main_reports_missing_document() -> None:
    rc = auditor.main(["--doc", "docs/does_not_exist.md"])
    assert rc == 2


def test_has_is_insensitive_to_markdown_line_wrapping() -> None:
    """Hard-wrapped prose must not fail a phrase lookup."""
    a = auditor.Auditor("a run that did not\nhappen at all")
    assert a.has("did not happen")


def test_check_int_accepts_either_integer_format() -> None:
    a = auditor.Auditor("the split holds 25,250 images")
    a.check_int("size", 25250)
    assert a.failures == []


def test_auditor_fails_when_a_number_is_missing() -> None:
    a = auditor.Auditor("no numbers here")
    a.check("resolution", "0.00396")
    assert len(a.failures) == 1


def test_check_absent_flags_a_forbidden_phrase() -> None:
    a = auditor.Auditor("this shows published CBAM gains are noise.")
    a.check_absent("overclaim", "published CBAM gains are noise.")
    assert len(a.failures) == 1


def test_audit_rejects_document_that_overclaims(tmp_path: Path) -> None:
    """Asserting published gains are noise must fail the audit."""
    text = DOC.read_text().replace(
        "does **not** license the claim that published CBAM gains are within noise",
        "shows published CBAM gains are noise.",
    )
    result = auditor.audit(text, BENCH)
    assert any("must not appear" in f for f in result.failures)


def test_audit_rejects_document_missing_the_compute_caveat() -> None:
    text = DOC.read_text().replace("Not completed", "Completed")
    result = auditor.audit(text, BENCH)
    assert any("compute-limited" in f for f in result.failures)


def test_audit_rejects_drifted_resolution_number() -> None:
    """Changing the points-per-image figure in prose must be caught."""
    text = DOC.read_text().replace("0.00396", "0.00500")
    result = auditor.audit(text, BENCH)
    assert result.failures


def test_audit_rejects_inconsistent_source_json(tmp_path: Path) -> None:
    """A summary count that contradicts the per-paper records must fail."""
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    data["summary"]["code_publicly_available"] = 99
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    (bench / "attention_claims_audit.json").write_text(json.dumps(data))
    (bench / "food11_seed_variance.json").write_text(
        (BENCH / "food11_seed_variance.json").read_text()
    )
    result = auditor.audit(DOC.read_text(), bench)
    assert any("code count" in f for f in result.failures)


def test_audit_reports_missing_source_file(tmp_path: Path) -> None:
    result = auditor.audit("anything", tmp_path)
    assert any("missing source" in f for f in result.failures)


def test_audit_rejects_claim_that_a_paper_reports_variance(tmp_path: Path) -> None:
    """If a paper were marked as reporting variance, the headline must fail."""
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    data["papers"][0]["seed_variance_reported"] = True
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    (bench / "attention_claims_audit.json").write_text(json.dumps(data))
    (bench / "food11_seed_variance.json").write_text(
        (BENCH / "food11_seed_variance.json").read_text()
    )
    result = auditor.audit(DOC.read_text(), bench)
    assert any("seed variance" in f for f in result.failures)
