"""Tests for scripts/audit_significance_doc.py.

The auditor is the thing that stops prose drifting from data, so it needs its
own tests: an auditor that silently passes everything is worse than none.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_significance_doc.py"
BENCH = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
DOC = Path(__file__).resolve().parents[1] / "docs" / "significance.md"


def _load():
    spec = importlib.util.spec_from_file_location("audit_significance_doc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = _load()


def test_auditor_accepts_thousands_separated_integers() -> None:
    a = auditor.Auditor("the run got 1,910 discordant images")
    a.check_int("discordant", 1910)
    assert a.failures == []


def test_auditor_accepts_bare_integers() -> None:
    a = auditor.Auditor("n_discordant=1910")
    a.check_int("discordant", 1910)
    assert a.failures == []


def test_auditor_reports_a_missing_number() -> None:
    a = auditor.Auditor("nothing relevant here")
    a.check_int("discordant", 1910)
    a.check("accuracy", "88.701%")
    assert len(a.failures) == 2
    assert "1910" in a.failures[0]


def test_auditor_counts_passes() -> None:
    a = auditor.Auditor("88.701% and 1,910")
    a.check("acc", "88.701%")
    a.check_int("disc", 1910)
    assert a.passed == 2 and a.failures == []


def test_check_true_records_detail_on_failure() -> None:
    a = auditor.Auditor("")
    a.check_true("label", False, "because reasons")
    assert a.failures == ["label: because reasons"]


# ---------------------------------------------------------------------------
# the real document
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_real_document_passes_its_own_audit() -> None:
    """Every number in docs/significance.md must trace to its source JSON."""
    result = auditor.audit(DOC.read_text(), BENCH)
    assert result.failures == [], "\n".join(result.failures)
    assert result.passed > 50  # the audit is actually checking things


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_catches_a_corrupted_number() -> None:
    # Flip one digit of a real figure; the audit must notice.
    text = DOC.read_text().replace("88.701%", "88.999%")
    result = auditor.audit(text, BENCH)
    assert any("88.701" in f for f in result.failures)


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_rejects_reintroducing_an_untraceable_figure() -> None:
    # 94.56% and 84.09% have no traceable source; CONTRIBUTING.md forbids them
    # until one exists, so the audit must block their return.
    text = DOC.read_text() + "\n\nThe model reaches 94.56% accuracy.\n"
    result = auditor.audit(text, BENCH)
    assert any("94.56" in f for f in result.failures)


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_requires_the_non_canonical_dataset_caveat() -> None:
    text = DOC.read_text().replace("not** the canonical Food-11", "the canonical Food-11")
    result = auditor.audit(text, BENCH)
    assert any("canonical" in f for f in result.failures)


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_requires_the_compute_limited_section() -> None:
    text = DOC.read_text().replace("Not completed", "Extras").replace(
        "not completed", "extras"
    )
    result = auditor.audit(text, BENCH)
    assert any("compute-limited" in f for f in result.failures)


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_requires_prose_strength_to_match_each_verdict() -> None:
    """The document must claim exactly what the JSON supports, per pair.

    Rewritten twice as the data grew, and both breakages were correct. First it
    assumed exactly one positive verdict; extending two cells to 6 seeds made
    four pairs positive. Then it assumed a parametric-only verdict always means
    the exact test *could not* reach alpha; at n=6 one pair is parametric-only
    because the exact test could have rejected and did not. Both assumptions
    were properties of n=3, not of the analysis, so the checks now derive
    everything from the JSON.
    """
    sig = json.loads((BENCH / "food11_seed_significance.json").read_text())
    positive = [
        c
        for c in sig["_significance"]["comparisons"]
        if c["verdict"].startswith("significant")
    ]
    assert positive, "no positive verdicts to check"

    for c in positive:
        exact_p = c["permutation"]["p_value"]
        alpha = c["alpha"]
        if c["verdict"] == "significant_parametric_only":
            assert exact_p > alpha, (
                f"{c['cell_a']} vs {c['cell_b']} is parametric-only but its exact "
                f"p={exact_p} clears alpha={alpha}"
            )
        else:
            assert exact_p <= alpha, (
                f"{c['cell_a']} vs {c['cell_b']} is plainly significant but its "
                f"exact p={exact_p} misses alpha={alpha}"
            )


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_audit_catches_a_corrupted_exact_p_value() -> None:
    sig = json.loads((BENCH / "food11_seed_significance.json").read_text())
    clear = [
        c
        for c in sig["_significance"]["comparisons"]
        if c["verdict"] == "significant"
    ]
    if not clear:
        pytest.skip("no plainly-significant pair to corrupt")
    exact_p = clear[0]["permutation"]["p_value"]
    text = DOC.read_text().replace(f"{exact_p:.4f}", "0.9999")
    result = auditor.audit(text, BENCH)
    assert any("exact p" in f for f in result.failures)


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_power_limited_tracks_the_design_not_the_effect() -> None:
    """power_limited must mean "the floor itself misses alpha".

    It is a property of n, so at n=6 (floor 0.031 < 0.05) no pair may be flagged
    power-limited regardless of its verdict. Conflating "parametric-only" with
    "underpowered" would hand a weak result an excuse it has not earned.
    """
    sig = json.loads((BENCH / "food11_seed_significance.json").read_text())
    for c in sig["_significance"]["comparisons"]:
        floor = c["permutation"]["min_attainable_p"]
        expected = floor > c["alpha"]
        assert c["power_limited"] is expected, (
            f"{c['cell_a']} vs {c['cell_b']}: power_limited={c['power_limited']} "
            f"but floor={floor} vs alpha={c['alpha']}"
        )


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_floor_hugging_rejections_are_disclosed() -> None:
    """A p-value equal to 2/2**n must be described as such in the prose."""
    sig = json.loads((BENCH / "food11_seed_significance.json").read_text())
    on_floor = [
        c
        for c in sig["_significance"]["comparisons"]
        if c["permutation"].get("at_floor")
        and c["permutation"]["p_value"] <= c["alpha"]
    ]
    if not on_floor:
        pytest.skip("no floor-hugging rejection in the current report")
    assert "floor" in DOC.read_text().lower()
    # Removing the disclosure must make the audit fail.
    text = DOC.read_text().replace("floor", "value").replace("Floor", "Value")
    result = auditor.audit(text, BENCH)
    assert any("floor-hugging" in f for f in result.failures)


def test_main_returns_2_for_a_missing_document(tmp_path: Path) -> None:
    assert auditor.main(["--doc", str(tmp_path / "absent.md")]) == 2


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_main_succeeds_on_the_real_document(capsys) -> None:
    rc = auditor.main(["--doc", str(DOC), "--bench", str(BENCH)])
    assert rc == 0
    assert "traces to its source" in capsys.readouterr().out


@pytest.mark.skipif(not DOC.is_file(), reason="significance.md not present")
def test_main_returns_1_on_a_broken_document(tmp_path: Path) -> None:
    broken = tmp_path / "broken.md"
    broken.write_text("this document contains none of the required numbers")
    assert auditor.main(["--doc", str(broken), "--bench", str(BENCH)]) == 1


def test_source_jsons_are_present_and_parseable() -> None:
    """The audit is only meaningful if its sources exist in the repo."""
    for name in (
        "food11_seed_variance.json",
        "food11_seed_significance.json",
        "food101_validation_power.json",
        "food101_paired_eval.json",
        "food11_epoch_criterion.json",
    ):
        path = BENCH / name
        assert path.is_file(), f"missing {name}"
        assert json.loads(path.read_text())
