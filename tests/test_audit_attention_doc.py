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

import pytest

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
    """Asserting published gains are noise must fail the audit.

    The whole sentence is replaced, not a fragment of it: swapping only the
    middle would leave the surrounding "It does ... ." scaffolding and produce
    a sentence that is ungrammatical rather than a genuine overclaim.
    """
    text = DOC.read_text().replace(
        "It does **not** license the claim that published CBAM gains are within noise.",
        "This shows published CBAM gains are within noise.",
    )
    result = auditor.audit(text, BENCH)
    assert any("asserts published gains are noise" in f for f in result.failures)


# --- the banned-claim detector -------------------------------------------
# Regression: the original check was a single exact string comparison against
# "published CBAM gains are noise.", so any paraphrase passed all 56 checks
# with exit status 0. The variants below were verified to slip through that
# version; each must now be caught. The first one is the exact string used to
# demonstrate the bypass.

BANNED_VARIANTS = [
    "Published attention gains are noise.",
    "Published CBAM gains are noise.",
    "Published attention gains are just noise.",
    "The reported gains are merely seed noise.",
    "Prior work's improvements are random noise.",
    "Their reported improvement is noise.",
    "These papers' gains fall within seed noise.",
    "Published improvements lie inside the noise.",
    "The published gain is not statistically significant.",
    "Prior work's reported gains are statistically insignificant.",
    "Published attention improvements are explained by seed noise.",
    "Their gains are indistinguishable from seed noise.",
    "The reported improvements are attributable to random noise.",
    "Published gains are no better than noise.",
    "Published **attention gains** are  noise.",
    "PUBLISHED ATTENTION GAINS ARE NOISE.",
    "Their improvements were within noise.",
    "Previous work's results are purely noise.",
]


@pytest.mark.parametrize("sentence", BANNED_VARIANTS)
def test_detector_catches_each_rewrite(sentence: str) -> None:
    assert auditor.find_banned_noise_claims(sentence), sentence


@pytest.mark.parametrize("sentence", BANNED_VARIANTS)
def test_audit_rejects_each_rewrite_appended_to_the_document(sentence: str) -> None:
    """End-to-end: appending any variant to the real document must fail it."""
    result = auditor.audit(DOC.read_text() + "\n\n" + sentence + "\n", BENCH)
    assert any("asserts published gains are noise" in f for f in result.failures), sentence


# Sentences the document legitimately contains, or could contain. Flagging any
# of these would make the guard-rail unusable and invite its removal, so the
# false-positive side is tested as explicitly as the true-positive side.
ALLOWED_SENTENCES = [
    # the document's own title
    "Are published attention gains in food recognition larger than seed noise?",
    # the Q2 column head
    "Q2: Does each claimed gain fall inside its own seed noise?",
    # docs/attention_variance.md line 190 - the denial that must keep passing
    "It does **not** license the claim that published CBAM gains are within noise.",
    "**Not claimed.** That the headline +1.04 pts (263 images) is noise.",
    "Whether published gains fall within seed noise is not established here.",
    "This does not show that published gains are noise.",
    "No evidence is offered that their reported improvement is noise.",
    "We cannot say published gains are within noise.",
    # this project's own result - about its own experiment, not published work
    "On this setup the CBAM gain is not separable from seed noise at n=3.",
    "The measured seed spread is 0.15-0.76 pts.",
]


@pytest.mark.parametrize("sentence", ALLOWED_SENTENCES)
def test_detector_allows_legitimate_sentences(sentence: str) -> None:
    assert auditor.find_banned_noise_claims(sentence) == [], sentence


def test_line_190_denial_survives_in_full_document_context() -> None:
    """The real denial must pass with its actual surrounding prose, not alone.

    This is the false-positive case that matters most: tightening the detector
    until it flags the document's own disclaimer would be a silent regression.
    """
    text = DOC.read_text()
    assert "does **not** license the claim that published CBAM gains are within noise" in text
    assert auditor.find_banned_noise_claims(text) == []


def test_detector_ignores_negation_that_follows_the_claim() -> None:
    """A trailing "not" in a separate clause must not excuse the assertion."""
    assert auditor.find_banned_noise_claims("Published gains are noise, not signal.")


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


def test_audit_rejects_variance_count_that_contradicts_the_records(tmp_path: Path) -> None:
    """The variance count must be checked against the records, not assumed zero.

    Regression: this audit used to assert ``not variance_yes`` outright, which
    encoded the false belief that no triaged paper reports repeated runs.
    Rokhva & Teimourpour report five from-scratch runs, so the invariant is now
    consistency between the per-paper records and the summary count.
    """
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    # records say 1 reports repeated runs; claim 0 in the summary
    data["summary"]["papers_reporting_seed_variance"] = 0
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    (bench / "attention_claims_audit.json").write_text(json.dumps(data))
    for name in ("food11_seed_variance.json", "attention_corpus_17_stats.json"):
        (bench / name).write_text((BENCH / name).read_text())
    result = auditor.audit(DOC.read_text(), bench)
    assert any("variance count" in f for f in result.failures)


def test_rokhva_multi_run_values_average_to_the_reported_headline() -> None:
    """The five per-run accuracies must reproduce the 96.40% headline."""
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    rokhva = next(p for p in data["papers"] if p["id"] == "rokhva2025")
    mrp = rokhva["multi_run_protocol"]
    assert mrp["n_runs"] == 5
    assert len(mrp["per_run_accuracy_percent"]) == 5
    assert abs(sum(mrp["per_run_accuracy_percent"]) / 5 - 96.40) < 0.005
    assert abs(mrp["range_points"] - 0.27) < 1e-9


def test_rokhva_no_ablation_conclusion_survives_the_variance_correction() -> None:
    """Correcting the multi-run field must not erase the no-ablation finding.

    These are separate facts: the paper repeats runs but isolates no attention
    gain. Losing the second would let the audit imply a testable published gain
    exists where none does.
    """
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    rokhva = next(p for p in data["papers"] if p["id"] == "rokhva2025")
    assert rokhva["seed_variance_reported"] is True
    assert rokhva["attention_ablation"]["isolated_no_cbam_control"] is False


def test_corpus_crosstab_is_internally_consistent() -> None:
    """K=17 and the 2x2 must agree with each other."""
    stats = json.loads((BENCH / "attention_corpus_17_stats.json").read_text())
    cross = stats["crosstab_2x2_variance_x_ablation"]
    assert stats["funnel"]["included_K"] == 17
    assert cross["n_total"] == 17
    assert (
        cross["both"]["n"]
        + cross["variance_only"]["n"]
        + cross["ablation_only"]["n"]
        + cross["neither"]["n"]
        == 17
    )
    assert cross["both"]["n"] == 0


def test_corpus_mapping_covers_every_triaged_paper() -> None:
    """Each of the 7 deep-audit papers must say where it sits in the corpus."""
    data = json.loads((BENCH / "attention_claims_audit.json").read_text())
    mapping = data["corpus_relationship"]["mapping"]
    assert {p["id"] for p in data["papers"]} == set(mapping)
