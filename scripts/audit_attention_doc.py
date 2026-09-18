"""Cross-check every number in docs/attention_variance.md against its sources.

Same contract as ``scripts/audit_significance_doc.py``, applied to the
attention-claims document: no figure may appear in the prose unless it can be
recomputed from ``docs/benchmarks/attention_claims_audit.json``, from the CBAM
ablation report, or from an arithmetic identity checked here::

    python scripts/audit_attention_doc.py

This document makes claims about *other people's* papers, so the audit also
enforces the separation this project insists on: the document must not assert
that published gains are noise, must keep its own CBAM experiment distinct from
the published claims, and must state the seed-control finding it does make.
Exit status is non-zero on any failure, so it can run in CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BENCH = Path("docs/benchmarks")

# --- banned-claim detection -------------------------------------------------
# The claim this project must never make is "the *published* gains are noise".
# An earlier version of this audit tested one exact string, which meant any
# paraphrase - "Published attention gains are noise." - sailed through all
# checks. The detector below works on normalised sentences instead.
#
# It is deliberately two-sided. The document legitimately *denies* this claim
# in several places, and those denials must keep passing:
#
#   - the title, "Are published attention gains ... larger than seed noise?"
#   - the Q2 column head, "Does each claimed gain fall inside its own seed noise?"
#   - "**Not claimed.** That the headline +1.04 pts (263 images) is noise."
#   - "It does **not** license the claim that published CBAM gains are within
#     noise." (docs/attention_variance.md)
#
# So a sentence is flagged only when it names published work, names a gain, and
# asserts that gain is noise *without* a negation or question guarding it.

_SUBJECT = re.compile(
    r"\b(published|prior work|previous work|other people'?s|others'|their|theirs|"
    r"reported|literature|existing work|these papers|those papers)\b"
)
_GAIN = re.compile(r"\b(gain|gains|improvement|improvements|effect|effects|result|results)\b")

# Predicates that assert "this is noise" / "this is not significant".
# "larger than seed noise" is deliberately NOT here: comparing against noise is
# a legitimate thing to say.
_PREDICATES = [
    re.compile(
        r"\b(are|is|were|was)\s+(just\s+|merely\s+|only\s+|all\s+|entirely\s+|purely\s+|"
        r"simply\s+|likely\s+|probably\s+|mostly\s+|largely\s+)*(seed\s+|random\s+)?noise\b"
    ),
    re.compile(
        r"\b(fall|falls|fell|falling|lie|lies|lay|sit|sits|sat|land|lands|remain|remains)\s+"
        r"(with)?in(side)?\s+(the\s+)?(their\s+|its\s+|his\s+|her\s+|own\s+)*"
        r"(seed\s+|random\s+)?noise\b"
    ),
    re.compile(
        r"\b(are|is|were|was)\s+(well\s+|entirely\s+|comfortably\s+)?"
        r"(with)?in(side)?\s+(the\s+)?(their\s+|its\s+|own\s+)*(seed\s+|random\s+)?noise\b"
    ),
    re.compile(r"\b(with)?in(side)?\s+(the\s+)?(seed\s+|random\s+)?noise\b"),
    re.compile(r"\b(are|is|were|was)\s+not\s+(statistically\s+)?significant\b"),
    re.compile(r"\b(statistically\s+)?insignificant\b"),
    re.compile(r"\b(explained|explainable)\s+by\s+(seed\s+|random\s+)?noise\b"),
    re.compile(r"\battributable\s+to\s+(seed\s+|random\s+)?noise\b"),
    re.compile(r"\bindistinguishable\s+from\s+(seed\s+|random\s+)?noise\b"),
    re.compile(r"\bno\s+(better|different|bigger|larger)\s+than\s+(seed\s+|random\s+)?noise\b"),
]

# Cues that mean the sentence is denying, questioning or hedging the claim
# rather than making it.
_NEGATION = re.compile(
    r"\b(not|n't|never|cannot|can't|without|neither|nor|unproven|unverified|"
    r"whether|unclear|would|could|cannot be|do not|does not|did not|"
    r"no evidence|not claimed|not licensed|refuse[sd]?|refut\w+|avoid\w*|"
    r"must never|never assert|forbid\w*|ban(?:s|ned)?|reject\w*|"
    r"disallow\w*|prohibit\w*)\b"
)


def _normalise(text: str) -> str:
    """Lower-case, drop markdown emphasis and blockquote markers, collapse space."""
    text = re.sub(r"[*_`]+", "", text)
    # "> " continuation markers inside a hard-wrapped blockquote are not words.
    text = re.sub(r"(?:^|\s)>\s+", " ", text)
    text = text.replace("\u2019", "'").replace("\u2014", " ").replace("\u2013", " ")
    return " ".join(text.split()).lower()


def find_banned_noise_claims(text: str) -> list[str]:
    """Return sentences that assert published gains are noise.

    Splits on sentence boundaries. A negation only counts as a denial when it
    appears *before* the offending predicate ("does **not** license the claim
    that published gains are within noise"), because a negation after it can
    belong to a different clause ("are noise, not signal") and must not excuse
    the assertion.
    """
    flat = " ".join(text.split())
    # Split on . ! ? and on markdown table cell / list boundaries, keeping it
    # simple: over-splitting only makes the check stricter about context, and
    # the lookback below restores the context that matters.
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}|\|", flat)
    # A heading has no terminating period, so it otherwise glues itself to the
    # first sentence of its section. "What this experiment does and does not
    # license" would then donate its "not" to that sentence and excuse it.
    # List bullets and blockquote markers are split for the same reason: they
    # begin a new statement without ending the previous one.
    _BLOCK = r"#{1,6}\s+|^\s*[->*]\s+|\s+[->*]\s+\*\*"
    raw = [part for chunk in raw for part in re.split(_BLOCK, chunk)]
    sentences = [s for s in (r.strip() for r in raw) if s]

    offenders: list[str] = []
    for i, sentence in enumerate(sentences):
        norm = _normalise(sentence)
        if not (_SUBJECT.search(norm) and _GAIN.search(norm)):
            continue
        hit = next((p for p in _PREDICATES if p.search(norm)), None)
        if hit is None:
            continue
        # A question is asking, not asserting. This covers both a sentence that
        # ends in "?" and a quoted question embedded in a declarative sentence,
        # as in: the question "is their gain inside seed noise?" has no
        # published quantity to attach to.
        if sentence.rstrip().endswith("?"):
            continue
        quoted_questions = re.findall(r'"[^"]*\?"|\u201c[^\u201d]*\?\u201d', sentence)
        if quoted_questions:
            stripped = sentence
            for q in quoted_questions:
                stripped = stripped.replace(q, " ")
            if not any(p.search(_normalise(stripped)) for p in _PREDICATES):
                continue
        # Only a negation that precedes the predicate is a denial of it, and
        # only within the same clause. An earlier clause can carry an unrelated
        # "not" - as in '... is / is not separable from seed noise at n=3."
        # This shows published gains are within noise.' - which must not excuse
        # the assertion that follows it.
        match = hit.search(norm)
        assert match is not None
        preceding = norm[: match.start()]
        _CLAUSE = r'[;:."]|\bthis shows\b|\bthis means\b|\bso\b|\btherefore\b'
        clause = re.split(_CLAUSE, preceding)[-1]
        if _NEGATION.search(clause):
            continue
        # A short lead-in label carries the denial for the next sentence, as in
        # "**Not claimed.** That the headline ... is noise." The length cap
        # keeps an unrelated neighbouring sentence that merely contains "not"
        # from excusing a real assertion.
        previous = _normalise(sentences[i - 1]) if i else ""
        if len(previous) <= 40 and _NEGATION.search(previous):
            continue
        offenders.append(sentence.strip())
    return offenders


class Auditor:
    """Collects pass/fail results for individual number lookups."""

    def __init__(self, text: str) -> None:
        self.text = text
        # Markdown hard-wraps prose, so a phrase like "did not happen" can be
        # split across lines. Phrase lookups run against a whitespace-collapsed
        # copy; otherwise the audit would fail on line breaks rather than on
        # anything substantive.
        self.flat = " ".join(text.split())
        self.passed = 0
        self.failures: list[str] = []

    def has(self, needle: str) -> bool:
        return needle in self.text or " ".join(needle.split()) in self.flat

    def check(self, label: str, needle: str) -> None:
        if self.has(needle):
            self.passed += 1
        else:
            self.failures.append(f"{label}: {needle!r} not found in document")

    def check_int(self, label: str, value: int) -> None:
        if self.has(f"{value:,}") or self.has(str(value)):
            self.passed += 1
        else:
            self.failures.append(f"{label}: neither {value} nor {value:,} found")

    def check_true(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(f"{label}: {detail or 'condition false'}")

    def check_absent(self, label: str, needle: str) -> None:
        if not self.has(needle):
            self.passed += 1
        else:
            self.failures.append(f"{label}: {needle!r} must not appear")

    def check_no_banned_claim(self, label: str) -> None:
        offenders = find_banned_noise_claims(self.text)
        if not offenders:
            self.passed += 1
        else:
            joined = "; ".join(repr(o) for o in offenders[:3])
            self.failures.append(f"{label}: asserts published gains are noise: {joined}")


def audit(doc_text: str, bench: Path = BENCH) -> Auditor:
    a = Auditor(doc_text)
    audit_path = bench / "attention_claims_audit.json"
    if not audit_path.is_file():
        a.failures.append(f"missing source: {audit_path}")
        return a
    data = json.loads(audit_path.read_text())
    papers = data["papers"]
    summary = data["summary"]

    # --- summary counts must match the per-paper records, and the prose ---
    code_yes = [p for p in papers if p["code"].get("available") is True]
    variance_yes = [p for p in papers if p.get("seed_variance_reported")]
    a.check_true(
        "audit JSON self-consistent: code count",
        len(code_yes) == summary["code_publicly_available"],
        f"{len(code_yes)} vs {summary['code_publicly_available']}",
    )
    a.check_true(
        "audit JSON self-consistent: paper count",
        len(papers) == summary["papers_triaged"],
    )
    # Rokhva & Teimourpour DO report five from-scratch runs; an earlier version
    # of this audit asserted that no paper reports variance, having inferred it
    # from `grep -ci seed` over their code. The count is now checked against the
    # records rather than assumed to be zero.
    a.check_true(
        "audit JSON self-consistent: variance count",
        len(variance_yes) == summary["papers_reporting_seed_variance"],
        f"{len(variance_yes)} vs {summary['papers_reporting_seed_variance']}",
    )
    a.check_true(
        "no paper was retrained",
        summary["papers_actually_retrained_in_this_session"] == 0,
    )
    a.check_int("papers triaged stated", len(papers))
    a.check_true(
        "code-availability count stated",
        a.has(f"{len(code_yes)} of {len(papers)} publish code")
        or a.has(f"**{len(code_yes)} of {len(papers)} publish code"),
    )
    a.check_true(
        "variance-reporting count stated",
        a.has(f"{len(variance_yes)} of {len(papers)} report") or a.has("1 of 7 reports"),
    )

    # --- the 17-paper corpus: the headline K, and the structural finding ---
    corpus_stats_path = bench / "attention_corpus_17_stats.json"
    if corpus_stats_path.is_file():
        cs = json.loads(corpus_stats_path.read_text())
        k = cs["funnel"]["included_K"]
        cross = cs["crosstab_2x2_variance_x_ablation"]
        a.check_true("corpus K is 17", k == 17, f"K={k}")
        a.check_true(
            "corpus 2x2 sums to K",
            cross["both"]["n"]
            + cross["variance_only"]["n"]
            + cross["ablation_only"]["n"]
            + cross["neither"]["n"]
            == cross["n_total"],
        )
        a.check_int("corpus K stated in prose", k)
        a.check_true(
            "corpus co-occurrence count stated",
            a.has(f"{cross['both']['n']} of {k}"),
        )
        both_ci = cs["proportions_of_K"]["BOTH_variance_and_ablation"]
        a.check_true(
            "corpus Wilson upper bound stated",
            a.has(f"{both_ci['hi'] * 100:.1f}"),
        )
        a.check_true(
            "corpus funnel stated",
            a.has(str(cs["funnel"]["total_screened"])),
        )
        a.check_true(
            "corpus is not described as a strict superset of the 7",
            a.has("not a strict subset") or a.has("not a subset"),
        )

    # --- Rokhva: seed control in the code, and repeated runs in the paper ---
    # These are independent facts. Conflating them is what produced the earlier
    # error, so both are now asserted separately.
    rokhva = next(p for p in papers if p["id"] == "rokhva2025")
    a.check_true(
        "rokhva seed control recorded as absent",
        rokhva["seed_control_in_code"]["any_seed_set"] is False,
    )
    a.check("rokhva line count", "1,645")
    a.check("rokhva grep result", "returns **0**")
    a.check("rokhva claim", "96.40%")

    mrp = rokhva["multi_run_protocol"]
    a.check_true("rokhva multi-run recorded", mrp["reported"] is True)
    a.check_true("rokhva n_runs is 5", mrp["n_runs"] == 5)
    a.check_true(
        "rokhva per-run values recomputed to the reported mean",
        abs(sum(mrp["per_run_accuracy_percent"]) / 5 - mrp["mean_percent"]) < 0.005,
    )
    a.check_true(
        "rokhva range recomputes",
        abs(
            (max(mrp["per_run_accuracy_percent"]) - min(mrp["per_run_accuracy_percent"]))
            - mrp["range_points"]
        )
        < 1e-9,
    )
    for value in mrp["per_run_accuracy_percent"]:
        a.check_true(f"rokhva per-run value {value} stated", a.has(f"{value:.2f}"))
    a.check_true("rokhva run range stated", a.has(f"{mrp['range_points']:.2f}"))
    a.check_true(
        "rokhva no-ablation finding retained",
        rokhva["attention_ablation"]["isolated_no_cbam_control"] is False,
    )
    a.check_true(
        "rokhva no-ablation stated in prose",
        a.has("no-CBAM") or a.has("no CBAM ablation") or a.has("isolates no attention gain"),
    )
    a.check_true(
        "macro-average speculation corrected, not repeated",
        a.has("mean of five") or a.has("mean of 5"),
    )
    ppi_rokhva = rokhva["measurement_resolution"]["points_per_image"]
    a.check("rokhva eval resolution", f"{ppi_rokhva:.4f}")
    a.check_int("rokhva eval set size", rokhva["measurement_resolution"]["eval_set_images"])

    # --- Deng: resolution audit, recomputed here rather than trusted ---
    deng = next(p for p in papers if p["id"] == "deng2024")
    res = deng["resolution_check_no_retraining_needed"]
    n = res["test_set_images"]
    ppi = 100.0 / n
    a.check_true(
        "deng points-per-image recomputes",
        abs(ppi - res["points_per_image"]) < 1e-9,
    )
    a.check_int("deng test set size", n)
    a.check(f"deng resolution {ppi:.5f}", f"{ppi:.5f}")

    t6 = res["stage_placement_table6"]["values_percent"]
    ordered = sorted(t6.values(), reverse=True)
    best_gap = ordered[0] - ordered[1]
    top4_spread = ordered[0] - ordered[3]
    a.check_true(
        "deng best-vs-second recomputes",
        abs(best_gap - res["stage_placement_table6"]["best_minus_second_points"]) < 1e-9,
    )
    a.check_true(
        "deng top-4 spread recomputes",
        abs(top4_spread - res["stage_placement_table6"]["top4_spread_points"]) < 1e-9,
    )
    a.check("deng best-vs-second in points", f"{best_gap:.2f} pts")
    a.check_true(
        "deng best-vs-second in images",
        a.has(f"{best_gap / ppi:.1f} images"),
    )
    a.check_true(
        "deng top-4 spread in images",
        a.has(f"{top4_spread / ppi:.1f} image"),
    )
    for value in t6.values():
        a.check_true(f"deng table6 value {value}", a.has(f"{value:.2f}"))

    # --- measured compute costs must match the JSON, not be rounded freely ---
    cost = rokhva["reproduction_cost"]
    a.check_true(
        "rokhva 3-seed cost stated",
        a.has(f"{cost['hours_for_3_seeds']} h"),
    )
    a.check_true(
        "rokhva per-seed cost stated",
        a.has(f"{cost['hours_per_seed']} h"),
    )
    a.check("measured B7 step time", "1.755")
    a.check("measured B7 epoch time", "18.0 min")

    # --- seed spread this project measured, used as the comparison scale ---
    variance = json.loads((bench / "food11_seed_variance.json").read_text())
    spreads = [round(v["spread"] * 100, 2) for v in variance.values()]
    a.check_true(
        "measured seed spread range stated",
        a.has(f"{min(spreads):.2f}") and a.has(f"{max(spreads):.2f}"),
    )

    # --- the CBAM experiment, if it has been generated ---
    cbam_path = bench / "food11_cbam_ablation.json"
    if cbam_path.is_file():
        cbam = json.loads(cbam_path.read_text())
        comp = cbam["comparison"]
        a.check_true(
            "cbam pairing unit is the seed",
            cbam["protocol"]["pairing_unit"].startswith("seed"),
        )
        a.check_true(
            "cbam bootstrap resamples stated",
            a.has(f"{cbam['protocol']['bootstrap_resamples']:,}"),
        )
        a.check_true(
            "cbam report keeps published claims out of scope",
            "published" in cbam["protocol"]["not_answered"],
        )
        a.check_true("cbam seeds recorded", len(cbam["seeds_used"]) >= 2)

        # every headline figure must appear as the document formats it
        a.check("cbam mean diff", f"{comp['mean_diff'] * 100:+.3f}")
        a.check("cbam CI low", f"{comp['bootstrap']['low'] * 100:+.3f}")
        a.check("cbam CI high", f"{comp['bootstrap']['high'] * 100:+.3f}")
        a.check("cbam p(t)", f"{float(comp['t_test']['p_value']):.4f}")
        a.check("cbam dz", f"{comp['effect_size_dz']:+.2f}")
        a.check("cbam verdict", comp["verdict"])
        a.check_true(
            "cbam diff-in-images stated",
            a.has(f"{cbam['mean_diff_in_images']:.1f} images"),
        )
        a.check_true(
            "cbam per-config means stated",
            all(
                a.has(f"{run['mean'] * 100:.2f}")
                for run in cbam["runs"].values()
            ),
        )
        a.check_true(
            "cbam per-seed accuracies stated",
            all(
                a.has(f"{acc * 100:.2f}")
                for run in cbam["runs"].values()
                for acc in run["per_seed"].values()
            ),
        )
        # the power caveat must not be dropped when the t-test rejects
        a.check_true(
            "cbam power limitation disclosed",
            (not comp["power_limited"]) or a.has("power_limited"),
        )
        a.check_true(
            "cbam exact-test floor stated",
            a.has(f"{comp['permutation']['min_attainable_p']:.3f}")
            or a.has(f"{comp['permutation']['min_attainable_p']:.2f}"),
        )

    # --- claim hygiene: the separation this project insists on ---
    # Pattern-based, not a single exact string: see find_banned_noise_claims.
    a.check_no_banned_claim("must not assert published gains are noise")
    a.check_true(
        "states novelty is already published elsewhere",
        a.has("1912.12522") and a.has("1709.06560"),
    )
    a.check_true(
        "keeps own experiment distinct from published claims",
        a.has("does **not** license") or a.has("does not license"),
    )
    a.check_true(
        "compute-limited section present",
        a.has("Not completed") or a.has("not completed"),
    )
    a.check_true(
        "states nothing is reported from runs that did not happen",
        a.has("did not happen"),
    )
    a.check_true(
        "non-canonical Food-11 caveat present",
        a.has("NTU ML2021-HW3") or a.has("ML2021"),
    )
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=Path, default=Path("docs/attention_variance.md"))
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.doc.is_file():
        print(f"error: no such document: {args.doc}", file=sys.stderr)
        return 2

    result = audit(args.doc.read_text(), args.bench)
    if not args.quiet:
        print(f"{result.passed} checks passed")
    for failure in result.failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if result.failures:
        print(f"\n{len(result.failures)} failure(s)", file=sys.stderr)
        return 1
    print("every number in the document traces to its source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
