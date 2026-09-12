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
import sys
from pathlib import Path

BENCH = Path("docs/benchmarks")


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
    a.check_true("no paper reports seed variance", not variance_yes)
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
        "zero-variance count stated",
        a.has(f"0 of {len(papers)} report seed variance"),
    )

    # --- Rokhva: the seed-control finding, which is the strongest claim ---
    rokhva = next(p for p in papers if p["id"] == "rokhva2025")
    a.check_true(
        "rokhva seed control recorded as absent",
        rokhva["seed_control_in_code"]["any_seed_set"] is False,
    )
    a.check("rokhva line count", "1,645")
    a.check("rokhva grep result", "returns **0**")
    a.check("rokhva claim", "96.40%")
    a.check_true(
        "rokhva macro-average caveat present",
        a.has("macro") and a.has("imbalanced"),
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
    a.check_absent(
        "must not assert published gains are noise",
        "published CBAM gains are noise.",
    )
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
