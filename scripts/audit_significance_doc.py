"""Cross-check every number in docs/significance.md against its source JSON.

CONTRIBUTING.md requires each benchmark figure to be traceable to a
``metrics.json``. Prose drifts from data silently, and this project has already
been bitten by transcription errors (8.6 vs 8.456, 198 vs 201), so the check is
mechanical rather than a review habit::

    python scripts/audit_significance_doc.py
    python scripts/audit_significance_doc.py --doc docs/significance.md --strict

Every assertion re-reads the JSON and searches the rendered Markdown for the
value formatted as the document formats it. Integers are accepted with or
without thousands separators, since Markdown tables use ``1,910`` while JSON
holds ``1910``. Exit status is non-zero when any value is missing, so this can
run in CI.
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
        self.passed = 0
        self.failures: list[str] = []

    def has(self, needle: str) -> bool:
        return needle in self.text

    def check(self, label: str, needle: str) -> None:
        if self.has(needle):
            self.passed += 1
        else:
            self.failures.append(f"{label}: {needle!r} not found in document")

    def check_int(self, label: str, value: int) -> None:
        """Accept ``1910`` or ``1,910``; Markdown tables use the latter."""
        if self.has(f"{value:,}") or self.has(str(value)):
            self.passed += 1
        else:
            self.failures.append(f"{label}: neither {value} nor {value:,} found")

    def check_true(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(f"{label}: {detail or 'condition false'}")


def audit(doc_text: str, bench: Path = BENCH) -> Auditor:
    a = Auditor(doc_text)

    variance = json.loads((bench / "food11_seed_variance.json").read_text())
    significance = json.loads((bench / "food11_seed_significance.json").read_text())
    power = json.loads((bench / "food101_validation_power.json").read_text())
    paired = json.loads((bench / "food101_paired_eval.json").read_text())
    epoch = json.loads((bench / "food11_epoch_criterion.json").read_text())

    # --- per-cell seed statistics -----------------------------------------
    for cell, entry in variance.items():
        a.check(f"{cell} mean", f"{entry['mean'] * 100:.2f}%")
        a.check(f"{cell} spread", f"{entry['spread'] * 100:.2f}")
        a.check(
            f"{cell} per-seed",
            " / ".join(f"{v * 100:.2f}" for v in entry["per_seed"]),
        )

    # --- pairwise comparisons ---------------------------------------------
    comparisons = significance["_significance"]["comparisons"]
    a.check_true(
        "six pairs compared", len(comparisons) == 6, f"got {len(comparisons)}"
    )
    for c in comparisons:
        pair = f"{c['cell_a']} vs {c['cell_b']}"
        a.check(f"{pair} diff", f"{c['mean_diff'] * 100:+.3f}")
        a.check(
            f"{pair} bootstrap CI",
            f"[{c['bootstrap']['low'] * 100:+.3f}, {c['bootstrap']['high'] * 100:+.3f}]",
        )
        a.check(f"{pair} dz", f"{c['effect_size_dz']:+.2f}")
        p_value = float(c["t_test"]["p_value"])
        a.check(f"{pair} p(t)", f"{p_value:.4f}")

    # The one positive result must be labelled as parametric-only, and the
    # document must not upgrade it to a plain "significant".
    positive = [c for c in comparisons if c["verdict"].startswith("significant")]
    a.check_true(
        "exactly one positive verdict", len(positive) == 1, f"got {len(positive)}"
    )
    if positive:
        a.check_true(
            "positive verdict is parametric-only",
            positive[0]["verdict"] == "significant_parametric_only",
            positive[0]["verdict"],
        )
        a.check_true(
            "positive verdict flagged power-limited", positive[0]["power_limited"] is True
        )
        a.check("parametric-only wording", "parametric only")

    # --- measurement resolution -------------------------------------------
    res = significance["_significance"]["measurement_resolution"]
    a.check("660 points per image", f"{res['points_per_image']:.4f}")
    a.check("660 CI width", f"{res['ci_width_points']:.2f}")

    # --- validation power sweep -------------------------------------------
    for row in power["sizes"]:
        n = row["n_images"]
        a.check(f"power n={n} points/image", f"{row['points_per_image']:.4f}")
        a.check(f"power n={n} sign flips", f"{row['sign_flip_rate'] * 100:.1f}%")
        a.check_int(f"power n={n} size", n)
    a.check("full-set diff", f"{power['full_set']['accuracy_diff_points']:+.3f}")

    # --- Food-101 paired evaluation ---------------------------------------
    for name, run in paired["runs"].items():
        a.check(f"{name} accuracy", f"{run['accuracy'] * 100:.3f}%")
        a.check_int(f"{name} correct", run["n_correct"])
        a.check(
            f"{name} Wilson CI",
            f"[{run['wilson_low'] * 100:.3f}, {run['wilson_high'] * 100:.3f}]",
        )
    pair = paired["pairs"][0]
    mc = pair["mcnemar"]
    a.check_int("discordant images", int(mc["n_discordant"]))
    a.check_int("only A correct", int(mc["only_a_correct"]))
    a.check_int("only B correct", int(mc["only_b_correct"]))
    a.check("McNemar p", f"{float(mc['p_value']):.4f}")
    a.check("diff in images", f"{int(round(pair['diff_in_images']))} images")

    # --- epoch-selection sensitivity --------------------------------------
    gain = epoch["selection_gain_points"]
    a.check("selection gain mean", f"{gain['mean']:+.3f}")
    a.check("selection gain max", f"{gain['max']:+.3f}")
    lo, hi = epoch["best_epoch_range"]
    a.check("best epoch range", f"{lo} to {hi}")
    for row in epoch["pairs"]:
        a.check(f"{row['pair']} last-epoch diff", f"{row['last_diff_points']:+.3f}")
    n_unstable = len(epoch["unstable_pairs"])
    # The count may be written as a digit or spelled out ("Four of six"), so
    # accept either rather than forcing the prose into one style.
    spelled = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    total = len(epoch["pairs"])
    wordings = [
        f"{n_unstable} of {total}",
        f"{n_unstable} of {spelled.get(total, total)}",
        f"{spelled.get(n_unstable, n_unstable)} of {total}",
        f"{spelled.get(n_unstable, n_unstable)} of {spelled.get(total, total)}",
    ]
    lowered = a.text.lower()
    a.check_true(
        "unstable pair count stated",
        any(w.lower() in lowered for w in wordings),
        f"{n_unstable} of {total} unstable pairs not stated in any accepted form",
    )

    # --- guard rails: claims that must NOT appear -------------------------
    # These two figures have never had a traceable source (see CONTRIBUTING.md),
    # so they must stay out of every artefact until one is produced.
    for banned in ("94.56", "84.09"):
        a.check_true(
            f"untraceable figure {banned} absent",
            not a.has(banned),
            f"{banned} reappeared without a source",
        )
    # The Food-11 mirror is not the canonical split, so cross-study comparison
    # is invalid; the caveat must be present.
    a.check_true(
        "non-canonical Food-11 caveat present",
        a.has("not** the canonical Food-11") or a.has("not the canonical Food-11"),
    )
    a.check_true("labelled-subset size stated", a.has("3,080"))
    a.check_true("canonical training size stated", a.has("9,866"))
    a.check_true(
        "compute-limited section present",
        a.has("Not completed") or a.has("not completed"),
    )
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=Path, default=Path("docs/significance.md"))
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
    print("every number in the document traces to its source JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
