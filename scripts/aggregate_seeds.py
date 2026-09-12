"""Aggregate multi-seed runs into mean/std per grid cell, and test the gaps.

The single-seed grid in ``docs/benchmarks/food11_ablation_grid.json`` ranks nine
cells, but a rank is only meaningful next to the run-to-run noise. This reads
per-seed ``metrics.json`` files and reports mean, sample standard deviation and
spread for every cell, so a claimed gap can be compared against the seed noise
underneath it.

Means alone still do not say whether a gap is real, so ``--significance`` adds a
paired analysis of every cell pair via :mod:`food_recognition.significance`:
a paired t-test, an exact sign-flip permutation test, a percentile bootstrap
confidence interval and the paired effect size. The protocol is fixed and
recorded in the output:

* **pairing unit** -- one seed. Cells are compared only on seeds they share, and
  the i-th difference is ``acc_A(seed_i) - acc_B(seed_i)``: same seed, same
  split, same schedule, so only the configuration differs.
* **bootstrap resampling unit** -- one paired per-seed difference, drawn with
  replacement. Images are never the resampling unit here, because the quantity
  being bounded is training noise, not label-sampling noise.
* **resamples** -- ``--resamples`` (default 10,000); **alpha** -- ``--alpha``
  (default 0.05, two-sided).

``--resolution N`` additionally reports how finely a validation set of ``N``
images can measure accuracy at all, which is a separate and additive source of
uncertainty from seed noise.

Usage::

    python scripts/aggregate_seeds.py SEED_DIR --grid docs/benchmarks/food11_ablation_grid.json
    python scripts/aggregate_seeds.py SEED_DIR --significance --resolution 660 \\
        --json-out docs/benchmarks/food11_seed_significance.json

``SEED_DIR`` holds directories named ``<cell>_s<seed>``, each containing the
``metrics.json`` written by ``food-recognition-train``. Seed 0 is read from the
grid JSON when present, so it does not need to be re-run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Importable both as a script and as a module: tests load this file directly by
# path, so a plain package import must keep working without the repo on sys.path.
if __package__ in (None, ""):  # pragma: no cover - depends on invocation style
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from food_recognition.significance import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    accuracy_resolution,
    pairwise_comparisons,
)

RUN_RE = re.compile(r"^(?P<cell>.+)_s(?P<seed>\d+)$")


def collect(
    seed_dir: Path, grid: dict | None, seeds: set[int] | None = None
) -> dict[str, dict[int, float]]:
    """Map cell -> {seed: accuracy}, merging seed 0 from the grid JSON.

    ``seeds``, when given, restricts the result to exactly those seed numbers.
    Without it the report silently changes shape as new runs land in
    ``seed_dir``: a report written while a sweep was still going had one cell at
    n=4 and the rest at n=3, which contradicted the n=3 table it was cited for.
    Pinning the seed set makes a committed report reproducible regardless of
    what else has since finished.
    """
    out: dict[str, dict[int, float]] = {}
    for child in sorted(seed_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_RE.match(child.name)
        metrics = child / "metrics.json"
        if match is None or not metrics.is_file():
            continue
        seed = int(match.group("seed"))
        if seeds is not None and seed not in seeds:
            continue
        acc = json.loads(metrics.read_text())["accuracy"]
        out.setdefault(match.group("cell"), {})[seed] = acc

    if grid:
        for cell in out:
            if cell in grid and 0 not in out[cell] and (seeds is None or 0 in seeds):
                out[cell][0] = grid[cell]["acc"]
    return out


def summarise(accs: dict[int, float]) -> dict[str, float | int]:
    values = [accs[s] for s in sorted(accs)]
    mean = statistics.fmean(values)
    # Sample stdev needs n >= 2; a single seed has no spread to report.
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "seeds": sorted(accs),
        "per_seed": [round(v, 6) for v in values],
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def significance_report(
    runs: dict[str, dict[int, float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    resolution_images: int | None = None,
    seed_filter: set[int] | None = None,
) -> dict:
    """Build the paired-comparison section, protocol included."""
    comparisons = pairwise_comparisons(
        runs, alpha=alpha, n_resamples=n_resamples, seed=seed
    )
    report: dict = {
        "protocol": {
            "pairing_unit": "seed",
            "pairing_detail": (
                "acc_A(seed_i) - acc_B(seed_i) over seeds present in both cells; "
                "same seed means same data order and same schedule"
            ),
            "bootstrap_resample_unit": "seed-level paired difference",
            "bootstrap_resamples": n_resamples,
            "bootstrap_type": "percentile",
            "bootstrap_rng_seed": seed,
            "alpha": alpha,
            "tails": 2,
            "tests": [
                "paired t-test",
                "exact sign-flip permutation test",
                "percentile bootstrap CI",
            ],
            "effect_size": "Cohen's dz (mean difference / SD of differences)",
            "seeds_included": (
                sorted(seed_filter) if seed_filter is not None else "all seeds found on disk"
            ),
            "multiplicity_correction": "none; p-values are per-pair and uncorrected",
            "units": "accuracies are fractions; *_points fields are percentage points",
        },
        "comparisons": [c.to_dict() for c in comparisons],
    }
    if resolution_images is not None:
        res = accuracy_resolution(resolution_images, alpha=alpha)
        report["measurement_resolution"] = res.to_dict()
    return report


def print_significance(report: dict) -> None:
    comparisons = report["comparisons"]
    if not comparisons:
        print("\nno cell pair shares enough seeds to test")
        return

    proto = report["protocol"]
    print(
        f"\npaired comparisons (unit=seed, bootstrap={proto['bootstrap_resamples']} "
        f"resamples over seed-level differences, alpha={proto['alpha']}, two-sided)"
    )
    header = (
        f"{'pair':22s} {'n':>2s} {'diff pts':>9s} {'95% CI pts':>18s} "
        f"{'p(t)':>7s} {'p(perm)':>8s} {'dz':>7s}  verdict"
    )
    print(header)
    for c in comparisons:
        p_t = c["t_test"]["p_value"]
        p_t_text = "n/a" if p_t is None or p_t != p_t else f"{p_t:.4f}"
        dz = c["effect_size_dz"]
        dz_text = "n/a" if dz is None or dz != dz else f"{dz:+.2f}"
        ci = f"[{c['bootstrap']['low'] * 100:+.3f}, {c['bootstrap']['high'] * 100:+.3f}]"
        print(
            f"{c['cell_a'] + ' vs ' + c['cell_b']:22s} {len(c['seeds']):2d} "
            f"{c['mean_diff'] * 100:+9.3f} {ci:>18s} {p_t_text:>7s} "
            f"{c['permutation']['p_value']:8.3f} {dz_text:>7s}  {c['verdict']}"
        )

    if "measurement_resolution" in report:
        res = report["measurement_resolution"]
        print(
            f"\nmeasurement resolution: {res['n_images']} validation images "
            f"-> 1 image = {res['points_per_image']:.4f} pts, "
            f"single-run 95% CI width {res['ci_width_points']:.3f} pts "
            f"at {res['reference_accuracy'] * 100:.0f}% accuracy"
        )

    floors = {c["permutation"]["min_attainable_p"] for c in comparisons}
    if all(f > report["protocol"]["alpha"] for f in floors):
        print(
            "note: with this many seeds the exact permutation test cannot reach "
            f"alpha={report['protocol']['alpha']}; every pair is underpowered by "
            "design, independent of the observed gaps"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_dir", type=Path)
    ap.add_argument("--grid", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--significance",
        action="store_true",
        help="Also run paired t-test / permutation / bootstrap on every cell pair.",
    )
    ap.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="Two-sided significance level."
    )
    ap.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES, help="Bootstrap resamples."
    )
    ap.add_argument(
        "--rng-seed", type=int, default=DEFAULT_SEED, help="Bootstrap RNG seed."
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=None,
        metavar="N",
        help="Report the measurement resolution of an N-image validation set.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated seeds to include, e.g. '0,1,2'. Pins the report to a "
            "fixed seed set so it stays reproducible while further runs land."
        ),
    )
    args = ap.parse_args(argv)

    seeds: set[int] | None = None
    if args.seeds is not None:
        try:
            seeds = {int(part) for part in args.seeds.split(",") if part.strip()}
        except ValueError:
            print(f"error: --seeds must be comma-separated integers: {args.seeds}",
                  file=sys.stderr)
            return 2
        if not seeds:
            print("error: --seeds was empty", file=sys.stderr)
            return 2

    if not args.seed_dir.is_dir():
        print(f"error: not a directory: {args.seed_dir}", file=sys.stderr)
        return 2

    grid = json.loads(args.grid.read_text()) if args.grid else None
    runs = collect(args.seed_dir, grid, seeds)
    if not runs:
        print(f"error: no <cell>_s<seed>/metrics.json under {args.seed_dir}", file=sys.stderr)
        return 1

    report = {cell: summarise(accs) for cell, accs in sorted(runs.items())}

    print(f"{'cell':10s} {'n':>2s} {'mean %':>8s} {'std':>6s} {'spread':>7s}   per-seed %")
    for cell, r in sorted(report.items(), key=lambda kv: -kv[1]["mean"]):
        per = "  ".join(f"{v * 100:.2f}" for v in r["per_seed"])
        print(
            f"{cell:10s} {r['n']:2d} {r['mean'] * 100:8.2f} "
            f"{r['std'] * 100:6.2f} {r['spread'] * 100:7.2f}   {per}"
        )

    singles = [c for c, r in report.items() if r["n"] < 2]
    if singles:
        print(f"\nnote: no spread available for {', '.join(sorted(singles))} (single seed)")

    payload: dict = dict(report)
    if args.significance:
        sig = significance_report(
            runs,
            alpha=args.alpha,
            n_resamples=args.resamples,
            seed=args.rng_seed,
            resolution_images=args.resolution,
            seed_filter=seeds,
        )
        print_significance(sig)
        # Nested under a reserved key so per-cell entries stay addressable by
        # name and existing readers of this JSON keep working.
        payload["_significance"] = sig
    elif args.resolution is not None:
        res = accuracy_resolution(args.resolution, alpha=args.alpha).to_dict()
        print(
            f"\nmeasurement resolution: {res['n_images']} validation images "
            f"-> 1 image = {res['points_per_image']:.4f} pts, "
            f"single-run 95% CI width {res['ci_width_points']:.3f} pts"
        )
        payload["_measurement_resolution"] = res

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
