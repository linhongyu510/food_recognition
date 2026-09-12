#!/usr/bin/env python
"""Build the CBAM-vs-baseline paired report from run directories.

This is the one experiment in the attention-variance audit that this project
can actually afford to run end to end: EfficientNet-B0 with and without CBAM,
three shared seeds each, everything else held fixed. It answers the narrow
question the published literature leaves open -- *is a CBAM gain of the size
people report distinguishable from seed noise?* -- on this project's own
infrastructure, where every input is under control.

It deliberately does **not** claim to reproduce any specific paper. See
``docs/benchmarks/attention_claims_audit.json`` for why the published works
could not be re-run, and ``docs/attention_variance.md`` for how this experiment
substitutes for that.

Usage::

    python scripts/cbam_ablation_report.py \\
        --run cbam=../abl/runs/cbamabl_efficientnet_b0_cbam_s{seed} \\
        --run nocbam=../abl/runs/cbamabl_efficientnet_b0_s{seed} \\
        --seeds 0 1 2 \\
        --json-out docs/benchmarks/food11_cbam_ablation.json

Each ``--run`` is ``NAME=PATH_TEMPLATE`` where the template contains ``{seed}``.
Accuracies are read from each run's ``metrics.json``; nothing is transcribed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - depends on invocation style
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from food_recognition.significance import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    accuracy_resolution,
    compare_cells,
)


def load_accuracies(template: str, seeds: list[int]) -> dict[int, float]:
    """Read ``metrics.json`` accuracy for each seed, keyed by seed."""
    out: dict[int, float] = {}
    for seed in seeds:
        metrics = Path(template.format(seed=seed)) / "metrics.json"
        if not metrics.is_file():
            continue
        out[seed] = float(json.loads(metrics.read_text())["accuracy"])
    return out


def parse_run(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH_TEMPLATE, got {spec!r}")
    name, template = spec.split("=", 1)
    if "{seed}" not in template:
        raise argparse.ArgumentTypeError(f"template must contain {{seed}}: {template!r}")
    return name, template


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=TEMPLATE")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--resolution", type=int, default=660)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if len(args.run) != 2:
        print("error: need exactly two --run entries to pair", file=sys.stderr)
        return 2

    runs: dict[str, dict[int, float]] = {}
    for spec in args.run:
        name, template = parse_run(spec)
        runs[name] = load_accuracies(template, args.seeds)

    names = list(runs)
    missing = {n: sorted(set(args.seeds) - set(runs[n])) for n in names}
    for name, gaps in missing.items():
        if gaps:
            print(f"warning: {name} missing seeds {gaps}", file=sys.stderr)

    shared = sorted(set(runs[names[0]]) & set(runs[names[1]]))
    if len(shared) < 2:
        print(
            f"error: need >=2 shared seeds, found {shared}. "
            "Runs are probably still training.",
            file=sys.stderr,
        )
        return 1

    comparison = compare_cells(
        names[0],
        runs[names[0]],
        names[1],
        runs[names[1]],
        alpha=args.alpha,
        n_resamples=args.resamples,
        seed=args.rng_seed,
    )
    resolution = accuracy_resolution(args.resolution)

    print(f"{'run':10s} {'n':>2s} {'mean %':>8s}   per-seed %")
    for name in names:
        accs = runs[name]
        vals = [accs[s] for s in shared]
        per = "  ".join(f"{v * 100:.2f}" for v in vals)
        print(f"{name:10s} {len(vals):2d} {sum(vals) / len(vals) * 100:8.2f}   {per}")
    print()
    print(comparison.format_line())
    print()
    print(
        f"measurement resolution: {resolution.n_images} images -> "
        f"1 image = {resolution.points_per_image:.4f} pts"
    )
    diff_images = comparison.mean_diff / (resolution.points_per_image / 100.0)
    print(f"the mean difference is worth {diff_images:.1f} images")
    if comparison.power_limited:
        print(
            f"note: with {len(shared)} seeds the exact test's floor is "
            f"{comparison.permutation['min_attainable_p']:.3f}, above alpha="
            f"{args.alpha}; an assumption-free rejection is impossible by design"
        )

    if args.json_out:
        report = {
            "protocol": {
                "question": (
                    "Holding everything but the CBAM block fixed, is the CBAM effect "
                    "on this setup separable from seed noise?"
                ),
                "pairing_unit": "seed (same seed, same split, same schedule)",
                "bootstrap_resample_unit": "one paired per-seed difference",
                "bootstrap_resamples": args.resamples,
                "bootstrap_rng_seed": args.rng_seed,
                "alpha": args.alpha,
                "tails": 2,
                "multiplicity_correction": "none; a single pre-planned comparison",
                "not_answered": (
                    "whether any specific published CBAM gain falls inside its own "
                    "seed noise; that needs those papers retrained, see "
                    "docs/benchmarks/attention_claims_audit.json"
                ),
            },
            "runs": {
                name: {
                    "per_seed": {str(s): runs[name][s] for s in shared},
                    "mean": sum(runs[name][s] for s in shared) / len(shared),
                    "missing_seeds": missing[name],
                }
                for name in names
            },
            "seeds_used": shared,
            "comparison": comparison.to_dict(),
            "measurement_resolution": resolution.to_dict(),
            "mean_diff_in_images": diff_images,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
