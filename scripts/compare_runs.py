"""Compare two runs on the same validation set, image by image.

Answers a narrower question than ``scripts/aggregate_seeds.py --significance``:
holding two *already trained* checkpoints fixed, is the validation set large
enough to separate them? The pairing unit is one image, so this measures
label-sampling noise only -- it says nothing about whether retraining either
model with a different seed would reorder the two, which is what the seed-level
analysis is for. Both numbers are needed, and neither substitutes for the other.

Consumes the correctness vectors written by ``scripts/dump_predictions.py``::

    python scripts/compare_runs.py \\
        --run b0_224=reval/b0_correct.json \\
        --run b4_224=reval/b4_correct.json \\
        --json-out docs/benchmarks/food101_paired_eval.json

Reports, for every pair: the accuracy difference, McNemar's exact test on the
discordant images, a paired image-level bootstrap CI, and each run's own Wilson
interval, together with what one image is worth on a set of that size.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
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
    bootstrap_accuracy_diff_ci,
    mcnemar_exact,
    wilson_interval,
)


def load_vector(path: Path) -> dict:
    payload = json.loads(path.read_text())
    for key in ("correct", "n_images", "order_sha256_16"):
        if key not in payload:
            raise ValueError(f"{path} is missing {key!r}; regenerate with dump_predictions.py")
    return payload


def build_report(
    runs: dict[str, dict],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    orders = {name: payload["order_sha256_16"] for name, payload in runs.items()}
    if len(set(orders.values())) > 1:
        raise ValueError(
            "runs were scored over different image orders, so they cannot be paired: "
            + ", ".join(f"{n}={h}" for n, h in orders.items())
        )

    n_images = next(iter(runs.values()))["n_images"]
    res = accuracy_resolution(n_images, alpha=alpha)

    per_run = {}
    for name, payload in sorted(runs.items()):
        n_correct = int(sum(payload["correct"]))
        low, high = wilson_interval(n_correct, n_images, alpha)
        per_run[name] = {
            "n_images": n_images,
            "n_correct": n_correct,
            "accuracy": n_correct / n_images,
            "wilson_low": low,
            "wilson_high": high,
            "wilson_width_points": (high - low) * 100,
            "checkpoint": payload.get("checkpoint"),
            "image_size": payload.get("image_size"),
        }

    pairs = []
    for a, b in combinations(sorted(runs), 2):
        # Order so the reported difference is non-negative.
        if per_run[a]["accuracy"] < per_run[b]["accuracy"]:
            a, b = b, a
        va, vb = runs[a]["correct"], runs[b]["correct"]
        mc = mcnemar_exact(va, vb)
        ci = bootstrap_accuracy_diff_ci(
            va, vb, alpha=alpha, n_resamples=n_resamples, seed=seed
        )
        pairs.append(
            {
                "run_a": a,
                "run_b": b,
                "accuracy_diff": mc["accuracy_diff"],
                "accuracy_diff_points": float(mc["accuracy_diff"]) * 100,
                "diff_in_images": float(mc["accuracy_diff"]) * n_images,
                "mcnemar": mc,
                "bootstrap": ci.to_dict(),
                "significant_at_alpha": bool(float(mc["p_value"]) < alpha)
                and ci.excludes_zero,
            }
        )
    pairs.sort(key=lambda p: -abs(float(p["accuracy_diff"])))

    return {
        "protocol": {
            "question": (
                "given these fixed checkpoints, is the validation set large enough "
                "to separate them?"
            ),
            "not_answered": (
                "whether retraining with another seed would reorder them; that "
                "requires the seed-level paired analysis in aggregate_seeds.py"
            ),
            "pairing_unit": "validation image",
            "bootstrap_resample_unit": "validation image (both runs resampled together)",
            "bootstrap_resamples": n_resamples,
            "bootstrap_type": "percentile",
            "bootstrap_rng_seed": seed,
            "alpha": alpha,
            "tails": 2,
            "tests": ["McNemar exact (binomial on discordant images)", "paired bootstrap CI"],
            "multiplicity_correction": "none; p-values are per-pair and uncorrected",
        },
        "measurement_resolution": res.to_dict(),
        "runs": per_run,
        "pairs": pairs,
    }


def print_report(report: dict) -> None:
    res = report["measurement_resolution"]
    print(
        f"validation set: {res['n_images']} images -> 1 image = "
        f"{res['points_per_image']:.4f} pts"
    )
    print(f"\n{'run':14s} {'accuracy':>9s} {'correct':>9s}  95% Wilson CI (pts wide)")
    for name, r in report["runs"].items():
        print(
            f"{name:14s} {r['accuracy'] * 100:8.3f}% {r['n_correct']:9d}  "
            f"[{r['wilson_low'] * 100:.3f}, {r['wilson_high'] * 100:.3f}] "
            f"({r['wilson_width_points']:.3f})"
        )

    print(f"\n{'pair':24s} {'diff pts':>9s} {'imgs':>7s} {'disc':>6s} {'p':>10s}  verdict")
    for p in report["pairs"]:
        mc = p["mcnemar"]
        verdict = "separable" if p["significant_at_alpha"] else "not separable"
        print(
            f"{p['run_a'] + ' vs ' + p['run_b']:24s} "
            f"{p['accuracy_diff_points']:+9.3f} {p['diff_in_images']:+7.0f} "
            f"{mc['n_discordant']:6d} {float(mc['p_value']):10.3e}  {verdict}"
        )
        print(
            f"{'':24s} only {p['run_a']} right: {mc['only_a_correct']}, "
            f"only {p['run_b']} right: {mc['only_b_correct']}, "
            f"bootstrap CI [{p['bootstrap']['low'] * 100:+.3f}, "
            f"{p['bootstrap']['high'] * 100:+.3f}] pts"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[], metavar="NAME=VECTOR.json")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    if len(args.run) < 2:
        print("error: need at least two --run NAME=VECTOR.json", file=sys.stderr)
        return 2

    runs: dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            print(f"error: --run needs NAME=path, got {spec!r}", file=sys.stderr)
            return 2
        name, raw = spec.split("=", 1)
        runs[name] = load_vector(Path(raw))

    try:
        report = build_report(
            runs, alpha=args.alpha, n_resamples=args.resamples, seed=args.rng_seed
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
