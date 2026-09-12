"""Measure what enlarging the validation set actually buys, by subsampling it.

The statistical-power argument for a bigger validation set is usually made from
a formula. This measures it directly on real predictions: it takes two runs'
image-level correctness vectors over a large labelled set, draws subsets of
increasing size *without replacement*, and records how often each subset size
would have (a) recovered the sign of the full-set difference and (b) reached
significance under McNemar's exact test.

The full set is the ground truth here, so "power" is power against a difference
known to exist rather than against an assumed one::

    python scripts/validation_power.py \\
        --run-a reval/b4_correct.json --run-b reval/b0_correct.json \\
        --sizes 660 2000 5000 12000 25250 \\
        --json-out docs/benchmarks/food101_validation_power.json

The 660 row matters because that is the size of the Food-11 validation split
this project's ablation grid was ranked on; the sign-flip rate at that size is a
direct estimate of how often that ranking was decided by which images happened
to be in the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - depends on invocation style
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from food_recognition.significance import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_SEED,
    accuracy_resolution,
    mcnemar_exact,
)

DEFAULT_TRIALS = 400


def sweep(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    sizes: list[int],
    *,
    trials: int = DEFAULT_TRIALS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict:
    """For each subset size, estimate sign recovery and significance rate."""
    if correct_a.shape != correct_b.shape:
        raise ValueError("correctness vectors must align image-by-image")
    n_total = int(correct_a.size)
    bad = [n for n in sizes if not 1 <= n <= n_total]
    if bad:
        raise ValueError(f"sizes must be in [1, {n_total}], got {bad}")

    full = mcnemar_exact(correct_a, correct_b)
    full_diff = float(full["accuracy_diff"])
    reference_sign = 1 if full_diff > 0 else (-1 if full_diff < 0 else 0)

    rng = np.random.default_rng(seed)
    rows = []
    for n in sorted(sizes):
        p_values = np.empty(trials)
        diffs = np.empty(trials)
        for t in range(trials):
            # Without replacement: a subset of the real validation set, which is
            # what "we could only afford to label N images" actually looks like.
            idx = rng.choice(n_total, size=n, replace=False)
            res = mcnemar_exact(correct_a[idx], correct_b[idx])
            p_values[t] = float(res["p_value"])
            diffs[t] = float(res["accuracy_diff"])

        signs = np.sign(diffs)
        res_n = accuracy_resolution(n, alpha=alpha)
        rows.append(
            {
                "n_images": n,
                "trials": trials,
                "points_per_image": res_n.points_per_image,
                "single_run_ci_width_points": res_n.ci_width_points,
                "power_at_alpha": float((p_values < alpha).mean()),
                "median_p": float(np.median(p_values)),
                "mean_diff_points": float(diffs.mean() * 100),
                "sd_diff_points": float(diffs.std(ddof=1) * 100) if trials > 1 else 0.0,
                "sign_flip_rate": float((signs != reference_sign).mean()),
                "diff_below_one_image_rate": float(
                    (np.abs(diffs) * n < 1.0).mean()
                ),
            }
        )

    return {
        "protocol": {
            "method": (
                "subsample the full labelled validation set without replacement, "
                "re-run McNemar's exact test on each subset"
            ),
            "resample_unit": "validation image",
            "trials_per_size": trials,
            "alpha": alpha,
            "rng_seed": seed,
            "reference": (
                "the full-set difference is treated as ground truth, so "
                "power is measured against a difference known to be present"
            ),
            "caveat": (
                "this isolates label-sampling noise for two fixed checkpoints; it "
                "does not capture training-seed noise"
            ),
        },
        "full_set": {
            "n_images": n_total,
            "accuracy_diff_points": full_diff * 100,
            "p_value": full["p_value"],
            "n_discordant": full["n_discordant"],
            "only_a_correct": full["only_a_correct"],
            "only_b_correct": full["only_b_correct"],
        },
        "sizes": rows,
    }


def print_sweep(report: dict) -> None:
    full = report["full_set"]
    print(
        f"full set: {full['n_images']} images, difference "
        f"{full['accuracy_diff_points']:+.3f} pts, p={float(full['p_value']):.4g}, "
        f"{full['n_discordant']} discordant images"
    )
    print(
        f"\n{'n':>7s} {'1 img pts':>10s} {'power':>7s} {'median p':>9s} "
        f"{'diff sd pts':>12s} {'sign flips':>11s}"
    )
    for row in report["sizes"]:
        print(
            f"{row['n_images']:7d} {row['points_per_image']:10.4f} "
            f"{row['power_at_alpha'] * 100:6.1f}% {row['median_p']:9.3f} "
            f"{row['sd_diff_points']:12.3f} {row['sign_flip_rate'] * 100:10.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    payload_a = json.loads(args.run_a.read_text())
    payload_b = json.loads(args.run_b.read_text())
    if payload_a.get("order_sha256_16") != payload_b.get("order_sha256_16"):
        print("error: the two runs were scored over different image orders", file=sys.stderr)
        return 1

    try:
        report = sweep(
            np.asarray(payload_a["correct"]),
            np.asarray(payload_b["correct"]),
            args.sizes,
            trials=args.trials,
            alpha=args.alpha,
            seed=args.rng_seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report["runs"] = {
        "a": {"path": str(args.run_a), "checkpoint": payload_a.get("checkpoint")},
        "b": {"path": str(args.run_b), "checkpoint": payload_b.get("checkpoint")},
    }
    print_sweep(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
