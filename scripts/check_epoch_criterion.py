"""Check whether a paired verdict survives changing the epoch-selection rule.

Every accuracy this project reports is the **best** validation accuracy over the
30-epoch schedule, and ``metrics.json`` is written from the epoch that achieved
it. That rule is applied identically to every run, so paired comparisons are
internally consistent -- but the epoch it lands on varies from 13 to 30 across
cells, and "best of 30" is itself a maximum over 30 correlated draws, so it is
biased upward by an amount that depends on how noisy each run's curve is.

If cell A only beats cell B because A's curve happened to spike at some epoch,
the gap should evaporate under the **last**-epoch rule, which involves no
selection at all. This script recomputes every paired comparison under both
rules from the runs' own ``history.json`` and flags any pair whose verdict or
sign depends on the choice::

    python scripts/check_epoch_criterion.py \\
        --run b0_300=../abl/runs/abl_b0_300 --run b0_300=../seeds/b0_300_s1 ...

or, more conveniently, point it at a manifest::

    python scripts/check_epoch_criterion.py --manifest docs/benchmarks/seed_runs.json

The manifest maps ``cell -> {seed: run_directory}``. Paths are resolved relative
to the manifest file, so it can live in the repo while the runs sit outside it.
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
    pairwise_comparisons,
)


def read_history(run_dir: Path) -> list[float]:
    """Return the per-epoch validation accuracies recorded by a run."""
    payload = json.loads((run_dir / "history.json").read_text())
    epochs = payload if isinstance(payload, list) else payload["epochs"]
    accs = [float(e["val_acc"]) for e in epochs if "val_acc" in e]
    if not accs:
        raise ValueError(f"no val_acc entries in {run_dir / 'history.json'}")
    return accs


def criteria_from_history(accs: list[float]) -> dict[str, float | int]:
    """Both selection rules plus the epoch the maximum landed on (1-based)."""
    best = max(accs)
    return {
        "best": best,
        "last": accs[-1],
        "best_epoch": accs.index(best) + 1,
        "n_epochs": len(accs),
        "selection_gain": best - accs[-1],
    }


def build(runs: dict[str, dict[int, Path]]) -> dict[str, dict]:
    """Read every run once and return per-cell, per-seed criterion values."""
    out: dict[str, dict] = {}
    for cell, seeds in sorted(runs.items()):
        for seed, path in sorted(seeds.items()):
            out.setdefault(cell, {})[seed] = criteria_from_history(read_history(path))
    return out


def accuracies(table: dict[str, dict], rule: str) -> dict[str, dict[int, float]]:
    return {
        cell: {seed: float(vals[rule]) for seed, vals in seeds.items()}
        for cell, seeds in table.items()
    }


def compare_rules(
    table: dict[str, dict],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Recompute all pairs under both rules and mark the unstable ones."""
    per_rule = {}
    for rule in ("best", "last"):
        per_rule[rule] = {
            (c.cell_a, c.cell_b): c
            for c in pairwise_comparisons(
                accuracies(table, rule), alpha=alpha, n_resamples=n_resamples, seed=seed
            )
        }

    rows = []
    for key, best in per_rule["best"].items():
        # pairwise_comparisons orders each pair by mean, so the same two cells
        # may come back swapped under the other rule; normalise before comparing.
        last = per_rule["last"].get(key)
        flipped = False
        if last is None:
            last = per_rule["last"].get((key[1], key[0]))
            flipped = last is not None
        if last is None:
            continue
        last_diff = -last.mean_diff if flipped else last.mean_diff
        rows.append(
            {
                "pair": f"{key[0]} vs {key[1]}",
                "best_diff_points": best.mean_diff * 100,
                "last_diff_points": last_diff * 100,
                "best_verdict": best.verdict,
                "last_verdict": last.verdict,
                "sign_flips": (best.mean_diff > 0) != (last_diff > 0),
                "verdict_changes": best.verdict != last.verdict,
                "best_p": best.t_test["p_value"],
                "last_p": last.t_test["p_value"],
            }
        )
    rows.sort(key=lambda r: -abs(float(r["best_diff_points"])))

    gains = [
        float(v["selection_gain"]) for seeds in table.values() for v in seeds.values()
    ]
    return {
        "criterion_note": (
            "every reported accuracy is the maximum validation accuracy over the "
            "schedule ('best'); 'last' is the final epoch, which involves no "
            "selection. Both rules are applied uniformly to all runs."
        ),
        "selection_gain_points": {
            "mean": sum(gains) / len(gains) * 100,
            "max": max(gains) * 100,
            "min": min(gains) * 100,
        },
        "best_epoch_range": [
            min(int(v["best_epoch"]) for seeds in table.values() for v in seeds.values()),
            max(int(v["best_epoch"]) for seeds in table.values() for v in seeds.values()),
        ],
        "per_run": {
            cell: {str(s): v for s, v in seeds.items()} for cell, seeds in table.items()
        },
        "pairs": rows,
        "unstable_pairs": [r["pair"] for r in rows if r["sign_flips"] or r["verdict_changes"]],
    }


def _parse_runs(specs: list[str], base: Path) -> dict[str, dict[int, Path]]:
    runs: dict[str, dict[int, Path]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--run needs cell=path, got {spec!r}")
        cell, raw = spec.split("=", 1)
        path = (base / raw).resolve()
        name = path.name
        seed = int(name.rsplit("_s", 1)[1]) if "_s" in name else 0
        runs.setdefault(cell, {})[seed] = path
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[], metavar="CELL=DIR")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    if args.manifest:
        base = args.manifest.parent
        raw = json.loads(args.manifest.read_text())
        runs = {
            cell: {int(s): (base / p).resolve() for s, p in seeds.items()}
            for cell, seeds in raw.items()
        }
    elif args.run:
        runs = _parse_runs(args.run, Path.cwd())
    else:
        print("error: pass --manifest or at least one --run", file=sys.stderr)
        return 2

    missing = [
        str(p)
        for seeds in runs.values()
        for p in seeds.values()
        if not (p / "history.json").is_file()
    ]
    if missing:
        print(f"error: no history.json in: {', '.join(missing)}", file=sys.stderr)
        return 1

    report = compare_rules(
        build(runs), alpha=args.alpha, n_resamples=args.resamples, seed=args.rng_seed
    )

    gain = report["selection_gain_points"]
    lo, hi = report["best_epoch_range"]
    print(
        f"best-epoch selection: epochs {lo}-{hi}; best exceeds last by "
        f"{gain['mean']:+.3f} pts on average (max {gain['max']:+.3f})"
    )
    print(f"\n{'pair':22s} {'best pts':>9s} {'last pts':>9s}  {'flips':>5s}  verdicts")
    for r in report["pairs"]:
        flag = "YES" if r["sign_flips"] else "no"
        print(
            f"{r['pair']:22s} {float(r['best_diff_points']):+9.3f} "
            f"{float(r['last_diff_points']):+9.3f}  {flag:>5s}  "
            f"{r['best_verdict']} -> {r['last_verdict']}"
        )

    if report["unstable_pairs"]:
        print(
            "\nverdict or sign depends on the epoch rule for: "
            + ", ".join(report["unstable_pairs"])
        )
    else:
        print("\nno pair changes sign or verdict between the two rules")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
