"""Aggregate multi-seed runs into mean/std per grid cell.

The single-seed grid in ``docs/benchmarks/food11_ablation_grid.json`` ranks nine
cells, but a rank is only meaningful next to the run-to-run noise. This reads
per-seed ``metrics.json`` files and reports mean, sample standard deviation and
spread for every cell, so a claimed gap can be compared against the seed noise
underneath it.

Usage::

    python scripts/aggregate_seeds.py SEED_DIR --grid docs/benchmarks/food11_ablation_grid.json

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

RUN_RE = re.compile(r"^(?P<cell>.+)_s(?P<seed>\d+)$")


def collect(seed_dir: Path, grid: dict | None) -> dict[str, dict[int, float]]:
    """Map cell -> {seed: accuracy}, merging seed 0 from the grid JSON."""
    out: dict[str, dict[int, float]] = {}
    for child in sorted(seed_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_RE.match(child.name)
        metrics = child / "metrics.json"
        if match is None or not metrics.is_file():
            continue
        acc = json.loads(metrics.read_text())["accuracy"]
        out.setdefault(match.group("cell"), {})[int(match.group("seed"))] = acc

    if grid:
        for cell in out:
            if cell in grid and 0 not in out[cell]:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_dir", type=Path)
    ap.add_argument("--grid", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.seed_dir.is_dir():
        print(f"error: not a directory: {args.seed_dir}", file=sys.stderr)
        return 2

    grid = json.loads(args.grid.read_text()) if args.grid else None
    runs = collect(args.seed_dir, grid)
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

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
