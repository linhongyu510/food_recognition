"""Plot the resolution x architecture ablation grid.

Reads the nine runs' own metrics.json/history.json rather than any transcribed
numbers, so the figure cannot disagree with the tables in the README::

    python scripts/plot_ablation.py --grid docs/benchmarks/food11_ablation_grid.json \\
        --output docs/benchmarks/food11_ablation.png

Pass ``--variance docs/benchmarks/food11_seed_variance.json`` to draw error bars
on the cells that were re-run with extra seeds. Without them the figure invites
the reader to rank cells whose gaps are smaller than the run-to-run noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("b0", "b3", "b4")
RESOLUTIONS = (224, 300, 380)
NATIVE = {"b0": 224, "b3": 300, "b4": 380}
PARAMS = {"b0": 4.2, "b3": 11.0, "b4": 18.0}
COLOURS = {"b0": "#1f77b4", "b3": "#ff7f0e", "b4": "#2ca02c"}


def plot(grid: dict, output: Path, variance: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variance = variance or {}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    for model in MODELS:
        cells = [f"{model}_{px}" for px in RESOLUTIONS]
        # Where extra seeds exist, plot their mean instead of the seed-0 point:
        # a single seed is one sample, not the cell's behaviour.
        accuracies = [
            (variance[c]["mean"] if c in variance else grid[c]["acc"]) * 100 for c in cells
        ]
        errors = [variance[c]["std"] * 100 if c in variance else 0.0 for c in cells]
        minutes = [grid[c]["min"] for c in cells]
        label = f"EfficientNet-{model.upper()} + CBAM ({PARAMS[model]:.1f} M)"

        left.errorbar(
            RESOLUTIONS,
            accuracies,
            yerr=errors,
            fmt="o-",
            color=COLOURS[model],
            label=label,
            linewidth=2,
            capsize=4,
        )
        # Ring the native resolution so "trained at" is visually separable from "best".
        native_index = RESOLUTIONS.index(NATIVE[model])
        left.plot(
            NATIVE[model],
            accuracies[native_index],
            "o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor=COLOURS[model],
            markeredgewidth=2,
        )
        right.errorbar(
            minutes,
            accuracies,
            yerr=errors,
            fmt="o-",
            color=COLOURS[model],
            label=label,
            linewidth=2,
            capsize=4,
        )
        for px, minute, accuracy in zip(RESOLUTIONS, minutes, accuracies):
            right.annotate(
                f"{px}px",
                (minute, accuracy),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
                color=COLOURS[model],
            )

    left.set_xlabel("Input resolution (px)")
    left.set_ylabel("Validation accuracy (%)")
    left.set_title("Accuracy vs input resolution\n(ring = the backbone's native resolution)")
    left.set_xticks(RESOLUTIONS)
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower right")

    right.set_xlabel("Training wall clock (min, 30 epochs on an M5 Pro)")
    right.set_ylabel("Validation accuracy (%)")
    right.set_title("Accuracy vs cost\n(up and to the left is better)")
    right.grid(alpha=0.3)
    right.legend(fontsize=8, loc="lower right")

    if variance:
        n_seeded = len(variance)
        worst = max(v["spread"] for v in variance.values()) * 100
        subtitle = (
            f"Food-11: resolution x architecture, 9 runs, identical schedule\n"
            f"{n_seeded} cells re-run with 3 seeds; error bars = 1 SD; "
            f"largest seed spread {worst:.2f} pts"
        )
    else:
        subtitle = "Food-11: resolution x architecture, 9 runs, identical schedule"

    fig.suptitle(subtitle, fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path, help="ablation grid JSON")
    parser.add_argument("--output", required=True, type=Path, help="PNG to write")
    parser.add_argument(
        "--variance", type=Path, default=None, help="multi-seed report from aggregate_seeds.py"
    )
    args = parser.parse_args(argv)

    grid = json.loads(args.grid.read_text())
    missing = [
        f"{m}_{px}" for m in MODELS for px in RESOLUTIONS if f"{m}_{px}" not in grid
    ]
    if missing:
        raise SystemExit(f"grid is incomplete, missing: {', '.join(missing)}")

    variance = json.loads(args.variance.read_text()) if args.variance else None
    plot(grid, args.output, variance)


if __name__ == "__main__":
    main()
