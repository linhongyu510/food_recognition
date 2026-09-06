"""Plot the resolution x architecture ablation grid.

Reads the nine runs' own metrics.json/history.json rather than any transcribed
numbers, so the figure cannot disagree with the tables in the README::

    python scripts/plot_ablation.py --grid docs/benchmarks/food11_ablation_grid.json \\
        --output docs/benchmarks/food11_ablation.png
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


def plot(grid: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    for model in MODELS:
        accuracies = [grid[f"{model}_{px}"]["acc"] * 100 for px in RESOLUTIONS]
        minutes = [grid[f"{model}_{px}"]["min"] for px in RESOLUTIONS]
        label = f"EfficientNet-{model.upper()} + CBAM ({PARAMS[model]:.1f} M)"

        left.plot(
            RESOLUTIONS, accuracies, "o-", color=COLOURS[model], label=label, linewidth=2
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
        right.plot(
            minutes, accuracies, "o-", color=COLOURS[model], label=label, linewidth=2
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

    fig.suptitle(
        "Food-11: resolution x architecture, 9 runs, identical schedule", fontsize=13
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path, help="ablation grid JSON")
    parser.add_argument("--output", required=True, type=Path, help="PNG to write")
    args = parser.parse_args(argv)

    grid = json.loads(args.grid.read_text())
    missing = [
        f"{m}_{px}" for m in MODELS for px in RESOLUTIONS if f"{m}_{px}" not in grid
    ]
    if missing:
        raise SystemExit(f"grid is incomplete, missing: {', '.join(missing)}")

    plot(grid, args.output)


if __name__ == "__main__":
    main()
