"""Plot seed distributions, paired confidence intervals and the power limit.

Reads the JSON written by ``scripts/aggregate_seeds.py --significance`` rather
than any transcribed number, so the figures cannot disagree with the tables::

    python scripts/plot_significance.py \\
        --report docs/benchmarks/food11_seed_significance.json \\
        --output-dir docs/benchmarks

Three panels are written:

``food11_seed_distribution.png``
    Every seed's accuracy per cell, with the mean and the +-1 SD band, plus a
    scale bar showing what one validation image is worth. When the per-seed
    points overlap across cells, the ranking is not a ranking.
``food11_paired_ci.png``
    The mean paired difference and its bootstrap CI for every cell pair. A pair
    whose interval crosses zero cannot be ordered from this data.
``food11_power.png``
    The exact sign-flip test's p-value floor against the number of paired seeds,
    marking where the current design sits and where alpha becomes reachable.
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

from food_recognition.significance import seeds_needed  # noqa: E402

# Colour-blind-safe; also distinguishable in greyscale by position.
POSITIVE = "#0072B2"
NEUTRAL = "#7F7F7F"
ACCENT = "#D55E00"
GRID_ALPHA = 0.3


def _cells(report: dict) -> dict[str, dict]:
    """Per-cell entries, i.e. everything not under a reserved ``_`` key."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def plot_seed_distribution(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = _cells(report)
    order = sorted(cells, key=lambda c: -cells[c]["mean"])
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for x, cell in enumerate(order):
        entry = cells[cell]
        accs = [v * 100 for v in entry["per_seed"]]
        mean, std = entry["mean"] * 100, entry["std"] * 100
        ax.add_patch(
            plt.Rectangle(
                (x - 0.28, mean - std), 0.56, 2 * std,
                facecolor=POSITIVE, alpha=0.16, edgecolor="none",
                label="±1 SD across seeds" if x == 0 else None,
            )
        )
        ax.hlines(mean, x - 0.3, x + 0.3, color=POSITIVE, linewidth=2.2,
                  label="mean" if x == 0 else None)
        # Jitter horizontally so coincident seeds stay countable.
        for i, (seed, acc) in enumerate(zip(entry["seeds"], accs)):
            ax.plot(x + (i - (len(accs) - 1) / 2) * 0.1, acc, "o", color=ACCENT,
                    markersize=7, zorder=3,
                    label="individual seed" if x == 0 and i == 0 else None)
            ax.annotate(f"s{seed}", (x + (i - (len(accs) - 1) / 2) * 0.1, acc),
                        textcoords="offset points", xytext=(7, -3), fontsize=7,
                        color=NEUTRAL)

    res = report.get("_significance", {}).get("measurement_resolution")
    subtitle = "Food-11 validation accuracy per seed (660 images, 30-epoch schedule)"
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    # Reserve a right-hand margin for the scale bar before reading the limits,
    # so the annotation lands inside the axes rather than over the spine.
    ax.set_xlim(-0.6, len(order) - 0.5 + 0.9)
    if res:
        per_image = res["points_per_image"]
        lo, hi = ax.get_ylim()
        y0 = lo + 0.10 * (hi - lo)
        x_bar = len(order) - 0.15
        ax.annotate(
            "", xy=(x_bar, y0 + per_image), xytext=(x_bar, y0),
            arrowprops={"arrowstyle": "|-|", "color": ACCENT, "linewidth": 1.4},
        )
        ax.annotate(
            f"1 image\n= {per_image:.3f} pts",
            (x_bar + 0.08, y0), fontsize=7.5, color=ACCENT, va="bottom",
        )
        subtitle = (
            f"Food-11 validation accuracy per seed ({res['n_images']} images, "
            "30-epoch schedule)\n"
            f"1 image = {per_image:.3f} points; single-run 95% CI width "
            f"{res['ci_width_points']:.2f} points"
        )

    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(subtitle, fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_paired_intervals(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = report.get("_significance")
    if not sig or not sig["comparisons"]:
        print("no comparisons in report; skipping paired-CI figure")
        return

    comps = sorted(sig["comparisons"], key=lambda c: c["mean_diff"])
    labels = [f"{c['cell_a']}\nvs {c['cell_b']}" for c in comps]
    means = [c["mean_diff"] * 100 for c in comps]
    lows = [c["bootstrap"]["low"] * 100 for c in comps]
    highs = [c["bootstrap"]["high"] * 100 for c in comps]

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(comps) + 2.4))
    for y, c in enumerate(comps):
        # Colour by the verdict, not by the bootstrap alone: an interval can
        # exclude zero while the t-test still fails to reject, and colouring on
        # the interval alone would paint those bars as if they were positive
        # findings while the label beside them says otherwise.
        colour = POSITIVE if c["verdict"].startswith("significant") else NEUTRAL
        ax.plot([lows[y], highs[y]], [y, y], color=colour, linewidth=2.6,
                solid_capstyle="butt")
        ax.plot(means[y], y, "o", color=colour, markersize=8, zorder=3)
        note = c["verdict"].replace("_", " ")
        if not c["bootstrap"]["excludes_zero"]:
            note += " (CI spans 0)"
        ax.annotate(note, (highs[y], y), textcoords="offset points", xytext=(8, -3),
                    fontsize=7.5, color=colour)

    ax.axvline(0, color=ACCENT, linewidth=1.3, linestyle="--", zorder=1)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Paired accuracy difference (percentage points)")
    proto = sig["protocol"]
    ax.set_title(
        "Paired differences with bootstrap confidence intervals\n"
        f"pairing unit = {proto['pairing_unit']}; "
        f"{proto['bootstrap_resamples']:,} percentile bootstrap resamples over "
        f"{proto['bootstrap_resample_unit']}s; alpha = {proto['alpha']}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=GRID_ALPHA)
    # Headroom on the right for the verdict labels.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.42 * (x1 - x0))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_power_floor(report: dict, output: Path, *, alpha: float = 0.05) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = list(range(2, 13))
    floors = [2.0 / 2**n for n in ns]
    needed = seeds_needed(alpha)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ns, floors, "o-", color=POSITIVE, linewidth=2,
            label="smallest p the exact test can return")
    ax.axhline(alpha, color=ACCENT, linestyle="--", linewidth=1.4,
               label=f"alpha = {alpha}")
    ax.axvline(needed, color=NEUTRAL, linestyle=":", linewidth=1.4,
               label=f"{needed} seeds: first design that can reject")

    sig = report.get("_significance")
    if sig and sig["comparisons"]:
        current = len(sig["comparisons"][0]["seeds"])
        ax.plot([current], [2.0 / 2**current], "s", color=ACCENT, markersize=11,
                zorder=4, label=f"this study: {current} seeds -> floor {2.0 / 2**current:.2f}")

    ax.set_yscale("log")
    ax.set_xlabel("Paired seeds per cell")
    ax.set_ylabel("p-value floor (log scale)")
    ax.set_title(
        "Why three seeds cannot produce an assumption-free rejection\n"
        "two-sided exact sign-flip test: the floor is 2 / 2^n, whatever the effect size",
        fontsize=10,
    )
    ax.set_xticks(ns)
    ax.grid(alpha=GRID_ALPHA, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="food11")
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text())
    if not _cells(report):
        print("error: report has no per-cell entries", file=sys.stderr)
        return 1

    plot_seed_distribution(report, args.output_dir / f"{args.prefix}_seed_distribution.png")
    plot_paired_intervals(report, args.output_dir / f"{args.prefix}_paired_ci.png")
    plot_power_floor(report, args.output_dir / f"{args.prefix}_power.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
