"""Tests for scripts/plot_significance.py.

Figure tests assert the files are produced and non-trivial; the numeric content
they draw is asserted in test_significance.py, so nothing here depends on pixels.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_significance.py"

pytest.importorskip("matplotlib")


def _load():
    spec = importlib.util.spec_from_file_location("plot_significance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = _load()


def _report(n_seeds: int = 3) -> dict:
    seeds = list(range(n_seeds))
    cells = {
        "b3_380": [0.954545, 0.956061, 0.956061][:n_seeds],
        "b0_380": [0.948485, 0.946970, 0.950000][:n_seeds],
    }
    report: dict = {}
    for cell, accs in cells.items():
        report[cell] = {
            "n": len(accs),
            "seeds": seeds,
            "per_seed": accs,
            "mean": sum(accs) / len(accs),
            "std": 0.001,
            "min": min(accs),
            "max": max(accs),
            "spread": max(accs) - min(accs),
        }
    report["_significance"] = {
        "protocol": {
            "pairing_unit": "seed",
            "bootstrap_resample_unit": "seed-level paired difference",
            "bootstrap_resamples": 10000,
            "alpha": 0.05,
        },
        "measurement_resolution": {
            "n_images": 660,
            "points_per_image": 0.15151515151515152,
            "ci_width_points": 3.3564849059046553,
        },
        "comparisons": [
            {
                "cell_a": "b3_380",
                "cell_b": "b0_380",
                "seeds": seeds,
                "mean_diff": 0.00707,
                "verdict": "significant_parametric_only",
                "bootstrap": {"low": 0.00606, "high": 0.00909, "excludes_zero": True},
                "t_test": {"p_value": 0.0198},
                "permutation": {"p_value": 0.25, "min_attainable_p": 0.25},
                "effect_size_dz": 4.04,
                "power_limited": True,
            }
        ],
    }
    return report


def test_cells_excludes_reserved_keys() -> None:
    assert set(plot._cells(_report())) == {"b3_380", "b0_380"}


def test_seed_distribution_is_written(tmp_path: Path) -> None:
    out = tmp_path / "dist.png"
    plot.plot_seed_distribution(_report(), out)
    assert out.is_file() and out.stat().st_size > 5000


def test_seed_distribution_works_without_resolution_block(tmp_path: Path) -> None:
    report = _report()
    del report["_significance"]["measurement_resolution"]
    out = tmp_path / "dist.png"
    plot.plot_seed_distribution(report, out)
    assert out.is_file()


def test_paired_ci_is_written(tmp_path: Path) -> None:
    out = tmp_path / "ci.png"
    plot.plot_paired_intervals(_report(), out)
    assert out.is_file() and out.stat().st_size > 5000


def test_paired_ci_skips_when_no_comparisons(tmp_path: Path, capsys) -> None:
    report = _report()
    report["_significance"]["comparisons"] = []
    out = tmp_path / "ci.png"
    plot.plot_paired_intervals(report, out)
    assert not out.exists()
    assert "skipping" in capsys.readouterr().out


def test_power_floor_is_written(tmp_path: Path) -> None:
    out = tmp_path / "power.png"
    plot.plot_power_floor(_report(), out)
    assert out.is_file() and out.stat().st_size > 5000


def test_power_floor_without_significance_block(tmp_path: Path) -> None:
    out = tmp_path / "power.png"
    plot.plot_power_floor({"cell": {"mean": 0.95}}, out)
    assert out.is_file()


def test_main_writes_all_three_panels(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()))
    rc = plot.main(
        ["--report", str(report_path), "--output-dir", str(tmp_path), "--prefix", "f11"]
    )
    assert rc == 0
    for name in ("seed_distribution", "paired_ci", "power"):
        assert (tmp_path / f"f11_{name}.png").is_file()


def _power_report() -> dict:
    """Minimal stand-in for the JSON written by ``validation_power.py``."""
    return {
        "full_set": {"accuracy_diff_points": 0.41188118811881225, "n_images": 25250},
        "protocol": {"trials_per_size": 400},
        "sizes": [
            {
                "n_images": 660,
                "sign_flip_rate": 0.3875,
                "points_per_image": 0.15151515151515152,
                "single_run_ci_width_points": 3.3564844916278647,
            },
            {
                "n_images": 5000,
                "sign_flip_rate": 0.145,
                "points_per_image": 0.02,
                "single_run_ci_width_points": 1.2097138349410397,
            },
            {
                "n_images": 25250,
                "sign_flip_rate": 0.0,
                "points_per_image": 0.0039603960396039604,
                "single_run_ci_width_points": 0.5376758288920591,
            },
        ],
    }


def test_validation_power_is_written(tmp_path: Path) -> None:
    out = tmp_path / "vp.png"
    plot.plot_validation_power(_power_report(), out)
    assert out.is_file() and out.stat().st_size > 5000


def test_validation_power_sorts_sizes(tmp_path: Path) -> None:
    """Rows arriving out of order must not produce a zig-zag line."""
    shuffled = _power_report()
    shuffled["sizes"] = list(reversed(shuffled["sizes"]))
    out = tmp_path / "vp_shuffled.png"
    plot.plot_validation_power(shuffled, out)
    assert out.is_file()


def test_main_adds_validation_power_panel_when_requested(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()))
    power_path = tmp_path / "power_report.json"
    power_path.write_text(json.dumps(_power_report()))
    rc = plot.main(
        [
            "--report",
            str(report_path),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "f11",
            "--power-report",
            str(power_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "f11_validation_power.png").is_file()


def test_main_omits_validation_power_panel_by_default(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()))
    rc = plot.main(
        ["--report", str(report_path), "--output-dir", str(tmp_path), "--prefix", "f11"]
    )
    assert rc == 0
    assert not (tmp_path / "f11_validation_power.png").exists()


def test_main_rejects_report_without_cells(tmp_path: Path) -> None:
    report_path = tmp_path / "empty.json"
    report_path.write_text(json.dumps({"_significance": {"comparisons": []}}))
    rc = plot.main(["--report", str(report_path), "--output-dir", str(tmp_path)])
    assert rc == 1


# ---------------------------------------------------------------------------
# the validation-size panel
# ---------------------------------------------------------------------------
def _power_report() -> dict:
    return {
        "protocol": {"trials_per_size": 400, "resample_unit": "validation image"},
        "full_set": {"n_images": 25250, "accuracy_diff_points": 0.4119, "p_value": 0.0184},
        "sizes": [
            {
                "n_images": 660,
                "points_per_image": 0.15151515151515152,
                "single_run_ci_width_points": 3.3564849059046553,
                "sign_flip_rate": 0.388,
                "power_at_alpha": 0.045,
            },
            {
                "n_images": 25250,
                "points_per_image": 0.0039603960396039604,
                "single_run_ci_width_points": 0.5376758941044724,
                "sign_flip_rate": 0.0,
                "power_at_alpha": 1.0,
            },
        ],
    }


def test_validation_power_panel_is_written(tmp_path: Path) -> None:
    out = tmp_path / "vp.png"
    plot.plot_validation_power(_power_report(), out)
    assert out.is_file() and out.stat().st_size > 5000


def test_validation_power_handles_unsorted_sizes(tmp_path: Path) -> None:
    report = _power_report()
    report["sizes"] = list(reversed(report["sizes"]))
    out = tmp_path / "vp.png"
    plot.plot_validation_power(report, out)
    assert out.is_file()


def test_main_adds_the_fourth_panel_only_when_asked(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()))
    power_path = tmp_path / "power.json"
    power_path.write_text(json.dumps(_power_report()))

    # Without --power-report: three panels.
    assert plot.main(["--report", str(report_path), "--output-dir", str(tmp_path),
                      "--prefix", "a"]) == 0
    assert not (tmp_path / "a_validation_power.png").exists()

    # With it: four.
    assert plot.main(["--report", str(report_path), "--output-dir", str(tmp_path),
                      "--prefix", "b", "--power-report", str(power_path)]) == 0
    assert (tmp_path / "b_validation_power.png").is_file()
