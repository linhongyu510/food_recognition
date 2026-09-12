"""Tests for scripts/compare_runs.py and scripts/validation_power.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmp_runs = _load("compare_runs")
vpower = _load("validation_power")


def _vector(tmp: Path, name: str, correct: list[int], order: str = "abc123") -> Path:
    path = tmp / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": f"runs/{name}/best.pt",
                "image_size": 224,
                "n_images": len(correct),
                "n_correct": sum(correct),
                "accuracy": sum(correct) / len(correct),
                "order_sha256_16": order,
                "correct": correct,
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------
def test_load_vector_reads_payload(tmp_path: Path) -> None:
    path = _vector(tmp_path, "a", [1, 0, 1])
    assert cmp_runs.load_vector(path)["n_images"] == 3


def test_load_vector_rejects_incomplete_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"correct": [1, 0]}))
    with pytest.raises(ValueError, match="is missing"):
        cmp_runs.load_vector(path)


def test_build_report_refuses_mismatched_image_order(tmp_path: Path) -> None:
    runs = {
        "a": json.loads(_vector(tmp_path, "a", [1, 0], order="AAA").read_text()),
        "b": json.loads(_vector(tmp_path, "b", [0, 1], order="BBB").read_text()),
    }
    # Pairing two runs scored over different orders would silently compare
    # unrelated images, so this must fail loudly rather than produce a number.
    with pytest.raises(ValueError, match="different image orders"):
        cmp_runs.build_report(runs, n_resamples=100)


def test_build_report_orders_pair_by_accuracy(tmp_path: Path) -> None:
    runs = {
        "worse": json.loads(_vector(tmp_path, "worse", [1, 0, 0, 0]).read_text()),
        "better": json.loads(_vector(tmp_path, "better", [1, 1, 1, 0]).read_text()),
    }
    report = cmp_runs.build_report(runs, n_resamples=200)
    (pair,) = report["pairs"]
    assert pair["run_a"] == "better" and pair["run_b"] == "worse"
    assert pair["accuracy_diff"] > 0


def test_build_report_records_protocol_and_resolution(tmp_path: Path) -> None:
    runs = {
        "a": json.loads(_vector(tmp_path, "a", [1] * 90 + [0] * 10).read_text()),
        "b": json.loads(_vector(tmp_path, "b", [1] * 80 + [0] * 20).read_text()),
    }
    report = cmp_runs.build_report(runs, n_resamples=300)
    proto = report["protocol"]
    assert proto["pairing_unit"] == "validation image"
    assert "does not" in proto["not_answered"] or "requires" in proto["not_answered"]
    assert report["measurement_resolution"]["n_images"] == 100
    assert report["runs"]["a"]["n_correct"] == 90
    assert report["runs"]["a"]["wilson_low"] < 0.90 < report["runs"]["a"]["wilson_high"]


def test_build_report_diff_in_images_is_the_actual_count(tmp_path: Path) -> None:
    # 90 vs 80 correct out of 100 is a 10-image gap; the report must say 10.
    runs = {
        "a": json.loads(_vector(tmp_path, "a", [1] * 90 + [0] * 10).read_text()),
        "b": json.loads(_vector(tmp_path, "b", [1] * 80 + [0] * 20).read_text()),
    }
    (pair,) = cmp_runs.build_report(runs, n_resamples=200)["pairs"]
    assert pair["diff_in_images"] == pytest.approx(10.0)


def test_build_report_calls_identical_runs_not_separable(tmp_path: Path) -> None:
    same = [1, 0, 1, 1, 0] * 20
    runs = {
        "a": json.loads(_vector(tmp_path, "a", same).read_text()),
        "b": json.loads(_vector(tmp_path, "b", list(same)).read_text()),
    }
    (pair,) = cmp_runs.build_report(runs, n_resamples=200)["pairs"]
    assert pair["mcnemar"]["n_discordant"] == 0
    assert pair["significant_at_alpha"] is False


def test_main_writes_json_and_prints(tmp_path: Path, capsys) -> None:
    a = _vector(tmp_path, "a", [1] * 200 + [0] * 50)
    b = _vector(tmp_path, "b", [1] * 150 + [0] * 100)
    out = tmp_path / "cmp.json"
    rc = cmp_runs.main(
        ["--run", f"a={a}", "--run", f"b={b}", "--json-out", str(out), "--resamples", "300"]
    )
    assert rc == 0
    assert json.loads(out.read_text())["pairs"]
    text = capsys.readouterr().out
    assert "validation set" in text and "separable" in text


def test_main_needs_two_runs(tmp_path: Path) -> None:
    a = _vector(tmp_path, "a", [1, 0])
    assert cmp_runs.main(["--run", f"a={a}"]) == 2


def test_main_rejects_bad_spec(tmp_path: Path) -> None:
    assert cmp_runs.main(["--run", "no_equals", "--run", "also_none"]) == 2


def test_main_reports_order_mismatch(tmp_path: Path) -> None:
    a = _vector(tmp_path, "a", [1, 0, 1], order="XXX")
    b = _vector(tmp_path, "b", [0, 1, 1], order="YYY")
    assert cmp_runs.main(["--run", f"a={a}", "--run", f"b={b}"]) == 1


# ---------------------------------------------------------------------------
# validation_power
# ---------------------------------------------------------------------------
def _pair_with_known_gap(n: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """``a`` beats ``b`` by exactly ``gap`` images, with overlap elsewhere."""
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    a[: n // 2] = 1
    b[: n // 2] = 1
    a[n // 2 : n // 2 + gap] = 1  # only a gets these
    return a, b


def test_sweep_power_rises_with_subset_size() -> None:
    a, b = _pair_with_known_gap(4000, 200)
    report = vpower.sweep(a, b, [100, 500, 4000], trials=60, seed=5)
    powers = [row["power_at_alpha"] for row in report["sizes"]]
    assert powers == sorted(powers)
    assert powers[-1] > powers[0]


def test_sweep_full_size_recovers_the_full_set_result() -> None:
    a, b = _pair_with_known_gap(2000, 120)
    report = vpower.sweep(a, b, [2000], trials=5, seed=1)
    (row,) = report["sizes"]
    # Sampling all n without replacement is the full set every time.
    assert row["sd_diff_points"] == pytest.approx(0.0, abs=1e-12)
    assert row["sign_flip_rate"] == 0.0
    assert row["mean_diff_points"] == pytest.approx(
        report["full_set"]["accuracy_diff_points"], abs=1e-9
    )


def test_sweep_small_subsets_flip_the_sign_more_often() -> None:
    a, b = _pair_with_known_gap(4000, 60)  # a small, noisy advantage
    report = vpower.sweep(a, b, [100, 4000], trials=80, seed=3)
    small, large = report["sizes"]
    assert small["sign_flip_rate"] > large["sign_flip_rate"]


def test_sweep_records_resolution_per_size() -> None:
    a, b = _pair_with_known_gap(1000, 50)
    report = vpower.sweep(a, b, [660, 1000], trials=10, seed=2)
    by_n = {row["n_images"]: row for row in report["sizes"]}
    assert by_n[660]["points_per_image"] == pytest.approx(100 / 660)
    assert by_n[1000]["points_per_image"] == pytest.approx(0.1)


def test_sweep_protocol_states_the_caveat() -> None:
    a, b = _pair_with_known_gap(500, 30)
    proto = vpower.sweep(a, b, [500], trials=5)["protocol"]
    assert proto["resample_unit"] == "validation image"
    assert "training-seed noise" in proto["caveat"]


def test_sweep_rejects_out_of_range_sizes() -> None:
    a, b = _pair_with_known_gap(100, 10)
    with pytest.raises(ValueError, match="sizes must be in"):
        vpower.sweep(a, b, [500], trials=2)


def test_sweep_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="align image-by-image"):
        vpower.sweep(np.array([1, 0]), np.array([1, 0, 1]), [2], trials=2)


def test_vpower_main_writes_json(tmp_path: Path, capsys) -> None:
    a, b = _pair_with_known_gap(1000, 80)
    pa = _vector(tmp_path, "a", a.tolist())
    pb = _vector(tmp_path, "b", b.tolist())
    out = tmp_path / "power.json"
    rc = vpower.main(
        ["--run-a", str(pa), "--run-b", str(pb), "--sizes", "200", "1000",
         "--trials", "20", "--json-out", str(out)]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert len(data["sizes"]) == 2
    assert "full set" in capsys.readouterr().out


def test_vpower_main_rejects_order_mismatch(tmp_path: Path) -> None:
    pa = _vector(tmp_path, "a", [1, 0, 1], order="AAA")
    pb = _vector(tmp_path, "b", [0, 1, 1], order="BBB")
    assert vpower.main(["--run-a", str(pa), "--run-b", str(pb), "--sizes", "3"]) == 1


def test_vpower_main_rejects_impossible_size(tmp_path: Path) -> None:
    pa = _vector(tmp_path, "a", [1, 0, 1])
    pb = _vector(tmp_path, "b", [0, 1, 1])
    assert vpower.main(["--run-a", str(pa), "--run-b", str(pb), "--sizes", "999"]) == 1
