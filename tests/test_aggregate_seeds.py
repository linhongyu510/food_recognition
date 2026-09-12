"""Tests for scripts/aggregate_seeds.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_seeds.py"


def _load():
    spec = importlib.util.spec_from_file_location("aggregate_seeds", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agg = _load()


def _run(tmp: Path, cell: str, seed: int, acc: float) -> None:
    d = tmp / f"{cell}_s{seed}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"accuracy": acc, "macro_f1": acc}))


def test_collect_groups_seeds_per_cell(tmp_path: Path) -> None:
    _run(tmp_path, "b0_300", 1, 0.95)
    _run(tmp_path, "b0_300", 2, 0.93)
    _run(tmp_path, "b4_380", 1, 0.94)
    got = agg.collect(tmp_path, None)
    assert got == {"b0_300": {1: 0.95, 2: 0.93}, "b4_380": {1: 0.94}}


def test_collect_merges_seed_zero_from_grid(tmp_path: Path) -> None:
    _run(tmp_path, "b0_300", 1, 0.95)
    got = agg.collect(tmp_path, {"b0_300": {"acc": 0.90}})
    assert got["b0_300"] == {0: 0.90, 1: 0.95}


def test_grid_does_not_override_existing_seed_zero(tmp_path: Path) -> None:
    _run(tmp_path, "b0_300", 0, 0.99)
    got = agg.collect(tmp_path, {"b0_300": {"acc": 0.11}})
    assert got["b0_300"][0] == 0.99


def test_collect_ignores_unparsable_and_incomplete(tmp_path: Path) -> None:
    _run(tmp_path, "good", 1, 0.9)
    (tmp_path / "no_seed_suffix").mkdir()
    (tmp_path / "pending_s2").mkdir()  # directory exists but no metrics.json
    (tmp_path / "stray.txt").write_text("x")
    assert set(agg.collect(tmp_path, None)) == {"good"}


def test_summarise_statistics() -> None:
    r = agg.summarise({0: 0.90, 1: 0.94, 2: 0.92})
    assert r["n"] == 3
    assert r["seeds"] == [0, 1, 2]
    assert r["mean"] == pytest.approx(0.92)
    assert r["std"] == pytest.approx(0.02)
    assert r["spread"] == pytest.approx(0.04)
    assert r["min"] == pytest.approx(0.90)
    assert r["max"] == pytest.approx(0.94)


def test_summarise_single_seed_has_zero_spread() -> None:
    r = agg.summarise({0: 0.9})
    assert r["n"] == 1
    assert r["std"] == 0.0
    assert r["spread"] == 0.0


def test_summarise_orders_per_seed_by_seed_number() -> None:
    r = agg.summarise({2: 0.3, 0: 0.1, 1: 0.2})
    assert r["per_seed"] == [0.1, 0.2, 0.3]


def test_main_writes_json_and_flags_single_seed(tmp_path: Path, capsys) -> None:
    _run(tmp_path, "solo", 1, 0.90)
    _run(tmp_path, "pair", 1, 0.90)
    _run(tmp_path, "pair", 2, 0.94)
    out = tmp_path / "report.json"
    rc = agg.main([str(tmp_path), "--json-out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "solo" in text and "single seed" in text
    data = json.loads(out.read_text())
    assert data["pair"]["mean"] == pytest.approx(0.92)
    assert data["solo"]["n"] == 1


def test_main_missing_dir_returns_2(tmp_path: Path) -> None:
    assert agg.main([str(tmp_path / "absent")]) == 2


def test_main_empty_dir_returns_1(tmp_path: Path) -> None:
    assert agg.main([str(tmp_path)]) == 1


# ---------------------------------------------------------------------------
# --significance / --resolution wiring
# ---------------------------------------------------------------------------
def test_significance_report_records_the_full_protocol() -> None:
    """The protocol must be machine-readable, not just described in prose."""
    runs = {
        "a": {0: 0.95, 1: 0.96, 2: 0.955},
        "b": {0: 0.94, 1: 0.945, 2: 0.942},
    }
    rep = agg.significance_report(runs, n_resamples=500, resolution_images=660)
    proto = rep["protocol"]
    assert proto["pairing_unit"] == "seed"
    assert proto["bootstrap_resample_unit"] == "seed-level paired difference"
    assert proto["bootstrap_resamples"] == 500
    assert proto["alpha"] == 0.05
    assert proto["tails"] == 2
    assert "paired t-test" in proto["tests"]
    assert proto["multiplicity_correction"] == "none; p-values are per-pair and uncorrected"
    assert rep["measurement_resolution"]["n_images"] == 660


def test_significance_report_compares_every_pair() -> None:
    runs = {n: {0: 0.9, 1: 0.91, 2: 0.92} for n in ("a", "b", "c")}
    rep = agg.significance_report(runs, n_resamples=200)
    assert len(rep["comparisons"]) == 3


def test_significance_report_each_comparison_carries_all_statistics() -> None:
    runs = {"a": {0: 0.95, 1: 0.96}, "b": {0: 0.94, 1: 0.93}}
    (comp,) = agg.significance_report(runs, n_resamples=200)["comparisons"]
    for key in ("t_test", "permutation", "bootstrap", "effect_size_dz", "verdict", "power_limited"):
        assert key in comp
    assert "p_value" in comp["t_test"]
    assert comp["bootstrap"]["resample_unit"] == "seed"


def test_significance_report_omits_resolution_when_not_requested() -> None:
    runs = {"a": {0: 0.95, 1: 0.96}, "b": {0: 0.94, 1: 0.93}}
    assert "measurement_resolution" not in agg.significance_report(runs, n_resamples=200)


def test_main_significance_nests_under_reserved_key(tmp_path: Path, capsys) -> None:
    # Per-cell entries must stay addressable by name so existing readers of this
    # JSON (plot_ablation.py --variance) keep working unchanged.
    for seed, (a, b) in enumerate([(0.95, 0.94), (0.96, 0.945), (0.955, 0.942)]):
        _run(tmp_path, "cell_a", seed, a)
        _run(tmp_path, "cell_b", seed, b)
    out = tmp_path / "sig.json"
    rc = agg.main([str(tmp_path), "--significance", "--resolution", "660",
                   "--resamples", "500", "--json-out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["cell_a"]["n"] == 3          # unchanged per-cell shape
    assert "_significance" in data
    assert data["_significance"]["protocol"]["pairing_unit"] == "seed"
    text = capsys.readouterr().out
    assert "paired comparisons" in text
    assert "measurement resolution" in text


def test_main_resolution_without_significance(tmp_path: Path, capsys) -> None:
    _run(tmp_path, "solo", 0, 0.95)
    out = tmp_path / "r.json"
    assert agg.main([str(tmp_path), "--resolution", "25250", "--json-out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["_measurement_resolution"]["n_images"] == 25250
    assert "measurement resolution" in capsys.readouterr().out


def test_main_significance_reports_when_no_pair_can_be_tested(tmp_path: Path, capsys) -> None:
    _run(tmp_path, "only_cell", 0, 0.95)
    _run(tmp_path, "only_cell", 1, 0.96)
    assert agg.main([str(tmp_path), "--significance", "--resamples", "200"]) == 0
    assert "no cell pair shares enough seeds" in capsys.readouterr().out


def test_main_significance_flags_power_limit_for_three_seeds(tmp_path: Path, capsys) -> None:
    for seed, (a, b) in enumerate([(0.95, 0.90), (0.96, 0.91), (0.955, 0.905)]):
        _run(tmp_path, "hi", seed, a)
        _run(tmp_path, "lo", seed, b)
    assert agg.main([str(tmp_path), "--significance", "--resamples", "500"]) == 0
    out = capsys.readouterr().out
    assert "underpowered by" in out or "cannot reach" in out


def test_main_json_stays_backward_compatible_without_significance(tmp_path: Path) -> None:
    _run(tmp_path, "cell", 0, 0.95)
    _run(tmp_path, "cell", 1, 0.96)
    out = tmp_path / "plain.json"
    assert agg.main([str(tmp_path), "--json-out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert set(data) == {"cell"}          # no extra top-level keys


# ---------------------------------------------------------------------------
# --seeds pinning
#
# Regression: a report generated while a sweep was still running had one cell
# at n=4 and the other three at n=3, because collect() took whatever was on
# disk. The paired comparisons were still correct (they intersect seeds), but
# the per-cell block contradicted the n=3 table the report was cited for.
# ---------------------------------------------------------------------------
def _sweep_dir(tmp_path: Path) -> Path:
    seed_dir = tmp_path / "seeds"
    accs = {
        ("a", 0): 0.90, ("a", 1): 0.91, ("a", 2): 0.92, ("a", 3): 0.93,
        ("b", 0): 0.80, ("b", 1): 0.81, ("b", 2): 0.82,
    }
    for (cell, seed), acc in accs.items():
        run = seed_dir / f"{cell}_s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "metrics.json").write_text(json.dumps({"accuracy": acc}))
    return seed_dir


def test_collect_without_seeds_takes_everything_on_disk(tmp_path: Path) -> None:
    runs = agg.collect(_sweep_dir(tmp_path), None)
    assert sorted(runs["a"]) == [0, 1, 2, 3]
    assert sorted(runs["b"]) == [0, 1, 2]


def test_collect_pins_to_the_requested_seeds(tmp_path: Path) -> None:
    runs = agg.collect(_sweep_dir(tmp_path), None, {0, 1, 2})
    assert sorted(runs["a"]) == [0, 1, 2]
    assert sorted(runs["b"]) == [0, 1, 2]


def test_pinned_seeds_give_every_cell_the_same_n(tmp_path: Path) -> None:
    """The actual defect: unequal n across cells in one report."""
    unpinned = agg.collect(_sweep_dir(tmp_path), None)
    assert len({len(v) for v in unpinned.values()}) == 2  # the bug

    pinned = agg.collect(_sweep_dir(tmp_path), None, {0, 1, 2})
    assert len({len(v) for v in pinned.values()}) == 1  # fixed


def test_seed_filter_excludes_grid_seed_zero_when_not_requested(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seeds"
    run = seed_dir / "a_s1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.91}))
    grid = {"a": {"acc": 0.90}}

    with_zero = agg.collect(seed_dir, grid, {0, 1})
    assert sorted(with_zero["a"]) == [0, 1]

    without_zero = agg.collect(seed_dir, grid, {1})
    assert sorted(without_zero["a"]) == [1]


def test_protocol_records_the_pinned_seed_set(tmp_path: Path) -> None:
    runs = agg.collect(_sweep_dir(tmp_path), None, {0, 1, 2})
    report = agg.significance_report(runs, seed_filter={0, 1, 2})
    assert report["protocol"]["seeds_included"] == [0, 1, 2]


def test_protocol_says_so_when_seeds_are_not_pinned(tmp_path: Path) -> None:
    runs = agg.collect(_sweep_dir(tmp_path), None)
    report = agg.significance_report(runs)
    assert report["protocol"]["seeds_included"] == "all seeds found on disk"


def test_main_accepts_a_seed_list(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    rc = agg.main([str(_sweep_dir(tmp_path)), "--significance",
                   "--seeds", "0,1,2", "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["a"]["n"] == payload["b"]["n"] == 3
    assert payload["_significance"]["protocol"]["seeds_included"] == [0, 1, 2]


def test_main_rejects_a_malformed_seed_list(tmp_path: Path) -> None:
    assert agg.main([str(_sweep_dir(tmp_path)), "--seeds", "0,x,2"]) == 2


def test_main_rejects_an_empty_seed_list(tmp_path: Path) -> None:
    assert agg.main([str(_sweep_dir(tmp_path)), "--seeds", ","]) == 2
