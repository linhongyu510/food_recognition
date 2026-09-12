"""Tests for scripts/check_epoch_criterion.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_epoch_criterion.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_epoch_criterion", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chk = _load()


def _run_dir(tmp: Path, name: str, accs: list[float]) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "history.json").write_text(
        json.dumps([{"epoch": i + 1, "val_acc": a} for i, a in enumerate(accs)])
    )
    return d


def test_read_history_returns_every_epoch(tmp_path: Path) -> None:
    d = _run_dir(tmp_path, "r", [0.90, 0.93, 0.91])
    assert chk.read_history(d) == [0.90, 0.93, 0.91]


def test_read_history_accepts_dict_shaped_history(tmp_path: Path) -> None:
    d = tmp_path / "r"
    d.mkdir()
    (d / "history.json").write_text(json.dumps({"epochs": [{"val_acc": 0.9}, {"val_acc": 0.95}]}))
    assert chk.read_history(d) == [0.9, 0.95]


def test_read_history_rejects_history_without_val_acc(tmp_path: Path) -> None:
    d = tmp_path / "r"
    d.mkdir()
    (d / "history.json").write_text(json.dumps([{"loss": 1.0}]))
    with pytest.raises(ValueError, match="no val_acc entries"):
        chk.read_history(d)


def test_criteria_separates_best_from_last() -> None:
    # Peak at epoch 2, then a decline: "best" and "last" must not coincide.
    got = chk.criteria_from_history([0.90, 0.96, 0.92, 0.91])
    assert got["best"] == pytest.approx(0.96)
    assert got["last"] == pytest.approx(0.91)
    assert got["best_epoch"] == 2
    assert got["n_epochs"] == 4
    assert got["selection_gain"] == pytest.approx(0.05)


def test_criteria_selection_gain_is_zero_for_monotonic_runs() -> None:
    got = chk.criteria_from_history([0.90, 0.92, 0.95])
    assert got["best_epoch"] == 3
    assert got["selection_gain"] == pytest.approx(0.0)


def test_compare_rules_detects_a_selection_only_gap(tmp_path: Path) -> None:
    # Cell "spiky" wins on best-epoch purely because of a one-epoch spike; on
    # last-epoch it loses. This is exactly the confound the script exists to
    # find, so the pair must come back flagged.
    table = chk.build(
        {
            "spiky": {
                s: _run_dir(tmp_path, f"spiky_s{s}", [0.90, 0.99, 0.90])
                for s in (0, 1, 2)
            },
            "steady": {
                s: _run_dir(tmp_path, f"steady_s{s}", [0.90, 0.93, 0.95])
                for s in (0, 1, 2)
            },
        }
    )
    report = chk.compare_rules(table, n_resamples=300)
    (row,) = report["pairs"]
    assert row["best_diff_points"] > 0  # spiky wins under best
    assert row["last_diff_points"] < 0  # and loses under last
    assert row["sign_flips"] is True
    assert report["unstable_pairs"] == [row["pair"]]


def test_compare_rules_reports_stability_when_the_gap_is_real(tmp_path: Path) -> None:
    table = chk.build(
        {
            "good": {
                s: _run_dir(tmp_path, f"good_s{s}", [0.90, 0.94, 0.95 + 0.001 * s])
                for s in (0, 1, 2)
            },
            "bad": {
                s: _run_dir(tmp_path, f"bad_s{s}", [0.80, 0.84, 0.85 + 0.001 * s])
                for s in (0, 1, 2)
            },
        }
    )
    report = chk.compare_rules(table, n_resamples=300)
    (row,) = report["pairs"]
    assert row["sign_flips"] is False
    assert report["unstable_pairs"] == []


def test_compare_rules_records_selection_gain_and_epoch_range(tmp_path: Path) -> None:
    table = chk.build(
        {
            "a": {0: _run_dir(tmp_path, "a_s0", [0.90, 0.99, 0.90])},
            "b": {0: _run_dir(tmp_path, "b_s0", [0.90, 0.92, 0.95])},
        }
    )
    report = chk.compare_rules(table, n_resamples=200)
    assert report["best_epoch_range"] == [2, 3]
    assert report["selection_gain_points"]["max"] == pytest.approx(9.0, abs=1e-6)
    assert report["selection_gain_points"]["min"] == pytest.approx(0.0, abs=1e-6)
    assert "no selection" in report["criterion_note"]


def test_accuracies_projects_one_rule(tmp_path: Path) -> None:
    table = chk.build({"a": {0: _run_dir(tmp_path, "a_s0", [0.90, 0.99, 0.93])}})
    assert chk.accuracies(table, "best") == {"a": {0: pytest.approx(0.99)}}
    assert chk.accuracies(table, "last") == {"a": {0: pytest.approx(0.93)}}


def test_main_with_manifest_writes_json(tmp_path: Path, capsys) -> None:
    for cell, accs in (("a", [0.90, 0.95, 0.94]), ("b", [0.88, 0.91, 0.92])):
        for s in (0, 1):
            _run_dir(tmp_path, f"{cell}_s{s}", [v + 0.001 * s for v in accs])
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {c: {str(s): f"{c}_s{s}" for s in (0, 1)} for c in ("a", "b")}
        )
    )
    out = tmp_path / "sens.json"
    rc = chk.main(["--manifest", str(manifest), "--json-out", str(out), "--resamples", "200"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["pairs"] and "best_epoch_range" in data
    assert "best-epoch selection" in capsys.readouterr().out


def test_main_requires_a_source() -> None:
    assert chk.main([]) == 2


def test_main_reports_missing_history(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"a": {"0": "nonexistent"}}))
    assert chk.main(["--manifest", str(manifest)]) == 1


def test_parse_runs_infers_seed_from_directory_suffix(tmp_path: Path) -> None:
    runs = chk._parse_runs(["cell=runs/cell_s7", "cell=runs/plain"], tmp_path)
    assert set(runs["cell"]) == {7, 0}


def test_parse_runs_rejects_malformed_spec(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs cell=path"):
        chk._parse_runs(["missing_equals"], tmp_path)
