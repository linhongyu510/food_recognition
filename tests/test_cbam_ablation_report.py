"""Tests for ``scripts/cbam_ablation_report.py``.

The script is the reporting layer over a real paired experiment, so the tests
check the contract that matters: accuracies come from ``metrics.json`` and are
never transcribed, only shared seeds are paired, and a half-finished sweep is
refused rather than silently reported on one seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cbam_ablation_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("cbam_ablation_report", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load()


def _make_run(root: Path, name: str, seed: int, accuracy: float) -> None:
    run = root / f"{name}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.json").write_text(json.dumps({"accuracy": accuracy}))


def test_parse_run_accepts_name_and_template() -> None:
    name, template = report.parse_run("cbam=runs/x_s{seed}")
    assert name == "cbam"
    assert template == "runs/x_s{seed}"


def test_parse_run_rejects_missing_equals() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        report.parse_run("no-equals-sign")


def test_parse_run_rejects_template_without_seed_placeholder() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        report.parse_run("cbam=runs/fixed_path")


def test_load_accuracies_reads_metrics_json(tmp_path: Path) -> None:
    _make_run(tmp_path, "cbam", 0, 0.93)
    _make_run(tmp_path, "cbam", 1, 0.94)
    accs = report.load_accuracies(str(tmp_path / "cbam_s{seed}"), [0, 1])
    assert accs == {0: 0.93, 1: 0.94}


def test_load_accuracies_skips_absent_runs(tmp_path: Path) -> None:
    """A seed that has not finished training yet must simply be absent."""
    _make_run(tmp_path, "cbam", 0, 0.93)
    accs = report.load_accuracies(str(tmp_path / "cbam_s{seed}"), [0, 1, 2])
    assert accs == {0: 0.93}


def test_main_writes_report_and_pairs_on_shared_seeds(tmp_path: Path) -> None:
    for seed, (a, b) in enumerate([(0.95, 0.94), (0.96, 0.95), (0.94, 0.93)]):
        _make_run(tmp_path, "cbam", seed, a)
        _make_run(tmp_path, "nocbam", seed, b)
    out = tmp_path / "out.json"
    rc = report.main(
        [
            "--run", f"cbam={tmp_path / 'cbam_s{seed}'}",
            "--run", f"nocbam={tmp_path / 'nocbam_s{seed}'}",
            "--seeds", "0", "1", "2",
            "--resamples", "200",
            "--json-out", str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["seeds_used"] == [0, 1, 2]
    # every per-seed difference is +0.01 here, so the mean must be exactly that
    assert data["comparison"]["mean_diff"] == pytest.approx(0.01)
    assert data["protocol"]["pairing_unit"].startswith("seed")
    assert data["comparison"]["bootstrap"]["resample_unit"] == "seed"


def test_main_refuses_fewer_than_two_shared_seeds(tmp_path: Path, capsys) -> None:
    """A sweep still in progress must be refused, not reported on n=1."""
    _make_run(tmp_path, "cbam", 0, 0.95)
    _make_run(tmp_path, "nocbam", 0, 0.94)
    rc = report.main(
        [
            "--run", f"cbam={tmp_path / 'cbam_s{seed}'}",
            "--run", f"nocbam={tmp_path / 'nocbam_s{seed}'}",
            "--seeds", "0", "1", "2",
        ]
    )
    assert rc == 1
    assert "shared seeds" in capsys.readouterr().err


def test_main_requires_exactly_two_runs(tmp_path: Path, capsys) -> None:
    _make_run(tmp_path, "cbam", 0, 0.95)
    rc = report.main(
        ["--run", f"cbam={tmp_path / 'cbam_s{seed}'}", "--seeds", "0"]
    )
    assert rc == 2
    assert "exactly two" in capsys.readouterr().err


def test_main_records_missing_seeds(tmp_path: Path) -> None:
    for seed in (0, 1):
        _make_run(tmp_path, "cbam", seed, 0.95)
        _make_run(tmp_path, "nocbam", seed, 0.94)
    _make_run(tmp_path, "cbam", 2, 0.96)  # nocbam seed 2 absent
    out = tmp_path / "out.json"
    rc = report.main(
        [
            "--run", f"cbam={tmp_path / 'cbam_s{seed}'}",
            "--run", f"nocbam={tmp_path / 'nocbam_s{seed}'}",
            "--seeds", "0", "1", "2",
            "--resamples", "200",
            "--json-out", str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["seeds_used"] == [0, 1]
    assert data["runs"]["nocbam"]["missing_seeds"] == [2]


def test_report_states_what_it_does_not_answer(tmp_path: Path) -> None:
    """The protocol must keep this experiment separate from the paper claims."""
    for seed in (0, 1):
        _make_run(tmp_path, "cbam", seed, 0.95)
        _make_run(tmp_path, "nocbam", seed, 0.94)
    out = tmp_path / "out.json"
    report.main(
        [
            "--run", f"cbam={tmp_path / 'cbam_s{seed}'}",
            "--run", f"nocbam={tmp_path / 'nocbam_s{seed}'}",
            "--seeds", "0", "1",
            "--resamples", "200",
            "--json-out", str(out),
        ]
    )
    not_answered = json.loads(out.read_text())["protocol"]["not_answered"]
    assert "published" in not_answered
