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
