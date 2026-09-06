"""Tests for the ablation plotting script.

The figure is read as evidence for the claims in the README, so these check that
it is built from the grid file rather than from anything hard-coded, and that an
incomplete grid is refused instead of silently plotted with gaps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="plotting needs the [viz] extra")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_ablation.py"


def _load_script():
    """Import plot_ablation.py, which lives outside the installed package."""
    spec = importlib.util.spec_from_file_location("plot_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_ablation"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _full_grid() -> dict:
    """A complete 3x3 grid with distinguishable values."""
    grid = {}
    for i, model in enumerate(("b0", "b3", "b4")):
        for j, px in enumerate((224, 300, 380)):
            grid[f"{model}_{px}"] = {
                "acc": 0.90 + 0.01 * i + 0.005 * j,
                "f1": 0.89,
                "ep": 20,
                "min": 10.0 + 5 * i + 3 * j,
            }
    return grid


def test_writes_a_figure_from_the_grid(script, tmp_path: Path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps(_full_grid()))
    output = tmp_path / "nested" / "ablation.png"

    script.main(["--grid", str(grid), "--output", str(output)])

    assert output.is_file()
    # A real PNG, not an empty file: check the magic bytes.
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert output.stat().st_size > 5000


def test_incomplete_grid_is_refused(script, tmp_path: Path):
    """A missing cell should error, not produce a figure with a silent gap."""
    grid = _full_grid()
    del grid["b3_300"]
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(SystemExit, match="b3_300"):
        script.main(["--grid", str(path), "--output", str(tmp_path / "out.png")])

    assert not (tmp_path / "out.png").exists()


def test_native_resolutions_match_torchvision(script):
    """The rings mark native resolution, so they must not drift from the factory."""
    pytest.importorskip("torch")
    from food_recognition.models import initialize_model

    for short, full in (
        ("b0", "efficientnet_b0_cbam"),
        ("b3", "efficientnet_b3_cbam"),
        ("b4", "efficientnet_b4_cbam"),
    ):
        _, native = initialize_model(full, 11, use_pretrained=False)
        assert script.NATIVE[short] == native


def test_shipped_grid_is_complete_and_matches_the_benchmark_files():
    """The committed grid must agree with the per-run metrics beside it."""
    benchmarks = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
    grid_path = benchmarks / "food11_ablation_grid.json"
    if not grid_path.is_file():
        pytest.skip("ablation grid not present in this checkout")

    grid = json.loads(grid_path.read_text())
    assert len(grid) == 9

    # Three cells predate the ablation and are stored under their original names.
    aliases = {
        "b0_224": "food11_efficientnet_b0_cbam",
        "b4_224": "food11_efficientnet_b4_cbam_224",
        "b4_380": "food11_efficientnet_b4_cbam",
    }
    for key, cell in grid.items():
        model, px = key.split("_")
        stem = aliases.get(key, f"food11_efficientnet_{model}_cbam_{px}")
        metrics_path = benchmarks / f"{stem}_metrics.json"
        assert metrics_path.is_file(), f"{key} -> {metrics_path.name} missing"

        metrics = json.loads(metrics_path.read_text())
        assert abs(cell["acc"] - metrics["accuracy"]) < 1e-9, key
        assert abs(cell["f1"] - metrics["macro_f1"]) < 1e-9, key
