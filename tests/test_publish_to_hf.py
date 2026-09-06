"""Tests for the Hugging Face publishing script.

The model card is the artefact a downstream user reads to decide whether to
trust a checkpoint, so these verify that every published number is derived from
the run's own files rather than hand-written -- and that Food-101's non-permissive
terms are not papered over with a default licence.

Nothing here contacts the Hub: ``build_card`` is pure, and the upload path is
exercised only through ``--dry-run``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_to_hf.py"


def _load_script():
    """Import publish_to_hf.py, which lives outside the installed package."""
    spec = importlib.util.spec_from_file_location("publish_to_hf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_to_hf"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _make_run(
    root: Path,
    *,
    classes: list[str],
    accuracy: float,
    model_name: str = "efficientnet_b0_cbam",
    image_size: int = 224,
) -> Path:
    """Build a miniature run directory in the layout Trainer writes."""
    run = root / "run"
    (run / "checkpoints").mkdir(parents=True)

    torch.save(
        {
            "epoch": 7,
            "classes": classes,
            "model_state": {"fc.weight": torch.zeros(len(classes), 4)},
            "config": {
                "model_name": model_name,
                "image_size": image_size,
                "num_classes": len(classes),
                "epochs": 30,
                "batch_size": 64,
                "learning_rate": 0.0001,
                "scheduler": "cosine",
                "dropout": 0.3,
            },
        },
        run / "checkpoints" / "best.pt",
    )

    (run / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": 0.8907,
                "per_class": [
                    {"name": classes[0], "f1": 0.42, "precision": 1.0,
                     "recall": 1.0, "support": 5},
                    {"name": classes[-1], "f1": 0.99, "precision": 1.0,
                     "recall": 1.0, "support": 5},
                ],
            }
        )
    )
    (run / "history.json").write_text(
        json.dumps([{"epoch": 1, "duration_sec": 60.0}, {"epoch": 2, "duration_sec": 120.0}])
    )
    return run


def test_card_reports_the_runs_own_accuracy(script, tmp_path: Path):
    """The headline figure must come from metrics.json, not an argument."""
    run = _make_run(tmp_path, classes=["00", "01"], accuracy=0.936364)
    payload, metrics, history = script._load_run(run)

    card = script.build_card(
        payload,
        metrics,
        history,
        dataset="Food-11",
        licence="mit",
        repo_id="u/r",
        source_commit="abc1234",
    )

    assert "**93.64%**" in card
    assert "0.8907" in card
    # 60s + 120s = 3 minutes, summed from history rather than declared.
    assert "3 min" in card
    assert "abc1234" in card
    assert "efficientnet_b0_cbam" in card


def test_card_names_the_hardest_and_easiest_class(script, tmp_path: Path):
    """Per-class extremes are computed, so a reader sees the weak spot."""
    run = _make_run(tmp_path, classes=["steak", "edamame"], accuracy=0.89)
    payload, metrics, history = script._load_run(run)

    card = script.build_card(
        payload,
        metrics,
        history,
        dataset="Food-101",
        licence="other",
        repo_id="u/r",
        source_commit=None,
    )

    assert "steak (F1 0.420)" in card
    assert "edamame (F1 0.990)" in card


def test_food101_card_carries_the_dataset_licence_warning(script, tmp_path: Path):
    """Food-101 permits scientific fair use only; the card must say so."""
    run = _make_run(tmp_path, classes=[f"c{i}" for i in range(101)], accuracy=0.891129)
    payload, metrics, history = script._load_run(run)

    card = script.build_card(
        payload,
        metrics,
        history,
        dataset="Food-101",
        licence="other",
        repo_id="u/r",
        source_commit=None,
    )

    assert "Dataset licence" in card
    assert "scientific fair use" in card
    assert "license: other" in card


def test_food11_card_omits_the_food101_licence_section(script, tmp_path: Path):
    """The warning is specific to Food-101 and must not leak onto other cards."""
    run = _make_run(tmp_path, classes=["00", "01"], accuracy=0.95)
    payload, metrics, history = script._load_run(run)

    card = script.build_card(
        payload,
        metrics,
        history,
        dataset="Food-11",
        licence="mit",
        repo_id="u/r",
        source_commit=None,
    )

    assert "Dataset licence" not in card
    assert "license: mit" in card


def test_card_states_limitations(script, tmp_path: Path):
    """A published classifier needs its scope written down, not implied."""
    run = _make_run(tmp_path, classes=["00", "01"], accuracy=0.95)
    payload, metrics, history = script._load_run(run)

    card = script.build_card(
        payload,
        metrics,
        history,
        dataset="Food-11",
        licence="mit",
        repo_id="u/r",
        source_commit=None,
    )

    assert "Limitations" in card
    assert "nutrition" in card


def test_dry_run_defaults_food101_to_a_non_permissive_licence(script, tmp_path: Path, capsys):
    """101 classes implies Food-101, which must not default to MIT."""
    run = _make_run(tmp_path, classes=[f"c{i}" for i in range(101)], accuracy=0.891129)

    script.main(["--run", str(run), "--repo-id", "u/r", "--dry-run"])

    out = capsys.readouterr().out
    assert "dataset    : Food-101" in out
    assert "licence    : other" in out
    # --dry-run must not create the repo README as a side effect.
    assert not (run / "README.md").exists()


def test_missing_run_files_fail_loudly(script, tmp_path: Path):
    """A half-finished run should error, not publish an empty card."""
    empty = tmp_path / "nothing"
    empty.mkdir()

    with pytest.raises(SystemExit, match="missing"):
        script._load_run(empty)


def _extract_card(stdout: str) -> str:
    """Return just the model card from the script's stdout.

    The progress log above the card contains its own `--- ... ---` separators, so
    a naive `stdout.index("---")` finds the wrong one. Anchor on a line that is
    exactly the YAML fence.
    """
    lines = stdout.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(fences) >= 2, "no YAML front matter found in output"
    return "\n".join(lines[fences[0] :])


def test_card_carries_machine_readable_model_index(script, tmp_path: Path, capsys):
    """The Hub renders 'Evaluation results' from model-index, so it must be valid
    YAML carrying the same numbers as the prose table."""
    yaml = pytest.importorskip("yaml")
    run = _make_run(tmp_path, classes=[f"c{i}" for i in range(11)], accuracy=0.954545)

    script.main(["--run", str(run), "--repo-id", "u/r", "--dry-run"])
    out = capsys.readouterr().out

    card = _extract_card(out)
    front = yaml.safe_load(card.split("---")[1])

    results = front["model-index"][0]["results"]
    by_type = {m["type"]: m["value"] for r in results for m in r["metrics"]}
    metrics = json.loads((run / "metrics.json").read_text())
    assert by_type["accuracy"] == metrics["accuracy"]
    assert by_type["f1"] == metrics["macro_f1"]
    assert results[0]["task"]["type"] == "image-classification"


def test_card_passes_hub_validation(script, tmp_path: Path, capsys):
    """huggingface_hub's own validator must accept the card we generate."""
    hub = pytest.importorskip("huggingface_hub")
    run = _make_run(tmp_path, classes=[f"c{i}" for i in range(11)], accuracy=0.9545)

    script.main(["--run", str(run), "--repo-id", "u/r", "--dry-run"])
    out = capsys.readouterr().out

    card_text = _extract_card(out)
    card = hub.ModelCard(card_text)
    card.validate()  # raises if the Hub would reject the metadata
    assert card.data.to_dict()["model-index"]
