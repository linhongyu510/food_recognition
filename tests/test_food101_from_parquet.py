"""Tests for the Parquet -> official-layout Food-101 converter.

The real mirror is ~4.8 GB, so these build miniature Parquet shards carrying the
same schema the mirror uses (image struct of bytes+path, int64 label, class names
in the huggingface schema metadata) and verify the conversion against them.

The property that matters is that the official split survives the round trip:
meta/train.txt must name exactly the images the shards said were training
images, keyed by the original filename rather than by row order.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pyarrow", reason="pyarrow is an optional [food101] extra")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "food101_from_parquet.py"

CLASSES = ["apple_pie", "baby_back_ribs", "waffles"]


def _load_script():
    """Import food101_from_parquet.py, which lives outside the package."""
    spec = importlib.util.spec_from_file_location("food101_from_parquet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["food101_from_parquet"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_shard(path: Path, rows: list[tuple[int, str, bytes]]) -> None:
    """Write one shard with the mirror's schema, including label metadata."""
    table = pa.table(
        {
            "image": pa.array(
                [{"bytes": data, "path": name} for _, name, data in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "label": pa.array([label for label, _, _ in rows], type=pa.int64()),
        }
    )
    blob = json.dumps(
        {
            "info": {
                "features": {
                    "image": {"_type": "Image"},
                    "label": {"_type": "ClassLabel", "names": CLASSES},
                }
            }
        }
    )
    table = table.replace_schema_metadata({"huggingface": blob})
    pq.write_table(table, path)


@pytest.fixture
def fake_mirror(tmp_path: Path, script) -> Path:
    """Miniature shards matching the real name pattern; 2 train + 1 test each."""
    shards = tmp_path / "parquet"
    shards.mkdir()

    # Deliberately spread each class across shards, so a converter that assumed
    # shard order == class order would produce a wrong mapping.
    train = [
        [(0, "1001.jpg", _jpeg((200, 10, 10))), (1, "2001.jpg", _jpeg((10, 200, 10)))],
        [(2, "3001.jpg", _jpeg((10, 10, 200))), (0, "1002.jpg", _jpeg((150, 20, 20)))],
    ]
    for index, rows in enumerate(train):
        name = f"train-{index:05d}-of-{len(train):05d}.parquet"
        _write_shard(shards / name, rows)

    test = [[(1, "2002.jpg", _jpeg((20, 150, 20))), (2, "3002.jpg", _jpeg((20, 20, 150)))]]
    for index, rows in enumerate(test):
        name = f"validation-{index:05d}-of-{len(test):05d}.parquet"
        _write_shard(shards / name, rows)

    # Point the module's shard counts at the miniature set.
    script.TRAIN_SHARDS = len(train)
    script.VAL_SHARDS = len(test)
    script.SPLITS = (
        ("train", "train", len(train)),
        ("test", "validation", len(test)),
    )
    script.EXPECTED_PER_CLASS = {"train": 2, "test": 1}
    return shards


def test_convert_rebuilds_official_layout(script, fake_mirror, tmp_path):
    """images/<class>/<stem>.jpg + meta/{train,test,classes}.txt all appear."""
    output = tmp_path / "food-101"
    counts = script.convert(fake_mirror, output)

    assert counts == {"train": 4, "test": 2}
    assert sorted(p.name for p in (output / "images").iterdir()) == sorted(CLASSES)
    for name in ("train.txt", "test.txt", "classes.txt"):
        assert (output / "meta" / name).is_file()
    assert (output / "meta" / "classes.txt").read_text().split() == CLASSES


def test_split_membership_follows_the_shards(script, fake_mirror, tmp_path):
    """The official split is preserved, keyed by class/stem — not by row order."""
    output = tmp_path / "food-101"
    script.convert(fake_mirror, output)

    train = set((output / "meta" / "train.txt").read_text().split())
    test = set((output / "meta" / "test.txt").read_text().split())

    assert train == {
        "apple_pie/1001",
        "baby_back_ribs/2001",
        "waffles/3001",
        "apple_pie/1002",
    }
    assert test == {"baby_back_ribs/2002", "waffles/3002"}
    # A leak here would silently inflate any accuracy measured afterwards.
    assert not train & test


def test_written_images_are_readable_and_correctly_placed(
    script, fake_mirror, tmp_path
):
    """Every meta entry resolves to a decodable JPEG under its own class."""
    output = tmp_path / "food-101"
    script.convert(fake_mirror, output)

    entries = (output / "meta" / "train.txt").read_text().split()
    entries += (output / "meta" / "test.txt").read_text().split()
    assert entries

    for entry in entries:
        class_name, stem = entry.split("/")
        path = output / "images" / class_name / f"{stem}.jpg"
        assert path.is_file(), f"{entry} listed in meta/ but missing on disk"
        with Image.open(path) as image:
            image.load()
            assert image.format == "JPEG"


def test_class_names_come_from_schema_metadata(script, fake_mirror):
    """Label order is read from the shard, not guessed alphabetically."""
    shard = fake_mirror / "validation-00000-of-00001.parquet"
    assert script._class_names(shard) == CLASSES


def test_missing_schema_metadata_is_reported(script, tmp_path):
    """A shard without label metadata fails loudly rather than mislabelling."""
    shard = tmp_path / "validation-00000-of-00001.parquet"
    table = pa.table(
        {
            "image": pa.array(
                [{"bytes": _jpeg((1, 2, 3)), "path": "1.jpg"}],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "label": pa.array([0], type=pa.int64()),
        }
    )
    pq.write_table(table, shard)

    with pytest.raises(ValueError, match="no huggingface schema metadata"):
        script._class_names(shard)


def test_missing_shard_is_reported(script, fake_mirror, tmp_path):
    """A deleted shard raises instead of silently writing a short split."""
    (fake_mirror / "train-00001-of-00002.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="missing shard"):
        script.convert(fake_mirror, tmp_path / "out")


def test_absent_parquet_dir_explains_how_to_fix(script, tmp_path):
    """The first-shard check names the file and suggests the remedy."""
    with pytest.raises(FileNotFoundError, match="Run without --parquet-dir"):
        script.convert(tmp_path / "nothing-here", tmp_path / "out")


def test_shard_names_match_the_mirror_pattern(script):
    """Names are built to the mirror's zero-padded convention."""
    assert script._shard_names("train", 8)[0] == "train-00000-of-00008.parquet"
    assert script._shard_names("train", 8)[-1] == "train-00007-of-00008.parquet"
    assert (
        script._shard_names("validation", 3)[1] == "validation-00001-of-00003.parquet"
    )


def test_cli_runs_end_to_end_on_local_shards(script, fake_mirror, tmp_path, capsys):
    """--parquet-dir skips downloading and converts what is already on disk."""
    output = tmp_path / "food-101"
    argv = [
        "food101_from_parquet.py",
        "--output",
        str(output),
        "--parquet-dir",
        str(fake_mirror),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        code = script.main()
    finally:
        sys.argv = original

    assert code == 0
    out = capsys.readouterr().out
    # The miniature set is smaller than the real one, so main() warns rather
    # than claiming the official totals.
    assert "prepare_food101.py" in out
    assert (output / "meta" / "train.txt").is_file()
    # --parquet-dir means the shards were not ours to delete.
    assert list(fake_mirror.glob("*.parquet"))
