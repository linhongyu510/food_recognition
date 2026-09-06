"""Publish a trained checkpoint to the Hugging Face Hub, with a model card.

The card is generated from the run's own ``metrics.json`` and the checkpoint's
embedded config, so the published numbers cannot drift from what was measured::

    pip install -e ".[app]"
    huggingface-cli login
    python scripts/publish_to_hf.py \\
        --run runs/food11_effnet_cbam \\
        --repo-id <user>/food-recognition-food11-effnet-b0-cbam

Use ``--dry-run`` to render the card and print the upload plan without
contacting the Hub.

Licensing note: Food-101's terms permit scientific fair use only -- the images
are not ETH Zurich's property. Weights trained on it are a derivative work, so
publish them for research use and say so on the card. This script refuses to
attach a permissive licence to a Food-101 model unless you override it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# 1000 rather than 1024: Hub file sizes are reported in decimal MB.
_BYTES_PER_MB = 1000 * 1000


def _load_run(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    """Read a run's checkpoint, metrics and history, failing loudly if absent."""
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    metrics_path = run_dir / "metrics.json"

    for path in (checkpoint_path, metrics_path):
        if not path.is_file():
            raise SystemExit(f"missing {path} -- is {run_dir} a finished run?")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text())

    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []

    return payload, metrics, history


def _per_class_extremes(metrics: dict) -> tuple[dict | None, dict | None]:
    """Return the worst and best class by F1, or (None, None) if unavailable."""
    per_class = metrics.get("per_class") or []
    if not per_class:
        return None, None
    ordered = sorted(per_class, key=lambda row: row["f1"])
    return ordered[0], ordered[-1]


def build_card(
    payload: dict,
    metrics: dict,
    history: list[dict],
    *,
    dataset: str,
    licence: str,
    repo_id: str,
    source_commit: str | None,
) -> str:
    """Render a model card whose every number comes from the run itself."""
    config = payload.get("config") or {}
    classes = payload.get("classes") or []
    model_name = config.get("model_name", "unknown")
    image_size = config.get("image_size", "unknown")
    epoch = payload.get("epoch", "unknown")

    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    total_minutes = sum(e.get("duration_sec", 0) for e in history) / 60 if history else 0

    worst, best = _per_class_extremes(metrics)

    tags = "\n".join(
        f"- {tag}"
        for tag in (
            "image-classification",
            "pytorch",
            "food",
            "cbam",
            "attention",
            dataset.lower().replace(" ", "-"),
        )
    )

    lines = [
        "---",
        f"license: {licence}",
        "library_name: pytorch",
        "pipeline_tag: image-classification",
        "tags:",
        tags,
        "---",
        "",
        f"# {model_name} — {dataset}",
        "",
        f"Food image classifier trained with "
        f"[food_recognition](https://github.com/linhongyu510/food_recognition). "
        f"`{model_name}` is a torchvision backbone with a "
        f"[CBAM](https://arxiv.org/abs/1807.06521) block "
        f"(channel then spatial attention) between the features and the "
        f"classification head.",
        "",
        "## Measured performance",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    if accuracy is not None:
        lines.append(f"| Validation accuracy | **{accuracy * 100:.2f}%** |")
    if macro_f1 is not None:
        lines.append(f"| Macro F1 | {macro_f1:.4f} |")
    lines += [
        f"| Classes | {len(classes)} |",
        f"| Input resolution | {image_size}px |",
        f"| Checkpoint epoch | {epoch} |",
    ]
    if total_minutes:
        lines.append(f"| Training wall clock | {total_minutes:.0f} min |")
    if worst and best:
        lines.append(
            f"| Hardest / easiest class | {worst['name']} (F1 {worst['f1']:.3f}) / "
            f"{best['name']} (F1 {best['f1']:.3f}) |"
        )

    lines += [
        "",
        "These figures come from this run's own `metrics.json`, computed over the "
        "full validation split. The repository re-scores every published "
        "checkpoint through a separate evaluation path and requires the two to "
        "agree to six decimal places.",
        "",
        "## Usage",
        "",
        "```bash",
        "pip install git+https://github.com/linhongyu510/food_recognition.git",
        "```",
        "",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "from food_recognition import load_predictor",
        "",
        f'checkpoint = hf_hub_download("{repo_id}", "best.pt")',
        "predictor = load_predictor(checkpoint)",
        "",
        'prediction = predictor.predict("photo.jpg", topk=3)',
        "print(prediction.label, prediction.topk)",
        "```",
        "",
        "The checkpoint embeds its architecture, input resolution and class "
        "names, so nothing needs to be specified again at load time.",
        "",
        "To see which pixels drove a prediction:",
        "",
        "```bash",
        "food-recognition-gradcam --checkpoint best.pt --input photo.jpg --side-by-side",
        "```",
        "",
        "## Training",
        "",
        f"- Architecture: `{model_name}`",
        f"- Input: {image_size}px",
        f"- Epochs: {config.get('epochs', 'unknown')}",
        f"- Batch size: {config.get('batch_size', 'unknown')}",
        f"- Learning rate: {config.get('learning_rate', 'unknown')}",
        f"- Scheduler: {config.get('scheduler', 'unknown')}",
        "- Fully supervised (no pseudo-labelling)"
        if not (config.get("semi_supervised") or {}).get("enabled")
        else "- Semi-supervised (pseudo-labelling enabled)",
    ]
    if source_commit:
        lines.append(f"- Source commit: `{source_commit}`")

    lines += [
        "",
        "## Limitations",
        "",
        f"- Trained on {dataset} only. Accuracy on dishes, cuisines or "
        "photographic conditions outside that distribution is unmeasured, and "
        "the model will still return a confident label for an image containing "
        "no food at all.",
        "- Predictions are not a nutrition, allergen or food-safety judgement. "
        "Do not use them where a misclassification carries a health risk.",
        "- The reported figure is validation accuracy on a public benchmark "
        "split, which is an optimistic estimate of field performance.",
    ]

    if dataset.lower().startswith("food-101"):
        lines += [
            "",
            "## Dataset licence",
            "",
            "Food-101 consists of Foodspotting images that are not the property "
            "of ETH Zurich; the dataset terms allow scientific fair use, and any "
            "use beyond that must be negotiated with the picture owners. These "
            "weights are a derivative of that data and are released for research "
            "use under the same understanding.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="run directory")
    parser.add_argument("--repo-id", required=True, help="target Hub repo, user/name")
    parser.add_argument(
        "--dataset", default=None, help="dataset name (default: inferred from classes)"
    )
    parser.add_argument(
        "--license",
        dest="licence",
        default=None,
        help="card licence (default: mit, or 'other' for Food-101)",
    )
    parser.add_argument("--commit", default=None, help="source commit to record")
    parser.add_argument("--private", action="store_true", help="create a private repo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the card and print the plan without uploading",
    )
    args = parser.parse_args(argv)

    payload, metrics, history = _load_run(args.run)
    classes = payload.get("classes") or []

    dataset = args.dataset or (
        "Food-101" if len(classes) == 101 else f"Food-{len(classes)}"
    )

    # Food-101's terms are not permissive, so do not let a default licence
    # silently mislabel a derivative of it.
    licence = args.licence or (
        "other" if dataset.lower().startswith("food-101") else "mit"
    )

    card = build_card(
        payload,
        metrics,
        history,
        dataset=dataset,
        licence=licence,
        repo_id=args.repo_id,
        source_commit=args.commit,
    )

    checkpoint_path = args.run / "checkpoints" / "best.pt"
    size_mb = checkpoint_path.stat().st_size / _BYTES_PER_MB

    print(f"run        : {args.run}")
    print(f"repo       : {args.repo_id} ({'private' if args.private else 'public'})")
    print(f"dataset    : {dataset}")
    print(f"licence    : {licence}")
    print(f"checkpoint : {checkpoint_path.name} ({size_mb:.0f} MB)")
    print(f"accuracy   : {metrics.get('accuracy')}")
    print()

    if args.dry_run:
        print("--- README.md (not uploaded) ---")
        print(card)
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "needs huggingface_hub; install with: pip install -e '.[app]'"
        ) from exc

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card_path = args.run / "README.md"
    card_path.write_text(card)

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="best.pt",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    for name in ("metrics.json", "history.json"):
        source = args.run / name
        if source.is_file():
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=name,
                repo_id=args.repo_id,
                repo_type="model",
            )

    print(f"published: https://huggingface.co/{args.repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
