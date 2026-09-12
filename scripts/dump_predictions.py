"""Dump per-image correctness vectors for one checkpoint on a labelled directory.

``food-recognition-eval`` reports aggregate metrics and a confusion matrix, which
is enough to compare two runs' accuracies but *not* enough to test whether the
difference between them is real: that needs to know which images each run got
right, so the two runs can be paired image by image.

This writes a compact JSON with one 0/1 entry per image, in a fixed sorted order,
so two checkpoints evaluated over the same directory produce aligned vectors::

    python scripts/dump_predictions.py \\
        --checkpoint runs/bench_food101/checkpoints/best.pt \\
        --data-dir data/food-101/validation \\
        --json-out reval/b0_correct.json

The vectors feed ``scripts/compare_runs.py``, which runs McNemar's exact test and
an image-level paired bootstrap over them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from food_recognition.data import LabeledImageDataset, build_transform
    from food_recognition.predict import load_predictor

    predictor = load_predictor(args.checkpoint, device=args.device)
    dataset = LabeledImageDataset(
        args.data_dir, transform=build_transform(predictor.image_size, is_train=False)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    correct: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = predictor.model(images.to(predictor.device, non_blocking=True))
            preds = logits.argmax(dim=1).cpu()
            correct.extend((preds == labels).to(torch.int64).tolist())

    # Hash the file order so a mismatched pairing is detectable rather than silent.
    paths = [str(p) for p, _ in dataset.samples]
    order_hash = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:16]

    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "image_size": predictor.image_size,
        "n_images": len(correct),
        "n_correct": sum(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "order_sha256_16": order_hash,
        "correct": correct,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload) + "\n")
    print(
        f"{payload['n_correct']}/{payload['n_images']} = {payload['accuracy']:.6f} "
        f"order={order_hash} -> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
