"""Gradio demo: classify a food photo and show what the model looked at.

Run locally::

    pip install -e ".[app]"
    python app.py --checkpoint runs/food11_effnet_cbam/checkpoints/best.pt

Or point it at a checkpoint published on the Hugging Face Hub::

    python app.py --hf-repo <user>/food-recognition-food11-effnet-b0-cbam

The same file works unchanged as a Hugging Face Space: with no arguments it
reads ``FR_CHECKPOINT``/``FR_HF_REPO`` from the environment, which is how a
Space passes configuration in.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from food_recognition import GradCAM, load_predictor, overlay_heatmap

# Grad-CAM needs gradients, so the heatmap costs a backward pass on top of the
# forward one. Measured on 2 CPU threads (Hugging Face's free CPU Basic tier):
# ~64 ms/image for efficientnet_b0_cbam at 224px, ~221 ms for B4 at 380px.
# Both are comfortable for an interactive demo without a GPU.

DESCRIPTION = """
# Food recognition

Upload a food photo. The model returns its top predictions, and the Grad-CAM
panel shows which pixels drove the answer — useful for spotting a model that is
right for the wrong reason (keying on a plate rim or watermark rather than the
food itself).
"""


def _resolve_checkpoint(explicit: str | None, hf_repo: str | None) -> Path:
    """Find a checkpoint locally, or download one from the Hub."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        return path

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise SystemExit(
                "--hf-repo needs huggingface_hub; install with: pip install -e '.[app]'"
            ) from exc
        return Path(hf_hub_download(repo_id=hf_repo, filename="best.pt"))

    raise SystemExit(
        "no checkpoint given; pass --checkpoint PATH, or --hf-repo REPO_ID, "
        "or set FR_CHECKPOINT / FR_HF_REPO in the environment"
    )


def build_interface(checkpoint: Path, *, device: str | None = None) -> gr.Blocks:
    """Build the demo around one checkpoint, loaded once at startup."""
    predictor = load_predictor(checkpoint, device=device)
    cam = GradCAM(predictor.model, classes=predictor.classes)
    image_size = predictor.image_size
    # Written into the UI so a visitor can tell which model produced the numbers.
    label = f"{len(predictor.classes)} classes · {image_size}px · {predictor.device}"

    def classify(image: Image.Image | None, topk: int, explain: bool):
        if image is None:
            return {}, None

        # Predictor and GradCAM both read from disk, which keeps the demo on the
        # exact same preprocessing path as the CLI rather than a parallel one.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            image.convert("RGB").save(path)

            prediction = predictor.predict(path, topk=int(topk))
            scores = dict(prediction.topk)

            if not explain:
                return scores, None

            result, display = cam.generate_from_path(path, image_size=image_size)
            return scores, overlay_heatmap(display, result.heatmap, alpha=0.5)

    with gr.Blocks(title="Food recognition") as demo:
        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"**Loaded model:** `{checkpoint.name}` — {label}")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Food photo")
                topk_slider = gr.Slider(
                    1, min(10, len(predictor.classes)), value=3, step=1, label="Top-k"
                )
                explain_toggle = gr.Checkbox(value=True, label="Show Grad-CAM")
                submit = gr.Button("Classify", variant="primary")
            with gr.Column():
                label_output = gr.Label(label="Predictions")
                cam_output = gr.Image(label="Grad-CAM overlay")

        inputs = [image_input, topk_slider, explain_toggle]
        outputs = [label_output, cam_output]
        submit.click(classify, inputs=inputs, outputs=outputs)
        image_input.upload(classify, inputs=inputs, outputs=outputs)

    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("FR_CHECKPOINT"),
        help="path to a best.pt (default: $FR_CHECKPOINT)",
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("FR_HF_REPO"),
        help="Hub model repo holding best.pt (default: $FR_HF_REPO)",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="expose a temporary public URL"
    )
    args = parser.parse_args(argv)

    checkpoint = _resolve_checkpoint(args.checkpoint, args.hf_repo)
    demo = build_interface(checkpoint, device=args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
