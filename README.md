# Food Recognition

Food image classification with PyTorch: a configurable training pipeline with
CBAM attention, self-training on unlabelled data, and CLI tools for training,
evaluation and inference.

[![CI](https://github.com/linhongyu510/food_recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/linhongyu510/food_recognition/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.4-ee4c2c)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status.** The pipeline, CLI and 176-test suite are verified and run in CI on
> every push. Food-11 accuracy is measured and recorded with full provenance in
> [Benchmarks](#benchmarks); Food-101 is not yet measured.

---

## Contents

- [Install](#install)
- [Try it in 30 seconds](#try-it-in-30-seconds)
- [Data layout](#data-layout)
- [Training](#training)
- [Evaluation and inference](#evaluation-and-inference)
- [Explaining predictions (Grad-CAM)](#explaining-predictions-grad-cam)
- [Python API](#python-api)
- [Models](#models)
- [Configuration](#configuration)
- [Self-training](#self-training)
- [Benchmarks](#benchmarks)
- [Project layout](#project-layout)
- [Development](#development)

---

## Install

Requires Python 3.9+ and PyTorch 2.4+.

```bash
git clone https://github.com/linhongyu510/food_recognition.git
cd food_recognition

python -m venv .venv && source .venv/bin/activate

# CPU-only torch (skips ~2GB of CUDA wheels). For GPU, follow pytorch.org.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e .
```

Installing exposes three commands: `food-recognition-train`,
`food-recognition-eval` and `food-recognition-predict`.

## Try it in 30 seconds

No dataset download needed — this generates synthetic data and runs the full
train → evaluate → predict loop on CPU:

```bash
python scripts/make_sample_data.py --output data/sample
food-recognition-train --config configs/smoke_test.yaml
```

```
device=cpu
model=simple_cnn trainable_params=1,553,475
train samples=24
val samples=12
epoch   1/2 | loss 1.0129 | acc 0.6667 | val_loss 0.1638 | val_acc 1.0000 | ... | <- best
epoch   2/2 | loss 0.1960 | acc 0.9667 | val_loss 0.0116 | val_acc 1.0000 | pseudo 6 | ...

best val accuracy : 1.0000 (epoch 1)
best checkpoint   : runs/smoke/checkpoints/best.pt

class         precision     recall         f1  support
00               1.0000     1.0000     1.0000        4
01               1.0000     1.0000     1.0000        4
02               1.0000     1.0000     1.0000        4

accuracy                               1.0000       12
macro avg        1.0000     1.0000     1.0000       12
weighted avg     1.0000     1.0000     1.0000       12
```

This toy task is deliberately trivial (three solid colours); it verifies the
plumbing, not model quality.

## Data layout

Labelled splits use one sub-directory per class. The unlabelled pool is a
**flat** directory:

```
data/food-11/
├── training/
│   ├── labeled/          # class sub-directories
│   │   ├── 00/  01/  02/  ...  10/
│   └── unlabeled/        # flat: images directly inside, no class dirs
│       ├── 0001.jpg  0002.jpg  ...
└── validation/
    ├── 00/  01/  02/  ...  10/
```

Class directories named with digits are sorted **numerically**, so `10/` maps to
index 10 rather than sorting between `1/` and `2/`. Any directory names work —
`apple_pie/`, `bread/` — and the resolved names are stored in the checkpoint so
predictions come back as readable labels.

To download Food-11:

```bash
pip install -e ".[download]"
python scripts/download_dataset.py
```

### Food-101

Food-101 ships in its own layout — `images/<class>/<hash>.jpg` plus
`meta/train.txt` and `meta/test.txt` listing the official splits. Download it
from [the ETH Zurich page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
(~5 GB), then convert it:

```bash
python scripts/prepare_food101.py --source /path/to/food-101 --output data/food-101
```

This creates symlinks by default, so it finishes in seconds and adds almost no
disk usage. The links are relative, so moving the source and output together
keeps them valid. Use `--copy` for real files (needs another ~5 GB) when the
output has to stand on its own, or `--limit-per-class 50` for a fast smoke run.

The split assignment comes from the official meta files rather than a reshuffle,
so results stay comparable with published numbers.

```bash
food-recognition-train --config configs/food101_efficientnet_cbam.yaml
```

Food-101 is 101 classes and 75,750 training images, so expect roughly 20–40
minutes per epoch on a single mid-range GPU — considerably heavier than Food-11.

## Training

```bash
# Built-in configs
food-recognition-train --config configs/food11_resnet18.yaml
food-recognition-train --config configs/food11_efficientnet_cbam.yaml

# Override any field from the command line
food-recognition-train --config configs/food11_resnet18.yaml \
    --model-name resnet50_cbam --epochs 40 --batch-size 64 --lr 1e-4

# Or skip configs entirely
food-recognition-train \
    --model-name efficientnet_b0_cbam --num-classes 11 \
    --train-dir data/food-11/training/labeled \
    --val-dir data/food-11/validation \
    --epochs 30 --device cuda
```

Each run writes to `output_dir`:

```
runs/food11_resnet18/
├── checkpoints/
│   ├── best.pt        # best validation accuracy
│   └── last.pt        # most recent epoch
├── history.json       # per-epoch loss/acc/LR/duration
└── metrics.json       # best epoch: per-class P/R/F1 + confusion matrix
```

`--help` lists every flag. Relative paths in configs resolve against the
current working directory, so run commands from the repository root.

## Evaluation and inference

```bash
food-recognition-eval \
    --checkpoint runs/food11_resnet18/checkpoints/best.pt \
    --data-dir data/food-11/validation \
    --json-out report.json
```

```bash
# Single image
food-recognition-predict --checkpoint runs/.../best.pt --input photo.jpg

# Whole directory, top-5, saved as JSON
food-recognition-predict --checkpoint runs/.../best.pt \
    --input data/food-11/testing --topk 5 --json-out predictions.json
```

```
photo.jpg
  -> 07 (0.9412)  [07 0.941, 03 0.038, 01 0.011]
```

Checkpoints embed their architecture, image size and class names, so neither
command needs to be told the model again.

## Explaining predictions (Grad-CAM)

Grad-CAM shows *which pixels* drove a prediction, which is how you catch a model
that is right for the wrong reason — keying on a plate rim or a watermark rather
than the food.

```bash
# One image; writes gradcam/photo_gradcam.png
food-recognition-gradcam \
    --checkpoint runs/food11_resnet18/checkpoints/best.pt \
    --input photo.jpg

# Original next to the overlay, for a whole directory
food-recognition-gradcam \
    --checkpoint runs/food11_resnet18/checkpoints/best.pt \
    --input data/food-11/testing \
    --output-dir cams --side-by-side
```

```
photo.jpg
  -> 07 (0.9412)  saved cams/photo_gradcam.png
```

Real output from the benchmarked `efficientnet_b0_cbam` model on a validation
image of class `06` (noodles/pasta), predicted correctly at 0.9104 confidence —
original left, overlay right. The heat sits on the pasta itself rather than the
plate rim or background, which is what you want to confirm before trusting a
model's accuracy number:

![Grad-CAM on Food-11 noodles/pasta](docs/benchmarks/gradcam_food11_noodles.png)

Pass `--class-index` to ask *"why not that other class?"* — it explains the class
you name instead of the predicted one, which is the useful view when a model is
confidently wrong.

```python
from food_recognition import GradCAM, load_predictor, overlay_heatmap

predictor = load_predictor("runs/food11_resnet18/checkpoints/best.pt")
cam = GradCAM(predictor.model, classes=predictor.classes)

result, display = cam.generate_from_path("photo.jpg", image_size=224)
print(result.class_name, result.confidence)   # 07 0.9412
print(result.heatmap.shape)                   # (224, 224), values in [0, 1]

overlay_heatmap(display, result.heatmap, alpha=0.5).save("cam.png")
```

The target convolution layer is resolved automatically per architecture
(`layer4` for ResNet, `features` for EfficientNet/VGG, the CBAM block for CBAM
models). Override it with `GradCAM(model, target_layer=...)`. Hooks are removed
in a `finally` block, so a failed call cannot leave them attached, and the
model's train/eval mode is restored afterwards.

Implementation notes: this uses `register_full_backward_hook` (PyTorch's
docstring marks `register_backward_hook` as deprecated, warning that "the
behavior of this function will change in future versions") and resizes with
`torch.nn.functional.interpolate`, so **OpenCV is not a dependency**. The
colormap is computed in NumPy, so matplotlib is not required either.

## Python API

```python
from food_recognition import TrainingConfig, train_model

cfg = TrainingConfig(
    model_name="efficientnet_b0_cbam",
    num_classes=11,
    train_dir="data/food-11/training/labeled",
    val_dir="data/food-11/validation",
    epochs=30,
    batch_size=32,
)
summary = train_model(cfg)

print(summary.best_accuracy, summary.best_epoch)
print(summary.final_report.format_table())
```

```python
from food_recognition import load_predictor

predictor = load_predictor("runs/exp/checkpoints/best.pt")
print(predictor.predict("photo.jpg").label)
```

Building blocks are importable directly — `CBAM`, `LabeledImageDataset`,
`compute_metrics`, `Trainer`, `initialize_model`, `EarlyStopper`.

## Models

Pass any of these as `model_name` (29 total, `available_models()` lists them):

| Family | Names |
| --- | --- |
| Baseline | `simple_cnn` |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101` |
| EfficientNet | `efficientnet_b0` … `efficientnet_b4` |
| Other | `alexnet`, `vgg11_bn`, `vgg16_bn`, `densenet121`, `squeezenet`, `googlenet`, `mobilenet_v3_large`, `convnext_tiny` |
| **+ CBAM** | append `_cbam` to any ResNet / EfficientNet / DenseNet / ConvNeXt name |

### CBAM

[CBAM](https://arxiv.org/abs/1807.06521) (Woo et al., ECCV 2018) applies channel
attention then spatial attention to the backbone's feature map:

```
backbone features → channel attention → spatial attention → pool → linear head
```

Channel attention pools with **both** average and max descriptors through a
shared MLP, per the paper. The attention width is read from the backbone at
build time, so `efficientnet_b4_cbam` correctly uses 1792 channels while
`efficientnet_b0_cbam` uses 1280.

## Configuration

Shipped configs: `configs/food11_resnet18.yaml`,
`configs/food11_efficientnet_cbam.yaml`, `configs/smoke_test.yaml`.

| Group | Keys |
| --- | --- |
| Model | `model_name`, `num_classes`, `use_pretrained`, `linear_probe`, `dropout` |
| Data | `train_dir`, `val_dir`, `unlabeled_dir`, `image_size`, `batch_size`, `num_workers`, `use_autoaugment`, `class_names` |
| Optimisation | `epochs`, `learning_rate`, `weight_decay`, `label_smoothing`, `grad_clip_norm`, `scheduler`, `warmup_epochs` |
| Runtime | `device`, `seed`, `use_amp`, `val_every_n_epochs`, `deterministic` |
| Output | `output_dir`, `checkpoint_name`, `save_last` |
| Nested | `early_stopping.*`, `semi_supervised.*` |

`scheduler` accepts `none`, `cosine`, `step` or `plateau`. `device` accepts
`auto` (CUDA → MPS → CPU), `cpu`, `cuda` or `mps`; an unavailable backend warns
and falls back rather than crashing.

Configs are validated before any data is loaded, and unknown keys are an error,
so a typo is reported instead of silently ignored.

## Self-training

Optional pseudo-labelling on the unlabelled pool:

```yaml
semi_supervised:
  enabled: true
  activation_threshold: 0.7    # wait until val accuracy reaches this
  confidence_threshold: 0.95   # keep predictions at least this confident
  refresh_interval: 5          # regenerate every N epochs
  max_ratio: 2.0               # cap pseudo-labels at 2x the labelled set
```

Only confident predictions are kept, and when the cap binds the
highest-confidence samples win. Reported epoch metrics are weighted by sample
count across the labelled and pseudo-labelled phases.

## Benchmarks

Measured on this codebase. Both runs are fully supervised — the 6,786-image
unlabelled pool is deliberately unused, so these are clean supervised baselines.

| Model | Params | Val accuracy | Macro F1 | Best epoch | Train time |
|---|---:|---:|---:|---:|---:|
| `resnet18` | 11.2 M | **88.64%** | 0.8851 | 26 / 30 | 6.0 min |
| `efficientnet_b0_cbam` | 4.2 M | **93.64%** | 0.9358 | 30 / 30 | 10.7 min |

CBAM on EfficientNet-B0 beats the ResNet18 baseline by **5.0 points with 2.6x
fewer parameters**, which is the result the attention module is there to
produce.

<details>
<summary>Provenance</summary>

| | |
|---|---|
| Commit | `d4ba3bc` |
| Dataset | Food-11, ML2021 HW3 split — Kaggle `zhaopang/ml2021springhw3` v1 |
| Train / val | 3,080 labelled (280 per class) / 660 (60 per class) |
| Hardware | Apple M5 Pro, 18 cores, 48 GB, MPS backend |
| Software | Python 3.12.14, torch 2.14.0, torchvision 0.29.0 |
| Configs | `configs/food11_bench_resnet18.yaml`, `configs/food11_bench_effnet_cbam.yaml` |
| Seed | 0 (`deterministic: true`) |

Accuracy is top-1 on the validation split, read from each run's
`metrics.json`. Both checkpoints were then re-scored through
`food-recognition-eval` — a different code path from training — and reproduced
the same figures.

`testing/` is not used: its 3,347 images sit in a single directory with no
labels, so validation is the only labelled held-out split.

</details>

<details>
<summary>Why 30 epochs, and what the longer runs showed</summary>

Both models were also run for 50 epochs. Neither improved:

| Run | Epochs run | Val accuracy |
|---|---:|---:|
| `resnet18`, 30 ep | 30 | **88.64%** |
| `resnet18`, 50 ep, no early stop | 50 | 87.73% |
| `efficientnet_b0_cbam`, 30 ep | 30 | **93.64%** |
| `efficientnet_b0_cbam`, 50 ep, no early stop | 50 | 93.48% |
| `efficientnet_b0_cbam`, 50 ep, early stop on | 14 | 91.36% |

The last row is worth noting: with `epochs: 50` and `patience: 8`, the run
stopped at epoch 14 and scored **2.3 points worse**. Cosine annealing spends
its middle epochs on a high learning-rate plateau, which early stopping reads
as "no improvement" and cuts before the anneal delivers its gain. If you
lengthen the schedule, raise `patience` with it.

</details>

### Reproducing

```bash
# Download (~918 MB) and link into place
pip install -e ".[download]"
python scripts/download_dataset.py
mkdir -p data && ln -s <printed-path>/food-11 data/food-11

food-recognition-train --config configs/food11_bench_effnet_cbam.yaml
cat runs/bench_effnet_cbam/metrics.json
```

Expect different numbers on different hardware: MPS, CUDA and CPU kernels do not
produce bit-identical results, and cuDNN autotuning varies between GPUs. The
seed makes a run repeatable on the *same* machine, not across machines.

### The two figures that used to be here

Earlier revisions claimed 94.56% on Food-11 and 84.09% on Food-101. Neither
could be traced to any script, log or checkpoint in this repository, and the
architecture named for the Food-101 figure (EfficientNet-B4 + CBAM) was never
implemented — the original experiments used EfficientNet-**B0**. They were
removed rather than carried forward unverified, and the table above replaces
them with numbers that ship with the config, commit and hardware needed to
check them.

Food-101 remains unmeasured. `configs/food101_efficientnet_cbam.yaml` is ready;
at 75,750 training images it is a much longer run than Food-11.

## Project layout

```
src/food_recognition/     # the package
├── config.py             # TrainingConfig + YAML loading and validation
├── data.py               # datasets, transforms, dataloaders, pseudo-labelling
├── models.py             # CBAM, model factory, 29 architectures
├── training.py           # Trainer: loop, validation, early stopping, scheduling
├── metrics.py            # accuracy, per-class P/R/F1, confusion matrix
├── predict.py            # Predictor and checkpoint loading
├── gradcam.py            # Grad-CAM attribution and heatmap overlays
├── utils.py              # seeding, devices, early stopping, checkpoint I/O
└── cli.py                # train / eval / predict / gradcam entry points

configs/                  # YAML configs
scripts/                  # sample data, Food-11 download, Food-101 conversion
tests/                    # 176 tests
docs/                     # thesis notes, reference PDF
experiments/              # object detection example
├── legacy/               # original single-file experiment scripts
legacy/model_utils/       # original helper modules
```

`experiments/legacy/` and `legacy/` preserve the original research scripts
verbatim for reference. They are not imported by the package, are excluded from
linting, and still contain hard-coded `cuda:0` device assignments.

## Development

```bash
pip install -e ".[dev]"

pytest -q                                    # 171 tests
pytest -q --cov=food_recognition             # with coverage
ruff check src tests scripts                 # lint
```

Tests run the real training loop on generated data rather than mocks, and cover
the regressions that motivated this refactor: numeric class-directory ordering,
flat unlabelled directories, config path resolution, and metric correctness
against hand-computed values.

CI runs the suite on Python 3.10 / 3.11 / 3.12, plus an end-to-end smoke test
and a distribution build.

## Citation

The `docs/` directory holds the original thesis notes and reference PDF that
this project accompanied.

## License

MIT — see [LICENSE](LICENSE).
