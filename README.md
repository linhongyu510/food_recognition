# Food Recognition

Food image classification with PyTorch: a configurable training pipeline with
CBAM attention, self-training on unlabelled data, and CLI tools for training,
evaluation and inference.

[![CI](https://github.com/linhongyu510/food_recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/linhongyu510/food_recognition/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.4-ee4c2c)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status.** The pipeline, CLI and 188-test suite are verified and run in CI on
> every push. Food-11 and Food-101 accuracy are both measured and recorded with
> full provenance in [Benchmarks](#benchmarks).

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
(~4.7 GB), then convert it:

```bash
python scripts/prepare_food101.py --source /path/to/food-101 --output data/food-101
```

> The canonical archive is `http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz`,
> but it served ~0.25 MB/s when this was measured — over 5 hours for one file.
> The [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) mirror on
> Hugging Face carries the same 75,750 / 25,250 official split and downloaded in
> about two minutes. It ships Parquet rather than JPEG trees, so use the helper
> below to rebuild the official layout first:
>
> ```bash
> pip install -e ".[food101]"
> python scripts/food101_from_parquet.py --output ~/data/food-101
> python scripts/prepare_food101.py --source ~/data/food-101 --output data/food-101
> ```
>
> The official split survives the round trip because the mirror keeps each
> original filename in the Parquet `image.path` field, and class ordering is read
> from the shard's schema metadata rather than guessed. Verified against a direct
> rebuild: identical `meta/` files and byte-identical images.

This creates symlinks by default, so it finishes in seconds and adds almost no
disk usage. The links are relative, so moving the source and output together
keeps them valid. Use `--copy` for real files (needs another ~5 GB) when the
output has to stand on its own, or `--limit-per-class 50` for a fast smoke run.

The split assignment comes from the official meta files rather than a reshuffle,
so results stay comparable with published numbers.

```bash
food-recognition-train --config configs/food101_efficientnet_cbam.yaml
```

Food-101 is 101 classes and 75,750 training images, so it is far heavier than
Food-11: **8.5 min per epoch** on the Apple M5 Pro used for
[Benchmarks](#benchmarks), i.e. about 4.2 hours for the full 30-epoch schedule.

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

### Input resolution

`image_size` defaults to 224, but the bigger EfficientNets were trained at higher
resolutions: B3 at 300px and B4 at 380px. Leaving the default in place runs B4 on
35% of the pixels it was designed for, and on Food-11 that costs **0.76 points**
(94.24% at 224px versus 95.00% at 380px — see [Benchmarks](#benchmarks)).

Training warns when `image_size` falls well below the backbone's native
resolution:

```
WARNING | image_size=224 is well below the native 380px for efficientnet_b4_cbam;
          expect to lose accuracy that the larger backbone would otherwise
          provide (set image_size=380 to use it fully)
```

Running below native resolution is a legitimate way to fit a compute budget —
380px costs 2.7x the time per epoch here — so the warning does not override the
setting. It exists because the accuracy loss is otherwise invisible.

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

Measured on this codebase. Every run is fully supervised — on Food-11 the
6,786-image unlabelled pool is deliberately unused — so these are clean
supervised baselines.

### Food-11

| Model | Input | Params | Val accuracy | Macro F1 | Best epoch | Train time |
|---|---:|---:|---:|---:|---:|---:|
| `resnet18` | 224px | 11.2 M | **88.64%** | 0.8851 | 26 / 30 | 6.0 min |
| `efficientnet_b0_cbam` | 224px | 4.2 M | **93.64%** | 0.9358 | 30 / 30 | 10.7 min |
| `efficientnet_b4_cbam` | 224px | 18.0 M | **94.24%** | 0.9418 | 21 / 30 | 29.0 min |
| `efficientnet_b4_cbam` | 380px | 18.0 M | **95.00%** | 0.9497 | 13 / 30 | 77.3 min |

CBAM on EfficientNet-B0 beats the ResNet18 baseline by **5.0 points with 2.6x
fewer parameters**, which is the result the attention module is there to
produce.

The two B4 rows exist to separate two variables that are easy to conflate.
Going B0 → B4 at a fixed 224px buys **+0.61 points for 4.3x the parameters and
2.7x the time**; raising that same B4 to its native 380px buys a further
**+0.76 points** and takes the total to 7.2x the B0 wall clock. Most of what B4
has to offer on this dataset comes from the resolution it was designed for, not
from the extra parameters alone — and B0 + CBAM remains the better accuracy-per-
minute choice unless the last point matters more than the time.

Grad-CAM from the 380px B4 model on a validation noodle plate, predicted at
0.9338 — heat on the pasta and its garnish, with the plate rim cold:

![Grad-CAM from EfficientNet-B4 + CBAM on Food-11 noodles](docs/benchmarks/gradcam_food11_b4_noodles.png)

### Food-101

| Model | Params | Val accuracy | Macro F1 | Best epoch | Train time |
|---|---:|---:|---:|---:|---:|
| `efficientnet_b0_cbam` | 4.3 M | **88.70%** | 0.8865 | 29 / 30 | 254 min |

101 classes over the official 75,750 / 25,250 split. Per-class F1 spans
`edamame` at 1.000 down to `steak` at 0.588 — the confusable meat and dessert
classes are where the errors concentrate, and the full per-class table is in
[`docs/benchmarks/`](docs/benchmarks/).

Grad-CAM from this model on a validation pizza, predicted at 0.9651 — heat on
the crust and pepperoni, not the box:

![Grad-CAM on Food-101 pizza](docs/benchmarks/gradcam_food101_pizza.png)

<details>
<summary>Provenance</summary>

| | Food-11 | Food-101 |
|---|---|---|
| Commit | `d4ba3bc` (resnet18, b0) / `8feb7a9` (b4) | `ac37bb4` |
| Dataset | ML2021 HW3 split — Kaggle `zhaopang/ml2021springhw3` v1 | Official split via [`ethz/food101`](https://huggingface.co/datasets/ethz/food101) |
| Train / val | 3,080 labelled (280/class) / 660 (60/class) | 75,750 (750/class) / 25,250 (250/class) |
| Config | `configs/food11_bench_resnet18.yaml`, `configs/food11_bench_effnet_cbam.yaml`, `configs/food11_bench_effnet_b4_cbam.yaml`, `configs/food11_bench_effnet_b4_cbam_224.yaml` | `configs/food101_bench_effnet_cbam.yaml` |
| Epoch cost | 11.9 s / 21.5 s / 58 s / 155 s | 8.5 min |

Common to all runs: Apple M5 Pro (18 cores, 48 GB) on the MPS backend, Python
3.12.14, torch 2.14.0, torchvision 0.29.0, seed 0 with `deterministic: true`,
and `use_amp: false` because autocast is unreliable on MPS.

Accuracy is top-1 on the validation split, read from each run's `metrics.json`.
Every checkpoint was then re-scored through `food-recognition-eval` — a
different code path from training — and reproduced the same figures; Food-101
matched to six decimal places across all 25,250 images.

Two caveats stated rather than buried. The Food-11 runs use the **3,080-image
HW3 split**, not the full 9,866-image Food-11, so they are not directly
comparable to papers using the latter; and Food-11's `testing/` directory is
unused because its 3,347 images sit in one directory with no labels.

</details>

<details>
<summary>Why 30 epochs, and what the longer runs showed</summary>

Both Food-11 models were also run for 50 epochs. Neither improved:

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

Food-101 shows the other half of that story, which is why early stopping is off
in its config. Validation accuracy sat at 86.9% around epoch 13, then kept
grinding upward through the anneal to peak at **epoch 29 of 30** — the default
`patience: 6` would have cut it somewhere in the middle and cost roughly a
point and a half.

</details>

### Reproducing

```bash
# Food-11 (~918 MB)
pip install -e ".[download]"
python scripts/download_dataset.py
mkdir -p data && ln -s <printed-path>/food-11 data/food-11
food-recognition-train --config configs/food11_bench_effnet_cbam.yaml

# Food-101 (~4.8 GB; see the Food-101 section above for why the mirror)
pip install -e ".[food101]"
python scripts/food101_from_parquet.py --output ~/data/food-101
python scripts/prepare_food101.py --source ~/data/food-101 --output data/food-101
food-recognition-train --config configs/food101_bench_effnet_cbam.yaml
```

Expect different numbers on different hardware: MPS, CUDA and CPU kernels do not
produce bit-identical results, and cuDNN autotuning varies between GPUs. The
seed makes a run repeatable on the *same* machine, not across machines.

### The two figures that used to be here

Earlier revisions claimed 94.56% on Food-11 and 84.09% on Food-101. Neither
could be traced to any script, log or checkpoint in this repository, and at the
time the architecture named for the Food-101 figure (EfficientNet-B4 + CBAM) had
never been trained here — the original experiments used EfficientNet-**B0**.
They were removed rather than carried forward unverified, and the tables above
replace them with numbers that ship with the config, commit and hardware needed
to check them.

The B4 gap has since been closed on Food-11: `efficientnet_b4_cbam` is now
benchmarked at both 224px and its native 380px, reaching **95.00%** — which is
above the old 94.56% claim, but on the 3,080-image HW3 split rather than the full
9,866-image Food-11, so it still is not the same measurement.

Neither old figure is contradicted by the new ones, because neither is directly
comparable. On Food-101 the old text named B4 while the measured 88.70% comes
from B0 + CBAM; on Food-11 the split differs as described above. Both old numbers
stay unverified rather than being retro-fitted to whichever new result sits
closest.

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
tests/                    # 188 tests
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
