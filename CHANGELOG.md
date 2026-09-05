# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-09-05

### Added

- **Measured Food-11 benchmarks**, replacing the "not yet measured" placeholder.
  `efficientnet_b0_cbam` reaches **93.64%** validation accuracy against
  `resnet18`'s **88.64%** — 5.0 points better with 2.6x fewer parameters, which
  is what the attention module exists to deliver. Recorded with commit, dataset
  version, hardware, software versions, seed and config, per the rule in
  CONTRIBUTING.md.
- `configs/food11_bench_resnet18.yaml` and `configs/food11_bench_effnet_cbam.yaml`
  reproduce those numbers.
- `docs/benchmarks/*.json` — the raw `metrics.json` and `history.json` from both
  runs, so every figure in the README can be checked against its source.

### Fixed

- **Checkpoints trained with `dropout > 0` could not be reloaded.** `dropout>0`
  wraps the classifier head in `Sequential(Dropout, Linear)`, which shifts the
  state_dict keys from `fc.weight` to `fc.1.weight`. `load_predictor` rebuilt the
  model with the default `dropout=0.0`, so loading failed with
  `Missing key(s) in state_dict: "fc.weight"`. Every shipped config sets
  `dropout > 0`, so this broke `food-recognition-eval`, `-predict` and
  `-gradcam` on the documented happy path. The value is now read back from the
  checkpoint's embedded config.

  Found by running `food-recognition-eval` against a real trained checkpoint —
  the existing round-trip test used the default `dropout=0`, so it never
  exercised the broken path. Now covered for `dropout` in {0.0, 0.2, 0.5}, plus
  a test asserting a reloaded model reproduces its training accuracy rather than
  merely loading without error.

### Changed

- Test suite grew from 172 to 176 tests.

## [0.3.0] - 2026-09-05

### Added

- **Grad-CAM attribution** (`food_recognition.gradcam`) with a
  `food-recognition-gradcam` CLI. Shows which pixels drove a prediction, which
  is how you catch a model that is right for the wrong reason.
  - Target layer is resolved automatically per architecture (`layer4` for
    ResNet, `features` for EfficientNet/VGG, the CBAM block for CBAM models)
    and can be overridden.
  - `--class-index` explains a class you name instead of the predicted one,
    which is the useful view when a model is confidently wrong.
  - `--side-by-side` writes the original and overlay together.
  - Hooks are removed in a `finally` block and train/eval mode is restored, so
    a failed call cannot leave the model altered.
  - No OpenCV or matplotlib dependency: resizing uses
    `torch.nn.functional.interpolate` and the colormap is computed in NumPy.
  - Verified against ground truth, not just for absence of crashes: on a task
    whose class signal is a coloured blob at a random position, the heatmap peak
    landed inside the true blob in 12/12 validation images, with ~4.9x more
    heat inside the blob than outside.
- **Food-101 support** via `scripts/prepare_food101.py`, which converts the
  official layout (`images/<class>/<hash>.jpg` + `meta/train.txt` /
  `meta/test.txt`) into the layout this project expects.
  - Uses relative symlinks by default, so conversion takes seconds and adds
    almost no disk usage; `--copy` produces a standalone tree.
  - Split membership comes from the official meta files rather than a
    reshuffle, so results stay comparable with published numbers.
  - `--limit-per-class` for fast experiments; warns when the source is
    incomplete instead of failing silently.
- `configs/food101_efficientnet_cbam.yaml` for 101-class training.

### Fixed

- `food-recognition-gradcam` on a directory wrote every overlay into one flat
  folder. Because dataset class directories reuse file names (`00/000.jpg`,
  `01/000.jpg`, ...), 12 input images silently produced only 4 output files.
  Output now mirrors the input's directory structure.
- Grad-CAM emitted a PyTorch `UserWarning` about the full backward hook on
  every call, because no input required grad. Users could not act on it.

### Changed

- Test suite grew from 118 to 172 tests; coverage held at 92%.

## [0.2.0] - 2026-09-05

Complete rebuild of the project into an installable, tested package. The `main`
branch previously contained no working code — the README described 20 files that
did not exist on any branch, and the real scripts sat unmerged on two side
branches.

### Added

- Installable `food_recognition` package with `src/` layout and three console
  scripts (`train`, `eval`, `predict`).
- 118-test suite at 92% coverage, running in CI on Python 3.10/3.11/3.12.
- `TrainingConfig` with YAML loading, unknown-key rejection and `validate()`
  that fails fast before any expensive work starts.
- `metrics.py`: per-class precision/recall/F1 and confusion matrix in pure
  torch, removing the scikit-learn dependency.
- Reproducible smoke test (`configs/smoke_test.yaml` +
  `scripts/make_sample_data.py`) that runs the full loop on CPU in seconds.

### Fixed

- Semi-supervised training could not run at all: the unlabelled pool is a flat
  directory, but the code loaded it with `ImageFolder`, which raises
  `FileNotFoundError: Couldn't find any class folder`.
- The default config was broken on a clean checkout: relative paths in YAML were
  resolved against the config file's directory rather than the working
  directory, so `python main.py` failed immediately.
- CBAM hard-coded 1280 channels, silently locking the model to EfficientNet-B0
  (B4 has 1792). Channel count is now inferred from the backbone.
- `SimpleConvNet` hard-coded a `512*7*7` classifier input and crashed on any
  input size other than 224. Now uses adaptive pooling.
- Pseudo-label epochs averaged labelled and unlabelled loss with equal weight
  regardless of sample count, distorting reported metrics. Now weighted by
  sample count.
- Non-zero-padded numeric class directories mapped to the wrong indices
  (`'10'` sorted to index 2). Numeric directory names are now sorted
  numerically.
- Replaced deprecated `pretrained=` (2 `UserWarning`s on torchvision 0.29) and
  `torch.cuda.amp.GradScaler` (`FutureWarning` on torch 2.14).

### Removed

- Unverified benchmark claims of 94.56% (Food-11) and 84.09% (Food-101). Neither
  could be traced to any script, log or checkpoint in the repository, and the
  architecture named for the Food-101 figure (EfficientNet-B4 + CBAM) was never
  implemented — the original experiments used EfficientNet-B0. See
  [Benchmarks](README.md#benchmarks).
