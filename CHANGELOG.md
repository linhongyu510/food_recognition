# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-09-06

### Added

- **A 3×3 resolution × architecture ablation on Food-11** — `efficientnet_b0`,
  `b3` and `b4` with CBAM, each at 224px, 300px and 380px, nine runs on an
  identical 30-epoch schedule differing only in `model_name` and `image_size`.
  Six new cells join the three already measured. All nine were re-scored through
  `food-recognition-eval` and reproduced their figures to six decimal places
  across all 660 validation images.
- `configs/food11_abl_effnet_b{0,3,4}_cbam_{224,300,380}.yaml` reproduce the new
  cells; `docs/benchmarks/food11_ablation_grid.json` holds the assembled grid and
  `scripts/plot_ablation.py` renders it into `docs/benchmarks/food11_ablation.png`
  from the grid file rather than transcribed numbers.

### Changed

- **Corrected the interpretation the earlier B4 rows had suggested.** Two of the
  grid's three findings contradict what 0.5.0 concluded from four rows alone:
  - **Resolution beats parameter count.** B0 gains +1.52 points from 224px →
    300px, while B0 → B4 at a fixed 224px buys +0.61 for 4.3x the parameters. The
    smallest model at 300px (95.15%, 19.0 min) *beats* the largest at its native
    380px (95.00%, 77.3 min) — +0.15 points for a quarter of the time.
  - **"Native resolution is optimal" is false here.** Only B4 peaks where it was
    pretrained. B0 is pretrained at 224px but peaks at 300px (+1.52 over its own
    native); B3 is pretrained at 300px but peaks at 380px (+0.30). The 0.5.0
    warning is still correctly aimed at *far* below native, but native is a floor,
    not a target — silence from the warning does not mean the resolution is
    optimal, and the README now says so.
  - **The best cell is the middle model.** `efficientnet_b3_cbam` at 380px
    reaches **95.45%**, the highest of the nine, in two thirds of B4@380's time.
    B4 is never the best choice at any resolution in this grid.
- The note on the two removed claims records that the architecture the Food-101
  figure named is not the one worth recommending on Food-11: it places fifth of
  nine, behind two configurations that are cheaper *and* more accurate.
- Stated caveats rather than burying them: one seed per cell on a 660-image split
  where a single image is 0.15 points, so B0@300 and B4@380 are tied rather than
  ranked; and the 3,080-image training set is small enough that the larger
  backbones are plausibly data-limited, the likeliest reason B4 never pulls ahead.
- Test suite grew from 201 to 205 tests.

## [0.6.0] - 2026-09-06

### Added

- **A Gradio demo app** (`app.py`) wrapping the same `Predictor` and `GradCAM`
  the CLI uses, rather than a parallel inference path: upload a photo, get top-k
  predictions and the heatmap beside them. It loads one checkpoint at startup and
  reads the architecture, resolution and class names from the file, so the same
  command works for any run. Runs on CPU — measured on 2 threads, which is what
  a free Hugging Face *CPU Basic* Space gets: 64 ms/image for
  `efficientnet_b0_cbam` at 224px, 180 ms for B4 at 224px, 221 ms for B4 at
  380px, with Grad-CAM roughly doubling each. Point it at a local checkpoint
  (`--checkpoint`) or one on the Hub (`--hf-repo`); as a Space, it reads
  `FR_CHECKPOINT` / `FR_HF_REPO` from the environment instead.
- **`scripts/publish_to_hf.py`** uploads a run's checkpoint to the Hugging Face
  Hub with a model card generated from that run's own `metrics.json` and
  embedded config — accuracy, macro F1, checkpoint epoch, wall clock, hardest
  and easiest class, full training configuration, and a limitations section — so
  a published number cannot drift from what was measured. `--dry-run` renders the
  card without contacting the Hub.
- The card defaults a 101-class model to `license: other` and appends Food-101's
  terms, since that dataset permits scientific fair use only and the images are
  not ETH Zurich's property; weights trained on it are a derivative work. Food-11
  models default to MIT. A permissive default is not allowed to imply more
  freedom than the data grants.
- New `app` extra (`pip install -e ".[app]"`) pulling in gradio and
  huggingface_hub, and now installed in CI so the new tests actually run there.

### Changed

- README documents the demo app and the Hub publishing path, including the
  measured CPU latencies and the fact that Gradio Spaces now require a paid plan
  (with a two-Space ZeroGPU exception on free personal accounts).
- `app.py` ships in the sdist, alongside the configs and scripts already there.
- Test suite grew from 190 to 201 tests.

### Fixed

- The Development section still advertised 171 tests, four counts out of date.

## [0.5.0] - 2026-09-06

### Fixed

- **`min_delta` no longer gates checkpoint saving**, only early-stopping patience.
  The two shared one threshold, so an epoch that improved by less than `min_delta`
  was not written to `best.pt`. Caught on the Food-101 B4 run: epoch 27 scored
  0.891366 against epoch 24's saved 0.891129 — a gain of 0.000238 under a
  `min_delta` of 0.0005 — so the better weights were discarded and `metrics.json`
  disagreed with the `history.json` beside it. Best-checkpoint tracking is now
  driven by any strict improvement, while `min_delta` still controls patience;
  both halves have regression tests. The five previously published figures were
  each re-checked against their history files and none were affected.

### Added

- **Measured EfficientNet-B4 + CBAM benchmarks**, the architecture a removed
  performance claim had named but which had never been trained in this repository.
  On Food-11, two runs so the architecture and resolution changes can be told
  apart: **94.24%** at 224px and **95.00%** at its native 380px, against 93.64%
  for `efficientnet_b0_cbam`. On Food-101 at matched 224px: **89.11%** against
  88.70% for B0. Going B0 → B4 at fixed resolution is worth +0.61 points on
  Food-11 and +0.41 on Food-101, for roughly 4.2x the parameters and 2.6-2.7x the
  time; the move to 380px on Food-11 adds a further +0.76. Every checkpoint was
  re-scored through `food-recognition-eval` and reproduced its figures to six
  decimal places — across all 660 Food-11 and all 25,250 Food-101 images.
- `configs/food11_bench_effnet_b4_cbam.yaml` (380px, native),
  `configs/food11_bench_effnet_b4_cbam_224.yaml` (224px control) and
  `configs/food101_bench_effnet_b4_cbam.yaml` reproduce them.
- A warning when `image_size` sits well below the backbone's native resolution.
  `image_size` defaults to 224 while EfficientNet-B3 and B4 were trained at 300px
  and 380px, so the default silently ran B4 on 35% of its intended pixels — worth
  0.76 points on Food-11 — with nothing in the output to say so. The trainer had
  been discarding the native size that `initialize_model` already returned. The
  setting is not overridden, since running below native resolution is a valid way
  to fit a compute budget: B4 at 380px on Food-101 was measured at 42.1 ms/img,
  which puts a 30-epoch run near 27 hours against the 11.2 hours 224px took.
- `docs/benchmarks/` gains `metrics.json` and `history.json` for all three B4
  runs, plus a Grad-CAM overlay from the 380px Food-11 model.

### Changed

- README documents the resolution trade-off under Models, records the
  checkpointing bug and its blast radius, and updates the note on the two removed
  figures: the B4 gap is now closed on both datasets, and the Food-101
  measurement lands 5.0 points *above* the 84.09% once claimed for that
  architecture. Both old numbers stay marked unverified rather than being
  retro-fitted to the nearest new result.
- Test suite grew from 185 to 190 tests.

## [0.4.0] - 2026-09-05

### Added

- **Measured Food-101 benchmark**, closing the last "not yet measured" gap.
  `efficientnet_b0_cbam` reaches **88.70%** top-1 (macro F1 0.8865) over the
  official 75,750 / 25,250 split, 101 classes, in 254 min on an Apple M5 Pro.
  Recorded with commit, dataset source, hardware, software versions, seed and
  config, per the rule in CONTRIBUTING.md, and re-scored through
  `food-recognition-eval` — a different code path from training — which matched
  to six decimal places across all 25,250 images.
- `scripts/food101_from_parquet.py` rebuilds the official `images/` + `meta/`
  layout from the `ethz/food101` Hugging Face mirror. The canonical ETH archive
  served 0.25 MB/s when measured — 5.6 hours for one 4.7 GB file — against about
  two minutes for the mirror. The official split survives the round trip because
  the mirror preserves each original filename in the Parquet `image.path` field;
  class ordering is read from the shard's schema metadata rather than assumed.
- `configs/food101_bench_effnet_cbam.yaml` reproduces the Food-101 numbers.
- `pyarrow` as an optional `[food101]` extra.
- `docs/benchmarks/` gains the Food-101 `metrics.json` (including the full
  per-class table) and `history.json`, plus a Grad-CAM overlay from the trained
  model.

### Changed

- Food-101 per-epoch cost in the README is now the measured 8.5 min rather than
  the guessed "20-40 minutes on a single mid-range GPU".
- Early stopping is off in the Food-101 benchmark config. Validation accuracy
  plateaued near 86.9% around epoch 13 and then climbed through the cosine
  anneal to peak at epoch 29 of 30; the default `patience: 6` would have cut it
  mid-plateau and cost roughly 1.5 points. This is the same trap already
  documented for Food-11, seen from the other side.
- Test suite grew from 176 to 185 tests.

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
