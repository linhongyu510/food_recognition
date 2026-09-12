# Statistical significance and validation-set power

**Status of this document.** Everything in the "Verified" sections was run on
this machine for this document and every number traces to a `metrics.json`, a
report JSON under `docs/benchmarks/`, or a command reproduced below. The
"Not completed" section lists what was scoped but not run, with measured cost
estimates rather than guesses. Nothing in the not-completed section is described
anywhere in this document as if it had been verified.

- **Commit:** `effaee2` (branch `feat/statistical-significance`), parent `2ee5f37`
- **Hardware:** Apple M5 Pro, 18 cores, 48 GB, macOS 26.5.1 arm64. MPS backend;
  CUDA unavailable
- **Environment:** Python 3.12.14, torch 2.14.0, torchvision 0.29.0, numpy 2.5.2.
  SciPy 1.18.1 is installed but is **only** used to cross-check the tests; the
  shipped code needs numpy alone
- **Date:** 2026-09-13

---

## 1. The two questions, kept apart

Two different questions get conflated whenever a benchmark table is read as a
ranking. They have different noise sources, different pairing units, and
different answers, so this document answers them separately and never uses one
to support a claim about the other.

| | Question A | Question B |
| --- | --- | --- |
| **Asks** | Given these two *fixed trained checkpoints*, is the validation set large enough to tell them apart? | Would *retraining* config A and config B put A ahead of B again? |
| **Noise source** | Which images are in the validation set | Random seed: init, data order, augmentation |
| **Pairing unit** | One validation image | One seed |
| **Test** | McNemar exact + image-level bootstrap | Paired t-test + exact sign-flip test + seed-level bootstrap |
| **Tool** | `scripts/compare_runs.py` | `scripts/aggregate_seeds.py --significance` |
| **Answered in** | §3 | §4 |

A significant result on Question A does **not** imply one on Question B. A
validation set can separate two particular checkpoints decisively while the
config-level ranking they imply remains unreproducible across seeds. That is
exactly the situation this project is in, and §3 and §4 show both halves.

### What this document does not claim

- **No cross-study comparison.** The Food-11 mirror used here is not the
  canonical Food-11 (see §2), so no number here is compared against any
  published Food-11 result. The two are not on the same footing.
- **No claim about published attention gains.** Whether the gains reported in
  the attention literature fall inside seed noise is a question about *those*
  papers and requires reproducing them. It is out of scope here, and the
  negative results in §4 are about this project's own configurations only.
- **Not a new finding.** That seed variance can reorder a leaderboard is
  established — see arXiv:1912.12522 and arXiv:1902.08142 on NAS, and
  *Deep Reinforcement Learning that Matters* (arXiv:1709.06560). This document
  measures the effect for this project; it does not claim to discover it.

---

## 2. Datasets, and one important caveat

### Food-11 (used for the seed analysis in §4)

The cached copy used by this project is **not** the canonical Food-11
distribution. It is the National Taiwan University ML2021 HW3 semi-supervised
re-split:

| Split | Images |
| --- | --- |
| `training/labeled` | 3,080 |
| `training/unlabeled` | 6,786 |
| `validation` | **660** |
| `testing` | 3,347, **unlabelled** |

`3,080 + 6,786 = 9,866`, which is exactly the canonical Food-11 training-set
size — confirming this is a relabelled re-split of it. This project trains on
the 3,080 labelled images only, i.e. **31.2%** of the canonical training set,
and validates on 660 images where the canonical split provides 3,430.

**Consequence:** no accuracy in this document may be compared with a published
Food-11 figure. The training sets differ by 3.2x and the validation splits are
different sets of images. Any such comparison would be measuring the data, not
the method.

The 660-image validation split is also the direct cause of the resolution
problem in §3: `testing/` is unlabelled, so it cannot be used to enlarge it.

### Food-101 (used for the power analysis in §3)

`validation/` holds 101 classes x 250 = **25,250** labelled images, all of which
are used. Food-101 images come from Foodspotting under terms permitting
scientific fair use only; no image is redistributed by this project.

---

## 3. Verified: validation-set resolution and power

### 3.1 Resolution accounting

`accuracy_resolution()` in `src/food_recognition/significance.py`. One
misclassified image moves accuracy by `100/n` percentage points:

| Validation set | n | 1 image | Single-run 95% Wilson CI width at 95% acc |
| --- | --- | --- | --- |
| Food-11 (this project) | 660 | **0.1515 pts** | 3.36 pts |
| Food-101 `validation/` | 25,250 | **0.0040 pts** | 0.54 pts |
| **Ratio** | 38.26x | **38.26x finer** | 6.24x narrower |

At 660 images, accuracy is quantised in steps of 0.1515 pts. A claimed gap of
0.10 pts is **smaller than the smallest change the set can represent** — it
cannot be the difference between two counts of correct images on the same set,
so it can only arise from comparing different splits or different runs. At
25,250 images the same 0.10 pts is 25 images and is representable.

Representable is not the same as readable. At 25,250 the single-run CI is still
0.54 pts wide, so a 0.10 pt gap remains well inside the noise band of either
measurement taken alone. This is why `AccuracyResolution.resolves()` requires a
gap to clear *both* one image and the CI width, and why it returns `False` for
0.10 pts at **both** sizes.

**Verdict on the original question.** Enlarging 660 -> 25,250 makes the
0.10–0.15 pt gaps *representable* (they stop being sub-quantum) but does **not**
make them *readable*: they stay inside the single-run confidence interval. The
38x gain in granularity buys a great deal, but not enough to adjudicate gaps
this small — see §3.3 for how much it does buy.

### 3.2 Re-evaluation on all 25,250 images

Both published Food-101 checkpoints were re-scored on the complete validation
set. This is a checkpoint-reuse evaluation; no retraining was involved.

```bash
food-recognition-eval --checkpoint runs/bench_food101/checkpoints/best.pt \
  --data-dir data/food-101/validation --batch-size 64 --num-workers 8 --device mps \
  --json-out reval/b0_224_full_val.json
```

| Config | Accuracy | Correct / 25,250 | 95% Wilson CI | metrics.json |
| --- | --- | --- | --- | --- |
| `efficientnet_b0_cbam` @224 | 88.701% | 22,397 | [88.305, 89.086] | `docs/benchmarks/food101_b0_cbam_224_fullval_metrics.json` |
| `efficientnet_b4_cbam` @224 | 89.113% | 22,501 | [88.723, 89.491] | `docs/benchmarks/food101_b4_cbam_224_fullval_metrics.json` |

Configs: `f101/configs/bench_food101.yaml` and `f101/configs/b4_food101_224.yaml`
(30 epochs, cosine, no early stopping, seed 0, fp32). Both reproduced their
stored accuracy **bit-exactly**, which confirms the evaluation path is
deterministic and that the previously recorded Food-101 numbers were already
computed on the full 25,250 images rather than a subset.

### 3.3 Question A: are these two checkpoints separable?

`scripts/compare_runs.py`, output in `docs/benchmarks/food101_paired_eval.json`.
Pairing unit: one image. Bootstrap: 10,000 percentile resamples over images,
both models scored on the same resampled images. alpha = 0.05, two-sided.

| Quantity | Value |
| --- | --- |
| Accuracy difference (b4 - b0) | **+0.412 pts** = +104 images |
| Discordant images | 1,910 (only b4 right: 1,007; only b0 right: 903) |
| McNemar exact p | **0.0184** |
| Image-level bootstrap 95% CI | [+0.075, +0.749] pts |
| Verdict | **separable** |

So on 25,250 images these two checkpoints *are* distinguishable. Note how narrow
the margin is: 1,910 images disagree, and the entire result rests on a 104-image
imbalance within them.

### 3.4 What the larger validation set actually buys

`scripts/validation_power.py` subsamples the real 25,250-image set without
replacement, treating the full-set difference (+0.412 pts) as ground truth, and
re-runs McNemar on each subset. 400 trials per size.
Output: `docs/benchmarks/food101_validation_power.json`.

| n | 1 image | Power at alpha=0.05 | Median p | SD of measured diff | **Sign-flip rate** |
| --- | --- | --- | --- | --- | --- |
| 660 | 0.1515 pts | **4.5%** | 0.597 | 1.052 pts | **38.8%** |
| 1,320 | 0.0758 pts | 6.8% | 0.486 | 0.746 pts | 28.7% |
| 2,000 | 0.0500 pts | 8.2% | 0.467 | 0.587 pts | 25.8% |
| 5,000 | 0.0200 pts | 14.2% | 0.325 | 0.356 pts | 14.5% |
| 12,000 | 0.0083 pts | 34.5% | 0.102 | 0.185 pts | 0.8% |
| 25,250 | 0.0040 pts | **100%** | 0.018 | 0.000 pts | 0.0% |

This is the strongest single result in this document. At 660 images — the size
this project's ablation grid was ranked on — a difference that genuinely exists
would have been:

- detected **4.5%** of the time (i.e. essentially never), and
- measured with the **wrong sign 38.8%** of the time.

A ranking produced at n=660 on a gap of this size is close to a coin flip. The
measured SD of 1.052 pts at n=660 also dwarfs every gap in the Food-11 grid,
whose largest pairwise difference is 0.707 pts.

**Caveat.** This isolates image-sampling noise for two fixed checkpoints. It
says nothing about seed noise, which §4 measures and which is additive to this.

---

## 4. Verified: seed-level significance on Food-11

### 4.1 Protocol

Recorded machine-readably in the `protocol` block of every report JSON.

- **Pairing unit:** one seed. Cells are compared only on seeds present in both;
  difference `i` is `acc_A(seed_i) - acc_B(seed_i)`. Same seed means same
  initialisation, data order and augmentation stream, so the difference isolates
  the configuration change.
- **Bootstrap resampling unit:** one seed-level paired difference, drawn with
  replacement. **Not** images — the quantity being bounded is training noise.
- **Bootstrap:** percentile, **10,000** resamples, RNG seed 20260913.
- **alpha:** 0.05, **two-sided**.
- **Effect size:** Cohen's *dz* = mean difference / SD of differences.
- **Multiplicity:** **none**. All six p-values are per-pair and uncorrected. With
  a Holm correction across six pairs, the smallest (0.0198) would need to clear
  0.0083 and would not survive.
- **Epoch selection:** best-of-30, applied uniformly to all 12 runs (§4.4).

Source of truth: `docs/benchmarks/food11_seed_variance.json`, 4 cells x 3 seeds,
all 30-epoch runs on the 660-image split.

```bash
python scripts/aggregate_seeds.py ../seeds \
  --grid docs/benchmarks/food11_ablation_grid.json \
  --significance --resolution 660 \
  --json-out docs/benchmarks/food11_seed_significance.json
```

### 4.2 Per-cell spread

| Cell | n | Mean | SD | Spread | Per-seed |
| --- | --- | --- | --- | --- | --- |
| `b3_380` | 3 | 95.56% | 0.09 | 0.15 | 95.45 / 95.61 / 95.61 |
| `b4_380` | 3 | 95.45% | 0.40 | 0.76 | 95.00 / 95.61 / 95.76 |
| `b0_300` | 3 | 95.00% | 0.40 | 0.76 | 95.15 / 95.30 / 94.55 |
| `b0_380` | 3 | 94.85% | 0.15 | 0.30 | 94.85 / 94.70 / 95.00 |

Largest spread 0.76 pts = 5.0 images out of 660.

### 4.3 Pairwise results

| Pair | diff (pts) | Bootstrap 95% CI | p (t) | p (exact) | dz | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `b3_380` vs `b0_380` | +0.707 | [+0.606, +0.909] | **0.0198** | 0.250 | +4.04 | **significant (parametric only)** |
| `b4_380` vs `b0_380` | +0.606 | [+0.152, +0.909] | 0.1201 | 0.250 | +1.51 | not significant |
| `b3_380` vs `b0_300` | +0.556 | [+0.303, +1.061] | 0.1588 | 0.250 | +1.27 | not significant |
| `b4_380` vs `b0_300` | +0.455 | [-0.152, +1.212] | 0.3745 | 0.500 | +0.65 | not significant |
| `b0_300` vs `b0_380` | +0.152 | [-0.455, +0.606] | 0.6784 | 0.750 | +0.28 | not significant |
| `b3_380` vs `b4_380` | +0.101 | [-0.152, +0.455] | 0.6349 | 1.000 | +0.32 | not significant |

**The conclusion is not uniform, and must not be reported as if it were.**

- **One pair separates:** `b3_380` vs `b0_380`, +0.707 pts, p=0.0198, dz=+4.04,
  bootstrap CI excluding zero. The three per-seed differences are +0.606, +0.909
  and +0.606 — same sign, similar magnitude, every seed agreeing. This is the
  strongest evidence in the dataset and it should not be flattened into "all
  differences fall inside the noise".
- **Five pairs are not decidable**, including the two closest cells
  (`b3_380` vs `b4_380`, 0.101 pts = 0.7 images), whose CI comfortably spans
  zero. The nominal top-two ranking is not supported.

**Three caveats on the one positive result**, all of which matter:

1. **The exact test cannot corroborate it.** The two-sided sign-flip test on 3
   paired seeds has a p-value floor of `2/2^3 = 0.25`. It cannot return anything
   below 0.25 no matter how large the effect, so it neither confirms nor refutes.
   `seeds_needed(0.05) = 6` is the smallest design that could. This is why the
   verdict is labelled `significant_parametric_only` with `power_limited: true`
   rather than plain "significant".
2. **The t-test's assumption is unverifiable at n=3.** p=0.0198 rests on
   normality of the differences, which three points cannot support. dz=+4.04 is
   implausibly large as a population estimate and is inflated by the tiny sample.
3. **It does not survive the epoch-selection check** (§4.4) or multiplicity
   correction (§4.1). Treat it as provisional.

### 4.4 Epoch-selection sensitivity — a genuine confound

Every accuracy above is the **best** validation accuracy over the 30-epoch
schedule. Verified uniform: for all 12 runs, `metrics.json["accuracy"]` equals
`max(history.json val_acc)` exactly, and all 12 ran the full 30 epochs. So the
rule is applied consistently and the pairing is internally valid.

But "best of 30" is a maximum over 30 correlated draws, so it is biased upward by
an amount that depends on each run's curve noise. Measured with
`scripts/check_epoch_criterion.py`:

- Best epoch ranges from **13 to 30** across runs.
- Best exceeds last-epoch accuracy by **+0.492 pts on average**, max **+1.364**.

**That average selection bonus is larger than four of the six pairwise gaps being
compared.** Recomputing everything under the last-epoch rule:

| Pair | best-epoch | last-epoch | Verdict change |
| --- | --- | --- | --- |
| `b3_380` vs `b0_380` | +0.707 | +0.657 | **significant (parametric) -> not significant** |
| `b4_380` vs `b0_380` | +0.606 | +1.061 | not significant -> **significant (parametric)** |
| `b3_380` vs `b0_300` | +0.556 | +0.657 | unchanged |
| `b4_380` vs `b0_300` | +0.455 | +1.061 | unchanged |
| `b0_300` vs `b0_380` | +0.152 | +0.000 | **sign lost** |
| `b3_380` vs `b4_380` | +0.101 | **-0.404** | **sign flips** |

Four of six pairs change verdict or sign. In particular the top-two pair
`b3_380` vs `b4_380` **reverses**: b3 leads by 0.101 pts under best-epoch and
trails by 0.404 pts under last-epoch. The single "significant" result of §4.3
does not survive the switch.

**Conclusion.** With 3 seeds, a 660-image validation split and best-of-30 epoch
selection, this grid cannot support any ranking claim. The one pair that clears
alpha under one epoch rule fails under the other, so nothing here is robust to a
defensible change in an arbitrary analysis choice.

---

## 5. Verified: implementation and tests

`src/food_recognition/significance.py`, wired into `scripts/aggregate_seeds.py`
via `--significance`. Pure numpy — the Student-t tail uses a Lentz continued
fraction for the regularized incomplete beta, so **SciPy is not a runtime
dependency**. Where SciPy is installed the test suite cross-checks against it and
agrees to ~1e-13 (`scipy.stats.ttest_1samp`, `scipy.stats.t.sf`,
`scipy.stats.binomtest`).

Test count: **217 -> 390** (+173). `ruff check src tests scripts app.py` clean
with no `--select` narrowing; full `pytest -q` green.

Three defects were found by these tests and fixed, each with a regression test
naming the original failure mode:

1. **Spurious significance from float residue.** `0.92-0.90`, `0.93-0.91` and
   `0.94-0.92` are all "0.02" but differ around 1e-17. An exact `sd == 0` guard
   missed that, producing t of order 1e14 and p ~ 0 — a "highly significant"
   verdict from three identical differences. Now uses a 1e-12 tolerance.
2. **McNemar p-value off by up to 0.078.** Summing the binomial PMF under a
   tolerance dropped one of each pair of mirror-image terms. Replaced with the
   exact symmetric tail `2 x P(K >= max(b01, b10))`, capped at 1.
3. **Wrong normal quantile.** The Wilson interval took *z* from the t quantile at
   df=1e7, which is wrong in the 7th decimal. Now derived from `math.erf`.

Cross-validation against an independently hand-computed reference: all six
pairwise `diff`, `sd` and `t` values reproduce to within 0.003.

---

## 6. Not completed — blocked by compute

Scoped but **not run**. No number below is reported anywhere else in this
document, and no conclusion in §3–§5 depends on any of them.

Measured costs on this machine (from `seeds/seeds_master.log` and
`f101/*.log`), used for the estimates:

| Run | Measured cost |
| --- | --- |
| Food-11 `b0_cbam` @224, 30 ep | 10.7 min |
| Food-11 `b0_cbam` @300/@380, 30 ep | ~18 / ~25 min |
| Food-11 `b3_cbam` @380, 30 ep | 51.5 min |
| Food-11 `b4_cbam` @380, 30 ep | 77.3 min |
| Food-101 `b0_cbam` @224, 30 ep | 254 min (4.2 h) |
| Food-101 `b4_cbam` @224, 30 ep | 670.6 min (11.2 h) |

Two MPS trainings run concurrently slow each other by ~2x, so all estimates
assume serial execution.

### 6.1 Six seeds per cell for all four Food-11 cells — partially started

Needed to lift the exact test's floor from 0.25 to 0.031 and make an
assumption-free verdict possible. Seeds 3–5 for `b0_380` and `b3_380` (the pair
carrying the one positive result) were **launched** for this document; see §6.5
for the status at the time of writing. Extending all four cells to 6 seeds costs
`3 x (18 + 25 + 51.5 + 77.3)` = **8.6 h** on top.

### 6.2 A larger labelled Food-11 validation split

The most valuable missing experiment. §3.4 shows n=660 gives 4.5% power and a
38.8% sign-flip rate, so no amount of seed averaging fixes a split this small.
`testing/` (3,347 images) is unlabelled and cannot be used. Options, none run:

- Re-split the 9,866 canonical training images to recover the canonical
  3,430-image validation split. Retraining every cell on the changed training
  set: `2 x (10.7 + 18 + 25 + 51.5 + 77.3 + ...)` — full 9-cell grid at 3 seeds
  is **~19 h**.
- Label a subset of `testing/` by hand. Not attempted; no ground truth available.

### 6.3 Multi-seed Food-101 runs

Food-101 has the validation set that actually resolves differences (§3.4) but
only seed 0 exists per config. Three seeds for both configs:
`2 x (254 + 670.6) x 3 / 60` = **92 h (3.9 days)**. Not started.

### 6.4 Last-epoch-only re-analysis with matched checkpoints

§4.4 recomputes last-epoch accuracy from `history.json`, which is sound for the
accuracy figure. Producing per-image correctness vectors under the last-epoch
rule would additionally require last-epoch *checkpoints*; only `best.pt` and
`last.pt` are retained per run, and `last.pt` exists, so this is feasible but was
not run. Estimated cost: evaluation only, ~1.5 min per Food-11 run, ~18 min for
all 12. Not done.

### 6.5 Status of the launched n=6 runs

Seeds 3–5 for `b0_380` and `b3_380` were started at 03:08 local time, serial,
estimated `3 x (25 + 51.5)` = **229 min (3.8 h)**. At the time this document was
finalised these runs had **not** all completed. See §7 for how to fold them in
when they do. Until then, **§4 stands at n=3 and every verdict in it carries the
n=3 power limitation.**

---

## 7. Reproducing, and folding in the pending seeds

```bash
# Seed-level analysis (Question B)
python scripts/aggregate_seeds.py ../seeds \
  --grid docs/benchmarks/food11_ablation_grid.json \
  --significance --resolution 660 \
  --json-out docs/benchmarks/food11_seed_significance.json

# Epoch-selection sensitivity
python scripts/check_epoch_criterion.py \
  --manifest docs/benchmarks/seed_runs_manifest.json \
  --json-out docs/benchmarks/food11_epoch_criterion.json

# Image-level analysis (Question A)
python scripts/dump_predictions.py --checkpoint <ckpt> \
  --data-dir data/food-101/validation --json-out reval/<name>_correct.json
python scripts/compare_runs.py --run b0_cbam_224=reval/b0_correct.json \
  --run b4_cbam_224=reval/b4_correct.json \
  --json-out docs/benchmarks/food101_paired_eval.json

# What a bigger validation set buys
python scripts/validation_power.py --run-a reval/b4_correct.json \
  --run-b reval/b0_correct.json --sizes 660 1320 2000 5000 12000 25250 \
  --json-out docs/benchmarks/food101_validation_power.json

# Figures
python scripts/plot_significance.py \
  --report docs/benchmarks/food11_seed_significance.json \
  --output-dir docs/benchmarks
```

When the §6.5 runs finish, re-running the first two commands picks the new seeds
up automatically (`aggregate_seeds.py` discovers `<cell>_s<seed>/metrics.json`).
The `b0_380` vs `b3_380` pair will then have 6 paired seeds and its exact-test
floor drops to 0.031, at which point `significant_parametric_only` can become
either `significant` or `not_significant`. Add the new run directories to
`docs/benchmarks/seed_runs_manifest.json` so the epoch check covers them too.

## 8. Figures

- `docs/benchmarks/food11_seed_distribution.png` — every seed's accuracy per
  cell with the ±1 SD band and a one-image scale bar.
- `docs/benchmarks/food11_paired_ci.png` — paired differences with bootstrap CIs,
  coloured by verdict; the zero line shows which pairs cannot be ordered.
- `docs/benchmarks/food11_power.png` — the exact test's p-value floor against
  seed count, marking where this study sits (n=3, floor 0.25) and where alpha
  becomes reachable (n=6).
- `docs/benchmarks/food11_validation_power.png` — the measured sign-flip rate and
  measurement granularity against validation subset size (§3.4), i.e. what
  enlarging the labelled set actually buys. Written by passing
  `--power-report docs/benchmarks/food101_validation_power.json` to
  `scripts/plot_significance.py`.
