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
measured SD of 1.052 pts at n=660 also exceeds every gap in the Food-11 grid,
whose largest pairwise difference is 0.985 pts.

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
  a Holm correction across six pairs, the smallest (0.0019) would need to clear
  0.0083 — it does, so the one positive result survives correction; no other pair
  comes close.
- **Epoch selection:** best-of-30, applied uniformly to all 18 runs (§4.4).

Source of truth: `docs/benchmarks/food11_seed_significance.json` — `b0_380` and
`b3_380` at 6 seeds, `b4_380` and `b0_300` at 3 (§6.1), all 30-epoch runs on the
660-image split. `docs/benchmarks/food11_seed_variance.json` is the frozen 3-seed
record kept for provenance.

```bash
python scripts/aggregate_seeds.py ../seeds \
  --grid docs/benchmarks/food11_ablation_grid.json \
  --significance --resolution 660 --seeds 0,1,2,3,4,5 \
  --json-out docs/benchmarks/food11_seed_significance.json
```

### 4.2 Per-cell spread

`b0_380` and `b3_380` were extended to **6 seeds** after the first pass, because
they carry the only pair that showed a real effect and n=3 could not corroborate
it (§4.3). The other two cells remain at 3.

| Cell | n | Mean | SD | Spread | Per-seed |
| --- | --- | --- | --- | --- | --- |
| `b3_380` | **6** | 95.71% | 0.21 | 0.61 | 95.45 / 95.61 / 95.61 / 95.76 / 96.06 / 95.76 |
| `b4_380` | 3 | 95.45% | 0.40 | 0.76 | 95.00 / 95.61 / 95.76 |
| `b0_300` | 3 | 95.00% | 0.40 | 0.76 | 95.15 / 95.30 / 94.55 |
| `b0_380` | **6** | 94.72% | 0.22 | 0.61 | 94.85 / 94.70 / 95.00 / 94.85 / 94.39 / 94.55 |

Largest spread 0.76 pts = 5.0 images out of 660. Note both 6-seed cells widened
their spread to 0.61 pts as seeds were added: 3 seeds understates run-to-run
variability, which is a reason to distrust tight-looking SDs at n=3, not to
prefer them.

### 4.3 Pairwise results

Pairs are formed only on seeds both cells share, so the extended pair is tested
at n=6 and the rest at n=3. That asymmetry is stated per row rather than hidden.

| Pair | n | diff (pts) | Bootstrap 95% CI | p (t) | p (exact) | dz | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `b3_380` vs `b0_380` | **6** | +0.985 | [+0.707, +1.288] | **0.0019** | **0.0312** | +2.44 | **significant** |
| `b4_380` vs `b0_380` | 3 | +0.606 | [+0.152, +0.909] | 0.1201 | 0.2500 | +1.51 | not significant |
| `b3_380` vs `b0_300` | 3 | +0.556 | [+0.303, +1.061] | 0.1588 | 0.2500 | +1.27 | not significant |
| `b4_380` vs `b0_300` | 3 | +0.455 | [-0.152, +1.212] | 0.3745 | 0.5000 | +0.65 | not significant |
| `b0_300` vs `b0_380` | 3 | +0.152 | [-0.455, +0.606] | 0.6784 | 0.7500 | +0.28 | not significant |
| `b3_380` vs `b4_380` | 3 | +0.101 | [-0.152, +0.455] | 0.6349 | 1.0000 | +0.32 | not significant |

**The conclusion is not uniform, and must not be reported as if it were.**

- **One pair separates, and now does so on every test.** `b3_380` over `b0_380`,
  **+0.985 pts** at n=6, p(t)=0.0019, **exact p=0.0312**, dz=+2.44, bootstrap CI
  [+0.707, +1.288] excluding zero. This is the one ranking claim in the grid that
  the data supports.
- **Five pairs remain undecidable**, including the two nominally best cells
  (`b3_380` vs `b4_380`, 0.101 pts = 0.7 images), whose CI comfortably spans zero.
  The nominal top-two ordering is not supported.

**What the extra three seeds changed.** At n=3 this pair was
`significant_parametric_only`: the t-test rejected at p=0.0198 but the exact
sign-flip test could not, because its two-sided p-value floor is `2/2^n` = 0.25 at
n=3 — unreachable at any effect size. `seeds_needed(0.05) = 6` predicted the fix,
and running the six seeds confirmed it: the floor drops to 0.03125 and the exact
test now clears alpha at exactly that value. The verdict is therefore plain
`significant` with `power_limited: false`. Note the honest reading of "exactly the
floor": with 6 seeds the exact test rejects only because **every** per-seed
difference has the same sign, which is the strongest pattern 6 paired runs can
show and also the only one that reaches alpha. It is significant, not
comfortably so.

Also note dz fell from +4.04 (n=3) to +2.44 (n=6). The n=3 figure was inflated by
the small sample, exactly as expected; the n=6 estimate is the more trustworthy
one, and still a large effect.

**Two caveats remain on this result:**

1. **Multiplicity.** p=0.0019 is uncorrected. Across 6 pairs, Holm at alpha=0.05
   tests the smallest p against 0.05/6 = 0.0083; 0.0019 passes that, so this
   particular result survives correction. The exact p=0.0312 would not.
2. **Epoch selection weakens but no longer overturns it** (§4.4).

### 4.4 Epoch-selection sensitivity — a genuine confound

Every accuracy above is the **best** validation accuracy over the 30-epoch
schedule. Verified uniform: for all 18 runs, `metrics.json["accuracy"]` equals
`max(history.json val_acc)` exactly, and every run completed the full 30 epochs.
So the rule is applied consistently and the pairing is internally valid.

But "best of 30" is a maximum over 30 correlated draws, so it is biased upward by
an amount that depends on each run's curve noise. Measured with
`scripts/check_epoch_criterion.py`:

- Best epoch ranges from **12 to 30** across runs.
- Best exceeds last-epoch accuracy by **+0.463 pts on average**, max **+1.364**.

**That average selection bonus is larger than four of the six pairwise gaps being
compared.** Recomputing everything under the last-epoch rule:

| Pair | best-epoch | last-epoch | Verdict change |
| --- | --- | --- | --- |
| `b3_380` vs `b0_380` | +0.985 | +0.808 | significant -> **significant (parametric only)**, p 0.0019 -> 0.0406 |
| `b4_380` vs `b0_380` | +0.606 | +1.061 | not significant -> **significant (parametric)** |
| `b3_380` vs `b0_300` | +0.556 | +0.657 | unchanged |
| `b4_380` vs `b0_300` | +0.455 | +1.061 | unchanged |
| `b0_300` vs `b0_380` | +0.152 | +0.000 | **sign lost** |
| `b3_380` vs `b4_380` | +0.101 | **-0.404** | **sign flips** |

Four of six pairs change verdict or sign. The top-two pair `b3_380` vs `b4_380`
still **reverses**: b3 leads by 0.101 pts under best-epoch and trails by 0.404
under last-epoch, so that ordering is an artifact of the epoch rule.

**What survived and what did not.** At n=3 the single positive result was
destroyed by this check. At n=6 it is not: the difference stays positive
(+0.808 pts) and the t-test still rejects (p=0.0406), though it drops to
parametric-only because the exact test at that magnitude no longer clears alpha.
So `b3_380` > `b0_380` is now the one claim that holds under **both** epoch rules,
which is why it is the only one stated as a finding.

**Conclusion.** With a 660-image validation split and best-of-30 selection, this
grid supports exactly one ranking claim — `b3_380` over `b0_380` — and only after
doubling that pair's seed count. Every other ordering, including the nominal
top-two, is indistinguishable from noise or depends on an arbitrary analysis
choice. Extending two cells to 6 seeds was enough to resolve one pair out of six;
resolving the rest would need the same treatment applied more widely (§6).

---

## 5. Verified: implementation and tests

`src/food_recognition/significance.py`, wired into `scripts/aggregate_seeds.py`
via `--significance`. Pure numpy — the Student-t tail uses a Lentz continued
fraction for the regularized incomplete beta, so **SciPy is not a runtime
dependency**. Where SciPy is installed the test suite cross-checks against it and
agrees to ~1e-13 (`scipy.stats.ttest_1samp`, `scipy.stats.t.sf`,
`scipy.stats.binomtest`).

Test count: **217 -> 423** (+206). `ruff check src tests scripts app.py` clean
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

### 6.1 Six seeds for the remaining two Food-11 cells — not done

`b0_380` and `b3_380` **were** extended to 6 seeds (completed; see §6.5), which
is what lifted that pair's exact-test floor from 0.25 to 0.031 and produced the
one assumption-free verdict in §4.3. `b4_380` and `b0_300` were **not**: they
remain at 3 seeds, so all five pairs involving them are still subject to the 0.25
floor and cannot be decided regardless of effect size.

Extending those two costs `3 x (51.5 + 77.3)` = **6.4 h** serial. This is the
single cheapest remaining action that would change a conclusion: it would make
`b3_380` vs `b4_380` — the nominal top-two comparison, currently undecidable —
testable for the first time.

### 6.2 A larger labelled Food-11 validation split

The most valuable missing experiment. §3.4 shows n=660 gives 4.5% power and a
38.8% sign-flip rate, so no amount of seed averaging fixes a split this small.
`testing/` (3,347 images) is unlabelled and cannot be used. Options, none run:

- Obtain the canonical Food-11 release and re-split it to recover the canonical
  3,430-image validation set. Note this cannot be done from the local cache
  alone: the 6,786 `training/unlabeled` images sit in a single `00/` directory
  with **no class labels**, so only 3,080 of the local 9,866 are usable as
  supervised data. Canonical labels would have to be fetched first. Retraining
  every cell on the changed training set is a full 9-cell grid at 3 seeds,
  **~19 h**, on top of that download.
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

### 6.5 The n=6 runs — completed and folded in

Seeds 3, 4 and 5 for `b0_380` and `b3_380` ran serially from 03:08 to 08:16 local
time, **308 min (5.1 h)** wall clock against a 229 min estimate — the estimate was
low by 34%, mostly because `b3_380` per-run cost ran above the 51.5 min figure
taken from the earlier single run.

All six completed with `rc=0`, all six ran the full 30 epochs, and all six satisfy
`metrics.json["accuracy"] == max(history val_acc)`, so they were folded into §4
under the same selection rule as the original three. Per-run accuracies:

| Run | Accuracy | Best epoch | Last epoch |
| --- | --- | --- | --- |
| `b0_380` seed 3 | 94.85% | 20 | 94.70% |
| `b0_380` seed 4 | 94.39% | 30 | 94.39% |
| `b0_380` seed 5 | 94.55% | 29 | 93.94% |
| `b3_380` seed 3 | 95.76% | 25 | 95.15% |
| `b3_380` seed 4 | 96.06% | 26 | 95.91% |
| `b3_380` seed 5 | 95.76% | 12 | 94.85% |

**This is the one part of §4 that moved from "compute-limited" to verified**, and
it changed the headline: `b3_380` vs `b0_380` went from
`significant_parametric_only` (n=3) to `significant` (n=6). Everything else in §6
remains not done.

---

## 7. Reproducing

```bash
# Seed-level analysis (Question B)
python scripts/aggregate_seeds.py ../seeds \
  --grid docs/benchmarks/food11_ablation_grid.json \
  --significance --resolution 660 --seeds 0,1,2,3,4,5 \
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
  --power-report docs/benchmarks/food101_validation_power.json \
  --output-dir docs/benchmarks
```

`--seeds 0,1,2,3,4,5` pins the report to an explicit seed set rather than taking
whatever `<cell>_s<seed>/metrics.json` directories happen to exist. This matters:
regenerating mid-sweep once produced a report with one cell at n=4 and the rest at
n=3, which contradicted the table it was cited for. Paired comparisons are
unaffected either way — cells are only ever compared on seeds they share, which is
why `b4_380` and `b0_300` correctly stay at n=3 here even though the list names six
seeds — but the per-cell block is only reproducible when the set is pinned. The
list is recorded in the output as `_significance.protocol.seeds_included`.

To extend a cell, train `<cell>_s<seed>` under the same config with only `--seed`
changed, add it to `docs/benchmarks/seed_runs_manifest.json`, widen the `--seeds`
list, and re-run the first two commands plus the figures.

When the §6.5 runs finish, re-run the first two commands with the seed list
widened (e.g. `--seeds 0,1,2,3,4,5`) rather than dropped, so the report stays
explicit about what went into it.
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
