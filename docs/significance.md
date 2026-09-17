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
- **Multiplicity:** **none**. All six p-values are per-pair and uncorrected. Under
  Holm at alpha=0.05 the two smallest (0.0019 and 0.0060) clear their thresholds
  and the other two positive results (0.0148, 0.0277) do not; §4.3 caveat 2 gives
  the detail. Reported uncorrected because the six pairs answer six separate
  questions, with the correction stated rather than silently applied.
- **Epoch selection:** best-of-30, applied uniformly to all 24 runs (§4.4).

Source of truth: `docs/benchmarks/food11_seed_significance.json` — all four cells
at 6 seeds, all 30-epoch runs on the 660-image split.
`docs/benchmarks/food11_seed_variance.json` is the frozen 3-seed record kept for
provenance; where the two disagree, the 6-seed report is current.

```bash
python scripts/aggregate_seeds.py ../seeds \
  --grid docs/benchmarks/food11_ablation_grid.json \
  --significance --resolution 660 --seeds 0,1,2,3,4,5 \
  --json-out docs/benchmarks/food11_seed_significance.json
```

### 4.2 Per-cell spread

All four cells are now at **6 seeds**. Three seeds cannot decide anything: the
two-sided exact sign-flip test's p-value floor is `2/2^n`, which is 0.25 at n=3
and unreachable at any effect size.

| Cell | n | Mean | SD | Spread | Per-seed |
| --- | --- | --- | --- | --- | --- |
| `b4_380` | 6 | 95.76% | 0.42 | 1.06 | 95.00 / 95.61 / 95.76 / 96.06 / 96.06 / 96.06 |
| `b3_380` | 6 | 95.71% | 0.21 | 0.61 | 95.45 / 95.61 / 95.61 / 95.76 / 96.06 / 95.76 |
| `b0_300` | 6 | 94.82% | 0.43 | 1.06 | 95.15 / 95.30 / 94.55 / 95.15 / 94.24 / 94.55 |
| `b0_380` | 6 | 94.72% | 0.22 | 0.61 | 94.85 / 94.70 / 95.00 / 94.85 / 94.39 / 94.55 |

Largest spread 1.06 pts = 7.0 images out of 660. **Every cell's spread grew when
seeds 3-5 were added** (b4_380 0.76 -> 1.06, b0_300 0.76 -> 1.06, and the two
extended earlier went 0.15/0.30 -> 0.61). Three seeds systematically understates
run-to-run variability, so a tight SD at n=3 is not evidence of stability.

Note also that the two 380px cells swapped places: at n=3 `b3_380` led with
95.56% against `b4_380`'s 95.45%; at n=6 `b4_380` leads with 95.76% against
95.71%. §4.3 shows that reordering is not significant either way.

### 4.3 Pairwise results

| Pair | n | diff (pts) | Bootstrap 95% CI | p (t) | p (exact) | dz | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `b4_380` vs `b0_380` | 6 | +1.035 | [+0.631, +1.414] | **0.0060** | **0.0312** | +1.87 | **significant** |
| `b3_380` vs `b0_380` | 6 | +0.985 | [+0.707, +1.288] | **0.0019** | **0.0312** | +2.44 | **significant** |
| `b4_380` vs `b0_300` | 6 | +0.934 | [+0.379, +1.465] | **0.0277** | 0.0625 | +1.25 | significant, parametric only |
| `b3_380` vs `b0_300` | 6 | +0.884 | [+0.480, +1.338] | **0.0148** | **0.0312** | +1.49 | **significant** |
| `b0_300` vs `b0_380` | 6 | +0.101 | [-0.177, +0.379] | 0.5430 | 0.6250 | +0.27 | not significant |
| `b4_380` vs `b3_380` | 6 | +0.051 | [-0.177, +0.227] | 0.6793 | 0.8750 | +0.18 | not significant |

**What n=6 resolved.** Both 380px cells beat `b0_380`, and `b3_380` beats
`b0_300`, on all three tests. A fourth pair (`b4_380` vs `b0_300`) rejects
parametrically but not exactly. So resolution matters: the two 380px cells are
genuinely ahead of the two B0 cells.

**What n=6 did not resolve, and this is the more important half.** The two
comparisons *within* each resolution tier remain undecidable:

- **`b4_380` vs `b3_380`: +0.051 pts, p=0.6793, CI [-0.177, +0.227].** This was
  the nominal top-two comparison and the whole reason for the extension. It is
  not merely unresolved — the sign **reversed** relative to n=3, where `b3_380`
  led by +0.101. Six seeds each, and the two cells are still indistinguishable;
  the CI is now tight enough (±0.2 pts) to say the true gap is *small*, not to
  say which is larger.
- **`b0_300` vs `b0_380`: +0.101 pts, p=0.5430**, CI spanning zero.

So doubling the seed count bought four verdicts and confirmed that the top-two
question has no answer at this scale. The three-way tie at the top of the grid is
now a **two-way** tie between `b4_380` and `b3_380`, established rather than
merely suspected.

**Three caveats on the positive results:**

1. **Three of the four rejections sit exactly on the exact test's floor**
   (p=0.0312 = `2/2^6`). That is the smallest p-value 6 paired seeds can produce,
   and it occurs precisely when every per-seed difference shares a sign. Such a
   result *is* a rejection, but a maximally fragile one: a single seed
   disagreeing would push p to 0.219 and the verdict to not-significant. It is
   flagged in the report as `permutation.at_floor` so the caveat travels with the
   number rather than depending on prose.
2. **Multiplicity.** p-values are uncorrected. Holm across six pairs tests the
   smallest against 0.05/6 = 0.0083: `b3_380` vs `b0_380` (0.0019) and `b4_380`
   vs `b0_380` (0.0060) survive; `b3_380` vs `b0_300` (0.0148) and `b4_380` vs
   `b0_300` (0.0277) do not. So two of the four positive results are robust to
   correction and two are not.
3. **Epoch selection changes three of the four verdicts** (§4.4), though it no
   longer reverses any sign among them.

### 4.4 Epoch-selection sensitivity — a genuine confound

Every accuracy above is the **best** validation accuracy over the 30-epoch
schedule. Verified uniform: for all 24 runs, `metrics.json["accuracy"]` equals
`max(history.json val_acc)` exactly, and every run completed the full 30 epochs.
So the rule is applied consistently and the pairing is internally valid.

But "best of 30" is a maximum over 30 correlated draws, so it is biased upward by
an amount that depends on each run's curve noise. Measured with
`scripts/check_epoch_criterion.py`:

- Best epoch ranges from **12 to 30** across runs.
- Best exceeds last-epoch accuracy by **+0.492 pts on average**, max **+1.364**.

That average selection bonus still exceeds the two unresolved gaps entirely.
Recomputing everything under the last-epoch rule:

| Pair | best-epoch | last-epoch | Verdict change |
| --- | --- | --- | --- |
| `b4_380` vs `b0_380` | +1.035 | +1.136 | unchanged (significant both ways) |
| `b3_380` vs `b0_380` | +0.985 | +0.808 | significant -> **parametric only** |
| `b4_380` vs `b0_300` | +0.934 | +1.313 | parametric only -> **significant** |
| `b3_380` vs `b0_300` | +0.884 | +0.985 | significant -> **parametric only** |
| `b0_300` vs `b0_380` | +0.101 | **-0.177** | **sign flips** (not significant either way) |
| `b4_380` vs `b3_380` | +0.051 | +0.328 | unchanged (not significant either way) |

**This is materially better than at n=3, and worth stating precisely.** With
three seeds, four of six pairs changed verdict *or sign*, and the single positive
result was destroyed by the switch. With six:

- **No sign flips among the four resolved pairs.** The only sign flip left is
  `b0_300` vs `b0_380`, which is not significant under either rule — so the flip
  changes nothing.
- **One pair is stable in both verdict and sign**: `b4_380` vs `b0_380`,
  significant either way. That is the single most robust claim in the grid.
- **Three pairs still move between `significant` and `parametric only`.** The
  direction is preserved in every case; what changes is whether the exact test
  clears alpha. Since three of them sit on the floor (caveat 1), a shift of a
  single seed's sign is enough to move them, so this instability is expected
  rather than surprising.
- **The top-two pair is not significant under either rule**, so no epoch choice
  rescues it.

**Conclusion.** Six seeds per cell, a 660-image validation split and best-of-30
selection support this much: **both 380px cells are genuinely better than
`b0_380`**, with `b4_380` vs `b0_380` robust to the epoch rule and to Holm
correction. They do **not** support any ordering *between* `b4_380` and `b3_380`,
or between `b0_300` and `b0_380` — and the top-two sign reversed between n=3 and
n=6, which is direct evidence that the earlier ranking was noise. Resolution is
the real effect here; backbone size within a resolution tier is not resolvable at
this validation-set size (§3.4 explains why 660 images is the binding limit).

---

## 5. Verified: implementation and tests

`src/food_recognition/significance.py`, wired into `scripts/aggregate_seeds.py`
via `--significance`. Pure numpy — the Student-t tail uses a Lentz continued
fraction for the regularized incomplete beta, so **SciPy is not a runtime
dependency**. Where SciPy is installed the test suite cross-checks against it and
agrees to ~1e-13 (`scipy.stats.ttest_1samp`, `scipy.stats.t.sf`,
`scipy.stats.binomtest`).

Test count: **217 -> 436** (+219). `ruff check src tests scripts app.py` clean
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
| Food-11 `b0_cbam` @300/@380, 30 ep | 17.7 / ~25 min |
| Food-11 `b3_cbam` @380, 30 ep | 51.5 min (86.5 min measured over seeds 3-5) |
| Food-11 `b4_cbam` @380, 30 ep | 72.4 min (mean of seeds 3-5) |
| Food-101 `b0_cbam` @224, 30 ep | 254 min (4.2 h) |
| Food-101 `b4_cbam` @224, 30 ep | 670.6 min (11.2 h) |

Two MPS trainings run concurrently slow each other by ~2x, so all estimates
assume serial execution.

### 6.1 Six seeds per cell — done

All four Food-11 cells are at 6 seeds; see §6.5 for the two extension sweeps and
their measured costs. Nothing in §4 is now seed-limited at n=6, which is what
made four of the six pairs decidable and confirmed the other two are not.

Going further would mean **8 seeds** (`seeds_needed(0.01) = 8`), which would drop
the exact floor from 0.031 to 0.0078 and let the three floor-hugging rejections in
§4.3 clear alpha with room to spare instead of exactly at the limit. Cost for two
more seeds on all four cells: `2 x (17.7 + 25 + 51.5 + 72.4)` = **5.6 h**. Not
done. It would firm up existing verdicts rather than change any, so it ranks below
§6.2.

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

### 6.5 The two n=6 extension sweeps — completed and folded in

**Sweep 1** — seeds 3-5 for `b0_380` and `b3_380`, the pair that at n=3 rejected
parametrically but could not be corroborated by the exact test. Ran serially
03:08 -> 08:16, **308 min (5.1 h)** against a 229 min estimate: **34% over**,
mostly because `b3_380` averaged 86.5 min/run rather than the 51.5 min taken from
its earlier single run.

**Sweep 2** — seeds 3-5 for `b4_380` and `b0_300`, the two cells still at n=3,
which left all pairs involving them undecidable by construction. Ran serially
02:07 -> 06:38, **270.7 min (4.5 h)** against a 255 min estimate: **6% over**.
Per-run: `b0_300` 18.3 / 17.8 / 17.7 min, `b4_380` 73.3 / 72.3 / 71.4 min.

> **Correction.** An earlier version of §6.1 priced sweep 2 at `3 x (51.5 + 77.3)`
> = 6.4 h. The 51.5 figure is `b3_380`'s per-run cost, not `b0_300`'s (~18 min) —
> the wrong cell was substituted. Re-deriving from the seed 1/2 logs of the cells
> actually being extended gave 255 min, which the 270.7 min actual confirms. The
> original 6.4 h estimate was 42% too high.

All twelve runs across both sweeps completed with `rc=0`, ran the full 30 epochs,
and satisfy `metrics.json["accuracy"] == max(history val_acc)`, so they were folded
into §4 under the same selection rule as seeds 0-2. Sweep 2 per-run accuracies:

| Run | Accuracy | Best epoch | Last epoch |
| --- | --- | --- | --- |
| `b0_300` seed 3 | 95.15% | 22 | 94.09% |
| `b0_300` seed 4 | 94.24% | 17 | 93.64% |
| `b0_300` seed 5 | 94.55% | 29 | 94.24% |
| `b4_380` seed 3 | 96.06% | 12 | 95.15% |
| `b4_380` seed 4 | 96.06% | 26 | 95.91% |
| `b4_380` seed 5 | 96.06% | 26 | 95.61% |

**What the two sweeps bought, stated plainly.** Sweep 1 turned one
`significant_parametric_only` verdict into `significant`. Sweep 2 was aimed at the
nominal top-two comparison `b4_380` vs `b3_380` and **did not resolve it**: the
gap is +0.051 pts with p=0.6793 and a CI of [-0.177, +0.227], and its sign
reversed relative to n=3. That is a negative result for the stated goal, and it is
the more informative outcome — it establishes that the top-two ordering is not
determinable at this validation-set size rather than merely unproven. Sweep 2 did
resolve three other pairs as a side effect (§4.3).

Total compute across both sweeps: **578.7 min (9.6 h)** for twelve 30-epoch runs.


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

Keep the seed list explicit rather than dropping the flag when adding seeds: a
report generated without it takes whatever run directories exist, which once
produced a table whose cells had unequal n.

## 8. Figures

- `docs/benchmarks/food11_seed_distribution.png` — every seed's accuracy per
  cell with the ±1 SD band and a one-image scale bar.
- `docs/benchmarks/food11_paired_ci.png` — paired differences with bootstrap CIs,
  coloured by verdict; the zero line shows which pairs cannot be ordered.
- `docs/benchmarks/food11_power.png` — the exact test's p-value floor against
  seed count, marking where this study sits (n=6, floor 0.031) and where alpha
  becomes reachable (n=6).
- `docs/benchmarks/food11_validation_power.png` — the measured sign-flip rate and
  measurement granularity against validation subset size (§3.4), i.e. what
  enlarging the labelled set actually buys. Written by passing
  `--power-report docs/benchmarks/food101_validation_power.json` to
  `scripts/plot_significance.py`.
