"""Paired significance testing for multi-seed benchmark comparisons.

Reporting only the mean of a few seeds cannot answer the question the ablation
grid actually poses: *is cell A better than cell B, or is the gap smaller than
the run-to-run noise?* This module answers it with three complementary
statistics over the **same seeds**:

* a two-sided **paired t-test** on the per-seed differences,
* an **exact paired sign-flip permutation test**, which makes no normality
  assumption and is the honest choice when the number of seeds is tiny,
* a **percentile bootstrap confidence interval** for the mean difference,

plus the effect size (Cohen's *dz*) so a "significant" verdict can be read
next to how large the effect is.

The three tests can disagree, and that disagreement is informative rather than a
defect: with only three seeds the exact test's p-value floor is 0.25, so it
cannot reject at alpha=0.05 even for an arbitrarily large true effect. A pair
where the t-test rejects but the exact test cannot is therefore reported as
``significant_parametric_only`` rather than being rounded to either
"significant" or "not significant" -- see :func:`compare_cells`.

Everything is implemented on top of :mod:`numpy` alone. The regularized
incomplete beta function behind the t-distribution tail is computed with a
Lentz continued fraction, so the package needs no SciPy at runtime; SciPy is
only used by the test suite to cross-check these values when it is installed.

Statistical conventions used throughout, stated once so results are quotable:

``pairing unit``
    One random seed. A comparison of cells A and B uses only the seeds present
    in *both* cells, and the i-th difference is ``acc_A(seed_i) - acc_B(seed_i)``
    -- same seed, same data split, same schedule, so the difference isolates
    the configuration change.
``bootstrap resampling unit``
    One paired difference, i.e. one seed. Resampling is done with replacement
    over the ``n`` seed-level differences, never over individual validation
    images, because the noise being estimated is run-to-run training noise.
``default alpha``
    0.05, two-sided.
``accuracies``
    Fractions in ``[0, 1]``. Differences are reported in the same unit;
    multiply by 100 for percentage points.

A separate helper, :func:`accuracy_resolution`, reports the *measurement*
granularity of a validation set (how much one image is worth), which is a
different and additive source of uncertainty from seed noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "AccuracyResolution",
    "BootstrapInterval",
    "PairedComparison",
    "accuracy_resolution",
    "bootstrap_accuracy_diff_ci",
    "bootstrap_mean_ci",
    "cohens_dz",
    "compare_cells",
    "mcnemar_exact",
    "pairwise_comparisons",
    "paired_differences",
    "paired_permutation_test",
    "paired_t_test",
    "seeds_needed",
    "student_t_sf",
    "wilson_interval",
]

DEFAULT_ALPHA = 0.05
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260913
# 2**20 sign patterns is about a million; beyond that, enumerate randomly.
MAX_EXACT_PERMUTATION_EXPONENT = 20
# Below this, a spread is floating-point residue rather than real variation.
# Subtracting equal accuracies leaves ~1e-17 rather than exactly 0, and treating
# that as real variance turns an undefined t statistic into one of order 1e14.
# One image out of 25,250 is 4e-5, so 1e-12 is far below anything measurable.
SD_DEGENERATE_ATOL = 1e-12


def _normal_ppf(prob: float) -> float:
    """Standard normal quantile, from the stdlib ``erf`` by bisection.

    Derived from ``erf`` rather than from the t quantile at a huge ``df``: t at
    ``df = 1e7`` still differs from the normal in the 7th decimal, which is
    enough to disagree with any independent reference.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# distribution helpers (no SciPy)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` for ``0 <= x <= 1``."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x in (0.0, 1.0):
        return x
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    # The continued fraction converges quickly only on one side of the mean;
    # reflect via I_x(a, b) = 1 - I_{1-x}(b, a) when x is past it.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper-tail probability ``P(T > t)`` of Student's t with ``df`` degrees."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(0.5 * df, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def _t_ppf(prob: float, df: float) -> float:
    """Inverse t CDF by bisection; accurate to ~1e-10 and dependency-free."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - student_t_sf(mid, df) < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wilson_interval(
    successes: int, total: int, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used to express how precisely a *single* run's accuracy is measured on a
    validation set of a given size. This is measurement uncertainty, not seed
    uncertainty: the two do not substitute for one another.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, {total}], got {successes}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # A normal quantile is what the Wilson interval is defined with.
    z = _normal_ppf(1.0 - alpha / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# measurement granularity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AccuracyResolution:
    """How finely a validation set of ``n`` images can measure accuracy."""

    n_images: int
    points_per_image: float
    ci_width_points: float
    alpha: float
    reference_accuracy: float

    def images_for(self, gap_points: float) -> float:
        """How many images a gap of ``gap_points`` percentage points is worth."""
        return gap_points / self.points_per_image

    def resolves(self, gap_points: float) -> bool:
        """True when a gap is at least one image wide *and* wider than the CI.

        Being wider than one image only means the gap is representable; being
        wider than the sampling interval is what makes it readable.
        """
        return gap_points >= self.points_per_image and gap_points > self.ci_width_points

    def to_dict(self) -> dict[str, float | int | bool]:
        data: dict[str, float | int | bool] = dict(asdict(self))
        return data


def accuracy_resolution(
    n_images: int, *, reference_accuracy: float = 0.95, alpha: float = DEFAULT_ALPHA
) -> AccuracyResolution:
    """Quantify the measurement granularity of an ``n_images`` validation set.

    ``points_per_image`` is the accuracy change caused by one more or one fewer
    correct image, in percentage points. ``ci_width_points`` is the width of the
    Wilson interval at ``reference_accuracy``, i.e. how wide a band a single
    measured accuracy really occupies.
    """
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if not 0.0 <= reference_accuracy <= 1.0:
        raise ValueError(f"reference_accuracy must be in [0, 1], got {reference_accuracy}")

    lo, hi = wilson_interval(round(reference_accuracy * n_images), n_images, alpha)
    return AccuracyResolution(
        n_images=n_images,
        points_per_image=100.0 / n_images,
        ci_width_points=(hi - lo) * 100.0,
        alpha=alpha,
        reference_accuracy=reference_accuracy,
    )


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def paired_differences(
    a: Mapping[int, float], b: Mapping[int, float]
) -> tuple[list[int], np.ndarray]:
    """Return the shared seeds and ``a - b`` on those seeds, ordered by seed.

    Seeds present in only one of the two cells are dropped: an unpaired run
    carries no information about the *difference* between configurations.
    """
    shared = sorted(set(a) & set(b))
    diffs = np.array([float(a[s]) - float(b[s]) for s in shared], dtype=float)
    return shared, diffs


def cohens_dz(diffs: Sequence[float] | np.ndarray) -> float:
    """Paired effect size: mean difference over the SD of the differences.

    Returns NaN when the SD is undefined (n < 2) or zero (all differences
    identical), rather than inventing an infinite effect.
    """
    values = np.asarray(diffs, dtype=float)
    if values.size < 2:
        return float("nan")
    sd = float(values.std(ddof=1))
    if sd <= SD_DEGENERATE_ATOL:
        return float("nan")
    return float(values.mean() / sd)


def paired_t_test(
    diffs: Sequence[float] | np.ndarray, alpha: float = DEFAULT_ALPHA
) -> dict[str, float | int]:
    """Two-sided one-sample t-test on paired differences against zero."""
    values = np.asarray(diffs, dtype=float)
    n = int(values.size)
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    df = n - 1
    if sd <= SD_DEGENERATE_ATOL:
        # Identical differences: t is undefined. A zero mean is "no evidence of
        # a difference" (p=1); a non-zero constant mean cannot be tested with
        # zero variance, so report it as undefined rather than as p=0.
        t_stat = 0.0 if abs(mean) <= SD_DEGENERATE_ATOL else float("nan")
        p_value = 1.0 if abs(mean) <= SD_DEGENERATE_ATOL else float("nan")
    else:
        t_stat = mean / (sd / math.sqrt(n))
        p_value = 2.0 * student_t_sf(abs(t_stat), df)

    half = _t_ppf(1.0 - alpha / 2.0, df) * sd / math.sqrt(n)
    return {
        "n": n,
        "df": df,
        "mean_diff": mean,
        "sd_diff": sd,
        "t_stat": t_stat,
        "p_value": p_value,
        "t_ci_low": mean - half,
        "t_ci_high": mean + half,
        "alpha": alpha,
    }


def paired_permutation_test(
    diffs: Sequence[float] | np.ndarray,
    *,
    max_exact_exponent: int = MAX_EXACT_PERMUTATION_EXPONENT,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int | bool]:
    """Exact two-sided sign-flip permutation test on paired differences.

    Under the null "the configuration change does nothing", the sign of each
    seed's difference is exchangeable, so every one of the ``2**n`` sign
    patterns is equally likely. With the handful of seeds this project can
    afford, all patterns are enumerated exactly, which sidesteps the normality
    assumption the t-test makes on 3 points. The reported p-value uses the
    standard convention of counting the observed statistic itself, so the
    smallest attainable p-value is ``2 / 2**n`` -- for ``n = 3`` that floor is
    0.25, which is *why* three seeds cannot produce significance at 0.05.
    """
    values = np.asarray(diffs, dtype=float)
    n = int(values.size)
    if n < 1:
        raise ValueError("need at least 1 paired observation")

    observed = abs(float(values.mean()))
    exact = n <= max_exact_exponent
    if exact:
        signs = 1.0 - 2.0 * (
            (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
        ).astype(float)
        total = 2**n
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
        signs[0] = 1.0  # keep the observed pattern in the reference set
        total = n_resamples

    stats = np.abs(signs @ values / n)
    # ``>=`` with a tolerance: the observed pattern must count itself even
    # after floating-point reordering inside the matrix product.
    hits = int(np.count_nonzero(stats >= observed - 1e-12))
    return {
        "n": n,
        "observed_abs_mean_diff": observed,
        "p_value": hits / total,
        "n_patterns": total,
        "exact": exact,
        "min_attainable_p": 2.0 / total if exact else 1.0 / total,
    }


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile bootstrap interval for the mean of paired differences."""

    low: float
    high: float
    mean: float
    alpha: float
    n_resamples: int
    resample_unit: str
    seed: int
    n_observations: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = dict(asdict(self))
        data["excludes_zero"] = self.excludes_zero
        return data


def bootstrap_mean_ci(
    diffs: Sequence[float] | np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    resample_unit: str = "seed",
) -> BootstrapInterval:
    """Percentile bootstrap CI for the mean paired difference.

    The resampling unit is one paired difference (one seed), stated explicitly
    in the returned object so a reader never has to guess whether images or
    runs were resampled.
    """
    values = np.asarray(diffs, dtype=float)
    n = int(values.size)
    if n < 1:
        raise ValueError("need at least 1 paired observation")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        low=float(low),
        high=float(high),
        mean=float(values.mean()),
        alpha=alpha,
        n_resamples=n_resamples,
        resample_unit=resample_unit,
        seed=seed,
        n_observations=n,
    )


@dataclass(frozen=True)
class PairedComparison:
    """The full verdict for one pair of configurations."""

    cell_a: str
    cell_b: str
    seeds: list[int]
    per_seed_diff: list[float]
    mean_a: float
    mean_b: float
    t_test: dict[str, float | int]
    permutation: dict[str, float | int | bool]
    bootstrap: BootstrapInterval
    effect_size_dz: float
    alpha: float
    verdict: str
    power_limited: bool
    notes: list[str] = field(default_factory=list)

    @property
    def mean_diff(self) -> float:
        return float(self.t_test["mean_diff"])

    @property
    def significant(self) -> bool:
        """True when the parametric test and the bootstrap both reject.

        Deliberately does *not* require the exact permutation test to reject:
        with a handful of seeds that test has a p-value floor above alpha, so
        requiring it would label every pair "no effect" however large the gap.
        Read alongside :attr:`power_limited`.
        """
        return self.verdict in {"significant", "significant_parametric_only"}

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_a": self.cell_a,
            "cell_b": self.cell_b,
            "seeds": list(self.seeds),
            "per_seed_diff": [round(v, 8) for v in self.per_seed_diff],
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "effect_size_dz": self.effect_size_dz,
            "t_test": dict(self.t_test),
            "permutation": dict(self.permutation),
            "bootstrap": self.bootstrap.to_dict(),
            "alpha": self.alpha,
            "verdict": self.verdict,
            "power_limited": self.power_limited,
            "notes": list(self.notes),
        }

    def format_line(self) -> str:
        """One quotable line, in percentage points."""
        p = self.t_test["p_value"]
        p_text = "n/a" if isinstance(p, float) and math.isnan(p) else f"{float(p):.4f}"
        dz = self.effect_size_dz
        dz_text = "n/a" if math.isnan(dz) else f"{dz:+.2f}"
        return (
            f"{self.cell_a} vs {self.cell_b}: "
            f"{self.mean_diff * 100:+.3f} pts "
            f"[{self.bootstrap.low * 100:+.3f}, {self.bootstrap.high * 100:+.3f}] "
            f"p={p_text} dz={dz_text} -> {self.verdict}"
        )


def compare_cells(
    cell_a: str,
    accs_a: Mapping[int, float],
    cell_b: str,
    accs_b: Mapping[int, float],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Compare two cells on their shared seeds and return every statistic.

    The verdict separates two questions a single label would conflate:

    ``"significant"``
        The paired t-test rejects, the bootstrap interval excludes zero, *and*
        the assumption-free exact permutation test also rejects.
    ``"significant_parametric_only"``
        The t-test rejects and the bootstrap excludes zero, but the exact
        sign-flip test cannot reach ``alpha`` at this many seeds, so it can
        neither confirm nor contradict. The effect is real *if* the normality
        assumption the t-test makes on a few points holds. Kept as its own
        label because calling it "significant" overstates it while calling it
        "not significant" discards a genuine rejection.
    ``"not_significant"``
        Nothing rejects; the data are consistent with no difference.

    ``power_limited`` is orthogonal to the verdict and is True whenever the
    exact test's p-value floor sits above ``alpha`` -- a property of the number
    of seeds, not of the configurations.
    """
    seeds, diffs = paired_differences(accs_a, accs_b)
    if diffs.size < 2:
        raise ValueError(
            f"{cell_a} and {cell_b} share {diffs.size} seed(s); need at least 2 to pair"
        )

    t_result = paired_t_test(diffs, alpha=alpha)
    perm = paired_permutation_test(diffs, n_resamples=n_resamples, seed=seed)
    boot = bootstrap_mean_ci(diffs, alpha=alpha, n_resamples=n_resamples, seed=seed)
    dz = cohens_dz(diffs)

    notes: list[str] = []
    floor = float(perm["min_attainable_p"])
    power_limited = floor > alpha
    if power_limited:
        notes.append(
            f"exact sign-flip test on {diffs.size} seeds cannot go below "
            f"p={floor:.3f}, so it cannot confirm anything at alpha={alpha}; "
            f"{seeds_needed(alpha)} paired seeds is the minimum that can"
        )

    p_value = t_result["p_value"]
    p_is_nan = isinstance(p_value, float) and math.isnan(p_value)
    t_says_yes = (not p_is_nan) and float(p_value) < alpha
    perm_says_yes = float(perm["p_value"]) < alpha

    if t_says_yes and boot.excludes_zero:
        verdict = "significant" if perm_says_yes else "significant_parametric_only"
        if not perm_says_yes:
            notes.append(
                "the t-test rejects, but it rests on a normality assumption that "
                f"{diffs.size} points cannot support; treat as provisional"
            )
    else:
        verdict = "not_significant"
        if t_says_yes and not boot.excludes_zero:
            notes.append("t-test rejects but the bootstrap interval spans zero")

    if float(t_result["sd_diff"]) <= SD_DEGENERATE_ATOL:
        notes.append("all per-seed differences are identical; the t statistic is undefined")

    return PairedComparison(
        cell_a=cell_a,
        cell_b=cell_b,
        seeds=seeds,
        per_seed_diff=[float(v) for v in diffs],
        mean_a=float(np.mean([accs_a[s] for s in seeds])),
        mean_b=float(np.mean([accs_b[s] for s in seeds])),
        t_test=t_result,
        permutation=perm,
        bootstrap=boot,
        effect_size_dz=dz,
        alpha=alpha,
        verdict=verdict,
        power_limited=power_limited,
        notes=notes,
    )


def seeds_needed(alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest paired sample size whose exact sign-flip floor reaches ``alpha``.

    The floor of the two-sided exact test is ``2 / 2**n``, so this returns the
    smallest ``n`` with ``2 / 2**n <= alpha``. At alpha=0.05 it is 6: three seeds
    per cell can never yield a permutation-test rejection however large the true
    effect, which is a fact about the design, not about the models.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = 1
    while 2.0 / (2**n) > alpha:
        n += 1
    return n


def pairwise_comparisons(
    cells: Mapping[str, Mapping[int, float]],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    min_shared_seeds: int = 2,
) -> list[PairedComparison]:
    """Compare every pair of cells that shares at least ``min_shared_seeds``.

    Pairs are ordered so the cell with the higher mean comes first, which makes
    ``mean_diff`` non-negative and the direction of any claim unambiguous.
    Cells with too few shared seeds are skipped rather than silently compared
    on unpaired runs.
    """
    if min_shared_seeds < 2:
        raise ValueError(f"min_shared_seeds must be at least 2, got {min_shared_seeds}")

    names = sorted(cells)
    out: list[PairedComparison] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared, _ = paired_differences(cells[first], cells[second])
            if len(shared) < min_shared_seeds:
                continue
            mean_first = float(np.mean([cells[first][s] for s in shared]))
            mean_second = float(np.mean([cells[second][s] for s in shared]))
            hi, lo = (
                (first, second) if mean_first >= mean_second else (second, first)
            )
            out.append(
                compare_cells(
                    hi,
                    cells[hi],
                    lo,
                    cells[lo],
                    alpha=alpha,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
    out.sort(key=lambda c: -abs(c.mean_diff))
    return out


# ---------------------------------------------------------------------------
# image-level pairing (one fixed model pair, one fixed validation set)
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict[str, float | int]:
    """Exact two-sided McNemar test on two 0/1 correctness vectors.

    The pairing unit here is **one validation image**, not one seed: entry ``i``
    of both vectors must refer to the same image. Only the discordant images
    matter -- the ones exactly one model got right -- because images both models
    handle identically carry no information about which is better.

    Under the null, each discordant image is a fair coin, so the count favouring
    ``a`` is Binomial(``b01 + b10``, 0.5) and the exact p-value is the two-sided
    binomial tail. This answers a strictly narrower question than
    :func:`compare_cells`: it holds the two trained models fixed and asks only
    whether the *validation set* is large enough to separate them. It says
    nothing about whether retraining with another seed would reorder them.
    """
    a = np.asarray(correct_a, dtype=np.int64)
    b = np.asarray(correct_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"vectors must align image-by-image, got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("need at least 1 paired image")
    if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
        raise ValueError("correctness vectors must contain only 0 and 1")

    b10 = int(np.count_nonzero((a == 1) & (b == 0)))  # only A got it right
    b01 = int(np.count_nonzero((a == 0) & (b == 1)))  # only B got it right
    n_disc = b10 + b01

    if n_disc == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial at p=0.5. The PMF is symmetric about
        # n_disc/2, so "at least as extreme as observed" is exactly the two
        # symmetric tails beyond max(b10, b01). Computing it as 2 x one tail
        # (capped at 1) is exact; summing a PMF under a tolerance is not,
        # because the two mirror-image terms differ in the last bits and one of
        # them silently drops out.
        extreme = max(b10, b01)
        log_pmf = np.array(
            [
                math.lgamma(n_disc + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n_disc - k + 1)
                - n_disc * math.log(2.0)
                for k in range(extreme, n_disc + 1)
            ]
        )
        p_value = float(min(1.0, 2.0 * np.exp(log_pmf).sum()))

    return {
        "n_images": int(a.size),
        "n_correct_a": int(a.sum()),
        "n_correct_b": int(b.sum()),
        "only_a_correct": b10,
        "only_b_correct": b01,
        "n_discordant": n_disc,
        "accuracy_diff": float(a.mean() - b.mean()),
        "p_value": p_value,
    }


def bootstrap_accuracy_diff_ci(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> BootstrapInterval:
    """Percentile bootstrap CI for an accuracy difference, resampling images.

    The resampling unit is one **image**, and both models are always scored on
    the same resampled images so the pairing survives. Reported separately from
    :func:`bootstrap_mean_ci` because the two answer different questions and
    their intervals are not interchangeable.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"vectors must align image-by-image, got {a.shape} vs {b.shape}")
    diffs = a - b
    interval = bootstrap_mean_ci(
        diffs, alpha=alpha, n_resamples=n_resamples, seed=seed, resample_unit="image"
    )
    return interval
