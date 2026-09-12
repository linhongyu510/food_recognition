"""Tests for food_recognition.significance.

Where a closed form or a hand-computable value exists, the test asserts against
that rather than against the implementation's own output, per CONTRIBUTING.md.
SciPy is used only as an independent cross-check and every such test skips
cleanly when it is absent, so CI never depends on it.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from food_recognition.significance import (
    AccuracyResolution,
    accuracy_resolution,
    bootstrap_accuracy_diff_ci,
    bootstrap_mean_ci,
    cohens_dz,
    compare_cells,
    mcnemar_exact,
    paired_differences,
    paired_permutation_test,
    paired_t_test,
    pairwise_comparisons,
    seeds_needed,
    student_t_sf,
    wilson_interval,
)


def _stats():
    """Return scipy.stats, or skip the calling test if SciPy is unavailable."""
    return pytest.importorskip("scipy.stats")


# ---------------------------------------------------------------------------
# t distribution
# ---------------------------------------------------------------------------
def test_student_t_sf_is_one_half_at_zero() -> None:
    for df in (1, 2, 5, 30, 1000):
        assert student_t_sf(0.0, df) == pytest.approx(0.5, abs=1e-12)


def test_student_t_sf_matches_cauchy_closed_form_at_df_one() -> None:
    # df=1 is the standard Cauchy, whose survival function is exactly
    # 0.5 - atan(t)/pi. An independent closed form, not the implementation.
    for t in (0.5, 1.0, 2.0, 7.5):
        assert student_t_sf(t, 1) == pytest.approx(0.5 - math.atan(t) / math.pi, abs=1e-10)


def test_student_t_sf_symmetry_and_monotonicity() -> None:
    assert student_t_sf(1.3, 8) + student_t_sf(-1.3, 8) == pytest.approx(1.0, abs=1e-12)
    tails = [student_t_sf(t, 8) for t in (0.5, 1.0, 2.0, 4.0)]
    assert tails == sorted(tails, reverse=True)


def test_student_t_sf_rejects_nonpositive_df() -> None:
    with pytest.raises(ValueError, match="df must be positive"):
        student_t_sf(1.0, 0)


def test_student_t_sf_matches_scipy() -> None:
    st = _stats()
    for df in (1, 2, 3, 7, 25, 200):
        for t in (0.1, 0.9, 1.96, 3.5, 9.0):
            assert student_t_sf(t, df) == pytest.approx(st.t.sf(t, df), rel=1e-10)


# ---------------------------------------------------------------------------
# Wilson interval and measurement resolution
# ---------------------------------------------------------------------------
def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = wilson_interval(627, 660)
    assert low < 627 / 660 < high


def test_wilson_interval_narrows_as_n_grows() -> None:
    # Same proportion, 38x the images: the band must shrink, which is the whole
    # argument for enlarging the validation set.
    small = wilson_interval(627, 660)
    large = wilson_interval(23_988, 25_250)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_stays_inside_zero_one_at_the_extremes() -> None:
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


@pytest.mark.parametrize(
    ("successes", "total", "match"),
    [(1, 0, "total must be positive"), (11, 10, r"successes must be in \[0, 10\]")],
)
def test_wilson_interval_validates_input(successes: int, total: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        wilson_interval(successes, total)


def test_wilson_interval_matches_scipy_normal_quantile() -> None:
    st = _stats()
    z = st.norm.ppf(0.975)
    n, k = 660, 627
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low, high = wilson_interval(k, n)
    assert low == pytest.approx(centre - half, abs=1e-9)
    assert high == pytest.approx(centre + half, abs=1e-9)


def test_accuracy_resolution_points_per_image_is_exact() -> None:
    # One image out of 660 moves accuracy by 1/660 = 0.1515... points; out of
    # 25,250 it moves it by 1/25250 = 0.00396 points. Hand-computed.
    assert accuracy_resolution(660).points_per_image == pytest.approx(0.15151515151, abs=1e-9)
    assert accuracy_resolution(25_250).points_per_image == pytest.approx(0.00396039604, abs=1e-9)


def test_accuracy_resolution_ratio_is_the_dataset_ratio() -> None:
    coarse = accuracy_resolution(660)
    fine = accuracy_resolution(25_250)
    assert coarse.points_per_image / fine.points_per_image == pytest.approx(25_250 / 660)


def test_accuracy_resolution_images_for_gap() -> None:
    coarse = accuracy_resolution(660)
    # A 0.76-point spread on 660 images is exactly 5 images.
    assert coarse.images_for(0.76) == pytest.approx(5.016, abs=1e-3)
    fine = accuracy_resolution(25_250)
    assert fine.images_for(0.76) == pytest.approx(191.9, abs=0.1)


def test_accuracy_resolution_does_not_call_a_sub_ci_gap_resolved() -> None:
    # The load-bearing negative result: a 0.10-point gap is under one image at
    # 660, and even at 25,250 it stays inside the single-run CI.
    assert accuracy_resolution(660).resolves(0.10) is False
    assert accuracy_resolution(25_250).resolves(0.10) is False
    # A gap far wider than the CI is resolved at the large size.
    assert accuracy_resolution(25_250).resolves(4.0) is True


def test_accuracy_resolution_ci_width_shrinks_with_n() -> None:
    assert accuracy_resolution(25_250).ci_width_points < accuracy_resolution(660).ci_width_points


def test_accuracy_resolution_to_dict_is_json_ready() -> None:
    data = accuracy_resolution(660).to_dict()
    assert set(data) == {
        "n_images",
        "points_per_image",
        "ci_width_points",
        "alpha",
        "reference_accuracy",
    }
    assert isinstance(data["n_images"], int)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_images": 0}, "n_images must be positive"),
        ({"n_images": 10, "reference_accuracy": 1.5}, "reference_accuracy must be in"),
    ],
)
def test_accuracy_resolution_validates_input(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        accuracy_resolution(**kwargs)


def test_accuracy_resolution_is_frozen() -> None:
    res = accuracy_resolution(660)
    assert isinstance(res, AccuracyResolution)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_images = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def test_paired_differences_uses_only_shared_seeds() -> None:
    seeds, diffs = paired_differences({0: 0.95, 1: 0.96, 3: 0.99}, {0: 0.94, 1: 0.94, 2: 0.5})
    assert seeds == [0, 1]
    np.testing.assert_allclose(diffs, [0.01, 0.02], atol=1e-12)


def test_paired_differences_is_ordered_by_seed_not_dict_order() -> None:
    seeds, diffs = paired_differences({2: 0.3, 0: 0.1, 1: 0.2}, {1: 0.0, 0: 0.0, 2: 0.0})
    assert seeds == [0, 1, 2]
    np.testing.assert_allclose(diffs, [0.1, 0.2, 0.3], atol=1e-12)


def test_paired_differences_with_no_overlap_is_empty() -> None:
    seeds, diffs = paired_differences({0: 0.9}, {1: 0.9})
    assert seeds == [] and diffs.size == 0


# ---------------------------------------------------------------------------
# effect size
# ---------------------------------------------------------------------------
def test_cohens_dz_hand_computed() -> None:
    # diffs = [1, 2, 3]: mean 2, sample sd 1, so dz = 2.
    assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_cohens_dz_is_scale_invariant() -> None:
    assert cohens_dz([0.01, 0.02, 0.03]) == pytest.approx(cohens_dz([1.0, 2.0, 3.0]))


def test_cohens_dz_sign_follows_the_mean() -> None:
    assert cohens_dz([-1.0, -2.0, -3.0]) == pytest.approx(-2.0)


def test_cohens_dz_is_nan_when_undefined() -> None:
    assert math.isnan(cohens_dz([0.5]))  # n < 2
    assert math.isnan(cohens_dz([0.5, 0.5, 0.5]))  # zero variance


# ---------------------------------------------------------------------------
# paired t-test
# ---------------------------------------------------------------------------
def test_paired_t_test_hand_computed() -> None:
    # diffs = [1, 2, 3]: mean 2, sd 1, n 3 -> t = 2 / (1/sqrt(3)) = 2*sqrt(3).
    r = paired_t_test([1.0, 2.0, 3.0])
    assert r["n"] == 3 and r["df"] == 2
    assert r["mean_diff"] == pytest.approx(2.0)
    assert r["sd_diff"] == pytest.approx(1.0)
    assert r["t_stat"] == pytest.approx(2.0 * math.sqrt(3.0))


def test_paired_t_test_ci_brackets_the_mean() -> None:
    r = paired_t_test([0.01, 0.02, 0.03])
    assert r["t_ci_low"] < r["mean_diff"] < r["t_ci_high"]


def test_paired_t_test_symmetric_data_gives_p_one() -> None:
    r = paired_t_test([-1.0, 0.0, 1.0])
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["p_value"] == pytest.approx(1.0)


def test_paired_t_test_constant_zero_differences_is_p_one_not_nan() -> None:
    r = paired_t_test([0.0, 0.0, 0.0])
    assert r["t_stat"] == 0.0 and r["p_value"] == 1.0


def test_paired_t_test_constant_nonzero_differences_is_undefined() -> None:
    # Zero variance with a non-zero mean cannot be tested; reporting p=0 here
    # would manufacture certainty out of a degenerate sample.
    r = paired_t_test([0.02, 0.02, 0.02])
    assert math.isnan(float(r["t_stat"])) and math.isnan(float(r["p_value"]))


def test_paired_t_test_treats_float_residue_as_zero_variance() -> None:
    # Regression: 0.92-0.90, 0.93-0.91 and 0.94-0.92 are all "0.02", but in
    # binary they differ around 1e-17. An exact ``sd == 0`` check missed that and
    # produced t of order 1e14 with p ~ 0, i.e. a spurious "highly significant"
    # verdict from three identical differences. The tolerance is what fixes it.
    diffs = [0.92 - 0.90, 0.93 - 0.91, 0.94 - 0.92]
    assert 0.0 < np.std(diffs, ddof=1) < 1e-12  # the residue really is present
    r = paired_t_test(diffs)
    assert math.isnan(float(r["t_stat"])) and math.isnan(float(r["p_value"]))
    assert math.isnan(cohens_dz(diffs))


def test_paired_t_test_larger_effect_gives_smaller_p() -> None:
    small = paired_t_test([0.001, 0.002, 0.0015])["p_value"]
    large = paired_t_test([0.05, 0.06, 0.055])["p_value"]
    assert float(large) < float(small)


@pytest.mark.parametrize(
    ("diffs", "kwargs", "match"),
    [
        ([0.1], {}, "need at least 2 paired observations"),
        ([0.1, 0.2], {"alpha": 0.0}, "alpha must be in"),
        ([0.1, 0.2], {"alpha": 1.0}, "alpha must be in"),
    ],
)
def test_paired_t_test_validates_input(diffs: list[float], kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        paired_t_test(diffs, **kwargs)


def test_paired_t_test_matches_scipy() -> None:
    st = _stats()
    rng = np.random.default_rng(7)
    for _ in range(50):
        d = rng.normal(0.3, 1.0, size=int(rng.integers(2, 40)))
        mine = paired_t_test(d)
        ref = st.ttest_1samp(d, 0.0)
        assert mine["t_stat"] == pytest.approx(ref.statistic, rel=1e-10)
        assert mine["p_value"] == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)


def test_paired_t_test_ci_matches_scipy_confidence_interval() -> None:
    st = _stats()
    d = np.array([0.01, -0.004, 0.021, 0.008])
    ref = st.ttest_1samp(d, 0.0).confidence_interval(0.95)
    mine = paired_t_test(d)
    assert mine["t_ci_low"] == pytest.approx(ref.low, rel=1e-8)
    assert mine["t_ci_high"] == pytest.approx(ref.high, rel=1e-8)


# ---------------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------------
def test_permutation_test_enumerates_exactly_for_small_n() -> None:
    r = paired_permutation_test([0.01, 0.02, 0.015])
    assert r["exact"] is True
    assert r["n_patterns"] == 8
    # All three differences share a sign, so no sign flip beats the observed
    # |mean| except the all-flipped mirror: 2/8 = 0.25.
    assert r["p_value"] == pytest.approx(0.25)


def test_permutation_test_floor_explains_why_three_seeds_cannot_reach_005() -> None:
    r = paired_permutation_test([0.001, 0.002, 0.003])
    assert r["min_attainable_p"] == pytest.approx(0.25)
    assert float(r["min_attainable_p"]) > 0.05


def test_permutation_test_floor_falls_as_seeds_are_added() -> None:
    floors = [
        float(paired_permutation_test([0.01] * n)["min_attainable_p"]) for n in (3, 5, 6, 8)
    ]
    assert floors == sorted(floors, reverse=True)
    # Six paired seeds are the first design that can reach alpha=0.05.
    assert floors[1] > 0.05 and floors[2] <= 0.05


def test_permutation_test_is_invariant_to_sign_of_all_differences() -> None:
    up = paired_permutation_test([0.01, 0.02, 0.03])
    down = paired_permutation_test([-0.01, -0.02, -0.03])
    assert up["p_value"] == down["p_value"]


def test_permutation_test_zero_differences_give_p_one() -> None:
    assert paired_permutation_test([0.0, 0.0, 0.0])["p_value"] == pytest.approx(1.0)


def test_permutation_test_switches_to_sampling_for_large_n() -> None:
    r = paired_permutation_test([0.01] * 12, max_exact_exponent=8, n_resamples=500, seed=3)
    assert r["exact"] is False and r["n_patterns"] == 500


def test_permutation_test_sampled_mode_is_deterministic_given_seed() -> None:
    kwargs = {"max_exact_exponent": 4, "n_resamples": 400, "seed": 11}
    d = [0.01, -0.002, 0.03, 0.004, 0.01, -0.001]
    assert (
        paired_permutation_test(d, **kwargs)["p_value"]
        == paired_permutation_test(d, **kwargs)["p_value"]
    )


def test_permutation_test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least 1 paired observation"):
        paired_permutation_test([])


def test_permutation_test_matches_brute_force_enumeration() -> None:
    import itertools

    d = np.array([0.03, -0.01, 0.02, 0.005])
    observed = abs(d.mean())
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=len(d))
        if abs(np.dot(signs, d) / len(d)) >= observed - 1e-12
    )
    assert paired_permutation_test(d)["p_value"] == pytest.approx(hits / 2 ** len(d))


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_observed_mean() -> None:
    ci = bootstrap_mean_ci([0.01, 0.02, 0.015], n_resamples=2000, seed=5)
    assert ci.low <= ci.mean <= ci.high


def test_bootstrap_ci_records_its_own_protocol() -> None:
    # The protocol must travel with the number, not live only in prose.
    ci = bootstrap_mean_ci([0.01, 0.02], n_resamples=1234, seed=99)
    assert ci.resample_unit == "seed"
    assert ci.n_resamples == 1234
    assert ci.seed == 99
    assert ci.n_observations == 2
    assert ci.alpha == 0.05
    assert ci.to_dict()["resample_unit"] == "seed"
    assert "excludes_zero" in ci.to_dict()


def test_bootstrap_ci_of_identical_values_is_degenerate() -> None:
    ci = bootstrap_mean_ci([0.02, 0.02, 0.02], n_resamples=500, seed=1)
    assert ci.low == pytest.approx(0.02) and ci.high == pytest.approx(0.02)


def test_bootstrap_excludes_zero_flag() -> None:
    clear = bootstrap_mean_ci([0.05] * 6, n_resamples=2000, seed=2)
    assert clear.excludes_zero is True
    straddling = bootstrap_mean_ci([-0.02, 0.03, -0.01, 0.02], n_resamples=4000, seed=2)
    assert straddling.excludes_zero is False


def test_bootstrap_is_deterministic_given_seed() -> None:
    a = bootstrap_mean_ci([0.01, -0.02, 0.03], n_resamples=1000, seed=42)
    b = bootstrap_mean_ci([0.01, -0.02, 0.03], n_resamples=1000, seed=42)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_ci_narrows_with_more_observations() -> None:
    rng = np.random.default_rng(4)
    narrow = bootstrap_mean_ci(rng.normal(0.01, 0.005, 400), n_resamples=3000, seed=1)
    wide = bootstrap_mean_ci(rng.normal(0.01, 0.005, 6), n_resamples=3000, seed=1)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha must be in"),
        ({"n_resamples": 0}, "n_resamples must be positive"),
    ],
)
def test_bootstrap_validates_input(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_mean_ci([0.1, 0.2], **kwargs)


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least 1 paired observation"):
        bootstrap_mean_ci([])


def test_bootstrap_ci_approximates_the_t_interval_on_ample_data() -> None:
    # With a few hundred well-behaved observations the two intervals should
    # agree closely; if they diverge here, one of them is wrong.
    rng = np.random.default_rng(8)
    d = rng.normal(0.02, 0.01, 500)
    boot = bootstrap_mean_ci(d, n_resamples=5000, seed=6)
    t_res = paired_t_test(d)
    assert boot.low == pytest.approx(float(t_res["t_ci_low"]), abs=5e-4)
    assert boot.high == pytest.approx(float(t_res["t_ci_high"]), abs=5e-4)


# ---------------------------------------------------------------------------
# compare_cells
# ---------------------------------------------------------------------------
def _cells() -> tuple[dict[int, float], dict[int, float]]:
    # The project's real b3_380 and b4_380 three-seed accuracies.
    b3 = {0: 0.954545, 1: 0.956061, 2: 0.956061}
    b4 = {0: 0.95, 1: 0.956061, 2: 0.957576}
    return b3, b4


def test_compare_cells_reports_every_required_statistic() -> None:
    b3, b4 = _cells()
    c = compare_cells("b3_380", b3, "b4_380", b4)
    for key in (
        "t_test",
        "permutation",
        "bootstrap",
        "effect_size_dz",
        "verdict",
        "mean_diff",
        "power_limited",
    ):
        assert key in c.to_dict()


def test_compare_cells_three_seeds_are_power_limited() -> None:
    b3, b4 = _cells()
    c = compare_cells("b3_380", b3, "b4_380", b4)
    assert c.power_limited is True
    assert c.verdict == "not_significant"
    assert c.significant is False
    assert any("cannot go below" in n for n in c.notes)


def test_power_limited_does_not_erase_a_real_t_test_rejection() -> None:
    # Regression: an earlier verdict function returned "underpowered" for every
    # 3-seed pair, which silently overwrote b3_380 vs b0_380 -- a pair whose
    # paired t-test rejects at p=0.0198 with dz=+4.04. Collapsing that to a
    # single "underpowered" label would have reported "no differences found"
    # across the whole grid, hiding the one pair that does separate.
    b3 = {0: 0.954545, 1: 0.956061, 2: 0.956061}
    b0_380 = {0: 0.948485, 1: 0.946970, 2: 0.950000}
    c = compare_cells("b3_380", b3, "b0_380", b0_380)
    assert float(c.t_test["p_value"]) < 0.05
    assert c.bootstrap.excludes_zero is True
    assert c.verdict == "significant_parametric_only"
    assert c.significant is True
    # ...while still recording that the exact test could not corroborate it.
    assert c.power_limited is True
    assert float(c.permutation["p_value"]) == pytest.approx(0.25)
    assert any("provisional" in n for n in c.notes)


def test_seeds_needed_is_six_at_the_default_alpha() -> None:
    # 2/2**6 = 0.03125 <= 0.05, while 2/2**5 = 0.0625 > 0.05.
    assert seeds_needed(0.05) == 6
    assert seeds_needed(0.01) == 8
    assert 2.0 / 2 ** seeds_needed(0.05) <= 0.05


def test_seeds_needed_validates_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        seeds_needed(0.0)


def test_compare_cells_mean_diff_matches_hand_arithmetic() -> None:
    b3, b4 = _cells()
    c = compare_cells("b3_380", b3, "b4_380", b4)
    expected = float(np.mean(list(b3.values())) - np.mean(list(b4.values())))
    assert c.mean_diff == pytest.approx(expected, abs=1e-9)
    assert c.mean_diff * 100 == pytest.approx(0.101, abs=1e-3)


def test_compare_cells_uses_only_shared_seeds() -> None:
    c = compare_cells("a", {0: 0.9, 1: 0.92, 5: 0.99}, "b", {0: 0.89, 1: 0.90})
    assert c.seeds == [0, 1]
    assert c.t_test["n"] == 2


def test_compare_cells_detects_a_large_real_difference() -> None:
    # Eight seeds with a consistent ~4-point gap and realistic per-seed jitter:
    # the design can reach alpha and the data clear it, so this must come back
    # significant. Without this test a module that always says "not significant"
    # would pass everything else.
    jitter_high = [0.0, 0.004, -0.003, 0.002, -0.001, 0.003, -0.002, 0.001]
    jitter_low = [0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.004, 0.0]
    high = {i: 0.950 + jitter_high[i] for i in range(8)}
    low = {i: 0.910 + jitter_low[i] for i in range(8)}
    c = compare_cells("high", high, "low", low, n_resamples=3000)
    assert c.verdict == "significant"
    assert c.significant is True
    assert float(c.t_test["p_value"]) < 0.05
    assert c.bootstrap.excludes_zero is True


def test_compare_cells_calls_a_noisy_gap_not_significant() -> None:
    rng = np.random.default_rng(12)
    high = {i: float(0.95 + rng.normal(0, 0.01)) for i in range(8)}
    low = {i: float(0.95 + rng.normal(0, 0.01)) for i in range(8)}
    c = compare_cells("h", high, "l", low, n_resamples=3000)
    assert c.verdict == "not_significant"
    assert c.significant is False


def test_compare_cells_flags_zero_variance_differences() -> None:
    c = compare_cells("a", {0: 0.92, 1: 0.93, 2: 0.94}, "b", {0: 0.90, 1: 0.91, 2: 0.92})
    assert any("undefined" in n for n in c.notes)


def test_compare_cells_requires_two_shared_seeds() -> None:
    with pytest.raises(ValueError, match="share 1 seed"):
        compare_cells("a", {0: 0.9, 1: 0.9}, "b", {0: 0.9})


def test_compare_cells_format_line_is_quotable() -> None:
    b3, b4 = _cells()
    line = compare_cells("b3_380", b3, "b4_380", b4).format_line()
    assert "b3_380 vs b4_380" in line
    assert "pts" in line and "p=" in line and "dz=" in line
    assert "not_significant" in line


def test_compare_cells_format_line_handles_undefined_p() -> None:
    line = compare_cells("a", {0: 0.92, 1: 0.93}, "b", {0: 0.90, 1: 0.91}).format_line()
    assert "p=n/a" in line and "dz=n/a" in line


def test_compare_cells_to_dict_is_json_serialisable() -> None:
    import json

    b3, b4 = _cells()
    text = json.dumps(compare_cells("b3_380", b3, "b4_380", b4).to_dict())
    assert "b3_380" in text


# ---------------------------------------------------------------------------
# pairwise_comparisons
# ---------------------------------------------------------------------------
def test_pairwise_comparisons_covers_every_pair() -> None:
    cells = {n: {0: 0.9, 1: 0.91, 2: 0.92} for n in ("a", "b", "c", "d")}
    assert len(pairwise_comparisons(cells, n_resamples=200)) == 6


def test_pairwise_comparisons_orders_higher_mean_first() -> None:
    cells = {"lo": {0: 0.90, 1: 0.90}, "hi": {0: 0.95, 1: 0.95}}
    (c,) = pairwise_comparisons(cells, n_resamples=200)
    assert (c.cell_a, c.cell_b) == ("hi", "lo")
    assert c.mean_diff > 0


def test_pairwise_comparisons_sorts_by_absolute_gap() -> None:
    cells = {
        "a": {0: 0.90, 1: 0.90},
        "b": {0: 0.901, 1: 0.901},
        "c": {0: 0.95, 1: 0.95},
    }
    gaps = [abs(c.mean_diff) for c in pairwise_comparisons(cells, n_resamples=200)]
    assert gaps == sorted(gaps, reverse=True)


def test_pairwise_comparisons_skips_pairs_without_shared_seeds() -> None:
    cells = {"a": {0: 0.9, 1: 0.9}, "b": {0: 0.8, 1: 0.8}, "orphan": {7: 0.99, 8: 0.99}}
    names = {(c.cell_a, c.cell_b) for c in pairwise_comparisons(cells, n_resamples=200)}
    assert names == {("a", "b")}


def test_pairwise_comparisons_rejects_min_shared_below_two() -> None:
    with pytest.raises(ValueError, match="min_shared_seeds must be at least 2"):
        pairwise_comparisons({"a": {0: 0.9}}, min_shared_seeds=1)


def test_pairwise_comparisons_on_empty_input_is_empty() -> None:
    assert pairwise_comparisons({}) == []


# ---------------------------------------------------------------------------
# McNemar / image-level pairing
# ---------------------------------------------------------------------------
def test_mcnemar_counts_discordant_pairs() -> None:
    a = [1, 1, 0, 0, 1, 1]
    b = [1, 0, 1, 0, 1, 0]
    r = mcnemar_exact(a, b)
    assert r["only_a_correct"] == 2  # indices 1 and 5
    assert r["only_b_correct"] == 1  # index 2
    assert r["n_discordant"] == 3
    assert r["n_correct_a"] == 4 and r["n_correct_b"] == 3


def test_mcnemar_identical_vectors_give_p_one() -> None:
    r = mcnemar_exact([1, 0, 1, 1], [1, 0, 1, 1])
    assert r["n_discordant"] == 0
    assert r["p_value"] == 1.0
    assert r["accuracy_diff"] == 0.0


def test_mcnemar_hand_computed_small_case() -> None:
    # 3 discordant images, all favouring A. Two-sided exact binomial:
    # 2 * (1/2)**3 = 0.25.
    r = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
    assert r["n_discordant"] == 3
    assert r["p_value"] == pytest.approx(0.25)


def test_mcnemar_symmetric_split_gives_p_one() -> None:
    r = mcnemar_exact([1, 0], [0, 1])
    assert r["p_value"] == pytest.approx(1.0)


def test_mcnemar_is_symmetric_in_its_arguments() -> None:
    a = [1, 1, 0, 1, 0, 0, 1]
    b = [0, 1, 1, 1, 0, 1, 1]
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(mcnemar_exact(b, a)["p_value"])
    assert mcnemar_exact(a, b)["accuracy_diff"] == pytest.approx(
        -mcnemar_exact(b, a)["accuracy_diff"]
    )


def test_mcnemar_large_consistent_gap_is_significant() -> None:
    a = [1] * 200 + [0] * 50
    b = [1] * 150 + [0] * 100
    assert float(mcnemar_exact(a, b)["p_value"]) < 1e-9


def test_mcnemar_p_value_never_exceeds_one() -> None:
    # A nearly even split can push 2 x one tail over 1 before the cap.
    for n in range(2, 40, 2):
        a = [1, 0] * (n // 2)
        b = [0, 1] * (n // 2)
        assert 0.0 <= float(mcnemar_exact(a, b)["p_value"]) <= 1.0


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ([1, 0], [1, 0, 1], "align image-by-image"),
        ([], [], "at least 1 paired image"),
        ([1, 2], [1, 0], "only 0 and 1"),
    ],
)
def test_mcnemar_validates_input(a: list[int], b: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        mcnemar_exact(a, b)


def test_mcnemar_matches_scipy_binomtest() -> None:
    st = _stats()
    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(20, 500))
        a = rng.integers(0, 2, n)
        b = rng.integers(0, 2, n)
        mine = mcnemar_exact(a, b)
        if mine["n_discordant"] == 0:
            continue
        ref = st.binomtest(
            int(mine["only_a_correct"]), int(mine["n_discordant"]), 0.5
        ).pvalue
        assert mine["p_value"] == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_bootstrap_accuracy_diff_records_image_as_the_unit() -> None:
    a = [1] * 90 + [0] * 10
    b = [1] * 80 + [0] * 20
    ci = bootstrap_accuracy_diff_ci(a, b, n_resamples=2000, seed=1)
    assert ci.resample_unit == "image"
    assert ci.n_observations == 100
    assert ci.mean == pytest.approx(0.10)


def test_bootstrap_accuracy_diff_brackets_the_observed_gap() -> None:
    rng = np.random.default_rng(9)
    a = rng.integers(0, 2, 800)
    b = rng.integers(0, 2, 800)
    ci = bootstrap_accuracy_diff_ci(a, b, n_resamples=2000, seed=2)
    assert ci.low <= ci.mean <= ci.high


def test_bootstrap_accuracy_diff_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="align image-by-image"):
        bootstrap_accuracy_diff_ci([1, 0], [1, 0, 1])
