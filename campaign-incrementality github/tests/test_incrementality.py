"""Tests that would actually catch a wrong answer.

The interesting tests here are not the type checks -- they are the three that
assert the estimator behaves correctly on data whose truth we control:

* ``test_recovers_known_effect`` -- a real lift is found, near its true size.
* ``test_returns_null_when_no_effect`` -- a campaign that did nothing reads
  as nothing. This is the test that catches a leak between the pre and post
  windows, which is the failure mode that silently manufactures results.
* ``test_displacement_is_detected`` -- spillover onto competitors is caught
  rather than being booked as incremental value.

Everything else is guard rails around them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from incrementality import (
    CausalImpactModel,
    EconomicsConfig,
    GroundTruth,
    SimulationSpec,
    frame_from_panel,
    load_cohort,
)
from incrementality.config import CohortConfig, ModelConfig
from incrementality.counterfactual import get_backend
from incrementality.economics import breakeven_relative_uplift, evaluate
from incrementality.features import build_design, build_donor_design, fourier_terms
from incrementality.inference import estimate_uplift
from incrementality.pipeline import _build_design_for
from incrementality.simulate import (
    aggregate_to_cohort,
    simulate_panel,
    split_control_pool,
    true_average_lift,
)
from incrementality.stacking import estimate_stacking_value
from incrementality.validation import displacement_test, placebo_in_time

# A deliberately small population: these tests check correctness of the
# machinery, not the precision achievable at production scale.
SMALL = {"n_treated": 900, "n_compset": 900, "n_distant": 1_200}


@pytest.fixture(scope="module")
def cfg() -> CohortConfig:
    return load_cohort("configs/mid_year_sale_2025.yaml")


def _panel(cfg: CohortConfig, seed: int = 4242, truth: GroundTruth | None = None):
    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=seed,
        truth=truth or GroundTruth(),
        **SMALL,
    )
    return split_control_pool(simulate_panel(spec), seed=seed + 1)


def _treated_series(cfg: CohortConfig, panel: pd.DataFrame) -> pd.DataFrame:
    series = aggregate_to_cohort(panel, cfg.outcomes)
    return series[series["cohort"] == "treated"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The three that matter
# ---------------------------------------------------------------------------


def test_recovers_known_effect_within_documented_conservatism(cfg):
    """The estimator is conservative by a known, bounded margin.

    It is not unbiased and this test does not pretend otherwise. Measured over
    five fixed seeds it understates the true lift by roughly 1.8 points, and
    the cause is identified: with high repeat participation the pre-period
    contains last year's campaign, and the block indicator that removes it
    also removes part of the seasonality at the same calendar position as the
    post window. Dropping the indicator is far worse -- bias moves from -1.8
    to -4.6 points -- so the correction stays and the residual is documented.

    The assertion is therefore two-sided and deliberately asymmetric: the
    estimate must be conservative, never inflated, and the conservatism must
    not drift beyond what has been measured. A refactor that makes the number
    larger is as much a regression as one that makes it smaller.
    """
    errors = []
    for seed in (4242, 4343, 4444, 4545, 4646):
        panel = _panel(cfg, seed=seed)
        truth = true_average_lift(panel, cfg.post_start, cfg.post_end)
        design = _build_design_for(_treated_series(cfg, panel), cfg.primary_outcome, cfg)
        result = estimate_uplift(
            get_backend("bsts"), design, alpha=0.10, n_bootstrap=500, backend_name="bsts"
        )
        assert result.is_significant, f"seed {seed}: a real 10% lift should be detected"
        assert result.relative_uplift > 0
        errors.append(result.relative_uplift - truth)

    bias = sum(errors) / len(errors)
    assert -0.035 < bias < 0.005, (
        f"mean bias {bias:+.2%} across {len(errors)} seeds is outside the "
        f"documented range of -3.5pp to +0.5pp"
    )


def test_returns_null_when_no_effect(cfg):
    """The most important test in the suite.

    A pre/post design that leaks -- a window boundary off by one, a covariate
    containing the outcome, a scaler fitted on the full sample -- will happily
    report a lift on data where nothing happened. If this goes red, no other
    result in the repo can be trusted.

    Judged on mean bias across seeds rather than a single draw, for the same
    reason as the effect test: one draw carries a couple of points of forecast
    error, so a single-draw assertion either tolerates real bias or fails at
    random.
    """
    quiet = GroundTruth(
        visits_lift=0.0,
        cvr_lift=0.0,
        los_lift=0.0,
        adr_lift=0.0,
        spillover=0.0,
        stacking_bonus=0.0,
        cross_sell_halo=0.0,
        marketplace_halo=0.0,
        prior_year_scale=0.0,
    )
    estimates = []
    for seed in (777, 888, 999, 1010, 1111):
        panel = _panel(cfg, seed=seed, truth=quiet)
        design = _build_design_for(_treated_series(cfg, panel), cfg.primary_outcome, cfg)
        result = estimate_uplift(
            get_backend("bsts"), design, alpha=0.10, n_bootstrap=500, backend_name="bsts"
        )
        estimates.append(result.relative_uplift)

    mean_estimate = sum(estimates) / len(estimates)
    assert abs(mean_estimate) < 0.015, (
        f"mean {mean_estimate:+.2%} on campaigns that did nothing"
    )
    # No single draw may look like a real campaign either.
    assert max(abs(e) for e in estimates) < 0.05


def test_displacement_is_detected(cfg):
    """Spillover onto the comp set must surface, not be booked as value."""
    spillover = -0.08
    panel = _panel(cfg, seed=888, truth=GroundTruth(spillover=spillover))
    outcome, relative = displacement_test(panel, cfg)

    # The injected parameter is NOT the expected GBV effect. Spillover hits a
    # competitor's conversion in full and its traffic at half, so an 8%
    # spillover composes to (1 - 0.04)(1 - 0.08) - 1 on booking value. Testing
    # against the raw parameter was wrong and hid a correct estimate.
    expected = (1 + 0.5 * spillover) * (1 + spillover) - 1.0
    assert relative < 0, "negative spillover must read as negative"
    assert abs(relative - expected) < 0.025, (
        f"recovered {relative:.2%}, composed truth {expected:.2%}"
    )
    assert not outcome.passed, "displacement this large must breach the -5% threshold"


# ---------------------------------------------------------------------------
# Falsification machinery
# ---------------------------------------------------------------------------


def test_in_time_placebo_is_null(cfg):
    panel = _panel(cfg)
    outcome = placebo_in_time(_treated_series(cfg, panel), cfg, shift_weeks=16, threshold=0.03)
    assert outcome.passed, f"placebo returned {outcome.statistic:+.2%}, expected ~0"


def test_intervals_widen_without_a_control_series(cfg):
    """Losing the contemporaneous control must cost precision.

    The central finding of the design study: a series that moves with the
    treated cohort week by week is what makes the estimate tight. A design
    relying only on year-lagged production must come back wider.

    Run at production scale on purpose. Precision is scale-dependent -- at
    the small cohort size used elsewhere in this suite the lagged aggregate
    is competitive, and pinning the comparison there would make the test a
    hostage to cohort size rather than a check on the claim.
    """
    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=5151,
        n_treated=2_500,
        n_compset=2_500,
        n_distant=3_000,
    )
    panel = split_control_pool(simulate_panel(spec), seed=5152)
    series = _treated_series(cfg, panel)
    lagged_only = replace(
        cfg, model=replace(cfg.model, design_version="market_prior_year")
    )

    def width(variant):
        result = estimate_uplift(
            get_backend("bsts"),
            _build_design_for(series, cfg.primary_outcome, variant),
            n_bootstrap=600,
        )
        return result.ci_relative[1] - result.ci_relative[0]

    assert width(lagged_only) > width(cfg)


def test_synthetic_control_agrees_with_bsts(cfg):
    """Triangulation: two estimators with different failure modes should agree.

    Synthetic control uses donor *units* and no covariates; BSTS uses
    covariates and no donors. Agreement is the only robustness evidence
    available when ground truth is unobservable in production.
    """
    panel = _panel(cfg)
    donors = build_donor_design(
        panel,
        cfg.primary_outcome,
        (cfg.pre_start, cfg.pre_end),
        (cfg.post_start, cfg.post_end),
    )
    fit = get_backend("synthetic_control").fit_predict(
        donors.y_pre, donors.X_pre, donors.X_post
    )
    assert np.isfinite(fit.mean).all()

    sc_lift = donors.inverse(donors.y_post).sum() / donors.inverse(fit.mean).sum() - 1
    truth = true_average_lift(panel, cfg.post_start, cfg.post_end)
    assert sc_lift > 0, f"synthetic control returned {sc_lift:.2%} on a real lift"
    assert abs(sc_lift - truth) < 0.05, (
        f"synthetic control {sc_lift:.2%} vs true {truth:.2%}"
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_scaling_uses_pre_period_only(cfg):
    """Post-period rows must not influence the scaler.

    Verified behaviourally: perturb only the post-period covariates and the
    pre-period design block must come back byte-identical.
    """
    panel = _panel(cfg)
    series = _treated_series(cfg, panel)
    baseline = _build_design_for(series, cfg.primary_outcome, cfg)

    tampered = series.copy()
    post_rows = tampered["week"] >= cfg.post_start
    tampered.loc[post_rows, "meta_impressions"] *= 50.0
    after = _build_design_for(tampered, cfg.primary_outcome, cfg)

    np.testing.assert_allclose(baseline.X_pre, after.X_pre)


def test_fourier_terms_shape():
    weeks = pd.date_range("2024-01-01", periods=52, freq="W-MON")
    assert fourier_terms(weeks, 3).shape == (52, 6)
    assert fourier_terms(weeks, 0).shape[1] == 0


def test_log_transform_rejects_non_positive(cfg):
    panel = _panel(cfg)
    series = _treated_series(cfg, panel)
    series.loc[0, "gbv"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        build_design(
            series,
            outcome="gbv",
            regressors=["clean_control_volume"],
            pre_window=(cfg.pre_start, cfg.pre_end),
            post_window=(cfg.post_start, cfg.post_end),
            log_transform=True,
        )


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


def test_breakeven_lift_yields_zero_net_value():
    """The solved break-even must actually break even.

    An algebraic identity check. The cost side moves with the lift, so a
    naive cost/margin ratio is wrong, and this catches it.
    """
    econ = EconomicsConfig()
    counterfactual = 250_000_000.0
    lift = breakeven_relative_uplift(counterfactual_bookings=counterfactual, econ=econ)
    result = evaluate(
        incremental_bookings=counterfactual * lift,
        counterfactual_bookings=counterfactual,
        actual_bookings=counterfactual * (1 + lift),
        econ=econ,
    )
    assert abs(result.net_value) < 1.0, f"net value at break-even was {result.net_value:,.2f}"


def test_discount_is_not_double_counted():
    """Cost must scale with the co-funded share, not the whole discount.

    At ``discount_funded_share = 0`` the partner funds the entire price
    reduction, which is already inside the measured GBV. Total cost then has
    to collapse to the fixed campaign cost alone.
    """
    econ = replace(EconomicsConfig(), discount_funded_share=0.0)
    result = evaluate(
        incremental_bookings=20_000_000.0,
        counterfactual_bookings=250_000_000.0,
        actual_bookings=270_000_000.0,
        econ=econ,
    )
    assert result.funded_discount_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(econ.fixed_campaign_cost)


def test_subsidy_splits_sum_to_discount_granted():
    result = evaluate(
        incremental_bookings=20_000_000.0,
        counterfactual_bookings=250_000_000.0,
        actual_bookings=270_000_000.0,
        econ=EconomicsConfig(),
    )
    assert result.subsidy_on_baseline + result.subsidy_on_incremental == pytest.approx(
        result.discount_value_granted
    )
    assert result.subsidy_on_baseline > result.subsidy_on_incremental, (
        "most of a broad promotion's discount should land on baseline demand"
    )


def test_unprofitable_at_every_lift_returns_inf():
    """Co-funding more than the margin can bear has no break-even."""
    econ = replace(
        EconomicsConfig(),
        discount_funded_share=1.0,
        discount_depth=0.30,
        discounted_booking_share=1.0,
    )
    assert breakeven_relative_uplift(counterfactual_bookings=1e8, econ=econ) == float("inf")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_activation_week_must_be_monday():
    with pytest.raises(ValueError, match="Monday"):
        CohortConfig(name="bad", activation_week=pd.Timestamp("2024-11-26"))


def test_short_pre_window_rejected():
    with pytest.raises(ValueError, match="pre_weeks"):
        CohortConfig(name="bad", activation_week=pd.Timestamp("2024-11-25"), pre_weeks=12)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        ModelConfig(backend="wishful_thinking")


def test_post_window_must_exceed_transition():
    with pytest.raises(ValueError, match="transition"):
        CohortConfig(
            name="bad",
            activation_week=pd.Timestamp("2024-11-25"),
            post_weeks=2,
            transition_weeks=2,
        )


def test_derived_windows_do_not_overlap(cfg):
    assert cfg.pre_end < cfg.post_start
    assert cfg.post_start == cfg.activation_week + pd.Timedelta(weeks=cfg.transition_weeks)
    assert cfg.n_post_weeks == cfg.post_weeks - cfg.transition_weeks


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def test_stacking_value_is_positive_and_bounded(cfg):
    panel = _panel(cfg)
    result = estimate_stacking_value(panel, cfg)
    assert result.n_stack > 0 and result.n_no_stack > 0
    assert 0.0 < result.incremental_stacking_value < 0.15
    assert result.variance_reduction >= 0.0


# ---------------------------------------------------------------------------
# CausalImpact-compatible surface
# ---------------------------------------------------------------------------


def test_compat_interface_matches_pipeline(cfg):
    """The convenience wrapper must not quietly disagree with the pipeline."""
    panel = _panel(cfg)
    frame = frame_from_panel(panel)
    ci = CausalImpactModel(
        frame,
        [str(cfg.pre_start.date()), str(cfg.pre_end.date())],
        [str(cfg.post_start.date()), str(cfg.post_end.date())],
        model_args={"n_bootstrap": 600},
    )
    assert "Relative effect" in ci.summary()
    assert "post-intervention" in ci.summary("report")
    assert list(ci.inferences.columns) == [
        "preds",
        "preds_lower",
        "preds_upper",
        "point_effects",
        "cumulative_effects",
        "response",
    ]
    assert len(ci.inferences) == cfg.n_post_weeks

    direct = estimate_uplift(
        get_backend("bsts"),
        _build_design_for(_treated_series(cfg, panel), cfg.primary_outcome, cfg),
        n_bootstrap=600,
    )
    # Not identical -- the compat frame carries market covariates as weekly
    # means across all cohorts -- but they must not tell different stories.
    assert abs(ci.result.relative_uplift - direct.relative_uplift) < 0.02


def test_compat_rejects_overlapping_windows(cfg):
    frame = frame_from_panel(_panel(cfg))
    with pytest.raises(ValueError, match="overlap"):
        CausalImpactModel(frame, ["2023-06-01", "2024-12-15"], ["2024-12-02", "2025-01-13"])


def test_compat_requires_a_control_series():
    frame = pd.DataFrame(
        {"y": np.arange(1, 60, dtype=float)},
        index=pd.date_range("2024-01-01", periods=59, freq="W-MON"),
    )
    with pytest.raises(ValueError, match="control series"):
        CausalImpactModel(frame, ["2024-01-01", "2024-10-01"], ["2024-10-08", "2024-12-01"])


# ---------------------------------------------------------------------------
# Design versions and the assumptions they rest on
# ---------------------------------------------------------------------------


def test_predictor_sets_are_distinct_and_resolvable():
    """Every named design must resolve to a non-empty, unique predictor list."""
    from incrementality import PREDICTOR_SETS

    assert set(PREDICTOR_SETS) == {
        "competitor_control",
        "own_prior_year",
        "market_prior_year",
        "clean_pool_control",
        "pruned_control",
        "marketplace_control",
        "cross_product_control",
        "negative_control",
    }
    for name, predictors in PREDICTOR_SETS.items():
        assert predictors, f"{name} resolves to an empty predictor list"
    # Design A must not reach for the clean pool; Design C must not reach for any
    # contemporaneous untreated series. These are the defining properties.
    assert "clean_control_volume" not in PREDICTOR_SETS["competitor_control"]
    assert "compset_volume" not in PREDICTOR_SETS["market_prior_year"]
    assert "clean_control_volume" not in PREDICTOR_SETS["market_prior_year"]
    assert "clean_control_volume" in PREDICTOR_SETS["clean_pool_control"]


def test_design_version_overrides_explicit_regressors(cfg):
    assert cfg.model.design_version == "cross_product_control"
    assert cfg.model.predictors[0] == "other_products_volume"
    with pytest.raises(ValueError, match="design_version"):
        ModelConfig(design_version="v9_wishful")
    with pytest.raises(ValueError, match="design_version or an explicit"):
        ModelConfig(design_version=None, regressors=[])


def test_v1_overstates_relative_to_v4(cfg):
    """The comp-set control must show its known upward bias.

    This is the finding that justified moving off Design A: using displaced
    competitors as the control series reads their depressed bookings as weak
    market demand and inflates the lift. If a refactor ever made Design A agree
    with Design D, the displacement mechanism has stopped working.
    """
    from incrementality import PREDICTOR_SETS

    # Competitor and clean-pool designs are only legal where a genuine
    # untreated group exists, so the comparison runs under a holdout scope.
    holdout = replace(cfg, campaign_scope="partial_holdout")
    panel = _panel(cfg, seed=555)
    series = _treated_series(cfg, panel)
    lifts = {}
    for name in ("competitor_control", "clean_pool_control"):
        variant = replace(holdout, model=replace(cfg.model, design_version=name))
        lifts[name] = estimate_uplift(
            get_backend("bsts"),
            _build_design_for(series, cfg.primary_outcome, variant),
            n_bootstrap=400,
        ).relative_uplift
        assert PREDICTOR_SETS[name]

    truth = true_average_lift(panel, cfg.post_start, cfg.post_end)
    assert lifts["competitor_control"] > lifts["clean_pool_control"]
    assert lifts["competitor_control"] > truth, "Design A should overstate a known lift"


def test_prior_campaign_indicator_reduces_bias(cfg):
    """Correcting for last year's campaign must improve accuracy.

    With high repeat participation the pre-period contains a treatment
    episode. Unmodelled, it is absorbed as seasonality and eats this year's
    effect. Judged on mean absolute error across seeds, because on any single
    draw forecast noise can swamp the correction.
    """
    without = replace(cfg, model=replace(cfg.model, control_prior_campaign=False))
    errors_with, errors_without = [], []
    for seed in (606, 707, 808):
        panel = _panel(cfg, seed=seed)
        series = _treated_series(cfg, panel)
        truth = true_average_lift(panel, cfg.post_start, cfg.post_end)
        for variant, bucket in ((cfg, errors_with), (without, errors_without)):
            estimate = estimate_uplift(
                get_backend("bsts"),
                _build_design_for(series, cfg.primary_outcome, variant),
                n_bootstrap=400,
            ).relative_uplift
            bucket.append(abs(estimate - truth))

    mae_with = sum(errors_with) / len(errors_with)
    mae_without = sum(errors_without) / len(errors_without)
    assert mae_with < mae_without, (
        f"correction made things worse: {mae_with:.2%} vs {mae_without:.2%}"
    )


def test_distant_spillover_contaminates_the_control_pool(cfg):
    """Leakage past the competitive set must be visible in the control series.

    Design D's precision rests on the distant pool being untouched. This asserts the
    simulator can actually violate that assumption, which is what makes the
    sensitivity study meaningful rather than decorative.
    """
    clean = _panel(cfg, seed=707, truth=GroundTruth(distant_spillover=0.0))
    leaky = _panel(cfg, seed=707, truth=GroundTruth(distant_spillover=-0.02))

    def pool_post(panel):
        pool = panel[panel["control_role"] == "covariate"].copy()
        pool["week"] = pd.to_datetime(pool["week"])
        window = pool[(pool["week"] >= cfg.post_start) & (pool["week"] <= cfg.post_end)]
        return float(window["gbv"].sum())

    assert pool_post(leaky) < pool_post(clean), "leakage must depress the control pool"
    # Pre-period must be untouched, or this is a level shift rather than a
    # campaign-window effect.
    def pool_pre(panel):
        pool = panel[panel["control_role"] == "covariate"].copy()
        pool["week"] = pd.to_datetime(pool["week"])
        window = pool[(pool["week"] >= cfg.pre_start) & (pool["week"] <= cfg.pre_end)]
        return float(window["gbv"].sum())

    assert pool_pre(leaky) == pytest.approx(pool_pre(clean), rel=1e-9)


def test_leakage_diagnostic_reports_its_own_weakness(cfg):
    """The leakage test must advertise that it is underpowered.

    A diagnostic that cannot detect the effect size that matters is dangerous
    if it reads as a clean bill of health, so the detail string has to say so.
    """
    from incrementality.validation import control_pool_leakage_test

    panel = _panel(cfg, seed=808)
    outcome, _ = control_pool_leakage_test(panel, cfg)
    assert "underpowered" in outcome.name.lower()
    assert "cannot confirm" in outcome.detail


# ---------------------------------------------------------------------------
# Drivers of growth
# ---------------------------------------------------------------------------


def test_driver_decomposition_reconciles(cfg):
    """visits x CVR x LoS x ADR must reproduce the GBV lift."""
    from incrementality.drivers import decompose

    # Production scale. Each driver carries its own forecast error and those
    # errors compound in the product, so at small cohort size the residual is
    # dominated by noise rather than by whether the identity holds.
    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=909,
        n_treated=2_500,
        n_compset=2_500,
        n_distant=3_000,
    )
    panel = split_control_pool(simulate_panel(spec), seed=910)
    series = _treated_series(cfg, panel)
    primary = estimate_uplift(
        get_backend("bsts"),
        _build_design_for(series, cfg.primary_outcome, cfg),
        n_bootstrap=400,
    )
    result = decompose(series, cfg, primary, n_bootstrap=400)

    assert set(result.drivers) == {"visits", "cvr", "los", "adr"}
    assert result.reconciles, (
        f"drivers compose to {result.composed_lift:.2%} vs "
        f"{result.outcome_lift:.2%} (residual {result.residual:.2%})"
    )
    frame = result.to_frame()
    assert frame["share_of_movement"].sum() == pytest.approx(1.0)


def test_discount_shows_up_as_negative_rate(cfg):
    """A real discount must lift room nights more than booking value.

    If RN and GBV move together the discount never reached travellers, which
    on real data means the panel or the rate plan is wrong. This is the
    cheapest available check on that.
    """
    panel = _panel(cfg, seed=1010)
    gbv = true_average_lift(panel, cfg.post_start, cfg.post_end, outcome="gbv")
    room_nights = true_average_lift(panel, cfg.post_start, cfg.post_end, outcome="room_nights")
    adr = true_average_lift(panel, cfg.post_start, cfg.post_end, outcome="adr")
    assert adr < 0, "the discount must reduce realised rate"
    assert room_nights > gbv, "room nights must outrun booking value when rate falls"


# ---------------------------------------------------------------------------
# Relationship stability: the second modelling assumption
# ---------------------------------------------------------------------------


def test_stability_test_is_quiet_when_the_relationship_holds(cfg):
    """No drift injected, so the diagnostic must not cry wolf."""
    from incrementality.validation import relationship_stability_test

    panel = _panel(cfg, seed=2101, truth=GroundTruth(relationship_drift=0.0))
    outcome, stats = relationship_stability_test(_treated_series(cfg, panel), cfg)
    assert outcome.passed, f"false alarm: {outcome.detail}"
    # Gated on the counterfactual gap, which is the statistic that actually
    # separates stable from drifted data.
    assert stats["divergence"] < 0.045


def test_stability_test_responds_to_injected_drift(cfg):
    """Severe drift must move the gating statistic and trip the gate.

    The gate is the counterfactual gap, which measures 1.8% on stable data
    and 6.5% under injected drift. Error growth is reported alongside but
    overlaps too heavily between the two to gate on.
    """
    from incrementality.validation import relationship_stability_test

    quiet = _panel(cfg, seed=2101, truth=GroundTruth(relationship_drift=0.0))
    drifting = _panel(cfg, seed=2101, truth=GroundTruth(relationship_drift=0.60))

    _, quiet_stats = relationship_stability_test(_treated_series(cfg, quiet), cfg)
    outcome, drift_stats = relationship_stability_test(_treated_series(cfg, drifting), cfg)

    assert drift_stats["divergence"] > quiet_stats["divergence"] * 1.5, (
        f"gap barely moved: {quiet_stats['divergence']:.3f} -> "
        f"{drift_stats['divergence']:.3f}"
    )
    assert not outcome.passed, f"missed injected drift: {outcome.detail}"


def test_drift_leaves_the_true_effect_untouched(cfg):
    """Drift must change the baseline, not the campaign effect.

    Otherwise the sweep would be measuring two things at once and the bias
    attributed to drift would be uninterpretable.
    """
    truths = [GroundTruth(relationship_drift=d) for d in (0.0, 0.35)]
    lifts = [
        true_average_lift(
            _panel(cfg, seed=2202, truth=t), cfg.post_start, cfg.post_end, truth=t
        )
        for t in truths
    ]
    assert abs(lifts[0] - lifts[1]) < 0.005, (
        f"true lift moved from {lifts[0]:.2%} to {lifts[1]:.2%} under drift"
    )


def test_battery_runs_seven_tests(cfg):
    """The scorecard must contain every falsification test, not a subset."""
    from incrementality.validation import run_battery

    panel = _panel(cfg, seed=2303)
    series = _treated_series(cfg, panel)
    observed = estimate_uplift(
        get_backend("bsts"),
        _build_design_for(series, cfg.primary_outcome, cfg),
        n_bootstrap=400,
    ).relative_uplift
    report = run_battery(panel, series, cfg, observed, n_placebo=8)

    names = {t.name for t in report.tests}
    assert len(report.tests) == 7, f"expected 7 tests, got {sorted(names)}"
    assert any("Relationship stability" in n for n in names)
    assert any("leakage" in n.lower() for n in names)


def test_incrementality_build_reconciles(cfg, tmp_path):
    """Net incrementality must equal measured lift less displacement."""
    from incrementality.pipeline import run_cohort

    panel = _panel(cfg, seed=2404)
    result = run_cohort(cfg, panel, output_dir=tmp_path, n_placebo=8, make_figures=False)
    inc = result.incrementality

    # Total incrementality is the participant effect PLUS the compset effect,
    # carrying its own sign: halo adds, cannibalisation subtracts, and an
    # effect that is not distinguishable contributes nothing.
    assert inc.net_value == pytest.approx(
        inc.gross_value + inc.displacement_value, rel=1e-9
    )
    assert inc.regime in {
        "halo",
        "cannibalisation",
        "no measurable effect on the competitive set",
    }
    if inc.regime == "cannibalisation":
        assert inc.net_lift < inc.gross_lift
    elif inc.regime == "halo":
        assert inc.net_lift > inc.gross_lift
    else:
        assert inc.net_lift == pytest.approx(inc.gross_lift, rel=1e-9)
    # The net interval must be wider than the gross one, since it carries the
    # displacement estimate's uncertainty too.
    assert (inc.net_ci[1] - inc.net_ci[0]) > (inc.gross_ci[1] - inc.gross_ci[0])


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(cfg):
    """One fitted result plus its design, shared across evaluation tests."""
    from incrementality import diagnostics as dg

    panel = _panel(cfg, seed=3101)
    series = _treated_series(cfg, panel)
    design = dg.build_design_for_eval(series, cfg)
    result = estimate_uplift(
        get_backend("bsts"),
        design,
        alpha=cfg.model.alpha,
        n_bootstrap=1_000,
        materiality_threshold=cfg.materiality_threshold,
    )
    return panel, series, design, result


def test_residual_checks_pass_at_production_scale(cfg):
    """Adequacy is scale-dependent, so it must be tested at a scale that is
    actually adequate.

    With a large enough participant cohort the control series is precise
    enough to absorb the shared market factor, and the residuals come back
    white. See the companion test below for the small-cohort case, where they
    do not -- that contrast is the useful finding, not a nuisance.
    """
    from incrementality.diagnostics import build_design_for_eval, residual_checks

    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=3101,
        n_treated=2_500,
        n_compset=2_500,
        n_distant=3_000,
    )
    panel = split_control_pool(simulate_panel(spec), seed=3102)
    series = _treated_series(cfg, panel)
    result = estimate_uplift(
        get_backend("bsts"), build_design_for_eval(series, cfg), n_bootstrap=500
    )
    failed = [c.name for c in residual_checks(result) if not c.passed]
    assert not failed, f"adequacy checks failed at production scale: {failed}"


def test_residual_checks_are_computed_at_small_cohort_size(evaluated):
    """Adequacy diagnostics must be produced whatever the cohort size."""
    from incrementality.diagnostics import residual_checks

    _, _, _, result = evaluated
    checks = {c.name: c for c in residual_checks(result)}
    assert checks, "residuals were not retained"
    # Which specific check trips depends on the design and the cohort size,
    # so what is pinned is that the diagnostics are computed and reported
    # rather than silently skipped. The companion test above asserts they
    # come back clean at production scale, which is the claim that matters.
    assert {
        "Residual autocorrelation (Ljung-Box)",
        "Residual normality (Shapiro-Wilk)",
        "First-order residual correlation",
    } <= set(checks)
    assert all(np.isfinite(c.value) for c in checks.values())


def test_mde_is_reported_and_ordered(evaluated, cfg):
    """Detection probability must rise monotonically and yield a usable MDE."""
    from incrementality.diagnostics import power_curve

    _, _, _, result = evaluated
    curve, mde = power_curve(result)

    assert curve["detection_probability"].is_monotonic_increasing
    assert curve.loc[curve["true_lift"] == 0.0, "detection_probability"].iloc[0] < 0.1
    assert 0.0 < mde < 0.10, f"implausible MDE {mde:.2%}"
    # A design cannot detect an effect smaller than its materiality floor.
    assert mde >= cfg.materiality_threshold


def test_dummy_outcome_refuter_passes_on_noise(evaluated, cfg):
    """A noise outcome must not produce a significant effect."""
    from incrementality.diagnostics import refutation_suite

    _, series, _, result = evaluated
    checks = {c.name: c for c in refutation_suite(series, cfg, result, n_bootstrap=400)}
    assert checks["Dummy outcome"].passed
    assert checks["Random common cause"].passed


def test_rmspe_ratio_exceeds_placebos_for_a_real_effect(evaluated, cfg):
    """Abadie's post/pre ratio must rank the treated unit as extreme."""
    from incrementality.diagnostics import rmspe_ratio_test

    panel, _, design, _ = evaluated
    ratio, p_value, placebos = rmspe_ratio_test(panel, cfg, design, n_placebo=8)
    assert ratio > 1.0, "post-period fit should be worse than pre-period fit"
    assert placebos.size > 0
    assert p_value <= 0.25, f"treated ratio {ratio:.2f} not extreme vs placebos"


def test_predictor_value_identifies_the_control_series(evaluated, cfg):
    """The contemporaneous control must rank first on forecast value.

    Ranking is on out-of-sample error removed, not on how much a predictor
    moves the answer -- a predictor that shifts the estimate without
    improving the forecast is a warning sign, not a useful control.
    """
    from incrementality.diagnostics import predictor_value

    _, series, _, _ = evaluated
    frame = predictor_value(series, cfg)
    assert frame.iloc[0]["predictor"] == "other_products_volume"
    assert frame.iloc[0]["verdict"] == "load-bearing"
    # Ranking is on forecast error, so values must be ordered.
    assert frame["error_added_by_dropping"].is_monotonic_decreasing


def test_design_comparison_screens_out_the_weak_designs(evaluated, cfg):
    """Backtest error must rank the prior-year-only designs last.

    And -- the point of the exercise -- it must NOT be trusted to separate the
    contaminated contemporaneous design from the clean one, since the
    contamination only acts during the post window. This test pins that
    limitation so nobody later promotes backtest error to a sufficient
    criterion.
    """
    from incrementality.diagnostics import compare_designs

    _, series, _, _ = evaluated
    frame = compare_designs(series, cfg).set_index("design")
    lagged = frame.loc[["own_prior_year", "market_prior_year"], "backtest_mape"].max()
    contemporaneous = frame.loc[
        ["competitor_control", "clean_pool_control"], "backtest_mape"
    ].max()
    assert contemporaneous < lagged, "designs without a current control should fit worse"

    gap = abs(
        frame.loc["competitor_control", "backtest_mape"]
        - frame.loc["clean_pool_control", "backtest_mape"]
    )
    assert gap < 0.01, (
        "backtest error separates the contaminated and clean designs; if this "
        "ever becomes true, the contamination story needs revisiting"
    )


def test_pruning_does_not_change_the_answer(evaluated, cfg):
    """Removing no-value predictors should tighten, not move, the estimate."""
    from incrementality.diagnostics import compare_designs

    _, series, _, _ = evaluated
    frame = compare_designs(
        series, cfg, designs=("clean_pool_control", "pruned_control")
    ).set_index("design")
    full, pruned = frame.loc["clean_pool_control"], frame.loc["pruned_control"]

    assert abs(full["estimate"] - pruned["estimate"]) < 0.01, "pruning moved the answer"
    # Deliberately no assertion that pruning narrows the interval. It does at
    # production scale, where the control series is precise; at the small
    # cohort size used here the discarded market covariates still carry some
    # signal and the interval widens slightly. Pruning is a scale-dependent
    # optimisation, and pinning the scale-dependent half would make this test
    # a hostage to cohort size.
    assert pruned["backtest_mape"] < full["backtest_mape"] * 1.25


def test_halo_and_displacement_bias_in_opposite_directions(cfg):
    """The two leakage channels must push the comparison group opposite ways.

    Displacement pulls untreated sales down; the site-wide halo pushes them
    up. If both ever moved the same way, the sizing table in the README would
    be wrong and a design contaminated by both could not look unbiased by
    coincidence.
    """
    def pool_post(panel):
        pool = panel[panel["control_role"] == "covariate"].copy()
        pool["week"] = pd.to_datetime(pool["week"])
        window = pool[(pool["week"] >= cfg.post_start) & (pool["week"] <= cfg.post_end)]
        return float(window["gbv"].sum())

    clean = pool_post(_panel(cfg, seed=4101, truth=GroundTruth()))
    displaced = pool_post(
        _panel(cfg, seed=4101, truth=GroundTruth(distant_spillover=-0.02))
    )
    haloed = pool_post(_panel(cfg, seed=4101, truth=GroundTruth(marketplace_halo=0.02)))

    assert displaced < clean, "displacement must depress the comparison group"
    assert haloed > clean, "site-wide halo must lift the comparison group"


def test_halo_does_not_reach_a_lagged_comparison_series(cfg):
    """A year-lagged series must be immune to this year's campaign.

    This is the structural property that makes the market-history design the
    safe fallback under a globally simultaneous campaign.
    """
    from incrementality.simulate import aggregate_to_cohort

    def market_history(panel):
        series = aggregate_to_cohort(panel, cfg.outcomes)
        treated = series[series["cohort"] == "treated"].copy()
        treated["week"] = pd.to_datetime(treated["week"])
        window = treated[
            (treated["week"] >= cfg.post_start) & (treated["week"] <= cfg.post_end)
        ]
        return float(window["market_total_ly"].sum())

    clean = market_history(_panel(cfg, seed=4202, truth=GroundTruth()))
    haloed = market_history(
        _panel(cfg, seed=4202, truth=GroundTruth(marketplace_halo=0.02))
    )
    assert clean == pytest.approx(haloed, rel=1e-9), (
        "a prior-year series must not move when this year's campaign changes"
    )
