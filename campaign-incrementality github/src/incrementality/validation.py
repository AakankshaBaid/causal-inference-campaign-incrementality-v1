"""The falsification battery.

A counterfactual estimate cannot be validated against ground truth, so it has
to be validated by trying to break it. Five tests, each answering a specific
objection a sceptical stakeholder will raise:

=========================  ==================================================
Objection                  Test
=========================  ==================================================
"Your model just can't     Rolling-origin backtest. Forecast eight weeks
 forecast this series."    ahead, repeatedly, inside the quiet pre-period.
"You'd find a lift even    In-time placebo. Move the intervention date back
 if nothing happened."     into the pre-period. The answer must be null.
"Any random group of       In-space placebo. Treat untreated cohorts as
 properties looks like     pseudo-treated and build the null distribution
 this."                    the real estimate has to beat.
"You moved bookings from   Displacement test. Estimate the effect on the
 the comp set, you didn't  untreated comp set. Negative effect there is
 create them."             cannibalisation, and it nets off the headline.
"You picked the window     Sensitivity sweep. Re-estimate across pre-window
 that gave the answer."    lengths and regressor subsets.
=========================  ==================================================

A run where any of these fails is not a run with a caveat. It is a number
that does not get presented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PREDICTOR_SETS, CohortConfig
from .counterfactual import get_backend
from .features import build_design
from .inference import estimate_uplift, rolling_origin_errors
from .simulate import aggregate_to_cohort

logger = logging.getLogger(__name__)


@dataclass
class TestOutcome:
    """One falsification test, reduced to something a slide can carry."""

    name: str
    passed: bool
    statistic: float
    threshold: float
    detail: str

    def as_row(self) -> dict[str, object]:
        return {
            "test": self.name,
            "result": "PASS" if self.passed else "FAIL",
            "statistic": round(self.statistic, 4),
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    tests: list[TestOutcome] = field(default_factory=list)
    placebo_distribution: np.ndarray | None = None
    sensitivity: pd.DataFrame | None = None
    displacement_relative: float = 0.0
    control_pool_leakage: float = 0.0
    stability: dict[str, float] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(test.passed for test in self.tests)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([test.as_row() for test in self.tests])


def _design_for(
    series: pd.DataFrame,
    outcome: str,
    cfg: CohortConfig,
    *,
    activation: pd.Timestamp | None = None,
    regressors: list[str] | None = None,
    pre_weeks: int | None = None,
):
    activation = activation or cfg.activation_week
    pre_weeks = pre_weeks or cfg.pre_weeks
    regressors = regressors if regressors is not None else cfg.model.predictors
    pre = (activation - pd.Timedelta(weeks=pre_weeks), activation - pd.Timedelta(weeks=1))
    post = (
        activation + pd.Timedelta(weeks=cfg.transition_weeks),
        activation + pd.Timedelta(weeks=cfg.post_weeks - 1),
    )
    prior_window = None
    if cfg.model.control_prior_campaign:
        # Anchored on the REAL activation week, never on a placebo's fake one.
        # Last year's campaign happened when it happened; sliding the dummy
        # along with a placebo date leaves the real prior episode unmodelled
        # inside the placebo's pre-period, which is enough to make a clean
        # placebo read as a 2% effect.
        prior_start = cfg.activation_week - pd.Timedelta(weeks=52)
        prior_window = (
            prior_start,
            prior_start + pd.Timedelta(weeks=cfg.model.prior_campaign_weeks - 1),
        )
    return build_design(
        series,
        outcome=outcome,
        regressors=regressors,
        pre_window=pre,
        post_window=post,
        prior_campaign_window=prior_window,
        prior_campaign_shape=(
            cfg.model.prior_campaign_ramp_weeks,
            cfg.model.prior_campaign_decay,
        ),
        log_transform=cfg.model.log_transform,
        annual_fourier_terms=cfg.model.annual_fourier_terms,
        holiday_iso_weeks=cfg.model.holiday_iso_weeks,
    )


def _quick_relative_uplift(design, backend_name: str) -> float:
    """Single fit, no bootstrap -- used where we need hundreds of fits."""
    model = get_backend(backend_name)
    fit = model.fit_predict(design.y_pre, design.X_pre, design.X_post)
    actual = design.inverse(design.y_post).sum()
    counterfactual = design.inverse(fit.mean).sum()
    return float(actual / counterfactual - 1.0)


# ---------------------------------------------------------------------------
# 1. Rolling-origin backtest
# ---------------------------------------------------------------------------


def backtest(design, backend_name: str, horizon: int, *, mape_threshold: float = 0.08) -> TestOutcome:
    """Can the model forecast this series at the horizon we care about?"""
    model = get_backend(backend_name)
    errors = rolling_origin_errors(model, design, horizon)
    # In log space, expm1 of the error is the proportional miss.
    pct_errors = np.expm1(errors) if design.log_transform else errors
    mape = float(np.mean(np.abs(pct_errors)))
    return TestOutcome(
        name="Rolling-origin backtest",
        passed=mape <= mape_threshold,
        statistic=mape,
        threshold=mape_threshold,
        detail=(
            f"{errors.shape[0]} origins, {horizon}-week horizon, "
            f"mean absolute error {mape:.2%}"
        ),
    )


# ---------------------------------------------------------------------------
# 2. In-time placebo
# ---------------------------------------------------------------------------


def placebo_in_time(
    series: pd.DataFrame,
    cfg: CohortConfig,
    *,
    shift_weeks: int = 12,
    threshold: float = 0.02,
) -> TestOutcome:
    """Pretend the campaign launched earlier, when nothing happened."""
    from .assumptions import _quiet_placebo_week

    fake_activation = _quiet_placebo_week(cfg, shift_weeks)
    shift_weeks = int((cfg.activation_week - fake_activation).days // 7)
    design = _design_for(series, cfg.primary_outcome, cfg, activation=fake_activation)
    relative = _quick_relative_uplift(design, cfg.model.backend)
    return TestOutcome(
        name="In-time placebo",
        passed=abs(relative) <= threshold,
        statistic=relative,
        threshold=threshold,
        detail=(
            f"fake activation {fake_activation.date()} ({shift_weeks} weeks early) "
            f"returns {relative:+.2%}"
        ),
    )


# ---------------------------------------------------------------------------
# 3. In-space placebo
# ---------------------------------------------------------------------------


def placebo_in_space(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    observed_relative: float,
    *,
    n_placebo: int = 40,
    seed: int = 11,
) -> tuple[TestOutcome, np.ndarray]:
    """Build the null distribution from untreated cohorts of the same size.

    Each pseudo-cohort is a random draw of untreated properties, aggregated
    and run through the identical pipeline. The share of pseudo-cohorts whose
    absolute effect matches or exceeds the observed one is a permutation
    p-value that makes no distributional assumptions at all.
    """
    rng = np.random.default_rng(seed)
    pool_mask = (
        panel["control_role"] == "placebo"
        if "control_role" in panel.columns
        else panel["cohort"] != "treated"
    )
    controls = panel[pool_mask]
    donor_ids = controls["property_id"].unique()
    n_treated = panel.loc[panel["cohort"] == "treated", "property_id"].nunique()
    draw_size = min(n_treated, len(donor_ids) // 2)
    if draw_size < 20:
        raise ValueError("placebo pool too small for an in-space placebo")

    placebo_effects = []
    for _ in range(n_placebo):
        chosen = rng.choice(donor_ids, size=draw_size, replace=False)
        subset = panel[
            panel["property_id"].isin(chosen)
            | (panel.get("control_role", panel["cohort"]) == "covariate")
        ].copy()
        subset.loc[subset["property_id"].isin(chosen), "cohort"] = "placebo"
        series = aggregate_to_cohort(subset, cfg.outcomes)
        series = series[series["cohort"] == "placebo"].reset_index(drop=True)
        try:
            design = _design_for(series, cfg.primary_outcome, cfg)
            placebo_effects.append(_quick_relative_uplift(design, cfg.model.backend))
        except Exception:  # a failed placebo fit is information, not a crash
            continue

    effects = np.asarray(placebo_effects, dtype=float)
    if effects.size == 0:
        raise RuntimeError("every placebo fit failed; investigate the panel")
    n_extreme = int(np.sum(np.abs(effects) >= abs(observed_relative)))
    p_value = (n_extreme + 1) / (effects.size + 1)
    return (
        TestOutcome(
            name="In-space placebo",
            passed=p_value <= 0.10,
            statistic=p_value,
            threshold=0.10,
            detail=(
                f"{effects.size} untreated cohorts of {draw_size} properties; "
                f"null sd {effects.std(ddof=1):.2%}; observed "
                f"{observed_relative:+.2%} ranks {n_extreme + 1}/{effects.size + 1}"
            ),
        ),
        effects,
    )


# ---------------------------------------------------------------------------
# 4. Displacement / cannibalisation
# ---------------------------------------------------------------------------


def displacement_test(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    *,
    threshold: float = -0.05,
) -> tuple[TestOutcome, float]:
    """Estimate the campaign's effect on the untreated comp set.

    If discounted properties won share from their direct competitors rather
    than growing the market, the comp set shows a negative effect over exactly
    the same weeks. That amount is not incremental and has to be netted off
    before anyone books the value.
    """
    compset_mask = panel["cohort"] == "compset"
    if not compset_mask.any():
        compset_mask = panel["cohort"] != "treated"
    subset = panel[compset_mask | (panel.get("control_role", panel["cohort"]) == "covariate")]
    series = aggregate_to_cohort(subset, cfg.outcomes)
    series = series[series["cohort"] == panel.loc[compset_mask, "cohort"].iloc[0]].reset_index(
        drop=True
    )
    design = _design_for(series, cfg.primary_outcome, cfg)
    relative = _quick_relative_uplift(design, cfg.model.backend)
    return (
        TestOutcome(
            name="Comp-set displacement",
            passed=relative >= threshold,
            statistic=relative,
            threshold=threshold,
            detail=(
                f"untreated comp set moves {relative:+.2%} over the same weeks; "
                + (
                    "consistent with market growth"
                    if relative >= 0
                    else "netted off the headline as displacement"
                )
            ),
        ),
        relative,
    )


# ---------------------------------------------------------------------------
# 5. Control-pool leakage
# ---------------------------------------------------------------------------


def control_pool_leakage_test(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    *,
    gross_threshold: float = -0.05,
) -> tuple[TestOutcome, float]:
    """Diagnostic: is the control series itself touched by the campaign?

    The Design D design uses untreated properties outside participants' competitive
    sets as a contemporaneous control. That is valid only if the campaign does
    not reach them, and destination substitution means it might: a discount in
    one market can pull demand from another.

    The test cannot reference the treated cohort or the comp set, since both
    are affected. It uses the only predictors this year's campaign cannot
    move -- market demand signals and marketplace-wide prior-year production,
    the Design C predictor set -- and asks whether the control pool underperformed
    them over the campaign weeks.

    **This diagnostic is deliberately reported as underpowered.** Estimating
    anything without a contemporaneous control series carries roughly 6-7%
    forecast error over a seven-week aggregate, so the minimum leakage this
    can reliably detect is far larger than the 1-2% that would actually
    matter. It is therefore a gate against *gross* contamination, not
    evidence that the control pool is clean, and it reports its own detectable
    floor so nobody mistakes a pass for a guarantee.

    The real protection is elsewhere: `studies/assumption_sensitivity.py`
    quantifies how much bias a given level of leakage buys, and the pipeline
    reports a Design C cross-check whose validity does not depend on this
    assumption at all. Absence of evidence is not evidence of absence, and
    the honest response to an untestable assumption is to price it rather
    than to assert it.
    """
    pool_mask = (
        panel["control_role"] == "covariate"
        if "control_role" in panel.columns
        else panel["cohort"] == "distant"
    )
    if not pool_mask.any():
        raise ValueError("no control pool to test for leakage")

    subset = panel[pool_mask].copy()
    subset["cohort"] = "control_pool"
    series = aggregate_to_cohort(subset, cfg.outcomes)
    series = series[series["cohort"] == "control_pool"].reset_index(drop=True)

    design = _design_for(
        series,
        cfg.primary_outcome,
        cfg,
        regressors=PREDICTOR_SETS["market_prior_year"],
    )
    model = get_backend(cfg.model.backend)
    result = estimate_uplift(
        model,
        design,
        alpha=cfg.model.alpha,
        n_bootstrap=400,
        block_length=cfg.model.block_length,
        backend_name=cfg.model.backend,
    )
    relative = result.relative_uplift
    detectable = (result.ci_relative[1] - result.ci_relative[0]) / 2.0
    uses_clean_control = "clean_control_volume" in cfg.model.predictors

    return (
        TestOutcome(
            name="Control-pool leakage (underpowered)",
            passed=relative >= gross_threshold,
            statistic=relative,
            threshold=gross_threshold,
            detail=(
                f"control pool reads {relative:+.1%} against unaffected demand, "
                f"detectable only to +/-{detectable:.1%} -- rules out gross "
                f"contamination, cannot confirm the 1-2% that would matter"
                + (
                    "; see the Design C cross-check and assumption_sensitivity study"
                    if uses_clean_control
                    else "; design does not use the pool as a control"
                )
            ),
        ),
        relative,
    )


# ---------------------------------------------------------------------------
# 6. Relationship stability (the second modelling assumption)
# ---------------------------------------------------------------------------


def relationship_stability_test(
    series: pd.DataFrame,
    cfg: CohortConfig,
    *,
    divergence_threshold: float = 0.045,
    error_growth_threshold: float = 1.75,
) -> tuple[TestOutcome, dict[str, float]]:
    """Does the pre-period relationship still hold over the campaign window?

    Every counterfactual design assumes two things. The first -- that the
    control series is untouched by the campaign -- gets all the attention.
    The second is that the relationship between control and target, fitted on
    the pre-period, still applies during the post-period. It is assumed far
    more often than it is checked, and when it fails the failure is silent:
    the counterfactual is projected from a relationship that has expired, and
    the whole error lands in the effect.

    The relationship cannot be observed during the post-period -- that is the
    entire problem. What *can* be observed is whether it has been stable
    historically, and a relationship that drifted through the pre-period is
    not one to extrapolate eight weeks past it. Two statistics:

    **Counterfactual divergence.** Fit the model on the first two-thirds of
    the pre-period and again on the last two-thirds, then project both onto
    the campaign window. Overlapping windows are used rather than clean halves
    so each fit keeps enough observations to be stable while still differing
    in what it has seen. If the coefficients are stable, the two
    counterfactuals agree; if they have drifted, the gap between them is a
    direct lower bound on the error the drift contributes -- and it is
    expressed in the units that matter, as a share of the counterfactual,
    rather than as an abstract test statistic.

    **Forecast-error growth.** Rolling-origin errors are split into the older
    and more recent halves of the pre-period. A model whose accuracy is
    decaying is being told, by its own track record, that the relationship is
    going stale.

    **Divergence gates the result; error growth is context.** Both statistics
    were calibrated on simulated campaigns, seven with no drift and seven
    with severe drift injected:

    ====================  ==============  ==============  ===============
    Statistic             Stable mean     Stable 90th     Drifted mean
    ====================  ==============  ==============  ===============
    Counterfactual gap    1.8%            3.5%            6.5%
    Error growth          1.01            1.50            1.59
    ====================  ==============  ==============  ===============

    The counterfactual gap separates cleanly -- roughly 3.6x between stable
    and drifted, with a stable maximum of 3.8% sitting well below the drifted
    mean. Its gate is set at 4.5%, just above anything observed on stable
    data. Error growth barely separates: stable runs reach 1.74 while drifted
    runs average 1.59, so the distributions overlap across most of their
    range and no threshold buys power without a heavy false-alarm rate. It is
    reported for context and does not gate.

    An earlier version of this docstring asserted the reverse -- that
    divergence could not discriminate and error growth could. Re-measuring
    showed the opposite, and the gate has been moved accordingly. Either way
    the diagnostic is a warning light rather than proof: even the gating
    statistic misses drift in a meaningful share of runs.

    Calibrating a diagnostic threshold on in-control data is standard
    practice and differs in kind from tuning a reported interval -- it
    changes when an analyst is warned, not what the estimate claims.
    Recalibrate on your own pre-period before relying on the defaults.

    Failing this test does not invalidate the estimate on its own. It says the
    counterfactual is resting on a relationship the data says is moving, and
    that the interval understates the risk -- because the conformal interval
    is built from historical forecast error and therefore prices the past
    relationship, not a drifting one.
    """
    design = _design_for(series, cfg.primary_outcome, cfg)
    n_pre = len(design.y_pre)
    window = max(int(round(n_pre * 2 / 3)), 40)
    if n_pre < window + 8:
        raise ValueError(
            f"need more pre-period weeks to split for a stability test; have {n_pre}"
        )

    model = get_backend(cfg.model.backend)
    early = model.fit_predict(design.y_pre[:window], design.X_pre[:window], design.X_post)
    late = model.fit_predict(design.y_pre[-window:], design.X_pre[-window:], design.X_post)

    cf_early = float(design.inverse(early.mean).sum())
    cf_late = float(design.inverse(late.mean).sum())
    divergence = abs(cf_early - cf_late) / cf_late if cf_late else float("nan")

    errors = rolling_origin_errors(model, design, len(design.y_post))
    pct = np.abs(np.expm1(errors) if design.log_transform else errors)
    split = max(len(pct) // 2, 1)
    older, recent = float(pct[:split].mean()), float(pct[split:].mean())
    error_growth = recent / older if older > 0 else float("nan")

    # Error growth is the gate; divergence is reported alongside it.
    passed = divergence <= divergence_threshold
    divergence_note = (
        "" if error_growth <= error_growth_threshold
        else " (error growth elevated -- context only; this statistic overlaps "
        "heavily between stable and drifted data and does not gate)"
    )
    return (
        TestOutcome(
            name="Relationship stability",
            passed=bool(passed),
            statistic=float(divergence),
            threshold=divergence_threshold,
            detail=(
                f"forecast error {error_growth:.2f}x from older to recent origins "
                f"({older:.2%} to {recent:.2%}); early vs late pre-period "
                f"counterfactuals diverge {divergence:.2%}{divergence_note}"
            ),
        ),
        {
            "divergence": float(divergence),
            "error_growth": error_growth,
            "counterfactual_early": cf_early,
            "counterfactual_late": cf_late,
        },
    )


# ---------------------------------------------------------------------------
# 7. Specification sensitivity
# ---------------------------------------------------------------------------


def sensitivity_sweep(
    series: pd.DataFrame,
    cfg: CohortConfig,
    observed_relative: float,
    *,
    pre_window_options: tuple[int, ...] = (52, 78, 104),
    spread_threshold: float = 0.025,
) -> tuple[TestOutcome, pd.DataFrame]:
    """Re-estimate under alternative windows, regressor sets and seasonality.

    Not every variant is evidence. The recovery study in ``studies/``
    established that dropping the comp-set control series raises RMSE by
    roughly eightfold and fails to detect 38% of genuinely effective
    campaigns -- it is a worse estimator, not a rival reading of the data.
    Counting its answer in a stability spread would conflate "this estimate is
    fragile" with "a model we have already rejected disagrees".

    So variants are tagged. The pass/fail criterion applies to the
    specification family the study validated; rejected variants are still
    computed and reported, because hiding them would be the more common and
    worse sin.
    """
    rows = []
    control_series = "clean_control_volume"

    for pre_weeks in pre_window_options:
        try:
            design = _design_for(series, cfg.primary_outcome, cfg, pre_weeks=pre_weeks)
            rows.append(
                {
                    "variant": f"pre-window {pre_weeks}w",
                    "family": "window",
                    "defensible": True,
                    "relative_uplift": _quick_relative_uplift(design, cfg.model.backend),
                }
            )
        except ValueError:
            continue

    for dropped in cfg.model.predictors:
        remaining = [r for r in cfg.model.predictors if r != dropped]
        if not remaining:
            continue
        design = _design_for(series, cfg.primary_outcome, cfg, regressors=remaining)
        rows.append(
            {
                "variant": f"drop {dropped}",
                "family": "regressors",
                # Dropping the control series is a rejected estimator, not an
                # alternative specification. See the module docstring.
                "defensible": dropped != control_series,
                "relative_uplift": _quick_relative_uplift(design, cfg.model.backend),
            }
        )

    rows.append(
        {
            "variant": "no seasonality terms",
            "family": "seasonality",
            "defensible": True,
            "relative_uplift": _quick_relative_uplift(
                build_design(
                    series,
                    outcome=cfg.primary_outcome,
                    regressors=cfg.model.predictors,
                    pre_window=(cfg.pre_start, cfg.pre_end),
                    post_window=(cfg.post_start, cfg.post_end),
                    log_transform=cfg.model.log_transform,
                    annual_fourier_terms=0,
                    holiday_iso_weeks=cfg.model.holiday_iso_weeks,
                ),
                cfg.model.backend,
            ),
        }
    )

    frame = pd.DataFrame(rows)
    frame["delta_vs_headline"] = frame["relative_uplift"] - observed_relative

    ok = frame[frame["defensible"]]
    spread = float(ok["relative_uplift"].max() - ok["relative_uplift"].min())

    rejected = frame[~frame["defensible"]]
    note = ""
    if not rejected.empty:
        worst = rejected.loc[rejected["delta_vs_headline"].abs().idxmax()]
        note = (
            f"; rejected variant '{worst['variant']}' reads "
            f"{worst['relative_uplift']:+.2%} "
            f"({worst['delta_vs_headline']:+.2%} vs headline), reported but not "
            f"counted -- see studies/recovery_study.py"
        )

    return (
        TestOutcome(
            name="Specification sensitivity",
            passed=spread <= spread_threshold,
            statistic=spread,
            threshold=spread_threshold,
            detail=(
                f"{len(ok)} validated variants span {ok['relative_uplift'].min():+.2%} "
                f"to {ok['relative_uplift'].max():+.2%} ({spread:.2%} wide){note}"
            ),
        ),
        frame,
    )


def run_battery(
    panel: pd.DataFrame,
    series: pd.DataFrame,
    cfg: CohortConfig,
    observed_relative: float,
    *,
    n_placebo: int = 40,
) -> ValidationReport:
    """Run every test and collect the scorecard."""
    report = ValidationReport()
    design = _design_for(series, cfg.primary_outcome, cfg)

    report.tests.append(backtest(design, cfg.model.backend, cfg.n_post_weeks))
    report.tests.append(placebo_in_time(series, cfg))

    space_test, effects = placebo_in_space(
        panel, cfg, observed_relative, n_placebo=n_placebo
    )
    report.tests.append(space_test)
    report.placebo_distribution = effects

    disp_test, disp_relative = displacement_test(panel, cfg)
    report.tests.append(disp_test)
    report.displacement_relative = disp_relative

    try:
        leak_test, leak_relative = control_pool_leakage_test(panel, cfg)
        report.tests.append(leak_test)
        report.control_pool_leakage = leak_relative
    except ValueError as exc:
        logger.warning("control-pool leakage test skipped: %s", exc)

    try:
        stability_test, stability_stats = relationship_stability_test(series, cfg)
        report.tests.append(stability_test)
        report.stability = stability_stats
    except ValueError as exc:
        logger.warning("relationship stability test skipped: %s", exc)

    sens_test, sens_frame = sensitivity_sweep(series, cfg, observed_relative)
    report.tests.append(sens_test)
    report.sensitivity = sens_frame

    return report
