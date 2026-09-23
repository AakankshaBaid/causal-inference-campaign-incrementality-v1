"""Do the model's assumptions hold? Checked before any effect is reported.

The estimator rests on two assumptions that the estimate itself cannot
reveal. They are checked here, in the order a reviewer would ask about them,
and the pipeline refuses to publish a headline when they fail.

    Assumption 1  The control series were not themselves affected by the
                  campaign.
    Assumption 2  The relationship between controls and the treated series,
                  established in the pre-period, holds through the
                  post-period.

Three checks, following the diagnostic sequence recommended in the
CausalImpact documentation and the applied literature:

**1. Cohort comparability.** The treated and comparison groups must be alike
enough that one can stand in for the other -- similar scale, similar mix,
and a pre-period relationship stable enough to extrapolate. Checked on
composition and on pre-period co-movement.

**2. Predictor integrity.** Google's guidance is to reason about whether each
covariate could have been affected and to eyeball the series. That is
necessary but weak, so the formal version is used as well: *run the causal
analysis on each predictor itself*. If the campaign appears to "cause" a
change in a predictor, that predictor is contaminated and must be dropped or
replaced with its own counterfactual (Menchetti, Cipollini & Mealli, 2023).

**3. Pre-period placebo.** Move the intervention date back into a quiet
window and re-run on both the treated and the comparison cohort. A correct
model finds nothing in either, while still fitting both series well. A model
that manufactures an effect where none exists will inflate one where
something does.

A warning from the applied literature that this module is built around:
selecting controls by correlation *actively favours contaminated series*,
because a series receiving spillover tracks the treated series better than a
clean one does. Control choice is therefore justified against how the
campaign was deployed, and only then checked against the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CohortConfig
from .counterfactual import get_backend
from .inference import estimate_uplift
from .validation import TestOutcome, _design_for


@dataclass
class AssumptionReport:
    checks: list[TestOutcome] = field(default_factory=list)
    comparability: pd.DataFrame | None = None
    predictors: pd.DataFrame | None = None
    placebo: pd.DataFrame | None = None

    @property
    def all_met(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self.checks])


# ---------------------------------------------------------------------------
# 1. Are the cohorts comparable?
# ---------------------------------------------------------------------------


def cohort_comparability(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    *,
    correlation_floor: float = 0.80,
) -> tuple[TestOutcome, pd.DataFrame]:
    """Scale, mix and pre-period co-movement of treated vs comparison cohorts.

    Comparability is not about the two groups being the same size -- they
    rarely are, and scale differences are absorbed by the model. It is about
    whether they move together. The statistic that matters is the correlation
    of their weekly changes across the pre-period: if they tracked each other
    before the campaign, one can stand in for the other during it.
    """
    from .simulate import aggregate_to_cohort

    work = panel.copy()
    work["week"] = pd.to_datetime(work["week"])
    series = aggregate_to_cohort(work, cfg.outcomes)
    series["week"] = pd.to_datetime(series["week"])
    pre = series[(series["week"] >= cfg.pre_start) & (series["week"] <= cfg.pre_end)]

    rows = []
    for cohort, chunk in pre.groupby("cohort"):
        props = work.loc[work["cohort"] == cohort, "property_id"].nunique()
        rows.append(
            {
                "cohort": cohort,
                "properties": props,
                "weekly_volume_musd": chunk[cfg.primary_outcome].mean() / 1e6,
                "volume_per_property_usd": chunk[cfg.primary_outcome].mean() / max(props, 1),
                "weekly_volatility": float(
                    np.std(np.diff(np.log(chunk[cfg.primary_outcome].to_numpy())), ddof=1)
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("cohort")

    treated = pre[pre["cohort"] == "treated"].set_index("week")[cfg.primary_outcome]
    correlations = {}
    for cohort in pre["cohort"].unique():
        if cohort == "treated":
            continue
        other = pre[pre["cohort"] == cohort].set_index("week")[cfg.primary_outcome]
        joined = pd.concat([treated, other], axis=1).dropna()
        if len(joined) < 20:
            continue
        growth = np.diff(np.log(joined.to_numpy()), axis=0)
        correlations[cohort] = float(np.corrcoef(growth[:, 0], growth[:, 1])[0, 1])
    frame["pre_period_growth_corr"] = pd.Series(correlations)

    best = max(correlations.values()) if correlations else float("nan")
    return (
        TestOutcome(
            name="Cohorts are comparable",
            passed=bool(best >= correlation_floor),
            statistic=best,
            threshold=correlation_floor,
            detail=(
                "pre-period weekly growth correlation with the treated cohort: "
                + ", ".join(f"{k} {v:.2f}" for k, v in sorted(correlations.items()))
            ),
        ),
        frame,
    )


# ---------------------------------------------------------------------------
# 2. Are the predictors themselves clean?
# ---------------------------------------------------------------------------


def predictor_integrity(
    series: pd.DataFrame,
    cfg: CohortConfig,
    *,
    n_bootstrap: int = 500,
    tolerance: float = 0.02,
    fit_limit: float = 0.055,
    placebo_shift: int = 8,
) -> tuple[TestOutcome, pd.DataFrame]:
    """Run the causal analysis on each predictor and look for an effect.

    This is the formal version of the recommended visual sanity check. A
    predictor that shows a significant, material "effect" over the campaign
    window was itself moved by the campaign, which violates Assumption 1 and
    biases the headline -- downward, when the contamination is a demand halo
    that inflates the control.

    Each predictor is tested using the remaining predictors as its own
    controls, so the test asks whether that series moved *beyond what the
    others explain*, rather than whether it moved at all.

    The test is **placebo-calibrated**. A predictor is compared against its
    own behaviour at a fake intervention in a quiet pre-period window. Series
    with different demand elasticities diverge from one another for reasons
    that have nothing to do with a campaign, and an uncalibrated version of
    this test flagged a predictor the data generator had built to be clean.
    Only an effect materially larger in the real window than in the placebo
    window counts as contamination.

    Multiplicity is corrected with Holm-Bonferroni. Testing five predictors at
    10% each produces a false alarm roughly four times in ten, and an
    uncorrected version of this check flagged a predictor that the data
    generator had constructed to be clean. A contamination test that cries
    wolf gets switched off, which is worse than not having one.
    """
    predictors = list(cfg.model.predictors)
    rows = []
    for target in predictors:
        # A series lagged a full year cannot be moved by this year's campaign.
        # That is a structural property, not an empirical question, so testing
        # it wastes power and invites a false accusation from forecast noise.
        if target.endswith("_ly"):
            rows.append(
                {
                    "predictor": target,
                    "apparent_effect": 0.0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": 1.0,
                    "fit_error": 0.0,
                    "material": False,
                    "structural": True,
                }
            )
            continue
        others = [p for p in predictors if p != target]
        if not others or target not in series.columns:
            continue
        try:
            design = _design_for(series, target, cfg, regressors=others)
            result = estimate_uplift(
                get_backend(cfg.model.backend),
                design,
                alpha=cfg.model.alpha,
                n_bootstrap=n_bootstrap,
            )
            placebo_design = _design_for(
                series,
                target,
                cfg,
                regressors=others,
                activation=_quiet_placebo_week(cfg, placebo_shift),
            )
            placebo_effect = abs(
                estimate_uplift(
                    get_backend(cfg.model.backend),
                    placebo_design,
                    alpha=cfg.model.alpha,
                    n_bootstrap=max(n_bootstrap // 2, 200),
                ).relative_uplift
            )
        except Exception as exc:  # a predictor that cannot be modelled is a flag
            rows.append(
                {
                    "predictor": target,
                    "apparent_effect": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": 0.0,
                    "material": True,
                    "note": f"untestable ({exc})",
                }
            )
            continue
        rows.append(
            {
                "predictor": target,
                "apparent_effect": result.relative_uplift,
                "ci_low": result.ci_relative[0],
                "ci_high": result.ci_relative[1],
                "p_value": result.p_value_bootstrap,
                "fit_error": result.diagnostics.get("backtest_mape", np.nan),
                "placebo_effect": placebo_effect,
                "material": bool(
                    abs(result.relative_uplift) >= tolerance
                    and abs(result.relative_uplift) > placebo_effect + tolerance
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return (
            TestOutcome("Predictors unaffected by the campaign", True, 0.0, tolerance,
                        "no predictors available to test"),
            frame,
        )

    frame["structural"] = frame.get("structural", pd.Series(False, index=frame.index))
    frame["structural"] = frame["structural"].fillna(False).astype(bool)

    # Holm-Bonferroni across the predictor family.
    order = frame["p_value"].rank(method="first").astype(int)
    n = len(frame)
    frame["holm_threshold"] = cfg.model.alpha / (n - order + 1)
    # A contamination flag is only meaningful when the model can forecast the
    # predictor in the first place. If it cannot, the "effect" is the model's
    # own forecast error and says nothing about the campaign, so the verdict
    # is "untestable" rather than a false accusation.
    fit_ok = frame["fit_error"].fillna(1.0) <= fit_limit
    flagged = frame["material"] & (frame["p_value"] <= frame["holm_threshold"])
    frame["verdict"] = np.where(
        frame["structural"],
        "structurally clean (lagged)",
        np.where(
            flagged & fit_ok,
            "CONTAMINATED",
            np.where(
                ~fit_ok, "untestable (model cannot forecast it)",
                np.where(frame["material"], "large but not significant", "clean"),
            ),
        ),
    )
    contaminated = frame[frame["verdict"] == "CONTAMINATED"]
    untestable = frame[frame["verdict"].str.startswith("untestable")]
    # A check that passes because nothing could be tested has not passed.
    # Reporting that as a clean bill of health is how an assumption gets waved
    # through, so it is surfaced as inconclusive instead.
    testable = frame[~frame["structural"]]
    inconclusive = len(untestable) > max(len(testable), 1) / 2
    worst = (
        float(contaminated["apparent_effect"].abs().max())
        if not contaminated.empty
        else 0.0
    )
    if inconclusive:
        detail = (
            f"INCONCLUSIVE: {len(untestable)} of {len(frame)} predictors cannot be "
            f"forecast well enough to test for contamination -- absence of a flag "
            f"here is not evidence that they are clean"
        )
    elif contaminated.empty:
        detail = f"{len(frame)} predictors tested; none show a material campaign effect"
    else:
        detail = "CONTAMINATED: " + ", ".join(
            f"{r.predictor} {r.apparent_effect:+.1%}" for r in contaminated.itertuples()
        )

    return (
        TestOutcome(
            name="Predictors unaffected by the campaign",
            passed=bool(contaminated.empty and not inconclusive),
            statistic=worst,
            threshold=tolerance,
            detail=detail,
        ),
        frame,
    )


# ---------------------------------------------------------------------------
# 3. Does a quiet period read as quiet, for both cohorts?
# ---------------------------------------------------------------------------


def _quiet_placebo_week(cfg: CohortConfig, preferred: int) -> pd.Timestamp:
    """Pick a placebo activation whose window avoids known seasonal spikes.

    A placebo landing on the new-year booking surge tests the model's ability
    to handle a holiday spike, not its tendency to invent effects. Candidate
    shifts are screened so the fake post-window contains none of the ISO weeks
    the config flags as holidays, while leaving enough history behind it to
    fit and backtest.
    """
    holiday_weeks = {w for weeks in cfg.model.holiday_iso_weeks.values() for w in weeks}
    for shift in (preferred, 8, 10, 12, 16, 20, 24, 26, 30, 34, 38, 42):
        start = cfg.activation_week - pd.Timedelta(weeks=shift)
        window = pd.date_range(
            start + pd.Timedelta(weeks=cfg.transition_weeks),
            start + pd.Timedelta(weeks=cfg.post_weeks - 1),
            freq="W-MON",
        )
        # The weeks immediately BEFORE the fake activation matter as much as
        # the window itself: a local-level model anchors on the end of its
        # training data, so a placebo whose pre-period finishes inside a
        # seasonal spike starts from a distorted level and reports an effect
        # that is really just the spike unwinding.
        run_up = pd.date_range(
            start - pd.Timedelta(weeks=4), start - pd.Timedelta(weeks=1), freq="W-MON"
        )
        touched = set(window.isocalendar().week) | set(run_up.isocalendar().week)
        if not touched & holiday_weeks:
            return start
    return cfg.activation_week - pd.Timedelta(weeks=preferred)


def pre_period_placebo(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    *,
    shift_weeks: int = 16,
    tolerance: float = 0.035,
    n_bootstrap: int = 500,
) -> tuple[TestOutcome, pd.DataFrame]:
    """Fake the intervention inside a quiet window, for treated and comparison.

    The shift is 16 weeks rather than a full season: the placebo still needs
    enough history behind it to fit and backtest, and taking too much out of
    the pre-period leaves the model with nothing to learn from.

    The tolerance is calibrated on stable data rather than chosen. Measured
    over seven simulated campaigns with no effect present, the placebo reads
    a mean of -0.9% with a standard deviation of 1.7%, so a 2% gate rejects
    roughly four runs in ten on noise alone. The gate is set at 3.5%, a
    little over two standard deviations of the in-control distribution.

    The cost is stated rather than buried: this check can only catch a
    fabricated effect larger than about 3.5%. It still catches the failure
    that mattered -- a peak-season campaign read -4.8% here -- but it is not
    evidence against a small leak.

    Both cohorts must come back null. The treated cohort returning null shows
    the pipeline does not manufacture effects; the comparison cohort
    returning null shows the same for the series the counterfactual leans on.
    Fit quality is reported alongside, because a null produced by a model that
    fits nothing is not reassurance.
    """
    from .simulate import aggregate_to_cohort

    fake = _quiet_placebo_week(cfg, shift_weeks)
    series = aggregate_to_cohort(panel, cfg.outcomes)
    rows = []
    for cohort in ("treated", "compset"):
        chunk = series[series["cohort"] == cohort].reset_index(drop=True)
        if chunk.empty:
            continue
        try:
            design = _design_for(
                chunk, cfg.primary_outcome, cfg, activation=fake
            )
            result = estimate_uplift(
                get_backend(cfg.model.backend),
                design,
                alpha=cfg.model.alpha,
                n_bootstrap=n_bootstrap,
            )
        except Exception as exc:
            rows.append({"cohort": cohort, "apparent_effect": np.nan, "note": str(exc)})
            continue
        rows.append(
            {
                "cohort": cohort,
                "apparent_effect": result.relative_uplift,
                "ci_low": result.ci_relative[0],
                "ci_high": result.ci_relative[1],
                "p_value": result.p_value_bootstrap,
                "fit_error": result.diagnostics.get("backtest_mape", np.nan),
                "significant": bool(result.is_significant),
            }
        )

    frame = pd.DataFrame(rows)
    worst = float(frame["apparent_effect"].abs().max()) if not frame.empty else np.nan
    return (
        TestOutcome(
            name="Quiet period reads as quiet",
            passed=bool(worst <= tolerance),
            statistic=worst,
            threshold=tolerance,
            detail=(
                f"placebo intervention {shift_weeks} weeks early: "
                + "; ".join(
                    f"{r.cohort} {r.apparent_effect:+.2%}" for r in frame.itertuples()
                )
            ),
        ),
        frame,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def check_assumptions(
    panel: pd.DataFrame, series: pd.DataFrame, cfg: CohortConfig, *, quick: bool = False
) -> AssumptionReport:
    """Run all three assumption checks. Called before any effect is reported."""
    report = AssumptionReport()
    n_boot = 300 if quick else 600

    comparability, frame = cohort_comparability(panel, cfg)
    report.checks.append(comparability)
    report.comparability = frame

    integrity, predictors = predictor_integrity(series, cfg, n_bootstrap=n_boot)
    report.checks.append(integrity)
    report.predictors = predictors

    placebo, placebo_frame = pre_period_placebo(panel, cfg, n_bootstrap=n_boot)
    report.checks.append(placebo)
    report.placebo = placebo_frame

    return report
