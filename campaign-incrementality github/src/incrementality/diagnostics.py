"""Model evaluation: is this the right model, and is it good enough?

The falsification battery in `validation.py` asks whether *this estimate* can
be broken. This module asks two prior questions that decide whether the
estimate should have been attempted at all:

1. **Is the model adequate?** A counterfactual is a forecast, so the standard
   forecasting adequacy checks apply: residuals should look like white noise,
   out-of-sample accuracy should be acceptable at the horizon that matters,
   and the specification should not be one of several that fit equally well
   while disagreeing about the answer.

2. **Is it powerful enough to answer the question asked?** A design has a
   minimum detectable effect fixed before launch. If the MDE exceeds the lift
   the campaign could plausibly produce, a null result means nothing and the
   study should not be run in that form. This is the check most often skipped,
   and the reason "no lift detected" gets mistaken for "no lift".

References for the specific statistics used, none of which are original here:

* Ljung-Box and Shapiro-Wilk residual diagnostics are the conventional
  adequacy pair for structural time-series models.
* The post/pre RMSPE ratio permutation test follows Abadie, Diamond and
  Hainmueller (2010). Its advantage over a raw post-period gap is robustness
  to poorly-fitting placebo units: a unit fitted badly before the
  intervention will usually be fitted badly after it too, so the ratio
  cancels rather than inflating the null.
* The refuter set mirrors the standard causal-inference refutation suite
  (random common cause, dummy outcome, data subset): each probes a distinct
  failure mode, and passing them raises confidence without proving
  correctness.
* MDE by power simulation -- inject known lifts and measure detection rate --
  is the accepted practice in geo-experiment design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PREDICTOR_SETS, CohortConfig
from .counterfactual import get_backend
from .inference import UpliftResult, estimate_uplift, rolling_origin_errors


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    guidance: str

    def as_row(self) -> dict[str, object]:
        return {
            "check": self.name,
            "result": "PASS" if self.passed else "REVIEW",
            "value": round(self.value, 4),
            "guidance": self.guidance,
        }


@dataclass
class ModelReport:
    adequacy: list[Check] = field(default_factory=list)
    power: pd.DataFrame | None = None
    mde: float = float("nan")
    rmspe_ratio: float = float("nan")
    rmspe_p_value: float = float("nan")
    refuters: list[Check] = field(default_factory=list)
    predictor_value: pd.DataFrame | None = None

    @property
    def fit_for_purpose(self) -> bool:
        return all(c.passed for c in self.adequacy) and all(c.passed for c in self.refuters)

    def to_frame(self) -> pd.DataFrame:
        rows = [{**c.as_row(), "family": "adequacy"} for c in self.adequacy]
        rows += [{**c.as_row(), "family": "refutation"} for c in self.refuters]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Model adequacy
# ---------------------------------------------------------------------------


def residual_checks(result: UpliftResult, *, lags: int = 12) -> list[Check]:
    """Do the pre-period residuals look like white noise?

    If they do not, the model has missed structure that is still in the data,
    and whatever it missed will be extrapolated into the counterfactual. This
    is the cheapest available warning that a specification is wrong, and it is
    diagnostic rather than decisive: rejection signals misspecification but
    does not invalidate a permutation p-value, which is distribution-free.
    """
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox

    if result.residuals is None or len(result.residuals) < lags + 5:
        return []
    resid = np.asarray(result.residuals, dtype=float)
    resid = resid[np.isfinite(resid)]

    lb = acorr_ljungbox(resid, lags=[min(lags, len(resid) // 4)], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    sw_p = float(stats.shapiro(resid).pvalue) if len(resid) <= 5000 else float("nan")
    acf1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])

    return [
        Check(
            "Residual autocorrelation (Ljung-Box)",
            lb_p > 0.05,
            lb_p,
            "p > 0.05 means no leftover serial structure. Failing it points to a "
            "missing seasonal or trend component, not to a wrong effect.",
        ),
        Check(
            "Residual normality (Shapiro-Wilk)",
            sw_p > 0.05 or np.isnan(sw_p),
            sw_p,
            "Affects the model's own interval only. The conformal interval used "
            "here does not assume normality, so this is informational.",
        ),
        Check(
            "First-order residual correlation",
            abs(acf1) < 0.25,
            acf1,
            "Large |acf1| means the local level is absorbing structure the "
            "covariates should explain.",
        ),
    ]


def accuracy_checks(
    result: UpliftResult, *, mape_limit: float = 0.05, horizon_weeks: int | None = None
) -> list[Check]:
    """Is out-of-sample accuracy good enough at the horizon that matters?"""
    mape = float(result.diagnostics.get("backtest_mape", np.nan))
    origins = int(result.diagnostics.get("n_backtest_origins", 0))
    return [
        Check(
            "Out-of-sample accuracy at campaign horizon",
            mape <= mape_limit,
            mape,
            f"Mean absolute forecast error over {origins} rolling origins. Above "
            f"{mape_limit:.0%} the counterfactual is too loose to price a campaign.",
        ),
        Check(
            "Backtest coverage of the pre-period",
            origins >= 10,
            float(origins),
            "Fewer than ~10 origins makes the error distribution -- and therefore "
            "the interval -- unreliable.",
        ),
    ]


# ---------------------------------------------------------------------------
# 2. Power and minimum detectable effect
# ---------------------------------------------------------------------------


def power_curve(
    result: UpliftResult,
    *,
    lifts: np.ndarray | None = None,
    target_power: float = 0.80,
) -> tuple[pd.DataFrame, float]:
    """Detection probability by true lift, and the MDE at target power.

    Derived non-parametrically from the bootstrap draws already computed for
    the interval. A campaign with a true lift of ``d`` produces estimates
    distributed like the observed draws re-centred on ``d``; detection
    requires clearing both the significance threshold and the materiality
    floor, which is the same rule used for the headline verdict.

    Reporting this alongside the estimate is what separates "no lift" from
    "this design could never have seen a lift of that size".
    """
    if result.relative_draws is None:
        raise ValueError("no bootstrap draws retained; cannot derive power")

    draws = np.asarray(result.relative_draws, dtype=float)
    centred = draws - draws.mean()
    se = float(centred.std(ddof=1))
    # Significance threshold implied by the interval, plus the materiality floor.
    z = abs(np.quantile(centred, 1 - result.alpha / 2))
    threshold = max(z, result.materiality_threshold)

    lifts = lifts if lifts is not None else np.arange(0.0, 0.1001, 0.005)
    rows = []
    for lift in lifts:
        detected = float(np.mean((centred + lift) >= threshold))
        rows.append({"true_lift": float(lift), "detection_probability": detected})
    frame = pd.DataFrame(rows)

    above = frame[frame["detection_probability"] >= target_power]
    mde = float(above["true_lift"].iloc[0]) if not above.empty else float("nan")
    frame.attrs["se"] = se
    frame.attrs["threshold"] = threshold
    return frame, mde


# ---------------------------------------------------------------------------
# 3. Abadie post/pre RMSPE ratio
# ---------------------------------------------------------------------------


def rmspe_ratio_test(
    panel: pd.DataFrame,
    cfg: CohortConfig,
    design,
    *,
    n_placebo: int = 20,
    seed: int = 17,
) -> tuple[float, float, np.ndarray]:
    """Post/pre RMSPE ratio against a placebo distribution.

    The treated unit's ratio measures how much worse the model fits after the
    intervention than before. Comparing ratios rather than post-period gaps
    is what makes the test robust to placebo units the model fits badly:
    a unit fitted poorly before will usually be fitted poorly after, and the
    ratio cancels the level of misfit instead of rewarding it.

    Returns the treated ratio, a permutation p-value, and the placebo ratios.
    """
    from .simulate import aggregate_to_cohort
    from .validation import _design_for

    model = get_backend(cfg.model.backend)

    def ratio(dm) -> float:
        fit = model.fit_predict(dm.y_pre, dm.X_pre, dm.X_post)
        pre = fit.residuals
        if pre is None or len(pre) == 0:
            return float("nan")
        post = dm.y_post - fit.mean
        return float(np.sqrt(np.mean(post**2)) / np.sqrt(np.mean(np.asarray(pre) ** 2)))

    treated_ratio = ratio(design)

    rng = np.random.default_rng(seed)
    pool_mask = (
        panel["control_role"] == "placebo"
        if "control_role" in panel.columns
        else panel["cohort"] != "treated"
    )
    pool = panel[pool_mask]
    ids = pool["property_id"].unique()
    n_treated = panel.loc[panel["cohort"] == "treated", "property_id"].nunique()
    size = min(n_treated, len(ids) // 2)

    ratios = []
    for _ in range(n_placebo):
        chosen = rng.choice(ids, size=size, replace=False)
        subset = panel[
            panel["property_id"].isin(chosen)
            | (panel.get("control_role", panel["cohort"]) == "covariate")
        ].copy()
        subset.loc[subset["property_id"].isin(chosen), "cohort"] = "placebo"
        series = aggregate_to_cohort(subset, cfg.outcomes)
        series = series[series["cohort"] == "placebo"].reset_index(drop=True)
        try:
            ratios.append(ratio(_design_for(series, cfg.primary_outcome, cfg)))
        except Exception:
            continue

    placebo = np.asarray([r for r in ratios if np.isfinite(r)], dtype=float)
    if placebo.size == 0:
        return treated_ratio, float("nan"), placebo
    p_value = float((np.sum(placebo >= treated_ratio) + 1) / (placebo.size + 1))
    return treated_ratio, p_value, placebo


# ---------------------------------------------------------------------------
# 4. Refuters
# ---------------------------------------------------------------------------


def refutation_suite(
    series: pd.DataFrame,
    cfg: CohortConfig,
    result: UpliftResult,
    *,
    seed: int = 23,
    n_bootstrap: int = 400,
) -> list[Check]:
    """Three probes of distinct failure modes.

    * **Random common cause** -- add a pure-noise predictor. A stable
      estimator should barely move; a large shift means the answer is
      sensitive to the adjustment set rather than to the data.
    * **Dummy outcome** -- replace the outcome with noise carrying the same
      level and volatility. The effect must collapse toward zero; anything
      else means the pipeline manufactures effects.
    * **Pre-period subset** -- re-estimate on a shortened pre-period. A large
      move means the answer depends on a particular slice of history.

    Passing all three does not prove the estimate correct. Failing any one
    localises the problem, which is the useful part.
    """
    from .validation import _design_for

    rng = np.random.default_rng(seed)
    model = cfg.model.backend
    baseline = result.relative_uplift
    checks: list[Check] = []

    def estimate_from(frame: pd.DataFrame, regressors: list[str], outcome: str):
        dm = _design_for(frame, outcome, cfg, regressors=regressors)
        return estimate_uplift(
            get_backend(model), dm, alpha=cfg.model.alpha, n_bootstrap=n_bootstrap
        )

    def lift_from(frame: pd.DataFrame, regressors: list[str], outcome: str) -> float:
        return estimate_from(frame, regressors, outcome).relative_uplift

    # -- random common cause ------------------------------------------------
    noised = series.copy()
    noised["random_noise"] = np.exp(rng.normal(0.0, 1.0, len(noised)) * 0.25) * 1e4
    shift = lift_from(
        noised, [*cfg.model.predictors, "random_noise"], cfg.primary_outcome
    ) - baseline
    checks.append(
        Check(
            "Random common cause",
            abs(shift) <= 0.01,
            float(shift),
            "Adding a noise predictor should move the estimate by well under a "
            "point. A large shift means the adjustment set is doing unstable work.",
        )
    )

    # -- dummy outcome ------------------------------------------------------
    dummy = series.copy()
    actual = dummy[cfg.primary_outcome].to_numpy(dtype=float)
    dummy["dummy_outcome"] = np.exp(
        rng.normal(np.log(actual.mean()), float(np.std(np.log(actual))), len(dummy))
    )
    # Judged on significance, not magnitude. A noise outcome is unforecastable
    # by construction, so the estimator correctly returns a wide interval and
    # a point estimate that wanders; the requirement is that it cannot
    # *conclude* anything. An earlier version tested |lift| < 3% and flagged a
    # -4.4% reading whose interval comfortably spanned zero -- penalising the
    # model for honest uncertainty rather than for manufacturing a result.
    dummy_est = estimate_from(dummy, cfg.model.predictors, "dummy_outcome")
    checks.append(
        Check(
            "Dummy outcome",
            not dummy_est.is_significant,
            float(dummy_est.relative_uplift),
            "A noise outcome must not yield a significant effect. The point "
            "estimate is free to wander; the interval must contain zero.",
        )
    )

    # -- shortened pre-period ----------------------------------------------
    # 62 weeks rather than 52: the rolling-origin backtest needs a 52-week
    # minimum training block plus the forecast horizon, so a 52-week subset
    # leaves nothing to backtest on.
    short = _design_for(series, cfg.primary_outcome, cfg, pre_weeks=62)
    short_lift = estimate_uplift(
        get_backend(model), short, alpha=cfg.model.alpha, n_bootstrap=n_bootstrap
    ).relative_uplift
    checks.append(
        Check(
            "Pre-period subset (62 weeks)",
            abs(short_lift - baseline) <= 0.015,
            float(short_lift - baseline),
            "Re-estimating on 20% less history should not move the answer "
            "much. If it does, the result is a property of the window chosen.",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# 5. Which predictors are earning their place
# ---------------------------------------------------------------------------


def predictor_value(series: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Rank predictors by the out-of-sample error they remove.

    A frequentist stand-in for the posterior inclusion probabilities a
    spike-and-slab prior would give: drop each predictor, re-run the
    rolling-origin backtest, and record how much worse the counterfactual
    forecast gets. Ranking on *forecast error* rather than on the effect
    estimate matters -- a predictor that changes the answer without improving
    the forecast is a red flag, not a useful control.
    """
    from .validation import _design_for

    model = get_backend(cfg.model.backend)
    horizon = cfg.n_post_weeks

    def mape(regressors: list[str]) -> float:
        dm = _design_for(series, cfg.primary_outcome, cfg, regressors=regressors)
        errors = rolling_origin_errors(model, dm, horizon)
        return float(np.mean(np.abs(np.expm1(errors) if dm.log_transform else errors)))

    full = mape(cfg.model.predictors)
    rows = []
    for dropped in cfg.model.predictors:
        remaining = [r for r in cfg.model.predictors if r != dropped]
        if not remaining:
            continue
        without = mape(remaining)
        rows.append(
            {
                "predictor": dropped,
                "mape_without": without,
                "mape_full": full,
                "error_added_by_dropping": without - full,
                "relative_value": (without - full) / full if full else np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values("error_added_by_dropping", ascending=False)
    frame["verdict"] = np.where(
        frame["relative_value"] > 0.05,
        "load-bearing",
        np.where(frame["relative_value"] > 0.0, "marginal", "no measurable value"),
    )
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_model(
    panel: pd.DataFrame,
    series: pd.DataFrame,
    cfg: CohortConfig,
    result: UpliftResult,
    design,
    *,
    n_placebo: int = 20,
    include_predictor_value: bool = True,
) -> ModelReport:
    """Full model-evaluation report for one campaign."""
    report = ModelReport()
    report.adequacy = residual_checks(result) + accuracy_checks(result)
    report.power, report.mde = power_curve(result)
    report.rmspe_ratio, report.rmspe_p_value, _ = rmspe_ratio_test(
        panel, cfg, design, n_placebo=n_placebo
    )
    report.refuters = refutation_suite(series, cfg, result)
    if include_predictor_value:
        report.predictor_value = predictor_value(series, cfg)
    return report


def compare_designs(
    series: pd.DataFrame, cfg: CohortConfig, *, designs: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Rank candidate designs on out-of-sample forecast error.

    On real data the effect is unobservable, so designs cannot be ranked by
    accuracy against truth. What *is* observable is how well each predicts the
    pre-period out of sample, and a design that cannot forecast the quiet
    period will not produce a trustworthy counterfactual for the loud one.
    Reported with the estimate each design implies, so a reviewer can see
    whether better forecasting changes the answer or merely tightens it.
    """
    from .validation import _design_for

    designs = designs or tuple(PREDICTOR_SETS)
    model = get_backend(cfg.model.backend)
    rows = []
    for name in designs:
        try:
            dm = _design_for(
                series, cfg.primary_outcome, cfg, regressors=PREDICTOR_SETS[name]
            )
            errors = rolling_origin_errors(model, dm, cfg.n_post_weeks)
            est = estimate_uplift(model, dm, alpha=cfg.model.alpha, n_bootstrap=600)
            rows.append(
                {
                    "design": name,
                    "backtest_mape": float(
                        np.mean(np.abs(np.expm1(errors) if dm.log_transform else errors))
                    ),
                    "estimate": est.relative_uplift,
                    "ci_low": est.ci_relative[0],
                    "ci_high": est.ci_relative[1],
                    "ci_width": est.ci_relative[1] - est.ci_relative[0],
                    "n_predictors": len(PREDICTOR_SETS[name]),
                }
            )
        except Exception as exc:  # a design that cannot be fitted is information
            rows.append({"design": name, "backtest_mape": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values("backtest_mape").reset_index(drop=True)


def build_design_for_eval(series: pd.DataFrame, cfg: CohortConfig):
    """Convenience: the headline design matrix, for callers outside the pipeline."""
    from .validation import _design_for

    return _design_for(series, cfg.primary_outcome, cfg)
