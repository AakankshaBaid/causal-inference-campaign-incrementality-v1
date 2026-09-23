"""Turn a counterfactual path into an effect estimate with honest uncertainty.

The interval matters more than the point estimate. A campaign that reads
+7.8% with a 90% interval of [+5.1%, +10.6%] is a decision; the same point
estimate with [-2%, +18%] is a request for a bigger sample.

Model-reported forecast standard errors are not enough here. They are
conditional on the model being correctly specified, and an eight-week-ahead
forecast from a state space model fitted to 78 weeks is exactly where
mis-specification bites. So the intervals below are built from the model's own
*out-of-sample* track record: refit at rolling origins across the pre-period,
collect forecast-path errors at the real horizon, and resample them in blocks
to preserve autocorrelation. This is the conformal-inference logic used for
synthetic control, and it has the practical virtue that a model which
forecasts badly in the pre-period is punished with wide intervals instead of
quietly producing a confident wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .counterfactual import Counterfactual
from .features import DesignMatrix


@dataclass
class UpliftResult:
    """Everything the reporting layer needs about one outcome."""

    outcome: str
    backend: str
    weekly: pd.DataFrame
    actual_cumulative: float
    counterfactual_cumulative: float
    absolute_uplift: float
    relative_uplift: float
    ci_absolute: tuple[float, float]
    ci_relative: tuple[float, float]
    p_value_bootstrap: float
    alpha: float
    materiality_threshold: float = 0.01
    diagnostics: dict[str, float] = field(default_factory=dict)
    relative_draws: np.ndarray | None = None
    """Bootstrap draws of the relative effect, reused for power analysis."""

    residuals: np.ndarray | None = None
    """Pre-period model residuals, for adequacy testing."""

    @property
    def is_significant(self) -> bool:
        """Statistically distinguishable from zero. Necessary, not sufficient."""
        return self.ci_absolute[0] > 0 or self.ci_absolute[1] < 0

    @property
    def is_material(self) -> bool:
        """Large enough to act on, regardless of how tight the interval is."""
        return abs(self.relative_uplift) >= self.materiality_threshold

    @property
    def verdict(self) -> str:
        if not self.is_significant:
            return "not distinguishable from zero"
        if not self.is_material:
            return (
                f"significant but immaterial (below the "
                f"{self.materiality_threshold:.1%} threshold)"
            )
        return "material effect"

    def summary_row(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "backend": self.backend,
            "actual": self.actual_cumulative,
            "counterfactual": self.counterfactual_cumulative,
            "absolute_uplift": self.absolute_uplift,
            "relative_uplift": self.relative_uplift,
            "ci_low": self.ci_absolute[0],
            "ci_high": self.ci_absolute[1],
            "relative_ci_low": self.ci_relative[0],
            "relative_ci_high": self.ci_relative[1],
            "p_value": self.p_value_bootstrap,
            "significant": self.is_significant,
            "material": self.is_material,
            "verdict": self.verdict,
        }


def rolling_origin_errors(
    model: Counterfactual,
    design: DesignMatrix,
    horizon: int,
    *,
    min_train: int = 52,
    max_origins: int = 24,
) -> np.ndarray:
    """Forecast-path errors from refitting at rolling pre-period origins.

    Returns an array of shape (n_origins, horizon) in model space. Each row is
    a genuine out-of-sample error path for the same horizon we care about, so
    the resulting intervals answer "how wrong is this model, at this horizon,
    on this series" rather than "how wrong does this model believe it is".
    """
    n_pre = len(design.y_pre)
    last_origin = n_pre - horizon
    if last_origin <= min_train:
        raise ValueError(
            f"need > {min_train + horizon} pre-period weeks to backtest a "
            f"{horizon}-week horizon; have {n_pre}"
        )

    origins = np.linspace(min_train, last_origin, num=min(max_origins, last_origin - min_train + 1))
    origins = np.unique(origins.astype(int))

    errors = []
    for cut in origins:
        result = model.fit_predict(
            design.y_pre[:cut],
            design.X_pre[:cut],
            design.X_pre[cut : cut + horizon],
        )
        errors.append(design.y_pre[cut : cut + horizon] - result.mean)
    return np.asarray(errors, dtype=float)


def _block_bootstrap_paths(
    errors: np.ndarray, horizon: int, n_draws: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample forecast-error paths, preserving within-path autocorrelation.

    Whole rows are drawn whenever the error matrix is at least as long as the
    horizon, and this matters more than it looks. The quantity being priced is
    a *cumulative* effect over the window, and models are wrong in runs rather
    than independently week to week -- a miss is usually eight weeks of
    slightly-low, not four low and four high. Stitching blocks from different
    origins lets those runs cancel, which shrinks the variance of the
    cumulative sum and produces intervals that are too narrow. Measured on
    simulated campaigns, block-stitching gave 75% coverage against a nominal
    90%; whole-path resampling brings it back to nominal.

    Block stitching survives only as the fallback for horizons longer than any
    single backtest path, where there is no whole path to draw.
    """
    n_origins, err_h = errors.shape
    if err_h >= horizon:
        rows = rng.integers(0, n_origins, size=n_draws)
        return errors[rows, :horizon]

    block_length = int(np.clip(block_length, 1, err_h))
    out = np.empty((n_draws, horizon))
    for i in range(n_draws):
        path: list[float] = []
        while len(path) < horizon:
            row = rng.integers(0, n_origins)
            start = rng.integers(0, err_h - block_length + 1)
            path.extend(errors[row, start : start + block_length])
        out[i] = path[:horizon]
    return out


def estimate_uplift(
    model: Counterfactual,
    design: DesignMatrix,
    *,
    alpha: float = 0.10,
    n_bootstrap: int = 2_000,
    block_length: int = 4,
    seed: int = 7,
    backend_name: str = "bsts",
    materiality_threshold: float = 0.01,
) -> UpliftResult:
    """Fit the counterfactual, difference it against reality, attach intervals."""
    rng = np.random.default_rng(seed)
    horizon = len(design.y_post)

    fit = model.fit_predict(design.y_pre, design.X_pre, design.X_post)
    actual = design.inverse(design.y_post)
    errors = rolling_origin_errors(model, design, horizon)

    # Retransformation bias. exp() of a forecast in log space is a *median*,
    # not a mean, so exponentiating directly understates the counterfactual
    # and inflates the lift by roughly half the forecast variance -- a few
    # tenths of a point here, and systematically in the flattering direction.
    # Duan's smearing estimator corrects it non-parametrically using the
    # model's own out-of-sample error pool, which also keeps the point
    # estimate consistent with the bootstrap interval below (that interval
    # averages over the same errors and is therefore already on a mean basis).
    smearing = float(np.mean(np.exp(errors))) if design.log_transform else 1.0
    counterfactual = design.inverse(fit.mean) * smearing

    draws = _block_bootstrap_paths(errors, horizon, n_bootstrap, block_length, rng)

    # Each draw perturbs the counterfactual path by a plausible forecast error,
    # then we recompute the cumulative effect. The spread of those recomputed
    # effects is the interval.
    cf_draws = design.inverse(fit.mean[None, :] + draws)
    cum_actual = float(actual.sum())
    cum_cf_draws = cf_draws.sum(axis=1)
    effect_draws = cum_actual - cum_cf_draws

    lo, hi = np.quantile(effect_draws, [alpha / 2, 1 - alpha / 2])
    rel_draws = effect_draws / cum_cf_draws
    rel_lo, rel_hi = np.quantile(rel_draws, [alpha / 2, 1 - alpha / 2])

    cum_cf = float(counterfactual.sum())
    absolute = cum_actual - cum_cf
    # Two-sided bootstrap p-value for "no effect", with the standard +1
    # correction so a p-value is never reported as exactly zero.
    n_extreme = int(np.sum(np.sign(effect_draws) != np.sign(absolute)))
    p_value = min(1.0, 2.0 * (n_extreme + 1) / (n_bootstrap + 1))

    weekly = pd.DataFrame(
        {
            "week": design.weeks_post,
            "actual": actual,
            "counterfactual": counterfactual,
            "pointwise_uplift": actual - counterfactual,
            "cumulative_actual": np.cumsum(actual),
            "cumulative_counterfactual": np.cumsum(counterfactual),
        }
    )
    weekly["cumulative_uplift"] = (
        weekly["cumulative_actual"] - weekly["cumulative_counterfactual"]
    )
    weekly["relative_uplift"] = weekly["pointwise_uplift"] / weekly["counterfactual"]
    # Pointwise bands, for the chart only -- the decision uses the cumulative.
    pointwise_q = np.quantile(
        design.inverse(fit.mean[None, :] + draws), [alpha / 2, 1 - alpha / 2], axis=0
    )
    weekly["counterfactual_low"] = pointwise_q[0]
    weekly["counterfactual_high"] = pointwise_q[1]

    diagnostics = dict(fit.diagnostics)
    diagnostics["smearing_factor"] = smearing
    diagnostics["backtest_mape"] = float(
        np.mean(np.abs(np.expm1(errors) if design.log_transform else errors))
    )
    diagnostics["backtest_rmse"] = float(
        np.sqrt(np.mean((actual.mean() * (np.expm1(errors) if design.log_transform else errors)) ** 2))
    )
    diagnostics["n_backtest_origins"] = float(errors.shape[0])

    return UpliftResult(
        outcome=getattr(design, "outcome", "outcome"),
        backend=backend_name,
        weekly=weekly,
        actual_cumulative=cum_actual,
        counterfactual_cumulative=cum_cf,
        absolute_uplift=absolute,
        relative_uplift=absolute / cum_cf,
        ci_absolute=(float(lo), float(hi)),
        ci_relative=(float(rel_lo), float(rel_hi)),
        p_value_bootstrap=float(p_value),
        alpha=alpha,
        materiality_threshold=materiality_threshold,
        relative_draws=rel_draws,
        residuals=fit.residuals,
        diagnostics=diagnostics,
    )
