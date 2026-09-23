"""Counterfactual estimators behind a single interface.

Three estimators, deliberately different in their assumptions, because
agreement between methods that fail in different ways is the only real
robustness evidence available when the ground truth is unobservable:

* ``bsts`` -- local-level state space with a regression component. This is the
  CausalImpact formulation: the level absorbs slow drift the covariates miss,
  the regression block borrows strength from market demand signals. Primary
  estimator.
* ``synthetic_control`` -- convex weights over untreated donor properties, no
  covariates at all. Fails differently: it is vulnerable to spillover onto
  donors, which is exactly why we run a displacement test.
* ``prophet`` -- additive trend/seasonality/regressors, optional dependency.
  Included because it is what the original production pipeline used, so the
  numbers stay comparable across the migration.

All three expose the same ``fit_predict`` call, which is what lets the
validation battery (rolling backtests, placebos, bootstraps) re-fit any of
them hundreds of times without knowing which one it has.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import optimize


@dataclass
class ForecastResult:
    """Counterfactual path in model space (log units when log_transform)."""

    mean: np.ndarray
    se: np.ndarray
    diagnostics: dict[str, float]
    residuals: np.ndarray | None = None
    """Pre-period one-step-ahead errors, for white-noise adequacy checks."""


class Counterfactual(Protocol):
    """Anything that can predict the no-campaign path from pre-period data."""

    name: str

    def fit_predict(
        self, y_pre: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray
    ) -> ForecastResult: ...


# ---------------------------------------------------------------------------
# Bayesian structural time series (local level + regression)
# ---------------------------------------------------------------------------


class BSTSCounterfactual:
    """Local-level state space with exogenous regressors, fitted by MLE.

    ``level='local level'`` rather than a local linear trend is intentional. A
    stochastic slope extrapolated eight weeks past the last observation is
    happy to invent a trend, and any drift it invents lands directly in the
    treatment effect. The level plus covariates keeps the counterfactual
    anchored to observable market demand.
    """

    name = "bsts"

    def __init__(self, level: str = "local level", maxiter: int = 250) -> None:
        self.level = level
        self.maxiter = maxiter

    def fit_predict(
        self, y_pre: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray
    ) -> ForecastResult:
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.UnobservedComponents(
                y_pre, level=self.level, exog=X_pre, freq_seasonal=None
            )
            fitted = model.fit(disp=False, maxiter=self.maxiter, method="lbfgs")
            forecast = fitted.get_forecast(steps=X_post.shape[0], exog=X_post)

        resid = np.asarray(fitted.resid, dtype=float)
        return ForecastResult(
            mean=np.asarray(forecast.predicted_mean, dtype=float),
            se=np.asarray(forecast.se_mean, dtype=float),
            residuals=resid[1:],
            diagnostics={
                "converged": float(bool(fitted.mle_retvals.get("converged", False))),
                "pre_resid_sd": float(np.nanstd(resid[1:], ddof=1)),
                "aic": float(fitted.aic),
            },
        )


# ---------------------------------------------------------------------------
# Synthetic control (convex donor weights)
# ---------------------------------------------------------------------------


class SyntheticControlCounterfactual:
    """Convex combination of untreated donor series.

    Weights solve min ||y - Xw|| subject to w >= 0 and sum(w) = 1, on
    pre-period-demeaned series. Demeaning (rather than requiring the donors to
    bracket the treated level) follows the standard fix for the case where no
    convex combination of donors can match the treated unit's level -- common
    when the treated cohort is systematically larger than the donor pool.
    """

    name = "synthetic_control"

    def __init__(self, sum_to_one_weight: float = 1e4) -> None:
        self.sum_to_one_weight = sum_to_one_weight
        self.weights_: np.ndarray | None = None

    def fit_predict(
        self, y_pre: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray
    ) -> ForecastResult:
        y_mu = float(np.mean(y_pre))
        x_mu = X_pre.mean(axis=0)
        y_c = y_pre - y_mu
        X_c = X_pre - x_mu

        # Append a heavily-weighted row of ones to impose sum(w) = 1 while
        # keeping the problem inside a single non-negative least squares call.
        A = np.vstack([X_c, np.full((1, X_c.shape[1]), self.sum_to_one_weight)])
        b = np.concatenate([y_c, [self.sum_to_one_weight]])
        weights, _ = optimize.nnls(A, b)
        total = weights.sum()
        weights = np.full(X_c.shape[1], 1.0 / X_c.shape[1]) if total <= 0 else weights / total
        self.weights_ = weights

        fitted_pre = X_c @ weights + y_mu
        resid = y_pre - fitted_pre
        resid_sd = float(np.std(resid, ddof=1))

        mean = (X_post - x_mu) @ weights + y_mu
        return ForecastResult(
            mean=mean,
            se=np.full(X_post.shape[0], resid_sd),
            residuals=resid,
            diagnostics={
                "pre_resid_sd": resid_sd,
                "n_donors_used": float((weights > 1e-4).sum()),
                "max_weight": float(weights.max()),
            },
        )


# ---------------------------------------------------------------------------
# Prophet (optional)
# ---------------------------------------------------------------------------


class ProphetCounterfactual:
    """Additive trend + seasonality + regressors, via Prophet.

    Optional dependency. Kept so results remain comparable with the original
    production pipeline, which used Prophet with the same regressor set.
    """

    name = "prophet"

    def __init__(self, weeks_pre=None, weeks_post=None) -> None:
        self.weeks_pre = weeks_pre
        self.weeks_post = weeks_post

    def fit_predict(
        self, y_pre: np.ndarray, X_pre: np.ndarray, X_post: np.ndarray
    ) -> ForecastResult:
        try:
            from prophet import Prophet
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "backend 'prophet' requires the optional dependency: "
                "pip install 'campaign-incrementality[prophet]'"
            ) from exc
        import pandas as pd

        if self.weeks_pre is None or self.weeks_post is None:
            raise ValueError("ProphetCounterfactual needs weeks_pre and weeks_post")

        cols = [f"x{i}" for i in range(X_pre.shape[1])]
        train = pd.DataFrame(X_pre, columns=cols)
        train["ds"] = pd.DatetimeIndex(self.weeks_pre)
        train["y"] = y_pre

        model = Prophet(weekly_seasonality=False, daily_seasonality=False, interval_width=0.9)
        for col in cols:
            model.add_regressor(col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train)

        future = pd.DataFrame(X_post, columns=cols)
        future["ds"] = pd.DatetimeIndex(self.weeks_post)
        forecast = model.predict(future)

        mean = forecast["yhat"].to_numpy(dtype=float)
        half_width = (
            forecast["yhat_upper"].to_numpy(dtype=float)
            - forecast["yhat_lower"].to_numpy(dtype=float)
        ) / 2.0
        return ForecastResult(
            mean=mean,
            se=half_width / 1.645,  # Prophet default interval is 80%/90%; see interval_width
            diagnostics={"pre_resid_sd": float(np.nan)},
        )


BACKENDS = {
    "bsts": BSTSCounterfactual,
    "synthetic_control": SyntheticControlCounterfactual,
    "prophet": ProphetCounterfactual,
}


def get_backend(name: str, **kwargs) -> Counterfactual:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
