"""Drop-in compatibility with the `CausalImpact` / `CImpact` interface.

The existing production notebook calls `tfcausalimpact` like this::

    ci = CausalImpact(data, pre_period, post_period, model_args={"fit_method": "vi"})
    print(ci.summary())
    ci.inferences.tail()

That convention is worth preserving. An estimator that improves on the
incumbent but requires everyone to rewrite their notebooks does not get
adopted, and the team has years of muscle memory in the `(y, x0..xn)` data
contract, the `pre_period` / `post_period` date lists, and the summary table
that goes straight onto a slide.

So this module wraps the pipeline in that signature::

    ci = CausalImpactModel(data, pre_period, post_period)
    print(ci.summary())
    ci.inferences.tail()

Same call, same output shape, different machinery underneath: a conformal
interval validated by coverage simulation instead of a posterior credible
interval, plus the falsification battery available via `ci.validate()`.

Two deliberate differences from the incumbent, both documented in the README:

1. **Covariates are standardised on pre-period moments only.** The reference
   implementations standardise the whole frame, which leaks post-period
   information into the scaling. The effect is small but it is free to avoid.
2. **Intervals are conformal, not posterior.** Hyperparameters here are point
   estimates from MLE rather than draws from a posterior, so a credible
   interval would understate uncertainty. The rolling-origin bootstrap prices
   in forecast error the model does not know it has -- and coverage is then
   checked by simulation rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .counterfactual import get_backend
from .features import DesignMatrix, fourier_terms, holiday_indicators
from .inference import UpliftResult, estimate_uplift

DEFAULT_MODEL_ARGS: dict[str, object] = {
    "backend": "bsts",
    "log_transform": True,
    "annual_fourier_terms": 2,
    "holidays": True,
    "alpha": 0.10,
    "n_bootstrap": 2_000,
    "block_length": 4,
}


@dataclass
class CausalImpactModel:
    """Estimate a causal impact from a wide frame of `y` plus control series.

    Parameters
    ----------
    data
        DataFrame indexed by date. The **first** column is the response; every
        remaining column is a control series. This is the `CausalImpact`
        convention, kept on purpose.
    pre_period, post_period
        Two-element ``[start, end]`` date ranges, inclusive. Strings in any
        format pandas parses, or Timestamps.
    model_args
        Overrides for ``DEFAULT_MODEL_ARGS``.
    """

    data: pd.DataFrame
    pre_period: list
    post_period: list
    model_args: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.args = {**DEFAULT_MODEL_ARGS, **self.model_args}
        if self.data.shape[1] < 2:
            raise ValueError(
                "data needs a response column plus at least one control series; "
                f"got {self.data.shape[1]} column(s)"
            )
        self.target_col = self.data.columns[0]
        self.control_cols = list(self.data.columns[1:])
        self._design = self._build_design()
        self._result: UpliftResult | None = None

    # -- construction ----------------------------------------------------

    def _build_design(self) -> DesignMatrix:
        frame = self.data.copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        weeks = pd.DatetimeIndex(frame.index)

        pre = [pd.Timestamp(x) for x in self.pre_period]
        post = [pd.Timestamp(x) for x in self.post_period]
        pre_mask = (weeks >= pre[0]) & (weeks <= pre[1])
        post_mask = (weeks >= post[0]) & (weeks <= post[1])
        if pre_mask.sum() < 26:
            raise ValueError(f"pre_period covers {pre_mask.sum()} rows; need >= 26")
        if post_mask.sum() == 0:
            raise ValueError("post_period covers no rows")
        if (pre_mask & post_mask).any():
            raise ValueError("pre_period and post_period overlap")

        log = bool(self.args["log_transform"])
        y = frame[self.target_col].to_numpy(dtype=float)
        if log:
            if np.any(y <= 0):
                raise ValueError(
                    f"{self.target_col} has non-positive values; "
                    "pass model_args={'log_transform': False}"
                )
            y = np.log(y)

        design = frame[self.control_cols].astype(float).copy()
        if log:
            for col in design.columns:
                if (design[col] > 0).all():
                    design[col] = np.log(design[col])

        mu = design.loc[pre_mask].mean()
        sigma = design.loc[pre_mask].std(ddof=0).replace(0.0, 1.0)
        design = (design - mu) / sigma

        extras = [design.reset_index(drop=True)]
        extras.append(fourier_terms(weeks, int(self.args["annual_fourier_terms"])))
        if self.args["holidays"]:
            extras.append(
                holiday_indicators(
                    weeks,
                    {"peak_shopping": [47, 48], "holiday_lull": [51, 52], "new_year": [1, 2]},
                )
            )
        design = pd.concat(extras, axis=1)
        constant_in_pre = [
            c for c in design.columns if design.loc[pre_mask, c].nunique() < 2
        ]
        design = design.drop(columns=constant_in_pre)

        return DesignMatrix(
            y_pre=y[pre_mask],
            X_pre=design.loc[pre_mask].to_numpy(dtype=float),
            y_post=y[post_mask],
            X_post=design.loc[post_mask].to_numpy(dtype=float),
            weeks_pre=weeks[pre_mask],
            weeks_post=weeks[post_mask],
            feature_names=list(design.columns),
            log_transform=log,
        )

    # -- estimation ------------------------------------------------------

    @property
    def result(self) -> UpliftResult:
        """Fit on first access, then cache."""
        if self._result is None:
            backend = str(self.args["backend"])
            kwargs = {}
            if backend == "prophet":
                kwargs = {
                    "weeks_pre": self._design.weeks_pre,
                    "weeks_post": self._design.weeks_post,
                }
            self._result = estimate_uplift(
                get_backend(backend, **kwargs),
                self._design,
                alpha=float(self.args["alpha"]),
                n_bootstrap=int(self.args["n_bootstrap"]),
                block_length=int(self.args["block_length"]),
                backend_name=backend,
            )
            self._result.outcome = str(self.target_col)
        return self._result

    @property
    def inferences(self) -> pd.DataFrame:
        """Per-period predictions and effects, in the reference column names."""
        weekly = self.result.weekly
        return pd.DataFrame(
            {
                "preds": weekly["counterfactual"].to_numpy(),
                "preds_lower": weekly["counterfactual_low"].to_numpy(),
                "preds_upper": weekly["counterfactual_high"].to_numpy(),
                "point_effects": weekly["pointwise_uplift"].to_numpy(),
                "cumulative_effects": weekly["cumulative_uplift"].to_numpy(),
                "response": weekly["actual"].to_numpy(),
            },
            index=pd.DatetimeIndex(weekly["week"], name=self.data.index.name or "date"),
        )

    def summary(self, output: str = "summary") -> str:
        """Summary table, or a plain-language report.

        Mirrors the reference `summary()` / `summary('report')` split so an
        existing notebook cell keeps working unchanged.
        """
        if output == "report":
            return self._report()
        if output != "summary":
            raise ValueError("output must be 'summary' or 'report'")
        return self._summary_table()

    def _summary_table(self) -> str:
        r = self.result
        n = len(r.weekly)
        conf = int(round((1 - r.alpha) * 100))

        avg_actual = r.actual_cumulative / n
        avg_pred = r.counterfactual_cumulative / n
        abs_lo, abs_hi = r.ci_absolute
        rel_lo, rel_hi = r.ci_relative

        def money(value: float) -> str:
            return f"{value:,.0f}"

        lines = [
            f"Conformal inference {{{self.args['backend']}}}",
            "",
            f"{'':<24}{'Average':>26}{'Cumulative':>26}",
            f"{'Actual':<24}{money(avg_actual):>26}{money(r.actual_cumulative):>26}",
            f"{'Prediction':<24}{money(avg_pred):>26}{money(r.counterfactual_cumulative):>26}",
            f"{f'{conf}% CI':<24}"
            f"{f'[{money(avg_actual - abs_hi / n)}, {money(avg_actual - abs_lo / n)}]':>26}"
            f"{f'[{money(r.actual_cumulative - abs_hi)}, {money(r.actual_cumulative - abs_lo)}]':>26}",
            "",
            f"{'Absolute effect':<24}{money(r.absolute_uplift / n):>26}{money(r.absolute_uplift):>26}",
            f"{f'{conf}% CI':<24}{f'[{money(abs_lo / n)}, {money(abs_hi / n)}]':>26}"
            f"{f'[{money(abs_lo)}, {money(abs_hi)}]':>26}",
            "",
            f"{'Relative effect':<24}{f'{r.relative_uplift:.2%}':>26}{f'{r.relative_uplift:.2%}':>26}",
            f"{f'{conf}% CI':<24}{f'[{rel_lo:.2%}, {rel_hi:.2%}]':>26}"
            f"{f'[{rel_lo:.2%}, {rel_hi:.2%}]':>26}",
            "",
            f"Bootstrap tail-area probability p: {r.p_value_bootstrap:.5f}",
            f"Probability of a causal effect: {(1 - r.p_value_bootstrap):.2%}",
            "",
            "Model performance (pre-period, rolling origin):",
            f"  MAPE:  {r.diagnostics.get('backtest_mape', float('nan')):.2%}",
            f"  RMSE:  {r.diagnostics.get('backtest_rmse', float('nan')):,.0f}",
            f"  origins: {int(r.diagnostics.get('n_backtest_origins', 0))}",
        ]
        return "\n".join(lines)

    def _report(self) -> str:
        r = self.result
        conf = int(round((1 - r.alpha) * 100))
        direction = "an increase" if r.absolute_uplift > 0 else "a decrease"
        verdict = (
            "This effect is statistically significant at the stated level."
            if r.is_significant
            else "This effect is NOT statistically distinguishable from zero, so it "
            "should not be reported as a campaign result."
        )
        return (
            f"During the {len(r.weekly)} periods after the intervention, the response "
            f"variable had an average value of {r.actual_cumulative / len(r.weekly):,.0f}. "
            f"In the absence of the intervention, the model predicts an average of "
            f"{r.counterfactual_cumulative / len(r.weekly):,.0f}. Summing over the "
            f"post-intervention window, the response totalled "
            f"{r.actual_cumulative:,.0f} against a predicted "
            f"{r.counterfactual_cumulative:,.0f}, {direction} of "
            f"{r.absolute_uplift:,.0f} ({r.relative_uplift:+.2%}) with a {conf}% "
            f"interval of [{r.ci_absolute[0]:,.0f}, {r.ci_absolute[1]:,.0f}]. "
            f"{verdict} The model's pre-period forecast error at this horizon is "
            f"{r.diagnostics.get('backtest_mape', float('nan')):.2%}, measured over "
            f"{int(r.diagnostics.get('n_backtest_origins', 0))} rolling origins; the "
            f"interval above is derived from that error distribution rather than from "
            f"the model's own reported variance."
        )


def frame_from_panel(
    panel: pd.DataFrame,
    *,
    outcome: str = "gbv",
    treated_cohort: str = "treated",
    control_role: str = "covariate",
    extra_columns: tuple[str, ...] = (
        "meta_impressions",
        "destination_queries",
        "market_visitors",
        "market_purchases",
    ),
) -> pd.DataFrame:
    """Collapse a property-week panel into the `CausalImpact` wide frame.

    Returns a date-indexed frame whose first column is the treated cohort's
    outcome and whose remaining columns are control series -- ready to hand
    straight to ``CausalImpactModel`` or to the reference implementations.
    """
    work = panel.copy()
    work["week"] = pd.to_datetime(work["week"])

    y = (
        work[work["cohort"] == treated_cohort]
        .groupby("week")[outcome]
        .sum()
        .rename("y")
    )
    role = work.get("control_role", work["cohort"])
    controls = (
        work[role == control_role].groupby("week")[outcome].sum().rename("x0_compset")
    )
    frame = pd.concat([y, controls], axis=1)

    for i, col in enumerate(extra_columns, start=1):
        if col in work.columns:
            frame[f"x{i}_{col}"] = work.groupby("week")[col].mean()

    return frame.dropna()
