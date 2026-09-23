"""Incremental value of logged-in discount stacking, via difference-in-differences.

Stacking is an extra discount shown only to signed-in members. Every
participating property gets the base campaign deal; only some allow stacking
on top. That contract split is the natural experiment: compare the campaign
uplift of stack-enabled properties against stack-disabled ones, and the
difference is the marginal value of stacking.

This is a cleaner identification problem than the campaign question itself --
both groups are treated, both face the same market, so the market-wide
confounders that force us into a counterfactual model above difference out.
What it is not is randomised: properties choose whether to allow stacking,
and the ones that do may differ systematically. Two defences below:

* a parallel-trends check on the pre-period, which is the assumption the
  design rests on and the first thing a reviewer should attack,
* CUPED-style adjustment using each property's own pre-period level, which
  strips out the between-property variance that has nothing to do with the
  treatment and typically halves the standard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import CohortConfig


@dataclass
class StackingResult:
    n_stack: int
    n_no_stack: int
    stack_uplift: float
    no_stack_uplift: float
    incremental_stacking_value: float
    ci: tuple[float, float]
    p_value: float
    parallel_trends_p: float
    variance_reduction: float

    @property
    def parallel_trends_holds(self) -> bool:
        return self.parallel_trends_p > 0.10

    def summary_row(self) -> dict[str, object]:
        return {
            "stack_enabled_properties": self.n_stack,
            "stack_disabled_properties": self.n_no_stack,
            "stack_enabled_uplift": self.stack_uplift,
            "stack_disabled_uplift": self.no_stack_uplift,
            "stacking_marginal_value": self.incremental_stacking_value,
            "ci_low": self.ci[0],
            "ci_high": self.ci[1],
            "p_value": self.p_value,
            "parallel_trends_p": self.parallel_trends_p,
            "cuped_variance_reduction": self.variance_reduction,
        }


def _property_window_means(
    panel: pd.DataFrame, cfg: CohortConfig, outcome: str
) -> pd.DataFrame:
    """Per-property mean outcome in the pre and post windows."""
    treated = panel[panel["cohort"] == "treated"].copy()
    treated["week"] = pd.to_datetime(treated["week"])

    pre = treated[(treated["week"] >= cfg.pre_start) & (treated["week"] <= cfg.pre_end)]
    post = treated[(treated["week"] >= cfg.post_start) & (treated["week"] <= cfg.post_end)]

    # Same calendar weeks one year earlier, as the seasonal reference for the
    # post window. Comparing post to a full-year pre average would confound
    # the treatment with seasonality.
    ly = treated.copy()
    ly["week"] = ly["week"] + pd.Timedelta(weeks=52)
    ly_post = ly[(ly["week"] >= cfg.post_start) & (ly["week"] <= cfg.post_end)]

    frame = (
        post.groupby(["property_id", "stack_eligible"], as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "post_mean"})
    )
    frame = frame.merge(
        pre.groupby("property_id", as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "pre_mean"}),
        on="property_id",
    )
    frame = frame.merge(
        ly_post.groupby("property_id", as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "ly_post_mean"}),
        on="property_id",
        how="left",
    )
    frame = frame[(frame["pre_mean"] > 0) & (frame["ly_post_mean"] > 0)]
    # Seasonally-referenced growth: this property's post-window level versus
    # the same weeks last year.
    frame["growth"] = np.log(frame["post_mean"] / frame["ly_post_mean"])
    frame["pre_growth"] = np.log(frame["pre_mean"] / frame["ly_post_mean"])
    return frame


def estimate_stacking_value(
    panel: pd.DataFrame, cfg: CohortConfig, outcome: str | None = None
) -> StackingResult:
    """Difference-in-differences estimate of the marginal value of stacking."""
    outcome = outcome or cfg.primary_outcome
    frame = _property_window_means(panel, cfg, outcome)
    if frame["stack_eligible"].nunique() < 2:
        raise ValueError("need both stack-enabled and stack-disabled properties")

    stack = frame[frame["stack_eligible"] == 1]
    no_stack = frame[frame["stack_eligible"] == 0]

    # -- parallel trends: do the two groups track each other pre-campaign? --
    trend_stat = stats.ttest_ind(
        stack["pre_growth"], no_stack["pre_growth"], equal_var=False
    )
    parallel_p = float(trend_stat.pvalue)

    # -- CUPED adjustment on the pre-period covariate -----------------------
    y = frame["growth"].to_numpy(dtype=float)
    x = frame["pre_growth"].to_numpy(dtype=float)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x) > 0 else 0.0
    y_adj = y - theta * (x - x.mean())
    variance_reduction = 1.0 - float(np.var(y_adj, ddof=1) / np.var(y, ddof=1))

    frame = frame.assign(growth_adj=y_adj)
    stack_adj = frame.loc[frame["stack_eligible"] == 1, "growth_adj"].to_numpy()
    no_stack_adj = frame.loc[frame["stack_eligible"] == 0, "growth_adj"].to_numpy()

    diff = float(stack_adj.mean() - no_stack_adj.mean())
    se = float(
        np.sqrt(
            stack_adj.var(ddof=1) / stack_adj.size
            + no_stack_adj.var(ddof=1) / no_stack_adj.size
        )
    )
    test = stats.ttest_ind(stack_adj, no_stack_adj, equal_var=False)
    z = stats.norm.ppf(0.95)

    # Log-point differences convert to a multiplicative effect.
    return StackingResult(
        n_stack=int(stack.shape[0]),
        n_no_stack=int(no_stack.shape[0]),
        stack_uplift=float(np.expm1(stack_adj.mean())),
        no_stack_uplift=float(np.expm1(no_stack_adj.mean())),
        incremental_stacking_value=float(np.expm1(diff)),
        ci=(float(np.expm1(diff - z * se)), float(np.expm1(diff + z * se))),
        p_value=float(test.pvalue),
        parallel_trends_p=parallel_p,
        variance_reduction=variance_reduction,
    )
