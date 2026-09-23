"""Translate a volume lift into money, and then into a decision.

A relative uplift is not a result. "We drove $31M of net incremental booking
value, which clears break-even at 1.3x, and the programme stops paying for
itself below a 7.4% lift" is a result, because it tells the next person what
to do.

## The double-counting trap

The most common error in promotional ROI, and one this module is built to
prevent: **charging the full discount as a cost when the outcome metric is
measured at transacted prices.**

GBV is booking value at the price the traveller actually paid. So the treated
series already reflects the discount -- the counterfactual is full-price and
lower-volume, the actual is discounted-price and higher-volume, and the
difference between them is the *net* effect with the price reduction already
subtracted. Subtracting the discount again double-counts it, and it is a
large enough number to flip a profitable campaign to unprofitable on paper.

What legitimately remains as cost:

* the share of the discount funded in cash by us rather than by the supply
  partner -- a real outflow on top of the price reduction,
* variable servicing cost on the incremental volume,
* fixed media, creative and ops cost.

The subsidy-on-baseline figure below is still worth reporting even though it
is not a P&L line: it is the share of the discount that went to travellers
who were going to book anyway, and it is the number that makes the case for
targeting the promotion more tightly next time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import EconomicsConfig


@dataclass
class EconomicsResult:
    incremental_bookings: float
    baseline_bookings: float
    total_bookings: float
    incremental_revenue: float
    incremental_contribution: float
    discounted_bookings: float
    discount_value_granted: float
    subsidy_on_baseline: float
    subsidy_on_incremental: float
    funded_discount_cost: float
    fixed_cost: float
    total_cost: float
    net_value: float
    return_on_spend: float
    net_roi: float
    breakeven_relative_uplift: float
    breakeven_discount_depth: float

    def summary_rows(self) -> pd.DataFrame:
        """Waterfall-ready view, in presentation order."""
        rows = [
            ("Incremental booking value", self.incremental_bookings),
            ("Incremental contribution", self.incremental_contribution),
            ("Cash-funded discount", -self.funded_discount_cost),
            ("Fixed campaign cost", -self.fixed_cost),
            ("Net value", self.net_value),
        ]
        return pd.DataFrame(rows, columns=["line_item", "usd"])

    def pnl_rows(self) -> pd.DataFrame:
        """P&L lines only, for the value-build chart.

        Incremental booking value is deliberately excluded: it is a volume
        figure an order of magnitude larger than the margin and cost lines,
        and plotting them on one axis makes the decision-relevant bars
        invisible. It belongs in the memo table, not the waterfall.
        """
        rows = [
            ("Incremental\ncontribution", self.incremental_contribution),
            ("Cash-funded\ndiscount", -self.funded_discount_cost),
            ("Fixed campaign\ncost", -self.fixed_cost),
            ("Net value", self.net_value),
        ]
        return pd.DataFrame(rows, columns=["line_item", "usd"])

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def _discount_granted(discounted_bookings: float, depth: float) -> float:
    """Face value of the discount on a given volume of discounted bookings.

    ``discounted_bookings`` is post-discount value, so the pre-discount
    equivalent is ``v / (1 - depth)`` and the discount granted is the
    difference.
    """
    if not 0.0 <= depth < 1.0:
        raise ValueError(f"discount_depth must be in [0, 1), got {depth}")
    return discounted_bookings / (1.0 - depth) * depth


def evaluate(
    *,
    incremental_bookings: float,
    counterfactual_bookings: float,
    actual_bookings: float,
    econ: EconomicsConfig,
) -> EconomicsResult:
    """Full P&L and break-even evaluation for one campaign.

    Parameters
    ----------
    incremental_bookings
        Net incremental booking value, after any displacement adjustment.
        Already net of the price reduction (see module docstring).
    counterfactual_bookings
        Modelled no-campaign booking value over the post window.
    actual_bookings
        Observed booking value over the post window.
    """
    margin_rate = econ.take_rate - econ.variable_cost_rate
    if margin_rate <= 0:
        raise ValueError("take_rate must exceed variable_cost_rate")

    incremental_revenue = incremental_bookings * econ.take_rate
    incremental_contribution = incremental_bookings * margin_rate

    discounted_bookings = actual_bookings * econ.discounted_booking_share
    discount_granted = _discount_granted(discounted_bookings, econ.discount_depth)
    funded = discount_granted * econ.discount_funded_share

    total_cost = funded + econ.fixed_campaign_cost
    net_value = incremental_contribution - total_cost
    return_on_spend = incremental_contribution / total_cost if total_cost > 0 else np.nan
    net_roi = net_value / total_cost if total_cost > 0 else np.nan

    # Where the discount landed. Incremental demand is assumed to transact on
    # the promotional rate, so everything above that is subsidy to travellers
    # who would have booked regardless.
    if discounted_bookings > 0:
        incremental_share = (
            min(max(incremental_bookings, 0.0), discounted_bookings) / discounted_bookings
        )
    else:
        incremental_share = float("nan")
    subsidy_on_incremental = discount_granted * incremental_share
    subsidy_on_baseline = discount_granted - subsidy_on_incremental

    return EconomicsResult(
        incremental_bookings=incremental_bookings,
        baseline_bookings=counterfactual_bookings,
        total_bookings=actual_bookings,
        incremental_revenue=incremental_revenue,
        incremental_contribution=incremental_contribution,
        discounted_bookings=discounted_bookings,
        discount_value_granted=discount_granted,
        subsidy_on_baseline=subsidy_on_baseline,
        subsidy_on_incremental=subsidy_on_incremental,
        funded_discount_cost=funded,
        fixed_cost=econ.fixed_campaign_cost,
        total_cost=total_cost,
        net_value=net_value,
        return_on_spend=return_on_spend,
        net_roi=net_roi,
        breakeven_relative_uplift=breakeven_relative_uplift(
            counterfactual_bookings=counterfactual_bookings, econ=econ
        ),
        breakeven_discount_depth=_solve_breakeven_depth(
            incremental_bookings=incremental_bookings,
            actual_bookings=actual_bookings,
            econ=econ,
        ),
    )


def breakeven_relative_uplift(
    *, counterfactual_bookings: float, econ: EconomicsConfig
) -> float:
    """The lift at which net value is exactly zero.

    Solved rather than approximated, because the cost side moves with the
    answer: a bigger lift means more discounted volume, which means more
    cash-funded discount. Writing

        contribution(L) = m * C * L
        funded(L)       = C * (1 + L) * s * d / (1 - d) * f

    and setting ``contribution - funded - F = 0`` gives

        L = (F + C * k) / (C * (m - k)),   k = s * d / (1 - d) * f

    A non-positive denominator means the promotion loses money at every lift:
    each additional discounted booking costs more in funded discount than it
    returns in margin, so no volume response can rescue it. That is a finding,
    not a numerical failure, so it returns inf rather than raising.
    """
    margin_rate = econ.take_rate - econ.variable_cost_rate
    k = (
        econ.discounted_booking_share
        * econ.discount_depth
        / (1.0 - econ.discount_depth)
        * econ.discount_funded_share
    )
    denominator = counterfactual_bookings * (margin_rate - k)
    if denominator <= 0:
        return float("inf")
    return float((econ.fixed_campaign_cost + counterfactual_bookings * k) / denominator)


def _solve_breakeven_depth(
    *, incremental_bookings: float, actual_bookings: float, econ: EconomicsConfig
) -> float:
    """Deepest discount that still breaks even, holding the lift fixed.

    Holding volume fixed is conservative in one direction and optimistic in
    the other: a shallower discount would presumably have produced less lift.
    Reported as a ceiling on depth, not a forecast.
    """
    margin_rate = econ.take_rate - econ.variable_cost_rate
    available = incremental_bookings * margin_rate - econ.fixed_campaign_cost
    if available <= 0:
        return 0.0
    base = actual_bookings * econ.discounted_booking_share * econ.discount_funded_share
    if base <= 0:
        return 0.99
    k = available / base
    return float(min(k / (1.0 + k), 0.99))


def sensitivity_grid(
    *,
    counterfactual_bookings: float,
    econ: EconomicsConfig,
    lift_range: tuple[float, float] = (0.0, 0.16),
    depth_range: tuple[float, float] = (0.04, 0.28),
    n: int = 13,
) -> pd.DataFrame:
    """Net value across the lift x discount-depth plane.

    This is the exhibit that changes next year's campaign design: it shows
    the region where the promotion is worth running at all, and how little
    room there is between the observed lift and the break-even line.
    """
    rows = []
    for lift in np.linspace(*lift_range, n):
        for depth in np.linspace(*depth_range, n):
            local = replace(econ, discount_depth=float(depth))
            actual = counterfactual_bookings * (1.0 + lift)
            result = evaluate(
                incremental_bookings=counterfactual_bookings * lift,
                counterfactual_bookings=counterfactual_bookings,
                actual_bookings=actual,
                econ=local,
            )
            rows.append(
                {
                    "relative_uplift": float(lift),
                    "discount_depth": float(depth),
                    "net_value": result.net_value,
                    "return_on_spend": result.return_on_spend,
                }
            )
    return pd.DataFrame(rows)
