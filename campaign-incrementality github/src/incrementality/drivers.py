"""Decompose the booking-value lift into its drivers.

A campaign readout that stops at "+9.3% GBV" cannot answer the question the
business actually asks next, which is *how*. Traffic, conversion, length of
stay and rate imply completely different follow-up actions, and a campaign
that lifts conversion while giving back rate is a different animal from one
that buys traffic at full price.

The identity used is multiplicative:

    GBV = visits x CVR x LoS x ADR
    RN = visits x CVR x LoS

so each driver is run through the identical counterfactual machinery and the
lifts compose. Because the estimates are made independently, their product
will not reconcile to the GBV estimate exactly -- the residual is reported
rather than hidden, and a large residual is a signal that something is wrong
with the panel (usually a ratio metric that was averaged instead of being
rebuilt from summed levels).

Two things this makes visible that the headline number hides:

* **Room nights lift more than booking value** when the discount is real.
  ADR falls, so RN outruns GBV. If they move together, either the discount
  did not reach travellers or the panel is mis-specified.
* **Conversion versus traffic.** Conversion lift is the deal working on
  demand that was already arriving. Traffic lift is the promotion bringing
  new demand. The first scales with eligibility, the second with media.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CohortConfig
from .counterfactual import get_backend
from .features import build_design
from .inference import UpliftResult, estimate_uplift

# Which drivers multiply to which outcome.
COMPOSITION: dict[str, tuple[str, ...]] = {
    "gbv": ("visits", "cvr", "los", "adr"),
    "room_nights": ("visits", "cvr", "los"),
    "bookings": ("visits", "cvr"),
}

LABELS = {
    "visits": "Traffic (visits)",
    "cvr": "Conversion rate",
    "los": "Length of stay",
    "adr": "Average daily rate",
}


@dataclass
class DriverDecomposition:
    outcome: str
    outcome_lift: float
    drivers: dict[str, UpliftResult] = field(default_factory=dict)

    @property
    def composed_lift(self) -> float:
        """Product of the driver lifts, which should reproduce the outcome."""
        product = 1.0
        for name in COMPOSITION[self.outcome]:
            if name in self.drivers:
                product *= 1.0 + self.drivers[name].relative_uplift
        return product - 1.0

    @property
    def residual(self) -> float:
        return self.outcome_lift - self.composed_lift

    @property
    def reconciles(self) -> bool:
        """Within half a point is close enough for a decomposition exhibit."""
        return abs(self.residual) < 0.005

    def to_frame(self) -> pd.DataFrame:
        """Presentation-ready table, with each driver's share of the lift."""
        rows = []
        for name in COMPOSITION[self.outcome]:
            if name not in self.drivers:
                continue
            result = self.drivers[name]
            # Log contribution apportions the multiplicative lift additively,
            # which is the only way a share column sums to something sensible
            # when one driver is negative.
            rows.append(
                {
                    "driver": LABELS.get(name, name),
                    "metric": name,
                    "lift": result.relative_uplift,
                    "ci_low": result.ci_relative[0],
                    "ci_high": result.ci_relative[1],
                    "log_contribution": np.log1p(result.relative_uplift),
                    "significant": result.is_significant,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        total_log = frame["log_contribution"].abs().sum()
        frame["share_of_movement"] = frame["log_contribution"].abs() / total_log
        return frame

    def narrative(self) -> str:
        """One sentence a stakeholder can repeat without the table."""
        frame = self.to_frame()
        if frame.empty:
            return "Driver decomposition unavailable."
        gainers = frame[frame["lift"] > 0].sort_values("log_contribution", ascending=False)
        givers = frame[frame["lift"] < 0].sort_values("log_contribution")
        parts = []
        if not gainers.empty:
            top = gainers.iloc[0]
            parts.append(f"{top['driver'].lower()} at {top['lift']:+.1%}")
            if len(gainers) > 1:
                second = gainers.iloc[1]
                parts.append(f"{second['driver'].lower()} at {second['lift']:+.1%}")
        text = f"The {self.outcome.upper()} lift is carried by " + " and ".join(parts)
        if not givers.empty:
            worst = givers.iloc[0]
            text += f", partly given back through {worst['driver'].lower()} at {worst['lift']:+.1%}"
        return text + "."


def decompose(
    series: pd.DataFrame,
    cfg: CohortConfig,
    outcome_result: UpliftResult,
    *,
    outcome: str | None = None,
    n_bootstrap: int | None = None,
) -> DriverDecomposition:
    """Estimate each driver's lift with the same counterfactual as the outcome."""
    outcome = outcome or cfg.primary_outcome
    if outcome not in COMPOSITION:
        raise KeyError(
            f"no driver composition defined for {outcome!r}; "
            f"known: {sorted(COMPOSITION)}"
        )

    prior_window = None
    if cfg.model.control_prior_campaign:
        prior_start = cfg.activation_week - pd.Timedelta(weeks=52)
        prior_window = (
            prior_start,
            prior_start + pd.Timedelta(weeks=cfg.model.prior_campaign_weeks - 1),
        )

    decomposition = DriverDecomposition(
        outcome=outcome, outcome_lift=outcome_result.relative_uplift
    )
    for name in COMPOSITION[outcome]:
        if name not in series.columns:
            continue
        design = build_design(
            series,
            outcome=name,
            regressors=cfg.model.predictors,
            pre_window=(cfg.pre_start, cfg.pre_end),
            post_window=(cfg.post_start, cfg.post_end),
            log_transform=cfg.model.log_transform,
            annual_fourier_terms=cfg.model.annual_fourier_terms,
            holiday_iso_weeks=cfg.model.holiday_iso_weeks,
            prior_campaign_window=prior_window,
        prior_campaign_shape=(
            cfg.model.prior_campaign_ramp_weeks,
            cfg.model.prior_campaign_decay,
        ),
        )
        result = estimate_uplift(
            get_backend(cfg.model.backend),
            design,
            alpha=cfg.model.alpha,
            n_bootstrap=n_bootstrap or cfg.model.n_bootstrap,
            block_length=cfg.model.block_length,
            backend_name=cfg.model.backend,
            materiality_threshold=cfg.materiality_threshold,
        )
        result.outcome = name
        decomposition.drivers[name] = result

    return decomposition
