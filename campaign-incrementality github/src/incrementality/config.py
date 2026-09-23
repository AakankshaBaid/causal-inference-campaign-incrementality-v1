"""Typed, validated configuration for a single measurement cohort.

Every campaign we measure is described by one YAML file. Nothing about a
campaign is hard-coded in the analysis code, which is what lets a new analyst
measure a new campaign by writing 20 lines of YAML instead of forking a
notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Marketplace-wide demand signals. Contemporaneous, and outside the lodging
# promotion: metasearch and paid-search demand, destination intent, and total
# site traffic across every market.
MARKETPLACE_DEMAND = [
    "meta_impressions",
    "search_clicks",
    "destination_queries",
    "market_visitors",
]

MARKET_DEMAND = [
    "meta_impressions",
    "destination_queries",
    "market_visitors",
    "market_purchases",
]

PREDICTOR_SETS: dict[str, list[str]] = {
    # Named after the iteration history of the measurement design, so a config
    # states which known trade-off it is accepting rather than listing opaque
    # column names.
    #
    # Design A -- comp-set properties as the control series.
    #   Precise, because competitors move with the same market shocks week by
    #   week. Biased, because the campaign displaces them: the model reads
    #   depressed competitors as weak market demand and inflates the lift.
    "competitor_control": ["compset_volume"],
    #
    # Design B -- market demand signals plus participants' own prior-year production.
    #   Removes the contemporaneous contamination, introduces a worse one.
    #   With high repeat participation the prior-year series contains last
    #   year's campaign lift, so the model believes the baseline was already
    #   elevated and UNDER-states this year's effect.
    "own_prior_year": ["outcome_ly", *MARKET_DEMAND],
    #
    # Design C -- market demand signals plus marketplace-wide prior-year production.
    #   Participants are a minority of the marketplace, so last year's lift is
    #   diluted and the predictor is far cleaner. Cost: a year-lagged series
    #   carries no information about *this* year's market shocks, so precision
    #   falls back on the noisy demand covariates.
    "market_prior_year": ["market_total_ly", *MARKET_DEMAND],
    #
    # Design D -- Design C plus a contemporaneous control series drawn from untreated
    #   properties OUTSIDE participants' competitive sets.
    #   Recovers Design A's precision without Design A's contamination, because these
    #   properties are exposed to the same market and not to the campaign.
    #   Requires reserving a clean donor pool at campaign design time.
    "clean_pool_control": ["clean_control_volume", "market_total_ly", *MARKET_DEMAND],
    #
    # Pruned -- the same design with predictors that carry no measurable
    # out-of-sample value removed. Motivated by `diagnostics.predictor_value`,
    # which found three of the six market-demand covariates actively degrade
    # the counterfactual forecast: with a clean contemporaneous control series
    # present they add parameters without adding signal. Kept as a separate
    # named design rather than silently editing the default, so the pruning
    # decision stays visible and reversible.
    "pruned_control": ["clean_control_volume", "market_total_ly", "meta_impressions"],
    #
    # Marketplace control -- the design used for globally simultaneous
    # campaigns, and the default.
    #   Contemporaneous predictors are marketplace-wide *demand* only: clicks,
    #   metasearch, destination intent, site traffic. Production enters only
    #   lagged, as marketplace-wide volume one year back across lodging AND
    #   the non-lodging product lines.
    #   Production is deliberately excluded from the contemporaneous block.
    #   Today's marketplace total contains the treated properties, so using it
    #   would regress the outcome partly on itself and shrink the effect.
    #   Lagging removes that: last year's total cannot be moved by this year's
    #   campaign, however global the campaign is.
    #   The residual exposure is that campaign marketing lifts marketplace
    #   demand itself, which is strongest in peak trading weeks -- so the
    #   design is sounder off-peak than on.
    "marketplace_control": ["marketplace_ex_participants_ly", *MARKETPLACE_DEMAND],
    #
    # Cross-product control -- contemporaneous, and structurally outside a
    # lodging promotion.
    #   Non-lodging production (air, car, activities) runs in every market,
    #   moves with the same demand conditions week by week, and is not
    #   discounted by a lodging merchandising deal. That combination is what
    #   no untreated *lodging* series can offer once the campaign is global.
    #
    #   It also sidesteps a trap in any lagged aggregate: when a campaign
    #   recurs annually, last year's production is elevated during exactly the
    #   weeks being measured, because the lagged series is showing last year's
    #   campaign. A control that spikes when the treatment does pulls the
    #   counterfactual up and eats the effect.
    #
    #   Residual exposure: cross-sell. A traveller taking the hotel deal may
    #   attach a car, so the control carries a little of the campaign's own
    #   lift and the estimate is mildly conservative.
    "cross_product_control": ["other_products_volume", *MARKETPLACE_DEMAND],
    #
    # Design E -- negative control outcome.
    #   For campaigns that run in every market at once, no untreated market
    #   exists and Designs A and D are both unavailable: every untreated
    #   property has discounting competitors. The only contemporaneous series
    #   left is *within* the treated properties -- revenue the promotion
    #   cannot reach (corporate and negotiated rates, non-qualifying room
    #   types, stays outside the promotional window). It shares the market
    #   confounders and not the treatment, which is the definition of a
    #   negative control outcome.
    #   Fails if the promotion cannibalises the ineligible segment, so that
    #   substitution has to be tested, not assumed.
    "negative_control": ["nonqualifying_volume", "market_total_ly", "meta_impressions"],
}

# Which designs are available depends on how the campaign was deployed. This
# is a deployment fact, not a modelling preference, and getting it wrong is
# the most expensive error available: a globally simultaneous campaign has no
# untreated market, so any design resting on one silently inherits the
# displacement it was meant to avoid.
SCOPE_DESIGNS = {
    "global_simultaneous": (
        "cross_product_control",
        "marketplace_control",
        "market_prior_year",
        "own_prior_year",
        "negative_control",
    ),
    "partial_holdout": tuple(PREDICTOR_SETS),
}


@dataclass(frozen=True)
class EconomicsConfig:
    """Unit economics needed to turn a volume lift into a P&L number."""

    take_rate: float = 0.155
    """Share of gross booking value we retain as revenue."""

    variable_cost_rate: float = 0.030
    """Payment, servicing and support cost as a share of booking value."""

    discount_depth: float = 0.12
    """Average price reduction on bookings made at the promotional rate."""

    discounted_booking_share: float = 0.42
    """Share of post-window booking value transacted on the promotional rate.

    Not every booking at a participating property uses the deal: travellers
    book non-qualifying room types, dates outside the promotional window, or
    rates sourced through channels the deal does not touch. Charging the
    discount against the property's entire volume is the second most common
    error in promo ROI after double-counting it.
    """

    discount_funded_share: float = 0.10
    """Share of the discount paid in cash by us rather than the partner.

    For partner-marketing campaigns the supply partner funds most of the
    price reduction; our cash exposure is the co-funded remainder. This is
    the only part of the discount that is a genuine incremental cost, because
    the price reduction itself is already inside the measured GBV.
    """

    fixed_campaign_cost: float = 1_200_000.0
    """Media, creative and ops cost booked against the campaign."""

    def __post_init__(self) -> None:
        for name in (
            "take_rate",
            "variable_cost_rate",
            "discount_depth",
            "discounted_booking_share",
            "discount_funded_share",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"economics.{name} must be in [0, 1], got {value}")
        if self.fixed_campaign_cost < 0:
            raise ValueError("economics.fixed_campaign_cost must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    """Which counterfactual estimator to run, and how."""

    backend: str = "bsts"
    """One of: bsts, synthetic_control, prophet."""

    design_version: str | None = "cross_product_control"
    """Named predictor set from PREDICTOR_SETS. Overrides `regressors`."""

    regressors: list[str] = field(default_factory=list)
    """Explicit predictor list. Used only when design_version is null."""

    control_prior_campaign: bool = True
    """Add an indicator for last year's campaign window.

    If the same campaign ran a year earlier over largely the same properties,
    the pre-period contains a treatment episode. Left unmodelled, the fit
    absorbs last year's lift as ordinary seasonality, raises the
    counterfactual and under-states this year's effect -- a second channel of
    repeat-participation bias, distinct from the prior-year predictor itself.
    """

    prior_campaign_weeks: int = 10
    """Length of the prior-year campaign window, in weeks."""

    prior_campaign_ramp_weeks: int = 2
    prior_campaign_decay: float = 0.06
    """Shape of the prior-year effect: ramp, then geometric fade.

    The correction is a shaped profile rather than a flat on/off block. A
    block dummy removes the whole window's level, and for an annually
    recurring campaign that window sits at the same calendar position as the
    measurement window -- so the dummy strips exactly the seasonality the
    forecast depends on. Measured cost of that side effect: 0.8 points of
    downward bias even when there is no prior campaign to correct. A shaped
    term removes the campaign's profile and leaves the season alone.
    """
    """Length of the prior-year campaign window, in weeks."""

    annual_fourier_terms: int = 2
    """Deterministic annual seasonality, mirroring Prophet's yearly component."""

    holiday_iso_weeks: dict[str, list[int]] = field(
        default_factory=lambda: {
            "peak_shopping": [47, 48],
            "holiday_lull": [51, 52],
            "new_year": [1, 2],
        }
    )
    """Named blocks of ISO weeks that get their own indicator."""

    log_transform: bool = True
    """Model log(outcome) so the treatment effect is multiplicative."""

    alpha: float = 0.10
    """1 - alpha is the credible/conformal interval coverage."""

    n_bootstrap: int = 2_000
    block_length: int = 4
    """Moving-block length (weeks) for the residual bootstrap."""

    VALID_BACKENDS = ("bsts", "synthetic_control", "prophet")

    def __post_init__(self) -> None:
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"model.backend must be one of {self.VALID_BACKENDS}, got {self.backend!r}"
            )
        if not 0.0 < self.alpha < 0.5:
            raise ValueError(f"model.alpha must be in (0, 0.5), got {self.alpha}")
        if self.annual_fourier_terms < 0:
            raise ValueError("model.annual_fourier_terms must be non-negative")
        if self.design_version is not None and self.design_version not in PREDICTOR_SETS:
            raise ValueError(
                f"model.design_version must be one of {sorted(PREDICTOR_SETS)}, "
                f"got {self.design_version!r}"
            )
        if self.design_version is None and not self.regressors:
            raise ValueError(
                "set either model.design_version or an explicit model.regressors list"
            )

    @property
    def predictors(self) -> list[str]:
        """The predictor list actually used, resolving the design version."""
        if self.design_version is not None:
            return list(PREDICTOR_SETS[self.design_version])
        return list(self.regressors)


@dataclass(frozen=True)
class CohortConfig:
    """Full specification of one measurement run."""

    name: str
    activation_week: pd.Timestamp
    pre_weeks: int = 78
    post_weeks: int = 8
    transition_weeks: int = 1
    """Weeks after activation that are excluded from both windows.

    Deals do not go live cleanly on day one -- rate plans propagate, caches
    refresh, partners opt out. Scoring the transition week as 'post' drags the
    estimate toward zero, so we drop it.
    """

    campaign_scope: str = "partial_holdout"
    """How the campaign was deployed: global_simultaneous or partial_holdout.

    ``global_simultaneous`` -- the promotion runs in every market at the same
    seasonal moment. No untreated market exists, so contemporaneous
    cross-market controls are unavailable however they are constructed.

    ``partial_holdout`` -- some markets, segments or property groups were
    genuinely unexposed and reserved before launch.
    """

    outcomes: list[str] = field(default_factory=lambda: ["gbv", "room_nights"])
    primary_outcome: str = "gbv"

    materiality_threshold: float = 0.01
    """Smallest relative lift worth acting on, independent of significance.

    A precise estimator makes statistical significance cheap. With a clean
    control series the interval on this design is roughly +/-1 point wide, so
    a +0.05% lift can clear p < 0.10 while being obvious noise to anyone
    reading the slide. Significance answers "is it distinguishable from
    zero"; this answers "is it big enough to care", and the campaign verdict
    requires both.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.activation_week.dayofweek != 0:
            # Weekly panels key on Monday. Silently snapping would hide a
            # config error that shifts every window by a few days.
            raise ValueError(
                f"activation_week {self.activation_week.date()} is a "
                f"{self.activation_week.day_name()}; weekly panels must key on Monday"
            )
        if self.pre_weeks < 26:
            raise ValueError(
                "pre_weeks < 26 leaves too little history to identify annual "
                f"seasonality; got {self.pre_weeks}"
            )
        if self.post_weeks <= self.transition_weeks:
            raise ValueError("post_weeks must exceed transition_weeks")
        if self.campaign_scope not in SCOPE_DESIGNS:
            raise ValueError(
                f"campaign_scope must be one of {sorted(SCOPE_DESIGNS)}, "
                f"got {self.campaign_scope!r}"
            )
        allowed = SCOPE_DESIGNS[self.campaign_scope]
        version = self.model.design_version
        if version is not None and version not in allowed:
            raise ValueError(
                f"design {version!r} needs untreated units that a "
                f"{self.campaign_scope!r} campaign does not have. "
                f"Available designs for this scope: {sorted(allowed)}"
            )
        if not 0.0 <= self.materiality_threshold < 1.0:
            raise ValueError("materiality_threshold must be in [0, 1)")
        if self.primary_outcome not in self.outcomes:
            raise ValueError(
                f"primary_outcome {self.primary_outcome!r} not in outcomes {self.outcomes}"
            )

    # -- derived windows -------------------------------------------------

    @property
    def pre_start(self) -> pd.Timestamp:
        return self.activation_week - pd.Timedelta(weeks=self.pre_weeks)

    @property
    def pre_end(self) -> pd.Timestamp:
        return self.activation_week - pd.Timedelta(weeks=1)

    @property
    def post_start(self) -> pd.Timestamp:
        return self.activation_week + pd.Timedelta(weeks=self.transition_weeks)

    @property
    def post_end(self) -> pd.Timestamp:
        return self.activation_week + pd.Timedelta(weeks=self.post_weeks - 1)

    @property
    def n_post_weeks(self) -> int:
        return self.post_weeks - self.transition_weeks

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["activation_week"] = str(self.activation_week.date())
        out["windows"] = {
            "pre": [str(self.pre_start.date()), str(self.pre_end.date())],
            "post": [str(self.post_start.date()), str(self.post_end.date())],
        }
        return out


def load_cohort(path: str | Path) -> CohortConfig:
    """Load and validate a cohort config from YAML."""
    path = Path(path)
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    model = ModelConfig(**raw.pop("model", {}))
    economics = EconomicsConfig(**raw.pop("economics", {}))
    activation = pd.Timestamp(raw.pop("activation_week"))
    return CohortConfig(activation_week=activation, model=model, economics=economics, **raw)
