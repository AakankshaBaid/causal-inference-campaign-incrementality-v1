"""Generate a weekly property-level panel with a *known* treatment effect.

Why a simulator instead of a data file: the whole point of a counterfactual
estimator is that the ground truth is unobservable, so on real data you can
never check whether your answer was right. Here we know the answer, which
turns "my model produced a number" into "my model recovers a 10.0% lift to
within 0.1pp, and returns a null when there is nothing there."

## Structure of the generating process

Bookings are built from their drivers rather than simulated as a single
series, because the business question is not only "how much" but "through
what":

    GBV = visits x CVR x LoS x ADR
    RN = visits x CVR x LoS

A merchandising campaign moves those four in different directions -- it buys
traffic and conversion and gives back rate -- so a decomposition is the only
way to distinguish "we sold more room nights" from "we sold the same room
nights cheaper". RN therefore lifts *more* than GBV, which is the signature
of a discount-driven campaign and a useful sanity check on real output.

## Three populations

* ``treated``  -- campaign participants.
* ``compset``  -- untreated properties competing directly with participants,
  exposed to displacement. Right units for the cannibalisation test, wrong
  units for a control series.
* ``distant``  -- untreated properties in unrelated markets, not exposed to
  displacement. Safe as a control series and as the placebo pool.

## Repeat participation

Most participants also ran last year's campaign. That makes each property's
own prior-year production contain a prior-year campaign lift, so using it as
a predictor tells the model the counterfactual baseline was already elevated
and *under*-states this year's effect. The simulator reproduces this on
purpose so the bias can be measured rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEK_FREQ = "W-MON"

DRIVERS = ("visits", "cvr", "los", "adr")
RATIO_METRICS = {"cvr": ("bookings", "visits"), "los": ("room_nights", "bookings"), "adr": ("gbv", "room_nights")}


@dataclass(frozen=True)
class GroundTruth:
    """The parameters the estimator is supposed to recover.

    Effects are specified per driver. The GBV lift is their product, so it is
    never set directly -- a campaign that lifts conversion 9% while giving
    back 3% of rate cannot also be declared to lift GBV by 9%.
    """

    visits_lift: float = 0.030
    """Campaign promotion drives incremental traffic to participants."""

    cvr_lift: float = 0.090
    """The main channel: a better price converts the traffic already arriving."""

    los_lift: float = 0.010
    adr_lift: float = -0.030
    """Rate given back through the discount. Negative by construction."""

    ramp_weeks: int = 2
    decay_per_week: float = 0.06
    """Geometric fade after the ramp (novelty decay)."""

    spillover: float = -0.021
    """Effect on directly competing untreated properties (displacement)."""

    nonqualifying_substitution: float = 0.0
    """Leakage of the promotion into the non-qualifying segment.

    If travellers switch from a corporate rate to the promotional rate, the
    negative control is depressed by the campaign and stops being a valid
    control -- the same failure mode as a contaminated donor pool, one level
    down. Zero by default; swept in the sensitivity study.
    """

    cross_sell_halo: float = 0.004
    """Lift on non-lodging product lines during the campaign.

    A lodging promotion does not discount flights or cars, so those product
    lines are structurally outside the treatment -- which is what makes them
    usable as a contemporaneous control when the campaign runs in every
    market at once. They are not perfectly clean: a traveller who books a
    discounted hotel sometimes attaches a car, so a small cross-sell lift
    leaks in. It is set small but non-zero on purpose, so the design's real
    weakness is measured rather than assumed away.
    """

    marketplace_halo: float = 0.0
    """Site-wide traffic lift reaching ALL untreated properties.

    The second spillover channel, and the one geographic distance cannot
    block. Campaign marketing drives incremental visitors to the marketplace
    as a whole; some of that traffic lands on properties that never joined
    the promotion, anywhere in the world. Distance separates a property from
    a discounted *competitor*; it does not separate it from the site's own
    homepage.

    This matters because these campaigns run globally and simultaneously.
    There is no market that is "off" while others are "on", so there is no
    geography where an untreated property is genuinely unexposed. Any
    contemporaneous untreated control series therefore carries some halo.

    Critically, halo biases in the OPPOSITE direction to displacement: it
    inflates the control series, raises the modelled counterfactual, and
    UNDERSTATES the campaign. A design can be contaminated by both at once
    and look unbiased by coincidence.
    """

    distant_spillover: float = 0.0
    """Effect on untreated properties OUTSIDE participants' competitive sets.

    This is the load-bearing assumption of the Design D design, isolated as a knob
    so it can be falsified instead of believed. Design D uses the distant pool as a
    contemporaneous control series, which is only legitimate if the campaign
    does not reach it. Set it to zero and Design D is the best estimator available;
    set it equal to `spillover` and the campaign depresses the whole untreated
    universe, the control series is contaminated exactly as Design A's is, and Design D
    inherits Design A's upward bias.

    Reality is somewhere between. A traveller comparing a discounted hotel
    against one two cities away is rarer than one comparing it against the
    hotel across the street, but it is not impossible -- and destination
    substitution puts a floor under how clean any domestic pool can be. See
    `studies/assumption_sensitivity.py` for the measured break-down point,
    and `control_pool_leakage_test` for the diagnostic that decides which
    design a given campaign can support.
    """

    stacking_bonus: float = 0.028
    """Extra conversion lift where logged-in stacking is also permitted."""

    prior_year_scale: float = 0.95
    """Last year's campaign intensity, relative to this year's."""

    repeat_participation: float = 0.85
    """Share of this year's participants who also ran last year's campaign."""

    relationship_drift: float = 0.0
    """Drift in participants' sensitivity to market demand across the sample.

    The second load-bearing assumption of any counterfactual design: the
    relationship between the control series and the target, fitted on the
    pre-period, still holds through the post-period. Estimators do not check
    this, and when it fails they fail quietly -- the counterfactual is built
    from a relationship that no longer applies, and the error goes straight
    into the effect.

    This parameter breaks the assumption on demand. Participants' market
    elasticity is scaled from ``1 - drift`` at the start of the sample to
    ``1 + drift`` at the end, so the pre-period relationship is systematically
    stale by the time the campaign runs. At zero the assumption holds exactly.
    Used by `relationship_stability_test` to prove the diagnostic has power,
    and by `studies/assumption_sensitivity.py` to price the failure.
    """

    @property
    def composite_nbv_lift(self) -> float:
        return (
            (1 + self.visits_lift)
            * (1 + self.cvr_lift)
            * (1 + self.los_lift)
            * (1 + self.adr_lift)
            - 1.0
        )


@dataclass(frozen=True)
class SimulationSpec:
    n_treated: int = 2_500
    n_compset: int = 3_000
    n_distant: int = 4_000
    activation_week: str = "2024-11-25"
    pre_weeks: int = 78
    post_weeks: int = 10
    campaign_length: int = 10
    """Weeks the deal stays live. Bounds the prior-year effect window too."""

    stack_eligible_share: float = 0.45

    nonqualifying_correlation: float = 0.55
    """How much demand noise the ineligible segment shares with the eligible one.

    The negative control's value depends entirely on this. At 1.0 it is a
    perfect mirror and the design looks unbeatable; at 0.0 it is an unrelated
    series and adds nothing beyond market seasonality. Real ineligible
    segments -- corporate contracts, non-qualifying room types -- sit
    somewhere in between, which is why it is a parameter to be estimated from
    the pre-period rather than a number to assume.
    """

    non_qualifying_share: float = 0.32
    """Share of each property's revenue that cannot use the promotion.

    Corporate and negotiated rates, non-qualifying room types, and stays
    outside the promotional window. This segment sits at the *same*
    properties, sees the *same* market conditions, and is not eligible for
    the discount -- which makes it a negative control outcome: a series that
    shares the confounders but not the treatment.

    It matters because a globally simultaneous campaign leaves no untreated
    market to use as a control. When every property everywhere is exposed,
    the only remaining contemporaneous comparison is *within* the property.

    Modelled as a parallel revenue stream rather than by re-splitting the
    primary outcome, so the headline lift stays the lift on eligible volume.
    """

    seed: int = 20240101
    truth: GroundTruth = GroundTruth()


def _shape(week_index: np.ndarray, truth: GroundTruth, length: int) -> np.ndarray:
    """Ramp-then-decay intensity by weeks since activation, zero outside."""
    live = (week_index >= 0) & (week_index < length)
    ramp = np.clip((week_index + 1) / max(truth.ramp_weeks, 1), 0.0, 1.0)
    past_ramp = np.clip(week_index - truth.ramp_weeks + 1, 0, None)
    decay = (1.0 - truth.decay_per_week) ** past_ramp
    return np.where(live, ramp * decay, 0.0)


def _annual_seasonality(weeks: pd.DatetimeIndex) -> np.ndarray:
    doy = weeks.dayofyear.to_numpy(dtype=float)
    phase = 2 * np.pi * doy / 365.25
    return 1.0 + 0.22 * np.sin(phase - 1.1) + 0.07 * np.sin(2 * phase + 0.4)


def _holiday_index(weeks: pd.DatetimeIndex) -> np.ndarray:
    idx = np.ones(len(weeks))
    iso_week = weeks.isocalendar().week.to_numpy()
    idx[np.isin(iso_week, [47, 48])] *= 1.34   # BFCM shopping window
    idx[np.isin(iso_week, [1, 2])] *= 1.18     # new-year trip planning
    idx[np.isin(iso_week, [51, 52])] *= 0.82   # holiday booking lull
    return idx


def simulate_panel(spec: SimulationSpec | None = None) -> pd.DataFrame:
    """Return a tidy weekly panel: one row per property-week.

    Columns
    -------
    property_id, week, cohort, control_role (after ``split_control_pool``),
    stack_eligible, repeat_participant, deal_live
    visits, bookings, room_nights, gbv          -- levels
    cvr, los, adr                       -- driver ratios
    outcome_ly                          -- this property's GBV, prior ISO year
    meta_impressions, destination_queries,
    market_visitors, market_purchases   -- market demand covariates
    """
    spec = spec or SimulationSpec()
    rng = np.random.default_rng(spec.seed)

    activation = pd.Timestamp(spec.activation_week)
    # Two extra years of burn-in so prior-year columns are never null.
    start = activation - pd.Timedelta(weeks=spec.pre_weeks + 104)
    end = activation + pd.Timedelta(weeks=spec.post_weeks)
    weeks = pd.date_range(start, end, freq=WEEK_FREQ)
    n_weeks = len(weeks)

    # -- shared market factor: AR(1) in logs, common to all properties ----
    shock = np.zeros(n_weeks)
    innovation = rng.normal(0.0, 0.055, n_weeks)
    for t in range(1, n_weeks):
        shock[t] = 0.72 * shock[t - 1] + innovation[t]
    market_factor = np.exp(shock)

    seasonal = _annual_seasonality(weeks)
    holiday = _holiday_index(weeks)
    trend = np.exp(np.linspace(0.0, 0.11, n_weeks))
    traffic_level = market_factor * seasonal * holiday * trend

    # Drift in participants' seasonal sensitivity, applied to participants
    # only, so the *relationship* between target and control changes while the
    # market itself does not. Seasonality is the right channel to drift rather
    # than the market factor: a local-level state absorbs slow changes in
    # amplitude, but the campaign window sits on the seasonal peak, so a
    # participant cohort whose holiday response is shifting is mis-predicted
    # exactly where it counts.
    elasticity = 1.0 + spec.truth.relationship_drift * np.linspace(-1.0, 1.0, n_weeks)
    # Rate is seasonal but far less volatile than traffic.
    adr_level = 1.0 + 0.09 * (seasonal - 1.0) + 0.04 * (market_factor - 1.0)

    # -- properties --------------------------------------------------------
    n_props = spec.n_treated + spec.n_compset + spec.n_distant
    prop_ids = np.arange(100_000, 100_000 + n_props)
    position = np.arange(n_props)
    cohort = np.where(
        position < spec.n_treated,
        "treated",
        np.where(position < spec.n_treated + spec.n_compset, "compset", "distant"),
    )
    is_treated = (cohort == "treated")[:, None]
    is_compset = (cohort == "compset")[:, None]

    base_visits = np.exp(rng.normal(np.log(1_300), 0.80, n_props))
    base_cvr = rng.beta(4.0, 150.0, n_props).clip(0.004, 0.13)
    base_los = rng.normal(2.45, 0.42, n_props).clip(1.05, 6.0)
    base_adr = rng.normal(188.0, 34.0, n_props).clip(70, 420)

    stack_eligible = (rng.random(n_props) < spec.stack_eligible_share).astype(int)
    stack_eligible[cohort != "treated"] = 0
    repeat = (rng.random(n_props) < spec.truth.repeat_participation).astype(int)
    repeat[cohort != "treated"] = 0

    # -- effect timing -----------------------------------------------------
    weeks_since = ((weeks - activation).days // 7).to_numpy()
    shape_now = _shape(weeks_since, spec.truth, spec.campaign_length)
    # Last year's campaign occupied the same ISO weeks, 52 weeks earlier.
    shape_prior = _shape(weeks_since + 52, spec.truth, spec.campaign_length)
    shape_spill = np.where(
        (weeks_since >= 0) & (weeks_since < spec.campaign_length), 1.0, 0.0
    )

    t = spec.truth
    # Stacking lifts conversion specifically, not traffic or rate.
    stack_extra = t.stacking_bonus * stack_eligible[:, None]

    def driver_effect(base_lift: float, extra: np.ndarray | float = 0.0) -> np.ndarray:
        """Multiplicative factor per (property, week) for one driver."""
        this_year = (base_lift + extra) * shape_now[None, :]
        last_year = (
            (base_lift * t.prior_year_scale + extra * t.prior_year_scale)
            * shape_prior[None, :]
            * repeat[:, None]
        )
        treated_path = 1.0 + this_year + last_year
        # Displacement hits competitors' traffic and conversion, not their rate.
        return np.where(is_treated, treated_path, 1.0)

    visit_effect = driver_effect(t.visits_lift)
    cvr_effect = driver_effect(t.cvr_lift, stack_extra)
    los_effect = driver_effect(t.los_lift)
    adr_effect = driver_effect(t.adr_lift)

    # Displacement hits competitors' conversion hardest and their traffic
    # about half as much: a traveller still arrives, then books elsewhere.
    spill = 1.0 + t.spillover * shape_spill[None, :]
    half_spill = 1.0 + 0.5 * t.spillover * shape_spill[None, :]
    visit_effect = np.where(is_compset, half_spill, visit_effect)
    cvr_effect = np.where(is_compset, spill, cvr_effect)

    # Leakage past the competitive-set boundary, onto the pool Design D relies on.
    is_distant = (cohort == "distant")[:, None]
    if t.distant_spillover != 0.0:
        far_spill = 1.0 + t.distant_spillover * shape_spill[None, :]
        far_half = 1.0 + 0.5 * t.distant_spillover * shape_spill[None, :]
        visit_effect = np.where(is_distant, far_half, visit_effect)
        cvr_effect = np.where(is_distant, far_spill, cvr_effect)

    # Site-wide halo: applies to every untreated property regardless of
    # distance, because it travels through the marketplace's own traffic
    # rather than through local competition. Acts on visits, which is the
    # channel campaign marketing actually moves.
    if t.marketplace_halo != 0.0:
        halo = 1.0 + t.marketplace_halo * shape_spill[None, :]
        visit_effect = np.where(is_treated, visit_effect, visit_effect * halo)

    # -- compose -----------------------------------------------------------
    noise = lambda sd: np.exp(rng.normal(0.0, sd, (n_props, n_weeks)))  # noqa: E731
    treated_traffic = (
        market_factor * trend * (seasonal * holiday) ** elasticity
    )[None, :]
    level = np.where(is_treated, treated_traffic, traffic_level[None, :])
    visits = base_visits[:, None] * level * noise(0.16) * visit_effect
    cvr = (base_cvr[:, None] * noise(0.10) * cvr_effect).clip(1e-4, 0.40)
    los = (base_los[:, None] * noise(0.05) * los_effect).clip(1.0, None)
    adr = base_adr[:, None] * adr_level[None, :] * noise(0.04) * adr_effect

    bookings = visits * cvr
    room_nights = bookings * los
    gbv = room_nights * adr

    # -- negative control outcome: the ineligible segment ------------------
    # Same properties, same market factor and seasonality, no campaign
    # exposure. Built from its own demand draw rather than by dividing the
    # effect back out of GBV: reusing GBV's noise would make the control a
    # near-perfect mirror of the treated series and flatter the design
    # enormously. A corporate-rate booker is not the same customer as a
    # deal-seeker, so the two streams share the market and little else.
    # `nonqualifying_correlation` sets how much demand noise they do share.
    nq_scale = rng.beta(4.0, 4.0, n_props) * 2 * spec.non_qualifying_share
    rho = float(np.clip(spec.nonqualifying_correlation, 0.0, 1.0))
    shared = np.log(np.clip(visits / (base_visits[:, None] * level), 1e-9, None))
    own = rng.normal(0.0, 1.0, (n_props, n_weeks))
    nq_noise = np.exp(0.16 * (rho * shared / 0.16 + np.sqrt(1 - rho**2) * own))
    substitution = 1.0 + t.nonqualifying_substitution * shape_spill[None, :]
    gbv_nonqualifying = (
        base_visits[:, None]
        * nq_scale[:, None]
        * traffic_level[None, :]
        * base_cvr[:, None]
        * base_los[:, None]
        * base_adr[:, None]
        * adr_level[None, :]
        * nq_noise
        * np.where(is_treated, substitution, 1.0)
    )

    panel = pd.DataFrame(
        {
            "property_id": np.repeat(prop_ids, n_weeks),
            "week": np.tile(weeks.to_numpy(), n_props),
            "cohort": np.repeat(cohort, n_weeks),
            "stack_eligible": np.repeat(stack_eligible, n_weeks),
            "repeat_participant": np.repeat(repeat, n_weeks),
            "deal_live": (
                is_treated & (weeks_since[None, :] >= 0)
                & (weeks_since[None, :] < spec.campaign_length)
            ).ravel().astype(int),
            "visits": visits.ravel(),
            "bookings": bookings.ravel(),
            "room_nights": room_nights.ravel(),
            "gbv": gbv.ravel(),
            "cvr": cvr.ravel(),
            "los": los.ravel(),
            "adr": adr.ravel(),
            "gbv_nonqualifying": gbv_nonqualifying.ravel(),
        }
    )

    # -- market demand covariates -----------------------------------------
    # Market-level, so they cannot be contaminated by the campaign itself.
    # They carry real signal about the shared market factor, but noisily --
    # which is why a contemporaneous control series still beats them.
    cov = pd.DataFrame({"week": weeks})
    # Idiosyncratic noise on these series is small because they are
    # marketplace-wide aggregates summed across every market and millions of
    # sessions. An earlier version used single-market noise levels (9-11%
    # week to week), which is realistic for one country's metasearch feed and
    # far too volatile for a global total -- it made the demand block nearly
    # uninformative and left the contamination test with no power.
    cov["meta_impressions"] = (
        42e6 * market_factor**0.85 * seasonal * np.exp(rng.normal(0, 0.035, n_weeks))
    )
    cov["destination_queries"] = (
        7.8e6 * market_factor**0.7 * seasonal * holiday**0.5
        * np.exp(rng.normal(0, 0.040, n_weeks))
    )
    cov["market_visitors"] = (
        19e6 * market_factor**0.9 * seasonal * holiday**0.8
        * np.exp(rng.normal(0, 0.025, n_weeks))
    )
    cov["market_purchases"] = 0.031 * cov["market_visitors"] * np.exp(
        rng.normal(0, 0.035, n_weeks)
    )
    # Paid-search clicks: a marketplace-level demand signal, contemporaneous
    # and outside the lodging promotion.
    cov["search_clicks"] = (
        11e6 * market_factor**0.8 * seasonal * holiday**0.6
        * np.exp(rng.normal(0, 0.030, n_weeks))
    )
    # Non-lodging product lines (air, car, activities). Part of marketplace
    # production, not part of a lodging merchandising deal, so they dilute the
    # campaign's footprint in any marketplace-wide aggregate.
    cov["other_products_volume"] = (
        62e6 * market_factor**0.75 * (1 + 0.14 * (seasonal - 1)) * trend
        * np.exp(rng.normal(0, 0.030, n_weeks))
        * (1.0 + spec.truth.cross_sell_halo * shape_spill)
    )
    panel = panel.merge(cov, on="week", how="left")

    # -- this property's own prior-year production ------------------------
    # Joined on (prior ISO year, same ISO week) rather than a fixed 52-week
    # offset. For Monday-keyed weeks the two usually agree, because 52 weeks
    # is exactly 364 days -- but they diverge around 53-week ISO years (2020,
    # 2026), where a fixed offset slides the comparison by a week. Harmless in
    # June, material in late December, which is when campaigns run.
    iso = panel["week"].dt.isocalendar()
    panel["iso_year"] = iso["year"].to_numpy()
    panel["iso_week"] = iso["week"].to_numpy()
    ly = panel[["property_id", "iso_year", "iso_week", "gbv"]].copy()
    ly["iso_year"] = ly["iso_year"] + 1
    ly = ly.rename(columns={"gbv": "outcome_ly"})
    panel = panel.merge(ly, on=["property_id", "iso_year", "iso_week"], how="left")

    # -- marketplace-wide production, computed before trimming -------------
    # Lodging across every cohort plus the non-lodging product lines. The
    # campaign is a small share of this, which is what makes its prior-year
    # value usable as a control. Computed here rather than after trimming so
    # the prior-year join has a full year of history behind it.
    lodging_total = panel.groupby("week", as_index=False)["gbv"].sum()
    lodging_total = lodging_total.rename(columns={"gbv": "_lodging_total"})
    totals = lodging_total.merge(
        cov[["week", "other_products_volume"]], on="week", how="left"
    )
    totals["marketplace_total"] = (
        totals["_lodging_total"] + totals["other_products_volume"]
    )
    # The same aggregate with participating properties removed.
    #
    # This matters because of repeat participation. The prior-year value of a
    # marketplace-wide total, read during the post window, refers to LAST
    # year's campaign weeks -- when most of the same properties were also
    # discounting. The control is therefore elevated exactly where it is used,
    # and the model charges part of this year's lift to it. Excluding
    # participants removes the channel; the aggregate is still marketplace
    # scale because participants are a minority of lodging and lodging is a
    # minority of the marketplace.
    ex_participants = (
        panel[panel["cohort"] != "treated"]
        .groupby("week", as_index=False)["gbv"]
        .sum()
        .rename(columns={"gbv": "_ex_participant_lodging"})
    )
    totals = totals.merge(ex_participants, on="week", how="left")
    totals["marketplace_ex_participants"] = (
        totals["_ex_participant_lodging"] + totals["other_products_volume"]
    )
    t_iso = totals["week"].dt.isocalendar()
    totals["iso_year"] = t_iso["year"].to_numpy()
    totals["iso_week"] = t_iso["week"].to_numpy()
    t_ly = totals[
        ["iso_year", "iso_week", "marketplace_total", "marketplace_ex_participants"]
    ].copy()
    t_ly["iso_year"] = t_ly["iso_year"] + 1
    t_ly = t_ly.rename(
        columns={
            "marketplace_total": "marketplace_total_ly",
            "marketplace_ex_participants": "marketplace_ex_participants_ly",
        }
    )
    totals = totals.merge(t_ly, on=["iso_year", "iso_week"], how="left")
    panel = panel.merge(
        totals[
            [
                "week",
                "marketplace_total",
                "marketplace_total_ly",
                "marketplace_ex_participants_ly",
            ]
        ],
        on="week",
        how="left",
    )

    # An extra year beyond the configured pre-window is retained. The window
    # used for the headline estimate is unchanged, but a placebo analysis has
    # to move the intervention date backwards and still keep a full training
    # history behind it. Trimming to exactly `pre_weeks` left no room, which
    # forced every placebo for a spring campaign into the winter holidays.
    panel = panel[panel["week"] >= activation - pd.Timedelta(weeks=spec.pre_weeks + 52)]
    return panel.sort_values(["property_id", "week"]).reset_index(drop=True)


def split_control_pool(
    panel: pd.DataFrame, *, covariate_share: float = 0.5, seed: int = 99
) -> pd.DataFrame:
    """Split the undisplaced pool into a covariate half and a placebo half.

    The untreated-and-undisplaced pool has two jobs: it supplies the control
    series that soaks up market-wide demand movement, and it supplies the
    pseudo-treated cohorts for the in-space placebo test. Using the same
    properties for both makes the placebo test circular -- the null
    distribution would be built from units already inside the model's
    counterfactual. So the pool is split once, by property, and the halves
    never mix.
    """
    out = panel.copy()
    rng = np.random.default_rng(seed)
    distant_ids = out.loc[out["cohort"] == "distant", "property_id"].unique()
    n_cov = int(round(len(distant_ids) * covariate_share))
    covariate_ids = set(rng.choice(distant_ids, size=n_cov, replace=False).tolist())
    out["control_role"] = np.select(
        [
            out["cohort"] == "treated",
            out["cohort"] == "compset",
            out["property_id"].isin(covariate_ids),
        ],
        ["treated", "compset", "covariate"],
        default="placebo",
    )
    return out


def aggregate_to_cohort(panel: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    """Collapse the property panel to one weekly row per cohort.

    Aggregating before modelling is deliberate: the decision is made on the
    portfolio total, and a single aggregate series with 78 weeks of history is
    far better identified than thousands of short, noisy property series.

    Ratio drivers are recomputed from the summed levels, never averaged.
    Averaging a conversion rate across properties of wildly different traffic
    answers a question nobody asked.
    """
    level_cols = [
        c
        for c in ("visits", "bookings", "room_nights", "gbv", "outcome_ly")
        if c in panel.columns
    ]
    market_cols = [
        c
        for c in (
            "meta_impressions",
            "destination_queries",
            "market_visitors",
            "market_purchases",
            "search_clicks",
            "other_products_volume",
            "marketplace_total",
            "marketplace_total_ly",
            "marketplace_ex_participants_ly",
        )
        if c in panel.columns
    ]

    agg = (
        panel.groupby(["cohort", "week"], as_index=False)[level_cols]
        .sum(min_count=1)
        .sort_values(["cohort", "week"])
    )
    counts = panel.groupby(["cohort", "week"])["property_id"].nunique().rename("n_props")
    agg = agg.merge(counts, on=["cohort", "week"])

    if market_cols:
        market = panel.groupby("week", as_index=False)[market_cols].mean()
        agg = agg.merge(market, on="week", how="left")

    for name, (num, den) in RATIO_METRICS.items():
        if num in agg.columns and den in agg.columns:
            agg[name] = agg[num] / agg[den]

    # -- control series ----------------------------------------------------
    # Named for what it is. Calling this "compset volume" -- as an earlier
    # version did -- invites exactly the confusion the design exists to
    # prevent: it is drawn from properties OUTSIDE participants' competitive
    # sets, which is why it is safe to use as a predictor.
    if "control_role" in panel.columns:
        clean = (
            panel[panel["control_role"] == "covariate"]
            .groupby("week", as_index=False)[outcomes[0]]
            .sum()
            .rename(columns={outcomes[0]: "clean_control_volume"})
        )
        if not clean.empty:
            agg = agg.merge(clean, on="week", how="left")

        # Contemporaneous comp-set volume: the Design A predictor. Retained so the
        # design comparison can quantify its contamination rather than assert
        # it.
        compset = (
            panel[panel["cohort"] == "compset"]
            .groupby("week", as_index=False)[outcomes[0]]
            .sum()
            .rename(columns={outcomes[0]: "compset_volume"})
        )
        if not compset.empty:
            agg = agg.merge(compset, on="week", how="left")

    # -- negative control outcome: ineligible revenue at treated properties
    # The only contemporaneous control available when a campaign runs in every
    # market at once. Drawn from the treated cohort itself, from the revenue
    # the promotion cannot reach.
    # Computed per cohort rather than only for the treated group. A placebo
    # cohort assembled from untreated properties needs its *own* ineligible
    # revenue to serve as a control, otherwise the in-space placebo cannot be
    # run against this design at all -- and a design whose placebo test cannot
    # be run is a design with one fewer check on it.
    if "gbv_nonqualifying" in panel.columns:
        nq = (
            panel.groupby(["cohort", "week"], as_index=False)["gbv_nonqualifying"]
            .sum()
            .rename(columns={"gbv_nonqualifying": "nonqualifying_volume"})
        )
        if not nq.empty:
            agg = agg.merge(nq, on=["cohort", "week"], how="left")

    # -- lodging-only prior-year production -------------------------------
    # Whole-marketplace volume one ISO year back. Participants are only a
    # minority of it and last year's lift is diluted accordingly, so it is far
    # less contaminated than a participant's own prior-year series -- at the
    # cost of carrying no information about *this* year's market shocks.
    if "outcome_ly" in panel.columns:
        eg_ly = (
            panel.groupby("week", as_index=False)["outcome_ly"]
            .sum()
            .rename(columns={"outcome_ly": "market_total_ly"})
        )
        agg = agg.merge(eg_ly, on="week", how="left")

    return agg.reset_index(drop=True)


def true_average_lift(
    panel: pd.DataFrame,
    post_start: pd.Timestamp,
    post_end: pd.Timestamp,
    outcome: str = "gbv",
    *,
    truth: GroundTruth | None = None,
    campaign_length: int = 10,
) -> float:
    """The volume-weighted lift the estimator should recover.

    Recovered from the generated panel rather than from the parameters alone,
    so it includes the actual stacking mix and the actual seasonal weighting
    of the post weeks -- the same quantity the estimator targets.
    """
    truth = truth or GroundTruth()
    treated = panel[panel["cohort"] == "treated"].copy()
    treated["week"] = pd.to_datetime(treated["week"])
    window = treated[(treated["week"] >= post_start) & (treated["week"] <= post_end)]
    if window.empty:
        raise ValueError("post window selected no treated rows")

    activation = pd.Timestamp(treated.loc[treated["deal_live"] == 1, "week"].min())
    weeks = pd.DatetimeIndex(sorted(window["week"].unique()))
    shape = _shape(((weeks - activation).days // 7).to_numpy(), truth, campaign_length)

    stack_share = float(
        panel.loc[panel["cohort"] == "treated"]
        .groupby("property_id")["stack_eligible"]
        .first()
        .mean()
    )
    cvr_lift = truth.cvr_lift + truth.stacking_bonus * stack_share

    factors = {
        "visits": 1 + truth.visits_lift * shape,
        "cvr": 1 + cvr_lift * shape,
        "los": 1 + truth.los_lift * shape,
        "adr": 1 + truth.adr_lift * shape,
    }
    composition = {
        "gbv": ("visits", "cvr", "los", "adr"),
        "room_nights": ("visits", "cvr", "los"),
        "bookings": ("visits", "cvr"),
        "visits": ("visits",),
        "cvr": ("cvr",),
        "los": ("los",),
        "adr": ("adr",),
    }[outcome]

    weekly_lift = pd.Series(
        np.prod([factors[name] for name in composition], axis=0) - 1.0, index=weeks
    )
    observed = window.groupby("week")[outcome].sum()
    if outcome in RATIO_METRICS:
        num, den = RATIO_METRICS[outcome]
        observed = window.groupby("week")[num].sum() / window.groupby("week")[den].sum()
    counterfactual = observed / (1.0 + weekly_lift.reindex(observed.index))
    if outcome in RATIO_METRICS:
        return float(observed.mean() / counterfactual.mean() - 1.0)
    return float(observed.sum() / counterfactual.sum() - 1.0)
