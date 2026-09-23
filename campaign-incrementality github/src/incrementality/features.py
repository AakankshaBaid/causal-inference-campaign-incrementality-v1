"""Turn a weekly series plus covariates into a model-ready design matrix.

Two decisions live here and both matter for the estimate:

1. Seasonality is deterministic Fourier terms in the exogenous block rather
   than a stochastic seasonal state. With 78 weeks of history a stochastic
   annual seasonal is barely identified and tends to absorb part of the
   treatment effect; two Fourier harmonics cost four parameters and stay put.
2. Regressors are z-scored on *pre-period statistics only*. Scaling on the
   full window leaks post-period information into the fit -- a subtle
   version of training on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DesignMatrix:
    """Aligned pre/post arrays ready for a counterfactual model."""

    y_pre: np.ndarray
    X_pre: np.ndarray
    y_post: np.ndarray
    X_post: np.ndarray
    weeks_pre: pd.DatetimeIndex
    weeks_post: pd.DatetimeIndex
    feature_names: list[str]
    log_transform: bool

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Map model-space values back to the outcome's natural units."""
        return np.exp(values) if self.log_transform else values


def fourier_terms(weeks: pd.DatetimeIndex, n_terms: int, period: float = 365.25) -> pd.DataFrame:
    """Deterministic annual seasonality as sin/cos pairs."""
    if n_terms <= 0:
        return pd.DataFrame(index=range(len(weeks)))
    doy = weeks.dayofyear.to_numpy(dtype=float)
    out = {}
    for k in range(1, n_terms + 1):
        phase = 2 * np.pi * k * doy / period
        out[f"sin_{k}"] = np.sin(phase)
        out[f"cos_{k}"] = np.cos(phase)
    return pd.DataFrame(out, index=range(len(weeks)))


def holiday_indicators(
    weeks: pd.DatetimeIndex, holiday_iso_weeks: dict[str, list[int]] | None
) -> pd.DataFrame:
    """Block dummies for named promotional/seasonal weeks.

    Two Fourier harmonics describe a smooth annual cycle; they cannot hold a
    34% spike in the Black Friday shopping week followed by a 18% dip over
    Christmas. Leaving that unmodelled pushes the miss into the treatment
    effect, because the campaign window sits precisely on top of it.

    These are block dummies (a named group of ISO weeks) rather than one
    dummy per week: with 78 weeks of history a single ISO week appears once
    or twice, which is not enough to estimate its own coefficient without
    overfitting.
    """
    if not holiday_iso_weeks:
        return pd.DataFrame(index=range(len(weeks)))
    iso_week = weeks.isocalendar().week.to_numpy()
    out = {
        f"hol_{name}": np.isin(iso_week, iso_list).astype(float)
        for name, iso_list in holiday_iso_weeks.items()
    }
    return pd.DataFrame(out, index=range(len(weeks)))


def build_design(
    series: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    pre_window: tuple[pd.Timestamp, pd.Timestamp],
    post_window: tuple[pd.Timestamp, pd.Timestamp],
    *,
    log_transform: bool = True,
    annual_fourier_terms: int = 2,
    holiday_iso_weeks: dict[str, list[int]] | None = None,
    prior_campaign_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    prior_campaign_shape: tuple[int, float] | None = None,
    week_col: str = "week",
) -> DesignMatrix:
    """Assemble the design matrix for one outcome and one cohort.

    Parameters
    ----------
    series
        Weekly rows for a single cohort, containing ``week``, ``outcome`` and
        every name in ``regressors``.
    """
    missing = [c for c in [outcome, *regressors] if c not in series.columns]
    if missing:
        raise KeyError(f"series is missing required columns: {missing}")

    frame = series.sort_values(week_col).reset_index(drop=True).copy()
    frame[week_col] = pd.to_datetime(frame[week_col])

    usable = [outcome, *regressors]
    if frame[usable].isna().any().any():
        n_before = len(frame)
        frame = frame.dropna(subset=usable).reset_index(drop=True)
        if len(frame) < n_before * 0.9:
            raise ValueError(
                f"dropped {n_before - len(frame)} of {n_before} weeks to nulls; "
                "check the upstream panel before trusting this estimate"
            )

    weeks = pd.DatetimeIndex(frame[week_col])
    pre_mask = (weeks >= pre_window[0]) & (weeks <= pre_window[1])
    post_mask = (weeks >= post_window[0]) & (weeks <= post_window[1])
    if pre_mask.sum() < 26:
        raise ValueError(f"only {pre_mask.sum()} pre-period weeks available; need >= 26")
    if post_mask.sum() == 0:
        raise ValueError("post window selected zero weeks; check activation_week")

    y = frame[outcome].to_numpy(dtype=float)
    if log_transform:
        if np.any(y <= 0):
            raise ValueError(
                f"{outcome} has non-positive values; cannot log-transform. "
                "Either fix the panel or set model.log_transform: false"
            )
        y = np.log(y)

    design = frame[regressors].astype(float).copy()
    if log_transform:
        # Covariates enter in logs too, so coefficients read as elasticities.
        for col in design.columns:
            if (design[col] > 0).all():
                design[col] = np.log(design[col])

    # Scale on pre-period moments only -- no post-period leakage.
    mu = design.loc[pre_mask].mean()
    sigma = design.loc[pre_mask].std(ddof=0).replace(0.0, 1.0)
    design = (design - mu) / sigma

    seasonal = fourier_terms(weeks, annual_fourier_terms)
    holidays = holiday_indicators(weeks, holiday_iso_weeks)
    extras = [design.reset_index(drop=True), seasonal, holidays]

    if prior_campaign_window is not None:
        # Last year's campaign is a treatment episode sitting inside the
        # pre-period. An indicator lets the fit attribute that lift to the
        # campaign instead of to seasonality, which is what otherwise
        # inflates the counterfactual and eats this year's effect.
        in_prior = (
            (weeks >= prior_campaign_window[0]) & (weeks <= prior_campaign_window[1])
        )
        profile = np.zeros(len(weeks), dtype=float)
        if in_prior.any():
            ramp_weeks, decay = prior_campaign_shape or (2, 0.06)
            offsets = np.arange(int(in_prior.sum()))
            ramp = np.clip((offsets + 1) / max(ramp_weeks, 1), 0.0, 1.0)
            fade = (1.0 - decay) ** np.clip(offsets - ramp_weeks + 1, 0, None)
            profile[in_prior] = ramp * fade
        extras.append(
            pd.DataFrame({"prior_campaign": profile}, index=range(len(weeks)))
        )

    design = pd.concat(extras, axis=1)

    # A dummy that never fires in the pre-period has no identified coefficient
    # and would be silently extrapolated from nothing. Drop it and say so.
    indicator_cols = [
        c for c in (*holidays.columns, "prior_campaign") if c in design.columns
    ]
    constant_in_pre = [
        col for col in indicator_cols if design.loc[pre_mask, col].nunique() < 2
    ]
    if constant_in_pre:
        design = design.drop(columns=constant_in_pre)

    return DesignMatrix(
        y_pre=y[pre_mask],
        X_pre=design.loc[pre_mask].to_numpy(dtype=float),
        y_post=y[post_mask],
        X_post=design.loc[post_mask].to_numpy(dtype=float),
        weeks_pre=weeks[pre_mask],
        weeks_post=weeks[post_mask],
        feature_names=list(design.columns),
        log_transform=log_transform,
    )


def build_donor_design(
    panel: pd.DataFrame,
    outcome: str,
    pre_window: tuple[pd.Timestamp, pd.Timestamp],
    post_window: tuple[pd.Timestamp, pd.Timestamp],
    *,
    n_donors: int = 40,
    donor_role: str = "placebo",
    treated_cohort: str = "treated",
    log_transform: bool = True,
    seed: int = 13,
) -> DesignMatrix:
    """Donor-unit design matrix for synthetic control.

    Synthetic control needs a different input from the regression backends:
    its columns must be *outcome series for untreated units*, because the
    estimator forms a convex combination of them. Handing it the regression
    design (standardised covariates plus Fourier terms) is meaningless -- a
    weighted average of sine waves is not a counterfactual booking path, and
    an earlier version of this code did exactly that.

    Individual properties are too noisy to serve as donors at weekly grain,
    so untreated properties are pooled into ``n_donors`` buckets, each summed
    to a stable series. Bucketing is deterministic given ``seed`` so a rerun
    reproduces the same weights.

    Donors are drawn from the placebo-role pool, which is untreated *and*
    undisplaced. Using competitive-set properties as donors would import the
    campaign's own spillover into the counterfactual.
    """
    work = panel.copy()
    work["week"] = pd.to_datetime(work["week"])
    role = work["control_role"] if "control_role" in work.columns else work["cohort"]

    treated = (
        work[work["cohort"] == treated_cohort].groupby("week")[outcome].sum().sort_index()
    )
    donors_raw = work[role == donor_role]
    if donors_raw.empty:
        raise ValueError(f"no properties with control_role/cohort == {donor_role!r}")

    ids = np.sort(donors_raw["property_id"].unique())
    rng = np.random.default_rng(seed)
    bucket = pd.Series(
        rng.integers(0, min(n_donors, max(len(ids) // 5, 2)), size=len(ids)), index=ids
    )
    donors_raw = donors_raw.assign(_bucket=donors_raw["property_id"].map(bucket))
    donor_frame = (
        donors_raw.pivot_table(
            index="week", columns="_bucket", values=outcome, aggfunc="sum"
        )
        .sort_index()
        .reindex(treated.index)
    )
    donor_frame.columns = [f"donor_{c}" for c in donor_frame.columns]

    frame = pd.concat([treated.rename("y"), donor_frame], axis=1).dropna()
    weeks = pd.DatetimeIndex(frame.index)
    pre_mask = (weeks >= pre_window[0]) & (weeks <= pre_window[1])
    post_mask = (weeks >= post_window[0]) & (weeks <= post_window[1])
    if pre_mask.sum() < 26:
        raise ValueError(f"only {pre_mask.sum()} pre-period weeks available; need >= 26")

    y = frame["y"].to_numpy(dtype=float)
    donors = frame.drop(columns="y").to_numpy(dtype=float)
    if log_transform:
        if np.any(y <= 0) or np.any(donors <= 0):
            raise ValueError("donor design requires strictly positive series to log")
        y, donors = np.log(y), np.log(donors)

    return DesignMatrix(
        y_pre=y[pre_mask],
        X_pre=donors[pre_mask],
        y_post=y[post_mask],
        X_post=donors[post_mask],
        weeks_pre=weeks[pre_mask],
        weeks_post=weeks[post_mask],
        feature_names=[c for c in frame.columns if c != "y"],
        log_transform=log_transform,
    )
