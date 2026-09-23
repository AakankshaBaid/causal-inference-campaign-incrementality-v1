"""Orchestration: config in, decision memo out.

One entry point, `run_cohort`, because the discipline that makes the numbers
trustworthy is that nobody can run the estimate without also running the
tests. Estimation and falsification are the same function call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import assumptions as assumptions_mod
from . import drivers as drivers_mod
from . import economics, reporting
from . import stacking as stacking_mod
from .config import CohortConfig
from .counterfactual import get_backend
from .features import build_design
from .inference import UpliftResult, estimate_uplift
from .simulate import (
    SimulationSpec,
    aggregate_to_cohort,
    simulate_panel,
    split_control_pool,
)
from .validation import ValidationReport, run_battery

logger = logging.getLogger(__name__)


@dataclass
class CohortResult:
    config: CohortConfig
    uplift: dict[str, UpliftResult]
    validation: ValidationReport
    assumptions: assumptions_mod.AssumptionReport
    econ: economics.EconomicsResult
    gross_econ: economics.EconomicsResult
    incrementality: IncrementalityResult
    drivers: drivers_mod.DriverDecomposition | None
    stacking: stacking_mod.StackingResult | None
    net_incremental_bookings: float
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def primary(self) -> UpliftResult:
        return self.uplift[self.config.primary_outcome]

    def headline(self) -> str:
        return (
            f"{self.config.name}: {self.incrementality.net_lift:+.1%} net "
            f"incrementality on "
            f"{self.config.primary_outcome.upper()}, "
            f"${self.net_incremental_bookings / 1e6:,.1f}M net incremental, "
            f"{self.econ.return_on_spend:.2f}x return on spend, "
            f"{'all tests pass' if self.validation.all_passed else 'TESTS FAILED'}"
            f"{'' if self.assumptions.all_met else ', ASSUMPTIONS NOT MET'}"
        )


@dataclass
class IncrementalityResult:
    """Total incrementality: the participant effect plus the compset effect.

    Two causal models, not one. The first treats participating properties as
    the intervention and measures partner incrementality. The second treats
    their competitive set as the intervention and measures what happened to
    the properties next door.

    The second result is *added*, with its own sign, and that sign is the
    finding:

        compset significantly positive  -> halo; the campaign lifted
                                           non-participants too, and that
                                           volume is incremental
        compset not distinguishable     -> clean; total equals the
                                           participant effect
        compset significantly negative  -> cannibalisation; volume moved
                                           rather than appeared, and comes off

    Treating the compset effect as a deduction by default -- as an earlier
    version of this code did -- is wrong. It assumes the answer before
    measuring it, and it silently discards halo.
    """

    gross_lift: float
    gross_ci: tuple[float, float]
    gross_value: float
    displacement_rate: float
    displacement_ci: tuple[float, float]
    displacement_value: float
    displacement_significant: bool
    net_lift: float
    net_ci: tuple[float, float]
    net_value: float

    @property
    def regime(self) -> str:
        if not self.displacement_significant:
            return "no measurable effect on the competitive set"
        return "halo" if self.displacement_value > 0 else "cannibalisation"

    def build_rows(self) -> pd.DataFrame:
        label = {
            "halo": "Plus halo on competitive set",
            "cannibalisation": "Less cannibalisation of competitive set",
            "no measurable effect on the competitive set": (
                "Competitive set: no measurable effect"
            ),
        }[self.regime]
        contribution = (
            self.displacement_value if self.displacement_significant else 0.0
        )
        rate = self.displacement_rate if self.displacement_significant else 0.0
        return pd.DataFrame(
            [
                ("Participating properties", self.gross_lift, self.gross_value),
                (label, rate, contribution),
                ("Total incrementality", self.net_lift, self.net_value),
            ],
            columns=["line_item", "rate", "value"],
        )

    def summary_row(self) -> dict[str, object]:
        return {
            "gross_lift": self.gross_lift,
            "gross_ci_low": self.gross_ci[0],
            "gross_ci_high": self.gross_ci[1],
            "displacement_rate": self.displacement_rate,
            "displacement_value": self.displacement_value,
            "displacement_significant": self.displacement_significant,
            "regime": self.regime,
            "net_lift": self.net_lift,
            "net_ci_low": self.net_ci[0],
            "net_ci_high": self.net_ci[1],
            "net_value": self.net_value,
        }


def _prior_campaign_window(cfg: CohortConfig):
    if not cfg.model.control_prior_campaign:
        return None
    start = cfg.activation_week - pd.Timedelta(weeks=52)
    return (start, start + pd.Timedelta(weeks=cfg.model.prior_campaign_weeks - 1))


def _build_design_for(series: pd.DataFrame, outcome: str, cfg: CohortConfig):
    return build_design(
        series,
        outcome=outcome,
        regressors=cfg.model.predictors,
        pre_window=(cfg.pre_start, cfg.pre_end),
        post_window=(cfg.post_start, cfg.post_end),
        log_transform=cfg.model.log_transform,
        annual_fourier_terms=cfg.model.annual_fourier_terms,
        holiday_iso_weeks=cfg.model.holiday_iso_weeks,
        prior_campaign_window=_prior_campaign_window(cfg),
        prior_campaign_shape=(
            cfg.model.prior_campaign_ramp_weeks,
            cfg.model.prior_campaign_decay,
        ),
    )


def run_cohort(
    cfg: CohortConfig,
    panel: pd.DataFrame,
    *,
    output_dir: str | Path = "outputs",
    n_placebo: int = 40,
    make_figures: bool = True,
) -> CohortResult:
    """Estimate, falsify, monetise and write exhibits for one campaign."""
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    cohort_series = aggregate_to_cohort(panel, cfg.outcomes)
    treated = cohort_series[cohort_series["cohort"] == "treated"].reset_index(drop=True)

    # -- 1. estimate every outcome ----------------------------------------
    uplift: dict[str, UpliftResult] = {}
    for outcome in cfg.outcomes:
        design = _build_design_for(treated, outcome, cfg)
        backend_kwargs = {}
        if cfg.model.backend == "prophet":
            backend_kwargs = {"weeks_pre": design.weeks_pre, "weeks_post": design.weeks_post}
        model = get_backend(cfg.model.backend, **backend_kwargs)
        result = estimate_uplift(
            model,
            design,
            alpha=cfg.model.alpha,
            n_bootstrap=cfg.model.n_bootstrap,
            block_length=cfg.model.block_length,
            backend_name=cfg.model.backend,
            materiality_threshold=cfg.materiality_threshold,
        )
        result.outcome = outcome
        uplift[outcome] = result
        logger.info(
            "%s | %s: %+.2f%% (CI %+.2f%% to %+.2f%%)",
            cfg.name,
            outcome,
            result.relative_uplift * 100,
            result.ci_relative[0] * 100,
            result.ci_relative[1] * 100,
        )

    primary = uplift[cfg.primary_outcome]

    # -- 1b. assumption gate -----------------------------------------------
    # Checked before anything is reported. An estimate produced on top of a
    # violated assumption is not a weaker estimate, it is a different number.
    assumption_report = assumptions_mod.check_assumptions(panel, treated, cfg)
    if not assumption_report.all_met:
        failed = [c.name for c in assumption_report.checks if not c.passed]
        logger.warning("%s | ASSUMPTIONS NOT MET: %s", cfg.name, "; ".join(failed))

    # -- 2. try to break it ------------------------------------------------
    report = run_battery(panel, treated, cfg, primary.relative_uplift, n_placebo=n_placebo)

    # -- 3. net off displacement before monetising -------------------------
    # A negative comp-set effect is demand moved, not demand created. Scale it
    # by the comp set's own counterfactual volume to get the dollar offset.
    compset_label = "compset" if (panel["cohort"] == "compset").any() else "distant"
    control_series = cohort_series[cohort_series["cohort"] == compset_label]
    control_design = _build_design_for(
        control_series.reset_index(drop=True), cfg.primary_outcome, cfg
    )
    # Full inference on the comp set, not just a point estimate: the marketplace view
    # subtracts this number, so its uncertainty belongs in the marketplace interval.
    # Treating displacement as known would understate the uncertainty on the
    # figure that actually drives the P&L.
    displacement_result = estimate_uplift(
        get_backend(cfg.model.backend),
        control_design,
        alpha=cfg.model.alpha,
        n_bootstrap=cfg.model.n_bootstrap,
        block_length=cfg.model.block_length,
        backend_name=cfg.model.backend,
        materiality_threshold=cfg.materiality_threshold,
    )
    # The compset model's effect is ADDED with its own sign. Positive is halo
    # and counts as incremental; negative is cannibalisation and comes off.
    # An effect that is not statistically distinguishable contributes nothing.
    compset_contribution = (
        float(displacement_result.absolute_uplift)
        if displacement_result.is_significant
        else 0.0
    )
    displacement_dollars = -compset_contribution
    net_incremental = primary.absolute_uplift + compset_contribution

    # -- 4. money ----------------------------------------------------------
    econ_result = economics.evaluate(
        incremental_bookings=net_incremental,
        counterfactual_bookings=primary.counterfactual_cumulative,
        actual_bookings=primary.actual_cumulative,
        econ=cfg.economics,
    )
    # The same P&L on the *gross* uplift, i.e. what the readout would say if
    # the displacement assumption were accepted untested. Reported alongside
    # so the reader can see exactly how much the test is worth: when these two
    # land on opposite sides of break-even, the displacement test is the
    # decision rather than a footnote.
    gross_econ = economics.evaluate(
        incremental_bookings=primary.absolute_uplift,
        counterfactual_bookings=primary.counterfactual_cumulative,
        actual_bookings=primary.actual_cumulative,
        econ=cfg.economics,
    )
    grid = economics.sensitivity_grid(
        counterfactual_bookings=primary.counterfactual_cumulative, econ=cfg.economics
    )

    # -- 4b. the incrementality build --------------------------------------
    net_lift = (
        net_incremental / primary.counterfactual_cumulative
        if primary.counterfactual_cumulative
        else float("nan")
    )
    # Half-widths added in quadrature. The participant lift and the
    # displacement rate are estimated on disjoint property sets, so treating
    # their errors as independent is reasonable, and the combined interval is
    # wider than either -- which it must be, since the net figure depends on
    # both.
    half_participant = (primary.ci_absolute[1] - primary.ci_absolute[0]) / 2.0
    half_displacement = (
        (displacement_result.ci_absolute[1] - displacement_result.ci_absolute[0]) / 2.0
        if displacement_result.is_significant
        else 0.0
    )
    net_half = float(np.hypot(half_participant, half_displacement))
    incrementality = IncrementalityResult(
        gross_lift=primary.relative_uplift,
        gross_ci=primary.ci_relative,
        gross_value=primary.absolute_uplift,
        displacement_rate=displacement_result.relative_uplift,
        displacement_ci=displacement_result.ci_relative,
        displacement_value=compset_contribution,
        displacement_significant=bool(displacement_result.is_significant),
        net_lift=net_lift,
        net_ci=(
            (net_incremental - net_half) / primary.counterfactual_cumulative,
            (net_incremental + net_half) / primary.counterfactual_cumulative,
        ),
        net_value=net_incremental,
    )

    # -- 4c. drivers of growth ---------------------------------------------
    driver_result = None
    try:
        driver_result = drivers_mod.decompose(treated, cfg, primary)
    except (KeyError, ValueError) as exc:
        logger.warning("driver decomposition skipped: %s", exc)

    # -- 5. stacking (skipped when the panel has no contract split) --------
    stack_result = None
    if "stack_eligible" in panel.columns and panel["stack_eligible"].nunique() > 1:
        try:
            stack_result = stacking_mod.estimate_stacking_value(panel, cfg)
        except ValueError as exc:
            logger.warning("stacking analysis skipped: %s", exc)

    # -- 6. artifacts ------------------------------------------------------
    artifacts: dict[str, str] = {}
    tables = output_dir / "tables"

    pd.DataFrame([r.summary_row() for r in uplift.values()]).to_csv(
        tables / f"{_slug(cfg.name)}_uplift_summary.csv", index=False
    )
    artifacts["uplift_summary"] = str(tables / f"{_slug(cfg.name)}_uplift_summary.csv")

    primary.weekly.to_csv(tables / f"{_slug(cfg.name)}_weekly_detail.csv", index=False)
    artifacts["weekly_detail"] = str(tables / f"{_slug(cfg.name)}_weekly_detail.csv")

    report.to_frame().to_csv(tables / f"{_slug(cfg.name)}_validation.csv", index=False)
    assumption_report.to_frame().to_csv(
        tables / f"{_slug(cfg.name)}_assumptions.csv", index=False
    )
    if assumption_report.predictors is not None:
        assumption_report.predictors.to_csv(
            tables / f"{_slug(cfg.name)}_predictor_integrity.csv", index=False
        )
    artifacts["validation"] = str(tables / f"{_slug(cfg.name)}_validation.csv")

    report.sensitivity.to_csv(tables / f"{_slug(cfg.name)}_sensitivity.csv", index=False)
    if driver_result is not None:
        driver_result.to_frame().to_csv(
            tables / f"{_slug(cfg.name)}_drivers.csv", index=False
        )
        artifacts["drivers"] = str(tables / f"{_slug(cfg.name)}_drivers.csv")
    econ_result.summary_rows().to_csv(
        tables / f"{_slug(cfg.name)}_economics.csv", index=False
    )
    artifacts["economics"] = str(tables / f"{_slug(cfg.name)}_economics.csv")

    run_record = {
        "config": cfg.to_dict(),
        "headline": {
            outcome: r.summary_row() for outcome, r in uplift.items()
        },
        "displacement_relative": displacement_result.relative_uplift,
        "displacement_ci": list(displacement_result.ci_relative),
        "displacement_dollars": displacement_dollars,
        "net_incremental_bookings": net_incremental,
        "incrementality": incrementality.summary_row(),
        "drivers": (
            driver_result.to_frame().to_dict("records") if driver_result else None
        ),
        "driver_residual": driver_result.residual if driver_result else None,
        "net_relative_uplift": (
            net_incremental / primary.counterfactual_cumulative
            if primary.counterfactual_cumulative
            else None
        ),
        "economics": econ_result.to_dict(),
        "economics_gross_of_displacement": gross_econ.to_dict(),
        "validation_passed": report.all_passed,
        "assumptions_met": assumption_report.all_met,
        "assumption_checks": assumption_report.to_frame().to_dict("records"),
        "stacking": stack_result.summary_row() if stack_result else None,
    }
    run_path = tables / f"{_slug(cfg.name)}_run_record.json"
    run_path.write_text(json.dumps(run_record, indent=2, default=str))
    artifacts["run_record"] = str(run_path)

    if make_figures:
        figures = output_dir / "figures"
        slug = _slug(cfg.name)
        # Exhibits are lettered so the README can reference them inline and
        # catalogue them in one table. A and B are presentation assets built
        # separately by `incrementality assets`.
        artifacts["exhibit_c"] = str(
            reporting.plot_counterfactual(
                primary,
                cfg.activation_week,
                f"Weekly {cfg.primary_outcome.upper()}",
                figures / f"{slug}_exhibit_c_baseline.png",
                series=treated,
            )
        )
        artifacts["exhibit_d"] = str(
            reporting.plot_uplift(primary, figures / f"{slug}_exhibit_d_lift_build.png")
        )
        if driver_result is not None:
            try:
                artifacts["exhibit_e"] = str(
                    reporting.plot_drivers(
                        driver_result, figures / f"{slug}_exhibit_e_drivers.png"
                    )
                )
            except ValueError as exc:
                logger.warning("driver exhibit skipped: %s", exc)
        try:
            artifacts["exhibit_h"] = str(
                reporting.plot_assumptions(
                    assumption_report, treated, cfg,
                    figures / f"{slug}_exhibit_h_assumptions.png",
                )
            )
        except Exception as exc:
            logger.warning("assumption exhibit skipped: %s", exc)
        artifacts["exhibit_f"] = str(
            reporting.plot_validation(
                report,
                primary.relative_uplift,
                figures / f"{slug}_exhibit_f_validation.png",
            )
        )
        artifacts["exhibit_g"] = str(
            reporting.plot_economics(
                econ_result, grid, figures / f"{slug}_exhibit_g_economics.png"
            )
        )

    artifacts["memo"] = str(
        reporting.write_decision_memo(
            cfg=cfg,
            uplift_results=uplift,
            econ_result=econ_result,
            gross_econ=gross_econ,
            incrementality=incrementality,
            drivers=driver_result,
            report=report,
            stacking=stack_result,
            path=output_dir / f"{_slug(cfg.name)}_decision_memo.md",
        )
    )

    return CohortResult(
        config=cfg,
        uplift=uplift,
        validation=report,
        assumptions=assumption_report,
        econ=econ_result,
        gross_econ=gross_econ,
        incrementality=incrementality,
        drivers=driver_result,
        stacking=stack_result,
        net_incremental_bookings=net_incremental,
        artifacts=artifacts,
    )


def load_or_simulate_panel(
    cfg: CohortConfig, data_path: str | Path | None = None, seed: int = 20240101
) -> pd.DataFrame:
    """Read a real panel if one is supplied, otherwise simulate.

    In production this is the only function that changes: point it at the
    warehouse query in ``sql/`` and everything downstream is untouched.
    """
    if data_path is not None:
        path = Path(data_path)
        panel = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        panel["week"] = pd.to_datetime(panel["week"])
        required = {"property_id", "week", "cohort", *cfg.outcomes}
        missing = required - set(panel.columns)
        if missing:
            raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
        return panel

    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=seed,
    )
    return split_control_pool(simulate_panel(spec), seed=seed + 1)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
