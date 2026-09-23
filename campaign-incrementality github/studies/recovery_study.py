"""Monte Carlo recovery study: does the estimator find the right answer?

Runs the full estimation path over many simulated campaigns where the true
lift is known, and scores four things that matter more than any single point
estimate:

* **bias** -- is the estimator systematically flattering the campaign?
* **RMSE** -- how far off is a typical single-campaign read?
* **interval coverage** -- does the nominal 90% interval actually contain the
  truth 90% of the time, or is it advertising precision it does not have?
* **false positive rate** -- on campaigns with a true lift of exactly zero,
  how often does it declare a win?

It is also how the model specification was chosen. Comparing candidate
specifications on simulated data is legitimate; comparing them on the real
campaign and keeping the one with the best-looking answer is not.

Design notes
------------
*Paired*: one simulated campaign per seed, scored under every specification,
so specification differences are not confounded with draw-to-draw variation.

*Checkpointed*: every draw is appended to the draws CSV as it completes, so
the study can be run in chunks and resumed. Re-running a seed overwrites its
rows rather than duplicating them.

*Reduced scale by default*: calibration is a property of the estimator and
holds at any cohort size. Precision is not -- the headline run uses the full
cohort and reports a tighter interval than the mean width reported here.

Usage
-----
    python studies/recovery_study.py --arm effect --seeds 0-7
    python studies/recovery_study.py --arm effect --seeds 8-15
    python studies/recovery_study.py --arm null   --seeds 0-15
    python studies/recovery_study.py --score-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incrementality.config import PREDICTOR_SETS, load_cohort
from incrementality.counterfactual import get_backend
from incrementality.features import build_design
from incrementality.inference import estimate_uplift
from incrementality.simulate import (
    GroundTruth,
    SimulationSpec,
    aggregate_to_cohort,
    simulate_panel,
    split_control_pool,
    true_average_lift,
)

# The measurement design went through four iterations. Each fixed a named
# assumption violation in the previous one and, in two cases, introduced a new
# one. This study puts numbers on trade-offs that were previously argued
# qualitatively: "can over- or under-estimate incrementality" becomes a
# measured bias in percentage points.
SPECS: dict[str, dict] = {
    "A: competitor-set control": {"design": "competitor_control", "prior_control": True},
    "B: own prior-year": {"design": "own_prior_year", "prior_control": True},
    "C: market-wide prior-year": {"design": "market_prior_year", "prior_control": True},
    "D: clean-pool control": {"design": "clean_pool_control", "prior_control": True},
    "D: no prior-campaign control": {"design": "clean_pool_control", "prior_control": False},
}

# The zero-effect arm measures the false-positive rate of the design actually
# shipped, so it is named explicitly rather than taken from dict ordering.
HEADLINE_SPEC = "D: clean-pool control"

SCALES = {
    "small": {"n_treated": 1_200, "n_compset": 1_200, "n_distant": 1_500},
    "full": {"n_treated": 2_500, "n_compset": 3_000, "n_distant": 4_000},
}

# Arm labels avoid the literal string "null" on purpose. pandas treats "null"
# as a missing-value sentinel by default, so writing it to CSV and reading it
# back silently turns the whole arm into NaN and the rows vanish from every
# groupby. Cost me one confused debugging session; the label is now inert and
# the reader below is explicit about NA handling.
ARMS = {
    "effect": (GroundTruth(), 5_000),
    "zero_effect": (
        GroundTruth(
            visits_lift=0.0,
            cvr_lift=0.0,
            los_lift=0.0,
            adr_lift=0.0,
            spillover=0.0,
            stacking_bonus=0.0,
        ),
        9_000,
    ),
}


def _simulate(cfg, seed: int, truth: GroundTruth, scale: str):
    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=seed,
        truth=truth,
        **SCALES[scale],
    )
    panel = split_control_pool(simulate_panel(spec), seed=seed + 1)
    truth_lift = (
        true_average_lift(panel, cfg.post_start, cfg.post_end, truth=truth)
        if truth.composite_nbv_lift != 0.0
        else 0.0
    )
    series = aggregate_to_cohort(panel, cfg.outcomes)
    return series[series["cohort"] == "treated"].reset_index(drop=True), truth_lift


def _estimate(cfg, seed, treated, truth_lift, spec_kwargs, n_bootstrap):
    prior_window = None
    if spec_kwargs["prior_control"]:
        prior_start = cfg.activation_week - pd.Timedelta(weeks=52)
        prior_window = (
            prior_start,
            prior_start + pd.Timedelta(weeks=cfg.model.prior_campaign_weeks - 1),
        )
    design = build_design(
        treated,
        outcome=cfg.primary_outcome,
        regressors=PREDICTOR_SETS[spec_kwargs["design"]],
        pre_window=(cfg.pre_start, cfg.pre_end),
        post_window=(cfg.post_start, cfg.post_end),
        log_transform=cfg.model.log_transform,
        annual_fourier_terms=cfg.model.annual_fourier_terms,
        holiday_iso_weeks=cfg.model.holiday_iso_weeks,
        prior_campaign_window=prior_window,
    )
    result = estimate_uplift(
        get_backend(cfg.model.backend),
        design,
        alpha=cfg.model.alpha,
        n_bootstrap=n_bootstrap,
        block_length=cfg.model.block_length,
        seed=seed,
    )
    return {
        "seed": seed,
        "truth": truth_lift,
        "estimate": result.relative_uplift,
        "error": result.relative_uplift - truth_lift,
        "ci_low": result.ci_relative[0],
        "ci_high": result.ci_relative[1],
        "covered": bool(result.ci_relative[0] <= truth_lift <= result.ci_relative[1]),
        "declared_significant": bool(result.is_significant),
        "backtest_mape": result.diagnostics.get("backtest_mape", np.nan),
    }


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def _score(frame: pd.DataFrame, label: str, arm: str) -> dict:
    return {
        "specification": label,
        "arm": arm,
        "draws": int(len(frame)),
        "mean_truth": frame["truth"].mean(),
        "mean_estimate": frame["estimate"].mean(),
        "bias_pp": frame["error"].mean() * 100,
        "rmse_pp": float(np.sqrt((frame["error"] ** 2).mean()) * 100),
        "max_abs_error_pp": frame["error"].abs().max() * 100,
        "interval_coverage": frame["covered"].mean(),
        "declared_significant": frame["declared_significant"].mean(),
        "mean_interval_width_pp": (frame["ci_high"] - frame["ci_low"]).mean() * 100,
        "mean_backtest_mape": frame["backtest_mape"].mean(),
    }


def score_only(out: Path) -> int:
    path = out / "recovery_study_draws.csv"
    if not path.exists() or path.stat().st_size == 0:
        print(f"no draws at {path}; run the study first", file=sys.stderr)
        return 1
    draws = pd.read_csv(path, keep_default_na=False, na_values=[""])
    rows = [
        _score(group, label, arm)
        for (label, arm), group in draws.groupby(["specification", "arm"], sort=False)
    ]
    scores = pd.DataFrame(rows).sort_values(["arm", "rmse_pp"])
    scores.to_csv(out / "recovery_study_scores.csv", index=False)

    print(f"\n{'specification':<34} {'arm':<12} {'n':>3} {'bias':>7} {'rmse':>7} "
          f"{'cover':>6} {'width':>7} {'sig':>5}")
    print("-" * 86)
    for _, r in scores.iterrows():
        print(
            f"{r['specification']:<34} {r['arm']:<12} {r['draws']:>3} "
            f"{r['bias_pp']:>+6.2f}pp {r['rmse_pp']:>6.2f}pp "
            f"{r['interval_coverage']:>5.0%} {r['mean_interval_width_pp']:>6.1f}pp "
            f"{r['declared_significant']:>4.0%}"
        )
    print(f"\nwrote {out / 'recovery_study_scores.csv'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/black_friday_2024.yaml"))
    parser.add_argument("--arm", choices=tuple(ARMS), default="effect")
    parser.add_argument("--seeds", default="0-7", help="e.g. 0-7 or 0,3,9")
    parser.add_argument("--n-bootstrap", type=int, default=400)
    parser.add_argument("--scale", choices=tuple(SCALES), default="small")
    parser.add_argument("--out", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--specs", nargs="*", default=None)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        return score_only(args.out)

    cfg = load_cohort(args.config)
    truth, seed_base = ARMS[args.arm]
    # The zero-effect arm only needs the headline specification: its job is
    # the false-positive rate, not a specification comparison.
    specs = (
        {HEADLINE_SPEC: SPECS[HEADLINE_SPEC]}
        if args.arm == "zero_effect"
        else {k: v for k, v in SPECS.items() if args.specs is None or k in args.specs}
    )

    path = args.out / "recovery_study_draws.csv"
    # A zero-byte file is left behind when a run dies mid-write, and reading
    # it raises rather than returning empty.
    existing = pd.DataFrame()
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, keep_default_na=False, na_values=[""])
    started = time.time()
    rows = []

    def flush() -> None:
        """Persist after every seed.

        Writing only at the end means a run that is interrupted -- a CI
        timeout, a killed shell -- loses everything it computed. At ~40s per
        seed that is expensive enough to be worth a redundant write.
        """
        if not rows:
            return
        fresh = pd.DataFrame(rows)
        combined = fresh
        if not existing.empty:
            # Re-running a seed replaces its rows rather than duplicating them.
            keys = set(zip(fresh["seed"], fresh["specification"], fresh["arm"], strict=True))
            keep = [
                (a, b, c) not in keys
                for a, b, c in zip(
                    existing["seed"],
                    existing["specification"],
                    existing["arm"],
                    strict=True,
                )
            ]
            combined = pd.concat([existing[keep], fresh], ignore_index=True)
        combined.to_csv(path, index=False)

    for i in _parse_seeds(args.seeds):
        seed = seed_base + i
        try:
            treated, truth_lift = _simulate(cfg, seed, truth, args.scale)
        except Exception as exc:
            print(f"  seed {seed} simulation failed: {exc}", file=sys.stderr)
            continue
        for label, spec_kwargs in specs.items():
            try:
                row = _estimate(cfg, seed, treated, truth_lift, spec_kwargs, args.n_bootstrap)
                rows.append({**row, "specification": label, "arm": args.arm})
            except Exception as exc:
                print(f"  seed {seed} / {label} failed: {exc}", file=sys.stderr)
        flush()
        print(f"  seed {seed} done ({time.time() - started:.0f}s)", flush=True)

    print(f"\n{len(rows)} new rows -> {path}")
    return score_only(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
