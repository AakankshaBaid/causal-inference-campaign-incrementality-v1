"""Assumption sensitivity: what does it cost to be wrong?

Two assumptions carry the measurement design, and neither can be verified
from the data at useful precision. This study prices them instead.

## Sweep 1 -- spillover reach (`--sweep leakage`)

Design D uses untreated properties outside participants' competitive sets as a
contemporaneous control series. That is what makes it precise, and it is
valid only if the campaign does not reach those properties. If a discount
pulls demand across market boundaries through destination substitution, the
control series is depressed during the campaign window, the counterfactual
follows it down, and the lift is overstated -- with tight intervals, which is
the dangerous version of wrong.

The sweep drives leakage from zero (Design D's assumption holds exactly) to the
full comp-set spillover rate (the campaign depresses the entire untreated
universe, so Design D's control is no cleaner than Design A's) and measures the bias each
design inherits. Design C's production predictor is year-lagged and therefore
cannot be moved by this year's campaign at all, so Design C should be flat across
the sweep. Confirming that flatness is the point: it establishes Design C as the
assumption-light fallback rather than merely the less precise option.

## Sweep 2 -- repeat participation (`--sweep repeat`)

With high repeat participation, last year's campaign sits inside the
pre-period fit window as an unmodelled treatment episode, and the model
absorbs it as ordinary seasonality. The sweep turns repeat participation off
and on with the prior-campaign indicator enabled and disabled, which
identifies the bias by difference rather than by assertion -- if the
indicator's value collapses when there is no prior campaign in the data, the
attribution was right.

Usage:
    python studies/assumption_sensitivity.py --sweep leakage --seeds 0-3
    python studies/assumption_sensitivity.py --sweep repeat  --seeds 0-3
    python studies/assumption_sensitivity.py --score-only
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
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
from incrementality.validation import relationship_stability_test

SCALE = {"n_treated": 1_200, "n_compset": 1_200, "n_distant": 1_500}

DESIGNS = {
    "A: competitor-set control": {"design": "competitor_control", "prior_control": True},
    "C: market-wide prior-year": {"design": "market_prior_year", "prior_control": True},
    "D: clean-pool control": {"design": "clean_pool_control", "prior_control": True},
}

# Leakage as a share of the comp-set spillover rate, so the endpoints are
# interpretable: 0.0 = Design D's assumption holds, 1.0 = spillover is uniform
# across the whole untreated universe and Design D's control is as dirty as Design A's.
LEAKAGE_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 1.00)

# Site-wide traffic halo reaching every untreated property. Distance cannot
# block this channel, so it is the failure mode that applies to a global,
# simultaneous campaign.
HALO_LEVELS = (0.0, 0.005, 0.010, 0.020)

# Drift in participants' seasonal sensitivity, breaking the assumption that
# the pre-period relationship still holds over the campaign window.
DRIFT_LEVELS = (0.0, 0.10, 0.20, 0.35)

REPEAT_CONDITIONS = {
    "repeat 85%, indicator on": {"repeat": 0.85, "prior_control": True},
    "repeat 85%, indicator off": {"repeat": 0.85, "prior_control": False},
    "repeat 0%, indicator on": {"repeat": 0.0, "prior_control": True},
    "repeat 0%, indicator off": {"repeat": 0.0, "prior_control": False},
}


def _simulate(cfg, seed: int, truth: GroundTruth):
    spec = SimulationSpec(
        activation_week=str(cfg.activation_week.date()),
        pre_weeks=cfg.pre_weeks,
        post_weeks=cfg.post_weeks + 2,
        seed=seed,
        truth=truth,
        **SCALE,
    )
    panel = split_control_pool(simulate_panel(spec), seed=seed + 1)
    truth_lift = true_average_lift(panel, cfg.post_start, cfg.post_end, truth=truth)
    series = aggregate_to_cohort(panel, cfg.outcomes)
    return series[series["cohort"] == "treated"].reset_index(drop=True), truth_lift


def _estimate(cfg, seed, treated, truth_lift, design_key, prior_control, n_bootstrap):
    prior_window = None
    if prior_control:
        start = cfg.activation_week - pd.Timedelta(weeks=52)
        prior_window = (
            start,
            start + pd.Timedelta(weeks=cfg.model.prior_campaign_weeks - 1),
        )
    design = build_design(
        treated,
        outcome=cfg.primary_outcome,
        regressors=PREDICTOR_SETS[design_key],
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
    }


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def _load(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return pd.DataFrame()


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return fresh
    seen = set(map(tuple, fresh[keys].itertuples(index=False, name=None)))
    keep = [tuple(row) not in seen for row in existing[keys].itertuples(index=False, name=None)]
    return pd.concat([existing[keep], fresh], ignore_index=True)


def score(out: Path) -> int:
    for sweep, group_keys in (
        ("leakage", ["leakage_pct", "design"]),
        ("repeat", ["condition"]),
        ("drift", ["drift_pct"]),
        ("halo", ["halo_bps", "design"]),
    ):
        path = out / f"assumption_{sweep}_draws.csv"
        draws = _load(path)
        if draws.empty:
            continue
        rows = []
        for key, chunk in draws.groupby(group_keys, sort=True):
            label = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    **dict(zip(group_keys, label, strict=True)),
                    "draws": len(chunk),
                    "mean_truth": chunk["truth"].mean(),
                    "mean_estimate": chunk["estimate"].mean(),
                    "bias_pp": chunk["error"].mean() * 100,
                    "rmse_pp": float(np.sqrt((chunk["error"] ** 2).mean()) * 100),
                    "coverage": chunk["covered"].mean(),
                    "width_pp": (chunk["ci_high"] - chunk["ci_low"]).mean() * 100,
                    **(
                        {
                            "divergence_pct": chunk["divergence"].mean() * 100,
                            "error_growth": chunk["error_growth"].mean(),
                            # Recomputed from the stored statistics so thresholds
                        # can be re-evaluated without re-simulating.
                        "flagged_rate": (
                            (chunk["divergence"] > 0.02)
                            | (chunk["error_growth"] > 1.25)
                        ).mean(),
                        }
                        if "divergence" in chunk.columns
                        else {}
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"assumption_{sweep}_scores.csv", index=False)

        print(f"\n=== {sweep.upper()} SWEEP ===")
        if sweep == "halo":
            pivot = frame.pivot(index="design", columns="halo_bps", values="bias_pp")
            print("\nbias (pp) by site-wide halo, in basis points of traffic:")
            print(pivot.round(2).to_string())
            print("\nRMSE (pp):")
            print(frame.pivot(index="design", columns="halo_bps", values="rmse_pp").round(2).to_string())
        elif sweep == "leakage":
            pivot = frame.pivot(index="design", columns="leakage_pct", values="bias_pp")
            print("\nbias (pp) by leakage as % of comp-set spillover:")
            print(pivot.round(2).to_string())
            rmse = frame.pivot(index="design", columns="leakage_pct", values="rmse_pp")
            print("\nRMSE (pp):")
            print(rmse.round(2).to_string())
        elif sweep == "drift":
            cols = [
                "drift_pct", "draws", "bias_pp", "rmse_pp", "coverage",
                "divergence_pct", "error_growth", "flagged_rate",
            ]
            print(frame[cols].round(3).to_string(index=False))
        else:
            cols = ["condition", "draws", "bias_pp", "rmse_pp", "coverage", "width_pp"]
            print(frame[cols].round(3).to_string(index=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/black_friday_2024.yaml"))
    parser.add_argument(
        "--sweep", choices=("leakage", "repeat", "drift", "halo"), default="leakage"
    )
    parser.add_argument("--seeds", default="0-3")
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        return score(args.out)

    cfg = load_cohort(args.config)
    base = GroundTruth()
    path = args.out / f"assumption_{args.sweep}_draws.csv"
    existing = _load(path)
    rows: list[dict] = []
    started = time.time()

    for i in _parse_seeds(args.seeds):
        seed = 3_000 + i
        if args.sweep == "leakage":
            for fraction in LEAKAGE_FRACTIONS:
                truth = replace(base, distant_spillover=base.spillover * fraction)
                try:
                    treated, truth_lift = _simulate(cfg, seed, truth)
                except Exception as exc:
                    print(f"  seed {seed} leak {fraction} sim failed: {exc}", file=sys.stderr)
                    continue
                for label, spec in DESIGNS.items():
                    try:
                        row = _estimate(
                            cfg, seed, treated, truth_lift,
                            spec["design"], spec["prior_control"], args.n_bootstrap,
                        )
                        rows.append(
                            {**row, "design": label, "leakage_pct": int(fraction * 100)}
                        )
                    except Exception as exc:
                        print(f"  seed {seed} / {label} failed: {exc}", file=sys.stderr)
            keys = ["seed", "design", "leakage_pct"]
        elif args.sweep == "halo":
            for halo in HALO_LEVELS:
                truth = replace(base, marketplace_halo=halo)
                try:
                    treated, truth_lift = _simulate(cfg, seed, truth)
                except Exception as exc:
                    print(f"  seed {seed} halo {halo} sim failed: {exc}", file=sys.stderr)
                    continue
                for label, spec in DESIGNS.items():
                    try:
                        row = _estimate(
                            cfg, seed, treated, truth_lift,
                            spec["design"], spec["prior_control"], args.n_bootstrap,
                        )
                        rows.append({**row, "design": label, "halo_bps": int(halo * 10_000)})
                    except Exception as exc:
                        print(f"  seed {seed} / {label} failed: {exc}", file=sys.stderr)
            keys = ["seed", "design", "halo_bps"]
        elif args.sweep == "drift":
            for drift in DRIFT_LEVELS:
                truth = replace(base, relationship_drift=drift)
                try:
                    treated, truth_lift = _simulate(cfg, seed, truth)
                except Exception as exc:
                    print(f"  seed {seed} drift {drift} sim failed: {exc}", file=sys.stderr)
                    continue
                try:
                    row = _estimate(
                        cfg, seed, treated, truth_lift,
                        "clean_pool_control", True, args.n_bootstrap,
                    )
                    outcome, stats = relationship_stability_test(treated, cfg)
                    rows.append(
                        {
                            **row,
                            "drift_pct": int(drift * 100),
                            "divergence": stats["divergence"],
                            "error_growth": stats["error_growth"],
                            "flagged": bool(not outcome.passed),
                        }
                    )
                except Exception as exc:
                    print(f"  seed {seed} drift {drift} failed: {exc}", file=sys.stderr)
            keys = ["seed", "drift_pct"]
        else:
            for label, cond in REPEAT_CONDITIONS.items():
                truth = replace(base, repeat_participation=cond["repeat"])
                try:
                    treated, truth_lift = _simulate(cfg, seed, truth)
                except Exception as exc:
                    print(f"  seed {seed} {label} sim failed: {exc}", file=sys.stderr)
                    continue
                try:
                    row = _estimate(
                        cfg, seed, treated, truth_lift,
                        "clean_pool_control", cond["prior_control"], args.n_bootstrap,
                    )
                    rows.append({**row, "condition": label})
                except Exception as exc:
                    print(f"  seed {seed} / {label} failed: {exc}", file=sys.stderr)
            keys = ["seed", "condition"]

        if rows:
            _merge(existing, pd.DataFrame(rows), keys).to_csv(path, index=False)
        print(f"  seed {seed} done ({time.time() - started:.0f}s)", flush=True)

    print(f"\n{len(rows)} new rows -> {path}")
    return score(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
