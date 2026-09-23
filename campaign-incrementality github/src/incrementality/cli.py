"""Command line entry point.

    python -m incrementality run configs/bfcm_2024.yaml
    python -m incrementality run configs/bfcm_2024.yaml --backend synthetic_control
    python -m incrementality all
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_cohort
from .pipeline import load_or_simulate_panel, run_cohort


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_one(config_path: Path, args: argparse.Namespace) -> int:
    cfg = load_cohort(config_path)
    if args.backend:
        cfg = replace(cfg, model=replace(cfg.model, backend=args.backend))
    panel = load_or_simulate_panel(cfg, data_path=args.data, seed=args.seed)
    result = run_cohort(
        cfg,
        panel,
        output_dir=args.output_dir,
        n_placebo=args.n_placebo,
        make_figures=not args.no_figures,
    )
    print(result.headline())
    print(f"  memo: {result.artifacts['memo']}")
    return 0 if result.validation.all_passed else 1


def _diagnose(config_path: Path, args: argparse.Namespace) -> int:
    """Answer 'is this the right model, and is it good enough?' before trusting it."""
    from . import diagnostics as dg
    from .counterfactual import get_backend
    from .inference import estimate_uplift
    from .simulate import aggregate_to_cohort

    cfg = load_cohort(config_path)
    if args.backend:
        cfg = replace(cfg, model=replace(cfg.model, backend=args.backend))
    panel = load_or_simulate_panel(cfg, data_path=args.data, seed=args.seed)
    series = aggregate_to_cohort(panel, cfg.outcomes)
    treated = series[series["cohort"] == "treated"].reset_index(drop=True)

    design = dg.build_design_for_eval(treated, cfg)
    result = estimate_uplift(
        get_backend(cfg.model.backend),
        design,
        alpha=cfg.model.alpha,
        n_bootstrap=cfg.model.n_bootstrap,
        materiality_threshold=cfg.materiality_threshold,
    )
    report = dg.evaluate_model(
        panel, treated, cfg, result, design, n_placebo=args.n_placebo
    )
    comparison = dg.compare_designs(treated, cfg)

    tables = Path(args.output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in cfg.name.lower()).strip("_")
    report.to_frame().to_csv(tables / f"{slug}_model_checks.csv", index=False)
    report.power.to_csv(tables / f"{slug}_power_curve.csv", index=False)
    comparison.to_csv(tables / f"{slug}_design_comparison.csv", index=False)
    if report.predictor_value is not None:
        report.predictor_value.to_csv(tables / f"{slug}_predictor_value.csv", index=False)

    print(f"\n{cfg.name}: model evaluation")
    print(report.to_frame()[["family", "check", "result", "value"]].to_string(index=False))
    print(
        f"\nMDE at 80% power: {report.mde:.2%}  |  "
        f"post/pre RMSPE ratio {report.rmspe_ratio:.2f} "
        f"(permutation p={report.rmspe_p_value:.3f})"
    )
    print("\nDesign comparison (ranked by out-of-sample forecast error):")
    print(comparison.to_string(index=False))
    if report.predictor_value is not None:
        print("\nPredictor value:")
        print(report.predictor_value[["predictor", "relative_value", "verdict"]].to_string(index=False))
    print(f"\nfit for purpose: {report.fit_for_purpose}")
    return 0 if report.fit_for_purpose else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="incrementality", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "all", "diagnose", "assets"):
        sp = sub.add_parser(name)
        if name in ("run", "diagnose", "assets"):
            sp.add_argument("config", type=Path)
        else:
            sp.add_argument("--config-dir", type=Path, default=Path("configs"))
        sp.add_argument("--data", type=Path, default=None, help="real panel; omit to simulate")
        sp.add_argument("--output-dir", type=Path, default=Path("outputs"))
        sp.add_argument("--backend", choices=("bsts", "synthetic_control", "prophet"))
        sp.add_argument("--n-placebo", type=int, default=40)
        sp.add_argument("--seed", type=int, default=20240101)
        sp.add_argument("--no-figures", action="store_true")
        sp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "run":
        return _run_one(args.config, args)
    if args.command == "diagnose":
        return _diagnose(args.config, args)
    if args.command == "assets":
        from .assets import build_all

        cfg = load_cohort(args.config)
        slug = "".join(c if c.isalnum() else "_" for c in cfg.name.lower()).strip("_")
        record = Path(args.output_dir) / "tables" / f"{slug}_run_record.json"
        if not record.exists():
            print(f"run the campaign first: no record at {record}")
            return 1
        for made in build_all(record, Path(args.output_dir) / "figures"):
            print(f"  wrote {made}")
        return 0

    exit_code = 0
    for config_path in sorted(args.config_dir.glob("*.yaml")):
        exit_code |= _run_one(config_path, args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
