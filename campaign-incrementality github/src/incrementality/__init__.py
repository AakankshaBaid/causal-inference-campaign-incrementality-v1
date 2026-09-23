"""Causal measurement of promotional campaign incrementality.

Public surface:

>>> from incrementality import load_cohort, load_or_simulate_panel, run_cohort
>>> cfg = load_cohort("configs/bfcm_2024.yaml")
>>> panel = load_or_simulate_panel(cfg)
>>> result = run_cohort(cfg, panel)
>>> print(result.headline())
"""

from .assumptions import check_assumptions
from .compat import CausalImpactModel, frame_from_panel
from .config import (
    PREDICTOR_SETS,
    CohortConfig,
    EconomicsConfig,
    ModelConfig,
    load_cohort,
)
from .counterfactual import get_backend
from .diagnostics import compare_designs, evaluate_model
from .drivers import decompose
from .economics import evaluate as evaluate_economics
from .features import build_donor_design
from .inference import estimate_uplift
from .pipeline import CohortResult, load_or_simulate_panel, run_cohort
from .simulate import (
    GroundTruth,
    SimulationSpec,
    simulate_panel,
    split_control_pool,
)
from .stacking import estimate_stacking_value

__version__ = "0.3.0"

__all__ = [
    "CausalImpactModel",
    "check_assumptions",
    "CohortConfig",
    "CohortResult",
    "EconomicsConfig",
    "GroundTruth",
    "ModelConfig",
    "SimulationSpec",
    "estimate_stacking_value",
    "estimate_uplift",
    "PREDICTOR_SETS",
    "build_donor_design",
    "compare_designs",
    "evaluate_model",
    "decompose",
    "evaluate_economics",
    "frame_from_panel",
    "get_backend",
    "load_cohort",
    "load_or_simulate_panel",
    "run_cohort",
    "simulate_panel",
    "split_control_pool",
]
