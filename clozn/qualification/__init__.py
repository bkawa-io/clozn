"""Model-free qualification planning.

The planner is deliberately separate from the lab runners: it can inspect a GGUF header and
describe the work required to qualify an artifact without importing Torch/Transformers or starting
the private worker.
"""

from .planner import PLAN_SCHEMA, build_plan, plan_from_model
from .pipeline import RUN_SCHEMA, build_run, default_run_path, run_core, run_external_step, validate_jlens_artifact
from .lab import (
    LAB_SCHEMA, INSTALL_SCHEMA, acceptance_fixture, install_artifact,
    rollback_artifact, record_jlens_validation, run_battery,
    run_jlens_fit,
)

__all__ = [
    "PLAN_SCHEMA", "RUN_SCHEMA", "build_plan", "plan_from_model", "build_run", "run_core",
    "default_run_path",
    "run_external_step", "validate_jlens_artifact", "LAB_SCHEMA", "INSTALL_SCHEMA",
    "run_jlens_fit",
    "record_jlens_validation",
    "run_battery", "install_artifact",
    "rollback_artifact", "acceptance_fixture",
]
