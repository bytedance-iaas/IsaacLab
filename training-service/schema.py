"""Schema definition and validation for training requests.

Two layers:
  - a common base shared by every task: model / task / training_budget / sim2real_robustness /
    record_video / seed / output_name / behavior
  - a task-specific section described by profiles/<task>.yaml: goal_zones (spatial ranges)

Validation only needs the standard library and PyYAML, so it can run standalone; a backend
validates with it before handing the request to the launcher.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

# --- Budget presets: training-length tier -> scale (calibrated on A30, tune as needed) ---
BUDGET_PRESETS: dict[str, dict[str, int]] = {
    "quick": {"num_envs": 256, "max_iterations": 200},       # ~minutes, smoke test
    "standard": {"num_envs": 4096, "max_iterations": 1000},  # ~1 hour, usable policy
    "thorough": {"num_envs": 8192, "max_iterations": 3000},  # ~hours, high quality
}

VALID_MODELS = ("so100", "so101")
PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


class ValidationError(Exception):
    """Validation failure; the message is meant to be shown to the user."""


def available_tasks() -> list[str]:
    """Currently supported tasks (one yaml per task under profiles/)."""
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith(".yaml"))


def load_profile(task: str) -> dict[str, Any]:
    path = os.path.join(PROFILES_DIR, f"{task}.yaml")
    if not os.path.isfile(path):
        raise ValidationError(
            f"unknown task '{task}'; supported: {', '.join(available_tasks()) or '(none)'}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def _check_range(name: str, val: Any, limits: list[float]) -> list[str]:
    """Validate a [lo, hi] range: shape, lo <= hi, and that it stays within limits."""
    errs: list[str] = []
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        return [f"{name}: expected two numbers as [min, max], got {val!r}"]
    lo, hi = val
    if not all(isinstance(x, (int, float)) for x in (lo, hi)):
        return [f"{name}: bounds must be numbers"]
    if lo > hi:
        errs.append(f"{name}: lower bound {lo} must not exceed upper bound {hi}")
    lmin, lmax = limits
    if lo < lmin or hi > lmax:
        errs.append(f"{name}: range [{lo}, {hi}] is outside the allowed interval [{lmin}, {lmax}]")
    return errs


def validate(request: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a training request. Raises ValidationError with every problem found; on success
    returns the request with defaults filled in."""
    req = dict(request)  # shallow copy; never mutate the caller's object
    errors: list[str] = []

    # --- Common base ---
    task = req.get("task")
    if not task:
        raise ValidationError("missing 'task'")
    if profile is None:
        profile = load_profile(task)

    model = req.get("model")
    if model not in VALID_MODELS:
        errors.append(f"model must be one of {VALID_MODELS}, got {model!r}")
    elif model not in profile["task_ids"]:
        errors.append(f"task '{task}' does not support model '{model}' (supported: {list(profile['task_ids'])})")

    budget = req.setdefault("training_budget", "standard")
    if budget not in BUDGET_PRESETS:
        errors.append(f"training_budget must be one of {list(BUDGET_PRESETS)}, got {budget!r}")

    behavior = req.setdefault("behavior", "balanced")
    presets = profile.get("behavior_presets", {})
    if behavior not in presets:
        errors.append(f"behavior must be one of {list(presets)}, got {behavior!r}")

    req.setdefault("sim2real_robustness", True)
    if not isinstance(req["sim2real_robustness"], bool):
        errors.append("sim2real_robustness must be true or false")

    req.setdefault("record_video", False)
    req.setdefault("seed", 42)
    if not isinstance(req["seed"], int):
        errors.append("seed must be an integer")

    req.setdefault("output_name", f"{model}_{task}")

    # --- Task-specific section: goal_zones ---
    zones_spec = profile.get("goal_zones", {})
    goal = req.setdefault("goal", {})
    if not isinstance(goal, dict):
        raise ValidationError("goal must be an object")
    for zone_name, zone_val in goal.items():
        if zone_name not in zones_spec:
            errors.append(f"unknown zone '{zone_name}' (this task supports: {list(zones_spec)})")
            continue
        for axis, axis_val in zone_val.items():
            axis_spec = zones_spec[zone_name].get(axis)
            if axis_spec is None:
                errors.append(f"{zone_name}.{axis}: this zone has no such axis")
                continue
            errors += _check_range(f"{zone_name}.{axis}", axis_val, axis_spec["limits"])

    if errors:
        raise ValidationError("request validation failed:\n  - " + "\n  - ".join(errors))
    return req
