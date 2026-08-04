"""Schema definition and validation for training requests.

Two layers, matching the frontend:
  - robot catalog  profiles/robots.yaml      : robot -> supported tasks (task ids) and DR capability
  - task profile   profiles/tasks/<task>.yaml: goal_zones / behavior_presets / dr_off_overrides

A request likewise has two layers:
  - common base: robot / task / training_budget / sim2real_robustness / record_video / seed / output_name / behavior
  - task-specific: goal (spatial or velocity ranges)

Only the standard library and PyYAML are needed, so it runs standalone; a backend validates with
it before handing the request to the launcher.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

# Budget presets: training-length tier -> scale (calibrate on the target GPU and adjust)
BUDGET_PRESETS: dict[str, dict[str, int]] = {
    "quick": {"num_envs": 256, "max_iterations": 200},
    "standard": {"num_envs": 4096, "max_iterations": 1000},
    "thorough": {"num_envs": 8192, "max_iterations": 3000},
}

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")
ROBOTS_PATH = os.path.join(PROFILES_DIR, "robots.yaml")
TASKS_DIR = os.path.join(PROFILES_DIR, "tasks")


class ValidationError(Exception):
    """Validation failure; the message is meant to be shown to the user."""


def load_robots() -> dict[str, Any]:
    with open(ROBOTS_PATH) as f:
        return yaml.safe_load(f)


def load_task(task: str) -> dict[str, Any]:
    path = os.path.join(TASKS_DIR, f"{task}.yaml")
    if not os.path.isfile(path):
        raise ValidationError(f"unknown task type '{task}'")
    with open(path) as f:
        return yaml.safe_load(f)


def robot_dr(robot: str, robots: dict | None = None) -> Any:
    """Return the robot's DR capability: True / False / 'builtin'."""
    robots = robots or load_robots()
    return robots.get(robot, {}).get("dr", False)


def _check_range(name: str, val: Any, limits: list[float]) -> list[str]:
    if not (isinstance(val, (list, tuple)) and len(val) == 2):
        return [f"{name}: expected two numbers as [min, max], got {val!r}"]
    lo, hi = val
    if not all(isinstance(x, (int, float)) for x in (lo, hi)):
        return [f"{name}: bounds must be numbers"]
    errs = []
    if lo > hi:
        errs.append(f"{name}: lower bound {lo} must not exceed upper bound {hi}")
    lmin, lmax = limits
    if lo < lmin or hi > lmax:
        errs.append(f"{name}: range [{lo}, {hi}] is outside the feasible interval [{lmin}, {lmax}]")
    return errs


def validate(request: dict[str, Any]) -> dict[str, Any]:
    """Validate a training request. Raises ValidationError listing every problem found; on success
    returns the request with defaults filled in."""
    req = dict(request)
    errors: list[str] = []
    robots = load_robots()

    # --- Robot and task ---
    robot = req.get("robot")
    if robot not in robots:
        raise ValidationError(f"unknown robot '{robot}'; supported: {', '.join(robots)}")
    rob = robots[robot]

    task = req.get("task")
    if task not in rob["tasks"]:
        raise ValidationError(
            f"robot '{robot}' does not support task '{task}' (it supports: {', '.join(rob['tasks'])})"
        )
    profile = load_task(task)

    # --- Common base ---
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
    req.setdefault("output_name", f"{robot}_{task}")

    # --- Task-specific: goal_zones ---
    zones_spec = profile.get("goal_zones", {})
    goal = req.setdefault("goal", {})
    if not isinstance(goal, dict):
        raise ValidationError("goal must be an object")
    for zone_name, zone_val in goal.items():
        if zone_name not in zones_spec:
            errors.append(f"unknown zone '{zone_name}' (this task supports: {list(zones_spec)})")
            continue
        axes = zones_spec[zone_name]["axes"]
        for axis, axis_val in zone_val.items():
            if axis not in axes:
                errors.append(f"{zone_name}.{axis}: this zone has no such axis")
                continue
            label = axes[axis].get("name", axis)
            errs = _check_range(f"{zones_spec[zone_name]['label']} · {label}", axis_val, axes[axis]["limits"])
            errors += errs

    if errors:
        raise ValidationError("request validation failed:\n  - " + "\n  - ".join(errors))
    return req
