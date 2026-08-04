"""Generic training launcher: turns a user-supplied training request (YAML) into an Isaac Lab
training command and runs it.

Usage:
    python launcher.py --config request.yaml            # validate and start training
    python launcher.py --config request.yaml --dry-run  # print the command without running it

Task-agnostic: all task-specific knowledge lives in profiles/<task>.yaml, so adding a task
requires no change here. A backend can also import build_command(request) directly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

import yaml

import schema

ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/workspace/isaaclab")
ISAACLAB_SH = os.path.join(ISAACLAB_ROOT, "isaaclab.sh")


def _fmt(value: Any) -> str:
    """Format a value as the right-hand side of a Hydra override. Lists become [a,b] without
    spaces, which avoids shell and Hydra parsing issues."""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def build_command(request: dict[str, Any]) -> list[str]:
    """Validate the request and build the training command as an argv list (no shell, no quoting)."""
    profile = schema.load_profile(request["task"])
    req = schema.validate(request, profile)

    task_id = profile["task_ids"][req["model"]]
    budget = schema.BUDGET_PRESETS[req["training_budget"]]
    train_script = os.path.join(ISAACLAB_ROOT, profile["train_script"])

    argv: list[str] = [
        ISAACLAB_SH, "-p", train_script,
        "--task", task_id,
        "--headless",
        "--num_envs", str(budget["num_envs"]),
        "--max_iterations", str(budget["max_iterations"]),
        "--seed", str(req["seed"]),
    ]
    if req.get("record_video"):
        argv += ["--video", "--video_length", "200", "--video_interval", "2000"]

    # --- Hydra overrides ---
    overrides: list[str] = []

    # 1) Task-specific: goal_zones -> command/event ranges
    zones_spec = profile.get("goal_zones", {})
    for zone_name, zone_val in req.get("goal", {}).items():
        for axis, axis_val in zone_val.items():
            path = zones_spec[zone_name][axis]["path"]
            overrides.append(f"{path}={_fmt(axis_val)}")

    # 2) Behavior preset -> reward weights
    for path, val in profile.get("behavior_presets", {}).get(req["behavior"], {}).items():
        overrides.append(f"{path}={_fmt(val)}")

    # 3) sim2real off -> neutralize the domain-randomization terms
    if not req["sim2real_robustness"]:
        for path, val in profile.get("dr_off_overrides", {}).items():
            overrides.append(f"{path}={_fmt(val)}")

    # 4) Advanced: expert-supplied overrides, passed through verbatim
    overrides += req.get("advanced", {}).get("overrides", [])

    return argv + overrides


def main() -> int:
    ap = argparse.ArgumentParser(description="Isaac Lab training launcher")
    ap.add_argument("--config", required=True, help="training request YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the command without executing it")
    args = ap.parse_args()

    with open(args.config) as f:
        request = yaml.safe_load(f)

    try:
        cmd = build_command(request)
    except schema.ValidationError as e:
        print(f"[validation failed] {e}", file=sys.stderr)
        return 2

    printable = " ".join(cmd)
    if args.dry_run:
        print(printable)
        return 0

    print(f"[launcher] running:\n{printable}\n", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
