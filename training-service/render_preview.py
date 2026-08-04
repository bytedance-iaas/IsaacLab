"""Render a scene preview: rasterize the (templated) task scene to a PNG so the user sees the
real geometry in the UI.

The scene is built with template_env, so what is previewed is what is trained: the obstacles and
swapped objects match the training scene. A camera is then added to take the shot. Runs under
isaac python.

Usage:
  python render_preview.py --task <base_task> [--template t.yaml] --out preview.png
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description="Render a task scene preview")
parser.add_argument("--task", type=str, required=True, help="base task id")
parser.add_argument("--template", type=str, default=None, help="scene/reward template YAML")
parser.add_argument("--out", type=str, default="/tmp/preview.png", help="output PNG path")
parser.add_argument("--eye", type=float, nargs=3, default=[0.9, 0.9, 0.7], help="camera position")
parser.add_argument("--target", type=float, nargs=3, default=[0.2, 0.0, 0.15], help="camera look-at target")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args_cli, _ = parser.parse_known_args()

# Clear sys.argv after parsing so AppLauncher does not try to parse these arguments
_out, _task, _tmpl = args_cli.out, args_cli.task, args_cli.template
_eye, _target, _w, _h = args_cli.eye, args_cli.target, args_cli.width, args_cli.height
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app  # rendering requires enable_cameras

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.envs.mdp as core_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  registers the stock tasks
import isaac_so_arm101.tasks  # noqa: F401,E402  registers the SO-ARM tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, "/workspace/isaaclab/training-service")
import template_env  # noqa: E402


def _disable_debug_vis(env_cfg):
    """Turn off debug visualizations such as goal-pose and frame markers (the colored arrows) to
    produce a clean scene image."""
    if getattr(env_cfg, "commands", None) is not None:
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
    for asset in vars(env_cfg.scene).values():
        if hasattr(asset, "debug_vis"):
            asset.debug_vis = False


def main() -> int:
    env_cfg = parse_env_cfg(_task, num_envs=1)

    # Apply the scene/reward template using the same generator training uses
    if _tmpl:
        with open(_tmpl) as f:
            tmpl = yaml.safe_load(f)
        notes = template_env.apply_template(env_cfg, tmpl, core_mdp)
        print("[preview] template applied:", notes, flush=True)

    _disable_debug_vis(env_cfg)  # drop the debug arrows

    # Add a preview camera to the scene
    env_cfg.scene.preview_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/preview_cam",
        height=_h, width=_w, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(_eye), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    import gymnasium as gym
    env = gym.make(_task, cfg=env_cfg)
    env.reset()

    cam = env.unwrapped.scene["preview_cam"]
    cam.set_world_poses_from_view(
        eyes=torch.tensor([_eye], device=env.unwrapped.device),
        targets=torch.tensor([_target], device=env.unwrapped.device),
    )
    # Step a few frames so the render settles
    for _ in range(6):
        env.unwrapped.sim.step()
        cam.update(dt=env.unwrapped.sim.get_physics_dt())

    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8)

    from PIL import Image
    Image.fromarray(rgb).save(_out)
    print(f"[preview] rendered -> {_out}  ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        app.close()
    raise SystemExit(rc)
