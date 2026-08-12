"""Publish the Isaac Sim view into a LiveKit room over WebRTC, using LiveKit's own TURN/NAT
traversal.

Frames are captured from an Isaac viewport camera (the same approach as render_preview) and
published as a video track with the LiveKit Python SDK; a browser subscribes with a LiveKit
client to watch. Unlike Isaac's native WebRTC, LiveKit handles NAT traversal over the public
internet (advertising the node's public IP, with TCP 7881 as fallback).

Usage (inside the pod):
  ./isaaclab.sh -p training-service/livekit_publisher.py --task <task_id> [--template t.yaml]
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description="Publish Isaac Sim video to LiveKit")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--template", type=str, default=None)
parser.add_argument("--url", type=str, default="ws://115.191.16.193:7880")
parser.add_argument("--api-key", type=str, default="isaackey")
parser.add_argument("--api-secret", type=str, default="isaacsecretABCDEF0123456789abcdef")
parser.add_argument("--room", type=str, default="isaac")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fps", type=int, default=30)
a, _ = parser.parse_known_args()
_task, _tmpl, _url, _key, _secret, _room = a.task, a.template, a.url, a.api_key, a.api_secret, a.room
_w, _h, _fps = a.width, a.height, a.fps
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True).app

import asyncio  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from livekit import api, rtc  # noqa: E402

import isaaclab.envs.mdp as core_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import isaac_so_arm101.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, "/workspace/isaaclab/training-service")
import template_env  # noqa: E402


def setup_env():
    env_cfg = parse_env_cfg(_task, num_envs=1)
    if _tmpl:
        with open(_tmpl) as f:
            template_env.apply_template(env_cfg, yaml.safe_load(f), core_mdp)
    for grp in ("commands", "scene"):
        obj = getattr(env_cfg, grp, None)
        if obj is not None:
            for term in vars(obj).values():
                if hasattr(term, "debug_vis"):
                    term.debug_vis = False
    env_cfg.scene.stream_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/stream_cam", height=_h, width=_w, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.9, 0.9, 0.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    import gymnasium as gym
    env = gym.make(_task, cfg=env_cfg)
    env.reset()
    return env


# LiveKit runs in its own thread with its own event loop, to stay clear of Kit's asyncio loop.
# Isaac stays on the main thread and hands frames over through a shared VideoSource
# (capture_frame is thread-safe).
_shared = {"source": None, "ready": False, "err": None}


def _livekit_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        token = (
            api.AccessToken(_key, _secret)
            .with_identity("isaac-publisher")
            .with_name("Isaac Sim")
            .with_grants(api.VideoGrants(room_join=True, room=_room, can_publish=True, can_subscribe=True))
            .to_jwt()
        )
        room = rtc.Room()
        await room.connect(_url, token)
        print(f"[livekit] connected to room '{_room}' @ {_url}", flush=True)
        source = rtc.VideoSource(_w, _h)
        track = rtc.LocalVideoTrack.create_video_track("isaac-video", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
        _shared["source"] = source
        _shared["ready"] = True
        print("[livekit] publishing 'isaac-video'; subscribe from a browser to watch.", flush=True)
        while True:
            await asyncio.sleep(1)

    try:
        loop.run_until_complete(run())
    except Exception as e:  # noqa: BLE001
        _shared["err"] = repr(e)
        print("[livekit] ERROR:", e, flush=True)


def main():
    env = setup_env()
    cam = env.unwrapped.scene["stream_cam"]
    device = env.unwrapped.device
    act_dim = env.unwrapped.action_manager.total_action_dim
    target = torch.tensor([[0.2, 0.0, 0.15]], device=device)

    threading.Thread(target=_livekit_thread, daemon=True).start()
    while not _shared["ready"] and _shared["err"] is None and app.is_running():
        time.sleep(0.2)
    if _shared["err"]:
        return 1

    dt = 1.0 / _fps
    i = 0
    while app.is_running():
        act = (torch.rand((env.unwrapped.num_envs, act_dim), device=device) * 2 - 1) * 0.3
        env.step(act)
        ang = i * 0.02
        eye = torch.tensor([[0.9 * math.cos(ang), 0.9 * math.sin(ang), 0.7]], device=device)
        cam.set_world_poses_from_view(eyes=eye, targets=target)
        cam.update(dt=env.unwrapped.sim.get_physics_dt())

        rgb = cam.data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] == 3:
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        else:
            rgba = rgb
        _shared["source"].capture_frame(
            rtc.VideoFrame(_w, _h, rtc.VideoBufferType.RGBA, rgba.astype(np.uint8).tobytes()))
        i += 1
        time.sleep(dt)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        app.close()
    raise SystemExit(rc)
