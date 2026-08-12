"""Publish the live Isaac training view to LiveKit as two tracks: a panorama looking down on the
whole env grid, and a closeup of a single robot.

- Frames are read back (GPU to CPU) in the main thread's app update callback; touching CUDA from
  another thread causes an illegal memory access.
- The LiveKit thread only reads CPU frames and publishes the two video tracks, "pano" and "closeup".
- Called from train.py --stream; --stream_eye and --stream_target adjust the closeup camera.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

DEFAULT_URL = "ws://115.191.16.193:7880"
DEFAULT_KEY = "isaackey"
DEFAULT_SECRET = "isaacsecretABCDEF0123456789abcdef"

PANO_W, PANO_H = 960, 540      # panorama resolution
CLOSE_W, CLOSE_H = 640, 360    # closeup resolution


def _cam(prim, w, h):
    return CameraCfg(
        prim_path=prim, height=h, width=w, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.9, 0.9, 0.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def add_stream_camera(env_cfg):
    """Add the panorama and closeup cameras to the scene before the env is created, and disable the
    debug markers."""
    for grp in ("commands", "scene"):
        obj = getattr(env_cfg, grp, None)
        if obj is not None:
            for term in vars(obj).values():
                if hasattr(term, "debug_vis"):
                    term.debug_vis = False
    env_cfg.scene.stream_pano = _cam("{ENV_REGEX_NS}/stream_pano", PANO_W, PANO_H)
    env_cfg.scene.stream_close = _cam("{ENV_REGEX_NS}/stream_close", CLOSE_W, CLOSE_H)


def _set_pose(cam, eye, target, device, n):
    cam.set_world_poses_from_view(
        eyes=torch.tensor([list(eye)], device=device, dtype=torch.float32).repeat(n, 1),
        targets=torch.tensor([list(target)], device=device, dtype=torch.float32).repeat(n, 1),
    )


def start_publisher(env, room: str = "isaac", url: str = DEFAULT_URL,
                    key: str = DEFAULT_KEY, secret: str = DEFAULT_SECRET, fps: int = 15,
                    close_eye=(0.9, 0.9, 0.7), close_target=(0.2, 0.0, 0.15)):
    import omni.kit.app
    from livekit import api, rtc

    unwrapped = env.unwrapped
    device = unwrapped.device
    n = unwrapped.num_envs
    pano_cam = unwrapped.scene["stream_pano"]
    close_cam = unwrapped.scene["stream_close"]

    # Panorama: derive the overhead pose from the env grid
    origins = unwrapped.scene.env_origins.float()
    center = origins.mean(dim=0)
    span = float((origins.max(dim=0).values - origins.min(dim=0).values).max().item())
    d = span * 0.65 + 2.0
    pano_eye = (center + torch.tensor([0.0, -d, d * 0.85 + 1.5], device=device)).tolist()
    pano_target = (center + torch.tensor([0.0, 0.0, 0.15], device=device)).tolist()
    # Closeup: the offset is relative to env_0's origin, so add env_0's world position
    # (env_0 is usually not at the world origin)
    env0 = origins[0]
    close_eye_w = (env0 + torch.tensor(list(close_eye), device=device)).tolist()
    close_target_w = (env0 + torch.tensor(list(close_target), device=device)).tolist()
    try:
        _set_pose(pano_cam, pano_eye, pano_target, device, n)
        _set_pose(close_cam, close_eye_w, close_target_w, device, n)
        print(f"[livekit] cameras ready, panorama eye={[round(x,2) for x in pano_eye]} "
              f"closeup eye (world)={[round(x,2) for x in close_eye_w]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("[livekit] failed to set camera pose:", e, flush=True)

    shared = {"pano": None, "close": None, "srcs": None, "sub": None}
    period = 1.0 / fps
    last = [0.0]

    def _grab(cam):
        out = cam.data.output.get("rgb")
        if out is None or out.shape[0] == 0:
            return None
        rgb = out[0].detach().cpu().numpy()
        if rgb.shape[-1] == 3:
            rgb = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        return np.ascontiguousarray(rgb.astype(np.uint8))

    def _on_update(_e):
        now = time.time()
        if now - last[0] < period:
            return
        try:
            shared["pano"] = _grab(pano_cam)
            shared["close"] = _grab(close_cam)
            last[0] = now
        except Exception:  # noqa: BLE001
            pass

    shared["sub"] = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_update, name="livekit-capture")

    def _lk_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():
            token = (
                api.AccessToken(key, secret)
                .with_identity("isaac-train").with_name("Isaac Training")
                .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
                .to_jwt()
            )
            r = rtc.Room()
            await r.connect(url, token)
            src_p = rtc.VideoSource(PANO_W, PANO_H)
            src_c = rtc.VideoSource(CLOSE_W, CLOSE_H)
            await r.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("pano", src_p),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
            await r.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("closeup", src_c),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
            shared["srcs"] = (src_p, src_c)
            print(f"[livekit] publishing training video (panorama + closeup) to room '{room}'", flush=True)
            while True:
                if shared["pano"] is not None:
                    src_p.capture_frame(rtc.VideoFrame(PANO_W, PANO_H, rtc.VideoBufferType.RGBA, shared["pano"].tobytes()))
                if shared["close"] is not None:
                    src_c.capture_frame(rtc.VideoFrame(CLOSE_W, CLOSE_H, rtc.VideoBufferType.RGBA, shared["close"].tobytes()))
                await asyncio.sleep(period)

        try:
            loop.run_until_complete(run())
        except Exception as e:  # noqa: BLE001
            print("[livekit] ERROR:", e, flush=True)

    threading.Thread(target=_lk_thread, daemon=True).start()
