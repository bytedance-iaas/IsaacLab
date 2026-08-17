"""Publish the live Isaac training view to LiveKit as one track: a panorama looking down on the
whole env grid.

There was also a closeup track, framed by a fixed world-space pose. That pose was chosen for one
arm task, so on any other robot or terrain it pointed at empty space -- a second video pane showing
nothing is worse than no second pane. The panorama derives its framing from the env grid, so it
adapts on its own.

- Frames are read back (GPU to CPU) in the main thread's app update callback; touching CUDA from
  another thread causes an illegal memory access.
- The LiveKit thread only reads CPU frames and publishes the video track "pano".
- Called from train.py --stream.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# Where to publish. These are configuration, not constants: a deployment that is not ours has its
# own LiveKit, and a run that hardcodes ours would either fail to connect or -- worse -- succeed,
# putting someone else's training video on our server.
#
# The default is the in-cluster Service address rather than the load balancer's public IP: the
# publisher runs inside the same cluster, so going out to the public IP and back in is a hairpin
# that pays for a round trip and egress for nothing.
DEFAULT_URL = os.environ.get("LIVEKIT_URL", "ws://livekit-isaac-clb.default.svc.cluster.local:7880")
DEFAULT_KEY = os.environ.get("LIVEKIT_API_KEY", "")
DEFAULT_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

# One room per run, so two people training at the same time do not land in each other's video.
DEFAULT_ROOM = os.environ.get("LIVEKIT_ROOM") or f"train-{os.environ.get('JOB_NAME', 'local')}"

PANO_W, PANO_H = 960, 540      # panorama resolution


def _cam(prim, w, h):
    # No offset: the pose is set from the env grid once the scene exists, which is the only way to
    # frame a layout whose size is not known until then.
    return CameraCfg(
        prim_path=prim, height=h, width=w, data_types=["rgb"],
        # Far plane 1000, not 30: the camera pulls back to frame the whole env grid, and on a
        # generated rough terrain that puts the ground tens of metres away. With a 30 m far plane
        # everything but the sky got clipped, and the stream showed a gradient over nothing.
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 1000.0)),
    )


def add_stream_camera(env_cfg):
    """Add the panorama camera to the scene before the env is created, and disable the debug
    markers."""
    for grp in ("commands", "scene"):
        obj = getattr(env_cfg, grp, None)
        if obj is not None:
            for term in vars(obj).values():
                if hasattr(term, "debug_vis"):
                    term.debug_vis = False
    env_cfg.scene.stream_pano = _cam("{ENV_REGEX_NS}/stream_pano", PANO_W, PANO_H)


def _set_pose(cam, eye, target, device, n):
    cam.set_world_poses_from_view(
        eyes=torch.tensor([list(eye)], device=device, dtype=torch.float32).repeat(n, 1),
        targets=torch.tensor([list(target)], device=device, dtype=torch.float32).repeat(n, 1),
    )


def start_publisher(env, room: str = DEFAULT_ROOM, url: str = DEFAULT_URL,
                    key: str = DEFAULT_KEY, secret: str = DEFAULT_SECRET, fps: int = 15):
    import omni.kit.app
    from livekit import api, rtc

    if not key or not secret:
        # Say so and carry on: --stream is a convenience, and losing the training run because the
        # credentials for the video feed are missing would be the wrong trade.
        print(
            "[livekit] LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set, so no video will be"
            " published. Training continues normally.",
            flush=True,
        )
        return

    unwrapped = env.unwrapped
    device = unwrapped.device
    n = unwrapped.num_envs
    pano_cam = unwrapped.scene["stream_pano"]

    # Initial pose from the env grid, so there is a sensible view before the first frame.
    origins = unwrapped.scene.env_origins.float()
    center = origins.mean(dim=0)
    span = float((origins.max(dim=0).values - origins.min(dim=0).values).max().item())
    d = span * 0.65 + 2.0
    pano_eye = (center + torch.tensor([0.0, -d, d * 0.85 + 1.5], device=device)).tolist()
    pano_target = (center + torch.tensor([0.0, 0.0, 0.15], device=device)).tolist()
    try:
        _set_pose(pano_cam, pano_eye, pano_target, device, n)
        print(f"[livekit] camera ready, panorama eye={[round(x,2) for x in pano_eye]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("[livekit] failed to set camera pose:", e, flush=True)

    # Then follow the robots. Framing the origin grid only works when the grid is compact; on a
    # generated terrain the origins span the whole map and the robots end up sub-pixel. So track
    # the centroid of the actual robot positions, cap how much area the view tries to cover, and
    # smooth the motion so resets do not yank the camera. Scenes without a "robot" articulation
    # keep the static grid framing.
    robot = unwrapped.scene.articulations.get("robot")
    _MAX_VIEW_SPAN = 18.0   # metres of robot spread the view will try to contain
    _EMA = 0.05             # per-tick smoothing; ~1.3 s to settle at 15 fps
    _follow = {"c": None, "s": None}

    def _follow_cam():
        pos = robot.data.root_pos_w
        c = pos.mean(dim=0)
        s = float((pos.max(dim=0).values - pos.min(dim=0).values)[:2].max().item())
        s = min(s, _MAX_VIEW_SPAN)
        if _follow["c"] is None:
            _follow["c"], _follow["s"] = c.clone(), s
        else:
            _follow["c"].mul_(1 - _EMA).add_(c, alpha=_EMA)
            _follow["s"] = (1 - _EMA) * _follow["s"] + _EMA * s
        cs, ss = _follow["c"], _follow["s"]
        dd = ss * 0.65 + 2.0
        eye = cs + torch.tensor([0.0, -dd, dd * 0.85 + 1.5], device=device)
        tgt = cs + torch.tensor([0.0, 0.0, 0.15], device=device)
        pano_cam.set_world_poses_from_view(
            eyes=eye.unsqueeze(0).repeat(n, 1), targets=tgt.unsqueeze(0).repeat(n, 1)
        )

    # "sub" holds the update subscription: dropping the handle would let it be collected and the
    # capture callback would stop firing.
    shared = {"pano": None, "sub": None}
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
            if robot is not None:
                _follow_cam()
            shared["pano"] = _grab(pano_cam)
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
                .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=False))
                .to_jwt()
            )
            r = rtc.Room()
            await r.connect(url, token)
            src_p = rtc.VideoSource(PANO_W, PANO_H)
            await r.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("pano", src_p),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
            print(f"[livekit] publishing training video to room '{room}'", flush=True)
            while True:
                if shared["pano"] is not None:
                    src_p.capture_frame(rtc.VideoFrame(PANO_W, PANO_H, rtc.VideoBufferType.RGBA, shared["pano"].tobytes()))
                await asyncio.sleep(period)

        try:
            loop.run_until_complete(run())
        except Exception as e:  # noqa: BLE001
            print("[livekit] ERROR:", e, flush=True)

    threading.Thread(target=_lk_thread, daemon=True).start()
