# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the live Isaac training view to LiveKit as one track: a panorama looking down on the
whole env grid.

There was also a closeup track, framed by a fixed world-space pose. That pose was chosen for one
arm task, so on any other robot or terrain it pointed at empty space -- a second video pane showing
nothing is worse than no second pane. The panorama derives its framing from the env grid, so it
adapts on its own.

- Frames come from the env's own single-viewport render() -- the same machinery --video uses --
  not from per-env Camera sensors. A per-env camera was the original design, and it broke on any
  task whose scene has no other USD sensors: physics-replication cloning does not copy the camera
  prim into env_1..N, leaving a 1-instance sensor that the scene reset then indexes with N env
  ids (CUDA index-out-of-bounds at startup). One viewport also removes the "small --num_envs
  only" restriction that per-env cameras imposed.
- Frames are captured on the training thread, right after each env.step, by wrapping the
  wrapper's step() -- the same place RecordVideo captures from. A kit update-event callback was
  tried first and silently produced no frames: the headless training loop does not render per
  step, and triggering a render from inside the update callback does not work.
- The LiveKit thread only reads CPU frames and publishes the video track "pano".
- Called from train.py --stream. Requires the env to be created with render_mode="rgb_array".
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time

import numpy as np
import torch

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

# LIVEKIT_KEYS is LiveKit's own credential format -- "api_key: api_secret", one pair per line --
# and it is what the server itself reads from its Secret. Accepting it here lets the publisher
# mount that same Secret rather than a second copy in split-variable form. Two copies of one
# credential are free to drift, and the failure mode is a connection that authenticates and then
# fails, which looks nothing like a configuration mistake. The split variables win when set, so
# nothing that worked before changes.
if not DEFAULT_KEY:
    _keys = os.environ.get("LIVEKIT_KEYS", "").strip()
    if _keys:
        _key, _, _secret = _keys.splitlines()[0].partition(":")
        DEFAULT_KEY, DEFAULT_SECRET = _key.strip(), _secret.strip()


def default_room(task: str | None = None) -> str:
    """One room per run, so two trainings at the same time do not land in each other's video.

    The deployment can only contribute JOB_NAME, which identifies the pod -- and a single-replica
    StatefulSet has one pod, so two runs started by hand inside it would collide. The task and the
    start time are known only here, and they are also what makes the room recognisable in a viewer's
    list: "Velocity-Rough-G1 at 14:32" rather than a pod name repeated three times.
    """
    if os.environ.get("LIVEKIT_ROOM"):
        return os.environ["LIVEKIT_ROOM"]
    parts = ["train"]
    if task:
        # Isaac-Velocity-Rough-G1-v0 -> Velocity-Rough-G1: the prefix and version are on every task
        # and carry nothing that tells two rooms apart.
        short = re.sub(r"^Isaac-|-v\d+$", "", task)
        parts.append(re.sub(r"[^A-Za-z0-9]+", "-", short).strip("-"))
    job = os.environ.get("JOB_NAME", "").strip()
    if job:
        parts.append(job)
    parts.append(time.strftime("%H%M%S"))
    # Room names travel in URLs and tokens; keep them within a sane length.
    return "-".join(p for p in parts if p)[:64]


DEFAULT_ROOM = default_room()

PANO_W, PANO_H = 960, 540  # panorama resolution


def add_stream_camera(env_cfg):
    """Prepare the env config for streaming: set the viewer resolution the render product will
    use, and disable the debug markers so they do not litter the picture.

    Deliberately does NOT add a camera to the scene. The stream is one view, so it uses the
    env's built-in viewer camera through render(); see the module docstring for why per-env
    scene cameras were removed.
    """
    for grp in ("commands", "scene"):
        obj = getattr(env_cfg, grp, None)
        if obj is not None:
            for term in vars(obj).values():
                if hasattr(term, "debug_vis"):
                    term.debug_vis = False
    env_cfg.viewer.resolution = (PANO_W, PANO_H)


def start_publisher(
    env,
    room: str | None = None,
    url: str = DEFAULT_URL,
    key: str = DEFAULT_KEY,
    secret: str = DEFAULT_SECRET,
    fps: int = 15,
    task: str | None = None,
):
    # Resolved here rather than as a default argument: the task is only known at the call site, and
    # a default would have frozen the start time at import.
    room = room or default_room(task)

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

    # The frame source is env.render(), which only produces pixels when the env was created with
    # render_mode="rgb_array" (train.py sets this for --stream). Anything else means a wiring
    # mistake; say so in one readable line instead of failing later with something cryptic.
    if getattr(unwrapped, "render_mode", None) != "rgb_array":
        print(
            "[livekit] env render_mode is not 'rgb_array', so no video will be published."
            " Training continues normally.",
            flush=True,
        )
        return

    def _look(eye, target):
        """Point the viewer camera. The controller exists whenever rendering does; keep a direct
        fallback for the odd configuration where it does not."""
        ctrl = getattr(unwrapped, "viewport_camera_controller", None)
        if ctrl is not None:
            ctrl.update_view_location(eye=[float(x) for x in eye], lookat=[float(x) for x in target])
        else:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=[float(x) for x in eye],
                target=[float(x) for x in target],
                camera_prim_path=unwrapped.cfg.viewer.cam_prim_path,
            )

    # Initial pose from the env grid, so there is a sensible view before the first frame.
    origins = unwrapped.scene.env_origins.float()
    center = origins.mean(dim=0)
    span = float((origins.max(dim=0).values - origins.min(dim=0).values).max().item())
    d = span * 0.65 + 2.0
    pano_eye = (center + torch.tensor([0.0, -d, d * 0.85 + 1.5], device=device)).tolist()
    pano_target = (center + torch.tensor([0.0, 0.0, 0.15], device=device)).tolist()
    try:
        _look(pano_eye, pano_target)
        print(f"[livekit] camera ready, panorama eye={[round(x, 2) for x in pano_eye]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("[livekit] failed to set camera pose:", e, flush=True)

    # Then follow the robots. Framing the origin grid only works when the grid is compact; on a
    # generated terrain the origins span the whole map and the robots end up sub-pixel. So track
    # the centroid of the actual robot positions, cap how much area the view tries to cover, and
    # smooth the motion so resets do not yank the camera. Scenes without a "robot" articulation
    # keep the static grid framing.
    robot = unwrapped.scene.articulations.get("robot")
    _MAX_VIEW_SPAN = 18.0  # metres of robot spread the view will try to contain
    _EMA = 0.05  # per-tick smoothing; ~1.3 s to settle at 15 fps
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
        _look(eye.tolist(), tgt.tolist())

    shared = {"pano": None, "frames": 0}
    period = 1.0 / fps
    last = [0.0]

    _size_warned = [False]

    def _grab():
        rgb = unwrapped.render()
        if rgb is None:
            return None
        rgb = np.asarray(rgb)
        if rgb.shape[:2] != (PANO_H, PANO_W):
            # The render product was created at viewer.resolution; a mismatch means someone else
            # overrode it after add_stream_camera. Publishing a wrong-sized buffer would garble
            # the video, so warn once and skip.
            if not _size_warned[0]:
                print(f"[livekit] unexpected frame size {rgb.shape[:2]}, expected {(PANO_H, PANO_W)}", flush=True)
                _size_warned[0] = True
            return None
        if rgb.shape[-1] == 3:
            rgb = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        return np.ascontiguousarray(rgb.astype(np.uint8))

    _err_once = [False]

    def _capture():
        now = time.time()
        if now - last[0] < period:
            return
        try:
            if robot is not None:
                _follow_cam()
            frame = _grab()
            if frame is not None:
                if shared["frames"] == 0:
                    print(f"[livekit] first frame captured {frame.shape[1]}x{frame.shape[0]}", flush=True)
                shared["pano"] = frame
                shared["frames"] += 1
            last[0] = now
        except Exception as e:  # noqa: BLE001
            # Losing the stream must not lose the run -- but losing it silently must not
            # happen either. Say what broke, once.
            if not _err_once[0]:
                print("[livekit] frame capture failed:", repr(e), flush=True)
                _err_once[0] = True

    # Capture from the training thread, after each step: wrap the outermost env's step so the
    # render happens where RecordVideo's does -- on the main thread, after physics advanced.
    _orig_step = env.step

    def _step_and_capture(*args, **kwargs):
        out = _orig_step(*args, **kwargs)
        _capture()
        return out

    env.step = _step_and_capture

    def _lk_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():
            token = (
                api.AccessToken(key, secret)
                .with_identity("isaac-train")
                .with_name("Isaac Training")
                .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=False))
                .to_jwt()
            )
            r = rtc.Room()
            await r.connect(url, token)
            src_p = rtc.VideoSource(PANO_W, PANO_H)
            await r.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("pano", src_p),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"[livekit] publishing training video to room '{room}'", flush=True)
            while True:
                if shared["pano"] is not None:
                    src_p.capture_frame(
                        rtc.VideoFrame(PANO_W, PANO_H, rtc.VideoBufferType.RGBA, shared["pano"].tobytes())
                    )
                await asyncio.sleep(period)

        try:
            loop.run_until_complete(run())
        except Exception as e:  # noqa: BLE001
            print("[livekit] ERROR:", e, flush=True)

    threading.Thread(target=_lk_thread, daemon=True).start()
