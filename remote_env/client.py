"""rsl_rl VecEnv that forwards to a remote Isaac Lab env served by server.py."""
import grpc
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from types import SimpleNamespace

import wire


def _robot_view(ts, action_scale):
    """Just enough of env.unwrapped (scene['robot'] and the action scale) for SAC's action scaling."""
    robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=ts["robot/soft_limits"].unsqueeze(0),
                                                 default_joint_pos=ts["robot/default_pos"].unsqueeze(0)))
    term = SimpleNamespace(cfg=SimpleNamespace(scale=action_scale))
    return SimpleNamespace(scene={"robot": robot}, action_manager=SimpleNamespace(_terms={"action": term}))


class RemoteVecEnv(VecEnv):
    def __init__(self, target: str, token: str, authority: str | None = None, timeout: float = 600.0):
        opts = list(wire.CHANNEL_OPTIONS)
        if authority:
            opts.append(("grpc.default_authority", authority))
        self._ch = grpc.insecure_channel(target, options=opts)
        self._md = (("x-token", token),)
        self._timeout = timeout
        self._rpc = {
            m: self._ch.unary_unary(f"/{wire.SERVICE}/{m}", request_serializer=None, response_deserializer=None)
            for m in wire.METHODS
        }
        robot_ts, info = self._call("Init")
        self.unwrapped = _robot_view(robot_ts, info["action_scale"])
        self.num_envs = info["num_envs"]
        self.num_actions = info["num_actions"]
        self.max_episode_length = info["max_episode_length"]
        self.cfg = info
        self.device = "cpu"
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.last_rpc_s = 0.0

    def _call(self, method, tensors=None, meta=None):
        import time

        t0 = time.perf_counter()
        resp = self._rpc[method](wire.pack(tensors, meta), metadata=self._md, timeout=self._timeout, wait_for_ready=True)
        self.last_rpc_s = time.perf_counter() - t0
        return wire.unpack(resp)

    def _obs(self, ts):
        return TensorDict({k[4:]: v for k, v in ts.items() if k.startswith("obs/")}, batch_size=[self.num_envs])

    @property
    def episode_length_buf(self):
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self._episode_length_buf = v
        self._call("SetEpisodeLen", {"episode_length_buf": v})

    def reset(self):
        ts, meta = self._call("Reset")
        return self._obs(ts), {"log": meta.get("log", {})}

    def get_observations(self):
        ts, _ = self._call("GetObs")
        return self._obs(ts)

    def step(self, actions):
        ts, meta = self._call("Step", {"actions": actions.detach().float()})
        extras = {"log": meta.get("log", {})}
        if "time_outs" in ts:
            extras["time_outs"] = ts["time_outs"]
        return self._obs(ts), ts["rew"], ts["dones"], extras

    def close(self):
        try:
            self._call("Close")
        except grpc.RpcError:
            pass
        self._ch.close()
