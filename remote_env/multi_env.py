"""One rsl_rl VecEnv over one or more remote env servers (env shards); shards are stepped in parallel."""

from concurrent.futures import ThreadPoolExecutor

import torch
from client import RemoteVecEnv
from rsl_rl.env import VecEnv


class MultiRemoteVecEnv(VecEnv):
    def __init__(self, targets: list[str], token: str, authority: str | None = None, timeout: float = 600.0):
        self.shards = [RemoteVecEnv(t, token, authority=authority, timeout=timeout) for t in targets]
        self.sizes = [s.num_envs for s in self.shards]
        self.num_envs = sum(self.sizes)
        self.num_actions = self.shards[0].num_actions
        self.max_episode_length = self.shards[0].max_episode_length
        self.cfg = dict(self.shards[0].cfg, num_envs=self.num_envs, shards=len(self.shards))
        self.unwrapped = self.shards[0].unwrapped
        self.device = "cpu"
        self._pool = ThreadPoolExecutor(len(self.shards))
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def map(self, fn, args=None):
        """fn(shard, arg) on every shard concurrently; results in shard order."""
        return list(self._pool.map(fn, self.shards, args if args is not None else [None] * len(self.shards)))

    @staticmethod
    def merge_extras(extras: list[dict]) -> dict:
        logs = [e["log"] for e in extras]
        keys = set().union(*logs)
        out = {"log": {k: sum(lg[k] for lg in logs if k in lg) / sum(1 for lg in logs if k in lg) for k in keys}}
        if all("time_outs" in e for e in extras):
            out["time_outs"] = torch.cat([e["time_outs"] for e in extras])
        return out

    @property
    def episode_length_buf(self):
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self._episode_length_buf = v

        def set_len(shard, part):
            shard.episode_length_buf = part

        self.map(set_len, list(v.split(self.sizes)))

    def reset(self):
        res = self.map(lambda s, _: s.reset())
        return torch.cat([r[0] for r in res]), self.merge_extras([r[1] for r in res])

    def get_observations(self):
        return torch.cat(self.map(lambda s, _: s.get_observations()))

    def step(self, actions):
        res = self.map(lambda s, a: s.step(a), list(actions.detach().float().cpu().split(self.sizes)))
        return (
            torch.cat([r[0] for r in res]),
            torch.cat([r[1] for r in res]),
            torch.cat([r[2] for r in res]),
            self.merge_extras([r[3] for r in res]),
        )

    def close(self):
        self.map(lambda s, _: s.close())
