"""Learning loop for MultiRemoteVecEnv, bound onto an rsl_rl OnPolicyRunner (PPO) or OffPolicyRunner (SAC).

Shard-asynchronous collection: every env shard runs its rollout on its own thread, calling the learner's actor under
a lock, so no shard waits for the slowest one at each step. The actor, including its observation normalizer, does
not change during the rollout; the rollout is then replayed step by step through the algorithm's own
act / process_env_step with the executed actions. For PPO, normalizer updates are applied once per rollout instead
of once per step, which is the only difference from the stock loop.
"""
import os
import threading
import time

import torch


class _DeferredNormalization:
    """Queue update_normalization calls inside the block and apply them in order on exit."""

    def __init__(self, *models):
        self.models = [m for m in models if m is not None]
        self.pending = []

    def __enter__(self):
        for m in self.models:
            m.update_normalization = lambda obs, _m=m: self.pending.append((_m, obs))
        return self

    def __exit__(self, *exc):
        for m in self.models:
            del m.update_normalization
        for m, obs in self.pending:
            m.update_normalization(obs)


def _collect(env, actor, device, obs, steps):
    """Roll the actor for `steps` steps on every shard; returns per-step full-batch records and the last observation."""
    lock = threading.Lock()

    def run(shard, o):
        recs = []
        for _ in range(steps):
            with torch.inference_mode(), lock:
                a = actor(o.to(device), stochastic_output=True).cpu()
            nxt, rew, done, extras = shard.step(a)
            recs.append((o, a, nxt, rew, done, extras))
            o = nxt
        return recs

    per_shard = env.map(run, list(obs.split(env.sizes)))
    records = []
    for t in range(steps):
        rows = [ps[t] for ps in per_shard]
        records.append(tuple(torch.cat([r[i] for r in rows]) for i in range(5))
                       + (env.merge_extras([r[5] for r in rows]),))
    return records, records[-1][2]


def _replay(runner, records, last_obs, off_policy):
    """Feed a collected rollout through the algorithm as if it had been collected step by step."""
    alg, dev = runner.alg, runner.device

    def feed():
        for o, a, nxt, rew, done, extras in records:
            alg.act(o.to(dev))  # fills the transition (and, for PPO, values and the action distribution)
            a = a.to(dev)
            alg.transition.actions = a  # the action that was actually executed
            if not off_policy:
                alg.transition.actions_log_prob = alg.actor.get_output_log_prob(a).detach()
            rew, done = rew.to(dev), done.to(dev)
            alg.process_env_step(nxt.to(dev), rew, done, extras)
            runner.logger.process_env_step(rew, done, extras, None)

    with torch.inference_mode():
        if off_policy:
            feed()
        else:
            with _DeferredNormalization(alg.actor, alg.critic):
                feed()
            alg.compute_returns(last_obs.to(dev))


def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Drop-in for runner.learn."""
    off_policy = hasattr(self.alg, "replay_buffer")
    steps = self.cfg["num_steps_per_env"]
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs = self.env.get_observations()
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
        t0 = time.time()
        records, obs = _collect(self.env, self.alg.actor, self.device, obs, steps)
        _replay(self, records, obs, off_policy)
        t1 = time.time()
        loss_dict = self.alg.update() if not off_policy or it >= self.start_training else {}
        t2 = time.time()
        self.current_learning_iteration = it
        if off_policy:
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=t1 - t0, learn_time=t2 - t1,
                            loss_dict=loss_dict, learning_rate=self.alg.actor_learning_rate,
                            action_std=self.alg.get_policy().output_std, rnd_weight=None,
                            alpha=getattr(self.alg, "alpha", None), collection_size_override=steps * self.env.num_envs)
        else:
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=t1 - t0, learn_time=t2 - t1,
                            loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                            action_std=self.alg.get_policy().output_std, rnd_weight=None)
        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0 and (it != 0 or not off_policy):
            self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()
