"""Learning loop for MultiRemoteVecEnv, bound onto an rsl_rl OnPolicyRunner (PPO) or OffPolicyRunner (SAC).

- Shard-asynchronous collection (PPO and SAC): every env shard runs its rollout on its own thread with its own copy
  of the policy, so no shard waits for the slowest one at each step. The policy, including its observation
  normalizer, is frozen for the rollout; the rollout is then replayed step by step through the algorithm's own
  act / process_env_step with the executed actions. Normalizer updates are applied once per rollout instead of
  once per step, which is the only difference from the stock loop.
- Collection overlapped with the update (SAC only): while the update runs, the next rollout is collected with a copy
  of the pre-update policy, so the behavior policy lags one iteration (fine for off-policy learning).
"""
import copy
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


class _Policies:
    """Sampling copies of the actor, one per shard thread on CPU (a single locked copy otherwise).

    The copies are made once, before any update, and refreshed from the live actor with load_state_dict; refreshing
    before an overlapped update makes them the pre-update snapshot the next rollout is collected with.
    """

    def __init__(self, actor, n, device):
        self.device = device
        if device == "cpu":
            base = copy.deepcopy(actor).to("cpu").eval()
            self.copies, self.lock = [copy.deepcopy(base) for _ in range(n)], None
        else:
            self.copies, self.lock = [copy.deepcopy(actor).eval()] * n, threading.Lock()

    def refresh(self, actor):
        state = actor.state_dict()
        for p in {id(c): c for c in self.copies}.values():
            p.load_state_dict(state)

    def act(self, i, obs):
        with torch.inference_mode():
            if self.lock is None:
                return self.copies[i](obs, stochastic_output=True)
            with self.lock:
                return self.copies[i](obs.to(self.device), stochastic_output=True).cpu()


def _collect(env, policies, obs, steps, asynchronous):
    """Roll the frozen policy for `steps` steps; returns per-step full-batch records and the last observation."""
    records = []
    if asynchronous:
        def run(shard, arg):
            i, o = arg
            recs = []
            for _ in range(steps):
                a = policies.act(i, o)
                nxt, rew, done, extras = shard.step(a)
                recs.append((o, a, nxt, rew, done, extras))
                o = nxt
            return recs

        per_shard = env.map(run, list(enumerate(obs.split(env.sizes))))
        for t in range(steps):
            rows = [ps[t] for ps in per_shard]
            records.append(tuple(torch.cat([r[i] for r in rows]) for i in range(5))
                           + (env.merge_extras([r[5] for r in rows]),))
    else:
        for _ in range(steps):
            a = policies.act(0, obs)
            nxt, rew, done, extras = env.step(a)
            records.append((obs, a, nxt, rew, done, extras))
            obs = nxt
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
    """Drop-in for runner.learn; reads self.async_collect, self.overlap and self.collect_device."""
    off_policy = hasattr(self.alg, "replay_buffer")
    overlap = off_policy and self.overlap
    steps = self.cfg["num_steps_per_env"]
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs = self.env.get_observations()
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    policies = _Policies(self.alg.actor, len(self.env.shards), self.collect_device)
    pending = None
    for it in range(start_it, total_it):
        t0 = time.time()
        if pending is None:
            policies.refresh(self.alg.actor)
            pending = _collect(self.env, policies, obs, steps, self.async_collect)
        records, obs = pending
        pending = None
        _replay(self, records, obs, off_policy)
        t1 = time.time()
        do_update = not off_policy or it >= self.start_training
        if overlap and it < total_it - 1:
            policies.refresh(self.alg.actor)  # behavior policy for the next rollout: the pre-update actor
            result = {}

            def update():
                try:
                    result["loss"] = self.alg.update() if do_update else {}
                except BaseException as e:  # re-raised on the main thread
                    result["error"] = e

            th = threading.Thread(target=update)
            th.start()
            pending = _collect(self.env, policies, obs, steps, self.async_collect)
            th.join()
            if "error" in result:
                raise result["error"]
            loss_dict = result["loss"]
        else:
            loss_dict = self.alg.update() if do_update else {}
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
