"""Learning loop for MultiRemoteVecEnv, bound onto an rsl_rl OnPolicyRunner (PPO) or OffPolicyRunner (SAC).

Shard-asynchronous collection: every env shard runs its rollout on its own thread, calling the learner's actor under
a lock, so no shard waits for the slowest one at each step. The actor, including its observation normalizer, does
not change during the rollout; the rollout is then replayed step by step through the algorithm's own
act / process_env_step with the executed actions. For PPO, normalizer updates are applied once per rollout instead
of once per step, which is the only difference from the stock loop.

learn_overlap (PPO) also overlaps collection with the update: while rollout k is collected with a frozen copy of the
actor (on its own device stream), the update for rollout k-1 runs on a background thread. Rollout k enters the rollout
storage only after that update, so rollout k is collected by the actor from before update k-1, one version behind the
actor that trains on it. This uses the decoupled PPO loss of RLinf's async PPO (staleness 1, explicitly recomputed
proximal policy): the replay recomputes log-probs, values and distribution parameters with the actor about to train
(the proximal policy), so clipping and the adaptive-LR KL are taken around it; each sample's advantage is then scaled
by the behavior weight pi_prox / pi_behav, and samples whose weight exceeds behav_w_max are dropped (the rest rescaled
to keep the mean over kept samples). RLinf's optional dual clip is not applied. SAC overlap is in sac_overlap.py.
"""

import contextlib
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


def _collect(env, actor, device, obs, steps, stream=None, log_prob=False):
    """Roll the actor for `steps` steps on every shard; returns per-step full-batch records and the last observation.
    With `stream`, the actor runs on that device stream instead of the current one; with `log_prob`, each record
    also carries the behavior log-prob of the executed action."""
    lock = threading.Lock()
    # An actor with per-environment noise state (FlashSAC) is told which environments a shard holds
    stateful = getattr(actor, "stateful_noise", False)
    starts = [sum(env.sizes[:i]) for i in range(len(env.sizes))]

    def on_stream():
        return _device_module(device).stream(stream) if stream is not None else contextlib.nullcontext()

    def run(shard, arg):
        o, start, size = arg
        kw = {"env_ids": slice(start, start + size)} if stateful else {}
        recs = []
        for _ in range(steps):
            with torch.inference_mode(), lock, on_stream():
                a = actor(o.to(device), stochastic_output=True, **kw)
                lp = (actor.get_output_log_prob(a).cpu(),) if log_prob else ()
                a = a.cpu()
            nxt, rew, done, extras = shard.step(a)
            recs.append((o, a, nxt, rew, done, extras) + lp)
            o = nxt
        return recs

    per_shard = env.map(run, list(zip(obs.split(env.sizes), starts, env.sizes)))
    records = []
    for t in range(steps):
        rows = [ps[t] for ps in per_shard]
        cat = lambda i: torch.cat([r[i] for r in rows])  # noqa: E731
        records.append(
            tuple(cat(i) for i in range(5))
            + (env.merge_extras([r[5] for r in rows]),)
            + tuple(cat(i) for i in range(6, len(rows[0])))
        )
    return records, records[-1][2]


def _replay(runner, records, last_obs, off_policy):
    """Feed a collected rollout through the algorithm as if it had been collected step by step."""
    alg, dev = runner.alg, runner.device

    def feed():
        for o, a, nxt, rew, done, extras, *_ in records:
            if getattr(alg.actor, "stateful_noise", False):
                alg.transition.observations = o.to(dev)  # acting again would advance the noise state
            else:
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


def _device_module(device):
    """torch.cuda or torch.npu for a device string."""
    return getattr(torch, torch.device(device).type)


def _frozen_copy(model):
    """Deep copy for inference; drops instance-level method wrappers (e.g. ptime) that would call the original."""
    snap = copy.deepcopy(model)
    for k in [k for k, v in vars(snap).items() if getattr(v, "__wrapped_ptime__", False)]:
        delattr(snap, k)
    snap.eval()
    return snap


def _behavior_weights(alg, records, w_max):
    """Decoupled PPO: scale the stored advantages by pi_prox / pi_behav, dropping samples above w_max.
    The storage already holds the proximal log-probs (recomputed in _replay); the records hold the behavior ones."""
    st = alg.storage
    with torch.inference_mode():  # the storage tensors were created in inference mode
        behav = torch.stack([r[6] for r in records]).to(st.actions_log_prob.device).view_as(st.actions_log_prob)
        w = torch.exp(st.actions_log_prob - behav)
        keep = w <= w_max
        n_keep = int(keep.sum())
        st.advantages *= torch.where(keep, w, torch.zeros_like(w)) * (keep.numel() / max(n_keep, 1))
    return {"behav_weight": float(w.mean()), "behav_dropped": 1.0 - n_keep / keep.numel()}


def learn_overlap(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False, behav_w_max: float = 2.0
) -> None:
    """Drop-in for OnPolicyRunner.learn: collection of rollout k overlaps the update for rollout k-1 (see module
    docstring)."""
    steps = self.cfg["num_steps_per_env"]
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )
    obs = self.env.get_observations()
    self.alg.train_mode()
    self.logger.init_logging_writer()
    dev = _device_module(self.device)
    stream = dev.Stream()
    snap = _frozen_copy(self.alg.actor)
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations

    def update_async():
        out = {}

        def run():
            try:
                t0 = time.time()
                out["loss"] = self.alg.update()
                out["dt"] = time.time() - t0
            except BaseException as e:  # noqa: BLE001
                out["err"] = e

        th = threading.Thread(target=run, name="upd")
        th.start()
        return th, out

    behav = {}
    for it in range(start_it, total_it):
        t0 = time.time()
        # the copy into snap is queued on the current stream; the collection stream waits for it, not for the update
        stream.wait_stream(dev.current_stream())
        # the update for rollout it-1 (the stock loop runs it at iteration it-1)
        upd = update_async() if it - 1 >= start_it else None
        records, obs = _collect(self.env, snap, self.device, obs, steps, stream=stream, log_prob=True)
        t1 = time.time()
        loss_dict, learn_time = {}, 0.0
        if upd is not None:
            th, out = upd
            th.join()
            if "err" in out:
                raise out["err"]
            loss_dict, learn_time = dict(out["loss"], **behav), out["dt"]
        _replay(self, records, obs, off_policy=False)
        behav = _behavior_weights(self.alg, records, behav_w_max)
        # the next rollout is collected by the actor as of now (after update it-1); update it runs alongside it
        snap.load_state_dict(self.alg.actor.state_dict())
        self.current_learning_iteration = it
        self.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=t1 - t0,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=self.alg.learning_rate,
            action_std=self.alg.get_policy().output_std,
            rnd_weight=None,
        )
        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
            self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    # the stock loop's last update (for the last rollout) has no collection left to overlap with
    if total_it - 1 >= start_it:
        self.alg.update()
    if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()


def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Drop-in for runner.learn."""
    off_policy = hasattr(self.alg, "replay_buffer")
    steps = self.cfg["num_steps_per_env"]
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )
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
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=t1 - t0,
                learn_time=t2 - t1,
                loss_dict=loss_dict,
                learning_rate=self.alg.actor_learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
                alpha=getattr(self.alg, "alpha", None),
                collection_size_override=steps * self.env.num_envs,
            )
        else:
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=t1 - t0,
                learn_time=t2 - t1,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0 and (it != 0 or not off_policy):
            self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()
