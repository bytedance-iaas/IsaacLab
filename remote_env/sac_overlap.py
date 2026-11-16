"""SAC learning loop for remote environments that spreads each iteration's gradient steps over its environment steps,
bound onto an rsl_rl OffPolicyRunner (`learn_interleaved`).

The stock loop collects `num_steps_per_env` steps and then does `num_mini_batches` gradient steps while the simulator
waits. Here a worker thread inserts each step's transition into the replay buffer as soon as it arrives and does that
step's share of the gradient steps while the simulator computes the next step. The rollout is acted by `policy`, a copy
of the actor on its own device stream that takes the newest finished snapshot of the actor's parameters before each
step (see `_Snapshots`). Its observation normalizer receives the same per-step updates as the live actor would.
"""

import copy
import os
import queue
import threading
import time

import torch
from tensordict import TensorDict


def _true_next_obs(next_obs, extras, device):
    """The next observation SAC.process_env_step stores (final observations for timed-out envs), on `device`."""
    if "time_outs" in extras and "time_outs_obs" in extras:
        mask = extras["time_outs"].to(device).bool().squeeze(-1)
        final = extras["time_outs_obs"]
        return TensorDict(
            {k: torch.where(mask[:, None], final[k].to(device), next_obs[k]) for k in final.keys()},
            batch_size=next_obs.batch_size,
        )
    return next_obs


class _Snapshots:
    """Actor parameters handed from the update thread to the acting copy through two device buffers.

    The update thread writes a slot on its stream and records that slot's event; the reader copies from a slot on its
    own stream only once the event has completed, and records a read event the writer waits for (on the device) before
    writing that slot again. A slot is unlisted while it is being rewritten, all under one lock."""

    def __init__(self, src: torch.nn.Module, dst: torch.nn.Module, dev, every: int):
        self.src, self.dst = list(src.parameters()), list(dst.parameters())
        self.dev, self.every = dev, every
        with torch.no_grad():
            self.bufs = [[p.detach().clone() for p in self.src] for _ in range(2)]
        self.written = [dev.Event() for _ in range(2)]
        self.read = [dev.Event() for _ in range(2)]
        for e in self.read:
            e.record()
        self.lock = threading.Lock()
        self.listed = []  # (version, slot, actor steps), oldest first
        self.version = self.applied = self.steps = 0
        self.applied_steps = 0  # actor steps of the snapshot the copy holds (kept by the caller)

    def on_step(self, *_):
        """Optimizer post-step hook, on the update thread."""
        self.steps += 1
        if self.steps % self.every == 0:
            self.publish()

    def publish(self):
        with self.lock:
            slot = 0 if not self.listed else 1 - self.listed[-1][1]
            self.listed = [x for x in self.listed if x[1] != slot]
            self.dev.current_stream().wait_event(self.read[slot])
        with torch.no_grad():
            for b, p in zip(self.bufs[slot], self.src):
                b.copy_(p)
        self.written[slot].record()
        with self.lock:
            self.version += 1
            self.listed.append((self.version, slot, self.steps))

    def apply(self, stream) -> int | None:
        """On the collection thread: copy the newest finished snapshot into the copy; returns its actor-step count."""
        with self.lock:
            for version, slot, steps in reversed(self.listed):
                if version <= self.applied:
                    return None
                if self.written[slot].query():
                    stream.wait_event(self.written[slot])
                    with self.dev.stream(stream), torch.no_grad():
                        for d, b in zip(self.dst, self.bufs[slot]):
                            d.copy_(b)
                    self.read[slot].record(stream)
                    self.applied = version
                    return steps
        return None


def learn_interleaved(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False, live_every: int = 4
) -> None:
    """Drop-in for OffPolicyRunner.learn that spreads each iteration's update over the environment steps.

    The stock loop does `num_mini_batches` gradient steps after every rollout of `num_steps_per_env` steps; here a
    worker thread inserts each step's transition as soon as it arrives and then does that step's share of the gradient
    steps (the canonical SAC loop at the same update-to-data ratio), while the calling thread keeps stepping the
    environment with the newest finished parameter snapshot (see `_Snapshots`). Unlike the stock loop, gradient steps
    see the rollout in progress; the number of gradient steps per iteration, the mini-batch size and every other
    setting are unchanged. The worker drains at the end of each iteration, so logging and checkpoints see the same
    state as in the stock loop and the calling thread is never more than one rollout ahead.
    """
    alg, steps = self.alg, self.cfg["num_steps_per_env"]
    per_iter = alg.num_mini_batches
    quotas = [(s + 1) * per_iter // steps - s * per_iter // steps for s in range(steps)]
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )
    obs = self.env.get_observations().to(self.device)
    alg.train_mode()
    self.logger.init_logging_writer()
    dev = getattr(torch, torch.device(self.device).type)
    stream = dev.Stream()
    policy = copy.deepcopy(alg.actor)
    for k in [k for k, v in vars(policy).items() if getattr(v, "__wrapped_ptime__", False)]:
        delattr(policy, k)  # instance-level wrappers (ptime) would call the live actor
    snaps = _Snapshots(alg.actor, policy, dev, live_every)
    alg.actor_optimizer.register_step_post_hook(snaps.on_step)
    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations

    work, errors = queue.Queue(), []
    acc = {"loss": {}, "n": 0}

    def worker():
        while True:
            item = work.get()
            try:
                if item is None:
                    return
                if errors:
                    continue
                ready, o, a, nxt, rew, done, extras, quota = item
                cur = dev.current_stream()
                cur.wait_event(ready)
                # made on the collection stream and used here: keep the allocator from reusing them early
                for t in [*o.values(), a, *nxt.values(), rew, done]:
                    t.record_stream(cur)
                with torch.inference_mode():
                    alg.transition.observations = o
                    alg.transition.actions = a
                    alg.process_env_step(nxt, rew, done, extras)
                    self.logger.process_env_step(rew, done, extras, None)
                if quota:
                    alg.num_mini_batches = quota
                    try:
                        ld = alg.update()
                    finally:
                        alg.num_mini_batches = per_iter
                    for key, v in ld.items():
                        acc["loss"][key] = acc["loss"].get(key, 0.0) + v * quota
                    acc["n"] += quota
                    snaps.publish()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            finally:
                work.task_done()

    th = threading.Thread(target=worker, name="upd", daemon=True)
    th.start()

    def refresh():
        with torch.inference_mode():  # the copy's normalizer buffers were written in inference mode
            policy.load_state_dict(alg.actor.state_dict())

    log_collect = log_learn = 0.0
    log_iters = 0
    log_lag = []
    try:
        for it in range(start_iter, tot_iter):
            start = time.time()
            train = it >= self.start_training
            stream.wait_stream(dev.current_stream())  # the refresh of `policy` is queued on the current stream
            lag = []
            with torch.inference_mode():
                for s in range(steps):
                    k = snaps.apply(stream)
                    if train:
                        lag.append(snaps.steps - (k if k is not None else snaps.applied_steps))
                    if k is not None:
                        snaps.applied_steps = k
                    with dev.stream(stream):
                        actions = policy(obs, stochastic_output=True)
                        actions_env = actions.to(self.env.device)
                    next_obs, rewards, dones, extras = self.env.step(actions_env)
                    with dev.stream(stream):
                        next_obs, rewards, dones = (
                            next_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                        policy.update_normalization(_true_next_obs(next_obs, extras, self.device))
                        ready = dev.Event()
                        ready.record(stream)
                    work.put((ready, obs, actions, next_obs, rewards, dones, extras, quotas[s] if train else 0))
                    obs = next_obs
            collection_time = time.time() - start
            t0 = time.time()
            work.join()
            if errors:
                raise errors[0]
            learn_time = time.time() - t0  # the part of the update not hidden behind the rollout
            loss_dict = {key: v / acc["n"] for key, v in acc["loss"].items()} if acc["n"] else {}
            acc["loss"], acc["n"] = {}, 0
            refresh()
            if lag:
                log_lag.append(sum(lag) / len(lag))
            self.current_learning_iteration = it

            log_collect += collection_time
            log_learn += learn_time
            log_iters += 1
            if (it % self.log_interval == 0) or (it == tot_iter - 1):
                self.logger.log(
                    it=it,
                    start_it=start_iter,
                    total_it=tot_iter,
                    collect_time=log_collect,
                    learn_time=log_learn,
                    loss_dict=loss_dict,
                    learning_rate=alg.actor_learning_rate,
                    action_std=getattr(alg.get_policy(), "output_std", policy.output_std),
                    rnd_weight=None,
                    alpha=getattr(alg, "alpha", None),
                    collection_size_override=steps * self.env.num_envs * log_iters,
                )
                log_collect = log_learn = 0.0
                log_iters = 0
                if log_lag:
                    print(
                        f"[interleave] it {it}: acting weights {sum(log_lag) / len(log_lag):.1f} actor steps "
                        f"behind the worker ({per_iter} per iteration)",
                        flush=True,
                    )
                    log_lag = []
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0 and it != 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    finally:
        work.put(None)
        th.join()
    if self.logger.writer is not None:
        self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()
