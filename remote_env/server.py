"""Isaac Lab 3.0 gRPC env server: reuses the stock train entrypoint to launch the sim and build the env,
then swaps the rsl_rl runners for an env-only stub whose learn() is a serving loop (all sim calls on the main
thread). The server is algorithm-agnostic: the same process serves PPO and SAC learners."""
import os, queue, sys, threading, time
from concurrent import futures
import warp as wp
wp.config.enable_backward = False
import grpc, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire

PORT = int(os.environ.get("ENV_PORT", "18990"))

def serve(self, *a, **k):
    env = self.env
    N = env.num_envs
    token = os.environ["ENV_TOKEN"]
    def obs_tensors(obs): return {f"obs/{k}": v for k, v in obs.items()}
    def scalar_logs(extras):
        out = {}
        for k, v in (extras.get("log") or {}).items():
            out[k] = float(v.float().mean()) if torch.is_tensor(v) else float(v)
        return out
    def handle(method, req):
        ts, meta = wire.unpack(req)
        if method == "Init":
            # SAC derives per-joint action scaling from these (rsl_rl SAC._compute_action_scaling)
            robot = env.unwrapped.scene["robot"]
            scale = 1.0
            for term in env.unwrapped.action_manager._terms.values():
                if isinstance(getattr(term.cfg, "scale", None), (float, int)): scale = float(term.cfg.scale); break
            robot_t = {"robot/soft_limits": robot.data.soft_joint_pos_limits[0].float().cpu(),
                       "robot/default_pos": robot.data.default_joint_pos[0].float().cpu()}
            return wire.pack(robot_t, {"num_envs": N, "num_actions": env.num_actions, "action_scale": scale,
                                       "max_episode_length": int(env.max_episode_length),
                                       "is_finite_horizon": bool(env.unwrapped.cfg.is_finite_horizon)})
        if method == "Reset":
            obs, extras = env.reset(); return wire.pack(obs_tensors(obs), {"log": scalar_logs(extras)})
        if method == "GetObs":
            return wire.pack(obs_tensors(env.get_observations()))
        if method == "Step":
            obs, rew, dones, extras = env.step(ts["actions"].to(env.device))
            out = obs_tensors(obs); out["rew"], out["dones"] = rew, dones
            if "time_outs" in extras: out["time_outs"] = extras["time_outs"]
            return wire.pack(out, {"log": scalar_logs(extras)})
        if method == "SetEpisodeLen":
            env.episode_length_buf = ts["episode_length_buf"].to(env.device); return wire.pack()
        if method == "Close":
            if not os.environ.get("ENV_PERSIST"): stop.set()
            return wire.pack()
        raise ValueError(method)
    work = queue.Queue(); stop = threading.Event()
    def dispatch(method, req):
        fut = futures.Future(); work.put((method, req, fut)); return fut.result()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=wire.CHANNEL_OPTIONS)
    server.add_generic_rpc_handlers((wire.make_server_handler(dispatch, token),))
    server.add_insecure_port(f"0.0.0.0:{PORT}"); server.start()
    print(f"[server] env ready num_envs={N} listening on :{PORT}", flush=True)
    nstep = 0; tbusy = 0.0
    while not stop.is_set():
        try: method, req, fut = work.get(timeout=1.0)
        except queue.Empty: continue
        t0 = time.time()
        try: fut.set_result(handle(method, req))
        except Exception as e:
            fut.set_exception(e); print(f"[server] {method} failed: {e!r}", flush=True)
        if method == "Step":
            nstep += 1; tbusy += time.time() - t0
            if nstep % 500 == 0: print(f"[server] steps={nstep} sim/step={tbusy/nstep*1e3:.1f}ms", flush=True)
    server.stop(2)
    raise SystemExit(0)

class EnvOnlyRunner:
    """Stands in for OnPolicyRunner / OffPolicyRunner: keeps the env, builds no policy or replay buffer."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env

    def add_git_repo_to_log(self, *a, **k):
        pass

    learn = serve


import rsl_rl.runners as R
R.OnPolicyRunner = R.OffPolicyRunner = EnvOnlyRunner
from isaaclab_rl.entrypoints import run_train_cli
sys.exit(run_train_cli(sys.argv[1:]))
