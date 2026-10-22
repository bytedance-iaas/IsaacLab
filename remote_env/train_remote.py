"""PPO or SAC (--agent) with remote Isaac Lab 3.0 env servers (server.py). Mirrors the stock rsl_rl train entrypoint's
runner setup. --target takes one address or a comma-separated list of env shards (e.g. several CPU servers on this host)."""
import argparse, importlib.metadata as metadata, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp
wp.config.enable_backward = False
import torch

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-G1")
p.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="rsl_rl_cfg_entry_point (PPO) or rsl_rl_sac_cfg_entry_point (SAC)")
p.add_argument("--target", required=True, help="host:port[,host:port...]")
p.add_argument("--async_collect", type=int, default=1, help="1: shards step without a per-step barrier")
p.add_argument("--overlap", type=int, default=1, help="1: SAC collects the next rollout while updating (ignored for PPO)")
p.add_argument("--collect_device", default=None, help="policy device for collection; default cpu with >1 shard, else --device")
p.add_argument("--authority", default=None)
p.add_argument("--iters", type=int, default=1500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="cuda:0")
p.add_argument("--run_name", default="remote")
p.add_argument("--resume", default=None, help="checkpoint to continue from (PPO; SAC restarts with an empty replay buffer)")
args = p.parse_args()

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
sys.argv = sys.argv[:1]
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = args.device; agent_cfg.max_iterations = args.iters; agent_cfg.seed = args.seed
import rsl_rl.runners
import types
from multi_env import MultiRemoteVecEnv
import async_runner
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
env = MultiRemoteVecEnv(args.target.split(","), os.environ["ENV_TOKEN"], authority=args.authority)
print("connected:", env.cfg, "agent:", agent_cfg.class_name, flush=True)
log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
runner = getattr(rsl_rl.runners, agent_cfg.class_name)(env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
if args.async_collect or (args.overlap and agent_cfg.class_name == "OffPolicyRunner"):
    if args.collect_device == "cpu" or (args.collect_device is None and len(env.shards) > 1):
        torch.set_num_threads(1)  # many shard threads each run a small CPU forward
    runner.async_collect, runner.overlap = bool(args.async_collect), bool(args.overlap)
    runner.collect_device = args.collect_device or ("cpu" if len(env.shards) > 1 else args.device)
    runner.start_training = getattr(runner, "start_training", 0)
    runner.learn = types.MethodType(async_runner.learn, runner)
start_it = 0
if args.resume:
    runner.load(args.resume); start_it = runner.current_learning_iteration
    print("resumed at iteration", start_it, flush=True)
t0 = time.time()
runner.learn(num_learning_iterations=args.iters - start_it, init_at_random_ep_len=True)
print(f"TRAIN_DONE total {time.time() - t0:.1f}s for {args.iters} iters", flush=True)
env.close()
