"""PPO with a remote Isaac Lab 3.0 env (server_v3.py). Mirrors the stock rsl_rl train entrypoint's runner setup."""
import argparse, importlib.metadata as metadata, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp as wp
wp.config.enable_backward = False
import torch

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-G1")
p.add_argument("--target", required=True)
p.add_argument("--authority", default=None)
p.add_argument("--iters", type=int, default=1500)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="cuda:0")
p.add_argument("--run_name", default="remote")
args = p.parse_args()

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
sys.argv = sys.argv[:1]
env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = args.device; agent_cfg.max_iterations = args.iters; agent_cfg.seed = args.seed
from rsl_rl.runners import OnPolicyRunner
from client import RemoteVecEnv
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
env = RemoteVecEnv(args.target, os.environ["ENV_TOKEN"], authority=args.authority)
print("connected:", env.cfg, "agent:", agent_cfg.class_name, flush=True)
log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args.device)
t0 = time.time()
runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
print(f"TRAIN_DONE total {time.time() - t0:.1f}s for {args.iters} iters", flush=True)
env.close()
