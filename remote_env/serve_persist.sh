#!/bin/bash
# Persistent PhysX simulation server; ENV_TOKEN must be set by the caller.
: "${ENV_TOKEN:?set ENV_TOKEN}"
export ENV_PORT=${ENV_PORT:-18990} ENV_PERSIST=1
cd "$(dirname "$(readlink -f "$0")")"
python -u server_v3.py --rl_library rsl_rl --task=Isaac-Velocity-Flat-G1 --device cuda:0 --seed 42 \
  --num_envs 4096 --max_iterations 1 physics=isaacsim_physx --run_name persist_srv
