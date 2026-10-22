#!/bin/bash
# Persistent simulation server (serves PPO and SAC learners alike); ENV_TOKEN must be set by the caller.
# Overridable: TASK, NUM_ENVS, PHYSICS (newton_mjwarp | isaacsim_physx), ENV_PORT.
: "${ENV_TOKEN:?set ENV_TOKEN}"
export ENV_PORT=${ENV_PORT:-18990} ENV_PERSIST=1
cd "$(dirname "$(readlink -f "$0")")"
python -u server.py --rl_library rsl_rl --task=${TASK:-Isaac-Velocity-Flat-G1} --device cuda:0 --seed 42 \
  --num_envs ${NUM_ENVS:-4096} --max_iterations 1 physics=${PHYSICS:-newton_mjwarp} --run_name persist_srv
