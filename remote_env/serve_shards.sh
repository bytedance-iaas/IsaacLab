#!/bin/bash
# CPU env shards on this host: serve_shards.sh <num_shards> <total_envs> [base_port]
# Prints the comma-separated --target list for train_remote.py once every shard is listening; ENV_TOKEN must be set.
: "${ENV_TOKEN:?set ENV_TOKEN}"
N=$1; TOTAL=$2; BASE=${3:-19000}; LOGDIR=${SHARD_LOGDIR:-/tmp/env_shards}
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p "$LOGDIR"
export ENV_PERSIST=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TARGETS=""
for i in $(seq 0 $((N - 1))); do
  p=$((BASE + i)); TARGETS="$TARGETS,localhost:$p"
  ENV_PORT=$p python -u server.py --rl_library rsl_rl --task=${TASK:-Isaac-Velocity-Flat-G1} --device cpu \
    --num_envs $((TOTAL / N)) --max_iterations 1 --seed $((42 + i)) physics=${PHYSICS:-newton_mjwarp} agent.device=cpu \
    --run_name shard$i > "$LOGDIR/shard_$i.log" 2>&1 &
done
until [ "$(grep -l "listening on" "$LOGDIR"/shard_*.log 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 5; done
echo "${TARGETS#,}"
