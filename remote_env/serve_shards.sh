#!/bin/bash
# CPU env shards on this host: serve_shards.sh <num_shards> <total_envs> [base_port]
# Prints the comma-separated --target list for train_remote.py once every shard is listening; ENV_TOKEN must be set.
: "${ENV_TOKEN:?set ENV_TOKEN}"
N=$1; TOTAL=$2; BASE=${3:-19000}; LOGDIR=${SHARD_LOGDIR:-/tmp/env_shards}
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p "$LOGDIR"
export ENV_PERSIST=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# SHARD_CPUS (e.g. 8-31,40-127): pin each shard to one of these physical cores and its SMT sibling,
# or to a single hardware thread when there are more shards than cores
UNITS=()
if [ -n "$SHARD_CPUS" ]; then
  for r in ${SHARD_CPUS//,/ }; do
    for c in $(seq ${r%-*} ${r#*-}); do UNITS+=("$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"); done
  done
  [ "$N" -gt "${#UNITS[@]}" ] && UNITS=($(printf '%s\n' "${UNITS[@]}" | tr , '\n'))
fi
TARGETS=""
for i in $(seq 0 $((N - 1))); do
  p=$((BASE + i)); TARGETS="$TARGETS,localhost:$p"
  PIN=(); [ ${#UNITS[@]} -gt 0 ] && PIN=(taskset -c "${UNITS[i % ${#UNITS[@]}]}")
  ENV_PORT=$p "${PIN[@]}" python -u server.py --rl_library rsl_rl --task=${TASK:-Isaac-Velocity-Flat-G1} --device cpu \
    --num_envs $((TOTAL / N + (i < TOTAL % N ? 1 : 0))) --max_iterations 1 --seed $((42 + i)) physics=${PHYSICS:-newton_mjwarp} agent.device=cpu \
    --run_name shard$i > "$LOGDIR/shard_$i.log" 2>&1 &
done
until [ "$(grep -l "listening on" "$LOGDIR"/shard_*.log 2>/dev/null | wc -l)" -ge "$N" ]; do sleep 5; done
echo "${TARGETS#,}"
