#!/bin/bash
#SBATCH --account=nemotron_rl_algo
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=0
#SBATCH --exclusive

set -euo pipefail
umask 077

user_root=/lustre/fsw/portfolios/llmservice/users/cmunley
gym_root="$user_root/nemo-gym-hermesbench-20260722"
run_tag="${HERMES_BENCH_RUN_TAG:?set HERMES_BENCH_RUN_TAG}"
provider="${HERMES_BENCH_PROVIDER:-opensandbox}"
output_dir="$user_root/eval/hermesbench_${provider}/$run_tag"
container=/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-nemo-rl-b620e79.sqsh

mkdir -p "$output_dir/logs"

srun \
  --output "$output_dir/logs/srun_%j.log" \
  --container-image "$container" \
  --container-mounts "$user_root:/workspace" \
  --container-workdir /workspace/nemo-gym-hermesbench-20260722 \
  --no-container-mount-home \
  --container-env=HERMES_BENCH_PROVIDER,HERMES_BENCH_RUN_TAG,HERMES_BENCH_POLICY_MODEL,HERMES_BENCH_INPUT,HERMES_BENCH_LIMIT,HERMES_BENCH_REPEATS,HERMES_BENCH_PARALLEL,HERMES_BENCH_MAX_OUTPUT_TOKENS,HERMES_BENCH_IMAGE \
  bash /workspace/nemo-gym-hermesbench-20260722/responses_api_agents/hermes_bench/cluster_scripts/run_cluster_inside.sh
