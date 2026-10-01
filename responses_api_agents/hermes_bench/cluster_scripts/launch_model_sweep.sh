#!/bin/bash

set -euo pipefail

user_root=/lustre/fsw/portfolios/llmservice/users/cmunley
runner="$user_root/nemo-gym-hermesbench-20260722/responses_api_agents/hermes_bench/cluster_scripts/run_cluster.sh"
stamp="${HERMES_BENCH_SWEEP_STAMP:-$(date +%Y%m%d-%H%M%S)}"
models=(
  "sonnet46|aws/anthropic/bedrock-claude-sonnet-4-6"
  "sonnet5|azure/anthropic/claude-sonnet-5"
  "opus48|azure/anthropic/claude-opus-4-8"
  "gpt55|openai/openai/gpt-5.5"
  "gemini31pro|gcp/google/gemini-3.1-pro-preview"
  "qwen35-397b|nvidia/qwen/qwen3-5-397b-a17b"
  "kimik26|nvidia/moonshotai/kimi-k2.6"
  "nemotron3ultra|nvidia/nvidia/nemotron-3-ultra"
)

for entry in "${models[@]}"; do
  tag="${entry%%|*}"
  model="${entry#*|}"
  run_tag="hermesbench-osb-${tag}-avg5-full140-64way-${stamp}"
  log_dir="$user_root/eval/hermesbench_opensandbox/$run_tag/logs"
  mkdir -p "$log_dir"
  job=$(sbatch --parsable \
    --job-name="hb-${tag}-avg5" \
    --output="$log_dir/sbatch_%j.log" \
    --export="ALL,HERMES_BENCH_PROVIDER=opensandbox,HERMES_BENCH_RUN_TAG=$run_tag,HERMES_BENCH_POLICY_MODEL=$model,HERMES_BENCH_INPUT=responses_api_agents/hermes_bench/data/hermes_bench.jsonl,HERMES_BENCH_LIMIT=140,HERMES_BENCH_REPEATS=5,HERMES_BENCH_PARALLEL=64,HERMES_BENCH_MAX_OUTPUT_TOKENS=32768" \
    "$runner")
  printf '%s %s %s\n' "$tag" "$job" "$run_tag"
done
