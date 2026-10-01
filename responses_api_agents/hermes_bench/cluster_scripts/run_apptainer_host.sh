#!/bin/bash
#SBATCH --account=nemotron_rl_algo
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=hermesbench-apptainer-host

set -euo pipefail
umask 077

user_root=/lustre/fsw/portfolios/llmservice/users/cmunley
gym_root="$user_root/nemo-gym-hermesbench-20260722"
runtime_bin="$gym_root/responses_api_agents/hermes_bench/deps/hermes_bench_hermes_agent_deps/bin"
runtime_python="$runtime_bin/python"
run_tag="${HERMES_BENCH_RUN_TAG:-hermesbench-apptainer-host}"
output_dir="$user_root/eval/hermesbench_apptainer/$run_tag"
log_dir="$output_dir/logs"

mkdir -p "$log_dir" /tmp/appt /tmp/appc /tmp/apptainer-config
export PATH="$runtime_bin:$user_root/.local/bin:$user_root/apptainer-stage/apptainer-env/bin:$PATH"
export APPTAINER_BINPATH="$user_root/apptainer-stage/apptainer-env/bin"
export APPTAINER_TMPDIR=/tmp/appt
export APPTAINER_CACHEDIR=/tmp/appc
export APPTAINER_CONFIGDIR=/tmp/apptainer-config
export HERMES_BENCH_IMAGE="$user_root/hermesbench/images/hermesbench-eval-ca192cfb.sif"
export POLICY_API_KEY
POLICY_API_KEY="$(<"$user_root/.secrets/inference_api_key")"
export NOUS_API_KEY="$POLICY_API_KEY"
export SLURM_MASTER_NODE="${SLURM_MASTER_NODE:-$(hostname)}"
export RAY_ADDRESS=local
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_LINK_MODE=copy
export UV_LOCK_TIMEOUT=1800

cd "$gym_root"

cleanup() {
  if [ -n "${NG_RUN_PID:-}" ]; then
    kill "$NG_RUN_PID" 2>/dev/null || true
    wait "$NG_RUN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$runtime_python" "$runtime_bin/ng_run" \
  "+config_paths=[responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml,responses_api_agents/hermes_bench/configs/cluster_apptainer.yaml,responses_api_agents/hermes_bench/configs/sandbox_apptainer.yaml,responses_api_models/openai_model/configs/openai_model.yaml]" \
  +policy_base_url=https://inference-api.nvidia.com/v1 \
  '+policy_api_key=${oc.env:POLICY_API_KEY}' \
  "+policy_model_name=${HERMES_BENCH_POLICY_MODEL:-nvidia/nvidia/Nemotron-3-Nano-30B-A3B}" \
  ++uv_venv_dir="/tmp/nemo-gym-venvs/$run_tag" \
  >"$log_dir/ng_run.log" 2>&1 &
NG_RUN_PID=$!

until "$runtime_python" "$runtime_bin/ng_status" 2>&1 | grep -q '2 servers found (2 healthy'; do
  kill -0 "$NG_RUN_PID" 2>/dev/null || {
    wait "$NG_RUN_PID"
    exit 1
  }
  sleep 5
done

"$runtime_python" "$runtime_bin/ng_collect_rollouts" \
  "+input_jsonl_fpath=${HERMES_BENCH_INPUT:-responses_api_agents/hermes_bench/data/hermes_bench_task_memory.jsonl}" \
  +output_jsonl_fpath="$output_dir/rollouts.jsonl" \
  +agent_name=hermes_bench_hermes \
  "+limit=${HERMES_BENCH_LIMIT:-1}" \
  "+num_samples_in_parallel=${HERMES_BENCH_PARALLEL:-1}" \
  "+num_repeats=${HERMES_BENCH_REPEATS:-1}" \
  +responses_create_params.temperature=1.0 \
  +responses_create_params.top_p=0.95 \
  "+responses_create_params.max_output_tokens=${HERMES_BENCH_MAX_OUTPUT_TOKENS:-32768}" \
  +resume_from_cache=false \
  >"$log_dir/ng_collect.log" 2>&1
