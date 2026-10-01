#!/bin/bash

set -euo pipefail
umask 077

user_root=/workspace
gym_root="$user_root/nemo-gym-hermesbench-20260722"
provider="${HERMES_BENCH_PROVIDER:-opensandbox}"
run_tag="${HERMES_BENCH_RUN_TAG:?set HERMES_BENCH_RUN_TAG}"
policy_model="${HERMES_BENCH_POLICY_MODEL:?set HERMES_BENCH_POLICY_MODEL}"
input="${HERMES_BENCH_INPUT:-responses_api_agents/hermes_bench/data/hermes_bench.jsonl}"
limit="${HERMES_BENCH_LIMIT:-140}"
repeats="${HERMES_BENCH_REPEATS:-1}"
parallel="${HERMES_BENCH_PARALLEL:-64}"
max_output_tokens="${HERMES_BENCH_MAX_OUTPUT_TOKENS:-32768}"
output_dir="$user_root/eval/hermesbench_${provider}/$run_tag"
runtime_bin="$gym_root/.venv/bin"

export PATH="$runtime_bin:$user_root/apptainer-stage/apptainer-env/bin:$PATH"
export POLICY_API_KEY
POLICY_API_KEY="$(<"$user_root/.secrets/inference_api_key")"
export NOUS_API_KEY="$POLICY_API_KEY"
export SLURM_MASTER_NODE="${SLURM_MASTER_NODE:-$(hostname)}"
export RAY_ADDRESS=local
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export APPTAINER_BINPATH="$user_root/apptainer-stage/apptainer-env/bin"
export APPTAINER_TMPDIR=/tmp/appt
export APPTAINER_CACHEDIR=/tmp/appc
export APPTAINER_CONFIGDIR=/tmp/apptainer-config

mkdir -p "$output_dir/logs" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_CONFIGDIR"
cd "$gym_root"

case "$provider" in
  opensandbox)
    export HERMES_BENCH_IMAGE="${HERMES_BENCH_IMAGE:-gitlab-master.nvidia.com:5005/cmunley/nemo-containers/hermesbench-eval:ca192cfb}"
    config_paths="responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml,responses_api_agents/hermes_bench/configs/cluster_opensandbox.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
    ;;
  apptainer)
    export HERMES_BENCH_IMAGE="${HERMES_BENCH_IMAGE:-$user_root/hermesbench/images/hermesbench-eval-ca192cfb.sif}"
    config_paths="responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml,responses_api_agents/hermes_bench/configs/cluster_apptainer.yaml,responses_api_agents/hermes_bench/configs/sandbox_apptainer.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
    ;;
  *)
    echo "unknown provider: $provider" >&2
    exit 2
    ;;
esac

cleanup() {
  if [ -n "${NG_RUN_PID:-}" ]; then
    kill "$NG_RUN_PID" 2>/dev/null || true
    wait "$NG_RUN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$runtime_bin/ng_run" \
  "+config_paths=[$config_paths]" \
  +policy_base_url=https://inference-api.nvidia.com/v1 \
  '+policy_api_key=${oc.env:POLICY_API_KEY}' \
  "+policy_model_name=$policy_model" \
  "++uv_venv_dir=/tmp/nemo-gym-venvs/$run_tag" \
  >"$output_dir/logs/ng_run.log" 2>&1 &
NG_RUN_PID=$!

until "$runtime_bin/ng_status" 2>&1 | grep -q '2 servers found (2 healthy'; do
  kill -0 "$NG_RUN_PID" 2>/dev/null || {
    wait "$NG_RUN_PID"
    exit 1
  }
  sleep 5
done

"$runtime_bin/ng_collect_rollouts" \
  "+input_jsonl_fpath=$input" \
  "+output_jsonl_fpath=$output_dir/rollouts.jsonl" \
  +agent_name=hermes_bench_hermes \
  "+limit=$limit" \
  "+num_samples_in_parallel=$parallel" \
  "+num_repeats=$repeats" \
  +responses_create_params.temperature=1.0 \
  +responses_create_params.top_p=0.95 \
  "+responses_create_params.max_output_tokens=$max_output_tokens" \
  +resume_from_cache=false \
  >"$output_dir/logs/ng_collect.log" 2>&1
