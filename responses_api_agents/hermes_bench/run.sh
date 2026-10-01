#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Set $name before running HermesBench" >&2
    exit 2
  fi
}

require_env HERMES_BENCH_IMAGE
require_env POLICY_BASE_URL
require_env POLICY_API_KEY
require_env POLICY_MODEL_NAME

input="${HERMES_BENCH_INPUT:-responses_api_agents/hermes_bench/data/hermes_bench.jsonl}"
output="${HERMES_BENCH_OUTPUT:-results/hermes_bench.jsonl}"
provider_config="${HERMES_BENCH_PROVIDER_CONFIG:-responses_api_agents/hermes_bench/configs/sandbox_apptainer.yaml}"
model_config="${HERMES_BENCH_MODEL_CONFIG:-responses_api_models/openai_model/configs/openai_model.yaml}"
extra_config="${HERMES_BENCH_EXTRA_CONFIG:-}"
limit="${HERMES_BENCH_LIMIT:-140}"
parallel="${HERMES_BENCH_PARALLEL:-4}"
repeats="${HERMES_BENCH_REPEATS:-1}"
health_timeout="${HERMES_BENCH_HEALTH_TIMEOUT:-900}"

if [ ! -f "$input" ]; then
  echo "HermesBench input not found: $input" >&2
  exit 2
fi

config_paths="responses_api_agents/hermes_bench/configs/hermes_bench_hermes.yaml"
if [ -n "$extra_config" ]; then
  config_paths="$config_paths,$extra_config"
fi
config_paths="$config_paths,$provider_config,$model_config"

mkdir -p "$(dirname "$output")"
env_log="${output%.jsonl}.env.log"

cleanup() {
  if [ -n "${ng_run_pid:-}" ]; then
    kill "$ng_run_pid" 2>/dev/null || true
    wait "$ng_run_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ng_run \
  "+config_paths=[$config_paths]" \
  "+policy_base_url=$POLICY_BASE_URL" \
  '+policy_api_key=${oc.env:POLICY_API_KEY}' \
  "+policy_model_name=$POLICY_MODEL_NAME" \
  >"$env_log" 2>&1 &
ng_run_pid=$!

ready=false
for ((elapsed = 0; elapsed < health_timeout; elapsed += 5)); do
  if ng_status 2>&1 | grep -q '2 servers found (2 healthy'; then
    ready=true
    break
  fi
  if ! kill -0 "$ng_run_pid" 2>/dev/null; then
    wait "$ng_run_pid"
    exit 1
  fi
  sleep 5
done

if [ "$ready" != true ]; then
  echo "HermesBench services were not healthy after ${health_timeout}s" >&2
  exit 1
fi

ng_collect_rollouts \
  "+input_jsonl_fpath=$input" \
  "+output_jsonl_fpath=$output" \
  +agent_name=hermes_bench_hermes \
  "+limit=$limit" \
  "+num_samples_in_parallel=$parallel" \
  "+num_repeats=$repeats" \
  +responses_create_params.temperature=1.0 \
  +responses_create_params.top_p=0.95 \
  +responses_create_params.max_output_tokens=32768 \
  +resume_from_cache=false

echo "HermesBench results: $output"
