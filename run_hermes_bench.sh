#!/usr/bin/env bash
# Top-level HermesBench runner: image + dataset + preflight + eval in one shot.
#
# Usage:
#   export MODEL_API_KEY=...             # key for the agent-model endpoint
#   export NOUS_API_KEY=...              # architecture-diagram vision judge
#   ./run_hermes_bench.sh --provider openrouter --model anthropic/claude-sonnet-4.6
#   ./run_hermes_bench.sh --provider nous --model Hermes-4.5-405B
#
# Common options (all optional):
#   --provider NAME        openrouter | nous — sets the endpoint URL and,
#                          when MODEL_API_KEY is unset, falls back to
#                          OPENROUTER_API_KEY / NOUS_API_KEY respectively
#                          (default: openrouter)
#   --model MODEL          model id (default: anthropic/claude-sonnet-4.6)
#   --model-url URL        custom OpenAI-compatible endpoint (overrides --provider)
#   --output NAME          results basename (default: hermes_bench_<model>_<date>)
#   --source PATH          HermesBench repo checkout (default: /home/shared/hermesbench-eval)
#   --hermes-agent PATH    Hermes-Agent checkout @ ca192cfb for the kanban task
#                          fixture (required only when --prepare is used)
#   --image PATH           existing .sif (default: build/reuse .hermes_bench.sif)
#   --prepare              re-run prepare.py (needed after task/grader changes)
#   --tasks REGEX          run only tasks whose task_id or category matches
#                          (e.g. --tasks systematic_debugging, --tasks p5js,
#                          --tasks 'task_arxiv_.*'); default: all 140
#   --port N               head server port (default: 11000; unique per
#                          concurrent run on the same host)
#
# Note: the architecture-diagram vision judge always routes through the Nous
# inference API with NOUS_API_KEY regardless of --provider; this keeps judge
# scores comparable across runs.
#
# The script:
#   1. builds the sandbox image from image.def if missing
#   2. optionally re-prepares the dataset from the HermesBench source repo
#   3. clears stale aggregate-metrics files (they hard-crash the run)
#   4. stops orphaned nemo-gym apptainer instances from dead runs
#   5. runs the full suite via `gym eval run` with an isolated RAY_TMPDIR
#   6. prints the health verdict (score_valid / mask_sample) at the end

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
adapter="responses_api_agents/hermes_bench"

model="anthropic/claude-sonnet-4.6"
provider="openrouter"
model_url=""
output=""
source_repo="/home/shared/hermesbench-eval"
hermes_agent_path=""
image="$script_dir/.hermes_bench.sif"
do_prepare=false
task_filter=""
port=11000

while [ $# -gt 0 ]; do
  case "$1" in
    --provider) provider="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --model-url) model_url="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --source) source_repo="$2"; shift 2 ;;
    --hermes-agent) hermes_agent_path="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --prepare) do_prepare=true; shift ;;
    --tasks) task_filter="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# Resolve endpoint + key: --model-url wins over --provider; the key is
# MODEL_API_KEY when set, else the provider's conventional env var.
model_api_key="${MODEL_API_KEY:-}"
case "$provider" in
  openrouter)
    [ -n "$model_url" ] || model_url="https://openrouter.ai/api/v1"
    [ -n "$model_api_key" ] || model_api_key="${OPENROUTER_API_KEY:-}"
    ;;
  nous)
    [ -n "$model_url" ] || model_url="https://inference-api.nousresearch.com/v1"
    [ -n "$model_api_key" ] || model_api_key="${NOUS_API_KEY:-}"
    ;;
  *)
    [ -n "$model_url" ] || { echo "unknown --provider '$provider' (openrouter|nous) and no --model-url given" >&2; exit 2; }
    ;;
esac

[ -n "$model_api_key" ] || { echo "No model API key: set MODEL_API_KEY (or OPENROUTER_API_KEY / NOUS_API_KEY to match --provider)" >&2; exit 2; }
[ -n "${NOUS_API_KEY:-}" ] || echo "WARNING: NOUS_API_KEY unset — architecture-diagram judging will fail" >&2

if [ -z "$output" ]; then
  output="hermes_bench_$(echo "$model" | tr '/.' '__')_$(date +%Y%m%d_%H%M)"
fi

# 1. Sandbox image
if [ ! -f "$image" ]; then
  echo "==> Building sandbox image: $image"
  apptainer build --fakeroot "$image" "$adapter/image.def"
fi

# 2. Dataset
if $do_prepare; then
  [ -n "$hermes_agent_path" ] || { echo "--prepare needs --hermes-agent PATH (checkout @ ca192cfb)" >&2; exit 2; }
  echo "==> Preparing dataset from $source_repo"
  python "$adapter/prepare.py" "$source_repo" --hermes-agent-path "$hermes_agent_path"
fi
[ -f "$adapter/data/hermes_bench.jsonl" ] || { echo "No dataset — run with --prepare first" >&2; exit 2; }

# 3+4. Preflight
rm -f results/preprocessed_datasets/benchmark_metrics*.json "$adapter"/data/hermes_bench_metrics*.json
# Only reap orphaned sandboxes when no other eval is running — in parallel
# multi-model mode this cleanup would kill sibling runs' live sandboxes.
#
# Counting 'gym eval run' alone is racy: a sibling launched seconds earlier has
# not spawned that process yet, so the count reads 1 and we reap its sandboxes
# out from under it. Observed 2026-08-12 — four tasks in a co-running eval died
# with "no instance found with name nemo-gym-..." and scored 0.0, which looked
# exactly like model failure. Also honour an explicit opt-out so a caller who
# knows siblings are coming can disable reaping outright.
reap_guard_count="$(pgrep -fc 'gym eval run' || true)"
runner_siblings="$(pgrep -fc 'run_hermes_bench.sh' || true)"
if [ "${HERMES_BENCH_NO_REAP:-0}" = "1" ]; then
  echo "==> sandbox reaping disabled (HERMES_BENCH_NO_REAP=1)"
elif [ "$reap_guard_count" -le 1 ] && [ "${runner_siblings:-1}" -le 1 ]; then
  apptainer instance list 2>/dev/null | awk '/nemo-gym/ {print $1}' | xargs -r -I{} apptainer instance stop {} 2>/dev/null || true
else
  echo "==> skipping orphan-sandbox reap: $runner_siblings runner(s), $reap_guard_count eval(s) live"
fi

# Task-family filter: build a filtered JSONL + a config copy pointing at it
# (the --input CLI flag is silently ignored; jsonl_fpath in the config wins).
agent_config="$adapter/configs/hermes_bench_hermes.yaml"
if [ -n "$task_filter" ]; then
  filtered="$adapter/data/hermes_bench_filtered_$$.jsonl"
  TASK_FILTER="$task_filter" FILTERED_OUT="$filtered" python3 - <<'PY'
import json, os, re, sys
pat = re.compile(os.environ["TASK_FILTER"])
src = "responses_api_agents/hermes_bench/data/hermes_bench.jsonl"
n = 0
with open(src) as f, open(os.environ["FILTERED_OUT"], "w") as g:
    for line in f:
        d = json.loads(line)
        task = json.loads(d["responses_create_params"]["metadata"]["task"])
        hay = f'{task.get("task_id","")} {task.get("category","")}'
        if pat.search(hay):
            g.write(line if line.endswith("\n") else line + "\n")
            n += 1
if n == 0:
    sys.exit(f"--tasks '{pat.pattern}' matched 0 of 140 tasks")
print(f"==> --tasks '{pat.pattern}' matched {n} task(s)")
PY
  agent_config="results/hermes_bench_hermes_filtered_$$.yaml"
  mkdir -p results
  sed "s|jsonl_fpath:.*hermes_bench.jsonl|jsonl_fpath: $filtered|" \
    "$adapter/configs/hermes_bench_hermes.yaml" > "$agent_config"
  trap 'rm -f "$filtered" "$agent_config"' EXIT
fi

# 5. Run
export HERMES_BENCH_IMAGE="$image"
export RAY_TMPDIR="/tmp/ray_hb_$$"
mkdir -p "$RAY_TMPDIR" results

echo "==> Running with $model (output: results/${output}.jsonl)"
.venv/bin/gym eval run \
  --config "$agent_config" \
  --config "$adapter/configs/sandbox_apptainer.yaml" \
  --model-type openai_model \
  --agent hermes_bench_hermes \
  --output "results/${output}.jsonl" \
  --model "$model" \
  --model-url "$model_url" \
  --model-api-key "$model_api_key" \
  --split benchmark \
  '+head_server.host=127.0.0.1' \
  "+head_server.port=$port" \
  2>&1 | tee "results/${output}.log"

# 6. Verdict
metrics="results/${output}_aggregate_metrics.json"
if [ -f "$metrics" ]; then
  python3 - "$metrics" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
flat = json.dumps(m)
def pick(key):
    import re
    hit = re.search(rf'"{key}":\s*([0-9.]+)', flat)
    return float(hit.group(1)) if hit else None
reward, valid, mask = pick("mean/reward"), pick("mean/score_valid"), pick("mean/mask_sample")
print(f"\n==> mean/reward={reward}  score_valid={valid}  mask_sample={mask}")
if valid == 1.0 and mask == 0.0:
    print("==> HEALTHY RUN: every task graded, zero infrastructure failures")
else:
    print("==> WARNING: masked/invalid samples present — audit zero-reward rows before trusting the mean")
PY
fi
