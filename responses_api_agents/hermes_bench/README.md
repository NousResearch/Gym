# HermesBench

HermesBench evaluates Hermes-Agent skills, tool use, memory, compaction,
safety, and artifact creation. It has 140 tasks: 92 automated, 33 hybrid,
7 LLM judge, and 8 architecture diagram.

## Benchmark Behavior

Preparation uses the original `scripts/lib_tasks.py`, task definitions,
fixtures, setup, grading, rubrics, and weights. Each rollout uses one fresh
agent sandbox and one fresh grader sandbox.

The 61 main-suite tasks use `lib_grading.grade_task`. The 79 skills tasks use
`hermes_skills_runner.call_grade`. Architecture tasks use the source judge.
The Hermes harness keeps the source prompt, toolsets, `max_turns: 90`, timeout
multiplier, memory, session, and skills-runner behavior. Only Hermes-Agent is
supported.

The adapter normalizes two malformed Excalidraw rubrics. A missing required
artifact is a valid model failure with reward 0. Other grading infrastructure
failures set `score_valid: false` and `mask_sample: true`.

The ZIP's `scores_and_models.txt` is not extracted. Judge credentials come
from runtime environment variables.

## Data Preparation

Requires Python 3.12 or newer, `uv`, and `npm`. Pass
`hermesbench-eval-clean.zip` directly to the script without extracting it.
Its SHA-256 is
`9f2edc889680801a5e77e2c13d18f62eb91b97384600964b03ebd2c749bd944e`.

One task also requires
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) at
commit `ca192cfb773915c9d8113352b8646d3ce9329424`.
The evaluation runtime is pinned separately by
`responses_api_agents/hermes_agent/requirements.txt`.

```bash
uv run --frozen python responses_api_agents/hermes_bench/prepare.py \
  /path/to/hermesbench-eval-clean.zip \
  --hermes-agent-path /path/to/hermes-agent
```

This writes the JSONL, fixtures, and grading support to
`responses_api_agents/hermes_bench/data/`. The directory is ignored by Git.
The ZIP is not needed after preparation. Use `--task` or `--limit` for a
smaller dataset.

## Quickstart

Build the task image from the ZIP's original environment recipe:

```bash
docker build --platform linux/amd64 -t hermes-bench:local responses_api_agents/hermes_bench
```

Publish that image for OpenSandbox or convert it to a SIF for Apptainer, then:

```bash
export HERMES_BENCH_IMAGE=docker://registry.example/hermes-bench:pinned-tag
export POLICY_BASE_URL=https://provider.example/v1
export POLICY_API_KEY=...
export POLICY_MODEL_NAME=provider/model
uv run --frozen bash responses_api_agents/hermes_bench/run.sh
```

The command runs all 140 tasks with temperature 1 and top-p 0.95. A limit is
only a smoke test. The first eight rows are architecture tasks, so a prefix is
not representative of the full score.

OpenSandbox uses the same command with:

```bash
export HERMES_BENCH_IMAGE=registry.example/hermes-bench:pinned-tag
export HERMES_BENCH_PROVIDER_CONFIG=nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
```

`agent_runtime_source: auto` builds the pinned Hermes-Agent runtime on Linux
x86_64. Use a baked or portable runtime on macOS.

The image needs `bash`, `tar`, Python 3, Git, PyYAML, `requests`, task
dependencies, and browser rendering dependencies. The default config forwards
`NOUS_API_KEY` and `OPENROUTER_API_KEY`. Set `HERMES_BENCH_EXTRA_CONFIG` when
custom sandbox networking or judge routing needs another config overlay.
Comparable scores require the same policy model, judge routes, and all 140
tasks.

## Test

```bash
uv run --frozen --extra dev pytest -q responses_api_agents/hermes_bench/tests
```
