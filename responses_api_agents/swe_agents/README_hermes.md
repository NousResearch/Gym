# Hermes Agent SWE-bench Rollouts

Runs hermes-agent as the SWE-bench agent framework inside Apptainer
containers, mirroring the OpenHands / SWE-agent flow in the base README but
driving the full hermes toolset (terminal, file ops, search, patch, etc.).
Per-turn token IDs and logprobs from the model are captured in the
trajectory, which makes this the path used for RL training data collection.

See the base README in this directory for the general SWE-agent setup
(apptainer install, env.yaml, vLLM serving, ng_run / ng_collect_rollouts).
This doc only covers the hermes-specific bits.

## Architecture

```
ng_run / ng_collect_rollouts
  SWE agents server (swebench_hermes.yaml or swebench_hermes_training.yaml)
    RunHermesAgent  -> Apptainer SWE-bench container
          /testbed            -- repo at base_commit
          /opt/hermes-agent   -- baked into the SIF
          hermes_runner.py    -- AIAgent.run_conversation()
                git diff -> SWE-bench eval -> reward
```

## Components

- `run_hermes.py` -- `RunHermesAgent`, a subclass of `RunOpenHandsAgent`.
  Writes `HERMES_RUNNER_SCRIPT` into the rollout output dir and invokes
  `/opt/hermes-agent/venv/bin/python hermes_runner.py ...` against `/testbed`
  inside each SWE-bench SIF. AIAgent is instantiated with
  `use_streaming=False`, `temperature=1.0`, `insert_reasoning=False`, and
  context compression disabled -- needed for prompt/generation token-ID
  contiguity across turns.
- `configs/swebench_hermes.yaml` -- standalone eval / rollout config.
- `configs/swebench_hermes_training.yaml` -- training-data-collection
  variant (framework = hermes, train + validation dataset registrations).

## Prerequisites

In addition to the base README prerequisites:

- Model served via vLLM (or OpenAI-compatible) with a tool parser. For
  Nemotron-3 Nano / similar NV models: `--tool-call-parser qwen3_coder` and
  the `nano_v3` reasoning parser. For Qwen3 Coder: `--tool-call-parser
  qwen3_coder`. See the base README for a sample `vllm serve` invocation.
- SIF images with `hermes-agent` pre-installed at `/opt/hermes-agent`
  (see below).

## Installing hermes-agent into the SIFs

`RunHermesAgent` runs `/opt/hermes-agent/venv/bin/python hermes_runner.py`
inside each SWE-bench SIF. hermes-agent has to be baked into each SIF
directly: `apptainer build --sandbox`, install into `/opt/hermes-agent`,
rebuild. Core of the per-SIF loop:

```bash
HERMES_BRANCH=nemo-gym-changes
HERMES_REPO=https://github.com/NousResearch/hermes-agent.git

# 1. unpack the SIF into a writable sandbox
apptainer build --sandbox "$SANDBOX" "$SIF_PATH"

# 2. clone + install hermes-agent at /opt/hermes-agent (writes its own venv)
apptainer exec --writable --no-mount home "$SANDBOX" bash -c "
    set -e
    rm -rf /opt/hermes-agent
    cd /opt
    git clone --branch $HERMES_BRANCH $HERMES_REPO hermes-agent
    cd hermes-agent
    HERMES_INSTALL_DIR=/opt/hermes-agent \
      bash ./scripts/install.sh --skip-setup --branch $HERMES_BRANCH
"

# 3. repack to a new SIF and verify the venv imports AIAgent
apptainer build "$NEW_SIF" "$SANDBOX"
apptainer exec --no-mount home "$NEW_SIF" \
    /opt/hermes-agent/venv/bin/python -c \
    'from run_agent import AIAgent; print("OK")'

# 4. swap the new SIF in for the old one
mv "$NEW_SIF" "$SIF_PATH"
```

`hermes-agent/scripts/install.sh --skip-setup` produces a self-contained
Python venv at `/opt/hermes-agent/venv`, so the rebuilt SIF needs no host
mounts at runtime.

For large SIF sets, wrap this in a sharded Slurm array
(`SLURM_ARRAY_TASK_ID` -> slice of the SIF list), use `.done` marker files
so re-runs resume where you left off, and run several SIFs per task in
parallel since the install is mostly network + I/O bound.

To tweak the installed hermes-agent across all SIFs later (e.g. a source
patch), use the same sandbox -> edit `/opt/hermes-agent/run_agent.py` ->
rebuild SIF pattern.

## Run

Same as the base README, but point at the hermes config:

```bash
config_paths="responses_api_agents/swe_agents/configs/swebench_hermes.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[$config_paths]" \
    +swe_agents.responses_api_agents.swe_agents.container_formatter=/path/to/sif/swebench_sweb.eval.x86_64.\{instance_id\}.sif
```

For batch rollout collection, use `ng_collect_rollouts +agent_name=swe_agents
...` exactly as in the base README.

## Data format

Standard SWE-bench rollout JSONL with the hermes agent wired in:

```json
{
  "agent_ref": {"name": "swe_agents_train", "type": "responses_api_agents"},
  "responses_create_params": {
    "input": [{"role": "user", "content": "...problem statement..."}],
    "model": "/path/to/model",
    "metadata": {
      "instance_id": "repo__project-1234",
      "base_commit": "abc123",
      "dataset_name": "SWE-Gym/SWE-Gym",
      "split": "train",
      "problem_statement": "...",
      "instance_dict": "{\"instance_id\":\"...\", \"repo\":\"...\", ...}"
    }
  }
}
```

Notes:

- `agent_ref.name` is the top-level server key in the config
  (`swe_agents_train` / `swe_agents_val`), not the dataset name.
- `metadata.instance_dict` is a JSON string of the SWE-bench instance row.
- `input` must contain at least one user message.
- `model` must match the served model name exactly.

## Trajectory output

Each rollout emits a trajectory with per-turn token IDs and logprobs
(courtesy of the NeMo Gym VLLMModel proxy with
`return_token_id_information: true`):

```
[0] system  -- persona + SWE prompt
[1] user    -- problem statement
[2] assistant  prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
[3] tool    -- tool result
[4] assistant  prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
...
```

## Gotchas

- Context compression must stay disabled in `hermes_runner.py`; enabling it
  rewrites earlier messages and invalidates prompt token IDs from prior
  turns, which breaks RL use downstream.
- `apptainer exec --pid` breaks in nested containers (Pyxis -> Apptainer)
  and is omitted in both `run_openhands.py` and `run_hermes.py`.
- The data's `model` field must match the served model name exactly or the
  proxy rejects requests.
- `container_formatter` can be a list; fallbacks are tried in order. If an
  instance isn't found under any pattern the rollout will fail.
