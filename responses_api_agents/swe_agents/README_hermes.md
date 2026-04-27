# Hermes Agent SWE-RL Training

Runs hermes-agent as a SWE-bench agent framework inside Apptainer/Singularity
containers for NeMo RL GRPO training. Replaces the OpenHands rollout in the
Stage 2 pipeline with hermes-agent so we train on the full hermes toolset
(terminal, file ops, search, patch, etc.) and capture per-turn token IDs +
logprobs for RL.

## Architecture

```
NeMo RL (GRPO training loop)  -- launched by super_launch.sh -> ray.sub
  Megatron policy (EP=8, CP=8, TP=1)
  vLLM inference workers (TP=2, async engine, qwen3_coder tool parser,
                          nano_v3 reasoning parser, enable_thinking=true)
        NeMo Gym VLLMModel proxy (adds token IDs + logprobs)
  NeMo Gym rollout collection
        SWE agents server (swebench_hermes_training.yaml)
              RunHermesAgent  -> Apptainer SWE-bench container
                    /testbed            -- repo at base_commit
                    /opt/hermes-agent   -- bind-mounted into the SIF
                    hermes_runner.py    -- AIAgent.run_conversation()
                          git diff -> SWE-bench eval -> binary reward
```

Cluster layout is config-driven (`cluster.num_nodes`, `policy.generation.colocated`,
etc). The Stage 2 Nano config currently ships as 12 nodes, non-colocated.

## Components

### Gym (this repo)

- `run_hermes.py` -- `RunHermesAgent`, a subclass of `RunOpenHandsAgent`. Writes
  `HERMES_RUNNER_SCRIPT` into the rollout output dir, bind-mounts it plus
  `/opt/hermes-agent` into each SWE-bench SIF, and invokes
  `/opt/hermes-agent/venv/bin/python hermes_runner.py ...` against `/testbed`.
  AIAgent is instantiated with `use_streaming=False`, `temperature=1.0`,
  `insert_reasoning=False`, and context compression disabled -- all required for
  prompt/generation token-ID contiguity across turns.
- `configs/swebench_hermes_training.yaml` -- train/val server config (framework
  = hermes, dataset registrations, defaults). Most per-run knobs (concurrency,
  agent_max_turns, swebench_agent_timeout, container_formatter) are overridden
  from the RL-side config via the `env.nemo_gym.swe_agents_{train,val}` block,
  so the values in this YAML are just the defaults.

### RL repo

- `super_launch.sh` -- main entry point. Takes EXP_NAME / TRAIN_PATH / VAL_PATH
  / CONFIG_PATH / MODEL_PATH / CONTAINER / SANDBOX_CONTAINER / PERSISTENT_CACHE
  / SLURM_PARTITION / SLURM_ACCOUNT as required env vars, snapshots the code,
  wires up caches and mounts, reads `cluster.num_nodes` from the config, and
  submits `ray.sub` via sbatch. Optional: `SIF_DIR` (forwarded as
  `sif_dir=...` to the training CLI), `EXTRA_MOUNTS`, `SLURM_TIME_LIMIT`,
  `DRY_RUN=true`.
- `examples/configs/super/stage2_hermes_nano_8node.yaml` -- Stage 2 GRPO config
  for Nemotron 3 Nano 30B-A3B with hermes agent. (Name says 8node but
  `cluster.num_nodes: 12`; rename if it bugs you.) Wires the Gym configs via
  `env.nemo_gym.config_paths` and applies the RL-side overrides.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py` -- overrides
  `request.temperature` / `request.top_p` to match generation config before
  serving (hermes requests can drift and vLLM would otherwise assert).

## Prerequisites

- Container: whichever NeMo RL image you pass as `CONTAINER` in
  `super_launch.sh`. It must have the vLLM tool-parser plugins in use
  (`qwen3_coder`, `nano_v3`).
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (instruct; ships the
  `<tool_call>` chat template with thinking support). The stage2 config points
  at `/home/dakota/hermes-swe-rl/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- SIF images: per-instance Apptainer `.sif` files for SWE-bench.
  Download via `./examples/nemo_gym/download_swe_images.py --sif-dir $SIF_DIR`.
  The stage2 config resolves them with a three-way `container_formatter`
  fallback: `swegym_sweb.eval.x86_64.{id}.sif`,
  `swebench_sweb.eval.x86_64.{id}.sif`, `r2egym_{id}.sif`.
- Data: JSONL rollout inputs (see below).
- `hermes-agent` installed at `/opt/hermes-agent` inside the NeMo RL container
  (or bind-mounted there -- see Installing hermes-agent below).

## Installing hermes-agent into the container

`RunHermesAgent` runs `/opt/hermes-agent/venv/bin/python hermes_runner.py`
inside each SWE-bench SIF. That path must resolve, so hermes-agent has to be
available at `/opt/hermes-agent` one way or another:

1. Bake it into the NeMo RL image (clone + `uv venv` + `uv pip install -e .[all]`
   at `/opt/hermes-agent`), or
2. Prebuild onto shared NFS and bind-mount into the NeMo RL container as
   `/opt/hermes-agent`, then let Gym forward the bind-mount into each SIF.
   `~/hermes-swe-rl/build_hermes_agent.sh` does this today (sbatch once to
   produce `/home/dakota/hermes-swe-rl/hermes-agent-install/hermes-agent/`
   with a Python 3.12 venv).

Either way, the install must include:

- branch `nemo-gym-changes` of `NousResearch/hermes-agent`
- a Python 3.12 venv at `/opt/hermes-agent/venv`
- `.[all]` extras installed

The uv-managed Python dir (`/root/.local/share/uv/python` or
`~/.local/share/uv/python`) also needs to be reachable from inside the SWE-bench
SIF so the venv symlinks resolve. `RunHermesAgent._get_extra_agent_mounts()`
bind-mounts both `/opt/hermes-agent` and the uv Python dir automatically if
present; for option (2) you just have to make sure they're mounted into the
NeMo RL container first (`super_launch.sh` EXTRA_MOUNTS, or baked in).

The temperature / reasoning_content / streaming monkey-patches in
`build_hermes_agent.sh` predate `run_hermes.py` passing those kwargs to
AIAgent directly and can probably be dropped; they're idempotent so leaving
them in is harmless.

## Data format

Each JSONL line:

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
  (`swe_agents_train` or `swe_agents_val`), not the dataset name.
- `metadata.instance_dict` is a JSON string of the SWE-bench instance row.
- `input` must contain at least one user message.
- `model` must match the served vLLM model name exactly.

## RL-side overrides that actually run

`stage2_hermes_nano_8node.yaml` overrides the Gym defaults in
`swebench_hermes_training.yaml`. Values used at runtime:

- `agent_max_turns: 200`
- `concurrency: 256`
- `swebench_agent_timeout: 3600`
- `run_with_mixed_prompts: true` (train only)
- `dataset_path` is pinned from `data.{train,validation}.data_path`
- `container_formatter` is the three-way fallback list above

If something looks off at runtime, diff against the RL config, not the Gym YAML.

## Launch

```bash
cd ~/github/RL
EXP_NAME=hermes-nano-stage2 \
TRAIN_PATH=/home/dakota/hermes-swe-rl/data/swegym_train.jsonl \
VAL_PATH=/home/dakota/hermes-swe-rl/data/swebench_verified_val.jsonl \
CONFIG_PATH=examples/configs/super/stage2_hermes_nano_8node.yaml \
MODEL_PATH=/home/dakota/hermes-swe-rl/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
CONTAINER=<nemo-rl image with hermes-agent reachable at /opt/hermes-agent> \
SANDBOX_CONTAINER=<nemo-skills sandbox sif> \
PERSISTENT_CACHE=/home/dakota/hermes-swe-rl/cache \
SIF_DIR=/home/dakota/hermes-swe-rl/sif \
SLURM_PARTITION=<...> SLURM_ACCOUNT=<...> \
bash super_launch.sh
```

`super_launch.sh` snapshots the code, reads `cluster.num_nodes` from the
config, and submits `ray.sub`. Set `DRY_RUN=true` to print the sbatch
invocation without submitting.

## Trajectory output

Each rollout emits a trajectory with per-turn token IDs and logprobs:

```
[0] system  -- persona + SWE prompt
[1] user    -- problem statement
[2] assistant  prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
[3] tool    -- tool result
[4] assistant  prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
...
```

Token IDs are injected by the NeMo Gym VLLMModel proxy
(`return_token_id_information: true`).

## Known issues / gotchas

- Context compression must stay disabled in `hermes_runner.py`; enabling it
  rewrites earlier messages and invalidates prompt token IDs from prior turns.
- `apptainer exec --pid` breaks in nested containers (Pyxis -> Apptainer) and
  is omitted in both `run_openhands.py` and `run_hermes.py`.
- vLLM worker overrides `request.temperature` / `request.top_p` to match
  generation config; keep that override (see `vllm_worker_async.py` lines
  around the `generation_config` block).
- Data's `model` field must match the vLLM served model name exactly, or the
  proxy rejects requests.
- `container_formatter` fallbacks are tried in order; if an instance isn't in
  any of swegym / swebench / r2egym SIF dirs it will fail to find a container.
