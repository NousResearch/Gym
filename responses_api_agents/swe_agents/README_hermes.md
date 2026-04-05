# Hermes Agent SWE-RL Training

Runs hermes-agent as a SWE-bench agent framework inside Apptainer containers for
NeMo RL GRPO training. Replaces OpenHands in the Stage 2.2 pipeline with hermes-agent,
training on the full hermes toolset (terminal, file ops, search, patch, etc.).

## Architecture

```
NeMo RL (GRPO training loop)
  ├── Megatron policy workers (4 nodes, TP=2 EP=8)
  ├── vLLM inference workers (4 nodes, TP=4)
  │     └── NeMo Gym VLLMModel proxy (adds token IDs + logprobs)
  └── NeMo Gym rollout collection
        └── SWE agents server (swebench_hermes_training.yaml)
              └── RunHermesAgent → Apptainer container
                    ├── /testbed          — SWE-bench repo at base_commit
                    ├── /opt/hermes-agent — built at container startup
                    └── hermes_runner.py  — AIAgent.run_conversation()
                          ├── 16 tools (terminal, file, search, patch, etc.)
                          ├── token IDs captured per-turn
                          └── git diff → SWE-bench eval → binary reward
```

## What Changed vs OpenHands Pipeline

### Gym (this repo)

- `run_hermes.py` — `RunHermesAgent` class inheriting `RunOpenHandsAgent`
  - Builds hermes-agent inside container at `/opt/hermes-agent` (SETUP_COMMAND)
  - Mounts into Apptainer SWE-bench containers with uv Python
  - Removed `--pid` flag from apptainer exec (breaks in nested containers)
  - `HERMES_RUNNER_SCRIPT` — standalone script that runs inside Apptainer
- `configs/swebench_hermes_training.yaml` — train/val config for hermes framework
- `run_openhands.py` — also removed `--pid` from apptainer exec

### RL repo

- `examples/configs/super/stage2_hermes_nano_8node.yaml` — 8-node config for
  Nemotron 3 Nano 30B-A3B with hermes agent
- `ray.sub` — patched for cluster (pmi2, container-writable, NCCL IB, ssh instead
  of srun --overlap)
- `nemo_rl/models/generation/vllm/vllm_worker_async.py` — replaced temperature/top_p
  assert with override (hermes requests may not match generation config exactly)

## Prerequisites

- **Container**: `nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super` (official NGC)
  - The old manually-built sqsh lacked tool parser support — must use official container
- **Model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (instruct, has chat template
  with `<tool_call>` format and thinking support)
- **SIF images**: Apptainer .sif files for SWE-bench instances
  - Download: `./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif`
- **Data**: JSONL with `responses_create_params`, `agent_ref`, and `instance_dict`

## Data Format

Each JSONL line must have:

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
      "instance_dict": "{\"instance_id\":\"...\", \"repo\":\"...\", ...}",
      ...
    }
  }
}
```

Key fields:
- `agent_ref.name` must match the config key (`swe_agents_train` or `swe_agents_val`)
- `metadata.instance_dict` is a JSON-serialized string of the SWE-bench instance
- `input` must contain at least one user message (empty list causes IndexError)
- `model` must match the served model name (full path)

## Launch

```bash
# From ~/hermes-swe-rl/
bash launch_hermes_swe.sh
```

The launch script:
1. Installs apptainer on compute nodes (SETUP_COMMAND)
2. Clones + builds hermes-agent at `/opt/hermes-agent` inside container
3. Patches hermes to send `temperature=1.0` (vLLM asserts on temperature match)
4. Runs `run_grpo_nemo_gym.py` with the hermes stage2 config

## Trajectory Output

Each rollout produces a trajectory with per-turn token IDs and logprobs:

```
[0] system  — persona + SWE prompt
[1] user    — problem statement
[2] assistant ✅ prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
[3] tool    — tool result
[4] assistant ✅ prompt_token_ids + generation_token_ids + logprobs, tool_calls=1
...
```

Token IDs are injected by the NeMo Gym VLLMModel proxy (`return_token_id_information: true`).

## Known Issues

- **Container version matters**: The official NGC container has tool parsers
  (`qwen3_coder`). Manually-built sqsh from the super-v3 branch may lack them.
- **Temperature assert**: NeMo RL's vLLM worker asserts `request.temperature ==
  generation_config.temperature`. Hermes may send different values. Fixed by
  overriding in `vllm_worker_async.py`.
- **Apptainer --pid**: Fails in nested containers (Pyxis → Apptainer). Removed.
- **srun --overlap**: Doesn't work when all node resources are allocated. Use ssh.
- **Model name**: Data's `model` field must match the vLLM served model name exactly.
