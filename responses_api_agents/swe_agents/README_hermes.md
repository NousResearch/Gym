# Hermes SWE Agent for NeMo Gym

Runs hermes-agent as a SWE-bench agent framework inside Apptainer containers.
Token IDs and logprobs are captured from each model server response and included
in the trajectory output.

## Architecture

```
NeMo Gym app.py (SWEBenchWrapper)
  └─ Ray dispatch → run_hermes_swebench_evaluation()
       └─ RunHermesAgent.process_single_datapoint()
            └─ Apptainer container
                 ├─ /testbed          — SWE-bench repo at base_commit
                 ├─ /hermes_setup     — pre-built hermes-agent (install.sh)
                 ├─ /root/.hermes     — config, API keys, memory, skills
                 └─ /trajectories_mount — output (prediction + trajectory)
```

The agent runs `AIAgent.run_conversation()` inside the container — identical to
a normal `hermes` CLI session. No stripped-down tool sets, no skipped features.
The model trains on exactly what users see.

## Files

- `run_hermes.py` — `RunHermesAgent` class + in-container runner script
- `utils.py` — `run_hermes_swebench_evaluation()`, `setup_hermes_environment()`, `get_hermes_trajectory()`
- `app.py` — dispatches to hermes when `agent_framework: hermes`
- `configs/swebench_hermes.yaml` — config template

## Local Testing

### Prerequisites

```bash
# Apptainer (Ubuntu — may need AppArmor fix for user namespaces)
curl -fsSL -o /tmp/apptainer.deb \
  https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
sudo dpkg -i /tmp/apptainer.deb
sudo apt-get install -y uidmap  # for fakeroot support

# If you get "setgroups: Permission denied":
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

# Pull a base container image
apptainer build --sandbox /tmp/swe_test_sandbox docker://python:3.12

# Build hermes using the production setup function
cd /path/to/Gym
python -c "
from responses_api_agents.swe_agents.utils import setup_hermes_environment
print(setup_hermes_environment())
"
# Creates: responses_api_agents/swe_agents/swe_hermes_setup/hermes-agent/
# with its own venv, Python 3.12 via uv, all deps installed
```

### Start model server

```bash
# Raw vLLM
HF_HOME=/path/to/hf/cache vllm serve Qwen/Qwen3-8B \
  --port 13020 --trust-remote-code --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice --tool-call-parser hermes

# Optional: token ID proxy (simulates NeMo Gym model server's token ID injection)
# See /tmp/token_id_proxy.py or use the real NeMo Gym model server (ng_run)
```

### Run the test

```bash
HERMES_SETUP=responses_api_agents/swe_agents/swe_hermes_setup  # from setup step
UV_PYTHON=~/.local/share/uv/python        # uv-managed Python (venv symlinks here)
OUTPUT_DIR=/tmp/hermes_swe_test            # output directory
HERMES_HOME=~/.hermes                      # config, API keys, memory, skills
MODEL_URL=http://127.0.0.1:13020/v1        # model server (or proxy for token IDs)
MODEL_NAME=Qwen/Qwen3-8B

mkdir -p $OUTPUT_DIR

# Extract the runner script from run_hermes.py
python3 -c "
content = open('responses_api_agents/swe_agents/run_hermes.py').read()
script = content.split(\"HERMES_RUNNER_SCRIPT = r'''\")[1].split(\"'''\")[0]
open('$OUTPUT_DIR/hermes_runner.py', 'w').write(script)
"

# Run inside Apptainer — matches the production RunHermesAgent._execute_container_command()
apptainer exec --writable-tmpfs --cleanenv --pid --no-mount home,tmp,bind-paths \
    --mount type=bind,src=$OUTPUT_DIR,dst=/trajectories_mount \
    --mount type=bind,src=$HERMES_SETUP,dst=/hermes_setup,ro \
    --mount type=bind,src=$HERMES_SETUP,dst=$(realpath $HERMES_SETUP),ro \
    --mount type=bind,src=$UV_PYTHON,dst=$(realpath $UV_PYTHON),ro \
    --mount type=bind,src=$HERMES_HOME,dst=/root/.hermes \
    /tmp/swe_test_sandbox \
    bash -c '
echo "127.0.0.1 localhost" > /etc/hosts 2>/dev/null || true

# Set up a testbed with a bug
mkdir -p /testbed && cd /testbed
git init -q && git config user.email test@test.com && git config user.name Test
echo "def add(a, b): return a - b  # BUG" > main.py
git add . && git commit -m "initial with bug" -q

# Run hermes from /testbed so context files come from the target repo
cd /testbed
PYTHONPATH=/hermes_setup/hermes-agent \
/hermes_setup/hermes-agent/venv/bin/python /trajectories_mount/hermes_runner.py \
    --model-base-url '"$MODEL_URL"' \
    --model '"$MODEL_NAME"' \
    --instance-id test__bug-001 \
    --problem-statement "Fix add(): it subtracts instead of adding." \
    --output-file /trajectories_mount/test.jsonl \
    --max-turns 5
'
```

### Check output

```bash
# SWE-bench prediction (patch)
python3 -m json.tool $OUTPUT_DIR/test.jsonl

# Trajectory with token IDs
python3 -c "
import json
with open('$OUTPUT_DIR/test.trajectory.json') as f:
    d = json.load(f)
for i, m in enumerate(d['messages']):
    role = m.get('role', '?')
    has_tokens = 'prompt_token_ids' in m
    tc = len(m.get('tool_calls', []))
    if role == 'system':
        print(f'[{i}] system ({len(m[\"content\"])} chars)')
    elif has_tokens:
        p = len(m['prompt_token_ids'])
        g = len(m['generation_token_ids'])
        print(f'[{i}] {role} ✅ prompt={p} gen={g} tc={tc}')
    else:
        print(f'[{i}] {role} tc={tc}')
"
```

### Expected output

```
[0] system (2605 chars)              # persona + SWE prompt (no hermes AGENTS.md)
[1] user tc=0                        # problem statement
[2] assistant ✅ prompt=N gen=N tc=1  # reads the file
[3] tool tc=0                        # file contents
[4] assistant ✅ prompt=N gen=N tc=1  # patches the bug
[5] tool tc=0                        # patch result
[6] assistant ✅ prompt=N gen=N tc=0  # confirms fix
```

Token IDs require a model server that returns `prompt_token_ids`,
`generation_token_ids`, `generation_log_probs` on the response message
(NeMo Gym's `VLLMModel` does this). Raw vLLM won't have them — the agent
still works, just without token-level data in the trajectory.

## Key design decisions

- **Zero diff from production**: no `skip_context_files`, no `skip_memory`,
  no restricted toolsets. The container runs hermes exactly like a user would.
- **CWD = /testbed**: context file auto-injection scans the target repo, not
  hermes's source tree.
- **~/.hermes mounted**: config.yaml, .env (API keys), memory, skills all
  available inside the container.
- **Inherits from RunOpenHandsAgent**: reuses container finding, Apptainer
  execution, and SWE-bench evaluation infrastructure.
