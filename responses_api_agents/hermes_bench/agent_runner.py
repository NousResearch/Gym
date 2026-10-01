# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import importlib
import inspect
import json
import os
import sys
from pathlib import Path


if os.environ.get("NGHB_EXTERNAL_RUNTIME") != "1":
    sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/sandbox/agent_deps/bin:" + os.environ.get("PATH", "")


def _json_env(name: str) -> dict:
    encoded = os.environ.get(f"{name}_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode())
    return json.loads(os.environ.get(name, "{}"))


def _preserve_hermes_behavior(task: dict) -> None:
    if os.environ.get("NGHB_PRESERVE_HERMES_BEHAVIOR") != "1":
        return

    import run_agent
    import yaml

    source_file = str(task.get("source_file") or "").replace("\\", "/")
    is_skills_task = source_file.startswith("tasks/hermes_skills/")
    original_init = run_agent.AIAgent.__init__

    def hermes_bench_init(self, *args, **kwargs):
        kwargs["skip_context_files"] = is_skills_task
        kwargs["skip_memory"] = is_skills_task
        kwargs["persist_session"] = True
        original_init(self, *args, **kwargs)

    run_agent.AIAgent.__init__ = hermes_bench_init
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    memory = config.setdefault("memory", {})
    memory["memory_enabled"] = not is_skills_task
    memory["user_profile_enabled"] = not is_skills_task
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def _sync_hermes_skills(enabled_toolsets: list[str] | None) -> None:
    if not enabled_toolsets or "skills" not in enabled_toolsets:
        return

    bundled_skills = Path(sys.executable).parent.parent / "skills"
    if bundled_skills.is_dir():
        os.environ["HERMES_BUNDLED_SKILLS"] = str(bundled_skills)
    module = importlib.import_module("tools.skills_sync")
    module = importlib.reload(module)
    result = module.sync_skills(quiet=True)
    if not result.get("total_bundled"):
        raise RuntimeError("Hermes skills toolset is enabled but no bundled skills were found")


def _task_toolsets(task: dict, configured: list[str] | None) -> list[str] | None:
    selected = task.get("toolsets") or configured
    if selected is None:
        return None
    enabled = [selected] if isinstance(selected, str) else list(selected)
    if "skills" not in enabled:
        enabled.append("skills")
    return enabled


def main() -> None:
    workspace = Path(os.environ.get("NGHB_WORKSPACE", "/sandbox/workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)

    model_url = os.environ.get("NGHB_MODEL_URL", "")
    model_name = os.environ["NGHB_MODEL_NAME"]
    agent_kwargs = _json_env("NGHB_AGENT_KWARGS")
    task = json.loads(Path("/sandbox/control/task.json").read_text())

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig

    model_base_url = model_url.rstrip("/")
    if model_base_url.endswith("/v1"):
        model_base_url = model_base_url[:-3]
    global_config = {"policy_model": {"responses_api_models": {"policy_model": {}}}}
    model_server = ModelServerRef(name="policy_model", type="responses_api_models")
    client = ServerClient.model_construct(global_config_dict=global_config)
    client._build_server_base_url = lambda config: model_base_url

    enabled_toolsets = _task_toolsets(task, agent_kwargs.get("enabled_toolsets"))
    config_values = {
        "host": "0.0.0.0",
        "port": 0,
        "name": HermesAgent.__name__.lower(),
        "entrypoint": "app.py",
        "model_server": model_server,
        "resources_server": ResourcesServerRef(name="hermes_bench", type="resources_servers"),
        **agent_kwargs,
    }
    if enabled_toolsets is not None:
        config_values["enabled_toolsets"] = enabled_toolsets
    config = HermesAgentConfig(**config_values)
    agent = HermesAgent(config=config, server_client=client)
    _preserve_hermes_behavior(task)
    _sync_hermes_skills(getattr(config, "enabled_toolsets", None))

    body_data = json.loads(Path("/sandbox/control/body.json").read_text())
    body_data["model"] = model_name
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(body_data)
    call_kwargs = {"body": body}
    signature = inspect.signature(agent.responses)
    if "request" in signature.parameters:
        call_kwargs["request"] = None
    if "response" in signature.parameters:
        from starlette.responses import Response

        call_kwargs["response"] = Response()
    response = asyncio.run(agent.responses(**call_kwargs))
    Path("/sandbox/control/response.json").write_text(response.model_dump_json())
    print(f"agent finished with {len(response.output)} output items", flush=True)


if __name__ == "__main__":
    main()
