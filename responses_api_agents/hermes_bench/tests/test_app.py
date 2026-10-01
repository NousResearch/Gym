import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from nemo_gym.sandbox.config import resolve_provider_config
from responses_api_agents.hermes_bench.agent_runner import (
    _preserve_hermes_behavior,
    _sync_hermes_skills,
    _task_toolsets,
)
from responses_api_agents.hermes_bench.app import HermesBenchAgent, HermesBenchAgentConfig
from responses_api_agents.hermes_bench.grader_runner import (
    _configure_judge_transport,
    _mapped_judge_model,
    _normalized_rubric,
    _task_object,
    grade,
    response_to_transcript,
)


def config(**overrides) -> HermesBenchAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "hermes_bench",
        "sandbox_provider": "sandbox",
        "sandbox_spec": {"image": "registry.example/hermes-bench:task"},
    }
    values.update(overrides)
    return HermesBenchAgentConfig(**values)


def test_provider_is_a_named_generic_sandbox() -> None:
    assert config().sandbox_provider == "sandbox"
    assert config(sandbox_provider={"apptainer": {}}).sandbox_provider == {"apptainer": {}}
    assert config(sandbox_provider={"opensandbox": {}}).sandbox_provider == {"opensandbox": {}}


def test_apptainer_is_selected_by_config_only() -> None:
    provider_path = Path(__file__).parents[1] / "configs" / "sandbox_apptainer.yaml"
    named_configs = yaml.safe_load(provider_path.read_text())
    assert resolve_provider_config("sandbox", named_configs) == {"apptainer": named_configs["sandbox"]["apptainer"]}


def test_sandbox_spec_uses_only_configured_image() -> None:
    params = SimpleNamespace(
        sandbox_spec={
            "image": "docker://configured/image:tag",
            "workdir": "/sandbox/workspace",
            "resources": {"cpu": 2},
            "metadata": {"source": "test"},
        },
        grader_sandbox_spec={"resources": {"memory_mib": 4096}},
        sandbox_default_metadata={"sandbox-api": "test"},
        task={"task_id": "task_one"},
        task_id="task_one",
        agent_timeout_s=100,
        grader_timeout_s=20,
        timeout_multiplier=2.0,
    )
    agent_spec = HermesBenchAgent._sandbox_spec(HermesBenchAgent, params, "agent")
    grader_spec = HermesBenchAgent._sandbox_spec(HermesBenchAgent, params, "grader")
    assert agent_spec.image == "docker://configured/image:tag"
    assert agent_spec.resources.cpu == 2
    assert grader_spec.resources.cpu == 2
    assert grader_spec.resources.memory_mib == 4096
    assert grader_spec.metadata["phase"] == "grader"


def test_task_image_can_override_config() -> None:
    params = SimpleNamespace(
        sandbox_spec={"image": "base:image"},
        grader_sandbox_spec={},
        sandbox_default_metadata={},
        task={"task_id": "task_one", "image": "task:image"},
        task_id="task_one",
        agent_timeout_s=100,
        grader_timeout_s=20,
        timeout_multiplier=2.0,
    )
    spec = HermesBenchAgent._sandbox_spec(HermesBenchAgent, params, "agent")
    assert spec.image == "task:image"


def test_task_timeout_matches_source_multiplier_and_grace() -> None:
    params = SimpleNamespace(task={"timeout_seconds": 1800}, agent_timeout_s=100, timeout_multiplier=2.0)
    assert HermesBenchAgent._task_timeout_s(params) == 3630


def test_agent_model_falls_back_to_model_server_name() -> None:
    params = SimpleNamespace(
        agent_kwargs={},
        model_server_url="http://model:10240/v1",
        model_server=SimpleNamespace(name="policy_model"),
        body=SimpleNamespace(model=None),
        agent_deps_archive=Path("/tmp/deps.tar.gz"),
        agent_deps_url=None,
        preserve_hermes_behavior=True,
    )

    assert HermesBenchAgent._agent_env(params)["NGHB_MODEL_NAME"] == "policy_model"


def test_sandbox_model_url_can_resolve_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("responses_api_agents.hermes_bench.app.socket.gethostbyname", lambda host: "10.1.2.3")
    assert HermesBenchAgent._sandbox_model_url("http://model-node:10240/v1", True) == "http://10.1.2.3:10240/v1"


def test_sandbox_image_auth_loads_from_file(tmp_path: Path) -> None:
    auth_file = tmp_path / "registry_token"
    auth_file.write_text("private-token\n")
    agent_config = config(
        sandbox_image_auth_file=auth_file,
        sandbox_spec={
            "image": "registry.example/hermes-bench:task",
            "provider_options": {"image_auth": {"username": "runner"}},
        },
    )

    HermesBenchAgent._load_sandbox_image_auth(agent_config)

    assert agent_config.sandbox_spec["provider_options"]["image_auth"] == {
        "username": "runner",
        "password": "private-token",
    }


def test_sandbox_provider_api_key_loads_from_file(tmp_path: Path) -> None:
    api_key_file = tmp_path / "opensandbox_api_key"
    api_key_file.write_text("private-api-key\n")
    agent_config = config(
        sandbox_provider_api_key_file=api_key_file,
    )
    named_configs = {
        "sandbox": {
            "default_metadata": {"sandbox-api": "opensandbox-sdk"},
            "opensandbox": {"connection": {"domain": "sandbox.example", "api_key": "loaded-from-file"}},
        },
    }

    provider = HermesBenchAgent._load_sandbox_provider_api_key(agent_config, named_configs)

    assert provider["opensandbox"]["connection"]["api_key"] == "private-api-key"
    assert named_configs["sandbox"]["opensandbox"]["connection"]["api_key"] == "loaded-from-file"


def test_judge_model_map_preserves_original_families(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "HERMES_JUDGE_MODEL_MAP",
        json.dumps(
            {
                "anthropic/claude-opus-4.7": "aws/anthropic/bedrock-claude-opus-4-7",
                "google/gemini-3.5-flash": "gcp/google/gemini-3.5-flash",
            }
        ),
    )

    assert _mapped_judge_model("anthropic/claude-opus-4.7") == "aws/anthropic/bedrock-claude-opus-4-7"
    task = {
        "task_id": "architecture",
        "grading_type": "hybrid",
        "judge_config": {"model": "google/gemini-3.5-flash"},
    }
    assert _task_object(task).judge_config["model"] == "gcp/google/gemini-3.5-flash"


def test_architecture_judge_url_can_be_routed_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("lib_architecture_diagram_judge")
    module.NOUS_URL = "https://inference-api.nousresearch.com/v1/chat/completions"
    monkeypatch.setitem(sys.modules, "lib_architecture_diagram_judge", module)
    monkeypatch.setenv("HERMES_ARCHITECTURE_JUDGE_URL", "https://inference-api.nvidia.com/v1/chat/completions")

    _configure_judge_transport()

    assert module.NOUS_URL == "https://inference-api.nvidia.com/v1/chat/completions"


def test_architecture_score_cache_control_can_be_routed_by_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("responses_api_agents.hermes_bench.grader_runner.CONTROL", tmp_path)
    monkeypatch.setenv("HERMES_ARCHITECTURE_SCORE_CACHE_CONTROL", "false")
    task = {
        "task_id": "architecture",
        "grading_type": "architecture_diagram",
        "judge_config": {},
    }

    assert _task_object(task).judge_config["score_cache_control"] == "false"


def test_plain_numbered_rubric_is_normalized() -> None:
    rubric = """
1. **Element count**: Contains exactly four elements.
2. **Connection labels**: Every connection is labeled.
"""
    normalized = _normalized_rubric(rubric)
    assert "### Criterion 1: Element count (Weight: 50.000000%)" in normalized
    assert "### Criterion 2: Connection labels (Weight: 50.000000%)" in normalized


def test_task_toolsets_keep_skills_available() -> None:
    assert _task_toolsets({"toolsets": ["file", "terminal"]}, None) == ["file", "terminal", "skills"]
    assert _task_toolsets({}, ["terminal", "skills"]) == ["terminal", "skills"]


def test_skills_are_synced_after_hermes_home_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("tools.skills_sync")
    calls = []
    module.sync_skills = lambda quiet: calls.append(quiet) or {"total_bundled": 4}
    executable = tmp_path / "agent_deps" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    (tmp_path / "agent_deps" / "skills").mkdir()
    monkeypatch.setattr("responses_api_agents.hermes_bench.agent_runner.sys.executable", str(executable))
    monkeypatch.setattr("responses_api_agents.hermes_bench.agent_runner.importlib.import_module", lambda name: module)
    monkeypatch.setattr("responses_api_agents.hermes_bench.agent_runner.importlib.reload", lambda value: value)

    _sync_hermes_skills(["terminal", "skills"])

    assert calls == [True]
    assert os.environ["HERMES_BUNDLED_SKILLS"] == str(tmp_path / "agent_deps" / "skills")


def test_transcript_groups_parallel_calls() -> None:
    response = {
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "working"}]},
            {"type": "function_call", "name": "read_file", "arguments": '{"path":"a"}', "call_id": "one"},
            {"type": "function_call", "name": "read_file", "arguments": '{"path":"b"}', "call_id": "two"},
            {"type": "function_call_output", "call_id": "one", "output": "A"},
        ]
    }
    transcript = response_to_transcript(response)
    assert len(transcript[0]["tool_calls"]) == 2
    assert [item["name"] for item in transcript[0]["message"]["content"] if item["type"] == "toolCall"] == [
        "read_file",
        "read_file",
    ]
    assert transcript[1]["role"] == "tool"


def test_automated_grader_reads_real_workspace(tmp_path: Path, monkeypatch) -> None:
    import responses_api_agents.hermes_bench.grader_runner as runner

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("June 1, 2024 is the beta release deadline.")
    monkeypatch.setattr(runner, "WORKSPACE", workspace)
    grading = ModuleType("lib_grading")

    def grade_task(task, execution_result, judge_model=None):
        namespace = {}
        exec(task.automated_checks, namespace)
        scores = namespace["grade"](
            execution_result["transcript"],
            execution_result["workspace_path"],
            execution_result["output"],
        )
        numeric = [float(value) for value in scores.values() if isinstance(value, (int, float, bool))]
        return {
            "overall_score": sum(numeric) / len(numeric),
            "score_valid": True,
            "grading_status": "scored",
            "scores": {"automated": scores},
        }

    grading.grade_task = grade_task
    monkeypatch.setitem(sys.modules, "lib_grading", grading)
    task = {
        "task_id": "task_memory",
        "name": "Memory",
        "grading_type": "automated",
        "prompt": "Answer the question",
        "automated_checks": """
def grade(transcript, workspace_path, output=''):
    from pathlib import Path
    content = (Path(workspace_path) / 'answer.txt').read_text()
    return {
        'correct': 'June 1, 2024' in content,
        'saw_transcript': bool(transcript),
    }
""",
        "llm_judge_rubric": None,
        "grading_weights": None,
    }
    response = {
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
    }
    result = grade(task, response)
    assert result["score_valid"] is True
    assert result["reward"] == 1.0
    assert result["scores"]["automated"] == {
        "correct": True,
        "saw_transcript": True,
    }


async def test_timed_out_agent_preserves_workspace_and_returns_for_grading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSandbox:
        def __init__(self, *args):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def upload(self, source, target):
            pass

        async def exec(self, command, **kwargs):
            if command.startswith("tar -czf"):
                return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
            return SimpleNamespace(return_code=125, stdout="partial agent output", stderr="timed out", error_type="timeout")

        async def download(self, source, target):
            if source.endswith("workspace.tar.gz"):
                Path(target).write_bytes(b"partial workspace")
                return
            raise FileNotFoundError(source)

    class Harness:
        def _sandbox_spec(self, params, phase):
            return {}

        async def _seed_sandbox(self, sandbox, params):
            pass

        _agent_env = staticmethod(HermesBenchAgent._agent_env)
        _task_timeout_s = staticmethod(lambda params: 30)

    monkeypatch.setattr("responses_api_agents.hermes_bench.app.AsyncSandbox", FakeSandbox)
    persistent_dir = tmp_path / "result"
    persistent_dir.mkdir()
    (persistent_dir / "body.json").write_text("{}")
    params = SimpleNamespace(
        resolved_sandbox_provider={},
        persistent_dir=persistent_dir,
        task={},
        task_id="timeout_task",
        body=SimpleNamespace(model="test-model", parallel_tool_calls=True, tool_choice="auto", tools=[]),
        agent_deps_archive=None,
        agent_deps_url=None,
        agent_python="python3",
        agent_kwargs={},
        model_server_url="http://model/v1",
        model_server=SimpleNamespace(name="policy_model"),
        preserve_hermes_behavior=True,
    )

    response = await HermesBenchAgent._run_agent(Harness(), params)
    metrics = json.loads((persistent_dir / "agent_metrics.json").read_text())

    assert response.output == []
    assert (persistent_dir / "workspace.tar.gz").read_bytes() == b"partial workspace"
    assert metrics["agent_timed_out"] is True
    assert metrics["agent_return_code"] == 125


def test_missing_required_image_is_a_scored_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import responses_api_agents.hermes_bench.grader_runner as runner

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(runner, "WORKSPACE", workspace)
    skills_runner = ModuleType("hermes_skills_runner")
    skills_runner.call_grade = lambda task, record: {"file_created": 0.0}
    skills_runner.aggregate_score = lambda scores, weights: 0.0
    grading = ModuleType("lib_grading")
    grading.calculate_overall_score = lambda *args, **kwargs: 1.0
    grading.default_judge_backend = lambda: "nous"
    grading.run_llm_judge_full = lambda **kwargs: pytest.fail("judge should not run without its required image")
    monkeypatch.setitem(sys.modules, "hermes_skills_runner", skills_runner)
    monkeypatch.setitem(sys.modules, "lib_grading", grading)
    task = {
        "task_id": "p5",
        "name": "p5",
        "grading_type": "hybrid",
        "source_file": "tasks/hermes_skills/p5js/task.md",
        "llm_judge_rubric": "### Criterion 1: Render (Weight: 100%)",
        "judge_config": {"requires_images": True},
        "grading_weights": {"automated": 0.5, "llm_judge": 0.5},
    }

    result = grade(task, {"model": "test", "output": []})

    assert result["reward"] == 0.0
    assert result["score_valid"] is True
    assert result["grading_status"] == "scored_model_failure"


def test_missing_architecture_artifact_is_a_scored_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import responses_api_agents.hermes_bench.grader_runner as runner

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(runner, "WORKSPACE", workspace)
    monkeypatch.setattr(runner, "CONTROL", control)
    grading = ModuleType("lib_grading")
    grading.grade_task = lambda *args, **kwargs: {
        "overall_score": None,
        "score_valid": False,
        "grading_status": "grading_error",
        "scores": {
            "architecture_diagram": {
                "score": None,
                "score_valid": False,
                "error": "No architecture artifact found in the workspace",
            }
        },
        "feedback": "No architecture artifact found in the workspace",
    }
    monkeypatch.setitem(sys.modules, "lib_grading", grading)
    task = {
        "task_id": "architecture",
        "name": "architecture",
        "grading_type": "architecture_diagram",
        "architecture_rubric": "rubric",
        "judge_config": {},
    }

    result = grade(task, {"model": "test", "output": []})

    assert result["reward"] == 0.0
    assert result["score_valid"] is True
    assert result["grading_status"] == "scored_model_failure"


def test_runner_sources_compile() -> None:
    root = Path(__file__).parents[1]
    for name in ("agent_runner.py", "grader_runner.py", "p5_direct.py"):
        compile((root / name).read_text(), name, "exec")


def test_agent_runner_is_hermes_specific() -> None:
    source = (Path(__file__).parents[1] / "agent_runner.py").read_text()
    assert "from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig" in source
    assert "NGHB_AGENT_MODULE" not in source
    assert "NGHB_AGENT_CLASS" not in source
    assert "NGHB_AGENT_CONFIG_CLASS" not in source


def test_p5_runner_uses_a_writable_browser_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from responses_api_agents.hermes_bench import p5_direct

    recorded = {}

    def run(command, env):
        recorded["command"] = command
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(p5_direct.subprocess, "run", run)
    monkeypatch.setattr(
        p5_direct.sys,
        "argv",
        ["p5_direct.py", "sketch.html", str(tmp_path), "--output", "/out/render.png"],
    )

    assert p5_direct.main() == 0
    assert recorded["command"][-1] == str(tmp_path / "render.png")
    assert Path(recorded["env"]["HOME"]).name.startswith("hermes-p5-")
    assert recorded["env"]["XDG_CACHE_HOME"].startswith(recorded["env"]["HOME"])
    assert recorded["env"]["XDG_CONFIG_HOME"].startswith(recorded["env"]["HOME"])


def test_config_has_no_image_default() -> None:
    source = (Path(__file__).parents[1] / "configs" / "hermes_bench_hermes.yaml").read_text()
    assert "image: ${oc.env:HERMES_BENCH_IMAGE}" in source
    assert "python:3" not in source
    assert "ubuntu:" not in source


def test_config_matches_archived_hermes_harness() -> None:
    source = (Path(__file__).parents[1] / "configs" / "hermes_bench_hermes.yaml").read_text()
    assert "domain: agent" in source
    assert "type: benchmark" in source
    assert "prepare_script: responses_api_agents/hermes_bench/prepare.py" in source
    assert "max_turns: 90" in source
    assert "enabled_toolsets: [terminal, file, code_execution, memory, skills, session_search]" in source
    assert "terminal_backend: local" in source
    assert "agent_server_module" not in source
    assert "agent_server_class" not in source
    assert "agent_config_class" not in source
    assert "preserve_hermes_behavior: true" in source
    assert "NOUS_API_KEY: ${oc.env:NOUS_API_KEY,''}" in source
    assert "P5_RENDER_CMD: python3 /sandbox/control/p5_direct.py {sketch} {out} {extra}" in source
    assert "judge_api_key" not in source


def test_task_json_is_not_exposed_as_workspace_fixture(tmp_path: Path) -> None:
    task = {"task_id": "one", "workspace_files": [{"path": "secret", "content": "x"}]}
    public = {key: value for key, value in task.items() if key != "workspace_files"}
    assert "workspace_files" not in json.loads(json.dumps(public))


def test_sandbox_bootstrap_starts_from_existing_root() -> None:
    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert '"mkdir -p /sandbox/control /sandbox/workspace /sandbox/benchmark /sandbox/agent_deps",' in source
    assert 'cwd="/",' in source


def test_main_suite_restores_hermes_memory_and_session(tmp_path: Path, monkeypatch) -> None:
    import yaml

    run_agent = ModuleType("run_agent")

    class AIAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    run_agent.AIAgent = AIAgent
    monkeypatch.setitem(sys.modules, "run_agent", run_agent)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text("memory:\n  memory_enabled: false\n  user_profile_enabled: false\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("NGHB_PRESERVE_HERMES_BEHAVIOR", "1")

    _preserve_hermes_behavior({"source_file": "tasks/memory/task_memory.md"})
    values = run_agent.AIAgent(skip_memory=True, skip_context_files=True, persist_session=False).kwargs
    config = yaml.safe_load(config_path.read_text())

    assert values["skip_memory"] is False
    assert values["skip_context_files"] is False
    assert values["persist_session"] is True
    assert config["memory"] == {"memory_enabled": True, "user_profile_enabled": True}
