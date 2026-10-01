import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.hermes_bench.app import HermesBenchAgent


RUN_E2E = os.environ.get("HERMES_BENCH_RUN_DOCKER_E2E") == "1"
ROOT = Path(__file__).parents[1]


@dataclass
class Params:
    task: dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    fixture_archive: Path
    support_archive: Path
    resolved_sandbox_provider: dict[str, Any]
    sandbox_spec: dict[str, Any]
    grader_sandbox_spec: dict[str, Any] = field(default_factory=dict)
    sandbox_default_metadata: dict[str, Any] = field(default_factory=dict)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    model_server_url: str = ""
    agent_deps_archive: Path | None = None
    agent_deps_url: str | None = None
    preserve_hermes_behavior: bool = True
    agent_python: str = "python3"
    grader_python: str = "python3"
    agent_timeout_s: int = 120
    grader_timeout_s: int = 120
    timeout_multiplier: float = 1.0
    grader_env: dict[str, str] = field(default_factory=dict)
    grader_env_files: dict[str, Path] = field(default_factory=dict)
    judge_model: str | None = None

    @property
    def task_id(self) -> str:
        return self.task["task_id"]


class Harness:
    _agent_env = staticmethod(HermesBenchAgent._agent_env)
    _merge_spec = staticmethod(HermesBenchAgent._merge_spec)
    _task_timeout_s = staticmethod(HermesBenchAgent._task_timeout_s)
    _sandbox_spec = HermesBenchAgent._sandbox_spec
    _seed_sandbox = HermesBenchAgent._seed_sandbox


@pytest.mark.skipif(not RUN_E2E, reason="set HERMES_BENCH_RUN_DOCKER_E2E=1")
@pytest.mark.asyncio
async def test_agent_and_original_grader_in_fresh_docker_sandboxes(tmp_path: Path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    data_path = ROOT / "data" / "hermes_bench.jsonl"
    if not data_path.exists():
        pytest.skip("prepare HermesBench data first")

    tag = "hermes-bench-fake-runtime:e2e"
    subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(Path(__file__).parent / "fake_runtime")],
        check=True,
    )

    harness = Harness()
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    cases = (
        ("task_memory", 1.0, True),
        ("task_ascii_art_basic_banner", 0.0, True),
        ("task_ghcr_approve_clean_pr", 0.0, False),
    )
    for task_id, expected_reward, expected_valid in cases:
        row = next(
            row for row in rows if json.loads(row["responses_create_params"]["metadata"]["task"])["task_id"] == task_id
        )
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {**row["responses_create_params"], "model": "fake-model"}
        )
        task = json.loads(body.metadata["task"])
        persistent = tmp_path / task_id
        persistent.mkdir()
        (persistent / "body.json").write_text(body.model_dump_json())
        (persistent / "task.json").write_text(json.dumps(task))

        params = Params(
            task=task,
            body=body,
            persistent_dir=persistent,
            fixture_archive=(ROOT.parents[1] / body.metadata["fixture_archive"]).resolve(),
            support_archive=(ROOT.parents[1] / body.metadata["support_archive"]).resolve(),
            resolved_sandbox_provider={"docker": {}},
            sandbox_spec={"image": tag, "workdir": "/sandbox/workspace"},
            preserve_hermes_behavior=False,
        )
        response = await HermesBenchAgent._run_agent(harness, params)
        metrics = await HermesBenchAgent._run_grader(harness, params)

        assert response.output
        assert metrics.score_valid is expected_valid
        assert metrics.reward == expected_reward
        assert metrics.mask_sample is not expected_valid
        if task_id == "task_ghcr_approve_clean_pr":
            with tarfile.open(persistent / "workspace.tar.gz") as archive:
                assert "workspace/acme-utils/.git/HEAD" in archive.getnames()
            assert "NOUS_API_KEY is not set" in metrics.error
