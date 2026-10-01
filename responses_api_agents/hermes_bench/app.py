# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
import uuid
from asyncio import Semaphore, sleep
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata


class HermesBenchMetrics(BaseModel):
    task_id: str
    reward: float = 0.0
    score_valid: bool = False
    grading_status: str = "pending"
    scores: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    agent_timed_out: bool = False
    agent_return_code: Optional[int] = None
    agent_error: Optional[str] = None
    mask_sample: bool = False
    agent_run_time: Optional[float] = None
    grader_run_time: Optional[float] = None


class HermesBenchAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: Optional[ModelServerRef] = None
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)

    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_spec: dict[str, Any]
    grader_sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    sandbox_image_auth_file: Optional[Path] = None
    sandbox_provider_api_key_file: Optional[Path] = None
    sandbox_model_base_url: Optional[str] = None
    sandbox_model_resolve_host: bool = False
    agent_runtime_source: str = "baked"
    preserve_hermes_behavior: bool = True
    agent_python: str = "python3"
    grader_python: str = "python3"
    agent_timeout_s: int = 1800
    grader_timeout_s: int = 600
    timeout_multiplier: float = 2.0
    concurrency: int = 4

    judge_model: Optional[str] = None
    grader_env: dict[str, str] = Field(default_factory=dict)
    grader_env_files: dict[str, Path] = Field(default_factory=dict)


class HermesBenchServerConfig(BaseModel):
    run_session_id: str
    base_results_dir: Path
    model_server_url: str
    resolved_sandbox_provider: dict[str, Any]
    sandbox_default_metadata: dict[str, Any]
    agent_deps_archive: Optional[Path] = None
    agent_deps_url: Optional[str] = None


class HermesBenchInstanceConfig(HermesBenchAgentConfig, HermesBenchServerConfig):
    task: dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    fixture_archive: Path
    support_archive: Path

    @property
    def task_id(self) -> str:
        return self.task["task_id"]


class HermesBenchVerifyResponse(HermesBenchMetrics, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    instance_config: dict[str, Any]


def _redact(value: Any, key: str = "") -> Any:
    normalized = key.lower()
    if any(marker in normalized for marker in ("api_key", "secret", "password")) or normalized.endswith("token"):
        return "***"
    if isinstance(value, dict):
        return {child_key: _redact(child, child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _safe_config(params: HermesBenchInstanceConfig) -> dict[str, Any]:
    data = params.model_dump(mode="json")
    data.pop("body", None)
    data.pop("task", None)
    return _redact(data)


def _check(result: SandboxExecResult, label: str) -> None:
    if result.return_code != 0:
        stdout = (result.stdout or "")[-2000:]
        stderr = (result.stderr or "")[-2000:]
        raise RuntimeError(f"{label} failed with exit {result.return_code}: {stdout}{stderr}")


class AgentRuntime:
    def __init__(self):
        self.root = Path(__file__).parent

    def setup(self) -> Path:
        deps_dir = self.root / "deps" / "hermes_bench_hermes_agent_deps"
        script = self.root / "setup_scripts" / "hermes_agent_deps.sh"
        portable = self.root / "setup_scripts" / "_portable_python.sh"
        requirements = PARENT_DIR / "responses_api_agents" / "hermes_agent" / "requirements.txt"
        if not script.exists():
            raise ValueError(f"missing agent runtime setup script: {script}")
        recipe = hashlib.sha256(
            b"".join(path.read_bytes() for path in (script, portable, requirements) if path.exists())
        ).hexdigest()
        sentinel = deps_dir / ".installed"
        if sentinel.exists() and sentinel.read_text().strip() == recipe:
            return deps_dir
        lock = deps_dir.parent / f".{deps_dir.name}.lockdir"
        lock.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.time() - lock.stat().st_mtime > 3600:
                    shutil.rmtree(lock, ignore_errors=True)
                else:
                    time.sleep(2)
        try:
            if sentinel.exists() and sentinel.read_text().strip() == recipe:
                return deps_dir
            deps_dir.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "DEPS_DIR": str(deps_dir), "NEMO_GYM_ROOT": str(PARENT_DIR)}
            subprocess.run(["bash", str(script)], env=env, check=True)
            sentinel.write_text(recipe)
            return deps_dir
        finally:
            shutil.rmtree(lock, ignore_errors=True)


class HermesBenchAgent(SimpleResponsesAPIAgent):
    config: HermesBenchAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _sem: Optional[Semaphore] = None
    _server: Optional[HermesBenchServerConfig] = None

    @staticmethod
    def _sandbox_model_url(value: str, resolve_host: bool) -> str:
        if not value or not resolve_host:
            return value
        parsed = urlsplit(value)
        if not parsed.hostname:
            raise ValueError(f"sandbox model URL has no hostname: {value}")
        host = socket.gethostbyname(parsed.hostname)
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    @staticmethod
    def _load_sandbox_image_auth(config: HermesBenchAgentConfig) -> None:
        if config.sandbox_image_auth_file is None:
            return
        password = config.sandbox_image_auth_file.read_text().strip()
        if not password:
            raise ValueError(f"sandbox image auth file is empty: {config.sandbox_image_auth_file}")
        options = config.sandbox_spec.setdefault("provider_options", {})
        image_auth = options.setdefault("image_auth", {})
        image_auth["password"] = password

    @staticmethod
    def _load_sandbox_provider_api_key(
        config: HermesBenchAgentConfig,
        named_configs: dict[str, Any],
    ) -> str | dict[str, Any]:
        if config.sandbox_provider_api_key_file is None:
            return config.sandbox_provider
        api_key = config.sandbox_provider_api_key_file.read_text().strip()
        if not api_key:
            raise ValueError(f"sandbox provider API key file is empty: {config.sandbox_provider_api_key_file}")
        if isinstance(config.sandbox_provider, str):
            provider = copy.deepcopy(named_configs[config.sandbox_provider])
        else:
            provider = copy.deepcopy(config.sandbox_provider)
        provider_config = resolve_provider_config(provider)
        provider_name = next(iter(provider_config))
        provider[provider_name].setdefault("connection", {})["api_key"] = api_key
        return provider

    def model_post_init(self, context: Any) -> None:
        self._sem = Semaphore(self.config.concurrency)
        self._load_sandbox_image_auth(self.config)
        sandbox_provider = self._load_sandbox_provider_api_key(
            self.config,
            self.server_client.global_config_dict,
        )
        model_url = self.config.sandbox_model_base_url or ""
        if self.config.model_server is not None and not model_url:
            model_config = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            model_url = self.server_client._build_server_base_url(model_config)
        model_url = self._sandbox_model_url(model_url, self.config.sandbox_model_resolve_host)

        agent_deps_archive = None
        agent_deps_url = None
        if self.config.agent_runtime_source == "auto":
            deps_dir = AgentRuntime().setup()
            agent_deps_archive = Path(__file__).parent / f".{deps_dir.name}.tar.gz"
            sentinel = deps_dir / ".installed"
            if not agent_deps_archive.exists() or agent_deps_archive.stat().st_mtime < sentinel.stat().st_mtime:
                temporary = agent_deps_archive.with_suffix(".tmp")
                with tarfile.open(temporary, "w:gz") as archive:
                    archive.add(deps_dir, arcname=".")
                temporary.replace(agent_deps_archive)
        elif self.config.agent_runtime_source != "baked":
            if "://" in self.config.agent_runtime_source:
                agent_deps_url = self.config.agent_runtime_source
            else:
                agent_deps_archive = Path(self.config.agent_runtime_source).expanduser().resolve()
                if not agent_deps_archive.is_file():
                    raise ValueError(f"agent runtime archive not found: {agent_deps_archive}")

        session = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self._server = HermesBenchServerConfig(
            run_session_id=session,
            base_results_dir=Path(__file__).parent / "results" / session,
            model_server_url=model_url,
            resolved_sandbox_provider=resolve_provider_config(
                sandbox_provider,
                self.server_client.global_config_dict,
            ),
            sandbox_default_metadata=resolve_provider_metadata(
                sandbox_provider,
                self.server_client.global_config_dict,
            ),
            agent_deps_archive=agent_deps_archive,
            agent_deps_url=agent_deps_url,
        )
        super().model_post_init(context)

    @staticmethod
    def _merge_spec(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = {**base, **override}
        for key in ("env", "metadata", "provider_options", "resources", "files"):
            if key in base or key in override:
                merged[key] = {**base.get(key, {}), **override.get(key, {})}
        return merged

    @staticmethod
    def _task_timeout_s(params: HermesBenchInstanceConfig) -> int:
        base = int(params.task.get("timeout_seconds") or params.agent_timeout_s)
        return int(base * params.timeout_multiplier) + 30

    def _sandbox_spec(self, params: HermesBenchInstanceConfig, phase: str) -> SandboxSpec:
        config = dict(params.sandbox_spec)
        if phase == "grader":
            config = self._merge_spec(config, params.grader_sandbox_spec)
        configured_image = config.pop("image", None)
        image = params.task.get("image") or configured_image
        if not image:
            raise ValueError("HermesBench requires sandbox_spec.image or task metadata image")
        metadata = {
            **params.sandbox_default_metadata,
            **config.pop("metadata", {}),
            "nemo_gym_agent": "hermes_bench",
            "task_id": params.task_id[:63],
            "phase": phase,
        }
        resources = SandboxResources.from_mapping(config.pop("resources", {}))
        spec = SandboxSpec(
            image=str(image),
            ttl_s=config.pop("ttl_s", self._task_timeout_s(params) + params.grader_timeout_s + 900),
            ready_timeout_s=config.pop("ready_timeout_s", 600),
            workdir=config.pop("workdir", "/sandbox/workspace"),
            env={str(key): str(value) for key, value in config.pop("env", {}).items()},
            files={str(key): str(value) for key, value in config.pop("files", {}).items()},
            metadata={str(key): str(value) for key, value in metadata.items()},
            resources=resources,
            entrypoint=config.pop("entrypoint", None),
            provider_options=config.pop("provider_options", {}),
        )
        if config:
            raise ValueError(f"unknown sandbox_spec fields: {sorted(config)}")
        return spec

    @staticmethod
    def _resolve_archive(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PARENT_DIR / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _setup_params(self, body: NeMoGymResponseCreateParamsNonStreaming) -> HermesBenchInstanceConfig:
        metadata = body.metadata or {}
        task = json.loads(metadata["task"])
        task_id = task["task_id"]
        persistent_dir = self._server.base_results_dir / f"{task_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        persistent_dir.mkdir(parents=True, exist_ok=True)
        params = HermesBenchInstanceConfig(
            **self.config.model_dump(),
            **self._server.model_dump(),
            task=task,
            body=body,
            persistent_dir=persistent_dir,
            fixture_archive=self._resolve_archive(metadata["fixture_archive"]),
            support_archive=self._resolve_archive(metadata["support_archive"]),
        )
        (persistent_dir / "body.json").write_text(body.model_dump_json())
        (persistent_dir / "task.json").write_text(json.dumps(task))
        return params

    @staticmethod
    def _agent_env(params: HermesBenchInstanceConfig) -> dict[str, str]:
        encoded_kwargs = base64.b64encode(json.dumps(params.agent_kwargs).encode()).decode()
        return {
            "NGHB_AGENT_KWARGS_B64": encoded_kwargs,
            "NGHB_MODEL_URL": params.model_server_url,
            "NGHB_MODEL_NAME": params.body.model or params.model_server.name,
            "NGHB_WORKSPACE": "/sandbox/workspace",
            # Apptainer forwards the parent process environment by default, so a
            # leaked host TERMINAL_CWD (e.g. /home/<user>, which apptainer also
            # auto-binds) would silently redirect the agent's relative-path file
            # writes to the HOST home dir instead of the workspace. Pin it.
            "TERMINAL_CWD": "/sandbox/workspace",
            "NGHB_EXTERNAL_RUNTIME": "1" if params.agent_deps_archive or params.agent_deps_url else "0",
            "NGHB_PRESERVE_HERMES_BEHAVIOR": "1" if params.preserve_hermes_behavior else "0",
        }

    async def _seed_sandbox(self, sandbox: AsyncSandbox, params: HermesBenchInstanceConfig) -> None:
        _check(
            await sandbox.exec(
                "mkdir -p /sandbox/control /sandbox/workspace /sandbox/benchmark /sandbox/agent_deps",
                cwd="/",
                timeout_s=30,
                user="root",
            ),
            "sandbox directory setup",
        )
        uploads = [
            (params.fixture_archive, "/sandbox/control/fixture.tar.gz"),
            (params.support_archive, "/sandbox/control/support.tar.gz"),
            (params.persistent_dir / "task.json", "/sandbox/control/task.json"),
        ]
        for source, target in uploads:
            await sandbox.upload(source, target)
        _check(
            await sandbox.exec(
                "tar -xzf /sandbox/control/fixture.tar.gz -C /sandbox && tar -xzf /sandbox/control/support.tar.gz -C /sandbox",
                timeout_s=300,
                user="root",
            ),
            "fixture extraction",
        )

    async def _run_agent(self, params: HermesBenchInstanceConfig) -> NeMoGymResponse:
        sandbox = AsyncSandbox(params.resolved_sandbox_provider, self._sandbox_spec(params, "agent"))
        started = time.time()
        result = None
        try:
            await sandbox.start()
            await self._seed_sandbox(sandbox, params)
            await sandbox.upload(params.persistent_dir / "body.json", "/sandbox/control/body.json")
            await sandbox.upload(Path(__file__).with_name("agent_runner.py"), "/sandbox/control/agent_runner.py")
            setup_script = params.task.get("setup_script")
            if setup_script:
                setup_path = params.persistent_dir / "setup.sh"
                setup_path.write_text(setup_script)
                await sandbox.upload(setup_path, "/sandbox/control/setup.sh")
                _check(
                    await sandbox.exec(
                        "HERMESBENCH_REPO_ROOT=/sandbox/benchmark bash /sandbox/control/setup.sh",
                        cwd="/sandbox/workspace",
                        timeout_s=300,
                    ),
                    "task setup",
                )
            external_runtime = params.agent_deps_archive is not None or params.agent_deps_url is not None
            agent_python = params.agent_python
            if params.agent_deps_archive is not None:
                await sandbox.upload(params.agent_deps_archive, "/sandbox/control/agent_deps.tar.gz")
                _check(
                    await sandbox.exec(
                        "tar -xzf /sandbox/control/agent_deps.tar.gz -C /sandbox/agent_deps",
                        timeout_s=600,
                        user="root",
                    ),
                    "agent runtime extraction",
                )
                agent_python = "/sandbox/agent_deps/bin/python"
            elif params.agent_deps_url is not None:
                _check(
                    await sandbox.exec(
                        f"curl -fsSL {json.dumps(params.agent_deps_url)} | tar -xzf - -C /sandbox/agent_deps",
                        timeout_s=600,
                        user="root",
                    ),
                    "agent runtime download",
                )
                agent_python = "/sandbox/agent_deps/bin/python"
            env = self._agent_env(params)
            env["NGHB_EXTERNAL_RUNTIME"] = "1" if external_runtime else "0"
            command = "\n".join(
                [
                    "set -a",
                    "if [ -f .hb_setup_env ]; then . ./.hb_setup_env; fi",
                    "set +a",
                    f"exec {agent_python} /sandbox/control/agent_runner.py",
                ]
            )
            result = await sandbox.exec(
                command,
                cwd="/sandbox/workspace",
                env=env,
                timeout_s=self._task_timeout_s(params),
            )
            _check(
                await sandbox.exec(
                    "tar -czf /sandbox/control/workspace.tar.gz -C /sandbox workspace",
                    timeout_s=300,
                ),
                "workspace archive",
            )
            await sandbox.download("/sandbox/control/workspace.tar.gz", params.persistent_dir / "workspace.tar.gz")
            try:
                await sandbox.download("/sandbox/control/response.json", params.persistent_dir / "response.json")
            except Exception:
                if result.return_code == 0:
                    raise
                response = NeMoGymResponse(
                    id=f"hermes-bench-{params.task_id}",
                    created_at=int(time.time()),
                    model=params.body.model or "unknown",
                    object="response",
                    output=[],
                    parallel_tool_calls=bool(params.body.parallel_tool_calls),
                    tool_choice=params.body.tool_choice or "auto",
                    tools=params.body.tools or [],
                )
                (params.persistent_dir / "response.json").write_text(response.model_dump_json())
        finally:
            await sandbox.stop()
        return_code = int(result.return_code)
        stdout = (result.stdout or "")[-2000:]
        stderr = (result.stderr or "")[-2000:]
        agent_error = None
        if return_code != 0:
            agent_error = f"agent run exited {return_code}: {stdout}{stderr}"
        metrics = {
            "agent_run_time": time.time() - started,
            "agent_timed_out": result.error_type == "timeout" or return_code == -1,
            "agent_return_code": return_code,
            "agent_error": agent_error,
        }
        (params.persistent_dir / "agent_metrics.json").write_text(json.dumps(metrics))
        return NeMoGymResponse.model_validate_json((params.persistent_dir / "response.json").read_text())

    async def _run_grader(self, params: HermesBenchInstanceConfig) -> HermesBenchMetrics:
        sandbox = AsyncSandbox(params.resolved_sandbox_provider, self._sandbox_spec(params, "grader"))
        started = time.time()
        grade_path = params.persistent_dir / "grade.json"
        try:
            await sandbox.start()
            await self._seed_sandbox(sandbox, params)
            await sandbox.upload(params.persistent_dir / "workspace.tar.gz", "/sandbox/control/workspace.tar.gz")
            await sandbox.upload(params.persistent_dir / "response.json", "/sandbox/control/response.json")
            await sandbox.upload(params.persistent_dir / "agent_metrics.json", "/sandbox/control/agent_metrics.json")
            await sandbox.upload(Path(__file__).with_name("grader_runner.py"), "/sandbox/control/grader_runner.py")
            await sandbox.upload(Path(__file__).with_name("p5_direct.py"), "/sandbox/control/p5_direct.py")
            _check(
                await sandbox.exec(
                    "rm -rf /sandbox/workspace && tar -xzf /sandbox/control/workspace.tar.gz -C /sandbox",
                    timeout_s=300,
                    user="root",
                ),
                "grader workspace restore",
            )
            secret_env = {}
            for name, path in params.grader_env_files.items():
                value = path.read_text().strip()
                if not value:
                    raise ValueError(f"grader environment file is empty: {path}")
                secret_env[name] = value
            env = {
                **params.grader_env,
                **secret_env,
                "P5_RENDER_DIR": "/sandbox/benchmark/scripts/p5_render",
                "HERMES_JUDGE_MODEL": params.judge_model or "",
            }
            result = await sandbox.exec(
                f"{params.grader_python} /sandbox/control/grader_runner.py",
                cwd="/sandbox/workspace",
                env=env,
                timeout_s=params.grader_timeout_s,
            )
            for attempt in range(3):
                try:
                    await sandbox.download("/sandbox/control/grade.json", grade_path)
                    break
                except Exception:
                    if attempt == 2:
                        _check(result, "grader run")
                        raise
                    await sleep(1)

            artifact_result = await sandbox.exec(
                "\n".join(
                    [
                        "items=''",
                        '[ -d /sandbox/control/architecture ] && items="$items control/architecture"',
                        '[ -d /sandbox/workspace/.judge_artifacts ] && items="$items workspace/.judge_artifacts"',
                        '[ -n "$items" ] && tar -czf /sandbox/control/grader_artifacts.tar.gz -C /sandbox $items',
                    ]
                ),
                timeout_s=300,
            )
            if artifact_result.return_code == 0:
                try:
                    await sandbox.download(
                        "/sandbox/control/grader_artifacts.tar.gz",
                        params.persistent_dir / "grader_artifacts.tar.gz",
                    )
                except Exception:
                    pass
        finally:
            await sandbox.stop()
        grade = json.loads(grade_path.read_text())
        agent_metrics = json.loads((params.persistent_dir / "agent_metrics.json").read_text())
        return HermesBenchMetrics(
            task_id=params.task_id,
            reward=float(grade.get("reward", 0.0)),
            score_valid=bool(grade.get("score_valid")),
            grading_status=str(grade.get("grading_status", "grading_error")),
            scores=grade.get("scores") or {},
            error=grade.get("error"),
            mask_sample=not bool(grade.get("score_valid")),
            grader_run_time=time.time() - started,
            **agent_metrics,
        )

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        params = self._setup_params(body)
        (params.persistent_dir / "params.json").write_text(json.dumps(_safe_config(params), indent=2))
        try:
            response = await self._run_agent(params)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            (params.persistent_dir / "agent_error.txt").write_text(message)
            response = NeMoGymResponse(
                id=f"hermes-bench-{params.task_id}",
                created_at=int(time.time()),
                model=params.body.model or "unknown",
                object="response",
                output=[],
                parallel_tool_calls=bool(params.body.parallel_tool_calls),
                tool_choice=params.body.tool_choice or "auto",
                tools=params.body.tools or [],
            )
            metrics = HermesBenchMetrics(
                task_id=params.task_id,
                reward=0.0,
                score_valid=False,
                grading_status="agent_error",
                error=message,
                agent_timed_out="timeout" in message.lower() or "timed out" in message.lower(),
                mask_sample=True,
            )
        else:
            try:
                metrics = await self._run_grader(params)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                (params.persistent_dir / "grader_error.txt").write_text(message)
                agent_metrics = json.loads((params.persistent_dir / "agent_metrics.json").read_text())
                metrics = HermesBenchMetrics(
                    task_id=params.task_id,
                    reward=0.0,
                    score_valid=False,
                    grading_status="grading_error",
                    error=message,
                    mask_sample=True,
                    **agent_metrics,
                )
        response.metadata = {
            "hermes_bench_metrics": metrics.model_dump_json(),
            "hermes_bench_instance": json.dumps(_safe_config(params)),
        }
        return response

    async def run(self, body: BaseRunRequest) -> HermesBenchVerifyResponse:
        async with self._sem:
            response = await self.responses(body.responses_create_params)
            metadata = response.metadata or {}
            metrics = HermesBenchMetrics.model_validate_json(metadata["hermes_bench_metrics"])
            instance_config = json.loads(metadata["hermes_bench_instance"])
            response.metadata = None
            return HermesBenchVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=response,
                reward=metrics.reward,
                instance_config=instance_config,
                **metrics.model_dump(exclude={"reward"}),
            )


if __name__ == "__main__":
    HermesBenchAgent.run_webserver()
