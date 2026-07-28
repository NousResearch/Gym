# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import re
import sys
from asyncio import Semaphore
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import ray
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class HarborDatasetSourceConfig(BaseModel):
    local_dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    workdir: Optional[str] = None


class HarborAgentConfig(BaseResponsesAPIAgentConfig):
    concurrency: int

    # --- Harbor agent settings ---
    # Name of a built-in Harbor agent (e.g. "terminus-2", "claude-code", "aider").
    harbor_agent_name: Optional[str] = "terminus-2"
    # Python import path for a custom agent class (e.g. "my_pkg.my_mod:MyAgent").
    # Overrides harbor_agent_name when set.
    harbor_agent_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor AgentConfig (e.g. collect_rollout_details,
    # model_info). See harbor_agent.yaml for examples.
    harbor_agent_kwargs: Optional[dict[str, Any]] = None
    # Environment forwarded to installed Harbor agents. This is primarily for
    # agents that run inside the task container and therefore cannot reach a
    # Gym model server advertised on the host's loopback interface.
    harbor_agent_env: Optional[dict[str, str]] = None

    # --- Dataset routing ---
    # Map of dataset aliases to source definitions. Each alias must define exactly
    # one source:
    # 1) local: {"local_dataset_path": "..."}
    # 2) registry: {"dataset_name": "...", "dataset_version": "..."} (version optional)
    # Requests must provide instance_id in the form "<dataset_alias>::<task_name>".
    harbor_datasets: dict[str, HarborDatasetSourceConfig]

    # --- Environment ---
    # Harbor environment type: "singularity", "docker", "daytona", "modal", etc.
    harbor_environment_type: Optional[str] = "singularity"
    # Python import path for a custom environment class (e.g. "my_pkg.my_mod:MyEnv").
    # Overrides harbor_environment_type when set.
    harbor_environment_import_path: Optional[str] = None
    # Extra kwargs forwarded to the Harbor EnvironmentConfig (e.g.
    # singularity_image_cache_dir, singularity_force_pull).
    harbor_environment_kwargs: Optional[dict[str, Any]] = None

    # --- Timeouts ---
    # Override agent timeout (seconds). Replaces the task's own timeout entirely.
    # Use this to set a fixed timeout for all tasks regardless of task.toml.
    harbor_agent_override_timeout: Optional[int] = None
    # Cap agent timeout (seconds). Uses the task's own timeout but clamps it
    # to this maximum. Respects shorter per-task timeouts unlike harbor_agent_override_timeout.
    harbor_agent_max_timeout: Optional[int] = None
    # Override verifier timeout (seconds). Replaces the task's own verifier timeout.
    harbor_verifier_override_timeout: Optional[int] = None
    # Cap verifier timeout (seconds). Uses the task's own verifier timeout but
    # clamps it to this maximum.
    harbor_verifier_max_timeout: Optional[int] = None
    # Multiplier applied to all Harbor timeouts after override/cap. None = 1.0.
    harbor_timeout_multiplier: Optional[float] = None

    # --- Job output ---
    # Directory where Harbor writes job results and trial artifacts.
    harbor_jobs_dir: str = "jobs"

    # Keep Docker runtime images between trials (maps to EnvironmentConfig.delete=False).
    harbor_no_delete: bool = True

    # --- Model routing ---
    # NeMo Gym model server reference used to resolve Harbor model base URL.
    model_server: ModelServerRef


class HarborRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    instance_id: str


class HarborVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _find_trial_dir_with_result(job_dir: Path) -> Optional[Path]:
    if not job_dir.exists():
        return None
    for trial_dir in job_dir.iterdir():
        if trial_dir.is_dir() and (trial_dir / "result.json").exists():
            return trial_dir
    return None


def _trial_trajectory_paths(trial_dir: Path, trial_result: dict[str, Any]) -> list[Path]:
    """Return ATIF trajectories in execution order.

    Single-step Harbor trials write ``agent/trajectory.json``. Modern Harbor
    archives multi-step agent logs under ``steps/<step>/agent/trajectory.json``
    after each checkpoint. Prefer the ordered per-step files when present and
    retain the top-level path as a backwards-compatible fallback.
    """
    step_paths: list[Path] = []
    for step_result in trial_result.get("step_results") or []:
        step_name = step_result.get("step_name")
        if not isinstance(step_name, str) or not step_name:
            continue
        candidate = trial_dir / "steps" / step_name / "agent" / "trajectory.json"
        if candidate.exists():
            step_paths.append(candidate)

    if step_paths:
        return step_paths

    trajectory_path = trial_dir / "agent" / "trajectory.json"
    return [trajectory_path] if trajectory_path.exists() else []


def _load_trial_trajectories(
    trial_dir: Path,
    trial_result: dict[str, Any],
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for trajectory_path in _trial_trajectory_paths(trial_dir, trial_result):
        with trajectory_path.open() as f:
            trajectories.append(json.load(f))
    return trajectories


def _load_agent_error_flags(
    trial_dir: Path,
    trial_result: dict[str, Any],
) -> dict[str, bool]:
    """OR agent error flags across a single- or multi-step trial."""
    agent_dirs: list[Path] = []
    step_results = trial_result.get("step_results") or []
    for step_result in step_results:
        step_name = step_result.get("step_name")
        if isinstance(step_name, str) and step_name:
            agent_dirs.append(trial_dir / "steps" / step_name / "agent")
    if not agent_dirs:
        agent_dirs.append(trial_dir / "agent")

    combined: dict[str, bool] = {}
    for agent_dir in agent_dirs:
        flags_path = agent_dir / "agent_error_flags.json"
        if not flags_path.exists():
            continue
        with flags_path.open() as f:
            flags = json.load(f)
        for key, value in flags.items():
            combined[key] = combined.get(key, False) or bool(value)
    return combined


async def run_harbor_job(job_config_dict: dict) -> str:
    """Runs a single Harbor Job and returns the *absolute* trial directory path.

    The trial directory contains:
    - result.json: Summary result with reward, agent_result, verifier_result, etc.
    - agent/trajectory.json: Full ATIF trajectory for a single-step task.
    - steps/<step>/agent/trajectory.json: One ATIF trajectory per checkpoint
      for a multi-step task.

    Harbor writes result.json and trajectory.json to disk even when the trial
    fails (e.g. verifier timeout, reward file not found, OOM).  We recover the
    trial directory after an exception so the caller can still use the partial
    trajectory for training.

    This runs inside a Ray remote worker (see ``runner_ray_remote``), whose cwd
    is inherited from wherever the Ray cluster was started (the Gym repo root,
    via ``ng_run``) -- NOT the `harbor_agent` server process's own cwd (which
    `ng_run` `cd`s into `responses_api_agents/harbor_agent/` before launching).
    `config.jobs_dir` is a relative path (e.g. `responses_api_agents/harbor_agent/
    jobs/...`), so it only resolves correctly from the *former*. We therefore
    MUST resolve to an absolute path before returning, since the caller
    (`HarborAgent.run()`) re-opens `result.json` from the server process's own
    cwd -- a relative path here silently resolves to the wrong (nonexistent,
    doubled) location there, and every trial looks like it failed even though
    it ran and wrote everything correctly to disk.
    """
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    config = JobConfig(**job_config_dict)
    job = await Job.create(config)

    job_error = None
    try:
        await job.run()
    except Exception as e:
        job_error = e

    # Find the trial directory from the job output directory. Harbor writes
    # result.json before propagating most exceptions, so we can usually
    # recover the trial even when job.run() raised.
    job_dir = config.jobs_dir / config.job_name
    trial_dir = _find_trial_dir_with_result(job_dir)
    if trial_dir is not None:
        return str(trial_dir.resolve())

    # No trial directory found — re-raise the original error if there was one,
    # otherwise raise FileNotFoundError.
    if job_error is not None:
        raise job_error
    raise FileNotFoundError(f"No trial result found in {job_dir.resolve()}")


_RAY_WORKER_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_harbor_job_sync(job_config_dict: dict) -> str:
    """Synchronous wrapper for run_harbor_job for use in Ray remote.

    Ray workers are long-lived processes. Reusing a single event loop per worker
    avoids cross-loop issues with global async state (e.g., LiteLLM logging worker
    queues) when multiple jobs execute sequentially in the same process.
    """
    global _RAY_WORKER_EVENT_LOOP
    if _RAY_WORKER_EVENT_LOOP is None or _RAY_WORKER_EVENT_LOOP.is_closed():
        _RAY_WORKER_EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_RAY_WORKER_EVENT_LOOP)
    return _RAY_WORKER_EVENT_LOOP.run_until_complete(run_harbor_job(job_config_dict))


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


class HarborAgent(SimpleResponsesAPIAgent):
    config: HarborAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        # Registered explicitly because this override replaces (rather than extends)
        # SimpleResponsesAPIAgent.setup_webserver(), which would otherwise add this route
        # by default. HarborAgent doesn't override compute_metrics/get_key_metrics, so this
        # uses AggregateMetricsMixin's generic reward-based aggregation (mean/max/min/median/std
        # via RewardProfiler) over the `reward` field already present on every /run response.
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Aggregate Harbor's named checkpoint rewards for multi-step evals.

        Overall ``reward`` remains the benchmark's primary scalar. These
        diagnostics make checkpoint reachability and conditional checkpoint
        scores visible instead of hiding them inside result metadata.
        """
        rollouts = [rollout for task in tasks for rollout in task]
        rollout_count = len(rollouts)
        reached_by_step: dict[str, int] = {}
        rewards_by_step: dict[tuple[str, str], list[float]] = {}

        for rollout in rollouts:
            metadata = rollout.get("metadata") or {}
            for step_result in metadata.get("step_results") or []:
                step_name = step_result.get("step_name")
                if not isinstance(step_name, str) or not step_name:
                    continue
                reached_by_step[step_name] = reached_by_step.get(step_name, 0) + 1
                verifier_result = step_result.get("verifier_result") or {}
                for reward_name, reward_value in (verifier_result.get("rewards") or {}).items():
                    if isinstance(reward_value, (int, float)):
                        rewards_by_step.setdefault((step_name, reward_name), []).append(float(reward_value))

        metrics: dict[str, Any] = {}
        for step_name, reached_count in sorted(reached_by_step.items()):
            metrics[f"step/{step_name}/reached_count"] = reached_count
            metrics[f"step/{step_name}/reached_rate"] = reached_count / rollout_count if rollout_count else 0.0
        for (step_name, reward_name), values in sorted(rewards_by_step.items()):
            metrics[f"step/{step_name}/mean_when_reached/{reward_name}"] = sum(values) / len(values)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key_metrics = super().get_key_metrics(agent_metrics)
        key_metrics.update({key: value for key, value in agent_metrics.items() if key.startswith("step/")})
        return key_metrics

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: HarborRunRequest) -> HarborVerifyResponse:
        async with self.sem:
            global_config_dict = get_global_config_dict()

            policy_model_name = global_config_dict["policy_model_name"]
            base_url = self._resolve_model_base_url(global_config_dict)
            run_timestamp = datetime.now(timezone.utc)
            run_id = self._build_run_id(run_timestamp)

            instance_id = body.instance_id
            dataset_alias, task_name = self._parse_instance_id(instance_id)

            output_file_dir = self._get_results_output_dir(policy_model_name, dataset_alias, run_timestamp)
            jobs_dir = self._get_jobs_output_dir(policy_model_name, dataset_alias, run_timestamp)
            job_name = self._build_job_name(run_id)

            responses_create_params = body.responses_create_params.model_dump(
                exclude_unset=True,
                exclude_none=True,
            )

            job_config_dict = self._build_job_config(
                dataset_alias,
                task_name,
                policy_model_name,
                base_url,
                job_name=job_name,
                jobs_dir=jobs_dir,
                responses_create_params=responses_create_params,
            )

            try:
                params = dict(
                    job_config_dict=job_config_dict,
                )
                future = runner_ray_remote.remote(_run_harbor_job_sync, params)
                trial_dir_path = await asyncio.to_thread(ray.get, future)
                trial_dir = Path(trial_dir_path)

                # Read the trial result (summary: reward, agent_result, verifier_result)
                with open(trial_dir / "result.json", "r") as f:
                    trial_result = json.load(f)

                # Modern Harbor archives one ATIF trajectory and error-flags
                # file per checkpoint for multi-step tasks.
                trajectories = _load_trial_trajectories(trial_dir, trial_result)
                agent_error_flags = _load_agent_error_flags(trial_dir, trial_result)

                # Extract reward from verifier result
                verifier_result = trial_result.get("verifier_result")
                reward = HarborAgentUtils.extract_reward(verifier_result)

                # Convert Harbor outputs to NeMo Gym response items:
                # keep rich trajectory details, then overlay rollout token details when present.
                output_items = [
                    item
                    for trajectory in trajectories
                    for item in HarborAgentUtils.trial_result_to_responses(trial_result, trajectory)
                ]

                # Preserve each checkpoint instruction in execution order.
                input_messages = [
                    message
                    for trajectory in trajectories
                    for message in HarborAgentUtils.extract_input_from_trajectory(trajectory)
                ]

                # Sum usage across checkpoint trajectories, with a result.json
                # fallback for agents that do not emit ATIF metrics.
                usage = HarborAgentUtils.extract_usage_for_trial(trial_result, trajectories)

            except Exception as e:
                print(f"Error running Harbor job: {e}")
                trial_result = None
                trajectories = []
                agent_error_flags = {}
                output_items = []
                input_messages = []
                usage = None
                reward = 0.0

            response = HarborAgentUtils.get_default_response_object()
            response["model"] = policy_model_name
            response["temperature"] = responses_create_params.get("temperature")
            response["top_p"] = responses_create_params.get("top_p")
            response["output"] = output_items
            if usage:
                response["usage"] = usage

            # Update responses_create_params with the actual input sent to the agent
            updated_params = body.responses_create_params
            if input_messages:
                updated_params = body.responses_create_params.model_copy(update={"input": input_messages})

            verify_response = HarborVerifyResponse(
                responses_create_params=updated_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=trial_result if trial_result else {},
                context_length_exceeded_error=int(agent_error_flags.get("context_length_exceeded", False)),
                memory_limit_exceeded_error=int(agent_error_flags.get("memory_limit_exceeded", False)),
                agent_timeout_error=int(
                    ((trial_result or {}).get("exception_info") or {}).get("exception_type") == "AgentTimeoutError"
                ),
            )

            # Save result to disk (folder = run_id, file = task name)
            output_path = output_file_dir / run_id
            output_path.mkdir(parents=True, exist_ok=True)

            safe_instance_id = self._sanitize_path_component(instance_id)
            with open(output_path / f"{safe_instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f, indent=2)

            return verify_response

    def _get_results_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build immutable run output directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path.cwd() / "results" / "runs" / date_key / dataset_key / model_key

    def _get_jobs_output_dir(self, policy_model_name: str, dataset_alias: str, run_timestamp: datetime) -> Path:
        """Build Harbor jobs directory grouped by date/dataset/model."""
        date_key = run_timestamp.strftime("%Y%m%d")
        dataset_key = self._sanitize_path_component(dataset_alias)
        model_key = self._sanitize_path_component(self._extract_model_name(policy_model_name))
        return Path(self.config.harbor_jobs_dir) / date_key / dataset_key / model_key

    @staticmethod
    def _parse_instance_id(instance_id: str) -> tuple[str, str]:
        """Parse instance id in the required form: <dataset_alias>::<task_name>."""
        dataset_alias, sep, task_name = instance_id.partition("::")
        dataset_alias = dataset_alias.strip()
        task_name = task_name.strip()
        if not sep or not dataset_alias or not task_name:
            raise ValueError(f"instance_id must be in the form '<dataset_alias>::<task_name>' (got: {instance_id!r})")
        return dataset_alias, task_name

    def _build_run_id(self, run_timestamp: datetime) -> str:
        """Build a compact run id (time + short hash) for immutable file naming."""
        time_key = run_timestamp.strftime("%H%M%S")
        return f"{time_key}_{uuid4().hex[:8]}"

    def _build_job_name(self, run_id: str) -> str:
        """Build a Harbor job name from run id only."""
        return run_id

    @staticmethod
    def _extract_model_name(policy_model_name: str) -> str:
        """Extract the final model name from a full path or HF-style identifier.

        '/lustre/.../nano-v3-sft-...-hf'  -> 'nano-v3-sft-...-hf'
        'Qwen/Qwen3-8B'                   -> 'Qwen3-8B'
        'my-model'                         -> 'my-model'
        """
        return Path(policy_model_name).name or policy_model_name

    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize path components to avoid accidental nested directories."""
        sanitized = value.replace("/", "__").replace("\\", "__").replace(":", "__")
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unknown"

    def _resolve_model_base_url(self, global_config_dict: Any) -> str:
        """Resolve model base URL from required model_server reference."""
        server_name = self.config.model_server.name
        model_server_config = get_first_server_config_dict(
            global_config_dict,
            server_name,
        )
        return f"http://{model_server_config['host']}:{model_server_config['port']}/v1"

    def _build_job_config(
        self,
        dataset_alias: str,
        task_name: str,
        model_name: str,
        api_base: str,
        job_name: str,
        jobs_dir: Path,
        responses_create_params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Build a Harbor JobConfig dict for a single task."""
        from harbor.models.job.config import DatasetConfig, JobConfig
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            VerifierConfig,
        )

        agent_kwargs: dict[str, Any] = {"api_base": api_base}
        if responses_create_params:
            agent_kwargs["responses_create_params"] = responses_create_params
            # Terminus-2 accepts temperature as a top-level kwarg for trajectory metadata.
            if "temperature" in responses_create_params:
                agent_kwargs["temperature"] = responses_create_params["temperature"]
        if self.config.harbor_agent_kwargs:
            agent_kwargs.update(self.config.harbor_agent_kwargs)

        agent_env = {
            "OPENAI_API_KEY": "nemo-gym-internal",
            "OPENAI_BASE_URL": api_base,
        }
        if self.config.harbor_agent_env:
            agent_env.update(self.config.harbor_agent_env)

        agent_config = AgentConfig(
            name=self.config.harbor_agent_name if not self.config.harbor_agent_import_path else None,
            import_path=self.config.harbor_agent_import_path,
            model_name=model_name,
            # Harbor's installed agents (including ``hermes``) discover an
            # OpenAI-compatible model endpoint through environment variables,
            # whereas custom Gym-native agents consume ``api_base`` from
            # ``kwargs`` above. Point both paths at the same internal Gym model
            # server. The server does not require authentication, but clients
            # expect a non-empty key to select the OpenAI-compatible provider.
            env=agent_env,
            override_timeout_sec=(
                float(self.config.harbor_agent_override_timeout)
                if self.config.harbor_agent_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_agent_max_timeout)
                if self.config.harbor_agent_max_timeout is not None
                else None
            ),
            kwargs=agent_kwargs,
        )

        dataset_source = self.config.harbor_datasets.get(dataset_alias)
        if dataset_source is None:
            available = ", ".join(sorted(self.config.harbor_datasets.keys()))
            raise ValueError(
                f"Unknown dataset alias in instance_id: {dataset_alias!r}. Available aliases: [{available}]"
            )

        has_local = bool(dataset_source.local_dataset_path)
        has_registry = bool(dataset_source.dataset_name)
        if has_local == has_registry:
            raise ValueError(
                f"Dataset alias {dataset_alias!r} must define exactly one source: "
                "local_dataset_path OR dataset_name[/dataset_version]."
            )

        environment_kwargs = {}
        if self.config.harbor_environment_kwargs:
            environment_kwargs.update(self.config.harbor_environment_kwargs)
        # Dataset alias-level workdir overrides global harbor_environment_kwargs.workdir.
        if dataset_source.workdir is not None:
            environment_kwargs["workdir"] = dataset_source.workdir

        environment_config = EnvironmentConfig(
            type=self.config.harbor_environment_type if not self.config.harbor_environment_import_path else None,
            import_path=self.config.harbor_environment_import_path,
            delete=not self.config.harbor_no_delete,
            kwargs=environment_kwargs,
        )

        verifier_config = VerifierConfig(
            override_timeout_sec=(
                float(self.config.harbor_verifier_override_timeout)
                if self.config.harbor_verifier_override_timeout is not None
                else None
            ),
            max_timeout_sec=(
                float(self.config.harbor_verifier_max_timeout)
                if self.config.harbor_verifier_max_timeout is not None
                else None
            ),
        )

        if has_registry:
            dataset_config = DatasetConfig(
                name=dataset_source.dataset_name,
                version=dataset_source.dataset_version,
                task_names=[task_name],
            )
        else:
            dataset_config = DatasetConfig(
                path=Path(dataset_source.local_dataset_path),
                task_names=[task_name],
            )

        job_config = JobConfig(
            job_name=job_name,
            jobs_dir=jobs_dir,
            timeout_multiplier=(
                self.config.harbor_timeout_multiplier if self.config.harbor_timeout_multiplier is not None else 1.0
            ),
            n_concurrent_trials=1,
            quiet=True,
            environment=environment_config,
            verifier=verifier_config,
            agents=[agent_config],
            datasets=[dataset_config],
        )

        # This dump is an in-memory transport across the Ray boundary, not a
        # persistence/logging boundary. Preserve credentials here so the
        # reconstructed JobConfig receives the values the caller supplied;
        # Harbor still redacts sensitive env values when it persists artifacts.
        return job_config.model_dump(
            mode="json",
            context={"redact_sensitive_env": False},
        )


if __name__ == "__main__":
    HarborAgent.run_webserver()
