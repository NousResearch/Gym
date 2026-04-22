# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RunHermesAgent -- SWE-bench agent using hermes-agent inside Apptainer containers.

Inherits container execution and evaluation infrastructure from RunOpenHandsAgent.
Hermes-agent is pre-built in a setup directory (like OpenHands) and mounted into
the SWE-bench container at runtime. Token IDs and logprobs are captured by
HermesAgentLoop and written to the output trajectory.
"""

import json
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from responses_api_agents.swe_agents.run_openhands import (
    RunOpenHandsAgent,
    SweBenchGenerationConfig,
)


# ---------------------------------------------------------------------------
# In-container runner script
# ---------------------------------------------------------------------------

HERMES_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""
Standalone hermes-agent SWE runner for inside Apptainer containers.

Uses AIAgent.run_conversation() with terminal + file tools against the testbed
repo. Token IDs and logprobs are captured by run_agent.py's conversation loop
and included in the message history.
"""

import argparse
import json
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hermes_swe_runner")

SYSTEM_PROMPT = """\
You are an expert software engineer tasked with fixing a bug in a code repository.
The repository is located at /testbed and is checked out at the relevant commit.

Use the terminal and file tools to:
1. Understand the problem by reading relevant source code
2. Locate the buggy code
3. Implement a minimal, targeted fix
4. Verify your fix does not break anything (run relevant tests if possible)

When you are done, make sure all your changes are saved. Do NOT commit.

Important:
- Focus on minimal, targeted fixes — do not refactor unrelated code
- Do not modify test files unless the problem specifically requires it
- Work within /testbed
"""


def run(args):
    from run_agent import AIAgent

    agent = AIAgent(
        base_url=args.model_base_url,
        model=args.model,
        api_key="***",
        max_iterations=args.max_turns,
        save_trajectories=True,
        use_streaming=False,
        temperature=1.0,
        insert_reasoning=False,
    )

    # Disable context compression for RL training — compression rewrites
    # earlier messages, which invalidates prompt token IDs from those turns.
    agent.compression_enabled = False

    user_msg = (
        f"Please fix the following issue in the repository at /testbed.\n\n"
        f"Instance ID: {args.instance_id}\n\n"
        f"## Problem Statement\n\n"
        f"{args.problem_statement}"
    )

    result = agent.run_conversation(
        user_message=user_msg,
        system_message=SYSTEM_PROMPT,
    )

    # Extract git diff
    try:
        proc = subprocess.run(
            ["git", "diff"], cwd="/testbed",
            capture_output=True, text=True, timeout=30,
        )
        patch = proc.stdout.strip() or None
    except Exception as e:
        logger.error("Failed to extract patch: %s", e)
        patch = None

    # Write output in SWE-bench evaluation format
    output = {
        "model_name_or_path": args.model,
        "instance_id": args.instance_id,
        "model_patch": (patch + "\n") if patch and not patch.endswith("\n") else patch,
    }
    with open(args.output_file, "w") as f:
        f.write(json.dumps(output))

    # Write full message history (includes token IDs / logprobs per-message)
    traj_file = args.output_file.replace(".jsonl", ".trajectory.json")

    # Include the full system prompt (persona, memory, skills, context files,
    # tool schemas — everything the model actually saw) for debugging
    full_messages = result.get("messages", [])
    system_prompt = getattr(agent, "_cached_system_prompt", None)
    if system_prompt:
        full_messages = [{"role": "system", "content": system_prompt}] + full_messages

    traj_data = {
        "messages": full_messages,
        "completed": result.get("completed", False),
        "api_calls": result.get("api_calls", 0),
    }
    with open(traj_file, "w") as f:
        json.dump(traj_data, f, default=str)

    logger.info(
        "Agent finished: %d api calls, completed=%s, patch=%s",
        result.get("api_calls", 0), result.get("completed"), "yes" if patch else "no",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--problem-statement", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# RunHermesAgent
# ---------------------------------------------------------------------------


@dataclass
class RunHermesAgent(RunOpenHandsAgent):
    """
    Runs hermes-agent on SWE-bench instances inside Apptainer containers.

    Inherits container execution, container finding, and SWE-bench evaluation
    from RunOpenHandsAgent. Overrides the agent execution step to use
    hermes-agent's HermesAgentLoop instead of OpenHands/SWE-agent.

    hermes-agent is pre-built in hermes_setup_dir and mounted read-only
    into the container at /hermes_setup.
    """

    hermes_setup_dir: Path | None = None

    # Path to ~/.hermes (or equivalent) to mount into the container.
    # Provides config.yaml, .env (API keys), memory, skills — the full
    # user environment so the agent behaves identically to a real session.
    hermes_home_dir: Path | None = None

    def _get_extra_agent_mounts(self) -> list[str]:
        """Return additional Apptainer mount arguments for hermes agent mode."""
        mounts = []

        # Mount the in-container hermes-agent build (/opt/hermes-agent)
        # and its uv-managed Python so the venv's symlinks resolve.
        hermes_install = Path("/opt/hermes-agent")
        if hermes_install.exists():
            mounts.append(
                f"--mount type=bind,src={hermes_install},dst={hermes_install},ro"
            )
        elif self.hermes_setup_dir:
            mounts.append(
                f"--mount type=bind,src={self.hermes_setup_dir},dst=/hermes_setup,ro"
            )
            mounts.append(
                f"--mount type=bind,src={self.hermes_setup_dir},dst={self.hermes_setup_dir},ro"
            )

        # Mount the uv-managed Python directory so venv symlinks resolve
        uv_python_dir = Path("/root/.local/share/uv/python")
        if uv_python_dir.exists():
            mounts.append(
                f"--mount type=bind,src={uv_python_dir},dst={uv_python_dir},ro"
            )
        # Fallback: check home-relative path
        uv_python_dir2 = Path.home() / ".local" / "share" / "uv" / "python"
        if uv_python_dir2.exists() and uv_python_dir2 != uv_python_dir:
            mounts.append(
                f"--mount type=bind,src={uv_python_dir2},dst={uv_python_dir2},ro"
            )
        if self.hermes_home_dir:
            # Mount the user's ~/.hermes into the container so config.yaml,
            # .env (API keys), memory, and skills are all available.
            mounts.append(
                f"--mount type=bind,src={self.hermes_home_dir},dst=/root/.hermes"
            )
        return mounts

    async def _execute_container_command(
        self,
        data_point,
        command,
        expected_file_pattern,
        mode,
        max_retries=2,
        timeout=45 * 60,
        dataset_mount_path=None,
    ):
        """Override to add hermes-specific mounts for agent mode.

        For eval mode, delegates to parent unchanged.
        For agent mode, temporarily switches framework to swe_agent (to skip
        OpenHands mount logic in the parent) and injects hermes mounts by
        including them in the apptainer command via environment.
        """
        if mode != "agent" or not self.hermes_setup_dir:
            # Eval mode or no hermes setup — use parent directly
            return await super()._execute_container_command(
                data_point, command, expected_file_pattern,
                mode, max_retries, timeout, dataset_mount_path,
            )

        # Agent mode: we need hermes mounts. The parent's mount logic
        # is keyed on agent_framework, so we temporarily set it to swe_agent
        # (which adds no framework-specific mounts) and inject hermes mounts
        # by monkey-patching the container command.
        #
        # This is admittedly a hack; a cleaner approach would be to refactor
        # the parent to accept extra_mounts. But this keeps the parent untouched.
        import asyncio
        import glob
        import shlex as _shlex

        container_name = self._find_container(data_point)
        dataset_path_to_mount = dataset_mount_path or self.dataset_path
        if dataset_path_to_mount is None:
            raise ValueError("Dataset path is not set")
        dataset_path_to_mount = str(dataset_path_to_mount)

        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{data_point['instance_id']}_agent.log"

        container_commands = ["echo '127.0.0.1 localhost' >/etc/hosts"]
        container_commands.append(command)
        combined_command = " && ".join(container_commands)

        mount_args = [
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount",
        ]
        mount_args.extend(self._get_extra_agent_mounts())

        mount_str = " ".join(mount_args)

        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --no-mount home,tmp,bind-paths "
            f"{mount_str} "
            f" {container_name} bash -c {_shlex.quote(combined_command)}"
        )
        memory_limit_mb = self.cfg.apptainer_memory_limit_mb
        if memory_limit_mb is not None and memory_limit_mb > 0:
            memory_limit_kb = int(memory_limit_mb) * 1024
            apptainer_cmd = f"ulimit -v {memory_limit_kb} && {apptainer_cmd}"

        for attempt in range(max_retries):
            try:
                with open(log_file_path, "w") as log_file:
                    try:
                        process = await asyncio.create_subprocess_shell(
                            apptainer_cmd, stdout=log_file, stderr=log_file
                        )
                        await asyncio.wait_for(process.communicate(), timeout=timeout)
                        if process.returncode != 0:
                            raise ValueError(f"Command failed with return code {process.returncode}")
                    except asyncio.TimeoutError:
                        if process.returncode is None:
                            process.terminate()
                            try:
                                await asyncio.wait_for(process.wait(), timeout=10)
                            except asyncio.TimeoutError:
                                process.kill()
                                await process.wait()
                        attempt = max_retries
                        raise ValueError("Command timed out")

                pred_files = glob.glob(expected_file_pattern, recursive=True)
                if len(pred_files) == 1:
                    return pred_files[0]
                elif len(pred_files) > 1:
                    import os as _os
                    return max(pred_files, key=_os.path.getmtime)
                else:
                    raise ValueError(
                        f"Expected file matching {expected_file_pattern}, found {len(pred_files)}"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {data_point['instance_id']}: {e}", flush=True)
                    continue
                else:
                    raise ValueError(
                        f"All attempts failed for {data_point['instance_id']}. "
                        f"Logs: {log_file_path}. Error: {e}"
                    )

    async def _run_hermes(
        self,
        data_point: dict[str, Any],
        api_base: str,
        dataset_mount_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Run hermes-agent inside an Apptainer container on one SWE-bench instance.

        Returns the path to a .jsonl prediction file in SWE-bench evaluation format,
        or None if the agent failed.
        """
        assert self.hermes_setup_dir is not None, "Hermes setup directory is not set"

        # Write the runner script into the output directory so it can be mounted
        runner_script_path = Path(self.output_dir) / "hermes_runner.py"
        with open(runner_script_path, "w") as f:
            f.write(HERMES_RUNNER_SCRIPT)
            f.flush()
            os.fsync(f.fileno())

        # Write problem statement to a file (avoids shell quoting nightmares)
        problem_file = Path(self.output_dir) / f"problem_{data_point['instance_id']}.txt"
        with open(problem_file, "w") as f:
            f.write(data_point.get("problem_statement", ""))
            f.flush()
            os.fsync(f.fileno())

        output_filename = f"{data_point['instance_id']}.jsonl"

        # hermes-agent is built at /opt/hermes-agent inside the NeMo RL container
        # (via SETUP_COMMAND). We mount it into the SWE-bench Apptainer container
        # along with the uv-managed Python so the venv resolves correctly.
        hermes_install = "/opt/hermes-agent"

        hermes_cmd = (
            f"cd /testbed && "
            f"PYTHONPATH={hermes_install} "
            f"{hermes_install}/venv/bin/python "
            f"/trajectories_mount/hermes_runner.py "
            f"    --model-base-url {shlex.quote(api_base)} "
            f"    --model {shlex.quote(self.cfg.server['model'])} "
            f"    --instance-id {shlex.quote(data_point['instance_id'])} "
            f"    --problem-statement \"$(cat /trajectories_mount/{problem_file.name})\" "
            f"    --output-file /trajectories_mount/{output_filename} "
            f"    --max-turns {self.cfg.agent_max_turns} "
        )

        search_path = os.path.join(str(self.output_dir), output_filename)

        try:
            pred_file = await self._execute_container_command(
                data_point=data_point,
                command=hermes_cmd,
                expected_file_pattern=search_path,
                mode="agent",
                max_retries=1,
                timeout=self.cfg.swebench_agent_timeout + 60,
                dataset_mount_path=dataset_mount_path,
            )
        except Exception as e:
            print(f"Running hermes-agent failed: {e}", flush=True)
            return None
        finally:
            # Clean up problem statement file
            try:
                problem_file.unlink(missing_ok=True)
            except OSError:
                pass

        # Copy trajectory to the trajectories dir for get_trajectory_and_tools()
        traj_source = Path(str(pred_file).replace(".jsonl", ".trajectory.json"))
        if traj_source.exists():
            traj_dest = Path(self.output_dir) / "trajectories" / data_point["instance_id"]
            traj_dest.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(traj_source, traj_dest / "trajectory.json")

        return str(pred_file)

    async def process_single_datapoint(self, data_point: dict[str, Any]):
        """Run hermes agent and SWE-bench evaluation on a single instance."""
        self.output_dir = Path(self.cfg.output_file).parent

        agent_run_id = f"{data_point['instance_id']}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        instance_dataset_path = self._write_instance_dataset(data_point, agent_run_id)
        api_base = self.cfg.server["base_url"]

        import asyncio
        start_time = asyncio.get_running_loop().time()
        generation_time = None
        evaluation_time = None

        try:
            pred_file = await self._run_hermes(
                data_point,
                api_base,
                instance_dataset_path,
            )

            generation_time = asyncio.get_running_loop().time() - start_time

            if pred_file is None:
                report_json = {
                    data_point["instance_id"]: {
                        "resolved": False,
                        "patch_exists": False,
                        "patch_successfully_applied": False,
                        "generation_time": generation_time,
                        "evaluation_time": evaluation_time,
                    }
                }
            else:
                with open(pred_file, "r") as f:
                    trajectory_dict = json.loads(f.read().strip())

                has_patch = trajectory_dict.get("model_patch") is not None

                if not has_patch:
                    report_json = {
                        data_point["instance_id"]: {
                            "resolved": False,
                            "patch_exists": False,
                            "patch_successfully_applied": False,
                            "generation_time": generation_time,
                            "evaluation_time": evaluation_time,
                        }
                    }
                else:
                    # Reuse pred file path for eval (already in SWE-bench format)
                    pred_mounted_path = pred_file.replace(
                        str(self.output_dir), "/trajectories_mount"
                    )

                    try:
                        start_time = asyncio.get_running_loop().time()

                        if data_point["dataset_name"] == "nv-internal-1":
                            report_file = await self._run_nv_internal_eval(
                                data_point,
                                trajectory_dict["model_patch"],
                                instance_dataset_path,
                            )
                        elif "R2E-Gym" in data_point["dataset_name"]:
                            report_file = await self._run_r2e_gym_eval(
                                pred_mounted_path,
                                data_point,
                                agent_run_id,
                                instance_dataset_path,
                            )
                        else:
                            report_file = await self._run_swebench_eval(
                                pred_mounted_path,
                                data_point,
                                agent_run_id,
                                instance_dataset_path,
                            )
                        evaluation_time = asyncio.get_running_loop().time() - start_time

                    except ValueError:
                        print(
                            f"Failed to execute SWE-bench evaluation for {data_point['instance_id']}",
                            flush=True,
                        )
                        report_json = {
                            data_point["instance_id"]: {
                                "resolved": False,
                                "patch_exists": True,
                                "patch_successfully_applied": False,
                                "generation_time": generation_time,
                                "evaluation_time": evaluation_time,
                            }
                        }
                        report_file = None

                    if report_file is not None:
                        with open(report_file, "r") as f:
                            report_json = json.loads(f.read().strip())

            output_dict = {
                "swe-bench-metrics": report_json[data_point["instance_id"]],
                "hermes_metrics": {},
                "generation": "",
                "generation_time": generation_time,
                "evaluation_time": evaluation_time,
            }

            return output_dict

        finally:
            self._cleanup_instance_dataset(instance_dataset_path)
