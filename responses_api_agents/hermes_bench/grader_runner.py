# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


WORKSPACE = Path("/sandbox/workspace")
CONTROL = Path("/sandbox/control")
SUPPORT_SCRIPTS = Path("/sandbox/benchmark/scripts")


def _content_text(item: dict) -> str:
    content = item.get("content") or []
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def response_to_transcript(response: dict) -> list[dict]:
    transcript: list[dict] = []
    active_assistant: dict | None = None
    for item in response.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "assistant")
            text = _content_text(item)
            parts = [{"type": "text", "text": text}] if text else []
            entry = {
                "type": "message",
                "role": role,
                "content": parts,
                "content_text": text,
                "tool_calls": [],
                "message": {"role": role, "content": parts},
            }
            transcript.append(entry)
            active_assistant = entry if role == "assistant" else None
            continue
        if item_type == "function_call":
            if active_assistant is None:
                active_assistant = {
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "content_text": "",
                    "tool_calls": [],
                    "message": {"role": "assistant", "content": []},
                }
                transcript.append(active_assistant)
            raw_arguments = item.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                arguments = raw_arguments
            call_id = item.get("call_id") or item.get("id")
            call = {
                "type": "toolCall",
                "name": item.get("name", ""),
                "arguments": arguments,
                "params": arguments,
                "input": arguments,
                "id": call_id,
            }
            active_assistant["content"].append(call)
            active_assistant["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": raw_arguments},
                }
            )
            continue
        if item_type == "function_call_output":
            output = item.get("output", "")
            parts = [{"type": "text", "text": str(output)}]
            transcript.append(
                {
                    "type": "message",
                    "role": "tool",
                    "content": parts,
                    "content_text": str(output),
                    "tool_call_id": item.get("call_id"),
                    "message": {"role": "tool", "content": parts},
                }
            )
            active_assistant = None
    return transcript


def _assistant_output(response: dict) -> str:
    return "\n".join(
        _content_text(item)
        for item in response.get("output") or []
        if item.get("type") == "message" and item.get("role") == "assistant"
    )


def _mapped_judge_model(model: str | None) -> str | None:
    if not model:
        return model
    mapping = json.loads(os.environ.get("HERMES_JUDGE_MODEL_MAP", "{}"))
    return str(mapping.get(model, model))


def _configure_judge_transport() -> None:
    url = os.environ.get("HERMES_ARCHITECTURE_JUDGE_URL")
    if not url:
        return
    import lib_architecture_diagram_judge

    lib_architecture_diagram_judge.NOUS_URL = url


def _normalized_rubric(value: str | None) -> str | None:
    if not value or re.search(r"#{2,4}\s*Criterion\s+\d+:", value, re.IGNORECASE):
        return value
    items = re.findall(r"^\s*\d+\.\s+\*\*(.+?)\*\*:\s*(.+)$", value, re.MULTILINE)
    if not items:
        return value
    weight = 100.0 / len(items)
    sections = []
    for index, (name, description) in enumerate(items, 1):
        sections.append(
            "\n".join(
                [
                    f"### Criterion {index}: {name.strip()} (Weight: {weight:.6f}%)",
                    f"- **Score 1.0**: Fully satisfies this requirement: {description.strip()}",
                    f"- **Score 0.0**: Does not satisfy this requirement: {description.strip()}",
                ]
            )
        )
    return "\n\n".join(sections)


def _task_object(task: dict) -> SimpleNamespace:
    judge_config = dict(task.get("judge_config") or {})
    for key in ("model", "parser_model"):
        if judge_config.get(key):
            judge_config[key] = _mapped_judge_model(judge_config[key])
    if task["grading_type"] == "architecture_diagram":
        score_cache_control = os.environ.get("HERMES_ARCHITECTURE_SCORE_CACHE_CONTROL")
        if score_cache_control:
            judge_config["score_cache_control"] = score_cache_control
        rubric_path = CONTROL / "architecture_rubric.txt"
        rubric_path.write_text(task.get("architecture_rubric", ""))
        judge_config["rubric_path"] = str(rubric_path)
    return SimpleNamespace(
        task_id=task["task_id"],
        name=task.get("name", ""),
        category=task.get("category", ""),
        grading_type=task["grading_type"],
        timeout_seconds=int(task.get("timeout_seconds") or 120),
        workspace_files=[],
        prompt=task.get("prompt", ""),
        expected_behavior=task.get("expected_behavior", ""),
        grading_criteria=task.get("grading_criteria") or [],
        automated_checks=task.get("automated_checks"),
        llm_judge_rubric=_normalized_rubric(task.get("llm_judge_rubric")),
        grading_weights=task.get("grading_weights"),
        judge_config=judge_config,
        toolsets=task.get("toolsets"),
        setup_script=task.get("setup_script"),
        isolation=task.get("isolation"),
        file_path=None,
    )


def _tool_calls(transcript: list[dict]) -> list[dict]:
    calls = []
    for entry in transcript:
        for call in entry.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"name": function.get("name", ""), "arguments": arguments})
    return calls


def _skills_record(
    task: dict,
    response: dict,
    transcript: list[dict],
    agent_metrics: dict,
) -> dict:
    return {
        "task_id": task["task_id"],
        "task_name": task.get("name", ""),
        "run_id": uuid4().hex[:8],
        "source_tag": "nemo-gym",
        "model": response.get("model"),
        "provider": None,
        "toolsets": task.get("toolsets") or [],
        "isolation": {"enabled": True, "backend": "nemo_gym_sandbox"},
        "preload_skills": [],
        "started_at": time.time(),
        "started_at_iso": "",
        "wall_seconds": float(agent_metrics.get("agent_run_time") or 0),
        "timed_out": bool(agent_metrics.get("agent_timed_out")),
        "returncode": int(agent_metrics.get("agent_return_code") or 0),
        "session_id": None,
        "session": None,
        "stdout": _assistant_output(response),
        "stderr": str(agent_metrics.get("agent_error") or ""),
        "messages": transcript,
        "tool_calls": _tool_calls(transcript),
        "workspace_path": str(WORKSPACE),
        "workspace_listing": [str(path.relative_to(WORKSPACE)) for path in WORKSPACE.rglob("*") if path.is_file()],
        "state_db_path": "",
        "setup_env": {},
        "setup_env_host": {},
        "prompt": task.get("prompt", ""),
        "setup_error": None,
    }


def _image_paths(scores: dict) -> list[str]:
    raw = scores.get("_judge_image_paths") or scores.get("_judge_image_path") or []
    if isinstance(raw, (str, Path)):
        return [str(raw)]
    if isinstance(raw, list):
        return [str(path) for path in raw if path]
    return []


def _grade_skills(
    task: dict,
    response: dict,
    transcript: list[dict],
    agent_metrics: dict,
) -> dict:
    from hermes_skills_runner import aggregate_score, call_grade
    from lib_grading import calculate_overall_score, default_judge_backend, run_llm_judge_full

    task_object = _task_object(task)
    record = _skills_record(task, response, transcript, agent_metrics)
    scores = call_grade(task_object, record)
    judge_scores = {}
    judge_bundle = None
    judge_attempted = bool(task["grading_type"] in ("llm_judge", "hybrid") and task.get("llm_judge_rubric"))
    missing_required_images = False
    if judge_attempted:
        judge_config = task.get("judge_config") or {}
        requires_images = bool(judge_config.get("requires_images", False))
        image_paths = _image_paths(scores)
        missing_required_images = requires_images and not image_paths
        if not requires_images or image_paths:
            try:
                judge_model = (
                    judge_config.get("model") or os.environ.get("HERMES_JUDGE_MODEL") or "anthropic/claude-opus-4.7"
                )
                judge_bundle = run_llm_judge_full(
                    task=task_object,
                    execution_result={
                        "workspace_path": str(WORKSPACE),
                        "output": record["stdout"],
                        "transcript": transcript,
                    },
                    judge_model=_mapped_judge_model(judge_model),
                    judge_backend=judge_config.get("backend")
                    or os.environ.get("HERMES_JUDGE_BACKEND")
                    or default_judge_backend(),
                    judge_source_tag=f"hermesbench-judge-{task['task_id']}-{record['run_id']}",
                    image_paths=image_paths,
                    temperature=judge_config.get("temperature"),
                    top_p=judge_config.get("top_p"),
                    max_tokens=judge_config.get("max_tokens"),
                )
                judge_scores = judge_bundle.get("scores", {})
            except Exception as error:
                judge_scores = {"_error": str(error)}

    judge_missing = not judge_scores or "_error" in judge_scores
    score_valid = True
    error = None
    if task["grading_type"] == "llm_judge" and judge_missing:
        if scores.get("_error") or missing_required_images:
            aggregate = 0.0
        else:
            aggregate = None
            score_valid = False
            error = judge_scores.get("_error", "judge returned no scores")
    elif task["grading_type"] == "hybrid" and judge_missing and judge_attempted:
        if scores.get("_error") or missing_required_images:
            aggregate = 0.0
        else:
            aggregate = None
            score_valid = False
            error = judge_scores.get("_error", "judge returned no scores")
    elif task["grading_type"] == "automated" or judge_missing:
        aggregate = aggregate_score(scores, task.get("grading_weights"))
    else:
        # Keep `_defect_groups` alongside the numeric axes: it is a dict, so a
        # plain isinstance((int, float)) filter silently discarded it and
        # automated scoring fell back to the flat per-axis mean. Observed
        # 2026-08-14 — the collapsed scorer was live in lib_grading but never
        # received its grouping metadata, so the suite scored 0.789 where the
        # replay predicted 0.73.
        combined = {
            "automated": {
                key: value for key, value in scores.items()
                if isinstance(value, (int, float)) or key == "_defect_groups"
            },
            "llm_judge": {key: value for key, value in judge_scores.items() if isinstance(value, (int, float))},
        }
        aggregate = calculate_overall_score(
            task["grading_type"],
            task.get("grading_weights"),
            combined,
            judge_criteria=(judge_bundle or {}).get("criteria"),
        )
        scores = {
            **scores,
            **{f"judge__{key}": value for key, value in judge_scores.items() if isinstance(value, (int, float))},
        }
    return {
        "reward": min(max(float(aggregate or 0.0), 0.0), 1.0),
        "score_valid": score_valid,
        "grading_status": (
            "scored_model_failure"
            if score_valid and (scores.get("_error") or missing_required_images)
            else "scored"
            if score_valid
            else "grading_error"
        ),
        "scores": scores,
        "feedback": error or "",
        "error": error,
        "transcript_events": len(transcript),
        "agent": agent_metrics,
        "judge": judge_bundle,
    }


def _grade_main(
    task: dict,
    response: dict,
    transcript: list[dict],
    agent_metrics: dict,
) -> dict:
    sys.path.insert(0, str(SUPPORT_SCRIPTS))
    from lib_grading import grade_task

    execution_result = {
        "task_id": task["task_id"],
        "name": task.get("name", ""),
        "category": task.get("category", ""),
        "start_time": 0,
        "execution_time": 0,
        "success": True,
        "output": _assistant_output(response),
        "error": str(agent_metrics.get("agent_error") or ""),
        "timed_out": bool(agent_metrics.get("agent_timed_out")),
        "return_code": int(agent_metrics.get("agent_return_code") or 0),
        "transcript": transcript,
        "workspace_path": str(WORKSPACE),
        "model": response.get("model"),
    }
    if task["grading_type"] == "architecture_diagram":
        execution_result["artifact_root"] = str(CONTROL / "architecture")
    grading = grade_task(
        _task_object(task),
        execution_result,
        judge_model=_mapped_judge_model(os.environ.get("HERMES_JUDGE_MODEL") or None),
    )
    score_valid = bool(grading.get("score_valid", True))
    if task["grading_type"] == "architecture_diagram" and not score_valid:
        architecture = (grading.get("scores") or {}).get("architecture_diagram") or {}
        architecture_error = str(architecture.get("error") or grading.get("feedback") or "")
        if "no architecture artifact found" in architecture_error.lower():
            architecture.update(
                {
                    "score": 0.0,
                    "score_valid": True,
                    "grading_status": "model_failure",
                }
            )
            grading["overall_score"] = 0.0
            grading["score_valid"] = True
            grading["grading_status"] = "scored_model_failure"
            grading["feedback"] = architecture_error
            score_valid = True
    raw_score = grading.get("overall_score")
    reward = float(raw_score) if score_valid and raw_score is not None else 0.0
    return {
        "reward": min(max(reward, 0.0), 1.0),
        "score_valid": score_valid,
        "grading_status": grading.get("grading_status", "scored"),
        "scores": grading.get("scores") or {},
        "feedback": grading.get("feedback", ""),
        "error": None if score_valid else grading.get("feedback") or "grading failed",
        "transcript_events": len(transcript),
        "agent": agent_metrics,
    }


def grade(task: dict, response: dict, agent_metrics: dict | None = None) -> dict:
    sys.path.insert(0, str(SUPPORT_SCRIPTS))
    _configure_judge_transport()
    transcript = response_to_transcript(response)
    agent_metrics = agent_metrics or {}
    source_file = str(task.get("source_file") or "").replace("\\", "/")
    if source_file.startswith("tasks/hermes_skills/") and task["grading_type"] != "architecture_diagram":
        return _grade_skills(task, response, transcript, agent_metrics)
    return _grade_main(task, response, transcript, agent_metrics)


def main() -> None:
    task = json.loads((CONTROL / "task.json").read_text())
    response = json.loads((CONTROL / "response.json").read_text())
    metrics_path = CONTROL / "agent_metrics.json"
    agent_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    try:
        result = grade(task, response, agent_metrics)
    except Exception as error:
        result = {
            "reward": 0.0,
            "score_valid": False,
            "grading_status": "grading_error",
            "scores": {},
            "error": f"{type(error).__name__}: {error}",
        }
    (CONTROL / "grade.json").write_text(json.dumps(result, indent=2, default=str))
    if not result["score_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
