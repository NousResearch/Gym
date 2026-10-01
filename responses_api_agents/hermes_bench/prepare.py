# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


THIS_DIR = Path(__file__).parent
GYM_ROOT = THIS_DIR.parents[1]
HERMES_AGENT_TASK_ID = "task_architecture_diagram_kanban_telegram_functions_extreme"
SUPPORT_PATHS = [
    "scripts/github_emulator",
    "scripts/p5_render",
    "scripts/lib_architecture_diagram.py",
    "scripts/lib_architecture_diagram_judge.py",
    "scripts/hermes_skills_runner.py",
    "scripts/lib_cost.py",
    "scripts/lib_grading.py",
    "scripts/lib_tasks.py",
]


def _sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    name = None
    lines: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            if name:
                lines.append(line)
            continue
        match = None if in_fence else re.match(r"^##\s+(.+)$", line)
        if match:
            if name:
                sections[name] = "\n".join(lines).strip()
            name = match.group(1).strip()
            lines = []
        elif name:
            lines.append(line)
    if name:
        sections[name] = "\n".join(lines).strip()
    return sections


def _fenced_code(text: str, languages: str) -> str | None:
    blocks = re.findall(rf"```(?:{languages})\s*\n(.*?)```", text or "", re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return text.strip() if text and text.strip() else None


def parse_task(path: Path, source_root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError(f"{path} is missing YAML frontmatter")
    metadata = yaml.safe_load(match.group(1)) or {}
    parsed_sections = _sections(match.group(2))
    grading_type = metadata.get("grading_type") or "automated"
    valid_types = {"automated", "llm_judge", "hybrid", "architecture_diagram"}
    if grading_type not in valid_types:
        raise ValueError(f"{path} has unknown grading_type {grading_type!r}")
    timeout = int(metadata.get("timeout_seconds") or 120)
    if timeout <= 0:
        raise ValueError(f"{path} has non-positive timeout_seconds")
    judge_config = metadata.get("judge_config") if "judge_config" in metadata else metadata.get("judge")
    task = {
        "task_id": metadata.get("id") or "",
        "name": metadata.get("name") or "",
        "category": metadata.get("category") or "",
        "grading_type": grading_type,
        "timeout_seconds": timeout,
        "workspace_files": metadata.get("workspace_files") or [],
        "prompt": parsed_sections.get("Prompt", "").strip(),
        "expected_behavior": parsed_sections.get("Expected Behavior", "").strip(),
        "grading_criteria": [
            item.group(1)
            for line in parsed_sections.get("Grading Criteria", "").splitlines()
            if (item := re.match(r"^-\s+\[[ x]\]\s+(.+)$", line.strip()))
        ],
        "automated_checks": _fenced_code(parsed_sections.get("Automated Checks", ""), "python"),
        "llm_judge_rubric": parsed_sections.get("LLM Judge Rubric") or metadata.get("llm_judge_rubric"),
        "grading_weights": metadata.get("grading_weights"),
        "judge_config": judge_config or {},
        "toolsets": metadata.get("toolsets"),
        "setup_script": _fenced_code(parsed_sections.get("Setup Script", ""), "bash|sh"),
        "isolation": metadata.get("isolation"),
        "source_file": str(path.relative_to(source_root)),
    }
    if not task["task_id"]:
        raise ValueError(f"{path} has an empty id")
    if grading_type == "architecture_diagram":
        rubric_path = (task["judge_config"] or {}).get("rubric_path")
        if not rubric_path:
            raise ValueError(f"{path} has no architecture rubric_path")
        candidates = [path.parent / rubric_path, path.parent.parent / rubric_path, source_root / "tasks" / rubric_path]
        resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"architecture rubric not found for {task['task_id']}: {rubric_path}")
        task["architecture_rubric"] = resolved.read_text(encoding="utf-8", errors="replace")
    return task


def load_source_tasks(source_root: Path) -> list[dict[str, Any]]:
    library = source_root / "scripts" / "lib_tasks.py"
    if not library.is_file():
        raise FileNotFoundError(f"original HermesBench task loader is missing: {library}")
    spec = importlib.util.spec_from_file_location("hermes_bench_source_lib_tasks", library)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import original HermesBench task loader: {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.TaskLoader(source_root / "tasks")
    source_tasks = loader.load_all_tasks()
    if loader.load_failures:
        raise ValueError(f"original task loader failed on {len(loader.load_failures)} files: {loader.load_failures}")
    tasks = []
    for source_task in source_tasks:
        task = {
            "task_id": source_task.task_id,
            "name": source_task.name,
            "category": source_task.category,
            "grading_type": source_task.grading_type,
            "timeout_seconds": source_task.timeout_seconds,
            "workspace_files": source_task.workspace_files,
            "prompt": source_task.prompt,
            "expected_behavior": source_task.expected_behavior,
            "grading_criteria": source_task.grading_criteria,
            "automated_checks": source_task.automated_checks,
            "llm_judge_rubric": source_task.llm_judge_rubric,
            "grading_weights": source_task.grading_weights,
            "judge_config": source_task.judge_config,
            "toolsets": source_task.toolsets,
            "setup_script": source_task.setup_script,
            "isolation": source_task.isolation,
            "source_file": str(Path(source_task.file_path).relative_to(source_root)),
        }
        if task["grading_type"] == "architecture_diagram":
            rubric_path = (task["judge_config"] or {}).get("rubric_path")
            candidates = [
                Path(source_task.file_path).parent / str(rubric_path),
                Path(source_task.file_path).parent.parent / str(rubric_path),
                source_root / "tasks" / str(rubric_path),
            ]
            resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
            if resolved is None:
                raise FileNotFoundError(f"architecture rubric not found for {task['task_id']}: {rubric_path}")
            task["architecture_rubric"] = resolved.read_text(encoding="utf-8", errors="replace")
        tasks.append(task)
    return tasks


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"fixture path escapes workspace: {relative}")
    return destination


def _resolve_source_path(
    value: str,
    task_file: Path,
    source_root: Path,
    override: Path | None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [source_root / raw, task_file.parent / raw, source_root.parent / raw]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"source_path fixture is missing: {value}")
    return resolved


def build_fixture(
    task: dict[str, Any],
    source_root: Path,
    output: Path,
    source_override: Path | None = None,
) -> None:
    task_file = source_root / task["source_file"]
    with tempfile.TemporaryDirectory(prefix="hermes-bench-fixture-") as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()
        for spec in task["workspace_files"]:
            if "content" in spec:
                destination = _safe_destination(workspace, spec["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(str(spec["content"]), encoding="utf-8")
                continue
            if "source" in spec:
                source = (source_root / "assets" / spec["source"]).resolve()
            elif "source_path" in spec:
                source = _resolve_source_path(spec["source_path"], task_file, source_root, source_override)
            else:
                raise ValueError(f"unknown workspace fixture for {task['task_id']}: {spec}")
            if not source.exists():
                raise FileNotFoundError(f"fixture source is missing for {task['task_id']}: {source}")
            destination = _safe_destination(workspace, spec.get("dest") or spec.get("path") or source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)
            else:
                shutil.copy2(source, destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(workspace, arcname="workspace")


def build_support_archive(source_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hermes-bench-support-") as temp_dir:
        support_root = Path(temp_dir) / "benchmark"
        for relative in SUPPORT_PATHS:
            source = source_root / relative
            if not source.exists():
                raise FileNotFoundError(f"benchmark support path is missing: {source}")
            destination = support_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        p5_dir = support_root / "scripts/p5_render"
        if (p5_dir / "package-lock.json").exists():
            if not shutil.which("npm"):
                raise RuntimeError("npm is required to prepare the p5 renderer")
            subprocess.run(
                ["npm", "ci", "--omit=dev", "--ignore-scripts"],
                cwd=p5_dir,
                check=True,
            )

        with tarfile.open(output, "w:gz") as archive:
            archive.add(support_root, arcname="benchmark")


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(GYM_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def to_gym_row(task: dict[str, Any], fixture: Path, support: Path) -> dict[str, Any]:
    system_prompt = (
        "You are a helpful AI assistant. Complete the task described below. "
        "Create all files in the current working directory. IMPORTANT: You MUST "
        "call skills_list to check for relevant skills before doing any work. If "
        "a relevant skill exists, call skill_view to load it and follow its "
        "guidance. Do not skip this step."
    )
    public_task = {key: value for key, value in task.items() if key != "workspace_files"}
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["prompt"]},
            ],
            "metadata": {
                "task_id": task["task_id"],
                "dataset_name": "HermesBench",
                "fixture_archive": _portable_path(fixture),
                "support_archive": _portable_path(support),
                "task": json.dumps(public_task),
            },
        }
    }


@contextmanager
def source_tree(source: Path) -> Iterator[Path]:
    source = source.expanduser().resolve()
    if source.is_dir():
        yield source
        return
    if not zipfile.is_zipfile(source):
        raise ValueError(f"source must be a HermesBench directory or zip: {source}")
    with tempfile.TemporaryDirectory(prefix="hermes-bench-source-") as temp_dir:
        destination = Path(temp_dir)
        with zipfile.ZipFile(source) as archive:
            manifest = next((name for name in archive.namelist() if name.endswith("/tasks/manifest.yaml")), None)
            if manifest is None:
                raise ValueError("zip has no tasks/manifest.yaml")
            prefix = manifest[: -len("tasks/manifest.yaml")]
            wanted = [f"{prefix}{path}" for path in ["tasks/", "assets/", *SUPPORT_PATHS]]
            for member in archive.infolist():
                if not any(member.filename == path or member.filename.startswith(path) for path in wanted):
                    continue
                relative = Path(member.filename).relative_to(prefix)
                if ".." in relative.parts:
                    raise ValueError(f"unsafe zip member: {member.filename}")
                archive.extract(member, destination / "source")
        root = destination / "source" / prefix
        yield root.resolve()


def build_dataset(
    source: Path,
    output: Path,
    fixture_dir: Path,
    task_ids: set[str],
    limit: int | None,
    source_overrides: dict[str, Path],
) -> int:
    with source_tree(source) as source_root:
        tasks = load_source_tasks(source_root)
        ids = [task["task_id"] for task in tasks]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids: {duplicates}")
        if task_ids:
            tasks = [task for task in tasks if task["task_id"] in task_ids]
            missing = sorted(task_ids - {task["task_id"] for task in tasks})
            if missing:
                raise ValueError(f"unknown task ids: {missing}")
        if limit is not None:
            tasks = tasks[:limit]
        support = fixture_dir.parent / "support.tar.gz"
        build_support_archive(source_root, support)
        rows = []
        for task in tasks:
            fixture = fixture_dir / f"{task['task_id']}.tar.gz"
            build_fixture(task, source_root, fixture, source_overrides.get(task["task_id"]))
            rows.append(to_gym_row(task, fixture, support))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="HermesBench source directory or zip")
    parser.add_argument("--output", type=Path, default=THIS_DIR / "data" / "hermes_bench.jsonl")
    parser.add_argument("--fixture-dir", type=Path, default=THIS_DIR / "data" / "fixtures")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hermes-agent-path", type=Path)
    args = parser.parse_args()
    try:
        source_overrides = {}
        if args.hermes_agent_path is not None:
            source_overrides[HERMES_AGENT_TASK_ID] = args.hermes_agent_path
        count = build_dataset(
            source=args.source,
            output=args.output,
            fixture_dir=args.fixture_dir,
            task_ids=set(args.task),
            limit=args.limit,
            source_overrides=source_overrides,
        )
    except Exception as error:
        sys.exit(f"HermesBench preparation failed: {error}")
    print(f"Wrote {count} tasks to {args.output}")


if __name__ == "__main__":
    main()
