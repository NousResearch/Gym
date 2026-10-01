import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from responses_api_agents.hermes_bench.prepare import (
    build_dataset,
    build_support_archive,
    parse_task,
    to_gym_row,
)
from responses_api_agents.hermes_bench.prepare import (
    source_tree as extracted_source_tree,
)


TASK = """---
id: task_example
name: Example
category: test
grading_type: automated
timeout_seconds: 42
toolsets: [file, terminal]
workspace_files:
  - path: input.txt
    content: hello
---

## Prompt

Write answer.txt.

## Automated Checks

```python
def grade(transcript, workspace_path, output=''):
    from pathlib import Path
    return {'created': (Path(workspace_path) / 'answer.txt').exists()}
```
"""


def source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "tasks").mkdir(parents=True)
    (source / "tasks" / "task_example.md").write_text(TASK)
    (source / "tasks" / "manifest.yaml").write_text("categories: {}\n")
    (source / "scripts" / "github_emulator").mkdir(parents=True)
    (source / "scripts" / "github_emulator" / "gh_shim.py").write_text("")
    (source / "scripts" / "p5_render").mkdir(parents=True)
    (source / "scripts" / "p5_render" / "grader_helpers.py").write_text("")
    (source / "scripts" / "lib_architecture_diagram.py").write_text("")
    (source / "scripts" / "lib_architecture_diagram_judge.py").write_text("")
    (source / "scripts" / "hermes_skills_runner.py").write_text("")
    (source / "scripts" / "lib_cost.py").write_text("")
    (source / "scripts" / "lib_grading.py").write_text("")
    (source / "scripts" / "lib_tasks.py").write_text("")
    return source


def test_parse_preserves_execution_metadata(tmp_path: Path) -> None:
    source = source_tree(tmp_path)
    task = parse_task(source / "tasks" / "task_example.md", source)
    assert task["timeout_seconds"] == 42
    assert task["toolsets"] == ["file", "terminal"]
    assert "def grade" in task["automated_checks"]
    assert task["workspace_files"] == [{"path": "input.txt", "content": "hello"}]


def test_build_dataset_writes_fixture_and_full_metadata(tmp_path: Path, monkeypatch) -> None:
    import responses_api_agents.hermes_bench.prepare as prepare

    source = source_tree(tmp_path)
    monkeypatch.setattr(
        prepare, "load_source_tasks", lambda root: [parse_task(root / "tasks" / "task_example.md", root)]
    )
    output = tmp_path / "data" / "tasks.jsonl"
    fixture_dir = tmp_path / "data" / "fixtures"
    count = build_dataset(source, output, fixture_dir, set(), None, {})
    assert count == 1
    row = json.loads(output.read_text())
    task = json.loads(row["responses_create_params"]["metadata"]["task"])
    assert task["task_id"] == "task_example"
    assert "workspace_files" not in task
    with tarfile.open(fixture_dir / "task_example.tar.gz") as archive:
        assert archive.extractfile("workspace/input.txt").read() == b"hello"


def test_missing_source_path_fails_preparation(tmp_path: Path, monkeypatch) -> None:
    import responses_api_agents.hermes_bench.prepare as prepare

    source = source_tree(tmp_path)
    task_file = source / "tasks" / "task_example.md"
    task_file.write_text(
        TASK.replace(
            "workspace_files:\n  - path: input.txt\n    content: hello",
            "workspace_files:\n  - source_path: missing\n    dest: repo",
        )
    )
    monkeypatch.setattr(prepare, "load_source_tasks", lambda root: [parse_task(task_file, root)])
    output = tmp_path / "data" / "tasks.jsonl"
    with pytest.raises(FileNotFoundError, match="source_path fixture is missing"):
        build_dataset(source, output, tmp_path / "data" / "fixtures", set(), None, {})


def test_row_uses_archived_skills_prompt(tmp_path: Path) -> None:
    source = source_tree(tmp_path)
    task = parse_task(source / "tasks" / "task_example.md", source)
    row = to_gym_row(task, tmp_path / "fixture.tar.gz", tmp_path / "support.tar.gz")
    messages = row["responses_create_params"]["input"]
    assert messages[0]["role"] == "system"
    assert "call skills_list" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Write answer.txt."}


def test_support_archive_installs_p5_runtime(tmp_path: Path, monkeypatch) -> None:
    import responses_api_agents.hermes_bench.prepare as prepare

    source = source_tree(tmp_path)
    p5_dir = source / "scripts" / "p5_render"
    (p5_dir / "package-lock.json").write_text("{}")

    def npm_ci(command, cwd, check):
        assert command == ["npm", "ci", "--omit=dev", "--ignore-scripts"]
        assert cwd.name == "p5_render"
        assert check is True
        (cwd / "node_modules" / "puppeteer-core").mkdir(parents=True)
        (cwd / "node_modules" / "puppeteer-core" / "package.json").write_text("{}")

    monkeypatch.setattr(prepare.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(prepare.subprocess, "run", npm_ci)
    output = tmp_path / "support.tar.gz"

    build_support_archive(source, output)

    with tarfile.open(output) as archive:
        assert archive.getmember("benchmark/scripts/p5_render/node_modules/puppeteer-core/package.json").isfile()


def test_zip_extraction_excludes_score_credentials(tmp_path: Path) -> None:
    source = source_tree(tmp_path)
    (source / "scores_and_models.txt").write_text("must-not-be-extracted")
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, Path("hermesbench") / path.relative_to(source))
    with extracted_source_tree(archive_path) as extracted:
        assert not (extracted / "scores_and_models.txt").exists()
