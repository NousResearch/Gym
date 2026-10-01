# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    sketch, output_dir, *extra = sys.argv[1:]
    output_root = Path(output_dir)
    rewritten = [
        str(output_root / value.removeprefix("/out/")) if value.startswith("/out/") else value for value in extra
    ]
    script = Path("/sandbox/benchmark/scripts/p5_render/render_check.js")
    with tempfile.TemporaryDirectory(prefix="hermes-p5-") as browser_home:
        env = os.environ.copy()
        env["HOME"] = browser_home
        env["XDG_CACHE_HOME"] = str(Path(browser_home) / ".cache")
        env["XDG_CONFIG_HOME"] = str(Path(browser_home) / ".config")
        result = subprocess.run(["node", str(script), sketch, *rewritten], env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
