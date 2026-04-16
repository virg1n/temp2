from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def execute_buggy_task(
    buggy_python: str,
    asserts: list[str],
    *,
    timeout_seconds: float,
    max_output_chars: int,
    execution_mode: str,
    sandbox_command: list[str],
    allow_unsafe_host_execution: bool,
) -> str:
    script = buggy_python.rstrip() + "\n\n" + "\n".join(asserts) + "\n"
    with tempfile.TemporaryDirectory(prefix="acl_task_") as tmp_dir:
        script_path = Path(tmp_dir) / "task.py"
        script_path.write_text(script, encoding="utf-8")

        mode = execution_mode.strip().lower()
        if mode == "disabled":
            return "None"
        if mode == "sandbox":
            if not sandbox_command:
                return "Execution unavailable: sandbox mode is enabled but no sandbox command is configured"
            command = [
                part.format(
                    python=sys.executable,
                    script_path=str(script_path),
                    workdir=tmp_dir,
                )
                for part in sandbox_command
            ]
        elif mode == "host":
            if not allow_unsafe_host_execution:
                return "Execution blocked: unsafe host execution is disabled"
            command = [sys.executable, "-I", str(script_path)]
        else:
            return f"Execution unavailable: unsupported execution mode '{execution_mode}'"

        try:
            result = subprocess.run(
                command,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout_seconds:.1f} seconds"

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if result.returncode != 0:
        output = stderr if stderr else stdout if stdout else f"Process exited with code {result.returncode}"
    else:
        output = "None"
    return output[:max_output_chars]
