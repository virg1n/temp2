from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from .config import TaskExecutionConfig
from .schemas import PythonTask, TaskExecutionResult


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def execute_task(task: PythonTask, config: TaskExecutionConfig) -> TaskExecutionResult:
    program = task.combined_program().rstrip() + "\n"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="acl_task_") as temp_dir:
        script_path = Path(temp_dir) / "task.py"
        script_path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [config.python_executable, "-I", "-B", str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.timeout_seconds,
                check=False,
            )
            duration = time.perf_counter() - start
            stdout = _truncate(proc.stdout, config.capture_max_chars)
            stderr = _truncate(proc.stderr, config.capture_max_chars)
            if proc.returncode == 0:
                return TaskExecutionResult(
                    status="passed",
                    returncode=0,
                    error_message="Program exited successfully. No failing assertion or runtime error was reproduced.",
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                )
            error_message = stderr or stdout or f"Process failed with return code {proc.returncode}."
            return TaskExecutionResult(
                status="failed",
                returncode=int(proc.returncode),
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            stdout = _truncate((exc.stdout or ""), config.capture_max_chars)
            stderr = _truncate((exc.stderr or ""), config.capture_max_chars)
            return TaskExecutionResult(
                status="timeout",
                returncode=-9,
                error_message=f"TimeoutError: task execution exceeded {config.timeout_seconds} seconds.",
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )


def task_meets_shape_requirements(task: PythonTask, config: TaskExecutionConfig) -> bool:
    code_lines = [line for line in task.buggy_solution.splitlines() if line.strip()]
    assert_lines = [line for line in task.failing_asserts if line.strip().startswith("assert ")]
    return len(code_lines) >= config.min_code_lines and len(assert_lines) >= config.min_asserts
