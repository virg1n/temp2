from __future__ import annotations

import ast
import subprocess
import tempfile
from pathlib import Path

from .config import ExecutionConfig
from .schemas import RedTask, ValidationResult


FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
}

FORBIDDEN_IMPORT_ROOTS = {
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


def _parse_or_reject(code: str, label: str) -> tuple[ast.AST | None, list[str]]:
    try:
        return ast.parse(code), []
    except SyntaxError as exc:
        return None, [f"{label} has a syntax error: {exc.msg} at line {exc.lineno}"]


def static_safety_check(code: str, label: str, cfg: ExecutionConfig) -> list[str]:
    reasons: list[str] = []
    if len(code) > cfg.max_code_chars:
        reasons.append(f"{label} is too long ({len(code)} chars)")
        return reasons

    tree, parse_reasons = _parse_or_reject(code, label)
    reasons.extend(parse_reasons)
    if tree is None:
        return reasons

    allowed_imports = set(cfg.allowed_imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS or root not in allowed_imports:
                    reasons.append(f"{label} imports disallowed module '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS or root not in allowed_imports:
                reasons.append(f"{label} imports disallowed module '{node.module}'")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                reasons.append(f"{label} calls disallowed function '{func.id}'")
            elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALLS:
                reasons.append(f"{label} calls disallowed method '{func.attr}'")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            reasons.append(f"{label} uses dunder attribute '{node.attr}'")
    return reasons


def build_script(solution: str, asserts: list[str]) -> str:
    assert_block = "\n".join(asserts)
    return f"{solution.rstrip()}\n\n# ACT validation asserts\n{assert_block}\n"


def run_python(script: str, cfg: ExecutionConfig) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="act_task_") as tmpdir:
        script_path = Path(tmpdir) / "task.py"
        script_path.write_text(script, encoding="utf-8")
        return subprocess.run(
            [cfg.python_executable, "-I", str(script_path)],
            cwd=tmpdir,
            text=True,
            capture_output=True,
            timeout=cfg.timeout_seconds,
            check=False,
        )


def _failure_text(result: subprocess.CompletedProcess[str]) -> str:
    combined = "\n".join(part for part in [result.stderr.strip(), result.stdout.strip()] if part)
    if combined:
        return combined[-4000:]
    return f"Program exited with status {result.returncode} without a detailed message."


def validate_task(task: RedTask, cfg: ExecutionConfig) -> ValidationResult:
    reasons: list[str] = []
    if not task.asserts:
        reasons.append("No asserts were provided.")
    for idx, line in enumerate(task.asserts, start=1):
        if not line.strip().startswith("assert "):
            reasons.append(f"Assert {idx} is not an assert statement.")

    reference_script = build_script(task.reference_solution, task.asserts)
    buggy_script = build_script(task.buggy_solution, task.asserts)
    reasons.extend(static_safety_check(reference_script, "reference_solution", cfg))
    reasons.extend(static_safety_check(buggy_script, "buggy_solution", cfg))
    if reasons:
        return ValidationResult(valid=False, rejection_reasons=reasons)

    try:
        reference = run_python(reference_script, cfg)
    except subprocess.TimeoutExpired:
        return ValidationResult(valid=False, rejection_reasons=["reference_solution timed out"])

    if reference.returncode != 0:
        return ValidationResult(
            valid=False,
            rejection_reasons=["reference_solution fails its asserts"],
            reference_stdout=reference.stdout,
            reference_stderr=reference.stderr,
        )

    try:
        buggy = run_python(buggy_script, cfg)
    except subprocess.TimeoutExpired:
        return ValidationResult(
            valid=True,
            error_message="Program timed out while running the failing tests.",
            reference_stdout=reference.stdout,
            reference_stderr=reference.stderr,
            buggy_stderr="TimeoutExpired",
        )

    if buggy.returncode == 0:
        return ValidationResult(
            valid=False,
            rejection_reasons=["buggy_solution passes all asserts"],
            reference_stdout=reference.stdout,
            reference_stderr=reference.stderr,
            buggy_stdout=buggy.stdout,
            buggy_stderr=buggy.stderr,
        )

    return ValidationResult(
        valid=True,
        error_message=_failure_text(buggy),
        reference_stdout=reference.stdout,
        reference_stderr=reference.stderr,
        buggy_stdout=buggy.stdout,
        buggy_stderr=buggy.stderr,
    )
