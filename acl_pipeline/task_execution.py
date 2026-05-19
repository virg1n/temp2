from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TaskExecutionConfig
from .schemas import PythonTask, TaskExecutionResult


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _failure_status(error_message: str) -> str:
    lowered = str(error_message or "").lower()
    if "indentationerror" in lowered or "taberror" in lowered:
        return "indentation_error"
    if "syntaxerror" in lowered:
        return "syntax_error"
    if "nameerror" in lowered:
        return "nameerror"
    return "failed"


def _task_language(config: TaskExecutionConfig) -> str:
    language = str(getattr(config, "language", "python") or "python").strip().lower()
    if language in {"c++", "cc", "cxx"}:
        return "cpp"
    return "cpp" if language == "cpp" else "python"


_TASK_RUNNER = '''
import runpy
import sys
import types


class _RaisesContext:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("DID NOT RAISE")
        expected = self.expected
        try:
            if isinstance(expected, tuple):
                return any(issubclass(exc_type, item) for item in expected)
            return issubclass(exc_type, expected)
        except Exception:
            return False


pytest_stub = types.SimpleNamespace(raises=lambda expected: _RaisesContext(expected))
sys.modules.setdefault("pytest", pytest_stub)
runpy.run_path(sys.argv[1], run_name="__main__")
'''


def _execute_python_program(
    program: str,
    config: TaskExecutionConfig,
    *,
    timeout_seconds: Optional[int] = None,
) -> TaskExecutionResult:
    program = str(program or "").rstrip() + "\n"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="acl_task_") as temp_dir:
        script_path = Path(temp_dir) / "task.py"
        runner_path = Path(temp_dir) / "runner.py"
        script_path.write_text(program, encoding="utf-8")
        runner_path.write_text(_TASK_RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [config.python_executable, "-I", "-B", str(runner_path), str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds if timeout_seconds is not None else config.timeout_seconds,
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
                status=_failure_status(error_message),
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
                error_message=(
                    "TimeoutError: task execution exceeded "
                    f"{timeout_seconds if timeout_seconds is not None else config.timeout_seconds} seconds."
                ),
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )


def _compiler_is_msvc(compiler: str) -> bool:
    name = Path(str(compiler or "")).name.lower()
    return name in {"cl", "cl.exe"}


def _cpp_compile_command(source_path: Path, exe_path: Path, config: TaskExecutionConfig) -> List[str]:
    compiler = str(getattr(config, "cpp_compiler", "") or "g++")
    standard = str(getattr(config, "cpp_standard", "") or "").strip()
    extra_args = [str(item) for item in (getattr(config, "cpp_compile_args", []) or []) if str(item).strip()]
    if _compiler_is_msvc(compiler):
        args = [compiler, "/nologo"]
        if standard:
            args.append(f"/std:{standard}")
        args.extend(extra_args)
        args.extend([str(source_path), f"/Fe:{exe_path}"])
        return args

    args = [compiler]
    if standard and not any(arg.startswith("-std=") for arg in extra_args):
        args.append(f"-std={standard}")
    args.extend(extra_args)
    args.extend([str(source_path), "-o", str(exe_path)])
    return args


def _normalize_cpp_diagnostics(text: str, *, temp_dir: str, max_chars: int) -> str:
    raw = _truncate(text, max_chars)
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    temp_normalized = str(temp_dir).replace("\\", "/").rstrip("/")
    if temp_normalized:
        normalized = normalized.replace(temp_normalized + "/", "")
    normalized = re.sub(r"(?m)^[^\n]*task\.cpp", "task.cpp", normalized)
    return normalized.strip()


def _format_cpp_compile_error(proc: subprocess.CompletedProcess[str], *, temp_dir: str, config: TaskExecutionConfig) -> str:
    stdout = _normalize_cpp_diagnostics(proc.stdout or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
    stderr = _normalize_cpp_diagnostics(proc.stderr or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
    diagnostic = stderr or stdout or f"Compiler exited with return code {proc.returncode}."
    return "Compilation failed:\n" + diagnostic


def _cpp_failure_status(error_message: str) -> str:
    lowered = str(error_message or "").lower()
    if "assertion" in lowered or "assert failed" in lowered:
        return "failed"
    if "segmentation fault" in lowered or "access violation" in lowered:
        return "runtime_error"
    return "failed"


def _execute_cpp_program(
    program: str,
    config: TaskExecutionConfig,
    *,
    timeout_seconds: Optional[int] = None,
) -> TaskExecutionResult:
    program = str(program or "").rstrip() + "\n"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="acl_cpp_task_") as temp_dir:
        source_path = Path(temp_dir) / "task.cpp"
        exe_path = Path(temp_dir) / ("task.exe" if os.name == "nt" else "task")
        source_path.write_text(program, encoding="utf-8")
        compile_timeout = int(getattr(config, "compile_timeout_seconds", 0) or config.timeout_seconds)
        run_timeout = timeout_seconds if timeout_seconds is not None else config.timeout_seconds
        compile_cmd = _cpp_compile_command(source_path, exe_path, config)
        try:
            compile_proc = subprocess.run(
                compile_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=compile_timeout,
                check=False,
            )
        except OSError as exc:
            duration = time.perf_counter() - start
            command_text = " ".join(shlex.quote(part) for part in compile_cmd)
            return TaskExecutionResult(
                status="compile_error",
                returncode=-1,
                error_message=f"Compilation failed:\nUnable to run C++ compiler. Command: {command_text}\n{type(exc).__name__}: {exc}",
                stdout="",
                stderr=str(exc),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            stdout = _normalize_cpp_diagnostics((exc.stdout or ""), temp_dir=temp_dir, max_chars=config.capture_max_chars)
            stderr = _normalize_cpp_diagnostics((exc.stderr or ""), temp_dir=temp_dir, max_chars=config.capture_max_chars)
            return TaskExecutionResult(
                status="compile_error",
                returncode=-9,
                error_message=f"Compilation timed out after {compile_timeout} seconds.",
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        if compile_proc.returncode != 0:
            duration = time.perf_counter() - start
            stdout = _normalize_cpp_diagnostics(compile_proc.stdout or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
            stderr = _normalize_cpp_diagnostics(compile_proc.stderr or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
            return TaskExecutionResult(
                status="compile_error",
                returncode=int(compile_proc.returncode),
                error_message=_format_cpp_compile_error(compile_proc, temp_dir=temp_dir, config=config),
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        if not exe_path.exists():
            duration = time.perf_counter() - start
            command_text = " ".join(shlex.quote(part) for part in compile_cmd)
            return TaskExecutionResult(
                status="compile_error",
                returncode=-1,
                error_message=f"Compilation produced no executable: {exe_path.name}. Command: {command_text}",
                stdout=_normalize_cpp_diagnostics(compile_proc.stdout or "", temp_dir=temp_dir, max_chars=config.capture_max_chars),
                stderr=_normalize_cpp_diagnostics(compile_proc.stderr or "", temp_dir=temp_dir, max_chars=config.capture_max_chars),
                duration_seconds=duration,
            )

        try:
            proc = subprocess.run(
                [str(exe_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=run_timeout,
                check=False,
            )
            duration = time.perf_counter() - start
            stdout = _normalize_cpp_diagnostics(proc.stdout or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
            stderr = _normalize_cpp_diagnostics(proc.stderr or "", temp_dir=temp_dir, max_chars=config.capture_max_chars)
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
                status=_cpp_failure_status(error_message),
                returncode=int(proc.returncode),
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            stdout = _normalize_cpp_diagnostics((exc.stdout or ""), temp_dir=temp_dir, max_chars=config.capture_max_chars)
            stderr = _normalize_cpp_diagnostics((exc.stderr or ""), temp_dir=temp_dir, max_chars=config.capture_max_chars)
            return TaskExecutionResult(
                status="timeout",
                returncode=-9,
                error_message=f"TimeoutError: task execution exceeded {run_timeout} seconds.",
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )


def execute_program(
    program: str,
    config: TaskExecutionConfig,
    *,
    timeout_seconds: Optional[int] = None,
) -> TaskExecutionResult:
    if _task_language(config) == "cpp":
        return _execute_cpp_program(program, config, timeout_seconds=timeout_seconds)
    return _execute_python_program(program, config, timeout_seconds=timeout_seconds)


def execute_task(task: PythonTask, config: TaskExecutionConfig) -> TaskExecutionResult:
    return execute_program(task.combined_program(), config)


_ASSERT_AUDIT_SCRIPT = r'''
from __future__ import annotations

import ast
import copy
import json
import sys
import types
from typing import Any


class _RaisesContext:
    def __init__(self, expected: Any) -> None:
        self.expected = expected

    def __enter__(self) -> "_RaisesContext":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc_type is None:
            raise AssertionError("DID NOT RAISE")
        expected = self.expected
        try:
            if isinstance(expected, tuple):
                return any(issubclass(exc_type, item) for item in expected)
            return issubclass(exc_type, expected)
        except Exception:
            return False


class _PytestStub(types.SimpleNamespace):
    def raises(self, expected: Any) -> _RaisesContext:
        return _RaisesContext(expected)


def _install_pytest_stub() -> _PytestStub:
    stub = _PytestStub()
    sys.modules.setdefault("pytest", stub)
    return stub


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _is_pytest_raises_with(node: ast.AST) -> bool:
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        expr = item.context_expr
        if not isinstance(expr, ast.Call):
            continue
        func = expr.func
        if isinstance(func, ast.Attribute) and func.attr == "raises":
            return True
        if isinstance(func, ast.Name) and func.id == "raises":
            return True
    return False


def _test_nodes(tree: ast.Module) -> list[ast.AST]:
    return [node for node in tree.body if isinstance(node, ast.Assert) or _is_pytest_raises_with(node)]


def _implementation_tree(source: str) -> tuple[ast.Module, list[ast.AST]]:
    tree = ast.parse(source or "")
    tests = _test_nodes(tree)
    tree.body = [node for node in tree.body if node not in tests]
    ast.fix_missing_locations(tree)
    return tree, tests


def _load_namespace(source: str) -> tuple[dict[str, Any], list[ast.AST]]:
    tree, tests = _implementation_tree(source)
    pytest_stub = _install_pytest_stub()
    ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__acl_assert_audit__",
        "pytest": pytest_stub,
    }
    exec(compile(tree, "<acl_program>", "exec"), ns)
    return ns, tests


def _snapshot(ns: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in ns.items():
        if key.startswith("__") and key != "__builtins__":
            continue
        try:
            copied[key] = copy.deepcopy(value)
        except Exception:
            copied[key] = value
    copied.setdefault("__builtins__", __builtins__)
    return copied


def _compile_expr(expr: ast.AST) -> Any:
    wrapper = ast.Expression(expr)
    ast.fix_missing_locations(wrapper)
    return compile(wrapper, "<acl_assert_expr>", "eval")


def _eval_value(expr: ast.AST, ns: dict[str, Any]) -> dict[str, Any]:
    try:
        value = eval(_compile_expr(expr), ns)
        return {"ok": True, "value": value, "repr": repr(value)}
    except BaseException as exc:
        return {
            "ok": False,
            "value": None,
            "repr": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
        }


def _public_result(result: dict[str, Any]) -> str:
    return str(result.get("repr", ""))


def _same_result(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if bool(left.get("ok")) != bool(right.get("ok")):
        return False
    if not bool(left.get("ok")):
        return str(left.get("exception_type")) == str(right.get("exception_type")) and str(left.get("repr")) == str(right.get("repr"))
    try:
        equal = left.get("value") == right.get("value")
        if isinstance(equal, bool):
            return equal
    except Exception:
        pass
    return str(left.get("repr")) == str(right.get("repr"))


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)


def _first_call(node: ast.AST, *, skip_outer_isinstance: bool = False) -> ast.Call | None:
    if skip_outer_isinstance and isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            for arg in node.args:
                found = _first_call(arg)
                if found is not None:
                    return found
            return None
    collector = _CallCollector()
    collector.visit(node)
    if not collector.calls:
        return None
    return collector.calls[0]


def _call_expr_from_assert(node: ast.Assert) -> ast.AST | None:
    expr = node.test
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        if isinstance(expr.ops[0], (ast.Eq, ast.NotEq, ast.In)):
            if isinstance(expr.left, ast.Call):
                return expr.left
            if isinstance(expr.comparators[0], ast.Call):
                return expr.comparators[0]
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "isinstance":
        if expr.args:
            first = expr.args[0]
            return first if isinstance(first, ast.Call) else _first_call(first)
    return _first_call(expr, skip_outer_isinstance=True)


def _call_expr_from_raises(node: ast.With) -> ast.AST | None:
    for child in node.body:
        call = _first_call(child)
        if call is not None:
            return call
    return None


def _record_diff(index: int, expr: ast.AST, ref_ns: dict[str, Any], buggy_ns: dict[str, Any]) -> dict[str, Any]:
    ref = _eval_value(expr, _snapshot(ref_ns))
    buggy = _eval_value(expr, _snapshot(buggy_ns))
    return {
        "kind": "behavior_diff",
        "test_index": index,
        "expression": _unparse(expr),
        "args_repr": _unparse(expr),
        "reference_value": _public_result(ref),
        "buggy_value": _public_result(buggy),
        "equal": _same_result(ref, buggy),
    }


def _record_bool_diff(index: int, expr: ast.AST, ref_ns: dict[str, Any], buggy_ns: dict[str, Any]) -> dict[str, Any]:
    ref = _eval_value(expr, _snapshot(ref_ns))
    buggy = _eval_value(expr, _snapshot(buggy_ns))
    return {
        "kind": "behavior_diff_boolean",
        "test_index": index,
        "expression": _unparse(expr),
        "args_repr": _unparse(expr),
        "reference_value": _public_result(ref),
        "buggy_value": _public_result(buggy),
        "equal": _same_result(ref, buggy),
    }


def behavior_diff(reference_source: str, buggy_source: str) -> list[dict[str, Any]]:
    ref_ns, _ = _load_namespace(reference_source)
    buggy_ns, tests = _load_namespace(buggy_source)
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(tests, start=1):
        if isinstance(node, ast.Assert):
            expr = _call_expr_from_assert(node)
            rows.append(_record_diff(index, expr, ref_ns, buggy_ns) if expr is not None else _record_bool_diff(index, node.test, ref_ns, buggy_ns))
        elif _is_pytest_raises_with(node):
            expr = _call_expr_from_raises(node)
            if expr is not None:
                rows.append(_record_diff(index, expr, ref_ns, buggy_ns))
            else:
                rows.append(
                    {
                        "kind": "unsupported",
                        "test_index": index,
                        "expression": _unparse(node),
                        "args_repr": _unparse(node),
                        "reference_value": None,
                        "buggy_value": None,
                        "equal": None,
                    }
                )
    return rows


def _operator_name(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.In):
        return "in"
    return type(op).__name__


def _fallback_assert(index: int, node: ast.Assert, ns: dict[str, Any]) -> dict[str, Any]:
    expr = node.test
    base = {
        "kind": "assert_eval",
        "test_index": index,
        "expression": _unparse(expr),
        "args_repr": _unparse(expr),
        "reference_value": None,
        "buggy_value": None,
        "equal": None,
    }
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1 and isinstance(expr.ops[0], (ast.Eq, ast.NotEq, ast.In)):
        snap = _snapshot(ns)
        left = _eval_value(expr.left, snap)
        right = _eval_value(expr.comparators[0], snap)
        base.update(
            {
                "kind": "assert_compare_eval",
                "operator": _operator_name(expr.ops[0]),
                "left_value": _public_result(left),
                "right_value": _public_result(right),
            }
        )
        return base
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "isinstance" and len(expr.args) >= 2:
        snap = _snapshot(ns)
        value = _eval_value(expr.args[0], snap)
        type_value = _eval_value(expr.args[1], snap)
        base.update(
            {
                "kind": "assert_isinstance_eval",
                "left_value": _public_result(value),
                "right_value": _public_result(type_value),
            }
        )
        return base
    value = _eval_value(expr, _snapshot(ns))
    base["boolean_value"] = _public_result(value)
    return base


def _fallback_raises(index: int, node: ast.With, ns: dict[str, Any]) -> dict[str, Any]:
    expr = _call_expr_from_raises(node)
    if expr is None:
        return {
            "kind": "unsupported",
            "test_index": index,
            "expression": _unparse(node),
            "args_repr": _unparse(node),
            "reference_value": None,
            "buggy_value": None,
            "equal": None,
        }
    result = _eval_value(expr, _snapshot(ns))
    return {
        "kind": "raises_eval",
        "test_index": index,
        "expression": _unparse(expr),
        "args_repr": _unparse(expr),
        "buggy_value": _public_result(result),
        "reference_value": None,
        "equal": None,
    }


def fallback_audit(source: str) -> list[dict[str, Any]]:
    ns, tests = _load_namespace(source)
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(tests, start=1):
        if isinstance(node, ast.Assert):
            rows.append(_fallback_assert(index, node, ns))
        elif _is_pytest_raises_with(node):
            rows.append(_fallback_raises(index, node, ns))
    return rows


def main() -> None:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        if payload.get("reference_solution"):
            rows = behavior_diff(str(payload.get("reference_solution") or ""), str(payload.get("buggy_solution") or ""))
        else:
            rows = fallback_audit(str(payload.get("buggy_solution") or ""))
        print(json.dumps({"results": rows}, ensure_ascii=False))
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "kind": "audit_error",
                            "test_index": "error",
                            "expression": str(exc),
                            "args_repr": None,
                            "reference_value": None,
                            "buggy_value": None,
                            "equal": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
'''


def audit_buggy_solution_asserts(task: PythonTask, config: TaskExecutionConfig) -> List[Dict[str, Any]]:
    language = str(getattr(task, "language", "") or getattr(config, "language", "python") or "python").strip().lower()
    if language in {"cpp", "c++", "cc", "cxx"}:
        return [
            {
                "kind": "unsupported_cpp_assert_audit",
                "test_index": "unsupported",
                "expression": "C++ assert auditing is not implemented; use compiler/runtime diagnostics.",
                "args_repr": None,
                "reference_value": None,
                "buggy_value": None,
                "equal": None,
            }
        ]
    payload = {
        "reference_solution": str(task.reference_solution or task.metadata.get("reference_solution") or ""),
        "buggy_solution": task.combined_program(),
    }
    with tempfile.TemporaryDirectory(prefix="acl_assert_audit_") as temp_dir:
        temp_path = Path(temp_dir)
        script_path = temp_path / "audit.py"
        payload_path = temp_path / "payload.json"
        script_path.write_text(_ASSERT_AUDIT_SCRIPT, encoding="utf-8")
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            proc = subprocess.run(
                [config.python_executable, "-I", "-B", str(script_path), str(payload_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [
                {
                    "kind": "timeout",
                    "test_index": "timeout",
                    "expression": "timeout",
                    "args_repr": None,
                    "reference_value": None,
                    "buggy_value": None,
                    "equal": None,
                }
            ]
    try:
        parsed = json.loads(proc.stdout or "{}")
        results = parsed.get("results")
        if isinstance(results, list):
            return [dict(item) for item in results if isinstance(item, dict)]
    except Exception:
        pass
    return [
        {
            "kind": "audit_error",
            "test_index": "error",
            "expression": _truncate((proc.stderr or proc.stdout or "assert audit failed"), config.capture_max_chars),
            "args_repr": None,
            "reference_value": None,
            "buggy_value": None,
            "equal": None,
        }
    ]
