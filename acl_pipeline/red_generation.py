from __future__ import annotations

import json
import re
import ast
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from .logging_utils import StructuredLogger
from .prompts import build_red_messages
from .schemas import PythonTask, RedTaskSpec

if TYPE_CHECKING:
    from .modeling import RoleSession


def _cleanup_chat_artifacts(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\s*assistant\b[:\s-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _extract_json(text: str) -> Optional[Any]:
    raw = _cleanup_chat_artifacts(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    stripped = raw.replace("```json", "").replace("```", "").strip()
    for left, right in (("{", "}"), ("[", "]")):
        start = stripped.find(left)
        end = stripped.rfind(right)
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                continue
    return None


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


def _shared_test_signature(program: str) -> tuple[List[str], Optional[str]]:
    try:
        tree = ast.parse(program or "")
    except SyntaxError as exc:
        return [], f"syntax error while parsing tests: {exc.msg}"
    signature: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) or _is_pytest_raises_with(node):
            signature.append(ast.dump(node, include_attributes=False))
    return signature, None


class RedTaskGenerator:
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger

    def generate_raw_response(self, session: "RoleSession", messages: List[Dict[str, str]], *, topic: str) -> str:
        return session.generate([messages])[0]

    def parse_task_response(
        self,
        raw: str,
        *,
        requested_topic: str,
    ) -> tuple[Optional[PythonTask], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]

        topic = str(payload.get("topic") or "").strip()
        target_function = str(payload.get("target_function") or "").strip()
        intended_bug = str(payload.get("intended_bug") or "").strip()
        expected_first_failure = str(payload.get("expected_first_failure") or "").strip()
        statement = str(payload.get("statement") or "").strip()
        reference_solution = str(payload.get("reference_solution") or "").strip()
        solution = str(payload.get("buggy_solution") or "").strip()
        failing_asserts = payload.get("failing_asserts") or payload.get("asserts") or []
        if isinstance(failing_asserts, str):
            failing_asserts = [failing_asserts]
        failing_asserts = [str(item).strip() for item in failing_asserts if str(item).strip()]
        metadata = dict(payload.get("metadata") or {})
        difficulty = str(metadata.get("difficulty") or "").strip().lower()
        reasons: List[str] = []
        if not topic:
            reasons.append("missing topic")
        elif topic != requested_topic:
            reasons.append("wrong topic")
        if not target_function:
            reasons.append("missing target_function")
        if not intended_bug:
            reasons.append("missing intended_bug")
        if not expected_first_failure:
            reasons.append("missing expected_first_failure")
        if not statement:
            reasons.append("missing statement")
        if not reference_solution:
            reasons.append("missing reference_solution")
        if not solution:
            reasons.append("missing buggy_solution")
        if not str(metadata.get("failure_mode") or "").strip():
            reasons.append("missing metadata.failure_mode")
        if difficulty not in {"medium", "hard"}:
            reasons.append("invalid metadata.difficulty")
        if reference_solution and solution:
            reference_tests, reference_parse_error = _shared_test_signature(reference_solution)
            buggy_tests, buggy_parse_error = _shared_test_signature(solution)
            if reference_parse_error:
                reasons.append("reference_solution parse error")
            if buggy_parse_error:
                reasons.append("buggy_solution parse error")
            if not buggy_tests:
                reasons.append("missing shared tests in buggy_solution")
            if reference_tests and buggy_tests and reference_tests != buggy_tests:
                reasons.append("solutions do not share identical tests")
        if reasons:
            return None, list(dict.fromkeys(reasons))

        spec = RedTaskSpec(
            topic=topic,
            target_function=target_function,
            intended_bug=intended_bug,
            expected_first_failure=expected_first_failure,
            metadata=metadata,
        )

        task = PythonTask(
            task_id=uuid4().hex[:16],
            topic=topic,
            statement=statement,
            buggy_solution=solution,
            metadata=metadata,
            failing_asserts=failing_asserts,
            reference_solution=reference_solution,
        )
        task.metadata["reference_solution"] = reference_solution
        task.metadata["red_spec"] = spec.to_dict()
        task.metadata.setdefault("failure_mode", str(metadata.get("failure_mode") or intended_bug))
        task.metadata.setdefault("difficulty", difficulty)
        task.metadata.setdefault("observed_failure", "AssertionError")
        task.metadata["raw_response"] = raw
        task.metadata["red_format"] = "dual_solution_json_v1"
        self.logger.debug_dump("red_task", task=task)
        return task, []

    def parse_reference_response(
        self,
        raw: str,
        *,
        requested_topic: str,
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]

        topic = str(payload.get("topic") or "").strip()
        target_function = str(payload.get("target_function") or "").strip()
        intended_bug = str(payload.get("intended_bug") or "").strip()
        expected_first_failure = str(payload.get("expected_first_failure") or "").strip()
        statement = str(payload.get("statement") or "").strip()
        reference_solution = str(payload.get("reference_solution") or "").strip()
        metadata = dict(payload.get("metadata") or {})
        difficulty = str(metadata.get("difficulty") or "").strip().lower()
        reasons: List[str] = []

        if not topic:
            reasons.append("missing topic")
        elif topic != requested_topic:
            reasons.append("wrong topic")
        if not target_function:
            reasons.append("missing target_function")
        if not intended_bug:
            reasons.append("missing intended_bug")
        if not expected_first_failure:
            reasons.append("missing expected_first_failure")
        if not statement:
            reasons.append("missing statement")
        if not reference_solution:
            reasons.append("missing reference_solution")
        if payload.get("buggy_solution"):
            reasons.append("unexpected buggy_solution in reference stage")
        if not str(metadata.get("failure_mode") or "").strip():
            reasons.append("missing metadata.failure_mode")
        if difficulty not in {"medium", "hard"}:
            reasons.append("invalid metadata.difficulty")
        if reference_solution:
            reference_tests, reference_parse_error = _shared_test_signature(reference_solution)
            if reference_parse_error:
                reasons.append("reference_solution parse error")
            if not reference_tests:
                reasons.append("missing frozen tests in reference_solution")
        if reasons:
            return None, list(dict.fromkeys(reasons))

        normalized = {
            "topic": topic,
            "target_function": target_function,
            "intended_bug": intended_bug,
            "expected_first_failure": expected_first_failure,
            "statement": statement,
            "reference_solution": reference_solution,
            "metadata": metadata,
        }
        return normalized, []

    def generate_task(
        self,
        session: "RoleSession",
        *,
        topic: str,
        weakness_summary: Optional[str],
    ) -> Optional[PythonTask]:
        messages = build_red_messages(topic, weakness_summary)
        raw = self.generate_raw_response(session, messages, topic=topic)
        task, _ = self.parse_task_response(raw, requested_topic=topic)
        return task
