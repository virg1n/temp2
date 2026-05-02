from __future__ import annotations

import json
import re
import ast
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from .logging_utils import StructuredLogger
from .prompts import build_red_buggy_messages, build_red_messages
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


_TASK_FILE_LINE_RE = re.compile(r'File "[^"\n]*task\.py", line (\d+)')


def _assert_spans(program: str) -> List[tuple[int, int]]:
    """Return (start_line, end_line) for each assert in the module."""
    try:
        tree = ast.parse(program or "")
    except SyntaxError:
        return []
    spans: List[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            if start and end:
                spans.append((int(start), int(end)))
    return spans


def _failed_line_from_traceback(error_message: str) -> Optional[int]:
    """Extract the LAST 'File task.py, line N' from the traceback (innermost frame)."""
    matches = _TASK_FILE_LINE_RE.findall(str(error_message or ""))
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (TypeError, ValueError):
        return None


def relax_reference_drop_failing_assert(
    program: str,
    error_message: str,
    *,
    min_remaining_asserts: int = 2,
) -> Optional[str]:
    """If exactly one assert contains the failing line, return the program with that
    assert disabled. Otherwise return None.

    The caller still executes the relaxed program; this helper only creates a
    candidate when a single assert failure is identifiable.
    """
    program_text = str(program or "")
    failed_line = _failed_line_from_traceback(error_message)
    if failed_line is None:
        return None
    spans = _assert_spans(program_text)
    if len(spans) < min_remaining_asserts + 1:
        return None
    matching = [(s, e) for s, e in spans if s <= failed_line <= e]
    if len(matching) != 1:
        return None
    drop_start, _ = matching[0]
    try:
        tree = ast.parse(program_text)
    except SyntaxError:
        return None

    class _DropFailingAssert(ast.NodeTransformer):
        def visit_Assert(self, node: ast.Assert) -> ast.AST:
            if int(getattr(node, "lineno", -1)) == drop_start:
                replacement = ast.Pass()
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    relaxed = _DropFailingAssert().visit(tree)
    ast.fix_missing_locations(relaxed)
    try:
        return ast.unparse(relaxed)
    except Exception:
        return None


def _payload_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    if "failure_mode" not in metadata and payload.get("intended_bug"):
        metadata["failure_mode"] = str(payload.get("intended_bug") or "")
    return metadata


def _required_spec_reasons(
    payload: Dict[str, Any],
    requested_topic: str,
    *,
    require_reference: bool = True,
) -> List[str]:
    topic = str(payload.get("topic") or "").strip()
    metadata = _payload_metadata(payload)
    difficulty = str(metadata.get("difficulty") or "").strip().lower()
    reference_solution = str(payload.get("reference_solution") or "").strip()
    reasons: List[str] = []
    if not topic:
        reasons.append("missing topic")
    elif topic != requested_topic:
        reasons.append("wrong topic")
    for key in ("target_function", "intended_bug", "expected_first_failure", "statement"):
        if not str(payload.get(key) or "").strip():
            reasons.append(f"missing {key}")
    if require_reference and not reference_solution:
        reasons.append("missing reference_solution")
    if not str(metadata.get("failure_mode") or "").strip():
        reasons.append("missing metadata.failure_mode")
    if difficulty not in {"medium", "hard"}:
        reasons.append("invalid metadata.difficulty")
    if reference_solution:
        reference_tests, reference_parse_error = _shared_test_signature(reference_solution)
        if reference_parse_error:
            reasons.append("reference_solution parse error")
        if not reference_tests:
            reasons.append("missing tests in reference_solution")
    return list(dict.fromkeys(reasons))


def _normalized_spec_payload(payload: Dict[str, Any], requested_topic: str, raw: str) -> Dict[str, Any]:
    metadata = _payload_metadata(payload)
    difficulty = str(metadata.get("difficulty") or "").strip().lower()
    metadata["difficulty"] = difficulty
    return {
        "topic": str(payload.get("topic") or requested_topic).strip(),
        "target_function": str(payload.get("target_function") or "").strip(),
        "intended_bug": str(payload.get("intended_bug") or "").strip(),
        "expected_first_failure": str(payload.get("expected_first_failure") or "").strip(),
        "statement": str(payload.get("statement") or "").strip(),
        "reference_solution": str(payload.get("reference_solution") or "").strip(),
        "metadata": metadata,
        "raw_response": raw,
    }


def _normalized_description_payload(payload: Dict[str, Any], requested_topic: str, raw: str) -> Dict[str, Any]:
    metadata = _payload_metadata(payload)
    difficulty = str(metadata.get("difficulty") or "").strip().lower()
    metadata["difficulty"] = difficulty
    return {
        "topic": str(payload.get("topic") or requested_topic).strip(),
        "target_function": str(payload.get("target_function") or "").strip(),
        "intended_bug": str(payload.get("intended_bug") or "").strip(),
        "expected_first_failure": str(payload.get("expected_first_failure") or "").strip(),
        "statement": str(payload.get("statement") or "").strip(),
        "metadata": metadata,
        "raw_response": raw,
    }


def _same_text(left: Any, right: Any) -> bool:
    return str(left or "").strip() == str(right or "").strip()


def _locked_spec_reasons(payload: Dict[str, Any], locked_spec: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for key in ("topic", "target_function", "intended_bug", "expected_first_failure", "statement"):
        if not _same_text(payload.get(key), locked_spec.get(key)):
            reasons.append(f"changed {key}")

    if not _same_text(payload.get("reference_solution"), locked_spec.get("reference_solution")):
        reasons.append("changed reference_solution")

    locked_metadata = dict(locked_spec.get("metadata") or {})
    metadata = _payload_metadata(payload)
    for key in ("failure_mode", "difficulty"):
        if not _same_text(metadata.get(key), locked_metadata.get(key)):
            reasons.append(f"changed metadata.{key}")

    locked_tests, locked_parse_error = _shared_test_signature(str(locked_spec.get("reference_solution") or ""))
    buggy_tests, buggy_parse_error = _shared_test_signature(str(payload.get("buggy_solution") or ""))
    if locked_parse_error:
        reasons.append("locked reference_solution parse error")
    if buggy_parse_error:
        reasons.append("buggy_solution parse error")
    if locked_tests and buggy_tests and locked_tests != buggy_tests:
        reasons.append("changed tests")
    return list(dict.fromkeys(reasons))


def _locked_description_reasons(payload: Dict[str, Any], locked_spec: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for key in ("topic", "target_function", "intended_bug", "expected_first_failure", "statement"):
        if not _same_text(payload.get(key), locked_spec.get(key)):
            reasons.append(f"changed {key}")

    locked_metadata = dict(locked_spec.get("metadata") or {})
    metadata = _payload_metadata(payload)
    for key in ("failure_mode", "difficulty"):
        if not _same_text(metadata.get(key), locked_metadata.get(key)):
            reasons.append(f"changed metadata.{key}")
    return list(dict.fromkeys(reasons))


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
        locked_spec: Optional[Dict[str, Any]] = None,
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
        metadata = _payload_metadata(payload)
        difficulty = str(metadata.get("difficulty") or "").strip().lower()
        metadata["difficulty"] = difficulty
        reasons: List[str] = _required_spec_reasons(payload, requested_topic)
        if not solution:
            reasons.append("missing buggy_solution")
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
        if locked_spec is not None:
            reasons.extend(_locked_spec_reasons(payload, locked_spec))
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
        task.metadata["red_format"] = "iterative_dual_solution_json_v2" if locked_spec is not None else "dual_solution_json_v1"
        if locked_spec is not None:
            task.metadata["red_locked_spec"] = {
                key: value
                for key, value in locked_spec.items()
                if key in {"topic", "target_function", "intended_bug", "expected_first_failure", "statement", "reference_solution", "metadata"}
            }
        self.logger.debug_dump("red_task", task=task)
        return task, []

    def parse_task_description_response(
        self,
        raw: str,
        *,
        requested_topic: str,
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]
        reasons = _required_spec_reasons(payload, requested_topic, require_reference=False)
        if str(payload.get("reference_solution") or "").strip():
            reasons.append("description stage included reference_solution")
        if str(payload.get("buggy_solution") or "").strip():
            reasons.append("description stage included buggy_solution")
        if reasons:
            return None, list(dict.fromkeys(reasons))
        spec_payload = _normalized_description_payload(payload, requested_topic, raw)
        self.logger.debug_dump("red_task_description", spec=spec_payload)
        return spec_payload, []

    def parse_reference_response(
        self,
        raw: str,
        *,
        requested_topic: str,
        locked_spec: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]
        reasons = _required_spec_reasons(payload, requested_topic, require_reference=True)
        reasons.extend(_locked_description_reasons(payload, locked_spec))
        if str(payload.get("buggy_solution") or "").strip():
            reasons.append("reference stage included buggy_solution")
        if reasons:
            return None, list(dict.fromkeys(reasons))
        spec_payload = _normalized_spec_payload(payload, requested_topic, raw)
        self.logger.debug_dump("red_reference", spec=spec_payload)
        return spec_payload, []

    def parse_spec_response(
        self,
        raw: str,
        *,
        requested_topic: str,
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]
        reasons = _required_spec_reasons(payload, requested_topic, require_reference=True)
        if str(payload.get("buggy_solution") or "").strip():
            reasons.append("spec stage included buggy_solution")
        if reasons:
            return None, list(dict.fromkeys(reasons))
        spec_payload = _normalized_spec_payload(payload, requested_topic, raw)
        self.logger.debug_dump("red_spec", spec=spec_payload)
        return spec_payload, []

    def generate_task(
        self,
        session: "RoleSession",
        *,
        topic: str,
        weakness_summary: Optional[str],
    ) -> Optional[PythonTask]:
        messages = build_red_messages(topic, weakness_summary)
        raw_spec = self.generate_raw_response(session, messages, topic=topic)
        spec_payload, reasons = self.parse_spec_response(raw_spec, requested_topic=topic)
        if spec_payload is None or reasons:
            return None
        raw = self.generate_raw_response(session, build_red_buggy_messages(topic, spec_payload), topic=topic)
        task, _ = self.parse_task_response(raw, requested_topic=topic, locked_spec=spec_payload)
        return task
