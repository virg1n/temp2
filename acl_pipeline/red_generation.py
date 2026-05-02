from __future__ import annotations

import json
import re
import ast
import io
import tokenize
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from .logging_utils import StructuredLogger
from .prompts import build_red_buggy_messages, build_red_reference_messages, build_red_task_description_messages
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


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_LABEL_RE = re.compile(
    r"(?im)^\s*(TOPIC|TARGET_FUNCTION|DIFFICULTY|FAILURE_MODE|INTENDED_BUG|EXPECTED_FIRST_FAILURE|STATEMENT)\s*:\s*(.*)$"
)


def _strip_python_comments(source: str) -> str:
    code = str(source or "")
    if "#" not in code:
        return code.strip()

    output: List[str] = []
    last_line = 1
    last_col = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            token_type = token.type
            token_text = token.string
            start_line, start_col = token.start
            end_line, end_col = token.end
            line_text = token.line

            if token_type == tokenize.COMMENT:
                last_line = end_line
                last_col = end_col
                continue
            if token_type == tokenize.ENDMARKER:
                break

            if start_line > last_line:
                output.append("\n" * (start_line - last_line))
                last_col = 0
            if start_col > last_col:
                output.append(line_text[last_col:start_col])
            output.append(token_text)
            if token_type in {tokenize.NEWLINE, tokenize.NL}:
                last_line = start_line + 1
                last_col = 0
            else:
                last_line = end_line
                last_col = end_col
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code.strip()

    cleaned = "".join(output)
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def _clean_code_response(text: str) -> str:
    raw = _cleanup_chat_artifacts(text)
    match = _CODE_FENCE_RE.search(raw)
    if match:
        raw = match.group(1)
    raw = raw.replace("```python", "").replace("```py", "").replace("```", "").strip()
    lines = raw.splitlines()
    first_code_index: Optional[int] = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(from\s+\S+\s+import\s+|import\s+|def\s+|class\s+|@)", stripped):
            first_code_index = index
            break
        if stripped.startswith("#") and first_code_index is None:
            first_code_index = index
            break
    if first_code_index is not None and first_code_index > 0:
        raw = "\n".join(lines[first_code_index:])
    return _strip_python_comments(raw)


def _plain_code_from_response(raw: str, *, preferred_key: str) -> str:
    payload = _extract_json(raw)
    if isinstance(payload, dict):
        for key in (preferred_key, "reference_solution", "buggy_solution", "code", "program"):
            value = str(payload.get(key) or "").strip()
            if value:
                return _clean_code_response(value)
    return _clean_code_response(raw)


def _extract_labeled_fields(raw: str) -> Dict[str, str]:
    text = _cleanup_chat_artifacts(raw)
    matches = list(_LABEL_RE.finditer(text))
    fields: Dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        same_line = match.group(2) or ""
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = (same_line + text[start:end]).strip()
        fields[label] = value
    return fields


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return cleaned[:80] or "semantic_bug"


def _infer_target_function(statement: str) -> str:
    text = str(statement or "")
    for pattern in (
        r"`([A-Za-z_][A-Za-z0-9_]*)`",
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


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


def _test_count_reasons(program: str, *, min_tests: int = 3, max_tests: int = 6) -> List[str]:
    tests, parse_error = _shared_test_signature(program)
    if parse_error:
        return ["reference_solution parse error"]
    count = len(tests)
    if count < int(min_tests):
        return ["too few tests in reference_solution"]
    if count > int(max_tests):
        return ["too many tests in reference_solution"]
    return []


_TASK_FILE_LINE_RE = re.compile(r'File "[^"\n]*task\.py", line (\d+)')


def _task_file_lines_from_traceback(error_message: str) -> List[int]:
    lines: List[int] = []
    for match in _TASK_FILE_LINE_RE.finditer(str(error_message or "")):
        try:
            lines.append(int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return lines


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


def relax_reference_drop_failing_assert(
    program: str,
    error_message: str,
    *,
    min_remaining_asserts: int = 2,
) -> Optional[str]:
    """If exactly one assert is implicated by the traceback, return the program
    with that assert disabled. Otherwise return None.

    The caller still executes the relaxed program; this helper only creates a
    candidate when a single assert failure is identifiable.
    """
    program_text = str(program or "")
    traceback_lines = _task_file_lines_from_traceback(error_message)
    if not traceback_lines:
        return None
    spans = _assert_spans(program_text)
    if len(spans) < min_remaining_asserts + 1:
        return None
    matching = [
        (start, end)
        for start, end in spans
        if any(start <= line <= end for line in traceback_lines)
    ]
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
        "reference_solution": _clean_code_response(str(payload.get("reference_solution") or "")),
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
            if locked_spec is None:
                return None, ["non-json response"]
            payload = {
                "topic": locked_spec.get("topic", requested_topic),
                "target_function": locked_spec.get("target_function", ""),
                "intended_bug": locked_spec.get("intended_bug", ""),
                "expected_first_failure": locked_spec.get("expected_first_failure", ""),
                "statement": locked_spec.get("statement", ""),
                "reference_solution": locked_spec.get("reference_solution", ""),
                "buggy_solution": _plain_code_from_response(raw, preferred_key="buggy_solution"),
                "metadata": dict(locked_spec.get("metadata") or {}),
            }
            plain_code_response = True
        else:
            plain_code_response = False

        topic = str(payload.get("topic") or "").strip()
        target_function = str(payload.get("target_function") or "").strip()
        intended_bug = str(payload.get("intended_bug") or "").strip()
        expected_first_failure = str(payload.get("expected_first_failure") or "").strip()
        statement = str(payload.get("statement") or "").strip()
        reference_solution = _clean_code_response(str(payload.get("reference_solution") or ""))
        solution = _clean_code_response(str(payload.get("buggy_solution") or ""))
        payload["reference_solution"] = reference_solution
        payload["buggy_solution"] = solution
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
        task.metadata["red_format"] = (
            "iterative_plain_code_v3"
            if plain_code_response
            else "iterative_dual_solution_json_v2"
            if locked_spec is not None
            else "dual_solution_json_v1"
        )
        task.metadata["red_chosen_completion"] = solution
        task.metadata["red_buggy_chosen_completion"] = solution
        task.metadata["red_buggy_raw_response"] = raw
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
            fields = _extract_labeled_fields(raw)
            statement = fields.get("STATEMENT", "").strip()
            target_function = fields.get("TARGET_FUNCTION", "").strip() or _infer_target_function(statement)
            intended_bug = fields.get("INTENDED_BUG", "").strip()
            expected_first_failure = fields.get("EXPECTED_FIRST_FAILURE", "").strip()
            if not expected_first_failure and target_function:
                expected_first_failure = f"AssertionError in tests for {target_function}"
            difficulty = fields.get("DIFFICULTY", "").strip().lower()
            if difficulty not in {"medium", "hard"}:
                difficulty = "medium"
            failure_mode = fields.get("FAILURE_MODE", "").strip() or _slug(intended_bug)
            payload = {
                "topic": fields.get("TOPIC", requested_topic).strip() or requested_topic,
                "target_function": target_function,
                "intended_bug": intended_bug,
                "expected_first_failure": expected_first_failure,
                "statement": statement,
                "metadata": {
                    "failure_mode": failure_mode,
                    "difficulty": difficulty,
                },
            }
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
            payload = {
                "topic": locked_spec.get("topic", requested_topic),
                "target_function": locked_spec.get("target_function", ""),
                "intended_bug": locked_spec.get("intended_bug", ""),
                "expected_first_failure": locked_spec.get("expected_first_failure", ""),
                "statement": locked_spec.get("statement", ""),
                "reference_solution": _plain_code_from_response(raw, preferred_key="reference_solution"),
                "metadata": dict(locked_spec.get("metadata") or {}),
            }
        else:
            payload = {
                "topic": locked_spec.get("topic", requested_topic),
                "target_function": locked_spec.get("target_function", payload.get("target_function", "")),
                "intended_bug": locked_spec.get("intended_bug", payload.get("intended_bug", "")),
                "expected_first_failure": locked_spec.get("expected_first_failure", payload.get("expected_first_failure", "")),
                "statement": locked_spec.get("statement", payload.get("statement", "")),
                "reference_solution": _plain_code_from_response(raw, preferred_key="reference_solution"),
                "metadata": dict(locked_spec.get("metadata") or payload.get("metadata") or {}),
            }
        reasons = _required_spec_reasons(payload, requested_topic, require_reference=True)
        reasons.extend(_test_count_reasons(str(payload.get("reference_solution") or ""), min_tests=3, max_tests=6))
        if str(payload.get("buggy_solution") or "").strip():
            reasons.append("reference stage included buggy_solution")
        if reasons:
            return None, list(dict.fromkeys(reasons))
        spec_payload = _normalized_spec_payload(payload, requested_topic, raw)
        spec_payload["reference_chosen_completion"] = spec_payload["reference_solution"]
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
        raw_description = self.generate_raw_response(
            session,
            build_red_task_description_messages(topic, weakness_summary),
            topic=topic,
        )
        description_payload, reasons = self.parse_task_description_response(raw_description, requested_topic=topic)
        if description_payload is None or reasons:
            return None
        raw_reference = self.generate_raw_response(
            session,
            build_red_reference_messages(topic, description_payload),
            topic=topic,
        )
        spec_payload, reasons = self.parse_reference_response(
            raw_reference,
            requested_topic=topic,
            locked_spec=description_payload,
        )
        if spec_payload is None or reasons:
            return None
        raw = self.generate_raw_response(session, build_red_buggy_messages(topic, spec_payload), topic=topic)
        task, _ = self.parse_task_response(raw, requested_topic=topic, locked_spec=spec_payload)
        return task
