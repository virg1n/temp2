from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from .logging_utils import StructuredLogger
from .prompts import build_red_messages
from .schemas import PythonTask

if TYPE_CHECKING:
    from .modeling import RoleSession


def _extract_json(text: str) -> Optional[Any]:
    raw = (text or "").strip()
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


def _split_asserts_from_program(program: str) -> tuple[str, List[str]]:
    body_lines: List[str] = []
    assert_lines: List[str] = []
    for line in program.splitlines():
        if line.strip().startswith("assert "):
            assert_lines.append(line.rstrip())
        else:
            body_lines.append(line.rstrip())
    return "\n".join(body_lines).rstrip(), assert_lines


class RedTaskGenerator:
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger

    def generate_raw_response(self, session: "RoleSession", messages: List[Dict[str, str]]) -> str:
        return session.generate([messages])[0]

    def parse_task_response(self, raw: str, *, requested_topic: str) -> tuple[Optional[PythonTask], List[str]]:
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            return None, ["non-json response"]

        solution = str(payload.get("buggy_solution") or "").strip()
        failing_asserts = payload.get("failing_asserts") or payload.get("asserts") or []
        if isinstance(failing_asserts, str):
            failing_asserts = [failing_asserts]
        failing_asserts = [str(item).strip() for item in failing_asserts if str(item).strip()]

        if solution and not failing_asserts:
            solution, failing_asserts = _split_asserts_from_program(solution)

        if not solution or not failing_asserts:
            reasons: List[str] = []
            if not solution:
                reasons.append("missing buggy_solution")
            if not failing_asserts:
                reasons.append("missing failing_asserts")
            return None, reasons

        task = PythonTask(
            task_id=uuid4().hex[:16],
            topic=str(payload.get("topic") or requested_topic),
            statement=str(payload.get("statement") or f"Debug this Python task about {requested_topic}."),
            buggy_solution=solution,
            failing_asserts=failing_asserts,
            metadata=dict(payload.get("metadata") or {}),
        )
        task.metadata.setdefault("observed_failure", "AssertionError")
        task.metadata["raw_response"] = raw
        self.logger.debug_dump("red_task", task=task)
        return task, []

    def generate_task(
        self,
        session: "RoleSession",
        *,
        topic: str,
        weakness_summary: Optional[str],
    ) -> Optional[PythonTask]:
        raw = self.generate_raw_response(session, build_red_messages(topic, weakness_summary))
        task, _ = self.parse_task_response(raw, requested_topic=topic)
        return task
