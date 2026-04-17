from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .logging_utils import StructuredLogger
from .modeling import RoleSession
from .prompts import build_red_messages
from .schemas import PythonTask


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


def fallback_task(topic: str, reason: str) -> PythonTask:
    buggy = (
        "def solve(nums):\n"
        "    total = 1\n"
        "    for value in nums:\n"
        "        total += value\n"
        "    return total\n"
    )
    asserts = [
        "assert solve([1, 2, 3]) == 6",
        "assert solve([]) == 0",
    ]
    return PythonTask(
        task_id=uuid4().hex[:16],
        topic=topic,
        statement=f"Debug this Python task about {topic}. The function should return the sum of the input list.",
        buggy_solution=buggy,
        failing_asserts=asserts,
        metadata={
            "fallback": True,
            "fallback_reason": reason,
            "observed_failure": "AssertionError",
        },
    )


class RedTaskGenerator:
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger

    def generate_task(
        self,
        session: RoleSession,
        *,
        topic: str,
        weakness_summary: Optional[str],
    ) -> PythonTask:
        raw = session.generate([build_red_messages(topic, weakness_summary)])[0]
        payload = _extract_json(raw)
        if not isinstance(payload, dict):
            task = fallback_task(topic, "non_json_red_output")
            task.metadata["raw_response"] = raw
            return task

        solution = str(payload.get("buggy_solution") or "").strip()
        failing_asserts = payload.get("failing_asserts") or payload.get("asserts") or []
        if isinstance(failing_asserts, str):
            failing_asserts = [failing_asserts]
        failing_asserts = [str(item).strip() for item in failing_asserts if str(item).strip()]

        if solution and not failing_asserts:
            solution, failing_asserts = _split_asserts_from_program(solution)

        if not solution or not failing_asserts:
            task = fallback_task(topic, "missing_solution_or_asserts")
            task.metadata["raw_response"] = raw
            return task

        task = PythonTask(
            task_id=uuid4().hex[:16],
            topic=str(payload.get("topic") or topic),
            statement=str(payload.get("statement") or f"Debug this Python task about {topic}."),
            buggy_solution=solution,
            failing_asserts=failing_asserts,
            metadata=dict(payload.get("metadata") or {}),
        )
        task.metadata.setdefault("observed_failure", "AssertionError")
        task.metadata["raw_response"] = raw
        self.logger.debug_dump("red_task", task=task)
        return task
