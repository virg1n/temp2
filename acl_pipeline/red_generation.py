from __future__ import annotations

import json
import re
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
        solution = str(payload.get("buggy_solution") or "").strip()
        failing_asserts = payload.get("failing_asserts") or payload.get("asserts") or []
        if isinstance(failing_asserts, str):
            failing_asserts = [failing_asserts]
        failing_asserts = [str(item).strip() for item in failing_asserts if str(item).strip()]
        metadata = dict(payload.get("metadata") or {})
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
        if not solution:
            reasons.append("missing buggy_solution")
        if not metadata:
            reasons.append("missing metadata")
        else:
            failure_mode = str(metadata.get("failure_mode") or "").strip()
            difficulty = str(metadata.get("difficulty") or "").strip().lower()
            if not failure_mode:
                reasons.append("missing metadata.failure_mode")
            if difficulty not in {"medium", "hard"}:
                reasons.append("invalid metadata.difficulty")
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
            failing_asserts=failing_asserts,
            metadata=metadata,
        )
        task.metadata["red_spec"] = spec.to_dict()
        task.metadata.setdefault("failure_mode", str(metadata.get("failure_mode") or intended_bug))
        task.metadata.setdefault("observed_failure", "AssertionError")
        task.metadata["raw_response"] = raw
        task.metadata["red_format"] = "single_json_v1"
        self.logger.debug_dump("red_task", task=task)
        return task, []

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
