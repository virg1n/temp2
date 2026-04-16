from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from environment_engine import EnvironmentEngine
from logging_utils import get_logger, log_event
from prompts import build_red_generation_messages
from schemas import CurriculumTopic, PipelineSettings, TaskCandidate


LOGGER = get_logger(__name__)


def _normalize_asserts(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _stable_task_id(topic: str, statement: str, buggy_python: str, asserts: list[str]) -> str:
    payload = json.dumps(
        {
            "topic": topic,
            "task_statement": statement,
            "buggy_python": buggy_python,
            "asserts": asserts,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class RedTaskGenerator:
    def __init__(self, environment: EnvironmentEngine, settings: PipelineSettings) -> None:
        self.environment = environment
        self.settings = settings

    def _to_task_candidate(self, payload: dict[str, Any], adapter_path: str | None) -> TaskCandidate | None:
        topic = str(payload.get("topic", "")).strip()
        task_statement = str(payload.get("task_statement", "")).strip()
        buggy_python = str(payload.get("buggy_python", "")).strip()
        bug_summary = str(payload.get("bug_summary", "")).strip()
        educational_value = str(payload.get("educational_value", "")).strip()
        asserts = _normalize_asserts(payload.get("asserts"))
        if not topic or not task_statement or not buggy_python or not asserts:
            return None
        task_id = _stable_task_id(topic, task_statement, buggy_python, asserts)
        return TaskCandidate(
            task_id=task_id,
            topic=topic,
            task_statement=task_statement,
            buggy_python=buggy_python,
            asserts=asserts,
            bug_summary=bug_summary,
            educational_value=educational_value,
            source_model=self.settings.models.red.model_name_or_path,
            source_adapter_path=adapter_path,
            raw_payload=payload,
        )

    def generate_for_topics(self, topics: list[CurriculumTopic], *, use_base_model: bool = False) -> list[TaskCandidate]:
        model = self.environment.load_red(use_base_model=use_base_model)
        adapter_path = (
            None
            if use_base_model
            else self.environment.state.red.active_adapter_path or self.settings.models.red.adapter_path
        )
        candidates: list[TaskCandidate] = []

        for topic in topics:
            messages = build_red_generation_messages(topic, self.settings.curriculum.candidates_per_topic)
            payload = model.generate_json(messages, generation=self.settings.models.red.generation)
            items = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
            accepted = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                candidate = self._to_task_candidate(item, adapter_path)
                if candidate is None:
                    continue
                candidates.append(candidate)
                accepted += 1
            log_event(
                LOGGER,
                logging.INFO,
                "red_topic_generation",
                f"Generated Red candidates for {topic.name}",
                topic=topic.name,
                accepted=accepted,
                use_base_model=use_base_model,
            )
        return candidates
