from __future__ import annotations

import hashlib
import json
import logging

from environment_engine import EnvironmentEngine
from logging_utils import get_logger, log_event
from models_factory import extract_json
from prompts import build_task_validation_messages
from schemas import PipelineSettings, RejectedTask, TaskCandidate, ValidatedTask, ValidationResult
from storage import StorageManager


LOGGER = get_logger(__name__)


def _dedupe_key(task: TaskCandidate) -> str:
    normalized = json.dumps(
        {
            "topic": task.topic.strip().lower(),
            "task_statement": " ".join(task.task_statement.split()).lower(),
            "buggy_program": " ".join(task.buggy_program.split()).lower(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class TaskJudgeValidator:
    def __init__(self, environment: EnvironmentEngine, settings: PipelineSettings, storage: StorageManager) -> None:
        self.environment = environment
        self.settings = settings
        self.storage = storage
        self.seen_hashes = storage.load_seen_tasks()

    def _dedupe_candidates(self, candidates: list[TaskCandidate]) -> tuple[list[tuple[TaskCandidate, str]], list[RejectedTask]]:
        unique: list[tuple[TaskCandidate, str]] = []
        rejected: list[RejectedTask] = []
        batch_seen: set[str] = set()
        duplicates = 0
        for candidate in candidates:
            key = _dedupe_key(candidate)
            if key in self.seen_hashes or key in batch_seen:
                duplicates += 1
                rejected.append(
                    RejectedTask(
                        task=candidate,
                        dedupe_key=key,
                        rejection_reason="duplicate",
                        judge_score=0.0,
                        judge_feedback="Duplicate of a previously seen or same-batch task",
                    )
                )
                self.storage.append_event(
                    "task_rejected",
                    {
                        "task_id": candidate.task_id,
                        "topic": candidate.topic,
                        "reason": "duplicate",
                    },
                )
                continue
            batch_seen.add(key)
            unique.append((candidate, key))
        log_event(
            LOGGER,
            logging.INFO,
            "task_deduped",
            "Completed task deduplication",
            input_count=len(candidates),
            unique_count=len(unique),
            duplicate_count=duplicates,
        )
        return unique, rejected

    def validate_candidates(self, candidates: list[TaskCandidate]) -> ValidationResult:
        deduped, rejected_tasks = self._dedupe_candidates(candidates)
        if not deduped:
            return ValidationResult(valid_tasks=[], rejected_tasks=rejected_tasks)

        judge = self.environment.load_judge()
        valid_tasks: list[ValidatedTask] = []
        batch_size = self.settings.judge.task_validation_batch_size

        for start in range(0, len(deduped), batch_size):
            batch = deduped[start : start + batch_size]
            messages = build_task_validation_messages([item[0] for item in batch])
            raw = judge.generate(messages, generation=self.settings.models.judge.generation, num_return_sequences=1)
            payload = extract_json(raw[0] if raw else "")
            verdicts = payload if isinstance(payload, list) else []
            verdict_map = {
                str(item.get("task_id")): item
                for item in verdicts
                if isinstance(item, dict) and item.get("task_id") is not None
            }

            for candidate, key in batch:
                verdict = verdict_map.get(candidate.task_id, {})
                passed = bool(verdict.get("passed", False))
                score = float(verdict.get("score", 0.0) or 0.0)
                feedback = str(verdict.get("feedback", "Judge response missing")).strip()
                if passed:
                    validated = ValidatedTask(
                        task=candidate,
                        dedupe_key=key,
                        judge_passed=True,
                        judge_score=score,
                        judge_feedback=feedback,
                    )
                    valid_tasks.append(validated)
                    self.seen_hashes.add(key)
                else:
                    rejected_tasks.append(
                        RejectedTask(
                            task=candidate,
                            dedupe_key=key,
                            rejection_reason="semantic_validation_failed",
                            judge_score=score,
                            judge_feedback=feedback,
                        )
                    )
                    self.storage.append_event(
                        "task_rejected",
                        {
                            "task_id": candidate.task_id,
                            "topic": candidate.topic,
                            "reason": "semantic_validation_failed",
                            "score": score,
                            "feedback": feedback,
                        },
                    )

        self.storage.save_seen_tasks(self.seen_hashes)
        log_event(
            LOGGER,
            logging.INFO,
            "task_validation_finished",
            "Completed semantic task validation",
            valid_count=len(valid_tasks),
            rejected_count=len(rejected_tasks),
            requested_min=self.settings.judge.min_valid_tasks,
        )
        return ValidationResult(valid_tasks=valid_tasks, rejected_tasks=rejected_tasks)
