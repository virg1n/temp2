from __future__ import annotations

import logging

from environment_engine import EnvironmentEngine
from logging_utils import get_logger, log_event
from prompts import build_socratic_messages, build_socratic_user_prompt
from schemas import HintCandidate, PipelineSettings, TaskCandidate
from task_execution import execute_buggy_task


LOGGER = get_logger(__name__)


class SocraticHintGenerator:
    def __init__(self, environment: EnvironmentEngine, settings: PipelineSettings) -> None:
        self.environment = environment
        self.settings = settings

    def _ensure_observed_failure(self, task: TaskCandidate) -> None:
        if task.observed_failure is not None:
            return
        observed_failure = execute_buggy_task(
            task.buggy_python,
            task.asserts,
            timeout_seconds=self.settings.runtime.task_execution_timeout_seconds,
            max_output_chars=self.settings.runtime.task_execution_max_output_chars,
            execution_mode=self.settings.runtime.task_execution_mode,
            sandbox_command=self.settings.runtime.task_execution_sandbox_command,
            allow_unsafe_host_execution=self.settings.runtime.allow_unsafe_host_execution,
        )
        task.observed_failure = observed_failure

    def generate_hints(self, tasks: list[TaskCandidate]) -> list[HintCandidate]:
        model = self.environment.load_socratic()
        num_hints = self.settings.training.socratic.num_hints_per_task
        results: list[HintCandidate] = []

        for task in tasks:
            self._ensure_observed_failure(task)
            prompt_text = build_socratic_user_prompt(task)
            messages = build_socratic_messages(task)
            outputs = model.generate(
                messages,
                generation=self.settings.models.socratic.generation,
                num_return_sequences=num_hints,
            )
            for idx, text in enumerate(outputs):
                results.append(
                    HintCandidate(
                        hint_id=f"{task.task_id}-hint-{idx}",
                        task_id=task.task_id,
                        topic=task.topic,
                        prompt_text=prompt_text,
                        text=text.strip(),
                        sample_index=idx,
                        model_name=self.environment.state.socratic.active_model_path
                        or self.settings.models.socratic.model_name_or_path,
                    )
                )
            log_event(
                LOGGER,
                logging.INFO,
                "socratic_hints_generated",
                f"Generated Socratic hints for {task.task_id}",
                task_id=task.task_id,
                topic=task.topic,
                hint_count=len(outputs),
            )
        return results
