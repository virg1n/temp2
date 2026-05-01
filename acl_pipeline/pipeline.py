from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .config import GenerationSettings, PipelineConfig
from .curriculum import CurriculumManager
from .judge import JudgeService
from .logging_utils import build_logger
from .modeling import ModelPool, clear_cuda_memory, is_oom_error
from .prompts import (
    build_red_buggy_messages,
    build_red_buggy_repair_message,
    build_red_buggy_response_prefix,
    build_red_buggy_training_prompt,
    build_red_repair_message,
    build_red_spec_messages,
    build_red_spec_repair_message,
    build_red_spec_response_prefix,
    build_red_training_prompt,
)
from .red_generation import RedTaskGenerator
from .red_update import RedUpdater, serialize_task_json
from .schemas import EpisodeRecord, RedRejectedExample, RedTaskSpec, RedTrainingExample, SocraticPreferenceExample
from .socratic_dpo import SocraticDpoUpdater
from .socratic_generation import generate_socratic_hint, generate_socratic_hints
from .socratic_grpo import SocraticGrpoUpdater
from .storage import SimpleStorage
from .task_execution import audit_buggy_solution_asserts, execute_program, execute_task


class AdversarialCurriculumPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = build_logger(
            config.storage.root_dir,
            level=config.runtime.log_level,
            debug_all=config.runtime.debug_all,
        )
        self.storage = SimpleStorage(
            config.storage.root_dir,
            keep_last_n_checkpoints=config.storage.keep_last_n_checkpoints,
            hard_buffer_max_size=config.storage.hard_buffer_max_size,
        )
        self.curriculum = CurriculumManager(config.curriculum)
        saved_state = self.storage.load_curriculum_state()
        if saved_state is not None:
            self.curriculum.restore(saved_state)

        self.model_pool = ModelPool(config, self.logger)
        self.judge = JudgeService(self.model_pool, self.logger)
        self.red_generator = RedTaskGenerator(self.logger)
        self.red_updater = RedUpdater(config, self.model_pool, self.storage, self.logger)
        self.socratic_grpo_updater = SocraticGrpoUpdater(config, self.model_pool, self.judge, self.storage, self.logger)
        self.socratic_dpo_updater = SocraticDpoUpdater(config, self.model_pool, self.storage, self.logger)
        self.rng = random.Random(config.runtime.seed)
        self._red_adapter_failed_last_iteration = False
        self._red_rejections_by_iteration: Dict[int, List[RedRejectedExample]] = {}

        pointers = self.storage.load_pointers()
        self.completed_iterations = int(pointers.get("completed_iterations") or 0)
        self.current_socratic_model = str(pointers.get("socratic_model_path") or config.socratic.model_name_or_path)
        self.current_socratic_adapter = pointers.get("socratic_adapter_path") or config.socratic.base_adapter_path
        self.current_red_adapter = pointers.get("red_adapter_path") or config.red.base_adapter_path
        if self._using_uniform_curriculum(self.completed_iterations + 1):
            self.storage.save_curriculum_state(self.curriculum.uniformize_weights())

    def _iteration_size(self) -> int:
        return max(1, int(self.config.runtime.iteration_size or self.config.red.update.update_every_episodes or 1))

    def _red_base_cutoff_iteration(self) -> int:
        return max(0, int(self.config.red.force_base_generation_after_iteration))

    def _curriculum_adaptation_cutoff_iteration(self) -> int:
        return max(0, int(self.config.curriculum.adaptive_weighting_until_iteration))

    def _using_base_red_generation(self, iteration_index: int) -> bool:
        cutoff = self._red_base_cutoff_iteration()
        return cutoff > 0 and iteration_index > cutoff

    def _using_uniform_curriculum(self, iteration_index: int) -> bool:
        cutoff = self._curriculum_adaptation_cutoff_iteration()
        return cutoff > 0 and iteration_index > cutoff

    def _socratic_training_method(self) -> str:
        return str(self.config.socratic.training_method or "grpo").strip().lower()

    def _using_socratic_dpo(self) -> bool:
        return self._socratic_training_method() == "dpo"

    def _socratic_preference_text(self, hint) -> str:
        # Keep DPO aligned with Judge: the ranked score is based on raw_text,
        # so the preference pair should train on that same model output.
        return str(getattr(hint, "raw_text", "") or hint.text or "").strip()

    def _socratic_sanitized_text(self, hint) -> str:
        return str(getattr(hint, "metadata", {}).get("sanitized_text") or hint.text or "").strip()

    def _effective_red_generation_adapter(self, iteration_index: int) -> Optional[str]:
        if self._using_base_red_generation(iteration_index):
            return None
        return self.current_red_adapter

    def _prepare_iteration_modes(self, iteration_index: int) -> None:
        if self._using_uniform_curriculum(iteration_index):
            snapshot = self.curriculum.uniformize_weights()
            self.storage.save_curriculum_state(snapshot)
            self.logger.event(
                "curriculum_uniform_mode",
                iteration=iteration_index,
                curriculum_weights=snapshot.weights,
                running_topic_rewards=snapshot.running_topic_rewards,
            )

        if self._using_base_red_generation(iteration_index):
            self.logger.event(
                "red_base_generation_mode",
                iteration=iteration_index,
                adapter_path=self._effective_red_generation_adapter(iteration_index),
                stored_red_adapter=self.current_red_adapter,
            )

    def _attach_execution(self, task, execution_result) -> None:
        task.metadata["execution"] = execution_result.to_dict()
        task.metadata["execution_status"] = execution_result.status
        task.metadata["observed_failure"] = execution_result.error_message

    def _normalize_topic(self, topic: str) -> str:
        return " ".join(str(topic).lower().replace("_", " ").split())

    def _is_already_correct_red_rejection(self, reason: Any) -> bool:
        text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
        return "already_correct_code" in text or "already_correct" in text

    def _is_dual_solution_validation_rejection(self, reason: Any) -> bool:
        text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
        return "buggy_too_correct" in text

    def _is_trainable_red_dpo_rejection(self, reason: Any) -> bool:
        text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
        if self._is_already_correct_red_rejection(text):
            return True
        if self._is_dual_solution_validation_rejection(text):
            return True
        return any(
            marker in text
            for marker in (
                "non_json_response",
                "blocking_syntax_error",
                "blocking_indentation_error",
                "blocking_nameerror",
                "blocking_timeout",
                "unrelated_nameerror",
                "changed_tests",
                "changed_reference_solution",
                "changed_statement",
                "changed_target_function",
                "changed_intended_bug",
                "missing_buggy_solution",
            )
        )

    def _task_spec_from_metadata(self, task) -> Optional[RedTaskSpec]:
        payload = dict(task.metadata.get("red_spec") or {})
        if not payload:
            return None
        topic = str(payload.get("topic") or task.topic)
        target_function = str(payload.get("target_function") or "").strip()
        intended_bug = str(payload.get("intended_bug") or "").strip()
        expected_first_failure = str(payload.get("expected_first_failure") or "").strip()
        if not (target_function or intended_bug or expected_first_failure):
            return None
        return RedTaskSpec(
            topic=topic,
            target_function=target_function,
            intended_bug=intended_bug,
            expected_first_failure=expected_first_failure,
            metadata=dict(payload.get("metadata") or {}),
        )

    def _record_red_rejection(
        self,
        *,
        topic: str,
        prompt: str,
        rejected_completion: str,
        rejection_reason: str,
        spec: Optional[RedTaskSpec] = None,
        task_quality: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        iteration_index: Optional[int] = None,
    ) -> Optional[RedRejectedExample]:
        completion = str(rejected_completion or "").strip()
        if not completion:
            return None
        example_metadata = dict(metadata or {})
        if iteration_index is not None:
            example_metadata["iteration"] = iteration_index
        example = RedRejectedExample(
            example_id=uuid4().hex[:16],
            topic=topic,
            prompt=prompt,
            rejected_completion=completion,
            rejection_reason=rejection_reason,
            task_quality=task_quality,
            spec=spec.to_dict() if spec is not None else None,
            metadata=example_metadata,
        )
        self.storage.append_red_rejected_example(example)
        if iteration_index is not None:
            self._red_rejections_by_iteration.setdefault(iteration_index, []).append(example)
        self.logger.warning(
            "red_rejected_example_added",
            topic=topic,
            rejection_reason=rejection_reason,
            task_quality=task_quality,
        )
        return example

    def _remember_direct_red_rejection(self, item: Dict[str, Any], rejected: Optional[RedRejectedExample]) -> None:
        if rejected is None:
            return
        item.setdefault("red_direct_rejections", []).append(rejected.to_dict())
        task = item.get("task")
        if task is not None:
            direct = list(task.metadata.get("red_direct_rejections") or [])
            direct.append(rejected.to_dict())
            task.metadata["red_direct_rejections"] = direct

    def _primary_trainable_validation_reason(self, reasons: List[str]) -> Optional[str]:
        for reason in reasons:
            if self._is_dual_solution_validation_rejection(reason):
                return str(reason)
        for reason in reasons:
            if self._is_already_correct_red_rejection(reason):
                return "already_correct_code"
        for reason in reasons:
            if self._is_trainable_red_dpo_rejection(reason):
                return str(reason)
        return None

    def _format_assert_audit_repair_context(self, task, audit: List[Dict[str, Any]]) -> str:
        spec = self._task_spec_from_metadata(task)
        intended_bug = str(spec.intended_bug if spec is not None else task.metadata.get("failure_mode") or "").strip()
        lines = [
            "Your previous output produced a buggy_solution that incorrectly passed all tests.",
            "Here is what each assert evaluated to:",
        ]
        if not audit:
            lines.append("  Test 1: audit unavailable -> reference=None, buggy=None (unknown)")
        for offset, row in enumerate(audit, start=1):
            index = row.get("test_index")
            label = f"Test {index}" if isinstance(index, int) else f"Test {offset}"
            if row.get("kind") == "timeout":
                lines.append(f"  {label}: timeout -> reference=None, buggy=None (unknown)")
                continue
            expression = str(row.get("args_repr") or row.get("expression") or "assert").strip()
            if row.get("reference_value") is not None or row.get("buggy_value") is not None:
                equal = row.get("equal")
                relation = "equal" if equal is True else "different" if equal is False else "unknown"
                passed = "assert passed" if equal is True else "check the asserted condition"
                lines.append(
                    "  "
                    + f"{label}: {expression} -> reference={row.get('reference_value')}, "
                    + f"buggy={row.get('buggy_value')} ({relation} -> {passed})"
                )
                continue
            if row.get("left_value") is not None or row.get("right_value") is not None:
                operator = str(row.get("operator") or "?")
                lines.append(
                    "  "
                    + f"{label}: {expression} -> left={row.get('left_value')} {operator} "
                    + f"right={row.get('right_value')} (assert passed)"
                )
                continue
            lines.append(
                "  "
                + f"{label}: {expression} -> value={row.get('boolean_value', row.get('expression'))} "
                + "(assert passed)"
            )
        lines.extend(
            [
                f'The intended_bug ("{intended_bug}") is NOT present in your buggy_solution.',
                "Modify buggy_solution so its behavior diverges from reference_solution on at least one test, in the way intended_bug describes.",
                "Do NOT change the tests, reference_solution, statement, or spec.",
            ]
        )
        return "\n".join(lines)

    def _red_repair_message_for_item(self, item: Dict[str, Any], rejection_reasons: List[str]) -> Dict[str, str]:
        task = item.get("task")
        repair_context = None
        normalized = {
            str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
            for reason in rejection_reasons
        }
        if task is not None and "buggy_too_correct" in normalized:
            audit = list(item.get("assert_audit") or task.metadata.get("assert_audit") or [])
            repair_context = self._format_assert_audit_repair_context(task, audit)
        spec_payload = item.get("spec_payload")
        if isinstance(spec_payload, dict):
            return build_red_buggy_repair_message(
                str(item["topic"]),
                rejection_reasons,
                spec_payload,
                repair_context=repair_context,
            )
        return build_red_repair_message(str(item["topic"]), rejection_reasons, repair_context=repair_context)

    def _candidate_rejection_reasons(
        self,
        *,
        requested_topic: str,
        task,
        execution_result,
    ) -> List[str]:
        reasons: List[str] = []
        if self._normalize_topic(task.topic) != self._normalize_topic(requested_topic):
            reasons.append("wrong topic")

        program = task.combined_program()
        if any(char in program for char in ("\u2028", "\u2029")):
            reasons.append("invalid unicode line separator in code")

        repair_probability = float(self.config.task_execution.probabilistic_repair_probability)
        if task.non_empty_line_count() < int(self.config.task_execution.min_code_lines_for_repair):
            if self.rng.random() < repair_probability:
                reasons.append("too short")

        if execution_result is not None:
            if execution_result.status in {"syntax_error", "indentation_error"}:
                reasons.append(f"blocking {execution_result.status}")
            elif execution_result.status == "nameerror":
                spec = self._task_spec_from_metadata(task)
                signal = " ".join(
                    str(part or "").lower()
                    for part in (
                        getattr(spec, "intended_bug", "") if spec is not None else "",
                        getattr(spec, "expected_first_failure", "") if spec is not None else "",
                        task.metadata.get("failure_mode"),
                    )
                )
                if not any(marker in signal for marker in ("nameerror", "name error", "undefined", "not defined", "missing name")):
                    reasons.append("blocking nameerror")
            elif execution_result.status == "timeout":
                reasons.append("blocking timeout")
            elif execution_result.status == "passed":
                keep_probability = max(0.0, min(1.0, float(self.config.task_execution.passed_task_keep_probability)))
                if self.rng.random() >= keep_probability:
                    reasons.append("already correct code, there are no errors in asserts")
        return reasons

    def _validate_request_item(self, item: Dict[str, Any]) -> Tuple[Optional[Any], List[str]]:
        task = item.get("task")
        execution = None
        rejection_reasons: List[str] = []
        if task is None:
            return execution, rejection_reasons
        item.pop("assert_audit", None)
        if self.config.task_execution.enabled:
            reference_solution = str(task.reference_solution or task.metadata.get("reference_solution") or "").strip()
            if reference_solution:
                reference_execution = execute_program(reference_solution, self.config.task_execution)
                task.metadata["reference_execution"] = reference_execution.to_dict()
                item["validation_reference_execution"] = reference_execution
                if reference_execution.status != "passed":
                    task.metadata["reference_invalid_ignored"] = True
                    task.metadata["observed_failure"] = reference_execution.error_message
                    rejection_reasons.append("reference_solution_failed")

                execution = execute_task(task, self.config.task_execution)
                self._attach_execution(task, execution)
                if execution.status == "passed":
                    rejection_reasons.append("buggy_too_correct")
                    audit = audit_buggy_solution_asserts(task, self.config.task_execution)
                    item["assert_audit"] = audit
                    task.metadata["assert_audit"] = audit
                    task.metadata["validation_rejection_reason"] = "buggy_too_correct"
            else:
                execution = execute_task(task, self.config.task_execution)
                self._attach_execution(task, execution)
        rejection_reasons.extend(
            self._candidate_rejection_reasons(
                requested_topic=str(item["topic"]),
                task=task,
                execution_result=execution,
            )
        )
        rejection_reasons = list(dict.fromkeys(reason for reason in rejection_reasons if reason))
        item["validation_execution"] = execution
        item["validation_reasons"] = rejection_reasons
        return execution, rejection_reasons

    def _red_effective_batch_size(self, item_count: int) -> int:
        configured = max(1, int(self.config.red.generation.batch_size))
        default_target = max(1, min(4, self._iteration_size()))
        return max(1, min(item_count, max(configured, default_target)))

    def _batched_red_generate(
        self,
        red_session,
        messages_batch: List[List[Dict[str, str]]],
        *,
        stage: str,
        response_prefixes: Optional[List[str]] = None,
    ) -> List[str]:
        if not messages_batch:
            return []
        generation = GenerationSettings(
            batch_size=self._red_effective_batch_size(len(messages_batch)),
            max_new_tokens=int(self.config.red.generation.max_new_tokens),
            max_context_tokens=int(self.config.red.generation.max_context_tokens),
            temperature=float(self.config.red.generation.temperature),
            top_p=float(self.config.red.generation.top_p),
            do_sample=bool(self.config.red.generation.do_sample),
            repetition_penalty=float(self.config.red.generation.repetition_penalty),
        )
        self.logger.debug_dump(
            "red_batch_generate",
            stage=stage,
            prompt_count=len(messages_batch),
            effective_batch_size=generation.batch_size,
            max_new_tokens=generation.max_new_tokens,
        )
        return red_session.generate(
            messages_batch,
            generation=generation,
            response_prefixes=response_prefixes,
        )

    def _new_red_request(self, topic: str, weakness_summary: str) -> Dict[str, Any]:
        return {
            "topic": topic,
            "weakness_summary": weakness_summary,
            "spec_messages": build_red_spec_messages(topic, weakness_summary),
            "spec_prompt": build_red_training_prompt(topic, weakness_summary),
            "messages": [],
            "task_prompt": "",
            "spec_payload": None,
            "spec_raw_response": "",
            "task": None,
            "red_direct_rejections": [],
            "last_rejection_reasons": [],
            "validation_reasons": [],
            "validation_execution": None,
        }

    def _generate_red_tasks_batch(
        self,
        red_session,
        requests: List[Dict[str, Any]],
        iteration_index: int,
    ) -> List[Dict[str, Any]]:
        generated: List[Dict[str, Any]] = []
        max_attempts = int(self.config.task_execution.max_red_generation_attempts)

        for attempt in range(1, max_attempts + 1):
            pending_specs = [item for item in requests if item.get("spec_payload") is None]
            if not pending_specs:
                break
            raw_batch = self._batched_red_generate(
                red_session,
                [item["spec_messages"] for item in pending_specs],
                stage="task_spec",
                response_prefixes=[build_red_spec_response_prefix(str(item["topic"])) for item in pending_specs],
            )
            for item, raw in zip(pending_specs, raw_batch):
                topic = str(item["topic"])
                weakness_summary = str(item["weakness_summary"])
                spec_payload, parse_reasons = self.red_generator.parse_spec_response(raw, requested_topic=topic)
                rejection_reasons = list(dict.fromkeys(reason for reason in parse_reasons if reason))
                if spec_payload is not None and not rejection_reasons:
                    if self.config.task_execution.enabled:
                        reference_execution = execute_program(
                            str(spec_payload.get("reference_solution") or ""),
                            self.config.task_execution,
                        )
                        spec_payload["reference_execution"] = reference_execution.to_dict()
                        if reference_execution.status != "passed":
                            rejection_reasons = ["reference_solution_failed"]
                            self._record_red_rejection(
                                topic=topic,
                                prompt=str(item["spec_prompt"]),
                                rejected_completion=raw,
                                rejection_reason=", ".join(rejection_reasons),
                                metadata={
                                    "stage": "task_spec_reference_validation",
                                    "attempt": attempt,
                                    "weakness_summary": weakness_summary,
                                    "reference_execution": reference_execution.to_dict(),
                                },
                                iteration_index=iteration_index,
                            )
                            self.logger.warning(
                                "red_task_spec_reference_repair_requested",
                                topic=topic,
                                attempt=attempt,
                                rejection_reasons=rejection_reasons,
                                reference_status=reference_execution.status,
                            )
                            item["spec_messages"].append({"role": "assistant", "content": raw})
                            item["spec_messages"].append(build_red_spec_repair_message(topic, rejection_reasons))
                            continue
                    item["spec_payload"] = spec_payload
                    item["spec_raw_response"] = raw
                    item["messages"] = build_red_buggy_messages(topic, spec_payload)
                    item["task_prompt"] = build_red_buggy_training_prompt(topic, spec_payload)
                    self.logger.event(
                        "red_task_spec_generated",
                        iteration=iteration_index,
                        topic=topic,
                        attempt=attempt,
                        weakness_summary=weakness_summary,
                    )
                    continue

                item["last_rejection_reasons"] = rejection_reasons or ["non-json response"]
                self._record_red_rejection(
                    topic=topic,
                    prompt=str(item["spec_prompt"]),
                    rejected_completion=raw,
                    rejection_reason=", ".join(item["last_rejection_reasons"]),
                    metadata={
                        "stage": "task_spec",
                        "attempt": attempt,
                        "weakness_summary": weakness_summary,
                    },
                    iteration_index=iteration_index,
                )
                self.logger.warning(
                    "red_task_spec_repair_requested",
                    topic=topic,
                    attempt=attempt,
                    rejection_reasons=item["last_rejection_reasons"],
                )
                item["spec_messages"].append({"role": "assistant", "content": raw})
                item["spec_messages"].append(build_red_spec_repair_message(topic, item["last_rejection_reasons"]))

        for attempt in range(1, max_attempts + 1):
            pending = [item for item in requests if item.get("spec_payload") is not None and item.get("task") is None]
            if not pending:
                break
            raw_batch = self._batched_red_generate(
                red_session,
                [item["messages"] for item in pending],
                stage="task_buggy",
                response_prefixes=[build_red_buggy_response_prefix(str(item["topic"])) for item in pending],
            )
            for item, raw in zip(pending, raw_batch):
                topic = str(item["topic"])
                weakness_summary = str(item["weakness_summary"])
                task_prompt = str(item["task_prompt"])

                task, parse_reasons = self.red_generator.parse_task_response(
                    raw,
                    requested_topic=topic,
                    locked_spec=item.get("spec_payload"),
                )
                rejection_reasons = list(parse_reasons)

                if task is not None:
                    item["messages"].append({"role": "assistant", "content": raw})
                    task.metadata["red_prompt"] = task_prompt
                    task.metadata["weakness_summary"] = weakness_summary
                    task.metadata["red_spec_prompt"] = str(item.get("spec_prompt") or "")
                    task.metadata["red_spec_raw_response"] = str(item.get("spec_raw_response") or "")
                    task.metadata["red_direct_rejections"] = list(item.get("red_direct_rejections") or [])

                rejection_reasons = list(dict.fromkeys(reason for reason in rejection_reasons if reason))
                if task is not None and not rejection_reasons:
                    item["task"] = task
                    generated.append(item)
                    continue

                item["last_rejection_reasons"] = rejection_reasons or ["unspecified issue"]
                rejected = self._record_red_rejection(
                    topic=topic,
                    prompt=task_prompt,
                    rejected_completion=raw,
                    rejection_reason=", ".join(item["last_rejection_reasons"]),
                    metadata={
                        "stage": "task_buggy",
                        "attempt": attempt,
                        "weakness_summary": weakness_summary,
                        "execution_status": None,
                    },
                    iteration_index=iteration_index,
                )
                self._remember_direct_red_rejection(item, rejected)
                self.logger.warning(
                    "red_task_buggy_repair_requested",
                    topic=topic,
                    attempt=attempt,
                    rejection_reasons=item["last_rejection_reasons"],
                    execution_status=None,
                )
                item["messages"].append(self._red_repair_message_for_item(item, item["last_rejection_reasons"]))

        for item in requests:
            if item.get("task") is None:
                self.logger.warning(
                    "red_task_generation_failed",
                    topic=item["topic"],
                    weakness_summary=item["weakness_summary"],
                    rejection_reasons=item.get("last_rejection_reasons") or ["unspecified issue"],
                )
        return generated

    def _validate_generated_requests(self, requests: List[Dict[str, Any]], iteration_index: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        valid: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        for item in requests:
            task = item.get("task")
            if task is None:
                continue
            _, rejection_reasons = self._validate_request_item(item)
            if rejection_reasons:
                primary_rejection = self._primary_trainable_validation_reason(rejection_reasons)
                if primary_rejection:
                    spec = self._task_spec_from_metadata(task)
                    rejected = self._record_red_rejection(
                        topic=str(item["topic"]),
                        prompt=str(item.get("task_prompt") or task.metadata.get("red_prompt") or ""),
                        rejected_completion=serialize_task_json(task),
                        rejection_reason=primary_rejection,
                        spec=spec,
                        metadata={
                            "stage": "validation",
                            "weakness_summary": item.get("weakness_summary"),
                            "validation_reasons": list(rejection_reasons),
                            "assert_audit": list(item.get("assert_audit") or task.metadata.get("assert_audit") or []),
                            "reference_execution": (
                                item["validation_reference_execution"].to_dict()
                                if item.get("validation_reference_execution") is not None
                                else task.metadata.get("reference_execution")
                            ),
                            "execution_status": (
                                item["validation_execution"].status
                                if item.get("validation_execution") is not None
                                else None
                            ),
                            "observed_failure": task.observed_failure(),
                        },
                        iteration_index=iteration_index,
                    )
                    self._remember_direct_red_rejection(item, rejected)
                invalid.append(item)
            else:
                valid.append(item)

        self.logger.event(
            "red_validation_complete",
            iteration=iteration_index,
            candidate_count=len(requests),
            valid_count=len(valid),
            invalid_count=len(invalid),
            invalid_topics=[item["topic"] for item in invalid],
            invalid_reasons=[item.get("validation_reasons") for item in invalid],
        )
        return valid, invalid

    def _repair_generated_requests_with_revalidation(
        self,
        red_session,
        requests: List[Dict[str, Any]],
        iteration_index: int,
    ) -> List[Dict[str, Any]]:
        if not requests:
            return []

        accepted: List[Dict[str, Any]] = []
        max_attempts = int(self.config.task_execution.max_red_generation_attempts)
        pending = list(requests)
        chunk_size = max(1, min(4, self._red_effective_batch_size(len(pending))))

        for item in pending:
            repair_reasons = item.get("validation_reasons") or ["validation requested regeneration"]
            item["messages"].append(self._red_repair_message_for_item(item, repair_reasons))

        for attempt in range(1, max_attempts + 1):
            if not pending:
                break
            next_pending: List[Dict[str, Any]] = []
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                raw_batch = self._batched_red_generate(
                    red_session,
                    [item["messages"] for item in chunk],
                    stage="task_repair_final",
                    response_prefixes=[build_red_buggy_response_prefix(str(item["topic"])) for item in chunk],
                )
                for item, raw in zip(chunk, raw_batch):
                    topic = str(item["topic"])
                    repaired_task, parse_reasons = self.red_generator.parse_task_response(
                        raw,
                        requested_topic=topic,
                        locked_spec=item.get("spec_payload"),
                    )
                    if repaired_task is not None and not parse_reasons:
                        repaired_task.metadata["red_prompt"] = str(item["task_prompt"])
                        repaired_task.metadata["weakness_summary"] = str(item["weakness_summary"])
                        repaired_task.metadata["red_spec_prompt"] = str(item.get("spec_prompt") or "")
                        repaired_task.metadata["red_spec_raw_response"] = str(item.get("spec_raw_response") or "")
                        repaired_task.metadata["red_direct_rejections"] = list(item.get("red_direct_rejections") or [])
                        repaired_task.metadata["pre_repair_validation_reasons"] = list(item.get("validation_reasons") or [])
                        repaired_task.metadata["pre_repair_execution"] = (
                            item["validation_execution"].to_dict()
                            if item.get("validation_execution") is not None
                            else None
                        )
                        item["task"] = repaired_task
                        item["messages"].append({"role": "assistant", "content": raw})
                        _, validation_reasons = self._validate_request_item(item)
                        if not validation_reasons:
                            repaired_task.metadata["accepted_after_revalidation"] = True
                            accepted.append(item)
                            self.logger.event(
                                "red_task_repaired_after_revalidation",
                                iteration=iteration_index,
                                topic=topic,
                                prior_rejection_reasons=repaired_task.metadata.get("pre_repair_validation_reasons"),
                                attempt=attempt,
                                execution_status=repaired_task.metadata.get("execution_status"),
                            )
                            continue

                        spec = self._task_spec_from_metadata(repaired_task)
                        rejected = self._record_red_rejection(
                            topic=topic,
                            prompt=str(item["task_prompt"]),
                            rejected_completion=raw,
                            rejection_reason=", ".join(validation_reasons),
                            spec=spec,
                            metadata={
                                "stage": "task_repair_final_validation",
                                "attempt": attempt,
                                "weakness_summary": item["weakness_summary"],
                                "validation_reasons": list(validation_reasons),
                                "assert_audit": list(item.get("assert_audit") or repaired_task.metadata.get("assert_audit") or []),
                                "reference_execution": (
                                    item["validation_reference_execution"].to_dict()
                                    if item.get("validation_reference_execution") is not None
                                    else repaired_task.metadata.get("reference_execution")
                                ),
                                "execution_status": (
                                    item["validation_execution"].status
                                    if item.get("validation_execution") is not None
                                    else None
                                ),
                            },
                            iteration_index=iteration_index,
                        )
                        self._remember_direct_red_rejection(item, rejected)
                        self.logger.warning(
                            "red_task_repair_retry_requested",
                            iteration=iteration_index,
                            topic=topic,
                            attempt=attempt,
                            rejection_reasons=validation_reasons,
                        )
                        item["messages"].append(self._red_repair_message_for_item(item, validation_reasons))
                        next_pending.append(item)
                        continue

                    parse_reasons = list(dict.fromkeys(reason for reason in parse_reasons if reason))
                    item["last_rejection_reasons"] = parse_reasons or ["non-json response"]
                    rejected = self._record_red_rejection(
                        topic=topic,
                        prompt=str(item["task_prompt"]),
                        rejected_completion=raw,
                        rejection_reason=", ".join(item["last_rejection_reasons"]),
                        metadata={
                            "stage": "task_repair_final",
                            "attempt": attempt,
                            "weakness_summary": item["weakness_summary"],
                        },
                        iteration_index=iteration_index,
                    )
                    self._remember_direct_red_rejection(item, rejected)
                    self.logger.warning(
                        "red_task_repair_retry_requested",
                        iteration=iteration_index,
                        topic=topic,
                        attempt=attempt,
                        rejection_reasons=item["last_rejection_reasons"],
                    )
                    item["messages"].append(self._red_repair_message_for_item(item, item["last_rejection_reasons"]))
                    next_pending.append(item)
            pending = next_pending

        for item in pending:
            self.logger.warning(
                "red_task_repair_failed_drop_original",
                iteration=iteration_index,
                topic=item["topic"],
                validation_reasons=item.get("validation_reasons"),
            )

        return accepted

    def _score_and_filter_red_tasks(self, requests: List[Dict[str, Any]], iteration_index: int) -> List[Dict[str, Any]]:
        if not requests:
            return []
        tasks = [item["task"] for item in requests if item.get("task") is not None]
        assessments = self.judge.evaluate_red_tasks(tasks)
        by_task_id = {
            task.task_id: assessment
            for task, assessment in zip(tasks, assessments)
        }
        accepted: List[Dict[str, Any]] = []
        rejected_count = 0
        for item in requests:
            task = item.get("task")
            if task is None:
                continue
            assessment = dict(by_task_id.get(task.task_id) or task.metadata.get("red_judge") or {})
            item["red_judge"] = assessment
            task.metadata["red_judge"] = assessment
            task.metadata["red_task_quality"] = float(assessment.get("task_quality", 5.0))
            task.metadata["red_task_hardness"] = float(assessment.get("task_hardness", 5.0))
            task.metadata["red_reward"] = float(assessment.get("red_reward", 0.0))
            if bool(assessment.get("task_is_valid_for_socratic", True)):
                accepted.append(item)
                continue
            rejected_count += 1
            spec = self._task_spec_from_metadata(task)
            rejection_reason = str(assessment.get("red_rejection_reason") or "judge_bad_task")
            rejected = self._record_red_rejection(
                topic=task.topic,
                prompt=str(task.metadata.get("red_prompt") or item.get("task_prompt") or ""),
                rejected_completion=serialize_task_json(task),
                rejection_reason=rejection_reason,
                spec=spec,
                task_quality=float(assessment.get("task_quality") or 0.0),
                metadata={
                    "stage": "red_task_judge",
                    "iteration": iteration_index,
                    "weakness_summary": item.get("weakness_summary"),
                    "red_judge": assessment,
                    "observed_failure": task.observed_failure(),
                },
                iteration_index=iteration_index,
            )
            self._remember_direct_red_rejection(item, rejected)
            self.logger.warning(
                "red_task_rejected_by_task_judge",
                iteration=iteration_index,
                topic=task.topic,
                task_quality=assessment.get("task_quality"),
                task_hardness=assessment.get("task_hardness"),
                rejection_reason=rejection_reason,
            )
        self.logger.event(
            "red_task_judge_complete",
            iteration=iteration_index,
            candidate_count=len(requests),
            accepted_count=len(accepted),
            rejected_count=rejected_count,
            task_quality=[item.get("red_judge", {}).get("task_quality") for item in requests],
            task_hardness=[item.get("red_judge", {}).get("task_hardness") for item in requests],
            red_reward=[item.get("red_judge", {}).get("red_reward") for item in requests],
        )
        return accepted

    def _build_hard_example(self, episode: EpisodeRecord, weakness_summary: str) -> RedTrainingExample:
        prompt = str(episode.task.metadata.get("red_prompt") or build_red_training_prompt(episode.topic, weakness_summary))
        return RedTrainingExample(
            example_id=uuid4().hex[:16],
            topic=episode.topic,
            prompt=prompt,
            chosen_completion=serialize_task_json(episode.task),
            rejected_completion=None,
            reward=episode.judge.normalized_reward,
            task=episode.task,
            metadata={
                "episode_id": episode.episode_id,
                "iteration": episode.metadata.get("iteration"),
                "socratic_score": episode.judge.score,
                "weakness_summary": weakness_summary,
                "observed_failure": episode.task.observed_failure(),
                "red_task_quality": episode.task.metadata.get("red_task_quality"),
                "red_task_hardness": episode.task.metadata.get("red_task_hardness"),
                "red_reward": episode.task.metadata.get("red_reward"),
            },
        )

    def _matching_trainable_red_rejection(
        self,
        *,
        episode: EpisodeRecord,
        iteration_index: int,
    ) -> Optional[RedRejectedExample]:
        for payload in episode.task.metadata.get("red_direct_rejections") or []:
            try:
                rejected = RedRejectedExample(**payload)
            except Exception:
                continue
            if self._is_trainable_red_dpo_rejection(rejected.rejection_reason):
                return rejected
            metadata = dict(rejected.metadata or {})
            if self._is_trainable_red_dpo_rejection(metadata.get("rejection_reason")):
                return rejected
            if self._is_trainable_red_dpo_rejection(metadata.get("red_rejection_reason")):
                return rejected
            for reason in metadata.get("validation_reasons") or []:
                if self._is_trainable_red_dpo_rejection(reason):
                    return rejected
            if str(metadata.get("execution_status") or "").strip().lower() == "passed":
                return rejected
        return None

    def _attach_red_dpo_rejection(
        self,
        example: RedTrainingExample,
        rejected: Optional[RedRejectedExample],
    ) -> None:
        if rejected is None:
            return
        rejected_completion = str(rejected.rejected_completion or "").strip()
        if not rejected_completion or rejected_completion == str(example.chosen_completion or "").strip():
            return
        example.rejected_completion = rejected_completion
        example.metadata.update(
            {
                "red_dpo_rejection_id": rejected.example_id,
                "red_dpo_rejection_reason": rejected.rejection_reason,
                "red_dpo_rejection_stage": dict(rejected.metadata or {}).get("stage"),
                "red_dpo_pairing": "same_task_spec_iterative_rejection",
            }
        )

    def _store_hard_examples_for_iteration(self, iteration_records: List[EpisodeRecord], iteration_index: int) -> None:
        valid_records = [
            episode
            for episode in iteration_records
            if bool(episode.metadata.get("task_is_valid_for_socratic", True))
        ]
        if not valid_records:
            return
        bottom_fraction = float(self.config.red.update.mining_bottom_fraction)
        hard_reward_max = float(self.config.red.update.hard_reward_max)
        min_red_reward = float(self.config.red.update.min_red_reward)
        eligible_records = [
            episode
            for episode in valid_records
            if float(episode.judge.normalized_reward) <= hard_reward_max
            and self._episode_red_reward(episode) >= min_red_reward
        ]
        keep_count = max(1, math.ceil(len(valid_records) * bottom_fraction))
        selected = sorted(eligible_records, key=lambda episode: episode.judge.normalized_reward)[:keep_count]
        self.logger.event(
            "hard_example_iteration_selection",
            iteration=iteration_index,
            selected_episode_ids=[episode.episode_id for episode in selected],
            selected_rewards=[episode.judge.normalized_reward for episode in selected],
            iteration_episode_ids=[episode.episode_id for episode in iteration_records],
            valid_episode_ids=[episode.episode_id for episode in valid_records],
            eligible_episode_ids=[episode.episode_id for episode in eligible_records],
            eligible_red_rewards=[self._episode_red_reward(episode) for episode in eligible_records],
            mining_bottom_fraction=bottom_fraction,
            hard_reward_max=hard_reward_max,
            min_red_reward=min_red_reward,
        )
        for episode in selected:
            weakness_summary = str(episode.metadata.get("weakness_summary") or "")
            example = self._build_hard_example(episode, weakness_summary)
            matched_rejection = self._matching_trainable_red_rejection(
                episode=episode,
                iteration_index=iteration_index,
            )
            self._attach_red_dpo_rejection(example, matched_rejection)
            self.storage.append_hard_example(example)
            self.logger.event(
                "hard_example_added",
                episode_id=episode.episode_id,
                topic=episode.topic,
                reward=episode.judge.normalized_reward,
                mining_bottom_fraction=bottom_fraction,
                hard_reward_max=hard_reward_max,
                min_red_reward=min_red_reward,
                red_reward=self._episode_red_reward(episode),
                red_dpo_rejection_id=example.metadata.get("red_dpo_rejection_id"),
                red_dpo_pairing=example.metadata.get("red_dpo_pairing"),
            )

    def _store_socratic_preferences_for_ranked_candidates(
        self,
        *,
        item: Dict[str, Any],
        episode_id: int,
        iteration_index: int,
    ) -> int:
        if not self._using_socratic_dpo():
            return 0
        ranked = list(item.get("hint_candidate_rankings") or [])
        if len(ranked) < 2:
            return 0

        task = item["task"]
        def is_socratic_trainable_candidate(candidate: Dict[str, Any]) -> bool:
            metadata = dict(candidate["judge"].metadata or {})
            if bool(metadata.get("task_is_valid_for_socratic", True)):
                return True
            local_tiebreak = dict(metadata.get("local_tiebreak") or {})
            return (
                self._is_already_correct_red_rejection(metadata.get("red_rejection_reason"))
                or str(local_tiebreak.get("execution_status") or "").strip().lower() == "passed"
            )

        socratic_trainable_ranked = [
            candidate
            for candidate in ranked
            if is_socratic_trainable_candidate(candidate)
        ]
        if len(socratic_trainable_ranked) < 2:
            return 0

        settings = self.config.socratic.dpo
        chosen = socratic_trainable_ranked[0]
        chosen_score = float(chosen["judge"].metadata.get("post_normalize") or chosen["judge"].metadata.get("adjusted_score") or chosen["judge"].score)
        added = 0
        for rejected in socratic_trainable_ranked:
            if rejected is chosen:
                continue
            rejected_score = float(rejected["judge"].metadata.get("post_normalize") or rejected["judge"].metadata.get("adjusted_score") or rejected["judge"].score)
            if chosen_score - rejected_score < float(settings.min_score_gap):
                continue

            chosen_hint = chosen["hint"]
            rejected_hint = rejected["hint"]
            chosen_text = self._socratic_preference_text(chosen_hint)
            rejected_text = self._socratic_preference_text(rejected_hint)
            if not chosen_text or not rejected_text or chosen_text == rejected_text:
                continue

            example = SocraticPreferenceExample(
                example_id=uuid4().hex[:16],
                topic=task.topic,
                task=task,
                chosen_hint=chosen_text,
                rejected_hint=rejected_text,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                chosen_judge=chosen["judge"].to_dict(),
                rejected_judge=rejected["judge"].to_dict(),
                metadata={
                    "episode_id": episode_id,
                    "iteration": iteration_index,
                    "task_id": task.task_id,
                    "weakness_summary": item.get("weakness_summary"),
                    "chosen_candidate_index": chosen.get("candidate_index"),
                    "rejected_candidate_index": rejected.get("candidate_index"),
                    "chosen_rank": chosen.get("rank"),
                    "rejected_rank": rejected.get("rank"),
                    "score_gap": chosen_score - rejected_score,
                    "preference_text_source": "raw_text",
                    "chosen_clean_hint": self._socratic_sanitized_text(chosen_hint),
                    "rejected_clean_hint": self._socratic_sanitized_text(rejected_hint),
                },
            )
            self.storage.append_socratic_preference(example)
            added += 1
            if added >= int(settings.max_pairs_per_task):
                break

        if added:
            self.logger.event(
                "socratic_dpo_preferences_added",
                episode_id=episode_id,
                task_id=task.task_id,
                topic=task.topic,
                pairs_added=added,
                candidate_count=len(ranked),
                chosen_score=chosen_score,
            )
        return added

    def _episode_red_reward(self, episode: EpisodeRecord) -> float:
        for source in (
            episode.metadata,
            episode.judge.metadata,
            episode.task.metadata,
        ):
            try:
                return float(dict(source or {}).get("red_reward"))
            except Exception:
                continue
        return 0.0

    def _log_episode_debug(self, episode: EpisodeRecord, weakness_summary: str) -> None:
        self.logger.debug_dump(
            "episode_debug",
            episode_id=episode.episode_id,
            topic=episode.topic,
            weakness_summary=weakness_summary,
            broken_code=episode.task.combined_program(),
            execution=episode.task.metadata.get("execution"),
            socratic_hint=episode.hint.text,
            socratic_hint_raw=episode.hint.raw_text,
            judge_grade=episode.judge.score,
            judge_post_normalize=episode.judge.metadata.get("post_normalize"),
            judge_adjusted_score=episode.judge.metadata.get("adjusted_score"),
            judge_criteria=episode.judge.criteria_scores,
            judge_task_quality=episode.judge.metadata.get("task_quality"),
            judge_task_hardness=episode.judge.metadata.get("task_hardness"),
            red_reward=episode.judge.metadata.get("red_reward"),
            judge_task_is_valid=episode.judge.metadata.get("task_is_valid_for_socratic"),
            judge_task_rejection_reason=episode.judge.metadata.get("red_rejection_reason"),
            judge_hint_is_valid=episode.judge.metadata.get("hint_is_valid_for_socratic"),
            judge_hint_rejection_reason=episode.judge.metadata.get("hint_rejection_reason"),
            hint_corruption=episode.judge.metadata.get("hint_corruption"),
            hint_quality=episode.judge.metadata.get("local_tiebreak"),
            task_metadata=episode.task.metadata,
        )

    def _using_non_base_red_adapter(self) -> bool:
        base = self.config.red.base_adapter_path
        current = self.current_red_adapter
        if current is None:
            return False
        return str(current) != str(base)

    def _handle_red_adapter_failure(self, iteration_index: int, generated_tasks: int, *, reason: str = "red_adapter_failure") -> None:
        previous_adapter = self.current_red_adapter
        self.current_red_adapter = None
        self.storage.save_pointer("red_adapter_path", self.current_red_adapter)
        self._red_adapter_failed_last_iteration = True
        self.logger.warning(
            "red_adapter_failure_reset",
            iteration=iteration_index,
            generated_tasks=generated_tasks,
            previous_adapter=previous_adapter,
            fallback_adapter=self.config.red.base_adapter_path,
            reason=reason,
        )

    def _reset_red_and_curriculum(self, episode_id: int) -> None:
        self.current_red_adapter = self.config.red.base_adapter_path
        self.storage.save_pointer("red_adapter_path", self.current_red_adapter)
        snapshot = self.curriculum.snapshot()
        self.storage.save_curriculum_state(snapshot)
        self.logger.warning(
            "curriculum_reset",
            episode_id=episode_id,
            reason="same_topic_repeat_threshold",
            red_adapter_path=self.current_red_adapter,
            curriculum_weights=snapshot.weights,
        )

    def _flush_pending_batch(
        self,
        pending_batch: List[Dict[str, Any]],
        next_episode_id: int,
        iteration_index: int,
    ) -> Tuple[List[EpisodeRecord], int]:
        tasks = [item["task"] for item in pending_batch]
        if self._using_socratic_dpo():
            hint_groups = [
                list(item.get("hint_candidates") or [item["hint"]])
                for item in pending_batch
            ]
            ranked_groups = self.judge.rank_hint_candidates(
                tasks,
                hint_groups,
                apply_group_spread=True,
            )
            judge_outputs = []
            for item, ranked in zip(pending_batch, ranked_groups):
                item["hint_candidate_rankings"] = ranked
                if ranked:
                    item["hint"] = ranked[0]["hint"]
                    judge_outputs.append(ranked[0]["judge"])
                else:
                    judge_outputs.extend(
                        self.judge.evaluate_batch(
                            [item["task"]],
                            [item["hint"]],
                            apply_batch_spread=False,
                        )
                    )
            self.logger.event(
                "judge_hint_ranking_complete",
                candidate_groups=len(pending_batch),
                candidates_per_task=[len(group) for group in hint_groups],
                topics=[item["task"].topic for item in pending_batch],
                ranked_scores=[
                    [candidate["judge"].metadata.get("post_normalize") for candidate in item.get("hint_candidate_rankings", [])]
                    for item in pending_batch
                ],
                ranked_valid=[
                    [candidate["judge"].metadata.get("hint_is_valid_for_socratic") for candidate in item.get("hint_candidate_rankings", [])]
                    for item in pending_batch
                ],
            )
        else:
            hints = [item["hint"] for item in pending_batch]
            judge_outputs = self.judge.evaluate_batch(
                tasks,
                hints,
                apply_batch_spread=True,
            )
        for item, judge_output in zip(pending_batch, judge_outputs):
            red_judge = dict(item.get("red_judge") or item["task"].metadata.get("red_judge") or {})
            if not red_judge:
                continue
            judge_output.metadata["task_quality"] = float(red_judge.get("task_quality", judge_output.metadata.get("task_quality", 5.0)))
            judge_output.metadata["task_hardness"] = float(red_judge.get("task_hardness", 5.0))
            judge_output.metadata["red_reward"] = float(red_judge.get("red_reward", 0.0))
            judge_output.metadata["task_is_valid_for_socratic"] = bool(red_judge.get("task_is_valid_for_socratic", True))
            if red_judge.get("red_rejection_reason"):
                judge_output.metadata["red_rejection_reason"] = str(red_judge.get("red_rejection_reason"))
            elif bool(red_judge.get("task_is_valid_for_socratic", True)):
                judge_output.metadata["red_rejection_reason"] = ""
        self.logger.event(
            "judge_batch_complete",
            candidate_count=len(pending_batch),
            socratic_training_method=self._socratic_training_method(),
            topics=[item["task"].topic for item in pending_batch],
            raw_scores=[output.score for output in judge_outputs],
            post_normalize_scores=[output.metadata.get("post_normalize") for output in judge_outputs],
            adjusted_scores=[output.metadata.get("adjusted_score") for output in judge_outputs],
            adjusted_rewards=[output.normalized_reward for output in judge_outputs],
            task_quality=[output.metadata.get("task_quality") for output in judge_outputs],
            task_hardness=[output.metadata.get("task_hardness") for output in judge_outputs],
            red_reward=[output.metadata.get("red_reward") for output in judge_outputs],
            task_is_valid_for_socratic=[output.metadata.get("task_is_valid_for_socratic") for output in judge_outputs],
            hint_is_valid_for_socratic=[output.metadata.get("hint_is_valid_for_socratic") for output in judge_outputs],
            local_tiebreak=[output.metadata.get("local_tiebreak") for output in judge_outputs],
        )

        records: List[EpisodeRecord] = []
        for item, judge_output in zip(pending_batch, judge_outputs):
            task = item["task"]
            weakness_summary = item["weakness_summary"]
            task_is_valid = bool(judge_output.metadata.get("task_is_valid_for_socratic", True))
            hint_is_valid = bool(judge_output.metadata.get("hint_is_valid_for_socratic", True))
            if not task_is_valid:
                spec = self._task_spec_from_metadata(task)
                rejection_reason = str(judge_output.metadata.get("red_rejection_reason") or "judge_bad_task")
                self._record_red_rejection(
                    topic=task.topic,
                    prompt=str(task.metadata.get("red_prompt") or build_red_training_prompt(task.topic, weakness_summary)),
                    rejected_completion=serialize_task_json(task),
                    rejection_reason=rejection_reason,
                    spec=spec,
                    task_quality=float(judge_output.metadata.get("task_quality") or 0.0),
                    metadata={
                        "stage": "judge",
                        "weakness_summary": weakness_summary,
                        "observed_failure": task.observed_failure(),
                    },
                    iteration_index=iteration_index,
                )
                self.logger.warning(
                    "red_task_rejected_by_judge",
                    topic=task.topic,
                    task_quality=judge_output.metadata.get("task_quality"),
                    rejection_reason=rejection_reason,
                    observed_failure=task.observed_failure(),
                )

            if not hint_is_valid:
                self.logger.warning(
                    "socratic_hint_flagged_by_judge",
                    topic=task.topic,
                    reward=judge_output.normalized_reward,
                    reason=judge_output.metadata.get("hint_rejection_reason"),
                    local_tiebreak=judge_output.metadata.get("local_tiebreak"),
                    observed_failure=task.observed_failure(),
                )

            next_episode_id += 1
            preference_pairs_added = self._store_socratic_preferences_for_ranked_candidates(
                item=item,
                episode_id=next_episode_id,
                iteration_index=iteration_index,
            )
            ranked_candidates = list(item.get("hint_candidate_rankings") or [])
            episode = EpisodeRecord(
                episode_id=next_episode_id,
                topic=task.topic,
                task=task,
                hint=item["hint"],
                judge=judge_output,
                metadata={
                    "weakness_summary": weakness_summary,
                    "socratic_model": self.current_socratic_model,
                    "socratic_adapter": self.current_socratic_adapter,
                    "red_adapter": self._effective_red_generation_adapter(iteration_index),
                    "red_training_adapter": self.current_red_adapter,
                    "iteration": iteration_index,
                    "task_is_valid_for_socratic": task_is_valid,
                    "hint_is_valid_for_socratic": hint_is_valid,
                    "red_task_quality": judge_output.metadata.get("task_quality"),
                    "red_task_hardness": judge_output.metadata.get("task_hardness"),
                    "red_reward": judge_output.metadata.get("red_reward"),
                    "socratic_training_method": self._socratic_training_method(),
                    "socratic_candidate_count": len(item.get("hint_candidates") or [item["hint"]]),
                    "socratic_candidate_scores": [
                        candidate["judge"].metadata.get("post_normalize")
                        for candidate in ranked_candidates
                    ],
                    "socratic_dpo_pairs_added": preference_pairs_added,
                },
            )
            self.storage.append_episode(episode)
            should_reset, snapshot = self.curriculum.observe(
                episode.topic,
                judge_output.normalized_reward,
                update_weights=not self._using_uniform_curriculum(iteration_index),
            )
            self.storage.save_curriculum_state(snapshot)
            self.logger.event(
                "episode_complete",
                episode_id=episode.episode_id,
                topic=episode.topic,
                reward=judge_output.normalized_reward,
                score=judge_output.score,
                post_normalize=judge_output.metadata.get("post_normalize"),
                adjusted_score=judge_output.metadata.get("adjusted_score"),
                task_is_valid_for_socratic=task_is_valid,
                hint_is_valid_for_socratic=hint_is_valid,
                curriculum_weights=snapshot.weights,
            )
            self._log_episode_debug(episode, weakness_summary)
            if should_reset:
                self._reset_red_and_curriculum(episode.episode_id)
            records.append(episode)

        return records, next_episode_id

    def _load_red_generation_session(self, iteration_index: int):
        adapter_path = self._effective_red_generation_adapter(iteration_index)
        try:
            return self.model_pool.load_red_generation(
                adapter_path=adapter_path,
                allow_base_adapter_fallback=adapter_path is not None,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_cuda_memory()
            self.logger.warning(
                "red_generation_load_oom",
                iteration=iteration_index,
                adapter_path=adapter_path,
                error=str(exc),
            )
            if adapter_path is None:
                raise
            self._handle_red_adapter_failure(iteration_index, 0)
            return self.model_pool.load_red_generation(
                adapter_path=None,
                allow_base_adapter_fallback=True,
            )

    def _generate_iteration_tasks(self, target_count: int, iteration_index: int) -> List[Dict[str, Any]]:
        self._red_adapter_failed_last_iteration = False
        generated_requests: List[Dict[str, Any]] = []
        max_generation_attempts = max(1, target_count * max(2, self.config.task_execution.max_red_generation_attempts))
        generation_attempt = 0
        red_session = self._load_red_generation_session(iteration_index)
        try:
            while len(generated_requests) < target_count and generation_attempt < max_generation_attempts:
                remaining_slots = target_count - len(generated_requests)
                remaining_attempts = max_generation_attempts - generation_attempt
                wave_size = min(remaining_slots, remaining_attempts)
                if wave_size <= 0:
                    break
                request_batch: List[Dict[str, Any]] = []
                for _ in range(wave_size):
                    topic = self.curriculum.sample_topic(self.rng)
                    weakness_summary = self.curriculum.weakness_summary(topic)
                    request_batch.append(self._new_red_request(topic, weakness_summary))
                generation_attempt += len(request_batch)

                generated_requests.extend(self._generate_red_tasks_batch(red_session, request_batch, iteration_index))

                for item in request_batch:
                    if item.get("task") is not None:
                        continue
                    self.logger.warning(
                        "episode_skipped_red_failure",
                        iteration=iteration_index,
                        generation_attempt=generation_attempt,
                        topic=item["topic"],
                        weakness_summary=item["weakness_summary"],
                    )

            valid_requests, invalid_requests = self._validate_generated_requests(generated_requests, iteration_index)
            repaired_requests = self._repair_generated_requests_with_revalidation(
                red_session,
                invalid_requests,
                iteration_index,
            )
        finally:
            red_session.unload()

        final_requests = self._score_and_filter_red_tasks(valid_requests + repaired_requests, iteration_index)
        result_items = [
            {
                "task": item["task"],
                "weakness_summary": item["weakness_summary"],
                "red_judge": item.get("red_judge"),
            }
            for item in final_requests
            if item.get("task") is not None
        ]
        if (
            len(result_items) < target_count
            and not self._using_base_red_generation(iteration_index)
            and self._using_non_base_red_adapter()
        ):
            missing = target_count - len(result_items)
            self._handle_red_adapter_failure(
                iteration_index,
                len(result_items),
                reason="too_few_valid_red_tasks",
            )
            fallback_items = self._generate_iteration_tasks(missing, iteration_index)
            result_items.extend(fallback_items)
            self.logger.warning(
                "red_generation_base_fallback_fill",
                iteration=iteration_index,
                adapter_generated_tasks=len(final_requests),
                accepted_before_fallback=len(result_items) - len(fallback_items),
                fallback_tasks=len(fallback_items),
                requested_tasks=target_count,
            )
        self.logger.event(
            "iteration_red_generation_complete",
            iteration=iteration_index,
            requested_tasks=target_count,
            generated_tasks=len(result_items),
            raw_generated_tasks=len(generated_requests),
            validated_tasks=len(valid_requests),
            repaired_tasks=len(repaired_requests),
            attempts=generation_attempt,
            replica_count=1,
            shard_gpu_ids=self.config.red.hardware.gpu_ids,
            effective_batch_size=self._red_effective_batch_size(max(1, target_count)),
            red_generation_adapter=self._effective_red_generation_adapter(iteration_index),
        )
        return result_items

    def _generate_socratic_hints_for_iteration(self, items: List[Dict[str, Any]], iteration_index: int) -> List[Dict[str, Any]]:
        if not items:
            return []
        socratic_session = self.model_pool.get_socratic(
            model_source=self.current_socratic_model,
            adapter_path=self.current_socratic_adapter,
        )
        try:
            for item in items:
                if self._using_socratic_dpo():
                    candidate_count = max(2, int(self.config.socratic.dpo.num_hint_candidates))
                    candidates = generate_socratic_hints(
                        socratic_session,
                        item["task"],
                        count=candidate_count,
                        logger=self.logger,
                    )
                    item["hint_candidates"] = candidates
                    item["hint"] = candidates[0]
                else:
                    item["hint"] = generate_socratic_hint(socratic_session, item["task"], self.logger)
        finally:
            if not self.config.socratic.hardware.persistent:
                socratic_session.unload()
        self.logger.event(
            "iteration_socratic_generation_complete",
            iteration=iteration_index,
            hint_count=len(items),
            total_candidate_count=sum(len(item.get("hint_candidates") or [item["hint"]]) for item in items),
            socratic_training_method=self._socratic_training_method(),
        )
        return items

    def _run_iteration_updates(self, accepted_records: List[EpisodeRecord], iteration_index: int) -> None:
        step = accepted_records[-1].episode_id if accepted_records else self.storage.episode_count()
        if step <= 0:
            return

        if not accepted_records:
            self.logger.event(
                "iteration_updates_skipped",
                iteration=iteration_index,
                step=step,
                reason="zero_accepted_episodes",
                socratic_training_method=self._socratic_training_method(),
                socratic_adapter=self.current_socratic_adapter,
                red_adapter=self.current_red_adapter,
            )
            return

        if self._using_socratic_dpo():
            preferences = self.storage.load_socratic_preferences(self.config.socratic.dpo.max_training_pairs)
            socratic_result = self.socratic_dpo_updater.run(
                preferences=preferences,
                step=step,
                model_source=self.current_socratic_model,
                adapter_path=self.current_socratic_adapter,
            )
        else:
            recent_episodes = self.storage.load_recent_episodes(self.config.socratic.grpo.max_training_examples)
            socratic_result = self.socratic_grpo_updater.run(
                episodes=recent_episodes,
                step=step,
                model_source=self.current_socratic_model,
                adapter_path=self.current_socratic_adapter,
            )
        if socratic_result is not None:
            self.current_socratic_model = socratic_result.model_source
            self.current_socratic_adapter = socratic_result.adapter_path
            self.storage.save_pointer("socratic_model_path", self.current_socratic_model)
            self.storage.save_pointer("socratic_adapter_path", self.current_socratic_adapter)

        self.model_pool.release_socratic()
        clear_cuda_memory()
        hard_examples = self.storage.load_hard_examples(self.config.red.update.max_sft_examples)
        rejected_examples = self.storage.load_red_rejected_examples(
            max(self.config.red.update.max_dpo_pairs * 4, self.config.red.update.max_dpo_pairs)
        )
        recent_for_red = self.storage.load_recent_episodes(max(self.config.red.update.max_sft_examples, 256))
        if self._using_base_red_generation(iteration_index):
            self.logger.event(
                "red_update_skipped",
                iteration=iteration_index,
                step=step,
                reason="base_red_generation_mode_after_cutoff",
                red_generation_adapter=self._effective_red_generation_adapter(iteration_index),
                red_training_adapter=self.current_red_adapter,
            )
        else:
            red_result = self.red_updater.run(
                hard_examples=hard_examples,
                rejected_examples=rejected_examples,
                recent_episodes=recent_for_red,
                step=step,
                adapter_path=self.current_red_adapter,
            )
            if red_result.adapter_path:
                self.current_red_adapter = red_result.adapter_path
                self.storage.save_pointer("red_adapter_path", self.current_red_adapter)

        clear_cuda_memory()
        self.logger.event(
            "iteration_updates_complete",
            iteration=iteration_index,
            step=step,
            accepted_records=len(accepted_records),
            socratic_training_method=self._socratic_training_method(),
            socratic_adapter=self.current_socratic_adapter,
            red_adapter=self._effective_red_generation_adapter(iteration_index),
            red_training_adapter=self.current_red_adapter,
        )

    def _apply_iteration_curriculum_focus(self, iteration_index: int) -> None:
        weakest_topic, snapshot = self.curriculum.apply_iteration_focus_boost(
            enabled=not self._using_uniform_curriculum(iteration_index),
        )
        self.storage.save_curriculum_state(snapshot)
        self.logger.event(
            "curriculum_iteration_focus",
            iteration=iteration_index,
            weakest_topic=weakest_topic,
            curriculum_weights=snapshot.weights,
            running_topic_rewards=snapshot.running_topic_rewards,
        )

    def run(self) -> None:
        start_episode = self.storage.episode_count()
        self.logger.event(
            "pipeline_start",
            start_episode=start_episode,
            total_episodes=self.config.runtime.total_episodes,
            iteration_size=self._iteration_size(),
            completed_iterations=self.completed_iterations,
            socratic_model=self.current_socratic_model,
            socratic_adapter=self.current_socratic_adapter,
            socratic_training_method=self._socratic_training_method(),
            red_adapter=self.current_red_adapter,
        )
        self.model_pool.get_judge()

        try:
            target_episode = start_episode + self.config.runtime.total_episodes
            next_episode_id = start_episode
            iteration_index = self.completed_iterations
            stalled_iterations = 0

            while next_episode_id < target_episode:
                iteration_index += 1
                remaining = target_episode - next_episode_id
                requested_tasks = min(self._iteration_size(), remaining)
                self.logger.event(
                    "iteration_start",
                    iteration=iteration_index,
                    start_episode=next_episode_id,
                    requested_tasks=requested_tasks,
                )
                self._prepare_iteration_modes(iteration_index)

                generated = self._generate_iteration_tasks(requested_tasks, iteration_index)
                generated = self._generate_socratic_hints_for_iteration(generated, iteration_index)

                accepted_records: List[EpisodeRecord] = []
                pending_batch: List[Dict[str, Any]] = []
                for item in generated:
                    pending_batch.append(item)
                    if len(pending_batch) >= self.config.judge.episode_batch_size:
                        batch_records, next_episode_id = self._flush_pending_batch(
                            pending_batch,
                            next_episode_id,
                            iteration_index,
                        )
                        accepted_records.extend(batch_records)
                        pending_batch = []

                if pending_batch:
                    batch_records, next_episode_id = self._flush_pending_batch(
                        pending_batch,
                        next_episode_id,
                        iteration_index,
                    )
                    accepted_records.extend(batch_records)

                self._store_hard_examples_for_iteration(accepted_records, iteration_index)
                self._run_iteration_updates(accepted_records, iteration_index)
                self._apply_iteration_curriculum_focus(iteration_index)

                if accepted_records and accepted_records[-1].episode_id % self.config.runtime.checkpoint_every_episodes == 0:
                    self.storage.save_curriculum_state(self.curriculum.snapshot())
                    self.logger.event("checkpoint_marker", episode_id=accepted_records[-1].episode_id)

                self.logger.event(
                    "iteration_complete",
                    iteration=iteration_index,
                    accepted_episodes=len(accepted_records),
                    total_episodes=next_episode_id,
                )
                self.completed_iterations = iteration_index
                self.storage.save_pointer("completed_iterations", self.completed_iterations)
                if accepted_records:
                    stalled_iterations = 0
                else:
                    if self._red_adapter_failed_last_iteration:
                        stalled_iterations = 0
                    else:
                        stalled_iterations += 1
                    if stalled_iterations >= 5:
                        self.logger.error(
                            "pipeline_stalled",
                            iteration=iteration_index,
                            reason="five_iterations_without_accepted_episodes",
                        )
                        break
        finally:
            self.storage.save_curriculum_state(self.curriculum.snapshot())
            self.logger.event("pipeline_stop", model_pool=self.model_pool.debug_summary())
            self.model_pool.close()
