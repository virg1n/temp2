from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset

from .config import PipelineConfig
from .logging_utils import StructuredLogger
from .modeling import (
    ModelPool,
    attach_lora_adapter,
    clear_cuda_memory,
    is_oom_error,
    release_trainer_memory,
    render_chat_messages,
)
from .prompts import RED_SYSTEM_PROMPT, build_red_training_prompt
from .schemas import EpisodeRecord, PythonTask, RedRejectedExample, RedTrainingExample
from .storage import SimpleStorage

try:
    from trl import SFTConfig, SFTTrainer
except Exception:  # noqa: BLE001
    try:
        from trl.trainer.sft_config import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TRL with SFT support is required for Red updates.") from exc

try:
    from trl import DPOConfig, DPOTrainer
except Exception:  # noqa: BLE001
    try:
        from trl.trainer.dpo_config import DPOConfig
        from trl.trainer.dpo_trainer import DPOTrainer
    except Exception:  # noqa: BLE001
        DPOConfig = None
        DPOTrainer = None


@dataclass
class RedUpdateResult:
    adapter_path: Optional[str]
    skipped_reason: Optional[str] = None
    sft_example_ids: Optional[List[str]] = None
    dpo_chosen_example_ids: Optional[List[str]] = None
    dpo_rejected_example_ids: Optional[List[str]] = None


def _task_output_metadata(task: PythonTask) -> Dict[str, Any]:
    spec = dict(task.metadata.get("red_spec") or {})
    spec_metadata = dict(spec.get("metadata") or {})
    failure_mode = str(
        spec_metadata.get("failure_mode")
        or task.metadata.get("failure_mode")
        or spec.get("intended_bug")
        or "unspecified bug"
    ).strip()
    difficulty = str(spec_metadata.get("difficulty") or task.metadata.get("difficulty") or "").strip().lower()
    if difficulty not in {"medium", "hard"}:
        difficulty = "medium"
    return {
        "failure_mode": failure_mode,
        "difficulty": difficulty,
    }


def _task_buggy_solution_for_output(task: PythonTask) -> str:
    if any(str(item).strip() for item in task.failing_asserts):
        return task.combined_program()
    return task.buggy_solution


def serialize_task_json(task: PythonTask) -> str:
    spec = dict(task.metadata.get("red_spec") or {})
    reference_solution = str(task.reference_solution or task.metadata.get("reference_solution") or "").strip()
    payload = {
        "topic": task.topic,
        "target_function": spec.get("target_function", ""),
        "intended_bug": spec.get("intended_bug", task.metadata.get("failure_mode", "")),
        "expected_first_failure": spec.get("expected_first_failure", task.observed_failure()),
        "statement": task.statement,
        "reference_solution": reference_solution,
        "buggy_solution": _task_buggy_solution_for_output(task),
        "metadata": _task_output_metadata(task),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def serialize_red_completion(task: PythonTask) -> str:
    """Return the assistant completion matching the stored Red prompt format."""
    metadata = dict(task.metadata or {})
    for key in ("red_chosen_completion", "red_buggy_chosen_completion", "red_buggy_raw_response"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    if str(metadata.get("red_format") or "").startswith("iterative_plain_code"):
        return task.combined_program()
    return serialize_task_json(task)


def _allowed_init_params(cls: Any) -> Optional[set[str]]:
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return None
    allowed: set[str] = set()
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return None
        if param.name != "self":
            allowed.add(param.name)
    return allowed


def _filter_kwargs_for_init(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    allowed = _allowed_init_params(cls)
    if allowed is None:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in allowed}


def _safe_token_count(tokenizer: Any, text: str) -> Optional[int]:
    try:
        encoded = tokenizer(
            str(text or ""),
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if isinstance(input_ids, list):
            return len(input_ids)
    except Exception:
        return None
    return None


def _token_stats(values: List[Optional[int]]) -> Dict[str, Any]:
    known = [int(value) for value in values if value is not None]
    if not known:
        return {
            "count": len(values),
            "known": 0,
            "min": None,
            "max": None,
            "avg": None,
        }
    return {
        "count": len(values),
        "known": len(known),
        "min": min(known),
        "max": max(known),
        "avg": sum(known) / len(known),
    }


def _example_identity(example: RedTrainingExample) -> Dict[str, Any]:
    return {
        "example_id": example.example_id,
        "topic": example.topic,
        "stage": dict(example.metadata or {}).get("red_training_stage"),
        "episode_id": dict(example.metadata or {}).get("episode_id"),
        "reward": example.reward,
    }


def _build_sft_dataset(
    examples: List[RedTrainingExample],
    *,
    tokenizer: Any,
    enable_thinking: bool,
    max_length: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    ranked = sorted(examples, key=lambda item: item.reward)
    rows: List[Dict[str, str]] = []
    token_counts: List[Optional[int]] = []
    truncated_examples: List[Dict[str, Any]] = []
    for item in ranked:
        messages = [
            {"role": "system", "content": RED_SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
            {"role": "assistant", "content": item.chosen_completion},
        ]
        text = render_chat_messages(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
        )
        token_count = _safe_token_count(tokenizer, text)
        token_counts.append(token_count)
        if max_length > 0 and token_count is not None and token_count > max_length:
            truncated_examples.append(
                {
                    **_example_identity(item),
                    "tokens": token_count,
                    "max_length": max_length,
                    "prompt_chars": len(str(item.prompt or "")),
                    "completion_chars": len(str(item.chosen_completion or "")),
                }
            )
        rows.append({"text": text})
    return Dataset.from_list(rows), {
        "examples": len(examples),
        "rows": len(rows),
        "max_length": max_length,
        "dropped_by_max_length": 0,
        "would_truncate_by_max_length": len(truncated_examples),
        "token_counts": _token_stats(token_counts),
        "truncated_examples": truncated_examples[:10],
    }


def _normalize_topic(topic: str) -> str:
    return " ".join(str(topic or "").lower().replace("_", " ").split())


def _is_already_correct_rejection(reason: Any) -> bool:
    text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "already_correct_code" in text or "already_correct" in text


def _is_trainable_red_dpo_rejection_reason(reason: Any) -> bool:
    text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
    if _is_already_correct_rejection(text):
        return True
    return any(
        marker in text
        for marker in (
            "non_json_response",
            "buggy_too_correct",
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
            "buggy_solution_parse_error",
            "missing_shared_tests_in_buggy_solution",
            "solutions_do_not_share_identical_tests",
            "reference_solution_failed",
            "reference_solution_parse_error",
            "missing_tests_in_reference_solution",
            "too_few_tests_in_reference_solution",
            "too_many_tests_in_reference_solution",
        )
    )


def _rejected_example_is_already_correct(example: RedRejectedExample) -> bool:
    if _is_already_correct_rejection(example.rejection_reason):
        return True
    metadata = dict(example.metadata or {})
    if _is_already_correct_rejection(metadata.get("rejection_reason")):
        return True
    return str(metadata.get("execution_status") or "").strip().lower() == "passed"


def _rejected_example_is_trainable_for_red_dpo(example: RedRejectedExample) -> bool:
    if _is_trainable_red_dpo_rejection_reason(example.rejection_reason):
        return True
    metadata = dict(example.metadata or {})
    if _is_trainable_red_dpo_rejection_reason(metadata.get("rejection_reason")):
        return True
    if _is_trainable_red_dpo_rejection_reason(metadata.get("red_rejection_reason")):
        return True
    for reason in metadata.get("validation_reasons") or []:
        if _is_trainable_red_dpo_rejection_reason(reason):
            return True
    return _rejected_example_is_already_correct(example)


def _chosen_example_has_direct_trainable_rejection(example: RedTrainingExample) -> bool:
    metadata = dict(example.metadata or {})
    return (
        _is_trainable_red_dpo_rejection_reason(metadata.get("red_dpo_rejection_reason"))
        or _is_trainable_red_dpo_rejection_reason(metadata.get("rejection_reason"))
        or _is_trainable_red_dpo_rejection_reason(metadata.get("rejected_reason"))
    )


def _red_reward_from_example(example: RedTrainingExample) -> float:
    for source in (
        dict(example.metadata or {}),
        dict(getattr(example.task, "metadata", {}) or {}),
    ):
        try:
            return float(source.get("red_reward"))
        except Exception:
            continue
    return 0.0


def _red_reward_from_episode(episode: EpisodeRecord) -> float:
    for source in (
        dict(episode.metadata or {}),
        dict(episode.judge.metadata or {}),
        dict(episode.task.metadata or {}),
    ):
        try:
            return float(source.get("red_reward"))
        except Exception:
            continue
    return 0.0


def _is_current_plain_red_example(example: RedTrainingExample) -> bool:
    metadata = dict(example.metadata or {})
    if str(metadata.get("red_training_stage") or "") in {"task_reference", "task_buggy"}:
        return True
    prompt = str(example.prompt or "").lower()
    chosen = str(example.chosen_completion or "").lstrip()
    if not chosen or chosen.startswith(("{", "[")):
        return False
    return "strict json object" not in prompt and "json only" not in prompt


def _passes_red_reward_gate(value: float, min_red_reward: float) -> bool:
    return float(value) >= float(min_red_reward)


def _metadata_count(metadata: Dict[str, Any], key: str) -> int:
    try:
        return max(0, int(dict(metadata or {}).get(key) or 0))
    except Exception:
        return 0


def _under_use_cap(metadata: Dict[str, Any], key: str, max_uses: int, *, extra_uses: int = 0) -> bool:
    cap = int(max_uses)
    if cap <= 0:
        return True
    return _metadata_count(metadata, key) + int(extra_uses) < cap


def _build_dpo_dataset(
    chosen_examples: List[RedTrainingExample],
    rejected_examples: List[RedRejectedExample],
    *,
    limit: int,
    tokenizer: Any,
    max_length: int,
    max_prompt_length: int,
    max_dpo_uses: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    chosen_example_ids: List[str] = []
    rejected_example_ids: List[str] = []
    chosen_local_uses: Dict[str, int] = {}
    rejected_local_uses: Dict[str, int] = {}
    rejected_by_id = {str(item.example_id): item for item in rejected_examples}
    limit = max(0, int(limit))
    drop_counts: Dict[str, int] = {
        "missing_prompt": 0,
        "missing_chosen": 0,
        "missing_rejected": 0,
        "rejected_same_as_chosen": 0,
        "no_direct_trainable_rejection": 0,
        "chosen_dpo_use_cap": 0,
        "rejected_dpo_use_cap": 0,
        "missing_rejected_usage_record": 0,
    }
    prompt_token_counts: List[Optional[int]] = []
    chosen_total_token_counts: List[Optional[int]] = []
    rejected_total_token_counts: List[Optional[int]] = []
    truncated_pairs: List[Dict[str, Any]] = []

    def base_stats() -> Dict[str, Any]:
        return {
            "pairs": len(rows),
            "direct_pairs": len(rows),
            "topic_pairs": 0,
            "already_correct_rejections": sum(1 for item in rejected_examples if _rejected_example_is_already_correct(item)),
            "trainable_rejections": sum(1 for item in rejected_examples if _rejected_example_is_trainable_for_red_dpo(item)),
            "topic_matches": {},
            "candidate_chosen_examples": len(chosen_examples),
            "rejected_pool_examples": len(rejected_examples),
            "limit": limit,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "drop_counts": dict(drop_counts),
            "dropped_by_max_length": 0,
            "would_truncate_by_max_length": len(truncated_pairs),
            "prompt_token_counts": _token_stats(prompt_token_counts),
            "chosen_total_token_counts": _token_stats(chosen_total_token_counts),
            "rejected_total_token_counts": _token_stats(rejected_total_token_counts),
            "truncated_pairs": truncated_pairs[:10],
            "chosen_example_ids": list(chosen_example_ids),
            "rejected_example_ids": list(rejected_example_ids),
        }

    if limit <= 0:
        return Dataset.from_list(rows), base_stats()

    direct_pairs = 0
    for example in sorted(chosen_examples, key=lambda entry: entry.reward):
        metadata = dict(example.metadata or {})
        chosen_id = str(example.example_id)
        rejected_id = str(metadata.get("red_dpo_rejection_id") or "").strip()
        chosen_extra_uses = chosen_local_uses.get(chosen_id, 0)
        if not _under_use_cap(metadata, "red_dpo_use_count", max_dpo_uses, extra_uses=chosen_extra_uses):
            drop_counts["chosen_dpo_use_cap"] += 1
            continue
        rejected_example = rejected_by_id.get(rejected_id)
        if rejected_id and rejected_example is None:
            drop_counts["missing_rejected_usage_record"] += 1
            continue
        if rejected_example is not None:
            rejected_extra_uses = rejected_local_uses.get(rejected_id, 0)
            if not _under_use_cap(
                dict(rejected_example.metadata or {}),
                "red_dpo_use_count",
                max_dpo_uses,
                extra_uses=rejected_extra_uses,
            ):
                drop_counts["rejected_dpo_use_cap"] += 1
                continue
        rejected = str(example.rejected_completion or "").strip()
        prompt = str(example.prompt or example.task.metadata.get("red_prompt") or "").strip()
        chosen = str(example.chosen_completion or "").strip()
        if not prompt:
            drop_counts["missing_prompt"] += 1
            continue
        if not chosen:
            drop_counts["missing_chosen"] += 1
            continue
        if not rejected:
            drop_counts["missing_rejected"] += 1
            continue
        if rejected == chosen:
            drop_counts["rejected_same_as_chosen"] += 1
            continue
        if not _chosen_example_has_direct_trainable_rejection(example):
            drop_counts["no_direct_trainable_rejection"] += 1
            continue

        prompt_tokens = _safe_token_count(tokenizer, prompt)
        chosen_tokens = _safe_token_count(tokenizer, chosen)
        rejected_tokens = _safe_token_count(tokenizer, rejected)
        chosen_total = prompt_tokens + chosen_tokens if prompt_tokens is not None and chosen_tokens is not None else None
        rejected_total = prompt_tokens + rejected_tokens if prompt_tokens is not None and rejected_tokens is not None else None
        prompt_token_counts.append(prompt_tokens)
        chosen_total_token_counts.append(chosen_total)
        rejected_total_token_counts.append(rejected_total)
        would_truncate = (
            (max_prompt_length > 0 and prompt_tokens is not None and prompt_tokens > max_prompt_length)
            or (max_length > 0 and chosen_total is not None and chosen_total > max_length)
            or (max_length > 0 and rejected_total is not None and rejected_total > max_length)
        )
        if would_truncate:
            truncated_pairs.append(
                {
                    **_example_identity(example),
                    "prompt_tokens": prompt_tokens,
                    "chosen_total_tokens": chosen_total,
                    "rejected_total_tokens": rejected_total,
                    "max_length": max_length,
                    "max_prompt_length": max_prompt_length,
                }
            )
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        chosen_example_ids.append(chosen_id)
        if rejected_id:
            rejected_example_ids.append(rejected_id)
            rejected_local_uses[rejected_id] = rejected_local_uses.get(rejected_id, 0) + 1
        chosen_local_uses[chosen_id] = chosen_extra_uses + 1
        direct_pairs += 1
        if len(rows) >= limit:
            stats = base_stats()
            stats["direct_pairs"] = direct_pairs
            return Dataset.from_list(rows), stats

    stats = base_stats()
    stats["direct_pairs"] = direct_pairs
    return Dataset.from_list(rows), stats


def _hard_or_low_reward_episode_examples(
    episodes: List[EpisodeRecord],
    *,
    limit: int,
    max_reward: float,
    min_red_reward: float,
) -> List[RedTrainingExample]:
    rows: List[RedTrainingExample] = []
    valid_episodes = [
        episode
        for episode in episodes
        if episode.metadata.get("task_is_valid_for_socratic") is not False
        and float(episode.judge.normalized_reward) <= max_reward
        and _passes_red_reward_gate(_red_reward_from_episode(episode), min_red_reward)
    ]
    for episode in sorted(valid_episodes, key=lambda item: item.judge.normalized_reward)[:limit]:
        weakness_summary = str(episode.metadata.get("weakness_summary") or "general weakness probing")
        red_reward = _red_reward_from_episode(episode)
        rows.append(
            RedTrainingExample(
                example_id=f"low_reward_recent_{episode.episode_id}",
                topic=episode.topic,
                prompt=str(episode.task.metadata.get("red_prompt") or build_red_training_prompt(episode.topic, weakness_summary)),
                chosen_completion=serialize_red_completion(episode.task),
                rejected_completion=None,
                reward=episode.judge.normalized_reward,
                task=episode.task,
                metadata={
                    "episode_id": episode.episode_id,
                    "source": "low_reward_recent_episode",
                    "red_training_stage": (
                        "task_buggy"
                        if str(episode.task.metadata.get("red_format") or "").startswith("iterative_plain_code")
                        else ""
                    ),
                    "red_task_quality": episode.task.metadata.get("red_task_quality") or episode.judge.metadata.get("task_quality"),
                    "red_task_hardness": episode.task.metadata.get("red_task_hardness") or episode.judge.metadata.get("task_hardness"),
                    "red_reward": red_reward,
                },
            )
        )
    return rows


class RedUpdater:
    def __init__(self, config: PipelineConfig, model_pool: ModelPool, storage: SimpleStorage, logger: StructuredLogger) -> None:
        self.config = config
        self.model_pool = model_pool
        self.storage = storage
        self.logger = logger

    def run(
        self,
        *,
        hard_examples: List[RedTrainingExample],
        rejected_examples: List[RedRejectedExample],
        recent_episodes: List[EpisodeRecord],
        step: int,
        adapter_path: Optional[str],
    ) -> RedUpdateResult:
        settings = self.config.red.update
        hard_reward_max = float(settings.hard_reward_max)
        min_red_reward = float(settings.min_red_reward)
        max_sft_uses = int(getattr(settings, "max_sft_uses_per_example", 2))
        max_dpo_uses = int(getattr(settings, "max_dpo_uses_per_example", 3))
        eligible_hard_examples = [
            example
            for example in hard_examples
            if float(example.reward) <= hard_reward_max
            and _passes_red_reward_gate(_red_reward_from_example(example), min_red_reward)
        ]
        chosen_examples = [
            example
            for example in eligible_hard_examples
            if _under_use_cap(dict(example.metadata or {}), "red_sft_use_count", max_sft_uses)
        ]
        sft_use_cap_excluded = len(eligible_hard_examples) - len(chosen_examples)
        if len(chosen_examples) < settings.min_hard_examples and max_sft_uses <= 0:
            fallback = _hard_or_low_reward_episode_examples(
                recent_episodes,
                limit=settings.min_hard_examples - len(chosen_examples),
                max_reward=hard_reward_max,
                min_red_reward=min_red_reward,
            )
            existing_ids = {item.task.task_id for item in chosen_examples}
            for item in fallback:
                if item.task.task_id not in existing_ids:
                    chosen_examples.append(item)
                    existing_ids.add(item.task.task_id)

        unfiltered_chosen_count = len(chosen_examples)
        chosen_examples = [example for example in chosen_examples if _is_current_plain_red_example(example)]
        if len(chosen_examples) < settings.min_hard_examples:
            reason = f"need_{settings.min_hard_examples}_chosen_examples_have_{len(chosen_examples)}"
            self.logger.event(
                "red_update_skip",
                reason=reason,
                hard_examples=len(hard_examples),
                eligible_before_format_filter=unfiltered_chosen_count,
                eligible_chosen_examples=len(chosen_examples),
                sft_use_cap_excluded=sft_use_cap_excluded,
                max_sft_uses_per_example=max_sft_uses,
                hard_reward_max=hard_reward_max,
                min_red_reward=min_red_reward,
            )
            return RedUpdateResult(adapter_path=adapter_path, skipped_reason=reason)

        full_context_length = max(1024, int(settings.max_length))
        attempts = [
            {
                "max_length": full_context_length,
                "per_device_batch_size": 1,
                "dpo_enabled": settings.dpo_enabled,
            },
            {
                "max_length": max(768, full_context_length // 2),
                "per_device_batch_size": 1,
                "dpo_enabled": False,
            },
        ]

        for attempt_index, attempt in enumerate(attempts, start=1):
            session = None
            sft_trainer = None
            dpo_trainer = None
            sft_dataset = None
            dpo_dataset = None
            model = None
            load_adapter_path = adapter_path
            try:
                try:
                    session = self.model_pool.load_red_trainable(adapter_path=load_adapter_path)
                except RuntimeError as exc:
                    message = str(exc).lower()
                    adapter_load_failed = load_adapter_path is not None and (
                        is_oom_error(exc) or "failed to load adapter" in message
                    )
                    if not adapter_load_failed:
                        raise
                    self.logger.warning(
                        "red_update_adapter_load_fallback",
                        step=step,
                        attempt=attempt_index,
                        failed_adapter_path=load_adapter_path,
                        fallback_adapter_path=self.config.red.base_adapter_path,
                        error=str(exc),
                    )
                    clear_cuda_memory()
                    load_adapter_path = None
                    session = self.model_pool.load_red_trainable(adapter_path=None)

                model = session.model
                if self.config.red.lora.enabled and not hasattr(model, "peft_config"):
                    model = attach_lora_adapter(model, self.config.red.lora)
                    session.model = model

                sft_examples = chosen_examples[-settings.max_sft_examples :]
                sft_dataset, sft_stats = _build_sft_dataset(
                    sft_examples,
                    tokenizer=session.tokenizer,
                    enable_thinking=self.config.red.enable_thinking,
                    max_length=int(attempt["max_length"]),
                )
                self.logger.event(
                    "red_sft_dataset_built",
                    step=step,
                    attempt=attempt_index,
                    max_length=attempt["max_length"],
                    stats=sft_stats,
                )
                output_dir = str(self.storage.checkpoint_dir("red_tmp", step))

                sft_cfg_kwargs: Dict[str, Any] = {
                    "output_dir": output_dir,
                    "learning_rate": float(settings.learning_rate),
                    "num_train_epochs": int(settings.epochs),
                    "per_device_train_batch_size": int(attempt["per_device_batch_size"]),
                    "gradient_accumulation_steps": int(settings.gradient_accumulation_steps),
                    "max_seq_length": int(attempt["max_length"]),
                    "logging_steps": int(settings.logging_steps),
                    "save_strategy": "no",
                    "report_to": "none",
                    "optim": "adamw_torch",
                    "gradient_checkpointing": True,
                    "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                    "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
                }
                sft_cfg = SFTConfig(**_filter_kwargs_for_init(SFTConfig, sft_cfg_kwargs))
                trainer_kwargs: Dict[str, Any] = {
                    "model": model,
                    "args": sft_cfg,
                    "train_dataset": sft_dataset,
                    "processing_class": session.tokenizer,
                    "dataset_text_field": "text",
                }
                trainer_sig = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
                trainer_sig.discard("self")
                if "processing_class" not in trainer_sig and "tokenizer" in trainer_sig:
                    trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
                trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in trainer_sig}
                sft_trainer = SFTTrainer(**trainer_kwargs)
                sft_trainer.train()
                model = sft_trainer.model
                session.model = model
                release_trainer_memory(sft_trainer)
                sft_trainer = None
                sft_dataset = None
                clear_cuda_memory()

                dpo_pair_count = 0
                dpo_stats: Dict[str, Any] = {}
                if attempt["dpo_enabled"] and DPOTrainer is not None and DPOConfig is not None:
                    dpo_dataset, dpo_stats = _build_dpo_dataset(
                        chosen_examples,
                        rejected_examples,
                        limit=settings.max_dpo_pairs,
                        tokenizer=session.tokenizer,
                        max_length=int(attempt["max_length"]),
                        max_prompt_length=min(1024, int(attempt["max_length"]) // 2),
                        max_dpo_uses=max_dpo_uses,
                    )
                    dpo_pair_count = len(dpo_dataset)
                    self.logger.event(
                        "red_dpo_dataset_built",
                        step=step,
                        attempt=attempt_index,
                        max_length=attempt["max_length"],
                        pairs=dpo_pair_count,
                        stats=dpo_stats,
                    )
                    if len(dpo_dataset) > 0:
                        dpo_cfg_kwargs: Dict[str, Any] = {
                            "output_dir": output_dir,
                            "learning_rate": float(settings.learning_rate),
                            "num_train_epochs": int(settings.epochs),
                            "per_device_train_batch_size": int(attempt["per_device_batch_size"]),
                            "gradient_accumulation_steps": int(settings.gradient_accumulation_steps),
                            "max_length": int(attempt["max_length"]),
                            "max_prompt_length": min(1024, int(attempt["max_length"]) // 2),
                            "beta": float(settings.dpo_beta),
                            "logging_steps": int(settings.logging_steps),
                            "save_strategy": "no",
                            "report_to": "none",
                            "optim": "adamw_torch",
                            "gradient_checkpointing": True,
                            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                            "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
                        }
                        dpo_cfg = DPOConfig(**_filter_kwargs_for_init(DPOConfig, dpo_cfg_kwargs))
                        dpo_kwargs: Dict[str, Any] = {
                            "model": model,
                            "ref_model": None,
                            "args": dpo_cfg,
                            "train_dataset": dpo_dataset,
                            "processing_class": session.tokenizer,
                        }
                        dpo_sig = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
                        dpo_sig.discard("self")
                        if "processing_class" not in dpo_sig and "tokenizer" in dpo_sig:
                            dpo_kwargs["tokenizer"] = dpo_kwargs.pop("processing_class")
                        dpo_kwargs = {key: value for key, value in dpo_kwargs.items() if key in dpo_sig}
                        dpo_trainer = DPOTrainer(**dpo_kwargs)
                        dpo_trainer.train()
                        model = dpo_trainer.model
                        session.model = model
                        release_trainer_memory(dpo_trainer)
                        dpo_trainer = None
                        dpo_dataset = None
                        clear_cuda_memory()

                save_dir = self.storage.checkpoint_dir("red", step) / "adapter"
                model.save_pretrained(str(save_dir))
                session.tokenizer.save_pretrained(str(save_dir))
                self.storage.prune_role_checkpoints("red")
                self.logger.event(
                    "red_update_complete",
                    step=step,
                    adapter_path=str(save_dir),
                    hard_examples=len(hard_examples),
                    chosen_examples=len(chosen_examples),
                    hard_reward_max=hard_reward_max,
                    min_red_reward=min_red_reward,
                    max_sft_uses_per_example=max_sft_uses,
                    max_dpo_uses_per_example=max_dpo_uses,
                    sft_use_cap_excluded=sft_use_cap_excluded,
                    chosen_red_rewards=[_red_reward_from_example(example) for example in chosen_examples],
                    sft_example_ids=[example.example_id for example in sft_examples],
                    rejected_examples=len(rejected_examples),
                    red_dpo_pairs=dpo_pair_count,
                    red_dpo_stats=dpo_stats,
                    recent_episodes=len(recent_episodes),
                    attempt=attempt_index,
                    max_length=attempt["max_length"],
                    loaded_adapter_path=load_adapter_path,
                )
                return RedUpdateResult(
                    adapter_path=str(save_dir),
                    sft_example_ids=[example.example_id for example in sft_examples],
                    dpo_chosen_example_ids=list(dpo_stats.get("chosen_example_ids") or []),
                    dpo_rejected_example_ids=list(dpo_stats.get("rejected_example_ids") or []),
                )
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                for trainer in (dpo_trainer, sft_trainer):
                    if trainer is not None:
                        try:
                            release_trainer_memory(trainer)
                        except Exception:
                            pass
                self.logger.warning(
                    "red_update_oom_retry",
                    step=step,
                    attempt=attempt_index,
                    error=str(exc),
                )
            finally:
                if session is not None:
                    try:
                        session.unload()
                    except Exception:
                        pass
                session = None
                sft_trainer = None
                dpo_trainer = None
                sft_dataset = None
                dpo_dataset = None
                model = None
                # The OOM handler above creates a `for trainer in (...)` loop
                # whose variable persists in this scope and would otherwise
                # keep one trainer object alive past the explicit nulls.
                trainer = None  # noqa: F841
                clear_cuda_memory()

        return RedUpdateResult(adapter_path=adapter_path, skipped_reason="oom_after_retries")
