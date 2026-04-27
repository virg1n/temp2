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
    payload = {
        "topic": task.topic,
        "target_function": spec.get("target_function", ""),
        "intended_bug": spec.get("intended_bug", task.metadata.get("failure_mode", "")),
        "expected_first_failure": spec.get("expected_first_failure", task.observed_failure()),
        "statement": task.statement,
        "buggy_solution": _task_buggy_solution_for_output(task),
        "metadata": _task_output_metadata(task),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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


def _build_sft_dataset(
    examples: List[RedTrainingExample],
    *,
    tokenizer: Any,
    enable_thinking: bool,
) -> Dataset:
    ranked = sorted(examples, key=lambda item: item.reward)
    rows: List[Dict[str, str]] = []
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
        rows.append({"text": text})
    return Dataset.from_list(rows)


def _normalize_topic(topic: str) -> str:
    return " ".join(str(topic or "").lower().replace("_", " ").split())


def _is_already_correct_rejection(reason: Any) -> bool:
    text = str(reason or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "already_correct_code" in text or "already_correct" in text


def _rejected_example_is_already_correct(example: RedRejectedExample) -> bool:
    if _is_already_correct_rejection(example.rejection_reason):
        return True
    metadata = dict(example.metadata or {})
    if _is_already_correct_rejection(metadata.get("rejection_reason")):
        return True
    return str(metadata.get("execution_status") or "").strip().lower() == "passed"


def _chosen_example_has_direct_already_correct_rejection(example: RedTrainingExample) -> bool:
    metadata = dict(example.metadata or {})
    return (
        _is_already_correct_rejection(metadata.get("red_dpo_rejection_reason"))
        or _is_already_correct_rejection(metadata.get("rejection_reason"))
        or _is_already_correct_rejection(metadata.get("rejected_reason"))
    )


def _build_dpo_dataset(
    chosen_examples: List[RedTrainingExample],
    rejected_examples: List[RedRejectedExample],
    *,
    limit: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    limit = max(0, int(limit))
    if limit <= 0:
        return Dataset.from_list(rows), {
            "pairs": 0,
            "direct_pairs": 0,
            "topic_pairs": 0,
            "already_correct_rejections": 0,
            "topic_matches": {},
        }

    topic_index: Dict[str, List[RedTrainingExample]] = {}
    for example in sorted(chosen_examples, key=lambda entry: entry.reward):
        topic_key = _normalize_topic(example.topic or example.task.topic)
        if not topic_key:
            continue
        topic_index.setdefault(topic_key, []).append(example)

    direct_pairs = 0
    for example in sorted(chosen_examples, key=lambda entry: entry.reward):
        rejected = str(example.rejected_completion or "").strip()
        prompt = str(example.prompt or example.task.metadata.get("red_prompt") or "").strip()
        chosen = str(example.chosen_completion or "").strip()
        if (
            not prompt
            or not chosen
            or not rejected
            or rejected == chosen
            or not _chosen_example_has_direct_already_correct_rejection(example)
        ):
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        direct_pairs += 1
        if len(rows) >= limit:
            return Dataset.from_list(rows), {
                "pairs": len(rows),
                "direct_pairs": direct_pairs,
                "topic_pairs": 0,
                "already_correct_rejections": sum(1 for item in rejected_examples if _rejected_example_is_already_correct(item)),
                "topic_matches": {},
            }

    already_correct_rejections = [item for item in rejected_examples if _rejected_example_is_already_correct(item)]
    topic_offsets: Dict[str, int] = {}
    topic_matches: Dict[str, int] = {}
    topic_pairs = 0
    seen_pairs = {
        (row["prompt"], row["chosen"], row["rejected"])
        for row in rows
    }

    for item in already_correct_rejections:
        topic_key = _normalize_topic(item.topic)
        if not topic_key:
            continue
        candidates = topic_index.get(topic_key, [])
        if not candidates:
            continue
        offset = topic_offsets.get(topic_key, 0)
        candidate = candidates[offset % len(candidates)]
        topic_offsets[topic_key] = offset + 1
        prompt = str(candidate.prompt or candidate.task.metadata.get("red_prompt") or "").strip()
        chosen = str(candidate.chosen_completion or "").strip()
        rejected = str(item.rejected_completion or "").strip()
        pair_key = (prompt, chosen, rejected)
        if not prompt or not chosen or not rejected or chosen == rejected or pair_key in seen_pairs:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        seen_pairs.add(pair_key)
        topic_pairs += 1
        topic_matches[topic_key] = topic_matches.get(topic_key, 0) + 1
        if len(rows) >= limit:
            break

    return Dataset.from_list(rows), {
        "pairs": len(rows),
        "direct_pairs": direct_pairs,
        "topic_pairs": topic_pairs,
        "already_correct_rejections": len(already_correct_rejections),
        "topic_matches": topic_matches,
    }


def _hard_or_low_reward_episode_examples(episodes: List[EpisodeRecord], *, limit: int) -> List[RedTrainingExample]:
    rows: List[RedTrainingExample] = []
    valid_episodes = [
        episode
        for episode in episodes
        if episode.metadata.get("task_is_valid_for_socratic") is not False
    ]
    for episode in sorted(valid_episodes, key=lambda item: item.judge.normalized_reward)[:limit]:
        weakness_summary = str(episode.metadata.get("weakness_summary") or "general weakness probing")
        rows.append(
            RedTrainingExample(
                example_id=f"low_reward_recent_{episode.episode_id}",
                topic=episode.topic,
                prompt=str(episode.task.metadata.get("red_prompt") or build_red_training_prompt(episode.topic, weakness_summary)),
                chosen_completion=serialize_task_json(episode.task),
                rejected_completion=None,
                reward=episode.judge.normalized_reward,
                task=episode.task,
                metadata={
                    "episode_id": episode.episode_id,
                    "source": "low_reward_recent_episode",
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
        chosen_examples = list(hard_examples)
        if len(chosen_examples) < settings.min_hard_examples:
            fallback = _hard_or_low_reward_episode_examples(
                recent_episodes,
                limit=settings.min_hard_examples - len(chosen_examples),
            )
            existing_ids = {item.task.task_id for item in chosen_examples}
            for item in fallback:
                if item.task.task_id not in existing_ids:
                    chosen_examples.append(item)
                    existing_ids.add(item.task.task_id)

        if len(chosen_examples) < settings.min_hard_examples:
            reason = f"need_{settings.min_hard_examples}_chosen_examples_have_{len(chosen_examples)}"
            self.logger.event("red_update_skip", reason=reason)
            return RedUpdateResult(adapter_path=adapter_path, skipped_reason=reason)

        full_context_length = max(1024, min(int(settings.max_length), 1536))
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

                sft_dataset = _build_sft_dataset(
                    chosen_examples[-settings.max_sft_examples :],
                    tokenizer=session.tokenizer,
                    enable_thinking=self.config.red.enable_thinking,
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
                    "optim": "paged_adamw_8bit",
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
                            "optim": "paged_adamw_8bit",
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
                    rejected_examples=len(rejected_examples),
                    red_dpo_pairs=dpo_pair_count,
                    red_dpo_stats=dpo_stats,
                    recent_episodes=len(recent_episodes),
                    attempt=attempt_index,
                    max_length=attempt["max_length"],
                    loaded_adapter_path=load_adapter_path,
                )
                return RedUpdateResult(adapter_path=str(save_dir))
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
                clear_cuda_memory()

        return RedUpdateResult(adapter_path=adapter_path, skipped_reason="oom_after_retries")
