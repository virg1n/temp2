from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset

from .config import PipelineConfig
from .logging_utils import StructuredLogger
from .modeling import ModelPool, attach_lora_adapter, clear_cuda_memory, is_oom_error, render_chat_messages
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


def serialize_task_json(task: PythonTask) -> str:
    spec = dict(task.metadata.get("red_spec") or {})
    metadata = {
        key: value
        for key, value in dict(task.metadata).items()
        if key in {"failure_mode", "difficulty", "observed_failure", "execution_status"}
    }
    payload = {
        "topic": task.topic,
        "target_function": spec.get("target_function", ""),
        "intended_bug": spec.get("intended_bug", task.metadata.get("failure_mode", "")),
        "expected_first_failure": spec.get("expected_first_failure", task.observed_failure()),
        "statement": task.statement,
        "buggy_solution": task.combined_program(),
        "metadata": metadata,
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


def _build_dpo_dataset(
    hard_examples: List[RedTrainingExample],
    rejected_examples: List[RedRejectedExample],
    recent_episodes: List[EpisodeRecord],
    *,
    limit: int,
) -> Dataset:
    easy_by_topic: Dict[str, EpisodeRecord] = {}
    global_easy: Optional[EpisodeRecord] = None

    for episode in sorted(recent_episodes, key=lambda item: item.judge.normalized_reward, reverse=True):
        if global_easy is None:
            global_easy = episode
        easy_by_topic.setdefault(episode.topic, episode)

    rows: List[Dict[str, str]] = []
    for item in sorted(hard_examples, key=lambda entry: entry.reward)[:limit]:
        easy = easy_by_topic.get(item.topic) or global_easy
        if easy is None:
            continue
        rejected = serialize_task_json(easy.task)
        if rejected == item.chosen_completion:
            continue
        rows.append(
            {
                "prompt": item.prompt,
                "chosen": item.chosen_completion,
                "rejected": rejected,
            }
        )

    remaining = max(0, limit - len(rows))
    if remaining > 0:
        for item in rejected_examples[-remaining:]:
            chosen_episode = easy_by_topic.get(item.topic) or global_easy
            if chosen_episode is None:
                continue
            chosen = serialize_task_json(chosen_episode.task)
            rejected = str(item.rejected_completion or "").strip()
            if not rejected or rejected == chosen:
                continue
            rows.append(
                {
                    "prompt": item.prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    return Dataset.from_list(rows)


def _recent_episode_examples(episodes: List[EpisodeRecord], *, limit: int) -> List[RedTrainingExample]:
    rows: List[RedTrainingExample] = []
    for episode in sorted(episodes, key=lambda item: item.judge.normalized_reward, reverse=True)[:limit]:
        weakness_summary = str(episode.metadata.get("weakness_summary") or "general weakness probing")
        rows.append(
            RedTrainingExample(
                example_id=f"recent_{episode.episode_id}",
                topic=episode.topic,
                prompt=build_red_training_prompt(episode.topic, weakness_summary),
                chosen_completion=serialize_task_json(episode.task),
                rejected_completion=None,
                reward=episode.judge.normalized_reward,
                task=episode.task,
                metadata={
                    "episode_id": episode.episode_id,
                    "source": "recent_episode",
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
            fallback = _recent_episode_examples(
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

        attempts = [
            {
                "max_length": settings.max_length,
                "per_device_batch_size": settings.per_device_batch_size,
                "dpo_enabled": settings.dpo_enabled,
            },
            {
                "max_length": max(512, settings.max_length // 2),
                "per_device_batch_size": 1,
                "dpo_enabled": False,
            },
        ]

        for attempt_index, attempt in enumerate(attempts, start=1):
            session = None
            try:
                session = self.model_pool.load_red_trainable(adapter_path=adapter_path)
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

                if attempt["dpo_enabled"] and DPOTrainer is not None and DPOConfig is not None:
                    dpo_dataset = _build_dpo_dataset(
                        hard_examples,
                        rejected_examples,
                        recent_episodes,
                        limit=settings.max_dpo_pairs,
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
                    recent_episodes=len(recent_episodes),
                    attempt=attempt_index,
                )
                return RedUpdateResult(adapter_path=str(save_dir))
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                clear_cuda_memory()
                self.logger.warning(
                    "red_update_oom_retry",
                    step=step,
                    attempt=attempt_index,
                    error=str(exc),
                )
            finally:
                if session is not None:
                    session.unload()
                clear_cuda_memory()

        return RedUpdateResult(adapter_path=adapter_path, skipped_reason="oom_after_retries")
