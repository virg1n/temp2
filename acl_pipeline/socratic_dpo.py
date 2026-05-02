from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import set_seed

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
from .prompts import build_socratic_messages
from .schemas import SocraticPreferenceExample
from .storage import SimpleStorage

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
class SocraticDpoUpdateResult:
    model_source: str
    adapter_path: Optional[str]
    preference_example_ids: Optional[List[str]] = None


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


def _metadata_count(metadata: Dict[str, Any], key: str) -> int:
    try:
        return max(0, int(dict(metadata or {}).get(key) or 0))
    except Exception:
        return 0


def _under_use_cap(metadata: Dict[str, Any], key: str, max_uses: int) -> bool:
    cap = int(max_uses)
    if cap <= 0:
        return True
    return _metadata_count(metadata, key) < cap


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


def _build_dataset(
    examples: List[SocraticPreferenceExample],
    *,
    tokenizer: Any,
    enable_thinking: bool,
    max_pairs: int,
    max_length: int,
    max_prompt_length: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    rows: List[Dict[str, str]] = []
    used_example_ids: List[str] = []
    limit = max(0, int(max_pairs))
    drop_counts: Dict[str, int] = {
        "missing_chosen": 0,
        "missing_rejected": 0,
        "chosen_same_as_rejected": 0,
    }
    prompt_token_counts: List[Optional[int]] = []
    chosen_total_token_counts: List[Optional[int]] = []
    rejected_total_token_counts: List[Optional[int]] = []
    truncated_examples: List[Dict[str, Any]] = []

    def stats() -> Dict[str, Any]:
        return {
            "examples": len(examples),
            "limit": limit,
            "rows": len(rows),
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "drop_counts": dict(drop_counts),
            "dropped_by_max_length": 0,
            "would_truncate_by_max_length": len(truncated_examples),
            "prompt_token_counts": _token_stats(prompt_token_counts),
            "chosen_total_token_counts": _token_stats(chosen_total_token_counts),
            "rejected_total_token_counts": _token_stats(rejected_total_token_counts),
            "truncated_examples": truncated_examples[:10],
            "used_example_ids": list(used_example_ids),
        }

    if limit <= 0:
        return Dataset.from_list(rows), stats()
    for example in examples[-limit:]:
        chosen = str(example.chosen_hint or "").strip()
        rejected = str(example.rejected_hint or "").strip()
        if not chosen:
            drop_counts["missing_chosen"] += 1
            continue
        if not rejected:
            drop_counts["missing_rejected"] += 1
            continue
        if chosen == rejected:
            drop_counts["chosen_same_as_rejected"] += 1
            continue
        prompt = render_chat_messages(
            tokenizer,
            build_socratic_messages(example.task),
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
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
            truncated_examples.append(
                {
                    "example_id": example.example_id,
                    "topic": example.topic,
                    "episode_id": dict(example.metadata or {}).get("episode_id"),
                    "task_id": example.task.task_id,
                    "chosen_score": example.chosen_score,
                    "rejected_score": example.rejected_score,
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
        used_example_ids.append(str(example.example_id))
    return Dataset.from_list(rows), stats()


def _compat_dpo_config(config: PipelineConfig, output_dir: str) -> Any:
    args = config.socratic.dpo
    cfg_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "save_total_limit": int(args.save_total_limit),
        "save_steps": int(args.save_steps),
        "logging_steps": int(args.logging_steps),
        "per_device_train_batch_size": int(args.per_device_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "num_train_epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "weight_decay": float(args.weight_decay),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "optim": "adamw_torch",
        "max_length": int(args.max_length),
        "max_prompt_length": int(args.max_prompt_length),
        "beta": float(args.beta),
        "ddp_find_unused_parameters": False,
        "report_to": "none",
    }
    return DPOConfig(**_filter_kwargs_for_init(DPOConfig, cfg_kwargs))


class SocraticDpoUpdater:
    def __init__(
        self,
        config: PipelineConfig,
        model_pool: ModelPool,
        storage: SimpleStorage,
        logger: StructuredLogger,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self.storage = storage
        self.logger = logger

    def run(
        self,
        *,
        preferences: List[SocraticPreferenceExample],
        step: int,
        model_source: str,
        adapter_path: Optional[str],
    ) -> Optional[SocraticDpoUpdateResult]:
        if DPOConfig is None or DPOTrainer is None:
            raise RuntimeError("TRL with DPO support is required when socratic.training_method is 'dpo'.")

        settings = self.config.socratic.dpo
        max_uses = int(getattr(settings, "max_uses_per_preference", 3))
        eligible_preferences = [
            example
            for example in preferences
            if _under_use_cap(dict(example.metadata or {}), "socratic_dpo_use_count", max_uses)
        ]
        preference_use_cap_excluded = len(preferences) - len(eligible_preferences)
        if len(eligible_preferences) < settings.min_preference_pairs_before_update:
            self.logger.event(
                "socratic_dpo_skip",
                reason="not_enough_preference_pairs",
                have=len(eligible_preferences),
                need=settings.min_preference_pairs_before_update,
                total_preferences=len(preferences),
                preference_use_cap_excluded=preference_use_cap_excluded,
                max_uses_per_preference=max_uses,
            )
            return None

        set_seed(self.config.runtime.seed + int(step))
        session = None
        trainer = None
        dataset = None
        model = None
        try:
            session = self.model_pool.load_socratic_trainable(
                model_source=model_source,
                adapter_path=adapter_path,
            )
            model = session.model
            if (
                not settings.full_ft
                and self.config.socratic.lora.enabled
                and not hasattr(model, "peft_config")
            ):
                model = attach_lora_adapter(model, self.config.socratic.lora)
                session.model = model

            dataset, dataset_stats = _build_dataset(
                eligible_preferences,
                tokenizer=session.tokenizer,
                enable_thinking=False,
                max_pairs=settings.max_training_pairs,
                max_length=int(settings.max_length),
                max_prompt_length=int(settings.max_prompt_length),
            )
            self.logger.event(
                "socratic_dpo_dataset_built",
                step=step,
                max_length=settings.max_length,
                max_prompt_length=settings.max_prompt_length,
                max_uses_per_preference=max_uses,
                preference_use_cap_excluded=preference_use_cap_excluded,
                stats=dataset_stats,
            )
            if len(dataset) <= 0:
                self.logger.event(
                    "socratic_dpo_skip",
                    reason="empty_preference_dataset_after_filtering",
                    preferences=len(eligible_preferences),
                    preference_use_cap_excluded=preference_use_cap_excluded,
                )
                return None

            cfg = _compat_dpo_config(
                self.config,
                output_dir=str(self.storage.checkpoint_dir("socratic_tmp", step)),
            )
            trainer_kwargs: Dict[str, Any] = {
                "model": model,
                "ref_model": None,
                "args": cfg,
                "train_dataset": dataset,
                "processing_class": session.tokenizer,
            }

            trainer_sig = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
            trainer_sig.discard("self")
            if "processing_class" not in trainer_sig and "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
            trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in trainer_sig and value is not None}

            trainer = DPOTrainer(**trainer_kwargs)
            trainer.train()

            save_root = self.storage.checkpoint_dir("socratic", step)
            if settings.full_ft or not hasattr(trainer.model, "peft_config"):
                model_dir = save_root / "model"
                trainer.save_model(str(model_dir))
                session.tokenizer.save_pretrained(str(model_dir))
                result = SocraticDpoUpdateResult(
                    model_source=str(model_dir),
                    adapter_path=None,
                    preference_example_ids=list(dataset_stats.get("used_example_ids") or []),
                )
            else:
                adapter_dir = save_root / "adapter"
                trainer.model.save_pretrained(str(adapter_dir))
                session.tokenizer.save_pretrained(str(adapter_dir))
                result = SocraticDpoUpdateResult(
                    model_source=model_source,
                    adapter_path=str(adapter_dir),
                    preference_example_ids=list(dataset_stats.get("used_example_ids") or []),
                )

            preference_pairs_used = len(dataset)
            release_trainer_memory(trainer)
            trainer = None
            dataset = None
            model = None
            clear_cuda_memory()
            self.storage.prune_role_checkpoints("socratic")
            self.logger.event(
                "socratic_dpo_complete",
                step=step,
                model_source=result.model_source,
                adapter_path=result.adapter_path,
                preference_pairs_used=preference_pairs_used,
                preference_example_ids=result.preference_example_ids,
                max_uses_per_preference=max_uses,
                preference_use_cap_excluded=preference_use_cap_excluded,
            )
            return result
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_cuda_memory()
            self.logger.warning("socratic_dpo_oom", step=step, error=str(exc))
            return None
        finally:
            if trainer is not None:
                try:
                    release_trainer_memory(trainer)
                except Exception:
                    pass
            self.model_pool.release_socratic()
            if session is not None:
                session.unload()
            trainer = None
            dataset = None
            model = None
            clear_cuda_memory()
