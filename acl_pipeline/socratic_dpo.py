from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import set_seed

from .config import PipelineConfig
from .logging_utils import StructuredLogger
from .modeling import ModelPool, attach_lora_adapter, clear_cuda_memory, is_oom_error, render_chat_messages
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


def _build_dataset(
    examples: List[SocraticPreferenceExample],
    *,
    tokenizer: Any,
    enable_thinking: bool,
    max_pairs: int,
) -> Dataset:
    rows: List[Dict[str, str]] = []
    limit = max(0, int(max_pairs))
    if limit <= 0:
        return Dataset.from_list(rows)
    for example in examples[-limit:]:
        chosen = str(example.chosen_hint or "").strip()
        rejected = str(example.rejected_hint or "").strip()
        if not chosen or not rejected or chosen == rejected:
            continue
        prompt = render_chat_messages(
            tokenizer,
            build_socratic_messages(example.task),
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return Dataset.from_list(rows)


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
        if len(preferences) < settings.min_preference_pairs_before_update:
            self.logger.event(
                "socratic_dpo_skip",
                reason="not_enough_preference_pairs",
                have=len(preferences),
                need=settings.min_preference_pairs_before_update,
            )
            return None

        set_seed(self.config.runtime.seed + int(step))
        session = None
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

            dataset = _build_dataset(
                preferences,
                tokenizer=session.tokenizer,
                enable_thinking=False,
                max_pairs=settings.max_training_pairs,
            )
            if len(dataset) <= 0:
                self.logger.event(
                    "socratic_dpo_skip",
                    reason="empty_preference_dataset_after_filtering",
                    preferences=len(preferences),
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
                result = SocraticDpoUpdateResult(model_source=str(model_dir), adapter_path=None)
            else:
                adapter_dir = save_root / "adapter"
                trainer.model.save_pretrained(str(adapter_dir))
                session.tokenizer.save_pretrained(str(adapter_dir))
                result = SocraticDpoUpdateResult(model_source=model_source, adapter_path=str(adapter_dir))

            self.storage.prune_role_checkpoints("socratic")
            self.logger.event(
                "socratic_dpo_complete",
                step=step,
                model_source=result.model_source,
                adapter_path=result.adapter_path,
                preference_pairs_used=len(dataset),
            )
            return result
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_cuda_memory()
            self.logger.warning("socratic_dpo_oom", step=step, error=str(exc))
            return None
        finally:
            self.model_pool.release_socratic()
            if session is not None:
                session.unload()
            clear_cuda_memory()
