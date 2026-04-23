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
from .judge import JudgeService
from .logging_utils import StructuredLogger
from .modeling import ModelPool, attach_lora_adapter, clear_cuda_memory, is_oom_error, release_trainer_memory
from .prompts import build_socratic_messages
from .schemas import EpisodeRecord
from .storage import SimpleStorage

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:  # noqa: BLE001
    try:
        from trl.trainer.grpo_config import GRPOConfig
        from trl.trainer.grpo_trainer import GRPOTrainer
    except Exception:  # noqa: BLE001
        GRPOConfig = None
        GRPOTrainer = None


@dataclass
class SocraticUpdateResult:
    model_source: str
    adapter_path: Optional[str]


def _prompt_to_user_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "").strip()
        if prompt and isinstance(prompt[-1], dict):
            return str(prompt[-1].get("content") or "").strip()
        return ""
    return str(prompt or "").strip()


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return str(completion[0].get("content") or "").strip()
        return ""
    return str(completion or "").strip()


def _build_dataset(episodes: List[EpisodeRecord], max_examples: int) -> Dataset:
    records = [{"prompt": build_socratic_messages(item.task)} for item in episodes[-max_examples:]]
    if not records:
        raise ValueError("No episodes available for GRPO.")
    return Dataset.from_list(records)


def _compat_grpo_config(config: PipelineConfig, output_dir: str) -> Any:
    args = config.socratic.grpo
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
        "optim": "paged_adamw_8bit",
        "max_prompt_length": int(args.max_prompt_length),
        "max_completion_length": int(args.max_completion_length),
        "num_generations": int(args.num_generations),
        "temperature": float(config.socratic.generation.temperature),
        "top_p": float(config.socratic.generation.top_p),
        "beta": float(args.beta),
        "scale_rewards": True,
        "reward_weights": [1.0],
        "ddp_find_unused_parameters": False,
        "report_to": "none",
    }

    param_names = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    param_names.discard("self")
    alias_map: Dict[str, List[str]] = {
        "max_prompt_length": ["max_prompt_len", "max_prompt_tokens"],
        "max_completion_length": ["max_new_tokens", "max_completion_len"],
        "num_generations": ["num_return_sequences", "num_samples"],
        "top_p": ["p", "top_p_sampling"],
    }

    filtered: Dict[str, Any] = {}
    for key, value in cfg_kwargs.items():
        if key in param_names:
            filtered[key] = value
            continue
        for alt in alias_map.get(key, []):
            if alt in param_names:
                filtered[alt] = value
                break
    return GRPOConfig(**filtered)


class SocraticGrpoUpdater:
    def __init__(
        self,
        config: PipelineConfig,
        model_pool: ModelPool,
        judge: JudgeService,
        storage: SimpleStorage,
        logger: StructuredLogger,
    ) -> None:
        self.config = config
        self.model_pool = model_pool
        self.judge = judge
        self.storage = storage
        self.logger = logger

    def run(
        self,
        *,
        episodes: List[EpisodeRecord],
        step: int,
        model_source: str,
        adapter_path: Optional[str],
    ) -> Optional[SocraticUpdateResult]:
        if GRPOConfig is None or GRPOTrainer is None:
            raise RuntimeError("TRL with GRPO support is required when socratic.training_method is 'grpo'.")

        if len(episodes) < self.config.socratic.grpo.min_episodes_before_update:
            self.logger.event(
                "socratic_grpo_skip",
                reason="not_enough_episodes",
                have=len(episodes),
                need=self.config.socratic.grpo.min_episodes_before_update,
            )
            return None

        set_seed(self.config.runtime.seed + int(step))
        dataset = _build_dataset(episodes, self.config.socratic.grpo.max_training_examples)
        session = None
        try:
            session = self.model_pool.load_socratic_trainable(
                model_source=model_source,
                adapter_path=adapter_path,
            )
            model = session.model
            if (
                not self.config.socratic.grpo.full_ft
                and self.config.socratic.lora.enabled
                and not hasattr(model, "peft_config")
            ):
                model = attach_lora_adapter(model, self.config.socratic.lora)
                session.model = model

            def reward_judge(*, prompts: List[Any], completions: List[Any], **_: Any) -> List[float]:
                prompt_texts = [_prompt_to_user_text(item) for item in prompts]
                completion_texts = [_completion_to_text(item) for item in completions]
                return self.judge.score_pairs(prompt_texts, completion_texts)

            cfg = _compat_grpo_config(
                self.config,
                output_dir=str(self.storage.checkpoint_dir("socratic_tmp", step)),
            )
            trainer_kwargs: Dict[str, Any] = {
                "model": model,
                "reward_funcs": [reward_judge],
                "args": cfg,
                "train_dataset": dataset,
                "processing_class": session.tokenizer,
            }

            trainer_sig = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
            trainer_sig.discard("self")
            if "processing_class" not in trainer_sig and "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
            trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in trainer_sig and value is not None}

            trainer = GRPOTrainer(**trainer_kwargs)
            trainer.train()

            save_root = self.storage.checkpoint_dir("socratic", step)
            if self.config.socratic.grpo.full_ft or not hasattr(trainer.model, "peft_config"):
                model_dir = save_root / "model"
                trainer.save_model(str(model_dir))
                session.tokenizer.save_pretrained(str(model_dir))
                result = SocraticUpdateResult(model_source=str(model_dir), adapter_path=None)
            else:
                adapter_dir = save_root / "adapter"
                trainer.model.save_pretrained(str(adapter_dir))
                session.tokenizer.save_pretrained(str(adapter_dir))
                result = SocraticUpdateResult(model_source=model_source, adapter_path=str(adapter_dir))

            release_trainer_memory(trainer)
            self.storage.prune_role_checkpoints("socratic")
            self.logger.event(
                "socratic_grpo_complete",
                step=step,
                model_source=result.model_source,
                adapter_path=result.adapter_path,
                episodes_used=len(dataset),
            )
            return result
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_cuda_memory()
            self.logger.warning("socratic_grpo_oom", step=step, error=str(exc))
            return None
        finally:
            self.model_pool.release_socratic()
            if session is not None:
                session.unload()
            clear_cuda_memory()
