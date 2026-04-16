from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from judge_assess import HintJudgeAssessor
from logging_utils import get_logger, log_event
from prompts import build_socratic_messages
from schemas import PipelineSettings, SocraticState, TaskCandidate

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:  # pragma: no cover - version compatibility
    from trl.trainer.grpo_config import GRPOConfig  # type: ignore
    from trl.trainer.grpo_trainer import GRPOTrainer  # type: ignore


LOGGER = get_logger(__name__)


def _allowed_init_params(cls: Any) -> set[str] | None:
    try:
        signature = inspect.signature(cls.__init__)
    except Exception:
        return None
    allowed: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            return None
        if parameter.name != "self":
            allowed.add(parameter.name)
    return allowed


def _filter_kwargs_for_init(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = _allowed_init_params(cls)
    if allowed is None:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in allowed}


def _prompt_to_user_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "").strip()
    return str(prompt or "").strip()


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content") or "").strip()
    return str(completion or "").strip()


class SocraticGRPOTrainerWrapper:
    def __init__(self, settings: PipelineSettings, assessor: HintJudgeAssessor) -> None:
        self.settings = settings
        self.assessor = assessor

    def _build_dataset(self, tasks: list[TaskCandidate]) -> Dataset:
        records = [{"prompt": build_socratic_messages(task)} for task in tasks]
        return Dataset.from_list(records)

    def _build_reward_fn(self):
        def reward_fn(*, prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
            items: list[tuple[str, str, str]] = []
            for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
                items.append((f"grpo-item-{idx}", _prompt_to_user_text(prompt), _completion_to_text(completion)))
            evaluations = self.assessor.score_prompt_completion_pairs(items)
            return [
                float(evaluations.get(item_id).scores.final_reward if item_id in evaluations else 0.0)
                for item_id, _, _ in items
            ]

        return reward_fn

    def train(self, tasks: list[TaskCandidate], state: SocraticState, round_index: int) -> SocraticState:
        if not tasks:
            return state

        trainer_settings = self.settings.training.socratic
        output_dir = Path(trainer_settings.output_dir) / f"round_{round_index:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = state.active_model_path or self.settings.models.socratic.model_name_or_path
        adapter_path = state.active_adapter_path or self.settings.models.socratic.adapter_path

        set_seed(self.settings.runtime.seed + round_index)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=self.settings.models.socratic.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        reward_funcs = [self._build_reward_fn()]
        model_for_trainer: Any = model_path
        peft_config: Any | None = None
        model_load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.models.socratic.trust_remote_code,
            "torch_dtype": torch.bfloat16
            if self.settings.models.socratic.torch_dtype.lower() in {"bf16", "bfloat16"}
            else torch.float16,
        }
        if trainer_settings.full_ft:
            model_for_trainer = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_load_kwargs,
            )
        elif adapter_path:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_load_kwargs,
            )
            model_for_trainer = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        else:
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=trainer_settings.lora_r,
                lora_alpha=trainer_settings.lora_alpha,
                lora_dropout=trainer_settings.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=trainer_settings.lora_target_modules or None,
            )

        cfg_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "save_steps": trainer_settings.save_steps,
            "save_total_limit": trainer_settings.save_total_limit,
            "logging_steps": trainer_settings.log_steps,
            "per_device_train_batch_size": trainer_settings.per_device_batch_size,
            "gradient_accumulation_steps": trainer_settings.gradient_accumulation_steps,
            "num_train_epochs": trainer_settings.epochs,
            "learning_rate": trainer_settings.learning_rate,
            "warmup_ratio": trainer_settings.warmup_ratio,
            "weight_decay": trainer_settings.weight_decay,
            "optim": trainer_settings.optim,
            "bf16": trainer_settings.bf16,
            "fp16": trainer_settings.fp16,
            "gradient_checkpointing": trainer_settings.gradient_checkpointing,
            "max_prompt_length": trainer_settings.max_prompt_length,
            "max_completion_length": trainer_settings.max_completion_length,
            "num_generations": trainer_settings.num_hints_per_task,
            "temperature": self.settings.models.socratic.generation.temperature,
            "top_p": self.settings.models.socratic.generation.top_p,
            "beta": trainer_settings.beta,
            "deepspeed": trainer_settings.deepspeed,
            "report_to": "none",
            "ddp_find_unused_parameters": False,
            "model_init_kwargs": {"trust_remote_code": self.settings.models.socratic.trust_remote_code},
        }
        cfg = GRPOConfig(**_filter_kwargs_for_init(GRPOConfig, cfg_kwargs))
        if isinstance(model_for_trainer, str):
            allowed = _allowed_init_params(GRPOConfig) or set()
            if "model_init_kwargs" not in allowed and "model_kwargs" not in allowed:
                model_for_trainer = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)

        trainer_kwargs: dict[str, Any] = {
            "model": model_for_trainer,
            "reward_funcs": reward_funcs,
            "args": cfg,
            "train_dataset": self._build_dataset(tasks),
            "processing_class": tokenizer,
            "peft_config": peft_config,
        }
        trainer_kwargs = {key: value for key, value in _filter_kwargs_for_init(GRPOTrainer, trainer_kwargs).items() if value is not None}
        if "processing_class" not in trainer_kwargs and "tokenizer" in (_allowed_init_params(GRPOTrainer) or set()):
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()

        if trainer_settings.full_ft:
            model_out = output_dir / "model"
            trainer.save_model(str(model_out))
            tokenizer.save_pretrained(str(model_out))
            state.active_model_path = str(model_out)
            state.active_adapter_path = None
        else:
            adapter_out = output_dir / "adapter"
            trainer.model.save_pretrained(str(adapter_out))
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))
            state.active_model_path = model_path
            state.active_adapter_path = str(adapter_out)

        state.update_count += 1
        log_event(
            LOGGER,
            logging.INFO,
            "socratic_grpo_trained",
            "Completed Socratic GRPO update",
            round_index=round_index,
            task_count=len(tasks),
            active_model_path=state.active_model_path,
            active_adapter_path=state.active_adapter_path,
        )
        return state
