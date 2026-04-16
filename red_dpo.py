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

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from logging_utils import get_logger, log_event
from prompts import RED_SYSTEM_PROMPT
from schemas import DpoTrainingPair, PipelineSettings, RedAdaptationState, RedSFTExample

try:
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
except Exception:  # pragma: no cover - version compatibility
    from trl.trainer.dpo_config import DPOConfig  # type: ignore
    from trl.trainer.dpo_trainer import DPOTrainer  # type: ignore
    from trl.trainer.sft_config import SFTConfig  # type: ignore
    from trl.trainer.sft_trainer import SFTTrainer  # type: ignore


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


def _apply_red_chat_template(tokenizer: Any, prompt: str, response: str) -> str:
    messages = [
        {"role": "system", "content": RED_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class RedTrainer:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def _model_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.models.red.trust_remote_code,
            "device_map": self.settings.models.red.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16 if self.settings.models.red.torch_dtype.lower() in {"bf16", "bfloat16"} else torch.float16,
        }
        if self.settings.models.red.quantization.load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=float(self.settings.models.red.quantization.llm_int8_threshold),
            )
        return kwargs

    def _load_trainable_model(self, model_path: str, adapter_path: str | None) -> tuple[Any, Any]:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=self.settings.models.red.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, **self._model_load_kwargs())
        if self.settings.models.red.quantization.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            peft_config = LoraConfig(
                r=self.settings.training.red.lora_r,
                lora_alpha=self.settings.training.red.lora_alpha,
                lora_dropout=self.settings.training.red.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.settings.training.red.lora_target_modules or None,
            )
            model = get_peft_model(model, peft_config)
        return model, tokenizer

    def _load_reference_model(self, model_path: str, adapter_path: str | None) -> Any:
        from peft import PeftModel

        ref_model = AutoModelForCausalLM.from_pretrained(model_path, **self._model_load_kwargs())
        if adapter_path:
            ref_model = PeftModel.from_pretrained(ref_model, adapter_path, is_trainable=False)
        ref_model.eval()
        return ref_model

    def _build_sft_dataset(self, tokenizer: Any, examples: list[RedSFTExample]) -> Dataset:
        rows = [{"text": _apply_red_chat_template(tokenizer, item.prompt, item.response)} for item in examples]
        return Dataset.from_list(rows)

    def _build_dpo_dataset(self, pairs: list[DpoTrainingPair]) -> Dataset:
        rows = [
            {
                "prompt": item.prompt,
                "chosen": item.chosen,
                "rejected": item.rejected,
            }
            for item in pairs
        ]
        return Dataset.from_list(rows)

    def _run_sft(self, examples: list[RedSFTExample], state: RedAdaptationState, round_index: int) -> str:
        training = self.settings.training.red
        output_dir = Path(training.output_dir) / f"round_{round_index:05d}" / "sft"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.settings.models.red.model_name_or_path
        adapter_path = state.active_adapter_path or self.settings.models.red.adapter_path
        model, tokenizer = self._load_trainable_model(model_path, adapter_path)
        dataset = self._build_sft_dataset(tokenizer, examples)

        cfg_kwargs = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "save_steps": training.save_steps,
            "save_total_limit": training.save_total_limit,
            "logging_steps": training.log_steps,
            "per_device_train_batch_size": training.per_device_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "num_train_epochs": training.epochs,
            "learning_rate": training.learning_rate,
            "warmup_ratio": training.warmup_ratio,
            "weight_decay": training.weight_decay,
            "optim": training.optim,
            "bf16": training.bf16,
            "fp16": training.fp16,
            "gradient_checkpointing": training.gradient_checkpointing,
            "report_to": "none",
        }
        config = SFTConfig(**_filter_kwargs_for_init(SFTConfig, cfg_kwargs))

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": config,
            "train_dataset": dataset,
            "processing_class": tokenizer,
            "dataset_text_field": "text",
            "max_seq_length": training.max_length,
        }
        allowed = _allowed_init_params(SFTTrainer) or set()
        if "processing_class" not in allowed and "tokenizer" in allowed:
            trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
        trainer_kwargs = _filter_kwargs_for_init(SFTTrainer, trainer_kwargs)
        trainer = SFTTrainer(**trainer_kwargs)
        trainer.train()

        adapter_out = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_out))
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        return str(adapter_out)

    def _run_dpo(self, pairs: list[DpoTrainingPair], state: RedAdaptationState, round_index: int, adapter_path: str) -> str:
        training = self.settings.training.red
        output_dir = Path(training.output_dir) / f"round_{round_index:05d}" / "dpo"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.settings.models.red.model_name_or_path
        model, tokenizer = self._load_trainable_model(model_path, adapter_path)
        ref_model = self._load_reference_model(model_path, adapter_path)
        dataset = self._build_dpo_dataset(pairs)

        cfg_kwargs = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "save_steps": training.save_steps,
            "save_total_limit": training.save_total_limit,
            "logging_steps": training.log_steps,
            "per_device_train_batch_size": training.per_device_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "num_train_epochs": training.epochs,
            "learning_rate": training.learning_rate,
            "warmup_ratio": training.warmup_ratio,
            "weight_decay": training.weight_decay,
            "optim": training.optim,
            "bf16": training.bf16,
            "fp16": training.fp16,
            "gradient_checkpointing": training.gradient_checkpointing,
            "beta": training.dpo_beta,
            "report_to": "none",
        }
        config = DPOConfig(**_filter_kwargs_for_init(DPOConfig, cfg_kwargs))

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "ref_model": ref_model,
            "args": config,
            "train_dataset": dataset,
            "processing_class": tokenizer,
        }
        allowed = _allowed_init_params(DPOTrainer) or set()
        if "processing_class" not in allowed and "tokenizer" in allowed:
            trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
        trainer_kwargs = _filter_kwargs_for_init(DPOTrainer, trainer_kwargs)
        trainer = DPOTrainer(**trainer_kwargs)
        trainer.train()

        adapter_out = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_out))
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        return str(adapter_out)

    def train(
        self,
        sft_examples: list[RedSFTExample],
        dpo_pairs: list[DpoTrainingPair],
        state: RedAdaptationState,
        round_index: int,
    ) -> RedAdaptationState:
        training = self.settings.training.red
        if len(sft_examples) < training.min_positive_examples:
            return state

        set_seed(self.settings.runtime.seed + round_index)
        sft_adapter = self._run_sft(sft_examples, state, round_index)
        final_adapter = sft_adapter
        if len(dpo_pairs) >= training.min_dpo_pairs:
            final_adapter = self._run_dpo(dpo_pairs, state, round_index, sft_adapter)

        state.active_adapter_path = final_adapter
        state.update_count += 1
        log_event(
            LOGGER,
            logging.INFO,
            "red_updated",
            "Completed Red SFT/DPO update",
            round_index=round_index,
            sft_examples=len(sft_examples),
            dpo_pairs=len(dpo_pairs),
            active_adapter_path=state.active_adapter_path,
        )
        return state
