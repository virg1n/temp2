from __future__ import annotations

import gc
import json
import os
import re
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from logging_utils import get_logger, log_event
from schemas import GenerationSettings, ModelSettings


LOGGER = get_logger(__name__)


def extract_json(text: str) -> Any | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    cleaned = strip_thinking_text(raw).replace("```json", "").replace("```", "").strip()
    for left, right in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(left), cleaned.rfind(right)
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                continue
    return None


def strip_thinking_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return cleaned.strip()


def _truncate_on_stop_strings(text: str, stop_strings: list[str]) -> str:
    result = text
    for stop_string in stop_strings:
        if stop_string and stop_string in result:
            result = result.split(stop_string, 1)[0]
    return result.strip()


def _torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.bfloat16)


def _get_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


class TransformersChatModel:
    def __init__(self, config: ModelSettings, *, adapter_override: str | None = None) -> None:
        self.config = config
        self.adapter_override = adapter_override
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    @property
    def adapter_path(self) -> str | None:
        return self.adapter_override if self.adapter_override is not None else self.config.adapter_path

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        if self.config.quantization.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("bitsandbytes/transformers quantization support is required for load_in_8bit") from exc
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=float(self.config.quantization.llm_int8_threshold),
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
            "torch_dtype": _torch_dtype_from_name(self.config.torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("peft is required to load LoRA adapters") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=False)

        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        log_event(
            LOGGER,
            20,
            "model_loaded",
            f"Loaded {self.config.name}",
            model_name=self.config.model_name_or_path,
            adapter_path=self.adapter_path,
        )

    def _apply_chat_template(self, messages: list[dict[str, str]], generation: GenerationSettings) -> str:
        assert self.tokenizer is not None
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(generation.enable_thinking),
        }
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        generation: GenerationSettings | None = None,
        num_return_sequences: int | None = None,
    ) -> list[str]:
        self.load()
        assert self.model is not None and self.tokenizer is not None

        gen = generation or self.config.generation
        prompt_text = self._apply_chat_template(messages, gen)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        )
        device = _get_model_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        requested = int(num_return_sequences or gen.num_return_sequences or 1)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": bool(gen.do_sample),
            "temperature": float(gen.temperature),
            "top_p": float(gen.top_p),
            "repetition_penalty": float(gen.repetition_penalty),
            "num_return_sequences": requested,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not gen.do_sample:
            generate_kwargs.pop("temperature", None)
            generate_kwargs.pop("top_p", None)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        return [
            _truncate_on_stop_strings(strip_thinking_text(text), list(gen.stop_strings))
            for text in decoded
        ]

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        generation: GenerationSettings | None = None,
    ) -> Any | None:
        outputs = self.generate(messages, generation=generation, num_return_sequences=1)
        return extract_json(outputs[0] if outputs else "")

    def unload(self, *, empty_cuda_cache: bool = True) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_event(LOGGER, 20, "model_unloaded", f"Unloaded {self.config.name}", model_name=self.config.name)
