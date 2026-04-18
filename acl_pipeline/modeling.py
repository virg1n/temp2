from __future__ import annotations

import gc
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import GenerationSettings, HardwareAllocation, LoRASettings, PipelineConfig
from .logging_utils import StructuredLogger


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _dtype_for_runtime() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config(mode: Optional[str]) -> Optional[BitsAndBytesConfig]:
    normalized = str(mode or "none").lower()
    if normalized in {"none", "null"}:
        return None
    if normalized in {"8", "8bit", "int8"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    if normalized in {"4", "4bit", "int4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype_for_runtime(),
        )
    raise ValueError(f"Unsupported quantization mode: {mode}")


def build_max_memory(hardware: HardwareAllocation) -> Optional[Dict[Any, str]]:
    if not hardware.gpu_ids:
        return None
    max_memory: Dict[Any, str] = {gpu_id: f"{hardware.per_gpu_memory_gib}GiB" for gpu_id in hardware.gpu_ids}
    max_memory["cpu"] = f"{hardware.cpu_offload_gib}GiB"
    return max_memory


def single_gpu_hardware(hardware: HardwareAllocation, gpu_id: int) -> HardwareAllocation:
    return replace(hardware, gpu_ids=[int(gpu_id)])


def render_chat_messages(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        for kwargs in (
            {"enable_thinking": enable_thinking},
            {},
        ):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
            except TypeError:
                continue

    rendered: List[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user").upper()
        content = str(msg.get("content") or "")
        rendered.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def load_tokenizer(model_name_or_path: str, tokenizer_name_or_path: Optional[str] = None) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def infer_input_device(model: Any) -> torch.device:
    if hasattr(model, "device") and model.device is not None and str(model.device) != "meta":
        return model.device

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attach_lora_adapter(model: Any, lora: LoRASettings) -> Any:
    if not lora.enabled:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("LoRA is enabled but `peft` is not installed.") from exc

    config = LoraConfig(
        r=int(lora.r),
        lora_alpha=int(lora.alpha),
        lora_dropout=float(lora.dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora.target_modules) or None,
    )
    return get_peft_model(model, config)


def save_model_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


@dataclass
class RoleSession:
    role_name: str
    model_name_or_path: str
    tokenizer: Any
    model: Any
    generation: GenerationSettings
    enable_thinking: bool
    logger: StructuredLogger
    adapter_path: Optional[str] = None

    def generate(
        self,
        messages_batch: Iterable[List[Dict[str, str]]],
        *,
        generation: Optional[GenerationSettings] = None,
    ) -> List[str]:
        effective = generation or self.generation
        prompts = [
            render_chat_messages(
                self.tokenizer,
                list(messages),
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        return _generate_texts(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            generation=effective,
            logger=self.logger,
            role_name=self.role_name,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        clear_cuda_memory()


def _generate_texts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    generation: GenerationSettings,
    logger: StructuredLogger,
    role_name: str,
) -> List[str]:
    if not prompts:
        return []

    input_device = infer_input_device(model)
    batch_size = max(1, int(generation.batch_size))
    outputs: List[str] = []
    start = 0

    while start < len(prompts):
        chunk = prompts[start : start + batch_size]
        try:
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            do_sample = bool(generation.do_sample and generation.temperature > 0)
            gen_kwargs: Dict[str, Any] = {
                **encoded,
                "max_new_tokens": int(generation.max_new_tokens),
                "do_sample": do_sample,
                "repetition_penalty": float(generation.repetition_penalty),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(generation.temperature)
                gen_kwargs["top_p"] = float(generation.top_p)
            result = model.generate(**gen_kwargs)

            attention_mask = encoded["attention_mask"]
            for row_index in range(result.size(0)):
                prompt_len = int(attention_mask[row_index].sum().item())
                new_tokens = result[row_index, prompt_len:]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
            start += len(chunk)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_cuda_memory()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "oom_generate_retry",
                    role=role_name,
                    next_batch_size=batch_size,
                    max_new_tokens=generation.max_new_tokens,
                )
                continue
            if generation.max_new_tokens > 64:
                generation = GenerationSettings(
                    batch_size=1,
                    max_new_tokens=max(64, generation.max_new_tokens // 2),
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    do_sample=generation.do_sample,
                    repetition_penalty=generation.repetition_penalty,
                )
                logger.warning(
                    "oom_generate_shrink_tokens",
                    role=role_name,
                    next_max_new_tokens=generation.max_new_tokens,
                )
                continue
            raise

    return outputs


def load_role_session(
    *,
    role_name: str,
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str],
    hardware: HardwareAllocation,
    generation: GenerationSettings,
    quantization: Optional[str],
    enable_thinking: bool,
    logger: StructuredLogger,
    adapter_path: Optional[str] = None,
    trainable: bool = False,
    gradient_checkpointing: bool = False,
) -> RoleSession:
    tokenizer = load_tokenizer(model_name_or_path, tokenizer_name_or_path)
    quant_cfg = build_quantization_config(quantization)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if hardware.gpu_ids else None,
        "max_memory": build_max_memory(hardware),
    }
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    else:
        kwargs["torch_dtype"] = _dtype_for_runtime() if torch.cuda.is_available() else torch.float32

    if gradient_checkpointing:
        kwargs["use_cache"] = False

    model = None
    load_attempts = [quantization]
    if quantization and str(quantization).lower() == "8bit":
        load_attempts.append("4bit")
    load_attempts.append(None)

    last_exc: Optional[BaseException] = None
    for mode in load_attempts:
        try:
            local_kwargs = dict(kwargs)
            local_kwargs["quantization_config"] = build_quantization_config(mode)
            if local_kwargs["quantization_config"] is None:
                local_kwargs.pop("quantization_config", None)
                local_kwargs["torch_dtype"] = _dtype_for_runtime() if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **local_kwargs)
            if adapter_path:
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Failed to load adapter from {adapter_path}") from exc

            if trainable and local_kwargs.get("quantization_config") is not None:
                try:
                    from peft import prepare_model_for_kbit_training

                    model = prepare_model_for_kbit_training(
                        model,
                        use_gradient_checkpointing=gradient_checkpointing,
                    )
                except Exception:
                    pass

            if trainable and gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            break
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            last_exc = exc
            clear_cuda_memory()
            logger.warning(
                "oom_model_load_retry",
                role=role_name,
                attempted_quantization=mode,
                next_attempt="fallback",
            )

    if model is None:
        raise RuntimeError(f"Unable to load {role_name} model after OOM fallbacks: {last_exc}")

    logger.event(
        "model_loaded",
        role=role_name,
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        quantization=quantization,
        gpu_ids=hardware.gpu_ids,
        persistent=hardware.persistent,
        trainable=trainable,
    )
    return RoleSession(
        role_name=role_name,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        model=model,
        generation=generation,
        enable_thinking=enable_thinking,
        logger=logger,
        adapter_path=adapter_path,
    )


class ModelPool:
    def __init__(self, config: PipelineConfig, logger: StructuredLogger) -> None:
        self.config = config
        self.logger = logger
        self._judge: Optional[RoleSession] = None
        self._socratic: Optional[RoleSession] = None
        self._socratic_source: Optional[str] = None
        self._socratic_adapter: Optional[str] = None

    def get_judge(self) -> RoleSession:
        if self._judge is None:
            self._judge = load_role_session(
                role_name="judge",
                model_name_or_path=self.config.judge.model_name_or_path,
                tokenizer_name_or_path=self.config.judge.tokenizer_name_or_path,
                hardware=self.config.judge.hardware,
                generation=self.config.judge.generation,
                quantization=self.config.judge.quantization,
                enable_thinking=self.config.judge.enable_thinking,
                logger=self.logger,
                adapter_path=self.config.judge.base_adapter_path,
                trainable=False,
            )
        return self._judge

    def get_socratic(self, *, model_source: Optional[str] = None, adapter_path: Optional[str] = None) -> RoleSession:
        source = model_source or self.config.socratic.model_name_or_path
        if (
            self._socratic is not None
            and self._socratic_source == source
            and self._socratic_adapter == adapter_path
            and self.config.socratic.hardware.persistent
        ):
            return self._socratic

        self.release_socratic()
        session = load_role_session(
            role_name="socratic",
            model_name_or_path=source,
            tokenizer_name_or_path=self.config.socratic.tokenizer_name_or_path,
            hardware=self.config.socratic.hardware,
            generation=self.config.socratic.generation,
            quantization=self.config.socratic.quantization,
            enable_thinking=False,
            logger=self.logger,
            adapter_path=adapter_path or self.config.socratic.base_adapter_path,
            trainable=False,
        )
        if self.config.socratic.hardware.persistent:
            self._socratic = session
            self._socratic_source = source
            self._socratic_adapter = adapter_path
        return session

    def load_socratic_trainable(self, *, model_source: Optional[str] = None, adapter_path: Optional[str] = None) -> RoleSession:
        source = model_source or self.config.socratic.model_name_or_path
        return load_role_session(
            role_name="socratic_train",
            model_name_or_path=source,
            tokenizer_name_or_path=self.config.socratic.tokenizer_name_or_path,
            hardware=self.config.socratic.hardware,
            generation=self.config.socratic.generation,
            quantization=self.config.socratic.quantization,
            enable_thinking=False,
            logger=self.logger,
            adapter_path=adapter_path or self.config.socratic.base_adapter_path,
            trainable=True,
            gradient_checkpointing=self.config.socratic.grpo.gradient_checkpointing,
        )

    def load_red_generation(self, *, adapter_path: Optional[str] = None, gpu_id: Optional[int] = None) -> RoleSession:
        hardware = self.config.red.hardware if gpu_id is None else single_gpu_hardware(self.config.red.hardware, gpu_id)
        return load_role_session(
            role_name="red_generation" if gpu_id is None else f"red_generation_{gpu_id}",
            model_name_or_path=self.config.red.model_name_or_path,
            tokenizer_name_or_path=self.config.red.tokenizer_name_or_path,
            hardware=hardware,
            generation=self.config.red.generation,
            quantization=self.config.red.generation_quantization,
            enable_thinking=self.config.red.enable_thinking,
            logger=self.logger,
            adapter_path=adapter_path or self.config.red.base_adapter_path,
            trainable=False,
        )

    def load_red_trainable(self, *, adapter_path: Optional[str] = None) -> RoleSession:
        return load_role_session(
            role_name="red_train",
            model_name_or_path=self.config.red.model_name_or_path,
            tokenizer_name_or_path=self.config.red.tokenizer_name_or_path,
            hardware=self.config.red.hardware,
            generation=self.config.red.generation,
            quantization=self.config.red.update_quantization,
            enable_thinking=self.config.red.enable_thinking,
            logger=self.logger,
            adapter_path=adapter_path or self.config.red.base_adapter_path,
            trainable=True,
            gradient_checkpointing=True,
        )

    def release_socratic(self) -> None:
        if self._socratic is not None:
            self._socratic.unload()
            self._socratic = None
            self._socratic_source = None
            self._socratic_adapter = None

    def close(self) -> None:
        if self._judge is not None:
            self._judge.unload()
            self._judge = None
        self.release_socratic()

    def debug_summary(self) -> str:
        payload = {
            "judge_loaded": self._judge is not None,
            "socratic_loaded": self._socratic is not None,
            "socratic_source": self._socratic_source,
            "socratic_adapter": self._socratic_adapter,
        }
        return json.dumps(payload, ensure_ascii=False)
