from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(
    model_name: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    load_in_4bit: bool,
    local_rank: int,
):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


@contextlib.contextmanager
def disabled_adapter(model):
    target = getattr(model, "module", model)
    if hasattr(target, "disable_adapter"):
        with target.disable_adapter():
            yield
    else:
        yield


def completion_logps(model, tokenizer, prompts: list[str], completions: list[str], max_length: int) -> torch.Tensor:
    eos = tokenizer.eos_token or ""
    full_ids: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    lengths: list[int] = []

    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        text = prompt + completion + eos
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
        if len(ids) < 2:
            ids = ids + [tokenizer.eos_token_id]
        token_positions = torch.arange(1, len(ids), dtype=torch.long)
        mask = token_positions >= len(prompt_ids)
        if not bool(mask.any()):
            mask[-1] = True
        full_ids.append(torch.tensor(ids, dtype=torch.long))
        masks.append(mask)
        lengths.append(len(ids))

    padded = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention = torch.zeros_like(padded, dtype=torch.long)
    for idx, length in enumerate(lengths):
        attention[idx, :length] = 1
    max_shift = padded.shape[1] - 1
    padded_masks = torch.zeros((len(masks), max_shift), dtype=torch.bool)
    for idx, mask in enumerate(masks):
        padded_masks[idx, : len(mask)] = mask[:max_shift]

    device = next(model.parameters()).device
    padded = padded.to(device)
    attention = attention.to(device)
    padded_masks = padded_masks.to(device)

    outputs = model(input_ids=padded, attention_mask=attention)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = padded[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * padded_masks
    return token_logps.sum(dim=-1)


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = []
        for message in messages:
            rendered.append(f"{message['role'].upper()}:\n{message['content']}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)


def trainable_parameters(model) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))
