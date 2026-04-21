"""Dataset loading + prompt-masked collator for Judge LoRA training.

The critical anti-overfit ingredient: we compute loss *only* on the assistant
JSON reply. Everything up to and including the assistant header is set to -100.
With only ~630 training samples, letting the model memorize the shared system
prompt would burn capacity for nothing and inflate eval loss.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(records: list[dict[str, Any]]) -> Dataset:
    # Keep only what we need to minimize RAM.
    rows = [{"messages": r["messages"]} for r in records]
    return Dataset.from_list(rows)


def _render_and_mask(
    messages: list[dict[str, str]],
    tokenizer,
    max_seq_length: int,
    enable_thinking: bool,
) -> dict[str, list[int]]:
    """Tokenize the full conversation and mask all non-assistant-reply tokens.

    Strategy: render prompt-only (everything except the final assistant turn)
    with add_generation_prompt=True, then render the full conversation. The
    length of the prompt-only ids tells us where the completion starts.
    """
    assert messages and messages[-1]["role"] == "assistant", (
        "Last message must be the assistant target."
    )
    prompt_messages = messages[:-1]
    target_text = messages[-1]["content"]

    tmpl_kwargs: dict[str, Any] = {"tokenize": False}
    # enable_thinking is Qwen3-specific; ignore if tokenizer doesn't support it.
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **tmpl_kwargs,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, **tmpl_kwargs
        )

    # eos after the assistant content so the model learns to stop.
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + target_text + eos

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    # Truncate from the left of the prompt if we overflow — keep the full
    # completion intact so the loss signal is preserved. Judge targets are
    # short JSON (<200 tokens), so this is safe.
    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[overflow:]
        prompt_len = max(0, prompt_len - overflow)

    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


def preprocess(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int,
    enable_thinking: bool,
    num_proc: int = 2,
) -> Dataset:
    def _map(ex):
        return _render_and_mask(
            ex["messages"], tokenizer, max_seq_length, enable_thinking
        )

    return dataset.map(
        _map,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenize + mask prompt",
    )


@dataclass
class PromptMaskedCollator:
    """Right-pad input_ids/labels/attention_mask to the longest sample in batch."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [self.label_pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
