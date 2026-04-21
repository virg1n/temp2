"""Train a LoRA adapter on Qwen3-32B for the Socratic-hint judge task.

Launch on 4x48GB with:
    accelerate launch --num_processes 4 --num_machines 1 \
        Judge-lora/train.py --config Judge-lora/configs/qwen3_32b_judge.yaml

Design notes (anti-overfit with only 630 train samples):
  * Loss computed only on the assistant JSON (prompt tokens masked to -100).
  * Small effective batch (32) + cosine LR with warmup + weight decay.
  * Early stopping on val loss with load_best_model_at_end.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from data import (  # noqa: E402
    PromptMaskedCollator,
    build_hf_dataset,
    load_jsonl,
    preprocess,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_training_args(tcfg: dict) -> TrainingArguments:
    # Build kwargs and drop any not supported by the installed transformers.
    ta_kwargs = dict(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=tcfg.get("optim", "paged_adamw_8bit"),
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg.get("eval_steps"),
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg.get("save_steps"),
        save_total_limit=tcfg.get("save_total_limit", 3),
        load_best_model_at_end=tcfg.get("load_best_model_at_end", True),
        metric_for_best_model=tcfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=tcfg.get("greater_is_better", False),
        report_to=tcfg.get("report_to", "none"),
        seed=tcfg.get("seed", 42),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        group_by_length=tcfg.get("group_by_length", True),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    if "warmup_steps" in tcfg:
        ta_kwargs["warmup_steps"] = tcfg["warmup_steps"]
    elif "warmup_ratio" in tcfg:
        ta_kwargs["warmup_ratio"] = tcfg["warmup_ratio"]
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in ta_kwargs if k not in supported]
    for k in dropped:
        ta_kwargs.pop(k)
    if dropped and int(os.environ.get("RANK", 0)) == 0:
        print(f"[warn] TrainingArguments does not support: {dropped} (dropped)")
    return TrainingArguments(**ta_kwargs)


def build_model_and_tokenizer(mcfg: dict):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map[mcfg.get("bnb_4bit_compute_dtype", "bfloat16")]
    model_dtype = compute_dtype

    bnb_cfg = None
    if mcfg.get("load_in_4bit", False):
        quant_storage_dtype = dtype_map[
            mcfg.get(
                "bnb_4bit_quant_storage_dtype",
                mcfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            )
        ]
        model_dtype = quant_storage_dtype
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=mcfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )
    elif mcfg.get("load_in_8bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=mcfg.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["name_or_path"],
        trust_remote_code=mcfg.get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # DDP: each process gets one full quantized replica on its local GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    attn_impl = mcfg.get("attn_implementation", "sdpa")
    model_kwargs = dict(
        quantization_config=bnb_cfg,
        trust_remote_code=mcfg.get("trust_remote_code", True),
        dtype=model_dtype,
        attn_implementation=attn_impl,
    )
    model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["name_or_path"],
            **model_kwargs,
        )
    except (ImportError, ValueError) as e:
        if attn_impl == "flash_attention_2":
            print(f"[warn] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                mcfg["name_or_path"],
                **model_kwargs,
            )
        else:
            raise

    model.config.use_cache = False
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    lcfg = cfg["lora"]

    set_seed(tcfg.get("seed", 42))

    # ---- Model + tokenizer ----
    model, tokenizer = build_model_and_tokenizer(mcfg)

    if mcfg.get("load_in_4bit", False) or mcfg.get("load_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        )
    elif tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
        target_modules=lcfg["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()

    # ---- Data ----
    train_records = load_jsonl(dcfg["train_file"])
    val_records = load_jsonl(dcfg["val_file"])
    train_ds = preprocess(
        build_hf_dataset(train_records),
        tokenizer,
        dcfg["max_seq_length"],
        dcfg.get("enable_thinking", False),
    )
    val_ds = preprocess(
        build_hf_dataset(val_records),
        tokenizer,
        dcfg["max_seq_length"],
        dcfg.get("enable_thinking", False),
    )

    collator = PromptMaskedCollator(pad_token_id=tokenizer.pad_token_id)
    targs = build_training_args(tcfg)

    callbacks = []
    patience = tcfg.get("early_stopping_patience", 0)
    if patience and patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    # transformers >=4.46 renamed `tokenizer` -> `processing_class`.
    trainer_sig = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Save the adapter + tokenizer on rank 0.
    if trainer.is_world_process_zero():
        out = Path(tcfg["output_dir"]) / "final"
        trainer.model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"[ok] Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
