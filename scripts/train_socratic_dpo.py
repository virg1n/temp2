from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from act_pipeline.config import load_config
from act_pipeline.training_common import (
    completion_logps,
    disabled_adapter,
    env_local_rank,
    load_lora_model,
    load_tokenizer,
    read_jsonl,
    set_seed,
    trainable_parameters,
)


def collate(batch: list[dict]) -> dict[str, list[str]]:
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SocraticAI with DPO pairs from ACT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.train.socratic_dpo
    set_seed(cfg.run.seed)

    accelerator = Accelerator(gradient_accumulation_steps=train_cfg.gradient_accumulation_steps)
    tokenizer = load_tokenizer(cfg.models.socratic_base_model)
    model = load_lora_model(
        cfg.models.socratic_base_model,
        lora_r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        load_in_4bit=train_cfg.load_in_4bit,
        local_rank=env_local_rank(),
    )
    if not train_cfg.load_in_4bit and torch.cuda.is_available():
        model.to(accelerator.device)

    records = read_jsonl(args.train_file, limit=args.limit)
    loader = DataLoader(records, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=train_cfg.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for epoch in range(train_cfg.epochs):
        model.train()
        for step, batch in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                chosen_logps = completion_logps(model, tokenizer, batch["prompt"], batch["chosen"], train_cfg.max_length)
                rejected_logps = completion_logps(model, tokenizer, batch["prompt"], batch["rejected"], train_cfg.max_length)
                with torch.no_grad(), disabled_adapter(model):
                    ref_chosen = completion_logps(model, tokenizer, batch["prompt"], batch["chosen"], train_cfg.max_length)
                    ref_rejected = completion_logps(model, tokenizer, batch["prompt"], batch["rejected"], train_cfg.max_length)

                logits = (chosen_logps - rejected_logps) - (ref_chosen - ref_rejected)
                loss = -F.logsigmoid(train_cfg.beta * logits).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 10 == 0:
                print(f"[socratic-dpo] epoch={epoch + 1} step={step} loss={loss.detach().float().item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[socratic-dpo] saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
