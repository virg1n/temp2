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
    render_chat_prompt,
    set_seed,
    trainable_parameters,
)


def build_examples(records: list[dict], tokenizer, max_candidates: int) -> list[dict]:
    examples: list[dict] = []
    for record in records:
        candidates = [
            candidate
            for candidate in record.get("candidates", [])
            if isinstance(candidate.get("completion"), str) and candidate.get("completion", "").strip()
        ]
        if len(candidates) < 2:
            continue
        candidates = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:max_candidates]
        examples.append(
            {
                "prompt": render_chat_prompt(tokenizer, record.get("messages", [])),
                "completions": [candidate["completion"] for candidate in candidates],
                "scores": [float(candidate.get("score", 0.0)) for candidate in candidates],
            }
        )
    return examples


def collate(batch: list[dict]) -> list[dict]:
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Red with listwise K-wise Preference Optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.train.red_kpo
    set_seed(cfg.run.seed)

    accelerator = Accelerator(gradient_accumulation_steps=train_cfg.gradient_accumulation_steps)
    tokenizer = load_tokenizer(cfg.models.red_model)
    records = read_jsonl(args.train_file, limit=args.limit)
    examples = build_examples(records, tokenizer, train_cfg.max_candidates_per_group)
    loader = DataLoader(examples, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)

    model = load_lora_model(
        cfg.models.red_model,
        lora_r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        load_in_4bit=train_cfg.load_in_4bit,
        local_rank=env_local_rank(),
    )
    if not train_cfg.load_in_4bit and torch.cuda.is_available():
        model.to(accelerator.device)

    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=train_cfg.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for epoch in range(train_cfg.epochs):
        model.train()
        for step, batch in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                losses: list[torch.Tensor] = []
                for group in batch:
                    prompt = group["prompt"]
                    completions = group["completions"]
                    scores = torch.tensor(group["scores"], dtype=torch.float32, device=accelerator.device)
                    prompts = [prompt] * len(completions)

                    policy_logps = completion_logps(model, tokenizer, prompts, completions, train_cfg.max_length)
                    with torch.no_grad(), disabled_adapter(model):
                        ref_logps = completion_logps(model, tokenizer, prompts, completions, train_cfg.max_length)

                    target = F.softmax(scores / train_cfg.reward_temperature, dim=0)
                    relative_logits = train_cfg.beta * (policy_logps - ref_logps)
                    log_policy = F.log_softmax(relative_logits, dim=0)
                    losses.append(-(target * log_policy).sum())

                loss = torch.stack(losses).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 5 == 0:
                print(f"[red-kpo] epoch={epoch + 1} step={step} loss={loss.detach().float().item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[red-kpo] saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
