"""Merge a trained LoRA adapter into the base model (bf16) for serving.

Run on a single GPU host (needs ~64GB VRAM or use CPU offload with --device cpu):
    python Judge-lora/merge_lora.py \
        --base Qwen/Qwen3-32B \
        --adapter Judge-lora/outputs/qwen3-32b-judge-lora/final \
        --out   Judge-lora/outputs/qwen3-32b-judge-merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"[ok] Merged model saved to {out}")


if __name__ == "__main__":
    main()
