from __future__ import annotations

import re

from .config import ACTConfig


def clean_hint(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    return cleaned.strip()


class SocraticGenerator:
    def __init__(self, cfg: ACTConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for Socratic generation.") from exc

        self.cfg = cfg
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.models.socratic_base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.models.socratic_base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if cfg.models.socratic_adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install peft to load a Socratic adapter.") from exc
            self.model = PeftModel.from_pretrained(self.model, cfg.models.socratic_adapter)
        self.model.eval()

    def generate(self, prompt: str, n: int) -> list[str]:
        tokenizer = self.tokenizer
        model = self.model
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with self.torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=self.cfg.generation.socratic_temperature,
                top_p=self.cfg.generation.socratic_top_p,
                max_new_tokens=self.cfg.generation.socratic_max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        completions = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        return [clean_hint(text) for text in completions]
