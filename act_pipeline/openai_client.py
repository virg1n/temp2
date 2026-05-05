from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None = None


class OpenAIChatClient:
    """Small OpenAI-compatible client for vLLM chat completions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to call the vLLM endpoint.") from exc

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, request: ChatRequest) -> str:
        kwargs = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""
