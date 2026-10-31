"""Streaming LLM clients. Ollama runs locally; the OpenAI-compatible client covers hosted backends."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from ragbot.config import Settings


class LLMError(RuntimeError):
    """The LLM backend is unreachable or returned an error."""


@dataclass(slots=True)
class LLMEvent:
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    model: str

    def stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]: ...

    def warm_up(self) -> None: ...

    def is_available(self) -> bool: ...


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self._settings = settings
        self._http = httpx.Client(
            base_url=settings.llm_base_url, timeout=settings.llm_timeout_seconds
        )

    def stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,  # keep weights in VRAM: reloading would cost seconds per query
            "think": False,  # reasoning traces would use up the latency budget
            "options": {
                "temperature": 0,
                "seed": self._settings.llm_seed,
                "num_ctx": self._settings.llm_num_ctx,
                "num_predict": self._settings.llm_max_tokens,
            },
        }
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        yield LLMEvent(
                            prompt_tokens=data.get("prompt_eval_count"),
                            completion_tokens=data.get("eval_count"),
                            finish_reason=data.get("done_reason"),
                        )
                    elif text := data.get("message", {}).get("content"):
                        yield LLMEvent(text=text)
        except httpx.HTTPError as error:
            raise LLMError(f"Ollama request failed: {error}") from error

    def warm_up(self) -> None:
        for _ in self.stream([{"role": "user", "content": "Reply with: ok"}]):
            pass

    def is_available(self) -> bool:
        try:
            tags = self._http.get("/api/tags", timeout=3).raise_for_status().json()
        except httpx.HTTPError:
            return False
        return any(model.get("name") == self.model for model in tags.get("models", []))


class OpenAICompatibleClient:
    """Any /v1/chat/completions endpoint: Groq, vLLM, llama.cpp server, OpenAI, and so on."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self._settings = settings
        headers = (
            {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        )
        self._http = httpx.Client(
            base_url=settings.llm_base_url, headers=headers, timeout=settings.llm_timeout_seconds
        )

    def stream(self, messages: list[dict[str, str]]) -> Iterator[LLMEvent]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "seed": self._settings.llm_seed,
            "max_tokens": self._settings.llm_max_tokens,
        }
        finish_reason = None
        usage: dict = {}
        try:
            with self._http.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line.removeprefix("data:").strip()
                    if body == "[DONE]":
                        break
                    data = json.loads(body)
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if text := choice.get("delta", {}).get("content"):
                            yield LLMEvent(text=text)
        except httpx.HTTPError as error:
            raise LLMError(f"LLM request failed: {error}") from error
        yield LLMEvent(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )

    def warm_up(self) -> None:
        for _ in self.stream([{"role": "user", "content": "Reply with: ok"}]):
            pass

    def is_available(self) -> bool:
        try:
            self._http.get("/v1/models", timeout=3).raise_for_status()
        except httpx.HTTPError:
            return False
        return True


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAICompatibleClient(settings)
    return OllamaClient(settings)
