"""Runs both LLM clients against a live local Ollama server. Skipped when it is not running."""

from __future__ import annotations

import pytest

from ragbot.config import Settings
from ragbot.generation.llm import OllamaClient, OpenAICompatibleClient
from ragbot.generation.prompts import REFUSAL_TEXT, SYSTEM_PROMPT

pytestmark = pytest.mark.integration

CONTEXT = "[1] Manual, page 4\nThe relief valve on the Model K pump opens at 150 psi."


def _messages(question: str) -> list[dict[str, str]]:
    user = f"Sources:\n\n{CONTEXT}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@pytest.fixture(params=[OllamaClient, OpenAICompatibleClient], ids=["ollama", "openai_compatible"])
def client(request):
    instance = request.param(Settings(llm_max_tokens=60))
    if not instance.is_available():
        pytest.skip("local LLM is not running")
    return instance


def test_streams_grounded_answer_and_usage(client):
    events = list(client.stream(_messages("At what pressure does the relief valve open?")))
    text = "".join(event.text for event in events)
    assert "150" in text
    assert "[1]" in text
    final = events[-1]
    assert final.finish_reason == "stop"
    assert final.prompt_tokens > 0
    assert final.completion_tokens > 0


def test_refuses_when_sources_lack_the_answer(client):
    events = client.stream(_messages("What is the capital of Australia?"))
    assert REFUSAL_TEXT.rstrip(".") in "".join(event.text for event in events)
