from __future__ import annotations

from typing import ClassVar

import pytest

from ragbot import models
from ragbot.config import Settings


class FlakyEncoder:
    """Fails on the GPU the way a lost graphics device does, and works on the CPU."""

    loads: ClassVar[list[list[str]]] = []

    def __init__(self, name, cache_dir, threads, providers) -> None:
        self.providers = providers
        FlakyEncoder.loads.append(providers)

    def rerank(self, query, passages, batch_size):
        if "DmlExecutionProvider" in self.providers:
            raise RuntimeError("887A0006 The GPU will not respond to more commands")
        return [1.0 for _ in passages]


@pytest.fixture
def flaky(monkeypatch):
    FlakyEncoder.loads = []
    monkeypatch.setattr(models, "TextCrossEncoder", FlakyEncoder)
    monkeypatch.setattr(models, "_available_gpu_providers", lambda: ["DmlExecutionProvider"])


def test_reranker_falls_back_to_cpu_when_gpu_fails(flaky):
    reranker = models.Reranker(Settings())
    assert reranker.score("question", ["a", "b"]) == [1.0, 1.0]
    assert FlakyEncoder.loads[-1] == ["CPUExecutionProvider"]


def test_reranker_stays_on_cpu_after_falling_back(flaky):
    reranker = models.Reranker(Settings())
    reranker.score("question", ["a"])
    loads = len(FlakyEncoder.loads)
    assert reranker.score("question", ["b"]) == [1.0]
    assert len(FlakyEncoder.loads) == loads


def test_gpu_can_be_turned_off(flaky):
    reranker = models.Reranker(Settings(rerank_use_gpu=False))
    assert reranker.score("question", ["a"]) == [1.0]
    assert FlakyEncoder.loads == [["CPUExecutionProvider"]]
