from __future__ import annotations

from ragbot.generation.focus import focus_text

FILLER = " ".join(f"Unrelated sentence number {i} talks about gardening tools." for i in range(40))
FACT = "The relief valve on the pump opens at 150 psi."


def test_short_text_is_unchanged():
    assert focus_text("Short text.", "anything", 700) == "Short text."


def test_finds_the_relevant_sentence_anywhere_in_the_passage():
    text = f"{FILLER} {FACT} {FILLER}"
    excerpt = focus_text(text, "At what pressure does the relief valve open?", 400)
    assert FACT in excerpt
    assert len(excerpt) <= 400


def test_keeps_identifiers_with_punctuation():
    text = f"{FILLER} SI-07(13) was moved to CM-07(07). {FILLER}"
    assert "CM-07(07)" in focus_text(text, "What happened to SI-07(13)?", 400)


def test_is_deterministic():
    text = f"{FILLER} {FACT} {FILLER}"
    assert focus_text(text, "relief valve", 400) == focus_text(text, "relief valve", 400)


def test_no_matching_terms_still_returns_bounded_text():
    assert len(focus_text(FILLER, "what is it", 300)) <= 300
