"""Flesch-Kincaid readability scoring.

Used to show the "before vs after" reading level of a document once
legalese has been replaced with plain English.
"""

from __future__ import annotations

import re

from .tokenizer import count_words, split_sentences

_VOWEL_GROUP = re.compile(r"[aeiouy]+")
_SENTENCE_END = re.compile(r"[.!?]+")


def count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip("'\"")
    if not word:
        return 0
    if word.isdigit():
        return 1
    groups = _VOWEL_GROUP.findall(word)
    count = len(groups)
    # silent trailing 'e' (e.g. "make" -> 1 syllable, not 2)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def _sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    spans = split_sentences(text)
    if spans:
        return [s.text for s in spans]
    return [text]


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid Grade Level (US school grade)."""
    sentences = _sentences(text)
    words = count_words(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables(w) for w in re.findall(r"[A-Za-z]+", text))
    words_per_sentence = words / len(sentences)
    syllables_per_word = syllables / words
    score = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return round(max(0.0, score), 1)


def flesch_reading_ease(text: str) -> float:
    """Return the Flesch Reading Ease score (higher = easier)."""
    sentences = _sentences(text)
    words = count_words(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables(w) for w in re.findall(r"[A-Za-z]+", text))
    score = 206.835 - 1.015 * (words / len(sentences)) - 84.6 * (syllables / words)
    return round(max(0.0, min(100.0, score)), 1)
