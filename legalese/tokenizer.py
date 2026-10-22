"""Offset-preserving sentence and clause splitting.

Legal text is full of abbreviations (Sec., e.g., U.S.A.), section symbols,
and numbered lists, so this module keeps character offsets intact to make
tooltip anchoring reliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common abbreviations that should not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "vs", "etc", "no",
    "sec", "fig", "al", "inc", "corp", "ltd", "llc", "lp", "llp",
    "approx", "dept", "ed", "vol", "pp", "min", "max", "co",
    "u.s", "u.s.a", "e.g", "i.e", "a.m", "p.m",
}

_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z\.]*)$")


@dataclass(frozen=True)
class Span:
    """A slice of the original text with absolute offsets."""

    start: int
    end: int
    text: str


def _is_abbreviation(text: str, period_index: int) -> bool:
    m = _WORD_BEFORE.search(text[:period_index])
    if not m:
        return False
    raw = m.group(1)
    word = raw.lower().rstrip(".")
    if "." in raw:  # dotted forms: e.g. U.S. or e.g.
        return True
    return word in _ABBREVIATIONS or len(word) == 1


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentences, preserving character offsets.

    Boundaries: ``. ! ?`` followed by closing quotes/brackets and
    whitespace before a capital/digit/quote; blank lines always split.
    Leading/trailing whitespace is trimmed from each span.
    """
    spans: list[Span] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        boundary = None  # index at which the NEXT sentence starts
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in "\"')]}>":
                j += 1
            if j < n and text[j].isspace() and not _is_abbreviation(text, i):
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k >= n or text[k].isupper() or text[k].isdigit() or text[k] in "\"'(":
                    boundary = k
        elif ch == "\n":
            # blank line -> hard boundary
            m = re.match(r"\n\s*\n", text[i:])
            if m:
                boundary = i + m.end()
        if boundary is not None:
            seg_start, seg_end = start, i
            while seg_start < seg_end and text[seg_start].isspace():
                seg_start += 1
            while seg_end > seg_start and text[seg_end - 1].isspace():
                seg_end -= 1
            if seg_start < seg_end:
                spans.append(Span(seg_start, seg_end, text[seg_start:seg_end]))
            start = boundary
            i = boundary
        else:
            i += 1
    seg_start, seg_end = start, n
    while seg_start < seg_end and text[seg_start].isspace():
        seg_start += 1
    while seg_end > seg_start and text[seg_end - 1].isspace():
        seg_end -= 1
    if seg_start < seg_end:
        spans.append(Span(seg_start, seg_end, text[seg_start:seg_end]))
    return spans


def count_words(text: str) -> int:
    """Count word tokens (letters/digits runs) in ``text``."""
    return len(re.findall(r"[A-Za-z0-9]+", text))
