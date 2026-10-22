"""Dependency-lite clause analysis.

Parses the *function* of a clause around a jargon hit: who must do what
to whom, whether the obligation is negated or passive, and whether a
condition triggers it. This is the "contextual" half of the Contextual
Replacer — it lets the translator flip senses ("shall" vs "shall not",
active indemnity vs passive protection) and fill rewrite templates with
real subjects/objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Obligation / permission modals, longest forms first.
_MODAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bis\s+required\s+to\b", re.I), "obligation"),
    (re.compile(r"\bare\s+required\s+to\b", re.I), "obligation"),
    (re.compile(r"\bis\s+obligated\s+to\b", re.I), "obligation"),
    (re.compile(r"\bis\s+entitled\s+to\b", re.I), "permission"),
    (re.compile(r"\bagrees?\s+to\b", re.I), "obligation"),
    (re.compile(r"\bcovenants?\s+to\b", re.I), "obligation"),
    (re.compile(r"\bshall\s+not\b", re.I), "obligation_negated"),
    (re.compile(r"\bmay\s+not\b", re.I), "prohibition"),
    (re.compile(r"\bmust\s+not\b", re.I), "prohibition"),
    (re.compile(r"\bshall\b", re.I), "obligation"),
    (re.compile(r"\bmust\b", re.I), "obligation"),
    (re.compile(r"\bwill\b", re.I), "obligation"),
    (re.compile(r"\bmay\b", re.I), "permission"),
    (re.compile(r"\bcan\b", re.I), "permission"),
]

_NEGATION_RE = re.compile(
    r"\b(?:not|never|neither|nor|cannot|can't|won't|doesn't|don't|didn't"
    r"|isn't|aren't|wasn't|weren't|fails?\s+to|failed\s+to)\b",
    re.I,
)

# Words that end the scope of a negation: they start a new modifier that
# the earlier negation does not reach ("shall not assign X without consent"
# -> "without consent" is not part of the negated obligation).
_NEGATION_SCOPE_BREAKS = {
    "without", "except", "unless", "other", "than", "but", "notwithstanding",
    "subject", "provided", "whereas", "save", "absent",
}

_BE_FORMS = {"is", "are", "was", "were", "be", "been", "being"}
_BE_RE = re.compile(r"\b(is|are|was|were|be|been|being)\b", re.I)

_ADVERBS = {
    "not", "never", "hereby", "promptly", "reasonably", "fully", "duly",
    "automatically", "expressly", "hereinafter", "mutually", "collectively",
    "individually", "irrevocably", "unconditionally", "herein", "only",
}

_CONDITION_RE = re.compile(
    r"^\s*(?:If|In\s+the\s+event\s+(?:that|of)|Provided(?:,?\s*however,?)?\s*(?:that|if)?"
    r"|Unless|Whenever|Upon|Should\s+\S+\s+\S+|Whereas|Subject\s+to)\b[^,;]{0,160},\s*",
    re.I,
)

_LEADING_DETERMINERS = {
    "the", "a", "an", "each", "any", "no", "all", "both", "either", "neither",
    "such", "this", "that", "these", "those", "his", "her", "its", "their",
    "our", "your", "my", "every",
}

_LEAD_IN_WORDS = {
    "if", "when", "unless", "upon", "provided", "in", "notwithstanding",
    "should", "whereas", "subject",
}

_PREPOSITIONS = {
    "of", "in", "under", "by", "for", "with", "pursuant", "from", "against",
    "to", "at", "on", "between", "upon", "within", "through", "pursuant",
}

_STOPWORDS_AFTER_SUBJECT = _PREPOSITIONS | _BE_FORMS | {
    "shall", "must", "may", "will", "can", "agrees", "agrees", "covenants",
    "hereby", "not", "herein", "thereof",
}


def clause_bounds(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    """Expand from a jargon match to its enclosing clause.

    Left boundary: nearest ``; :`` or sentence punctuation before the
    match; right boundary: nearest ``; . ! ?`` or end of text.

    The left boundary defaults to the start of the text, so the clause
    keeps the subject that precedes the match ("The Tenant shall ...").
    """
    left = 0
    for i in range(match_start - 1, -1, -1):
        if text[i] in ";:.!?":
            left = i + 1
            break
        if text[i] == "\n":
            left = i + 1
            break
    right = len(text)
    for i in range(match_end, len(text)):
        if text[i] in ";.!?":
            right = i
            break
        if text[i] == "\n":
            right = i
            break
    return left, right


def _strip_determiners(words: list[str]) -> list[str]:
    while words and words[0].lower() in _LEADING_DETERMINERS:
        words = words[1:]
    return words


def _subject_window(text: str, upto: int) -> str:
    """Text just before position ``upto``, wide enough to contain the subject.

    The tight clause may begin *at* the jargon term ("shall"), which would
    hide the subject that precedes it ("The Tenant shall ..."). So we widen
    the window back to the previous sentence boundary, then keep only the
    text after the last comma or colon.
    """
    left = 0
    for i in range(upto - 1, -1, -1):
        if text[i] in ".;:!?\n":
            left = i + 1
            break
    segment = text[left:upto]
    # Keep only the final comma-delimited chunk: that is the subject slot.
    idx = segment.rfind(",")
    if idx != -1:
        segment = segment[idx + 1 :]
    return segment


def _extract_subject(clause: str, upto: int) -> str | None:
    """Grab the noun phrase before the modal/verb within ``clause[:upto]``."""
    segment = clause[:upto].strip()
    if not segment:
        return None
    words = segment.split()
    while words and words[0].lower().rstrip(",") in _LEAD_IN_WORDS:
        words = words[1:]
    kept: list[str] = []
    for w in words:
        bare = w.strip(",.;:()")
        if not bare:
            break
        if kept and bare.lower() in _STOPWORDS_AFTER_SUBJECT:
            break
        kept.append(bare)
    kept = _strip_determiners(kept)
    if not kept:
        return None
    return " ".join(kept[:6])


def _extract_object(clause: str, after: int) -> str | None:
    """Grab the noun phrase right after the jargon hit."""
    segment = clause[after:].lstrip()
    segment = re.split(r"[,;.:?!]", segment, maxsplit=1)[0]
    words = _strip_determiners(segment.split())
    # A leading "not" belongs to a negated modal, not to the object.
    while words and words[0].lower().strip("(),") == "not":
        words = words[1:]
    kept: list[str] = []
    for w in words:
        bare = w.strip("()\"'")
        if not bare:
            break
        if kept and bare.lower() in _PREPOSITIONS:
            break
        if bare.lower() in {"to", "that", "and", "or"} and kept:
            break
        kept.append(bare)
    if not kept:
        return None
    return " ".join(kept[:6])


def _extract_action(clause: str, after: int) -> str | None:
    """Grab the verb phrase after the jargon hit (for {ACTION} templates)."""
    segment = clause[after:].lstrip()
    segment = re.split(r"[,;:?!]", segment, maxsplit=1)[0].strip()
    words = segment.split()
    while words and words[0].lower().strip(",") == "not":
        words = words[1:]
    while words and words[0].lower().strip(",") in {"to", "the", "a", "an"}:
        words = words[1:]
    if not words:
        return None
    return " ".join(words[:8])


def _scoped_negation(clause: str, upto: int) -> bool:
    """Is the text before ``upto`` negated *at that point*?

    A negation only reaches forward until a scope-breaking word appears.
    "The Tenant shall not assign this Agreement without the Landlord's
    consent." - the "not" applies to the assigning, but the consent that
    follows is not itself negated.
    """
    segment = clause[:upto]
    last_negation = -1
    for m in _NEGATION_RE.finditer(segment):
        last_negation = m.start()
    if last_negation == -1:
        return False
    # A scope break after the last negation cancels it.
    tail = segment[last_negation:]
    for m in re.finditer(r"\b([A-Za-z]+)\b", tail):
        if m.start() == 0:
            continue
        if m.group(1).lower() in _NEGATION_SCOPE_BREAKS:
            return False
    return True


def _detect_condition(clause: str, match_offset: int) -> str | None:
    """If the clause opens with a condition containing the match, capture it."""
    m = _CONDITION_RE.match(clause)
    if not m:
        return None
    if m.end() >= match_offset - 2:
        return m.group().strip().rstrip(",").strip()
    return None


@dataclass
class ClauseContext:
    """Syntactic context for one jargon occurrence inside a clause."""

    clause_start: int  # absolute offsets into the original text
    clause_end: int
    subject: str | None = None
    obj: str | None = None
    action: str | None = None
    condition: str | None = None
    modal: str | None = None
    modal_kind: str | None = None
    negation: bool = False
    passive: bool = False
    entities: list = field(default_factory=list)

    @property
    def is_passive_negated(self) -> bool:
        return self.passive and self.negation


def analyze_clause(text: str, match_start: int, match_end: int) -> ClauseContext:
    """Build a :class:`ClauseContext` for the jargon span [start, end)."""
    c_start, c_end = clause_bounds(text, match_start, match_end)
    clause = text[c_start:c_end]
    offset = match_start - c_start  # match position within clause
    ctx = ClauseContext(clause_start=c_start, clause_end=c_end)

    # --- modal / obligation detection -------------------------------
    for pattern, kind in _MODAL_PATTERNS:
        m = pattern.search(clause, 0, offset)
        if not m:
            m = pattern.search(clause)
        if m:
            ctx.modal = m.group()
            ctx.modal_kind = kind
            # The subject may sit before the clause start (e.g. "The Tenant
            # shall ..."), so widen the window before extracting it.
            window = _subject_window(text, c_start + m.start())
            ctx.subject = _extract_subject(window, len(window))
            if not ctx.subject:
                ctx.subject = _extract_subject(clause, m.start())
            break

    # --- negation ---------------------------------------------------
    if _scoped_negation(clause, offset):
        ctx.negation = True
    if ctx.modal_kind in {"obligation_negated", "prohibition"}:
        ctx.negation = True
    # A negated modal ("shall not") only negates what follows it; a later
    # term in the same clause is not automatically negated.
    if ctx.modal_kind in {"obligation_negated", "prohibition"}:
        modal_neg_pos = clause.lower().find("not")
        if modal_neg_pos != -1 and modal_neg_pos < offset:
            ctx.negation = _scoped_negation(clause, offset)

    # --- passive voice ("shall be indemnified", "is waived") --------
    before = clause[:offset].rstrip()
    be_match = None
    for m in _BE_RE.finditer(before):
        be_match = m
    if be_match:
        gap_words = before[be_match.end():].split()
        if len(gap_words) <= 2 and all(
            w.lower().strip(",") in _ADVERBS or not w.isalpha() for w in gap_words
        ):
            ctx.passive = True

    # --- object / action --------------------------------------------
    ctx.obj = _extract_object(clause, offset + (match_end - match_start))
    ctx.action = _extract_action(clause, offset + (match_end - match_start))

    # --- condition ---------------------------------------------------
    ctx.condition = _detect_condition(clause, offset)

    return ctx

