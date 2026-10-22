"""Contextual translator: jargon detection -> contextual rewrite.

Pipeline
--------
1. ``JargonDictionary.find_matches`` locates jargon phrases (longest match).
2. ``analyze_clause`` builds a :class:`ClauseContext` for each hit.
3. The best matching *sense* is chosen from the entry's ``senses`` list.
4. The sense's ``plain`` template is filled with clause-derived values.
5. Results are assembled: annotated spans, plain-English bullets, stats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .dictionary import JargonDictionary, JargonEntry, Sense, load_dictionary
from .ner import Entity, extract_entities
from .parser import ClauseContext, analyze_clause
from .readability import flesch_kincaid_grade
from .tokenizer import count_words, split_sentences

__all__ = [
    "TranslationResult",
    "JargonHit",
    "PlainBullet",
    "TranslationStats",
    "translate",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class JargonHit:
    """One detected jargon occurrence, with its contextual explanation."""

    entry_id: str
    term: str
    surface: str
    start: int
    end: int
    category: str
    risk: str
    short: str
    definition: str
    meaning: str
    plain: str  # the filled-in plain-English rewrite
    template: str  # the raw template (before filling)
    sense: str  # which sense was used ("default" or a sense label)
    clause: str = ""
    subject: str | None = None
    obj: str | None = None
    condition: str | None = None
    negation: bool = False
    passive: bool = False
    modal: str | None = None
    sentence: str = ""
    replacement: str = ""  # inline text swapped into the document
    is_modal: bool = False  # True for duty modals (shall/must) that map 1:1

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "term": self.term,
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "category": self.category,
            "risk": self.risk,
            "short": self.short,
            "definition": self.definition,
            "meaning": self.meaning,
            "plain": self.plain,
            "template": self.template,
            "sense": self.sense,
            "clause": self.clause,
            "subject": self.subject,
            "object": self.obj,
            "condition": self.condition,
            "negation": self.negation,
            "passive": self.passive,
            "modal": self.modal,
            "sentence": self.sentence,
            "replacement": self.replacement,
            "is_modal": self.is_modal,
        }


@dataclass
class PlainBullet:
    """A plain-English restatement of one clause that contained jargon."""

    text: str
    source: str
    hits: list[str] = field(default_factory=list)
    risk: str = "low"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "hits": list(self.hits),
            "risk": self.risk,
        }


@dataclass
class TranslationStats:
    """Aggregate jargon-density metrics."""

    words: int = 0
    sentences: int = 0
    jargon_hits: int = 0
    unique_terms: int = 0
    by_risk: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    reading_level_before: float = 0.0
    reading_level_after: float = 0.0

    @property
    def density(self) -> float:
        """Percentage of words that were jargon."""
        if not self.words:
            return 0.0
        return round(100.0 * self.jargon_hits / self.words, 2)

    def to_dict(self) -> dict:
        return {
            "words": self.words,
            "sentences": self.sentences,
            "jargon_hits": self.jargon_hits,
            "unique_terms": self.unique_terms,
            "density": self.density,
            "by_risk": dict(self.by_risk),
            "by_category": dict(self.by_category),
            "reading_level_before": self.reading_level_before,
            "reading_level_after": self.reading_level_after,
        }


@dataclass
class TranslationResult:
    """Everything the app needs to render a translation."""

    original: str
    rewritten: str
    hits: list[JargonHit] = field(default_factory=list)
    bullets: list[PlainBullet] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    stats: TranslationStats = field(default_factory=TranslationStats)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "hits": [h.to_dict() for h in self.hits],
            "bullets": [b.to_dict() for b in self.bullets],
            "entities": [e.to_dict() for e in self.entities],
            "stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Sense selection
# ---------------------------------------------------------------------------
def _sense_matches(sense: Sense, ctx: ClauseContext, clause: str) -> bool:
    """Does this sense's ``when`` condition hold for the current context?"""
    when = sense.when or {}
    for key, expected in when.items():
        if key == "negation":
            if ctx.negation != bool(expected):
                return False
        elif key == "passive":
            if ctx.passive != bool(expected):
                return False
        elif key == "subject":
            if not ctx.subject or expected.lower() not in ctx.subject.lower():
                return False
        elif key == "object":
            if not ctx.obj or expected.lower() not in ctx.obj.lower():
                return False
        elif key == "contains":
            if expected.lower() not in clause.lower():
                return False
        elif key == "not_contains":
            if expected.lower() in clause.lower():
                return False
        elif key == "modal":
            if not ctx.modal or expected.lower() not in ctx.modal.lower():
                return False
        elif key == "condition":
            if not ctx.condition:
                return False
        else:
            # unknown key -> ignore rather than silently failing to match
            continue
    return True


def select_sense(entry: JargonEntry, ctx: ClauseContext, clause: str) -> tuple[Sense | None, str]:
    """Pick the best sense for this context.

    Returns ``(sense, label)`` where ``sense`` is ``None`` for the entry's
    default sense. Later senses win when several match (most specific
    wins), matching the dictionary's ordering convention.
    """
    best: Sense | None = None
    best_label = "default"
    for idx, sense in enumerate(entry.senses):
        if _sense_matches(sense, ctx, clause):
            best = sense
            best_label = f"sense[{idx}]"
    return best, best_label


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------
_DEFAULT_SUBJ = "the party"

# Terms whose surface form maps 1:1 onto plain English inside a sentence.
# Everything else is a *gloss*: it explains a phrase and is shown as a
# plain-English bullet rather than spliced into the running text.
_INLINE_TERMS: dict[str, dict[str, str]] = {
    "shall": {"affirmative": "must", "negated": "must not"},
    "covenant": {"affirmative": "promises to", "negated": "does not promise to"},
    "represent_warrant": {"affirmative": "states and promises", "negated": "does not state"},
    "commercially_reasonable": {
        "affirmative": "will use commercially reasonable efforts to",
        "negated": "will not use commercially reasonable efforts to",
    },
    "indemnify": {"affirmative": "must pay for", "negated": "does not have to pay for"},
    "hold_harmless": {"affirmative": "promises not to blame", "negated": "makes no promise"},
    "waiver": {"affirmative": "gives up", "negated": "does not give up"},
    "release": {"affirmative": "drops all claims against", "negated": "keeps all claims against"},
    "consent": {"affirmative": "agrees", "negated": "does not agree"},
    "notwithstanding": {"affirmative": "even so,", "negated": "even so,"},
    "provided_that": {"affirmative": "but only if", "negated": "but only if"},
    "in_event_that": {"affirmative": "if", "negated": "if"},
    "except_that": {"affirmative": "except when", "negated": "except when"},
    "subject_to": {"affirmative": "subject to", "negated": "subject to"},
}


# Surface forms that are genuinely *verbs* and so can be swapped inline.
# A noun usage ("the Landlord's consent", "a waiver of claims") reads badly
# when replaced by a verb phrase, so those fall back to a bullet.
_INLINE_VERB_ONLY: dict[str, set[str]] = {
    "covenant": {"covenant", "covenants", "covenanted", "agree to", "agrees to",
                 "agreed to", "undertake", "undertakes"},
    "represent_warrant": {"represent and warrant", "represents and warrants",
                          "warrants", "warrant", "hereby represents"},
    "commercially_reasonable": {"commercially reasonable", "reasonable efforts"},
    "indemnify": {"indemnify", "indemnifying"},
    "hold_harmless": {"hold harmless", "holds harmless", "holding harmless"},
    "waiver": {"waive", "waives", "waived", "waiving"},
    "release": {"release", "releases", "releasing"},
    "consent": {"you consent"},
}

# Surface forms that are conditionals/connectors and are safe to swap
# wherever they appear.
_INLINE_ANY_SURFACE: set[str] = {
    "shall", "notwithstanding", "provided_that", "in_event_that",
    "except_that", "subject_to",
}


def _inline_replacement(entry: JargonEntry, ctx: ClauseContext, surface: str) -> str | None:
    """Map a term onto an inline plain-English phrase, if one exists.

    Returns ``None`` for terms that have no 1:1 inline equivalent — those
    are rendered as explanatory bullets instead.
    """
    table = _INLINE_TERMS.get(entry.id)
    if table is None:
        return None
    if entry.id in _INLINE_VERB_ONLY:
        allowed = _INLINE_VERB_ONLY[entry.id]
        if surface.lower() not in allowed:
            return None
    key = "negated" if ctx.negation else "affirmative"
    return table.get(key)


def _extend_negated_modal(text: str, start: int, end: int) -> int:
    """Extend a negated modal span to swallow the trailing "not".

    "shall not assign" should become "must not assign", not
    "must not not assign". So the "not" is pulled into the replaced span.
    """
    tail = text[end : end + 6]
    m = re.match(r"\s+not\b", tail, re.I)
    if m:
        return end + m.end()
    return end


def _clean_phrase(text: str | None) -> str | None:
    """Normalize an extracted phrase for use inside a sentence."""
    if not text:
        return None
    cleaned = text.strip().strip(",;:")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or None


def _render(template: str, ctx: ClauseContext, entry: JargonEntry, entities: list[Entity]) -> str:
    """Fill a rewrite template with clause-derived values."""
    subj = _clean_phrase(ctx.subject) or _DEFAULT_SUBJ
    obj = _clean_phrase(ctx.obj)
    action = _clean_phrase(ctx.action)

    # Lowercase the subject when it is a generic placeholder.
    if subj == _DEFAULT_SUBJ:
        subj_render = subj
    else:
        subj_render = subj[0].lower() + subj[1:] if subj[:1].isupper() and subj.split()[0].lower() not in {
            "the", "a", "an", "each", "any", "all", "both", "either", "neither",
            "no", "such", "this", "that", "these", "those", "landlord", "tenant",
            "company", "provider", "customer", "client", "borrower", "lender",
        } else subj

    # Entities that the template can reference directly.
    money = next((e for e in entities if e.type == "MONEY"), None)
    jurisdiction = next((e for e in entities if e.type == "JURISDICTION"), None)
    duration = next((e for e in entities if e.type == "DURATION"), None)

    values = {
        "SUBJ": subj_render,
        "OBJ": obj or subj_render,
        "ACTION": action or "do this",
        "TRIG": _clean_phrase(ctx.condition) or "a specific condition applies",
        "TERM": entry.short,
        "CLAUSE": _clean_phrase(ctx.subject) or _DEFAULT_SUBJ,
        "MONEY": money.text if money else "the stated amount",
        "JURISDICTION": jurisdiction.text if jurisdiction else "the agreed jurisdiction",
        "DURATION": duration.text if duration else "the agreed period",
    }

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)

    # Collapse any placeholders we did not know about.
    rendered = re.sub(r"\{[A-Z_]+\}", "the relevant term", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip()
    if rendered and not rendered.endswith((".", "!", "?", ":")):
        rendered += "."
    return rendered


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _sentence_for_offset(sentences, start: int) -> str:
    for span in sentences:
        if span.start <= start < span.end:
            return span.text
    return ""


def _replace_spans(text: str, hits: list[JargonHit]) -> str:
    """Build the rewritten text by swapping jargon spans for plain English.

    Only terms with a 1:1 inline equivalent (``shall`` -> ``must``) are
    spliced into the running text. Gloss-only terms are left in place so
    the document still reads as a document; their plain-English meaning is
    delivered in :attr:`TranslationResult.bullets`.

    Adjacent inline hits that would stack two verbs ("agrees to" + "must
    pay for") are merged: the first one wins and the second is dropped,
    so the sentence keeps a single verb.

    Replacements are applied left to right, tracking the output offset.
    """
    pieces: list[str] = []
    cursor = 0
    prev_end = -1
    for hit in hits:
        if not hit.replacement:
            continue
        # Skip a hit that immediately follows another inline hit: two verbs
        # in a row read as broken English ("promises to must pay for").
        if prev_end != -1 and hit.start <= prev_end + 1:
            continue
        # For negated modals ("shall not"), swallow the trailing "not" so we
        # don't emit "does not have to not assign".
        start = hit.start
        end = hit.end
        if hit.negation:
            end = _extend_negated_modal(text, start, end)
        pieces.append(text[cursor:start])
        pieces.append(hit.replacement)
        cursor = end
        prev_end = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _build_bullets(text: str, hits: list[JargonHit], sentences) -> list[PlainBullet]:
    """Produce one plain-English bullet per jargon term found.

    Each bullet explains a single term in the context it appeared in, and
    links back to the source sentence so the UI can highlight it.
    """
    bullets: list[PlainBullet] = []
    risk_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    for hit in hits:
        source = hit.sentence or _sentence_for_offset(sentences, hit.start)
        bullets.append(
            PlainBullet(
                text=hit.plain,
                source=source,
                hits=[hit.term],
                risk=hit.risk,
            )
        )
    return bullets


def translate(
    text: str,
    domain: str | None = None,
    dictionary: JargonDictionary | None = None,
    include_entities: bool = True,
) -> TranslationResult:
    """Translate ``text`` from legalese into plain English.

    Args:
        text: the document or clause to translate.
        domain: optional domain filter (``saas``, ``finance``, ...).
        dictionary: an explicit dictionary; the bundled one is used otherwise.
        include_entities: whether to run the NER pass and attach entities.

    Returns:
        A :class:`TranslationResult` with the rewritten text, per-hit
        explanations, plain-English bullets, and jargon-density stats.
    """
    if not text or not text.strip():
        return TranslationResult(original=text, rewritten=text)

    dictionary = dictionary or load_dictionary()
    matches = dictionary.find_matches(text, domain=domain)
    sentences = split_sentences(text)
    entities = extract_entities(text) if include_entities else []

    hits: list[JargonHit] = []
    for match in matches:
        entry = dictionary.get(match.entry_id)
        if entry is None:
            continue
        ctx = analyze_clause(text, match.start, match.end)
        clause = text[ctx.clause_start : ctx.clause_end]
        sense, sense_label = select_sense(entry, ctx, clause)

        template = entry.plain
        short = entry.short
        definition = entry.definition
        meaning = entry.meaning
        if sense is not None:
            template = sense.plain or template
            short = sense.short or short
            definition = sense.definition or definition
            meaning = sense.meaning or meaning

        plain_text = _render(template, ctx, entry, entities)
        inline = _inline_replacement(entry, ctx, match.surface)
        hits.append(
            JargonHit(
                entry_id=entry.id,
                term=entry.term,
                surface=match.surface,
                start=match.start,
                end=match.end,
                category=entry.category,
                risk=entry.risk,
                short=short,
                definition=definition,
                meaning=meaning,
                plain=plain_text,
                template=template,
                sense=sense_label,
                clause=clause.strip(),
                subject=ctx.subject,
                obj=ctx.obj,
                condition=ctx.condition,
                negation=ctx.negation,
                passive=ctx.passive,
                modal=ctx.modal,
                sentence=_sentence_for_offset(sentences, match.start),
                replacement=inline or "",
                is_modal=inline is not None,
            )
        )

    rewritten = _replace_spans(text, hits)
    bullets = _build_bullets(text, hits, sentences)

    by_risk: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for hit in hits:
        by_risk[hit.risk] = by_risk.get(hit.risk, 0) + 1
        by_category[hit.category] = by_category.get(hit.category, 0) + 1

    stats = TranslationStats(
        words=count_words(text),
        sentences=len(sentences),
        jargon_hits=len(hits),
        unique_terms=len({h.entry_id for h in hits}),
        by_risk=by_risk,
        by_category=by_category,
        reading_level_before=flesch_kincaid_grade(text),
        reading_level_after=flesch_kincaid_grade(rewritten),
    )

    return TranslationResult(
        original=text,
        rewritten=rewritten,
        hits=hits,
        bullets=bullets,
        entities=entities,
        stats=stats,
    )


