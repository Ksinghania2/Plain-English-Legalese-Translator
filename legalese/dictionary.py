"""Semantic Jargon Dictionary: loading, validation, and trie-based matching.

The dictionary is a YAML file (``data/jargon_dictionary.yaml``) mapping
legalese surface forms to plain-English definitions and rewrite templates.
Matching is done over a normalized token trie so the longest phrase wins
("limitation of liability" beats "limit").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import yaml

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "jargon_dictionary.yaml"

VALID_RISKS = ("low", "medium", "high", "critical")
VALID_DOMAINS = ("general", "saas", "finance", "healthcare", "realestate")

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so surface variants line up."""
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Sense:
    """A context-specific override for a dictionary entry."""

    when: dict
    short: str | None = None
    definition: str | None = None
    plain: str | None = None
    meaning: str | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "Sense":
        return cls(
            when=raw.get("when") or {},
            short=raw.get("short"),
            definition=raw.get("definition"),
            plain=raw.get("plain"),
            meaning=raw.get("meaning"),
        )


@dataclass(frozen=True)
class JargonEntry:
    """One legalese term with its plain-English knowledge."""

    id: str
    term: str
    category: str
    domains: tuple[str, ...]
    variants: tuple[str, ...]
    risk: str
    short: str
    definition: str
    plain: str
    meaning: str
    case_sensitive: bool = False
    senses: tuple[Sense, ...] = ()

    @property
    def surface_forms(self) -> tuple[str, ...]:
        """All canonical forms (term + variants), deduplicated."""
        forms: list[str] = []
        for form in (self.term, *self.variants):
            if form and form not in forms:
                forms.append(form)
        return tuple(forms)

    def applies_to_domain(self, domain: str | None) -> bool:
        if domain is None or domain == "all":
            return True
        return domain in self.domains or "general" in self.domains

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "term": self.term,
            "category": self.category,
            "domains": list(self.domains),
            "variants": list(self.variants),
            "risk": self.risk,
            "short": self.short,
            "definition": self.definition,
            "plain": self.plain,
            "meaning": self.meaning,
            "case_sensitive": self.case_sensitive,
            "senses": [
                {
                    "when": s.when,
                    "short": s.short,
                    "definition": s.definition,
                    "plain": s.plain,
                    "meaning": s.meaning,
                }
                for s in self.senses
            ],
        }


class _TrieNode:
    __slots__ = ("children", "entry_ids")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.entry_ids: list[str] = []


class TrieMatch:
    """A phrase match found in the text (character offsets into original)."""

    __slots__ = ("entry_id", "start", "end", "surface", "num_words")

    def __init__(self, entry_id: str, start: int, end: int, surface: str, num_words: int):
        self.entry_id = entry_id
        self.start = start
        self.end = end
        self.surface = surface
        self.num_words = num_words


class JargonDictionary:
    """Indexed collection of :class:`JargonEntry` objects.

    Case-insensitive forms go into a word trie; case-sensitive forms
    (e.g. the defined term "Term") are matched separately with a
    boundary-aware regex so they only fire on the exact casing.
    """

    def __init__(self, entries: Sequence[JargonEntry], meta: dict | None = None):
        self.entries: dict[str, JargonEntry] = {}
        self.categories: dict[str, dict] = (meta or {}).get("categories", {})
        self.domains: dict[str, str] = (meta or {}).get("domains", {})
        self.version: str = (meta or {}).get("version", "?")
        self._root = _TrieNode()
        self._case_sensitive: list[tuple[re.Pattern, str]] = []
        for entry in entries:
            self._add(entry)

    # -- construction ---------------------------------------------------
    def _add(self, entry: JargonEntry) -> None:
        self.entries[entry.id] = entry
        if entry.case_sensitive:
            alts = "|".join(re.escape(f) for f in entry.surface_forms)
            pattern = re.compile(rf"(?<![A-Za-z0-9])({alts})(?![A-Za-z0-9])")
            self._case_sensitive.append((pattern, entry.id))
            return
        for form in entry.surface_forms:
            words = normalize(form).split()
            if not words:
                continue
            node = self._root
            for word in words:
                node = node.children.setdefault(word, _TrieNode())
            if entry.id not in node.entry_ids:
                node.entry_ids.append(entry.id)

    # -- lookup ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[JargonEntry]:
        return iter(self.entries.values())

    def get(self, entry_id: str) -> JargonEntry | None:
        return self.entries.get(entry_id)

    def filter(
        self,
        query: str | None = None,
        domain: str | None = None,
        category: str | None = None,
        risk: str | None = None,
    ) -> list[JargonEntry]:
        results = []
        q = (query or "").lower().strip()
        for entry in self:
            if domain and not entry.applies_to_domain(domain):
                continue
            if category and entry.category != category:
                continue
            if risk and entry.risk != risk:
                continue
            if q:
                haystack = " ".join([entry.term, entry.short, entry.definition, *entry.variants]).lower()
                if q not in haystack:
                    continue
            results.append(entry)
        # sort: risk weight desc, then alphabetically
        weight = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        results.sort(key=lambda e: (-weight.get(e.risk, 0), e.term.lower()))
        return results

    # -- matching ---------------------------------------------------------
    def find_matches(self, text: str, domain: str | None = None) -> list[TrieMatch]:
        """Find all jargon phrases in ``text`` (longest match wins).

        Returns non-overlapping matches sorted by position. When a
        case-sensitive hit and a case-insensitive hit collide, the longer
        one wins; ties go to the earliest start.
        """
        tokens: list[tuple[str, int, int]] = []  # (normalized word, start, end)
        for m in _WORD_RE.finditer(text):
            tokens.append((m.group().lower(), m.start(), m.end()))

        raw: list[TrieMatch] = []
        # 1) trie walk (case-insensitive, longest match from each token)
        i = 0
        while i < len(tokens):
            node = self._root
            best: TrieMatch | None = None
            j = i
            while j < len(tokens):
                node = node.children.get(tokens[j][0])
                if node is None:
                    break
                if node.entry_ids:
                    entry_id = self._select_entry(node.entry_ids, domain)
                    if entry_id:
                        best = TrieMatch(
                            entry_id=entry_id,
                            start=tokens[i][1],
                            end=tokens[j][2],
                            surface=text[tokens[i][1] : tokens[j][2]],
                            num_words=j - i + 1,
                        )
                j += 1
            if best:
                raw.append(best)
                i += best.num_words  # skip past the matched phrase
            else:
                i += 1

        # 2) case-sensitive regex pass (defined terms like "Term")
        taken = [(m.start, m.end) for m in raw]
        for pattern, entry_id in self._case_sensitive:
            if domain and not self.entries[entry_id].applies_to_domain(domain):
                continue
            for m in pattern.finditer(text):
                if any(s < m.end() and m.start() < e for s, e in taken):
                    continue
                raw.append(
                    TrieMatch(entry_id, m.start(), m.end(), m.group(), m.group().count(" ") + 1)
                )
                taken.append((m.start(), m.end()))

        raw.sort(key=lambda m: (m.start, -m.num_words))
        # 3) resolve overlaps: greedily keep earliest-longest
        result: list[TrieMatch] = []
        last_end = -1
        for m in raw:
            if m.start >= last_end:
                result.append(m)
                last_end = m.end
        return result

    def _select_entry(self, candidate_ids: Sequence[str], domain: str | None) -> str | None:
        """Pick the entry for a trie node given the active domain filter.

        Prefers higher-risk entries when several share a surface form.
        """
        weight = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        best_id = None
        best_weight = -1
        for cid in candidate_ids:
            entry = self.entries[cid]
            if not entry.applies_to_domain(domain):
                continue
            w = weight.get(entry.risk, 0)
            if w > best_weight:
                best_id, best_weight = cid, w
        return best_id


@lru_cache(maxsize=4)
def load_dictionary(path_str: str | None = None) -> JargonDictionary:
    """Load and validate the dictionary YAML. Cached per path."""
    path = Path(path_str) if path_str else DATA_PATH
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _from_raw(raw)


def _from_raw(raw: dict) -> JargonDictionary:
    meta = {
        "version": raw.get("version", "?"),
        "categories": raw.get("categories", {}),
        "domains": raw.get("domains", {}),
    }
    entries: list[JargonEntry] = []
    seen_ids: set[str] = set()
    for item in raw.get("entries", []):
        missing = [
            f
            for f in ("id", "term", "category", "risk", "short", "definition", "plain", "meaning")
            if not item.get(f)
        ]
        if missing:
            raise ValueError(
                f"dictionary entry {item.get('id', '<no id>')} missing fields: {missing}"
            )
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate dictionary id: {item['id']}")
        if item["risk"] not in VALID_RISKS:
            raise ValueError(f"entry {item['id']}: invalid risk {item['risk']!r}")
        if item["category"] not in meta["categories"]:
            raise ValueError(f"entry {item['id']}: unknown category {item['category']!r}")
        seen_ids.add(item["id"])
        entries.append(
            JargonEntry(
                id=item["id"],
                term=item["term"],
                category=item["category"],
                domains=tuple(item.get("domains", ["general"])),
                variants=tuple(item.get("variants", [])),
                risk=item["risk"],
                short=item["short"],
                definition=item["definition"],
                plain=item["plain"],
                meaning=item["meaning"],
                case_sensitive=bool(item.get("case_sensitive", False)),
                senses=tuple(Sense.from_dict(s) for s in item.get("senses", [])),
            )
        )
    return JargonDictionary(entries, meta)


