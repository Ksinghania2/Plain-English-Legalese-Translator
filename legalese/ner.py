"""Lightweight, rule-based NER tuned for legal/financial documents.

Extracts typed entities (money, percent, durations, dates, jurisdictions,
laws, section references, organizations, and party roles) with exact
character offsets. An optional spaCy backend can be layered on top by the
translator when the ``spacy`` extra is installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    type: str
    text: str
    start: int
    end: int
    value: str | None = None  # normalized value (e.g. raw money string)

    def to_dict(self) -> dict:
        d = {"type": self.type, "text": self.text, "start": self.start, "end": self.end}
        if self.value is not None:
            d["value"] = self.value
        return d


# Ordered by priority; longer/more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "SECTION_REF",
        re.compile(
            r"\b(?:Section|Article|Paragraph|Clause|Exhibit|Appendix|Schedule|Annex|Rule|Regulation)"
            r"\s+[A-Z0-9]+(?:\.\d+)*(?:\([a-z0-9]+\))*(?:\([a-z0-9]+\))*"
            r"|\u00a7+\s?\d+(?:\.\d+)*(?:\([a-z0-9]+\))*"
        ),
    ),
    (
        "LAW",
        re.compile(
            r"\b(?:GDPR|CCPA|CPRA|HIPAA|HITECH|CAN-SPAM|TCPA|FCRA|FDCPA|ADA|FMLA|SOX"
            r"|Sarbanes-Oxley|Sarbanes Oxley|PCI-DSS|PCI DSS|GLBA|COPPA|FERPA|DMCA"
            r"|Basel III|Dodd-Frank|Gramm-Leach-Bliley)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MONEY",
        re.compile(
            r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d{1,2})?(?:\s?(?:million|billion|thousand))?"
            r"|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|JPY|CAD)(?:\s?(?:million|billion|thousand))?"
            r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred)"
            r"\s+(?:million|billion|thousand)?\s*(?:dollars|euros|pounds))",
            re.IGNORECASE,
        ),
    ),
    (
        "PERCENT",
        re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s?cent)\b", re.IGNORECASE),
    ),
    (
        "DURATION",
        re.compile(
            r"\b(?:(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|twelve"
            r"|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
            r"\s*(?:\(\s*\d[\d,]*\s*\)\s*)?(?:calendar\s+|business\s+)?"
            r"(?:minute|hour|day|week|month|quarter|year)s?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}"
            r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
            r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        ),
    ),
    (
        "JURISDICTION",
        re.compile(
            r"\b(?:State|Commonwealth|County|City) of [A-Z][\w]+(?:\s[A-Z][\w]+)?"
            r"|\blaws of (?:the )?(?:State of )?[A-Z][\w]+(?:\s[A-Z][\w]+)?"
            r"|\bcourts? (?:located )?in [A-Z][\w]+(?:\s[A-Z][\w]+)?"
            r"|\bgoverned by (?:the )?laws of (?:the )?(?:State of )?[A-Z][\w]+(?:\s[A-Z][\w]+)?"
        ),
    ),
    (
        "ORG",
        re.compile(
            r"\b[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,4}\s+"
            r"(?:Inc|Corp|Corporation|Co|Company|Ltd|LLC|L\.L\.C|LP|LLP|plc|GmbH|Holdings|Group)\.?\b"
        ),
    ),
    (
        "PARTY",
        re.compile(
            r"\b(?:Company|Provider|Customer|Client|Tenant|Landlord|Borrower|Lender"
            r"|Disclosing Party|Receiving Party|Licensee|Licensor|Contractor"
            r"|Subcontractor|Consultant|Employee|Employer|Supplier|Vendor)\b"
        ),
    ),
    (
        "PERSON",
        re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][\w]+(?:\s[A-Z][\w]+)?"),
    ),
]

_PRIORITY = {
    "SECTION_REF": 0, "LAW": 1, "MONEY": 2, "PERCENT": 3, "DURATION": 4,
    "DATE": 5, "JURISDICTION": 6, "ORG": 7, "PARTY": 8, "PERSON": 9,
}


def extract_entities(text: str) -> list[Entity]:
    """Run all rule patterns over ``text`` and return non-overlapping entities.

    Longer spans win on collision; priority breaks remaining ties.
    Spans are sorted by start offset.
    """
    candidates: list[Entity] = []
    for etype, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            surface = m.group().strip()
            if not surface:
                continue
            # trim trailing whitespace offsets
            start = m.start() + m.group().index(surface)
            end = start + len(surface)
            value = None
            if etype == "MONEY":
                value = re.sub(r"\s+", "", surface)
            candidates.append(Entity(etype, surface, start, end, value))

    # sort by start, longer first; then greedily resolve overlaps
    candidates.sort(key=lambda e: (e.start, -(e.end - e.start), _PRIORITY.get(e.type, 99)))
    result: list[Entity] = []
    last_end = -1
    for ent in candidates:
        if ent.start >= last_end:
            result.append(ent)
            last_end = ent.end
    return result


def entities_by_type(entities: list[Entity]) -> dict[str, list[Entity]]:
    """Group entities by their type label."""
    grouped: dict[str, list[Entity]] = {}
    for ent in entities:
        grouped.setdefault(ent.type, []).append(ent)
    return grouped

