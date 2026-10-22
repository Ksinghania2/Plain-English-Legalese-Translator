"""Legalese Translator — a jargon-to-plain-English translator for
regulatory and legal documents.

Pipeline:  text -> jargon matcher (semantic dictionary trie)
                 -> clause parser (obligation / negation / passive voice)
                 -> contextual replacer (sense selection + template filling)
                 -> annotated spans + plain-English bullets + stats
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dictionary import JargonDictionary, JargonEntry, Sense, load_dictionary
from .translator import translate

__all__ = [
    "__version__",
    "JargonDictionary",
    "JargonEntry",
    "Sense",
    "load_dictionary",
    "translate",
]
