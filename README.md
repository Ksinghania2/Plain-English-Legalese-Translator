# Legalese Translator

> A "spellcheck for corporate legalese." It highlights dense regulatory/legal phrases
> and translates them into plain, transparent English — clause by clause, with context.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688) ![License](https://img.shields.io/badge/license-MIT-green)

## Why

Professionals in finance, healthcare, and real estate waste hours wading through dense
terms of service, compliance updates, and privacy policies. Generic summarizers flatten
the text and lose the binding clauses. **Legalese Translator** doesn't summarize — it
*strips away obfuscation* and rewrites specific clauses into plain English, while
preserving the document structure and anchoring every translation to its source sentence.

## How it works

Instead of training a massive LLM or doing naive chunk-and-summarize, the project uses a
**Semantic Jargon Dictionary + Contextual Replacer**:

1. **Semantic Jargon Dictionary** (`data/jargon_dictionary.yaml`) — 80+ regulatory/legal
   terms mapped to plain-English definitions, risk levels, and rewrite templates with
   placeholders (`{SUBJ}`, `{OBJ}`, `{ACTION}`, `{TRIG}`, …).
2. **Longest-match trie** (`legalese/dictionary.py`) — phrase matching so
   "limitation of liability" beats "limit".
3. **Rule-based NER** (`legalese/ner.py`) — money, percent, durations, dates,
   jurisdictions, organizations, party roles, laws, section references.
4. **Dependency-lite clause parser** (`legalese/parser.py`) — subject, object, action,
   modal, negation, passive voice, and triggering conditions.
5. **Contextual sense selection** (`legalese/translator.py`) — picks the right sense
   (e.g. "shall" vs "shall not", active indemnity vs passive protection) and fills
   templates with real subjects/objects.
6. **Readability scoring** (`legalese/readability.py`) — Flesch-Kincaid before/after.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
# optional: richer NER via spaCy
pip install -e ".[spacy]"
python -m spacy download en_core_web_sm
```

## Usage

### CLI

```bash
# Translate a file
legalese translate contract.txt

# From stdin
cat tos.txt | legalese translate --bullets --stats

# Full JSON
legalese translate contract.txt --domain finance --json

# Shorthand
legalese contract.txt
```

### Python

```python
from legalese.translator import translate

text = "The Tenant shall not assign this Agreement without the Landlord's consent."
result = translate(text)
print(result.rewritten)
for bullet in result.bullets:
    print("-", bullet.text)
print(result.stats.to_dict())
```

### Web UI / API

```bash
legalese serve --port 8000
# open http://127.0.0.1:8000
```

REST endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The Company shall indemnify the Customer.", "domain": "saas"}'
```

## Project layout

```
legalese-translator/
├── legalese/            # core engine
│   ├── dictionary.py    # YAML loader + trie matching
│   ├── ner.py           # entity extraction
│   ├── parser.py        # clause analysis
│   ├── readability.py   # Flesch-Kincaid
│   ├── tokenizer.py     # offset-preserving sentence splitting
│   ├── translator.py    # contextual replacer
│   └── cli.py           # command-line interface
├── app/                 # FastAPI web app + static UI
├── data/                # semantic jargon dictionary (YAML)
├── tests/               # pytest suite
├── smoke_test.py
├── pyproject.toml
└── requirements.txt
```

## Tests

```bash
pytest -q
```

## Domains

`general`, `saas`, `finance`, `healthcare`, `realestate`. Pass `--domain` / `domain`
to filter entries that don't apply.

## Risk levels

Each hit carries a `risk`: `low`, `medium`, `high`, `critical`, so the UI can color-code
the clauses that matter most.

## License

MIT — see [LICENSE](LICENSE).
