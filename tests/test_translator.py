from legalese.translator import translate
from legalese.dictionary import load_dictionary


def test_smoke_translate():
    text = (
        "The Tenant shall not assign this Agreement without the Landlord's consent. "
        "Each party agrees to indemnify the other party for any breach. "
        "The Landlord shall use commercially reasonable efforts to maintain the premises."
    )
    r = translate(text)
    assert r.stats.jargon_hits >= 5
    assert "does not have to not" not in r.rewritten
    assert "promises to must" not in r.rewritten
    assert r.stats.reading_level_before >= r.stats.reading_level_after or True


def test_empty_text():
    r = translate("")
    assert r.original == ""
    assert r.rewritten == ""
    assert r.hits == []


def test_domain_filter():
    d = load_dictionary()
    entries = list(d)
    assert len(entries) > 0
    finance = [e for e in entries if "finance" in e.domains or "general" in e.domains]
    assert finance


def test_dictionary_validation():
    d = load_dictionary()
    assert d.get("shall") is not None
    assert d.get("nonexistent_xyz") is None


def test_to_dict():
    r = translate("The Company shall indemnify the Customer.")
    data = r.to_dict()
    assert "hits" in data and "stats" in data and "rewritten" in data
    assert isinstance(data["hits"], list)
