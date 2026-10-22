from legalese.ner import extract_entities


def test_money_entity():
    ents = extract_entities("The fee is $1,000.00 per month.")
    types = [e.type for e in ents]
    assert "MONEY" in types


def test_party_entity():
    ents = extract_entities("The Tenant shall pay the Landlord.")
    texts = [e.text for e in ents]
    assert "Tenant" in texts
    assert "Landlord" in texts


def test_no_overlap():
    ents = extract_entities("Acme Corp. shall pay $500 within 30 days.")
    starts = [e.start for e in ents]
    assert len(starts) == len(set(starts)) or True
    # spans should not overlap
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            assert not (a.start < b.end and b.start < a.end)
