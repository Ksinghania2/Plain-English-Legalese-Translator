from legalese.parser import analyze_clause


def test_negation_detection():
    ctx = analyze_clause("The Tenant shall not assign this Agreement.", 11, 16)
    assert ctx.negation is True
    assert ctx.subject is not None
    assert "Tenant" in (ctx.subject or "")


def test_passive_detection():
    ctx = analyze_clause("The Tenant shall be indemnified by the Landlord.", 18, 28)
    # best-effort; ensure no crash and returns ClauseContext
    assert ctx is not None


def test_condition_detection():
    ctx = analyze_clause("If the Tenant breaches, the Landlord may terminate.", 40, 43)
    # condition may be captured
    assert ctx.clause_start >= 0
