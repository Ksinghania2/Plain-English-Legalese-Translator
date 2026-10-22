from legalese.translator import translate

TEXT = (
    "The Tenant shall not assign this Agreement without the Landlord's consent. "
    "Each party agrees to indemnify the other party for any breach. "
    "The Landlord shall use commercially reasonable efforts to maintain the premises."
)

r = translate(TEXT)
print("REWRITTEN:", r.rewritten)
print()
for b in r.bullets:
    print("BULLET:", b.text)
print()
for h in r.hits:
    print(
        "HIT", h.entry_id, repr(h.surface),
        "neg=", h.negation,
        "repl=", repr(h.replacement),
        "modal=", h.modal,
        "subj=", h.subject,
    )
print()
print("STATS:", r.stats.to_dict())
