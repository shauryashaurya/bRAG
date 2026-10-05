# PII boundary: the only place where raw personal data is turned into safe placeholder tokens.
# Rule from CLAUDE.md: PII must be tokenized before any model call. Everything downstream of
# this function (extraction, assessment, routing, database) only ever sees tokens.
# Known limits of this mock: tokens are fixed ([EMAIL_1], [NAME_1]) so two emails collide, names
# are only found after a label, and there is no reversible vault to restore the originals.
# A production version would issue unique tokens and keep the mapping in a protected store.
# Cert: CCAR-P safety, privacy and compliance by design; CCAR-F preparing context for the model.
from common.stubs import replace_with_token


# Returns the text with emails and names replaced by tokens.
def tokenize_pii(text):
    text = replace_with_token(text, "email", "[EMAIL_1]")
    text = replace_with_token(text, "name", "[NAME_1]")
    return text
