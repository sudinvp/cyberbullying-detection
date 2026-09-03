"""
Small curated lexicon of common insult/toxic terms.

Why this exists: the training set (cyberbullying_tweets.csv) is built from
real tweets about *identity-based* harassment (age, gender, ethnicity,
religion) — full sentences with context. It contains very few short, generic
insults like "idiot" or "you are bad" in isolation, so the neural model
never learned strong weights for plain profanity/name-calling on its own.

This lexicon is a pragmatic hybrid-system fix: a rule-based floor that
catches obvious insults the model under-scores, blended with the model's
own probability rather than replacing it. This is a standard technique in
production content-moderation systems (ML + rules), not a replacement for
a better/bigger training set.
"""

INSULT_TERMS = {
    "idiot", "idiotic", "stupid", "dumb", "moron", "moronic", "scoundrel",
    "loser", "pathetic", "ugly", "fool", "foolish", "worthless", "trash",
    "garbage", "disgusting", "hate", "hateful", "shut up", "kill yourself",
    "die", "freak", "psycho", "crazy", "retard", "retarded", "dumbass",
    "jerk", "creep", "coward", "clown", "pig", "rat", "scum", "vermin",
    "loathe", "despise", "detest", "useless", "shameful", "disgrace",
    "bully", "bullied", "harass", "harassment", "threat", "threaten",
}

# Multi-word phrases need substring matching against the raw (lowercased,
# but *not* stopword-stripped) text, since cleaning removes "you", "up", etc.
PHRASE_TERMS = {
    "shut up", "kill yourself", "go die", "you are bad", "you're bad",
    "i hate you", "no one likes you", "nobody likes you",
}


def lexicon_score(raw_text: str, cleaned_words: list) -> float:
    """Returns a boost score in [0, 1] if the text matches known insult terms."""
    lowered = raw_text.lower()
    for phrase in PHRASE_TERMS:
        if phrase in lowered:
            return 0.85

    word_set = set(cleaned_words)
    if word_set & INSULT_TERMS:
        return 0.80

    return 0.0
