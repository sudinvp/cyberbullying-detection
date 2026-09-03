"""
Loads the trained hybrid LSTM+CNN text model and exposes clean_text() /
sentences_to_indices() / predict() helpers used by the Flask routes.

This mirrors the preprocessing used in models/train_model.py so that
inference is consistent with training.
"""
import pickle
import re

import numpy as np

STOPWORDS = set("""
a an the this that these those is are was were be been being am
i you he she it we they me him her us them my your his its our their
mine yours hers ours theirs myself yourself himself herself itself ourselves
themselves and or but if then else so because as until while of at by for
with about against between into through during before after above below to
from up down in out on off over under again further once here there when
where why how all any both each few more most other some such no nor not
only own same than too very s t can will just don should now do does did
having have has had do does did doing would could should might must shall
will can may
""".split())

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")

_model = None
_word_to_index = None
_max_len = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def sentences_to_indices(sentences, word_to_index, max_len):
    m = len(sentences)
    X = np.zeros((m, max_len), dtype=np.int32)
    for i, sent in enumerate(sentences):
        j = 0
        for word in sent.split():
            if j >= max_len:
                break
            if word in word_to_index:
                X[i, j] = word_to_index[word]
                j += 1
    return X


def init(model_path, word_index_path, meta_path):
    """Load model + tokenizer once at app startup. Returns (word_to_index, max_len)."""
    global _model, _word_to_index, _max_len
    from tensorflow.keras.models import load_model

    _model = load_model(model_path)
    with open(word_index_path, "rb") as f:
        _word_to_index = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    _max_len = meta["max_len"]
    return _word_to_index, _max_len


def predict(text: str) -> float:
    """Returns a bullying-probability score in [0, 1] for a raw text string.

    This blends the trained LSTM+CNN model's score with a small rule-based
    lexicon check (see src/lexicon.py) — the model alone under-scores short,
    generic insults ("idiot", "you are bad") because the training data is
    mostly full-sentence identity-based harassment tweets, not one-line
    name-calling. We take the max of the two signals rather than either
    alone, so the model still drives nuanced/contextual detection while the
    lexicon catches the obvious cases it misses.
    """
    if _model is None:
        raise RuntimeError("model_utils.init() must be called before predict()")
    cleaned_text = clean_text(text)
    X = sentences_to_indices([cleaned_text], _word_to_index, _max_len)
    model_score = float(_model.predict(X, verbose=0)[0][0])

    from src.lexicon import lexicon_score
    rule_score = lexicon_score(text, cleaned_text.split())

    return max(model_score, rule_score)
