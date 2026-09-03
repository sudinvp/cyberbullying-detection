"""
Train the hybrid LSTM + CNN cyberbullying text classifier.

Architecture (matches the project report / notebook):
  Input (token indices) -> GloVe Embedding (frozen)
     -> Branch A: LSTM(64)               -> GlobalMaxPooling1D
     -> Branch B: Conv1D(128, k=5) -> MaxPooling1D -> GlobalMaxPooling1D
     -> concatenate(Branch A, Branch B) -> Dense(128, relu) -> Dropout(0.5)
     -> Dense(1, sigmoid)   [binary: 0 = not bullying, 1 = bullying]

Run:
    python models/train_model.py --glove /path/to/glove.6B.50d.txt --data /path/to/cyberbullying_tweets.csv
"""
import argparse
import pickle
import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- built-in stopword list (avoids needing an nltk data download) ----
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


def clean_text(text: str) -> str:
    """Lightweight cleaner: lowercase, strip urls/mentions/punct/stopwords."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def read_glove_vecs(glove_file):
    word_to_index, index_to_word, word_to_vec_map = {}, {}, {}
    with open(glove_file, "r", encoding="utf8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            word_to_index[word] = idx
            index_to_word[idx] = word
            word_to_vec_map[word] = np.asarray(parts[1:], dtype=np.float32)
    return word_to_index, index_to_word, word_to_vec_map


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--balanced_per_class", type=int, default=8000)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    print("Loading GloVe vectors ...")
    word_to_index, _, word_to_vec_map = read_glove_vecs(args.glove)
    print(f"  vocab size: {len(word_to_index)}")

    print("Loading dataset ...")
    df = pd.read_csv(args.data)
    df["label"] = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)

    # Balance classes so the model doesn't just learn to predict "bullying"
    not_bully = df[df["label"] == 0]
    bully = df[df["label"] == 1].sample(
        n=min(args.balanced_per_class, len(df[df["label"] == 1])), random_state=42
    )
    not_bully = not_bully.sample(n=min(args.balanced_per_class, len(not_bully)), random_state=42)
    data = pd.concat([not_bully, bully]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  training on {len(data)} balanced rows (0: {len(not_bully)}, 1: {len(bully)})")

    print("Cleaning text ...")
    data["clean"] = data["tweet_text"].apply(clean_text)
    data = data[data["clean"].str.len() > 0]

    X = sentences_to_indices(data["clean"].tolist(), word_to_index, args.max_len)
    y = data["label"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building embedding matrix ...")
    vocab_len = len(word_to_index)
    emb_dim = len(next(iter(word_to_vec_map.values())))
    embedding_matrix = np.zeros((vocab_len, emb_dim), dtype=np.float32)
    for word, idx in word_to_index.items():
        embedding_matrix[idx] = word_to_vec_map[word]

    from tensorflow.keras.layers import (
        Input, Embedding, LSTM, Conv1D, MaxPooling1D,
        GlobalMaxPooling1D, concatenate, Dense, Dropout,
    )
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(args.max_len,))
    embedding_layer = Embedding(
        input_dim=vocab_len, output_dim=emb_dim,
        weights=[embedding_matrix], input_length=args.max_len, trainable=False,
    )
    embedded = embedding_layer(input_layer)

    # LSTM branch
    lstm_branch = LSTM(64, return_sequences=True)(embedded)
    lstm_branch = GlobalMaxPooling1D()(lstm_branch)

    # CNN branch
    cnn_branch = Conv1D(filters=128, kernel_size=5, activation="relu", padding="same")(embedded)
    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = GlobalMaxPooling1D()(cnn_branch)

    merged = concatenate([lstm_branch, cnn_branch])
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    output_layer = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc*100:.2f}%")

    model.save(f"{args.out_dir}/cyberbullying_lstm_cnn.h5")
    with open(f"{args.out_dir}/word_to_index.pkl", "wb") as f:
        pickle.dump(word_to_index, f)
    with open(f"{args.out_dir}/meta.pkl", "wb") as f:
        pickle.dump({"max_len": args.max_len}, f)
    print("Saved model + tokenizer artifacts to", args.out_dir)


if __name__ == "__main__":
    main()
