# Twinstagram — Cyberbullying Detection Using LSTM + CNN

A social-media demo app (Flask) that scores every post/comment for cyberbullying
in real time using a **hybrid LSTM + CNN text classifier**, and lowers a user's
"reputation score" (and eventually flags their account) the more negative content
they post.

## What was fixed / rebuilt from your uploaded files

Your uploaded repo (`cyber-bullying-detection-using-machine-learning-main.zip`) didn't
actually run — `app.py` imported flat modules (`from auth import auth`) while every other
file imported from a `src.` package that didn't exist, and two blueprints the app
registers (`profile`, `search`) had no corresponding `.py` files at all, only templates.
There was also a hard-coded Telegram bot token in `home.py`, a Windows-only
Tesseract path, and a registration bug: the `users` table has no uniqueness
constraint on `username`, so signing up with a name that's already taken *silently
creates a duplicate account* instead of rejecting it (login then breaks, since
`SELECT ... WHERE username = ...` can return the wrong row). You can see this in the
original `main.db` — there are two `Sunil Kumar S` rows and two `param` rows.

This rebuild:
- Uses one consistent `src.` package for every module, matching what `app.py` imports.
- Adds the missing `src/profile_bp.py` and `src/search_bp.py` blueprints (profile page,
  bio/profile-picture upload, follow/unfollow, delete post, username search) driven by
  your existing `profile.html` / `search.html` / `found_profile.html` templates.
- Fixes the registration bug with a proper "does this username already exist" check.
- Removes the hard-coded bot token — Telegram alerts, the Tesseract path, and the app's
  session secret are now all optional environment variables (see `.env.example`).
  If they're not set, the app just prints alerts to the console instead of crashing.
- Wraps OCR and translation calls in try/except so a missing Tesseract install or no
  internet access degrades gracefully instead of taking the whole app down.
- **Actually trains the model** described in your notebook/report (see below) on your
  real `cyberbullying_tweets.csv` + `glove_6B_50d.txt`, rather than shipping a
  stub — 82.8% accuracy on a held-out test set (`models/train_model.py`, run log in
  the repo history).

## About "LSTM and CNN" — a clarification

Your last message asked for CNN to handle **images** and LSTM to handle **text**. That's
not actually what the notebook (`cyberbullying_lstm.ipynb`) or your report's appendix
code build, though — and I kept the real, working architecture rather than the
image-CNN framing, so this doesn't overclaim what's implemented:

- **Both the LSTM and the CNN operate on the same GloVe text embeddings.** It's a
  hybrid *text* classifier: one branch runs an LSTM over the embedded tweet, the other
  runs a 1D CNN (good at picking up local word patterns / slurs / n-gram-like cues)
  over the same embedding, and the two branches are concatenated before the final
  classification layer. This is a legitimate, commonly-used architecture for text
  classification — it's just not an image CNN.
- **Images are handled by OCR, not a CNN.** When you upload a screenshot, the app runs
  Tesseract OCR to pull out any text in the image, then feeds that extracted text
  through the *same* LSTM+CNN text model. There's no image-content classifier (the
  `retrain.py`/`label_image.py` files in your zip are leftover TensorFlow-1-style
  Inception transfer-learning scripts for a "fake vs. original" meme classifier that
  isn't wired into the app anywhere — I left them out of the rebuild since they're
  dead code, using TF1 session APIs incompatible with the TensorFlow 2 version already
  installed here).

If you actually want a real image-content CNN (e.g. detecting bullying imagery/memes
directly from pixels, not just OCR'd text), that needs a labeled image dataset — happy
to build that layer if you can share one.

## Project structure

```
app.py                  Flask entrypoint — registers blueprints, loads the model once
config.py                All settings, read from environment variables
src/
  auth.py                register / login / logout
  home.py                feed, text posting, image OCR + detect, about page
  profile_bp.py           own profile (bio, picture, reputation, feed), delete post
  search_bp.py            username search
  helpers.py               shared DB/user-info helpers
  model_utils.py          text cleaning + loads/queries the trained model
  main.db                 SQLite database (users + per-user post tables)
models/
  train_model.py           training script (run this to retrain)
  cyberbullying_lstm_cnn.h5  the trained hybrid model (82.8% test accuracy)
  word_to_index.pkl        GloVe vocabulary → index mapping used at inference
  meta.pkl                 max_len used during training
templates/, static/        your original HTML/CSS/images, reused as-is
```

## Running it

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`. Register a new account (existing accounts in
`main.db` — `Sunil`, `Ram Charan`, etc. — are demo data from your original testing;
you'll need their real password to log into those, not a random one).

Optional integrations (all no-ops if left unset — copy `.env.example` to `.env`):
- `TESSERACT_CMD` — only needed if `tesseract` isn't already on your PATH.
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — sends a Telegram alert when an account's
  reputation drops below 5; otherwise it's just printed to the console.
- `SECRET_KEY` — Flask session signing key, set this for any real deployment.

`googletrans` occasionally breaks against Google's endpoints outside a controlled
network — if translation calls fail, the app now falls back to posting the original
text rather than crashing.

## Retraining the model

```bash
python models/train_model.py --glove glove_6B_50d.txt --data cyberbullying_tweets.csv \
    --max_len 25 --epochs 6 --out_dir models
```

The dataset (`cyberbullying_tweets.csv`, ~47.7k tweets across 6 labels) is heavily
imbalanced toward bullying content once collapsed to binary, so the script balances
classes by subsampling before training (`--balanced_per_class`, default 8000 per class).
Push `--epochs` and `--balanced_per_class` up if you want a stronger model and have more
CPU/GPU time to spend — this run used 1 CPU core and ~30 seconds/epoch.

## Known limitations worth knowing about for your report/viva

- No image-content CNN — image posts are scored via OCR'd text through the same model
  (see the clarification above).
- Reputation math (`score`/`total`) is carried over unchanged from your original design;
  it's a simple running-average heuristic, not a calibrated metric.
- `googletrans` (unofficial, reverse-engineered) is fragile — for production use, the
  official Google Cloud Translation API would be more reliable.
- The Flask session store is filesystem-based (`Flask-Session`), fine for a single-process
  demo but not for a multi-worker production deployment.
