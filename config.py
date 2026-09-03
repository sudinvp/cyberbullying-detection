import os
from tempfile import mkdtemp

DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
TEMPLATES_AUTO_RELOAD = True
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

SESSION_FILE_DIR = mkdtemp()
SESSION_PERMANENT = False
SESSION_TYPE = "filesystem"

# --- Optional integrations (all disabled unless configured via env vars) ---
# Tesseract OCR binary path. Leave unset on Linux/Mac if tesseract is on PATH.
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")

# Telegram alert bot (optional). If unset, alerts are just logged to console.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Model artifacts
MODEL_PATH = os.environ.get("MODEL_PATH", "models/cyberbullying_lstm_cnn.h5")
WORD_INDEX_PATH = os.environ.get("WORD_INDEX_PATH", "models/word_to_index.pkl")
META_PATH = os.environ.get("META_PATH", "models/meta.pkl")

# Reputation thresholds (0-1 bullying probability)
NEGATIVE_THRESHOLD = 0.50
NEUTRAL_THRESHOLD = 0.30
