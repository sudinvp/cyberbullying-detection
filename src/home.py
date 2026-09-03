import socket

from cs50 import SQL
from flask import Blueprint, current_app, flash, redirect, render_template, request, session

from src.auth import login_required
from src.helpers import UserInfo, error
from src import model_utils

home = Blueprint("home", __name__, static_folder="static", template_folder="templates")
db = SQL("sqlite:///src/main.db")
blocked_ips = set()


def get_system_ip():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except OSError:
        return "unknown"


def send_alert(message):
    """Send a moderation alert via Telegram if configured, otherwise just log it."""
    token = current_app.config.get("TELEGRAM_BOT_TOKEN")
    chat_id = current_app.config.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            import telepot
            telepot.Bot(token).sendMessage(chat_id, message)
        except Exception as exc:
            print(f"[alert] failed to send telegram alert: {exc}")
    else:
        print(f"[alert] {message}")


def translate_text(text, source_lang):
    """Best-effort translation to English.

    googletrans (an unofficial, reverse-engineered library) can silently
    return garbled/wrong text when Google's internal endpoints change,
    rather than raising an error — which is why translation could look like
    it "worked" but produced a nonsense word. We try googletrans first, then
    fall back to deep-translator (a more actively maintained wrapper around
    the same public Google Translate web endpoint), and log every step so
    failures are visible in the console instead of silently swallowed.
    """
    if source_lang == "en" or not text:
        return text

    print(f"[translate] input ({source_lang}): {text!r}")

    try:
        from googletrans import Translator
        result = Translator().translate(text, src=source_lang, dest="en").text
        print(f"[translate] googletrans -> {result!r}")
        if result and result.strip().lower() != text.strip().lower():
            return result
    except Exception as exc:
        print(f"[translate] googletrans failed: {exc}")

    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=source_lang, target="en").translate(text)
        print(f"[translate] deep_translator -> {result!r}")
        if result:
            return result
    except Exception as exc:
        print(f"[translate] deep_translator failed: {exc}")

    print("[translate] all translators failed/unavailable, posting original text")
    return text


class OcrUnavailable(Exception):
    """Raised when Tesseract itself can't be reached (not installed / bad path)."""


def ocr_image(path):
    """Extract text from an uploaded image via Tesseract OCR.

    Raises OcrUnavailable if Tesseract can't be run at all (so the caller can
    tell the user to install it, instead of silently doing nothing). Returns
    "" (no exception) if Tesseract runs fine but just finds no text.
    """
    import cv2
    import pytesseract
    from pytesseract import TesseractNotFoundError

    cmd = current_app.config.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    img = cv2.imread(path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        return pytesseract.image_to_string(gray)
    except TesseractNotFoundError as exc:
        raise OcrUnavailable(str(exc))


def add_publisher(posts, publisher):
    for item in posts:
        item["publisher"] = publisher
    return posts


def get_timestamp(post):
    return post.get("timestamp")


def record_post(userInfo, ans, extra_update=True):
    """Update the user's reputation score/total based on a prediction score."""
    if ans < 0.4:
        score = 0.4 - ans
        total = "{:.2f}".format(userInfo["total"] + score)
        good_score = "{:.2f}".format(userInfo["score"] + score)
        if extra_update:
            db.execute(
                "UPDATE users SET score=:score, total=:total WHERE id=:user_id",
                score=good_score, total=total, user_id=session["user_id"],
            )
    else:
        score = ans - 0.4
        total = "{:.2f}".format(userInfo["total"] + score)
        if extra_update:
            db.execute(
                "UPDATE users SET total=:total WHERE id=:user_id",
                total=total, user_id=session["user_id"],
            )


@home.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    userInfo, dp = UserInfo(db)
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Choose an image file before clicking Post.")
        return redirect("/")

    import os
    os.makedirs("static/images", exist_ok=True)
    path = f"static/images/{file.filename}"
    file.save(path)

    try:
        text1 = ocr_image(path)
    except OcrUnavailable:
        flash(
            "Tesseract OCR isn't installed/configured, so text can't be read from "
            "images. Install it and set TESSERACT_CMD (see .env.example) — text "
            "posts work fine without it."
        )
        return redirect("/")

    post_text = text1.strip()
    if not post_text:
        flash("No readable text was found in that image, so nothing was posted.")
        return redirect("/")

    ans = model_utils.predict(post_text)
    db.execute(
        "INSERT INTO :tablename ('text', 'nature', 'image') VALUES (:post_text, :score, :post_img)",
        tablename=userInfo["username"], post_text=post_text, score=str(ans), post_img=path,
    )
    record_post(userInfo, ans)
    return redirect("/")


@home.route("/", methods=["GET", "POST"])
@login_required
def index():
    userInfo, dp = UserInfo(db)

    if request.method == "GET":
        get_posts = db.execute("SELECT * FROM :tablename", tablename=userInfo["username"])
        get_posts = add_publisher(get_posts, userInfo["username"])

        follow_metadata = db.execute("SELECT following FROM :tablename", tablename=userInfo["username"] + "Social")
        posts_metadata = {userInfo["username"]: dp}
        for following in follow_metadata:
            following_posts = db.execute("SELECT * FROM :tablename", tablename=following["following"])
            following_posts = add_publisher(following_posts, following["following"])
            other_user_info, other_user_dp = UserInfo(db, following["following"])
            posts_metadata[following["following"]] = other_user_dp
            get_posts.extend(following_posts)

        get_posts.sort(key=get_timestamp, reverse=True)

        if get_posts:
            reputation = (userInfo["score"] / userInfo["total"]) * 10
            if reputation < 5:
                blocked_ips.add(get_system_ip())
                send_alert(f"{userInfo['username']}'s account was flagged: reputation below 5")
                return render_template("index.html", msg="Your account has been flagged for repeated negative posts.")
            return render_template(
                "index.html", posts=get_posts, posts_metadata=posts_metadata,
                dp=dp, user=userInfo, reputation=round(reputation, 2),
            )
        return render_template("index.html", reputation=round((userInfo["score"] / userInfo["total"]) * 10, 2))

    # POST: new text post
    post_text = request.form.get("post")
    from_lang = request.form.get("lang", "en")
    if from_lang != "en":
        translated = translate_text(post_text, from_lang)
        if translated.encode("ascii", "ignore").decode().strip() == "" or translated.strip() == (post_text or "").strip():
            flash(
                "Translation to English failed, so this post couldn't be reliably "
                "scored. It was still posted, but treat the result with caution."
            )
        post_text = translated
    if not post_text:
        return redirect("/")

    ans = model_utils.predict(post_text)
    db.execute(
        "INSERT INTO :tablename ('text', 'nature') VALUES (:post_text, :score)",
        tablename=userInfo["username"], post_text=post_text, score=str(ans),
    )
    record_post(userInfo, ans)
    return redirect("/")


@home.route("/about", methods=["GET"])
@login_required
def about():
    return render_template("about.html")


@home.route("/unblock_my_ip")
def unblock_my_ip():
    my_ip = get_system_ip()
    blocked_ips.discard(my_ip)
    return f"IP address {my_ip} unblocked successfully"
