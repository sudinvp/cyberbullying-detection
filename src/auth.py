from functools import wraps

from cs50 import SQL
from flask import Blueprint, redirect, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

auth = Blueprint("auth", __name__, static_folder="static", template_folder="templates")
db = SQL("sqlite:///src/main.db")


@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username")
    password = request.form.get("password")
    confirm = request.form.get("confirm")

    if not username:
        return render_template("login.html", msg="You must provide a username")
    if not password:
        return render_template("login.html", msg="You must provide a password")
    if password != confirm:
        return render_template("login.html", msg="Your passwords do not match")

    existing = db.execute("SELECT * FROM users WHERE username = :username", username=username)
    if existing:
        return render_template("login.html", msg="Username already taken")

    try:
        db.execute(
            "INSERT INTO users (username, hash) VALUES (:username, :hash)",
            username=username, hash=generate_password_hash(password),
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS :tablename "
            "('id' INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'text' TEXT NOT NULL, "
            "'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP, 'image' TEXT, 'nature' TEXT DEFAULT 'na')",
            tablename=username,
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS :tablename "
            "('following' TEXT PRIMARY KEY NOT NULL, 'timestamp' DATETIME DEFAULT CURRENT_TIMESTAMP)",
            tablename=str(username) + "Social",
        )
        return redirect("/")
    except Exception:
        return render_template("login.html", msg="Username already taken")


@auth.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if request.method == "POST":
        if not request.form.get("username"):
            return render_template("login.html", msg="You must provide username")
        if not request.form.get("password"):
            return render_template("login.html", msg="You must provide password")

        account = db.execute(
            "SELECT * FROM users WHERE username = :username",
            username=request.form.get("username"),
        )
        if len(account) != 1 or not check_password_hash(account[0]["hash"], request.form.get("password")):
            return render_template("login.html", msg="Invalid username and/or password")

        session["user_id"] = account[0]["id"]
        return redirect("/")

    return render_template("login.html")


@auth.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function
