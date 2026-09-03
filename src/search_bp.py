from cs50 import SQL
from flask import Blueprint, render_template, request

from src.auth import login_required
from src.helpers import UserInfo

search = Blueprint("search", __name__, static_folder="static", template_folder="templates")
db = SQL("sqlite:///src/main.db")


@search.route("/search", methods=["GET", "POST"])
@login_required
def do_search():
    if request.method == "POST":
        username = request.form.get("username", "")
        rows = db.execute(
            "SELECT * FROM users WHERE username LIKE :pattern",
            pattern=f"%{username}%",
        )
        results = []
        dp = "../static/dp/default.png"
        for row in rows:
            results.append({"username": row["username"], "bio": row["bio"] or ""})
        return render_template("search.html", method="POST", results=results, dp=dp)

    return render_template("search.html", method="GET")
