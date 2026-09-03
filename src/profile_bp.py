import os

from cs50 import SQL
from flask import Blueprint, redirect, render_template, request, session

from src.auth import login_required
from src.helpers import UserInfo

profile = Blueprint("profile", __name__, static_folder="static", template_folder="templates")
db = SQL("sqlite:///src/main.db")


@profile.route("/me", methods=["GET", "POST"])
@login_required
def me():
    userInfo, dp = UserInfo(db)

    if request.method == "POST":
        if "dp_upload" in request.files and request.files["dp_upload"].filename:
            file = request.files["dp_upload"]
            ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "png"
            os.makedirs("static/dp", exist_ok=True)
            file.save(f"static/dp/{userInfo['username']}.{ext}")
            db.execute("UPDATE users SET dp=:ext WHERE id=:user_id", ext=ext, user_id=session["user_id"])
        elif request.form.get("bio") is not None:
            db.execute(
                "UPDATE users SET bio=:bio WHERE id=:user_id",
                bio=request.form.get("bio"), user_id=session["user_id"],
            )
        return redirect("/me")

    userInfo, dp = UserInfo(db)
    posts = db.execute("SELECT * FROM :tablename ORDER BY id DESC", tablename=userInfo["username"])
    reputation = round((userInfo["score"] / userInfo["total"]) * 10, 2) if userInfo["total"] else 10.0
    return render_template("profile.html", userInfo=userInfo, dp=dp, reputation=reputation, posts=posts)


@profile.route("/remove/<int:Id>")
@login_required
def Remove(Id):
    userInfo, dp = UserInfo(db)
    db.execute("DELETE FROM :tablename WHERE id=:id", tablename=userInfo["username"], id=Id)
    return redirect("/me")


@profile.route("/<username>", methods=["GET", "POST"])
@login_required
def view_profile(username):
    me_info, _ = UserInfo(db)
    user, dp = UserInfo(db, username)
    if user is None:
        return redirect("/search")

    if request.method == "POST":
        action = request.form.get("follow_button")
        if action == "follow":
            db.execute(
                "INSERT OR IGNORE INTO :tablename (following) VALUES (:username)",
                tablename=me_info["username"] + "Social", username=username,
            )
        elif action == "unfollow":
            db.execute(
                "DELETE FROM :tablename WHERE following=:username",
                tablename=me_info["username"] + "Social", username=username,
            )

    follow_info = db.execute(
        "SELECT * FROM :tablename WHERE following=:username",
        tablename=me_info["username"] + "Social", username=username,
    )
    posts = db.execute("SELECT * FROM :tablename ORDER BY id DESC", tablename=username)
    reputation = round((user["score"] / user["total"]) * 10, 2) if user["total"] else 10.0
    return render_template(
        "found_profile.html", user=user, dp=dp, reputation=reputation,
        posts=posts, follow_info=follow_info,
    )
