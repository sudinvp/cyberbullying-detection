import os

from flask import render_template, session


def UserInfo(db, username=None):
    """Look up a user row + their display picture path."""
    if not username:
        user_id_info = db.execute("SELECT * FROM users WHERE id = :id", id=session["user_id"])[0]
    else:
        rows = db.execute("SELECT * FROM users WHERE username = :username", username=username)
        if not rows:
            return None, "static/dp/default.png"
        user_id_info = rows[0]

    dp = f"static/dp/{user_id_info['username']}.{user_id_info['dp']}"
    if not os.path.exists(dp):
        dp = "../static/dp/default.png"
    else:
        dp = f"../static/dp/{user_id_info['username']}.{user_id_info['dp']}"
    return user_id_info, dp


def error(message, code=400):
    print(f"[error {code}] {message}")
    return render_template("error.html", message=message, code=code)
