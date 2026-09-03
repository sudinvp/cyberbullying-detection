import os

from flask import Flask
from flask_session import Session
from werkzeug.exceptions import HTTPException, InternalServerError

from src.auth import auth
from src.helpers import error
from src.home import home
from src.profile_bp import profile
from src.search_bp import search
from src import model_utils

app = Flask(__name__)
app.config.from_object("config")

# Register Blueprints
app.register_blueprint(auth, url_prefix="/")
app.register_blueprint(home, url_prefix="/")
app.register_blueprint(profile, url_prefix="/")
app.register_blueprint(search, url_prefix="/")

Session(app)

# Load the hybrid LSTM+CNN text model once, at startup, so requests are fast.
with app.app_context():
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("static/dp", exist_ok=True)
    model_utils.init(
        app.config["MODEL_PATH"],
        app.config["WORD_INDEX_PATH"],
        app.config["META_PATH"],
    )
    print("Cyberbullying detection model loaded.")


@app.errorhandler(Exception)
def errorhandler(e):
    print(str(e))
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return error(e.name, e.code)


if __name__ == "__main__":
    app.run(debug=app.config.get("DEBUG", True))
