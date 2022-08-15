from flask import Flask, render_template, request, session
from flask_session import Session

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

notes = []
@app.route('/', methods=["POST","GET"])
def index():
    if session.get("notes") is None:
        session["notes"]=[]
    if request.method=="POST":
        note = request.form["note"]
        if note!="":
            session["notes"].append(note)
    return render_template("home.html", notes=session["notes"])

if __name__ == '__main__':
    app.run(debug=True)