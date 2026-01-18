from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import os
import pandas as pd
import random
import re
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.secret_key = "simple_secret_key"

# ---------------------------------
# DATABASE
# ---------------------------------
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            result TEXT,
            confidence REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score INTEGER,
            total INTEGER,
            level TEXT
        )
    """)

    conn.commit()
    conn.close()

# ---------------------------------
# TEXT CLEANING
# ---------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------------
# AI MODEL
# ---------------------------------
def load_model():
    if os.path.exists("model/fraud_model.pkl"):
        with open("model/fraud_model.pkl", "rb") as f:
            return pickle.load(f)

    data = {
        "text": [
            # FRAUD
            "urgent verify your bank account",
            "click here to reset your password",
            "your account has been suspended",
            "confirm your login details immediately",
            "limited time offer claim your prize",
            "you have won a gift card",
            "update your payment information",
            "security alert unusual login detected",
            "act now to avoid account closure",
            "verify your identity now",

            # SAFE
            "meeting scheduled tomorrow at 3 pm",
            "invoice attached please review",
            "lunch at home today",
            "project deadline is next monday",
            "team meeting rescheduled",
            "please find the report attached",
            "can we discuss this tomorrow",
            "thanks for your help",
            "family dinner tonight",
            "see you at the office"
        ],
        "label": [
            "FRAUD","FRAUD","FRAUD","FRAUD","FRAUD",
            "FRAUD","FRAUD","FRAUD","FRAUD","FRAUD",
            "SAFE","SAFE","SAFE","SAFE","SAFE",
            "SAFE","SAFE","SAFE","SAFE","SAFE"
        ]
    }

    df = pd.DataFrame(data)
    df["text"] = df["text"].apply(clean_text)

    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        ),
        MultinomialNB()
    )

    model.fit(df["text"], df["label"])

    os.makedirs("model", exist_ok=True)
    with open("model/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

model = load_model()
init_db()

# ---------------------------------
# HOME ROUTE (DETECTOR)
# ---------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    email_text = ""
    highlighted_text = ""

    if request.method == "POST":
        email_text = request.form["email"]
        cleaned_email = clean_text(email_text)

        result = model.predict([cleaned_email])[0]
        confidence = round(model.predict_proba([cleaned_email]).max() * 100, 2)

        # Save scan to DB
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO scans (email, result, confidence) VALUES (?, ?, ?)",
            (email_text, result, confidence)
        )
        conn.commit()
        conn.close()

        suspicious_words = [
            "urgent", "verify", "click", "password",
            "bank", "account", "login", "confirm"
        ]

        highlighted_text = email_text

        if result == "FRAUD":
            for word in suspicious_words:
                highlighted_text = highlighted_text.replace(
                    word,
                    f"<span class='highlight'>{word}</span>"
                )
                highlighted_text = highlighted_text.replace(
                    word.capitalize(),
                    f"<span class='highlight'>{word.capitalize()}</span>"
                )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        email_text=email_text,
        highlighted_text=highlighted_text
    )

# ---------------------------------
# PHISHING TRAINING GAME
# ---------------------------------
QUIZ_QUESTIONS = [
    {
        "text": "Urgent! Verify your account immediately.",
        "answer": "FRAUD",
        "reason": "Creates urgency and asks for action."
    },
    {
        "text": "Meeting scheduled tomorrow at 3 PM.",
        "answer": "SAFE",
        "reason": "Normal internal communication."
    },
    {
        "text": "Click here to reset your password.",
        "answer": "FRAUD",
        "reason": "Suspicious link asking for credentials."
    },
    {
        "text": "Invoice attached. Please review.",
        "answer": "SAFE",
        "reason": "Common business email."
    }
]

@app.route("/game", methods=["GET", "POST"])
def game():

    if "score" not in session:
        session["score"] = 0
        session["total"] = 0

    feedback = None
    question = random.choice(QUIZ_QUESTIONS)

    if request.method == "POST":
        user_choice = request.form["choice"]
        correct_answer = request.form["correct"]
        reason = request.form["reason"]

        session["total"] += 1

        if user_choice == correct_answer:
            session["score"] += 1
            feedback = "✅ Correct! " + reason
        else:
            feedback = "❌ Wrong! " + reason

    score = session["score"]

    if score < 3:
        level = "Beginner"
    elif score < 6:
        level = "Intermediate"
    else:
        level = "Expert"

    # Save game score
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO game_scores (score, total, level) VALUES (?, ?, ?)",
        (session["score"], session["total"], level)
    )
    conn.commit()
    conn.close()

    return render_template(
        "game.html",
        question=question,
        feedback=feedback,
        score=session["score"],
        total=session["total"],
        level=level
    )

@app.route("/reset-game")
def reset_game():
    session.clear()
    return redirect(url_for("game"))


    # ---------------------------------
# VIEW SCAN HISTORY
# ---------------------------------
@app.route("/history")
def history():
    conn = get_db_connection()
    scans = conn.execute(
        "SELECT email, result, confidence FROM scans ORDER BY id DESC"
    ).fetchall()
    conn.close()

    return render_template("history.html", scans=scans)


# ---------------------------------
# VIEW GAME SCORES
# ---------------------------------
@app.route("/scores")
def scores():
    conn = get_db_connection()
    scores = conn.execute(
        "SELECT score, total, level FROM game_scores ORDER BY id DESC"
    ).fetchall()
    conn.close()

    return render_template("scores.html", scores=scores)


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
