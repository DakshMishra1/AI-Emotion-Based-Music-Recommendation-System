"""
app.py — Flask backend for AI Emotion-Based Music Recommendation System
Run: python app.py
"""

import os
import pickle
import sqlite3
import datetime
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = "c37dde5163e94cb99451a98c936a1b78"
SPOTIFY_CLIENT_SECRET = "033482a842e14c30b911b8cbd94cedb9"

# ── Your model uses EXACTLY these 4 labels (from dataset.csv) ──
# joy | sadness | anger | fear

EMOTION_QUERIES = {
    "joy":     "happy upbeat feel good pop hits",
    "sadness": "sad heartbreak melancholy indie songs",
    "anger":   "angry intense rock metal energy",
    "fear":    "calming peaceful ambient relaxing meditation",
}

FALLBACK_BY_EMOTION = {
    "joy": [
        {"name": "Happy",                  "artist": "Pharrell Williams",   "duration": "3:53"},
        {"name": "Good as Hell",           "artist": "Lizzo",               "duration": "2:39"},
        {"name": "Can't Stop the Feeling", "artist": "Justin Timberlake",   "duration": "3:56"},
        {"name": "Uptown Funk",            "artist": "Bruno Mars",          "duration": "4:30"},
        {"name": "Walking on Sunshine",    "artist": "Katrina & The Waves", "duration": "3:58"},
    ],
    "sadness": [
        {"name": "Someone Like You", "artist": "Adele",      "duration": "4:45"},
        {"name": "The Night We Met", "artist": "Lord Huron", "duration": "3:28"},
        {"name": "Skinny Love",      "artist": "Bon Iver",   "duration": "3:58"},
        {"name": "Fix You",          "artist": "Coldplay",   "duration": "4:55"},
        {"name": "Mad World",        "artist": "Gary Jules", "duration": "3:08"},
    ],
    "anger": [
        {"name": "Break Stuff",         "artist": "Limp Bizkit",              "duration": "2:46"},
        {"name": "Given Up",            "artist": "Linkin Park",              "duration": "3:08"},
        {"name": "Killing in the Name", "artist": "Rage Against the Machine", "duration": "5:13"},
        {"name": "Numb",                "artist": "Linkin Park",              "duration": "3:05"},
        {"name": "Cochise",             "artist": "Audioslave",               "duration": "3:44"},
    ],
    "fear": [
        {"name": "Weightless",     "artist": "Marconi Union",             "duration": "8:09"},
        {"name": "Breathe (2 AM)", "artist": "Anna Nalick",               "duration": "4:03"},
        {"name": "The Scientist",  "artist": "Coldplay",                  "duration": "5:09"},
        {"name": "Holocene",       "artist": "Bon Iver",                  "duration": "5:37"},
        {"name": "Atlas Hands",    "artist": "Benjamin Francis Leftwich", "duration": "4:03"},
    ],
}

SUGGESTIONS = {
    "joy":     "Your positive energy is contagious! Channel it into something creative — art, music, or reaching out to someone you care about.",
    "sadness": "It's okay to feel sad. Try journaling your thoughts or talking to someone you trust. This feeling is temporary.",
    "anger":   "Physical movement is the fastest way to discharge anger. Take a brisk walk, then revisit what upset you with fresh eyes.",
    "fear":    "Try 4-7-8 breathing: inhale 4 counts, hold 7, exhale 8. Writing down your worries can reduce their power over you.",
}

# ─────────────────────────────────────────────────────────────────────
#  SPOTIFY SETUP
# ─────────────────────────────────────────────────────────────────────
SPOTIFY_AVAILABLE = False
sp = None
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ))
    SPOTIFY_AVAILABLE = True
    print("✅ Spotify connected")
except Exception as e:
    print("⚠  Spotify error:", e)

# ─────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded")
print("   Emotion classes:", list(model.classes_))

# ─────────────────────────────────────────────────────────────────────
#  DATABASE  (defined BEFORE any route uses DB_PATH)
# ─────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "mood_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT,
            emotion   TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def get_tracks(emotion, limit=5):
    query_map = {
        "joy": [
            {"name": "Happy - Pharrell Williams", "url": "https://www.youtube.com/results?search_query=Happy+Pharrell+Williams"},
            {"name": "Can't Stop the Feeling - Justin Timberlake", "url": "https://www.youtube.com/results?search_query=Cant+Stop+the+Feeling"},
            {"name": "On Top of the World - Imagine Dragons", "url": "https://www.youtube.com/results?search_query=On+Top+of+the+World"},
            {"name": "Good Life - OneRepublic", "url": "https://www.youtube.com/results?search_query=Good+Life+OneRepublic"},
            {"name": "Best Day of My Life - American Authors", "url": "https://www.youtube.com/results?search_query=Best+Day+of+My+Life"}
        ],
        "sadness": [
            {"name": "Someone Like You - Adele", "url": "https://www.youtube.com/results?search_query=Someone+Like+You+Adele"},
            {"name": "Let Her Go - Passenger", "url": "https://www.youtube.com/results?search_query=Let+Her+Go"},
            {"name": "Fix You - Coldplay", "url": "https://www.youtube.com/results?search_query=Fix+You+Coldplay"},
            {"name": "All I Want - Kodaline", "url": "https://www.youtube.com/results?search_query=All+I+Want+Kodaline"},
            {"name": "Stay With Me - Sam Smith", "url": "https://www.youtube.com/results?search_query=Stay+With+Me+Sam+Smith"}
        ],
        "anger": [
            {"name": "Stronger - Kanye West", "url": "https://www.youtube.com/results?search_query=Stronger+Kanye+West"},
            {"name": "Lose Yourself - Eminem", "url": "https://www.youtube.com/results?search_query=Lose+Yourself+Eminem"},
            {"name": "Numb - Linkin Park", "url": "https://www.youtube.com/results?search_query=Numb+Linkin+Park"},
            {"name": "Believer - Imagine Dragons", "url": "https://www.youtube.com/results?search_query=Believer+Imagine+Dragons"},
            {"name": "Till I Collapse - Eminem", "url": "https://www.youtube.com/results?search_query=Till+I+Collapse"}
        ],
        "fear": [
            {"name": "Weightless - Ambient", "url": "https://www.youtube.com/results?search_query=Weightless+ambient"},
            {"name": "River Flows In You - Yiruma", "url": "https://www.youtube.com/results?search_query=River+Flows+In+You"},
            {"name": "Clair de Lune", "url": "https://www.youtube.com/results?search_query=Clair+de+Lune"},
            {"name": "Sunset Lover - Petit Biscuit", "url": "https://www.youtube.com/results?search_query=Sunset+Lover"},
            {"name": "Bloom - ODESZA", "url": "https://www.youtube.com/results?search_query=Bloom+ODESZA"}
        ]
    }

    return query_map.get(emotion, query_map["joy"])[:limit]

# ─────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "MoodTune API running"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Missing text"}), 400

    text    = data["text"].strip()
    cleaned = preprocess(text)
    vec     = vectorizer.transform([cleaned])

    emotion = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    scores  = {cls: round(float(p), 4) for cls, p in zip(model.classes_, proba)}

    # Save to history
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO mood_history (text, emotion, timestamp) VALUES (?, ?, ?)",
        (text, emotion, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify({
        "emotion":    emotion,
        "confidence": float(max(proba)),
        "scores":     scores,
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data    = request.get_json()
    emotion = data.get("emotion", "joy")
    tracks  = get_tracks(emotion)
    return jsonify({"tracks": tracks})


@app.route("/api/suggestions", methods=["POST"])
def suggestions():
    data    = request.get_json()
    emotion = data.get("emotion", "joy")
    tip     = SUGGESTIONS.get(emotion, "Take care of yourself today.")
    return jsonify({"suggestion": tip})


@app.route("/api/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 20))
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT id, text, emotion, timestamp FROM mood_history ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return jsonify({
        "history": [
            {"id": r[0], "text": r[1], "emotion": r[2], "timestamp": r[3]}
            for r in rows
        ]
    })


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎵 MoodTune Flask API")
    print("   → http://127.0.0.1:5500")
    print("   → Spotify:", "✅ connected" if SPOTIFY_AVAILABLE else "⚠  offline (using fallback songs)")
    print()
    app.run(debug=True, port=5000)
