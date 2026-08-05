from flask import Flask, render_template

app = Flask(__name__)

# Temporary hardcoded data — will be replaced with real model output later
now_playing = {
    "track_name": "Baby Can I Hold You",
    "artist": "Tracy Chapman",
    "genre": "acoustic"
}

similar_tracks = [
    {"track_name": "Fast Car", "artist": "Tracy Chapman"},
    {"track_name": "Talkin Bout a Revolution", "artist": "Tracy Chapman"},
    {"track_name": "Mountains O Things", "artist": "Tracy Chapman"},
    {"track_name": "For My Lover", "artist": "Tracy Chapman"},
]

@app.route("/")
def home():
    return render_template("index.html", now_playing=now_playing, similar_tracks=similar_tracks)

if __name__ == "__main__":
    app.run(debug=True)