from flask import Flask, render_template, jsonify
from knn_recommender import get_recommendations, data

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/genres")
def genres():
    # Return a sorted list of all unique genres
    genre_list = sorted(data["track_genre"].unique().tolist())
    return jsonify(genre_list)

@app.route("/songs/<genre>")
def songs_by_genre(genre):
    # Return songs belonging to the selected genre (name, artist, id)
    subset = data[data["track_genre"] == genre]
    subset = subset.drop_duplicates(subset=["track_name", "artists"])
    songs = subset[["track_id", "track_name", "artists"]].head(50).to_dict(orient="records")
    return jsonify(songs)

@app.route("/recommend/<track_id>")
def recommend(track_id):
    result, error = get_recommendations(track_id, k=5)
    if error:
        return jsonify({"error": error}), 404
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)