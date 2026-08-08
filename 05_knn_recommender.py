import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
data = pd.read_csv("spotify_tracks_clean.csv")

# Load the trained classifier and label encoder
classifier = joblib.load("genre_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Same feature columns used for training the classifier
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

def get_recommendations(track_id, k=5):
    """
    Given a track_id, predict its genre using the classifier,
    then find the k most similar songs within that genre using KNN.
    """
    # Find the input song's row
    song_row = data[data["track_id"] == track_id]
    if song_row.empty:
        return None, f"Track ID {track_id} not found in dataset."

    song_features = song_row[feature_cols]

    # Step 1: Predict genre using classifier (uses RAW features - correct, classifier was trained this way)
    predicted_genre_encoded = classifier.predict(song_features)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_encoded])[0]

    # Step 2: Filter dataset to only songs in that predicted genre
    genre_subset = data[data["track_genre"] == predicted_genre].reset_index(drop=True)

    # Step 3: Scale the features BEFORE computing distances
    scaler = StandardScaler()
    genre_features_scaled = scaler.fit_transform(genre_subset[feature_cols])
    song_features_scaled = scaler.transform(song_features)

    # Step 4: Fit KNN on the SCALED features
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(genre_features_scaled)

    # Step 5: Find nearest neighbors using the SCALED input song
    distances, indices = knn.kneighbors(song_features_scaled)

    # Step 6: Get the recommended songs (excluding the input song itself)
    recommended = genre_subset.iloc[indices[0]]
    recommended = recommended[recommended["track_id"] != track_id].head(k)

    return {
        "input_track": song_row.iloc[0]["track_name"],
        "predicted_genre": predicted_genre,
        "recommendations": recommended[["track_name", "artists", "track_genre"]].to_dict(orient="records")
    }, None


# --- Quick test ---
if __name__ == "__main__":
    # Test on a sample of 5 different songs, not just one
    sample_songs = data.sample(5, random_state=42)

    for _, row in sample_songs.iterrows():
        track_id = row["track_id"]
        actual_genre = row["track_genre"]

        result, error = get_recommendations(track_id, k=3)

        if error:
            print(error)
            continue

        match = "MATCH" if result["predicted_genre"] == actual_genre else "mismatch"
        print(f"\n{result['input_track']} | Actual: {actual_genre} | Predicted: {result['predicted_genre']} [{match}]")
        for track in result["recommendations"]:
            print(f"    -> {track['track_name']} by {track['artists']} ({track['track_genre']})")



