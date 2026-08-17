import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the dataset (includes both the fine-grained track_genre and the
# consolidated broad_genre columns, plus all 15 features the broad classifier
# was trained on).
data = pd.read_csv("spotify_tracks_broad_genre.csv")
data["explicit"] = data["explicit"].astype(int)  # match training: True/False -> 1/0

# Load the trained classifier and label encoder - using the improved
# broad-genre, SMOTE-balanced model (57.4% test accuracy), not the original
# 113-raw-genre model (32.7%).
classifier = joblib.load("genre_classifier_broad_smote.pkl")
label_encoder = joblib.load("label_encoder_broad.pkl")

# Same feature columns used for training the broad-genre classifier
# (see 07_train_broad_classifier.py)
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "popularity", "key", "mode", "explicit", "time_signature",
]

def get_recommendations(track_id, k=5):
    """
    Given a track_id, predict its broad genre using the classifier (for
    comparison/demonstration - see note below), then find the k most similar
    songs within its real, fine-grained genre using KNN.
    """
    # Find the input song's row
    song_row = data[data["track_id"] == track_id]
    if song_row.empty:
        return None, f"Track ID {track_id} not found in dataset."

    song_features = song_row[feature_cols]

    # Step 1: Use the dataset's known genre labels directly (stable, always
    # correct) rather than the classifier's prediction, to decide what to
    # recommend. We pool by the FINE-GRAINED genre (track_genre) rather than
    # the broad genre, so recommendations stay tightly matched instead of
    # being pooled across a much larger, more general bucket.
    actual_genre = song_row.iloc[0]["track_genre"]
    actual_broad_genre = song_row.iloc[0]["broad_genre"]

    # Step 1b: Also run the classifier, for comparison/demonstration purposes
    # only - this prediction never drives what gets recommended (see Step 2).
    # The classifier predicts a BROAD genre, so we compare it against the
    # song's real broad genre (actual_broad_genre), not its fine-grained one.
    predicted_genre_encoded = classifier.predict(song_features)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_encoded])[0]

    # Step 2: Filter dataset using the ACTUAL fine-grained genre (stable),
    # not the predicted one.
    genre_subset = data[data["track_genre"] == actual_genre].reset_index(drop=True)

    # Step 2.5: Remove near-duplicate songs (same title + artist, different track_id)
    genre_subset = genre_subset.drop_duplicates(subset=["track_name", "artists"], keep="first").reset_index(drop=True)

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
        "actual_genre": actual_genre,
        "actual_broad_genre": actual_broad_genre,
        "predicted_genre": predicted_genre,
        "recommendations": recommended[["track_id", "track_name", "artists", "track_genre"]].to_dict(orient="records")
    }, None


# --- Quick test ---
if __name__ == "__main__":
    # Test on a sample of 5 different songs, not just one
    sample_songs = data.sample(5, random_state=42)

    for _, row in sample_songs.iterrows():
        track_id = row["track_id"]

        result, error = get_recommendations(track_id, k=3)

        if error:
            print(error)
            continue

        match = "MATCH" if result["predicted_genre"] == result["actual_broad_genre"] else "mismatch"
        print(f"\n{result['input_track']} | Actual genre: {result['actual_genre']} (broad: {result['actual_broad_genre']}) "
              f"| Predicted broad genre: {result['predicted_genre']} [{match}]")
        for track in result["recommendations"]:
            print(f"    -> {track['track_name']} by {track['artists']} ({track['track_genre']})")