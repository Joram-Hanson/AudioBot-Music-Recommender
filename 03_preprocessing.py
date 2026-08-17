import pandas as pd

data = pd.read_csv("spotify-tracks-dataset.csv")
data = data.drop(columns=["Unnamed: 0.1"], errors="ignore")

# --- 1. Drop rows with missing critical values ---
critical_cols = ["track_name", "artists", "track_genre"]
data = data.dropna(subset=critical_cols)

# --- 2. Remove duplicate track_ids, keeping the first occurrence ---
# (Same song can appear under multiple genres; we keep one label per song
# so the classifier isn't trained on contradictory genre labels for identical features)
before = data.shape[0]
data = data.drop_duplicates(subset="track_id", keep="first")
after = data.shape[0]
print(f"Removed {before - after} duplicate track_id rows")

# --- 3. Select the columns needed for modeling ---
metadata_cols = ["track_id", "track_name", "artists", "track_genre"]
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "popularity", "key", "mode", "explicit", "time_signature"
]

data_clean = data[metadata_cols + feature_cols].copy()

# --- 4. Save the cleaned dataset ---
data_clean.to_csv("spotify_tracks_clean.csv", index=False)

print("\nCleaned dataset shape:", data_clean.shape)
print("Remaining unique genres:", data_clean["track_genre"].nunique())
print("Saved to spotify_tracks_clean.csv")
print("\nGenre counts after cleaning (smallest 10):\n", data_clean["track_genre"].value_counts().tail(10))
print("\nGenre counts after cleaning (largest 10):\n", data_clean["track_genre"].value_counts().head(10))