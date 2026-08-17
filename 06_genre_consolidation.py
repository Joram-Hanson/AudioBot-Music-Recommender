import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your cleaned dataset
data = pd.read_csv("spotify_tracks_clean.csv")

# --- Genre consolidation mapping: 113 genres -> 13 broad categories ---
# Mood/activity tags (sad, happy, chill, party, study, sleep, romance) are excluded
# since they represent a different taxonomy dimension than genre.
genre_mapping = {
    # Rock
    "alt-rock": "rock", "alternative": "rock", "british": "rock", "emo": "rock",
    "garage": "rock", "goth": "rock", "grunge": "rock", "hard-rock": "rock",
    "indie": "rock", "j-rock": "rock", "power-pop": "rock", "psych-rock": "rock",
    "punk": "rock", "punk-rock": "rock", "rock": "rock", "rock-n-roll": "rock",
    "rockabilly": "rock",

    # Metal
    "black-metal": "metal", "death-metal": "metal", "grindcore": "metal",
    "hardcore": "metal", "heavy-metal": "metal", "industrial": "metal",
    "metal": "metal", "metalcore": "metal",

    # Electronic/Dance
    "breakbeat": "electronic", "chicago-house": "electronic", "club": "electronic",
    "dance": "electronic", "deep-house": "electronic", "detroit-techno": "electronic",
    "disco": "electronic", "drum-and-bass": "electronic", "dubstep": "electronic",
    "edm": "electronic", "electro": "electronic", "electronic": "electronic",
    "hardstyle": "electronic", "house": "electronic", "idm": "electronic",
    "j-dance": "electronic", "minimal-techno": "electronic", "progressive-house": "electronic",
    "synth-pop": "electronic", "techno": "electronic", "trance": "electronic",

    # Hip-Hop / R&B / Funk / Soul
    "funk": "hiphop_rnb", "groove": "hiphop_rnb", "hip-hop": "hiphop_rnb",
    "r-n-b": "hiphop_rnb", "soul": "hiphop_rnb",

    # Pop
    "cantopop": "pop", "indie-pop": "pop", "j-idol": "pop", "j-pop": "pop",
    "k-pop": "pop", "mandopop": "pop", "pop": "pop", "pop-film": "pop",

    # Latin
    "brazil": "latin", "forro": "latin", "latin": "latin", "latino": "latin",
    "mpb": "latin", "pagode": "latin", "reggaeton": "latin", "salsa": "latin",
    "samba": "latin", "sertanejo": "latin", "spanish": "latin", "tango": "latin",

    # Jazz / Blues
    "blues": "jazz_blues", "jazz": "jazz_blues",

    # Classical / Instrumental
    "acoustic": "classical_instrumental", "ambient": "classical_instrumental",
    "classical": "classical_instrumental", "guitar": "classical_instrumental",
    "new-age": "classical_instrumental", "opera": "classical_instrumental",
    "piano": "classical_instrumental", "show-tunes": "classical_instrumental",

    # Folk / Country
    "bluegrass": "folk_country", "country": "folk_country", "folk": "folk_country",
    "honky-tonk": "folk_country", "singer-songwriter": "folk_country",

    # Reggae / Dub
    "dancehall": "reggae_dub", "dub": "reggae_dub", "reggae": "reggae_dub",
    "ska": "reggae_dub", "trip-hop": "reggae_dub",

    # World / Regional
    "afrobeat": "world_regional", "french": "world_regional", "german": "world_regional",
    "indian": "world_regional", "iranian": "world_regional", "malay": "world_regional",
    "swedish": "world_regional", "turkish": "world_regional", "world-music": "world_regional",

    # Kids / Family
    "anime": "kids_family", "children": "kids_family", "comedy": "kids_family",
    "disney": "kids_family", "kids": "kids_family",

    # Gospel
    "gospel": "gospel",
}

# Excluded mood/activity tags - not real genres
excluded_genres = ["sad", "happy", "chill", "party", "study", "sleep", "romance"]

print(f"Original dataset size: {len(data)}")
data_filtered = data[~data["track_genre"].isin(excluded_genres)].copy()
print(f"After removing mood/activity tags: {len(data_filtered)} ({len(data) - len(data_filtered)} rows removed)")

# Apply the mapping
data_filtered["broad_genre"] = data_filtered["track_genre"].map(genre_mapping)

# Check for any unmapped genres (safety check)
unmapped = data_filtered[data_filtered["broad_genre"].isna()]["track_genre"].unique()
if len(unmapped) > 0:
    print(f"WARNING: Unmapped genres found: {unmapped}")
else:
    print("All genres successfully mapped.")

print(f"\nBroad genre distribution:\n{data_filtered['broad_genre'].value_counts()}")

# Save this new dataset with the broad_genre column
data_filtered.to_csv("spotify_tracks_broad_genre.csv", index=False)
print("\nSaved to spotify_tracks_broad_genre.csv")