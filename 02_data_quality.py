import pandas as pd
data = pd.read_csv("spotify-tracks-dataset.csv")
data = data.drop(columns=["Unnamed: 0.1"], errors = "ignore")

#Missing values per column
print("Missing values per column:\n", data.isnull().sum())

#Duplicate rows
print("Number of duplicate rows:", data.duplicated().sum())

#Duplicate tracks by track_id (same song listened more than once)
print("Number of duplicate tracks by track_id:", data.duplicated(subset=["track_id"]).sum())

#Genre distribution - how many songs per genre
genre_counts = data["track_genre"].value_counts()
print("\nNumber of unique genres:", data["track_genre"].nunique())
print("\nGenre counts (top 10):\n", genre_counts.head(10))
print("\nGenre counts (bottom 10 — smallest genres):\n", genre_counts.tail(10))