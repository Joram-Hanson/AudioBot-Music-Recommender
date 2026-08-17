# AudioBot — AI-Powered Music Recommendation System

CS 254 (Introduction to Artificial Intelligence) Final Project — content-based
music recommendation system built on real Spotify track data.

## Overview
AudioBot recommends songs based on audio-feature similarity within a song's
genre, rather than relying on other users' listening history (which this
dataset does not include). A Random Forest classifier is trained and
evaluated as the project's core supervised-learning component; the live
recommendation feature itself is driven by each song's real, known genre
label plus a K-Nearest Neighbors (KNN) similarity search.

## System Architecture
1. User selects a song from the app.
2. The app looks up that song's audio features (tempo, energy, danceability,
   etc.) and its real genre label.
3. The Random Forest classifier also predicts the song's genre, shown
   alongside the real label for comparison — this prediction does **not**
   drive what gets recommended (see "Why the classifier's accuracy doesn't
   limit recommendation quality" below).
4. KNN finds the closest-matching songs within the song's real genre, using
   scaled audio-feature distance.
5. Recommended songs are displayed in the web app.

![Architecture Diagram](music_recommender_architecture.png)

## Dataset
Spotify Tracks Dataset (Kaggle), by Yash Dev:
https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset

Download the CSV and place it in the project root as
`spotify-tracks-dataset.csv` before running scripts 01-03 (this raw file is
not included in the repo — see `.gitignore`). The derived, cleaned datasets
(`spotify_tracks_clean.csv`, `spotify_tracks_broad_genre.csv`) **are**
included in this repo, so you can skip straight to training or running the
app without re-downloading anything, if you just want to see it work.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## How to Run

### Quick path — just run the app
The trained models and processed data are already included in this repo, so
you can go straight to:
```bash
python app.py
```
Then open the URL Flask prints (typically http://127.0.0.1:5000) in your
browser, pick a genre and a song, and view its recommendations.

### Full pipeline — reproduce everything from scratch
If you want to regenerate the data and retrain the models yourself (this
requires downloading the raw dataset first — see "Dataset" above):

1. `python 01_load_explore.py` — inspect the raw dataset (shape, columns,
   dtypes, sample rows).
2. `python 02_data_quality.py` — check missing values, duplicate rows, and
   the raw genre distribution.
3. `python 03_preprocessing.py` — clean the data (drops rows missing
   critical fields, removes duplicate track_ids) and save
   `spotify_tracks_clean.csv`.
4. `python 04_train_classifier.py` — train the baseline Random Forest on all
   113 raw genres. Saves `genre_classifier.pkl`, `label_encoder.pkl`, and a
   confusion-matrix heatmap (`confusion_matrix.png`). Real test accuracy:
   **32.7%**.
5. `python 06_genre_consolidation.py` — consolidate the 113 raw genres into
   13 broader, more acoustically coherent genres (drops non-genre mood/
   activity tags like "sad," "chill," "study"). Saves
   `spotify_tracks_broad_genre.csv`.
6. `python 07_train_broad_classifier.py` — train the improved Random Forest
   on the 13 consolidated genres, using SMOTE to balance underrepresented
   classes. Saves `genre_classifier_broad_smote.pkl` and
   `label_encoder_broad.pkl`. Real test accuracy: **57.4%**.
7. `python app.py` — launch the web app (uses the broad-genre model from
   step 6, via `knn_recommender.py`).

## Evaluation Results
| Model | Genres | Test Accuracy |
|---|---|---|
| Raw genre classifier (`04_train_classifier.py`) | 113 | 32.7% |
| Broad-genre classifier + SMOTE (`07_train_broad_classifier.py`) | 13 | 57.4% |

Both figures are measured on a held-out test set the model never saw during
training. Full evaluation detail, five diagnostic experiments explaining why
accuracy plateaus below our 85% target, and honest discussion of limitations
are in the accompanying Final Report.

## Why the classifier's accuracy doesn't limit recommendation quality
The live app's recommendation step uses each song's **real, already-known
genre label** to select which songs to compare against — not the
classifier's prediction. The classifier runs separately, purely for
comparison/demonstration. This means recommendation quality is unaffected by
the classifier's 57.4% ceiling; that figure evaluates a separate, required
component of the project (the supervised learning technique itself), not the
thing a user experiences when they click through the app.

## Known Data Limitations
- After deduplication, genre distribution is imbalanced (ranging from about
  74 to 1000+ songs per genre), addressed via genre consolidation and SMOTE.
- A meaningful portion of the raw dataset's genre tags describe language,
  mood, or listening context (e.g. "german," "kids," "study") rather than
  musical style, which caps achievable classification accuracy regardless of
  algorithm — discussed in detail in the Final Report.

## Project Structure
```
01_load_explore.py            - initial data inspection
02_data_quality.py            - missing values, duplicates, genre distribution
03_preprocessing.py           - cleaning -> spotify_tracks_clean.csv
04_train_classifier.py        - baseline classifier (113 genres, 32.7%)
06_genre_consolidation.py     - 113 -> 13 genre consolidation
07_train_broad_classifier.py  - improved classifier (13 genres + SMOTE, 57.4%)
knn_recommender.py            - recommendation logic (genre lookup + KNN)
app.py                        - Flask web app
templates/, static/           - web app front-end
```

## Team
- Joram Elvin Akuerteh Hanson — Project coordination, data preprocessing, feature engineering
- Joshua Elvis Ataa-Oko Hanson — Dataset collection, preprocessing, documentation
- Edward Anokye Junior — KNN recommendation implementation, system integration
- Adjoa Afriyie Adusei — UI development, testing, evaluation, report writing
