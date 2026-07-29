## Setup
````bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib
````

**Update "How to Run"** 
````markdown
## How to Run
1. `python 01_load_explore.py` — inspect the raw dataset
2. `python 02_data_quality.py` — check missing values, duplicates, genre distribution
3. `python 03_preprocessing.py` — clean the data (removes duplicate track_ids, drops missing values) and save `spotify_tracks_clean.csv`
````

````markdown
## Known Data Limitations
After deduplication, genre distribution is imbalanced (ranging from 74 to 1000 songs 
per genre). This is discussed further in our final report's evaluation and limitations sections.
````

## Full updated README to use now

````markdown
# AudioBot — AI-Powered Music Recommendation System

## Overview
AudioBot recommends songs based on audio-feature similarity within a predicted genre, 
rather than relying on historical listening behavior.

## System Architecture
The system follows a two-stage pipeline: a user's selected song is first classified 
by genre, then matched against other songs in that genre using audio-feature similarity.

![Architecture Diagram](music_recommender_architecture.png)

1. User selects a song
2. System looks up its audio features (tempo, energy, danceability, etc.)
3. A Random Forest classifier predicts the song's genre
4. KNN finds the closest-matching songs within that genre
5. Recommended songs are displayed in the web app

## Dataset
Spotify Tracks Dataset (Kaggle), by Yash Dev:
https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset

## IMPORTANT NOTICE
Download the CSV and place it in the project root as `spotify-tracks-dataset.csv` 
before running the scripts.



## Team
- Joram Elvin Akuerteh Hanson — Project coordination, data preprocessing, feature engineering
- Joshua Elvis Ataa-Oko Hanson — Dataset collection, preprocessing, documentation
- Edward Anokye Junior — KNN recommendation implementation, system integration
- Adjoa Afriyie Adusei — UI development, testing, evaluation, report writing
````

