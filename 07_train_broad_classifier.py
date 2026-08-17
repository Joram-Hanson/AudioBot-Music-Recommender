import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

data_filtered = pd.read_csv("spotify_tracks_broad_genre.csv")

data_filtered["explicit"] = data_filtered["explicit"].astype(int)  # converts True/False to 1/0

feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "popularity", "key", "mode", "explicit", "time_signature"
]

X = data_filtered[feature_cols]
y = data_filtered["broad_genre"]

label_encoder_broad = LabelEncoder()
y_encoded = label_encoder_broad.fit_transform(y)

# Split FIRST, before any SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Before SMOTE - Training set size: {X_train.shape[0]}")
print(f"Class distribution before SMOTE:\n{pd.Series(y_train).value_counts()}")

# Apply SMOTE only to the training data
# Cap SMOTE oversampling to avoid extreme inflation of very small classes
class_counts = pd.Series(y_train).value_counts()
sampling_strategy = {}
for label, count in class_counts.items():
    if count < 3000:
        sampling_strategy[label] = 3000  # oversample small classes up to 3000, not all the way to 13280
    # else: leave larger classes unchanged (don't include in dict = no change)

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nSampling strategy used: {sampling_strategy}")

print(f"\nAfter SMOTE - Training set size: {X_train_resampled.shape[0]}")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

# Train WITHOUT class_weight this time, since SMOTE already balances the data
model_smote = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest with SMOTE-balanced data...")
model_smote.fit(X_train_resampled, y_train_resampled)
print("Training complete.")

# Evaluate on the ORIGINAL (non-SMOTE) test set - this is important for honest evaluation
y_pred = model_smote.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (SMOTE-trained model, evaluated on real data): {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder_broad.classes_, zero_division=0)
print(report)

joblib.dump(model_smote, "genre_classifier_broad_smote.pkl", compress=3)
joblib.dump(label_encoder_broad, "label_encoder_broad.pkl")
print("\nModel and label encoder saved.")