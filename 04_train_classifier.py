import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.metrics import classification_report

# Load your cleaned dataset
data = pd.read_csv("spotify_tracks_clean.csv")

# Define features (X) and target (y)
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

X = data[feature_cols]
y = data["track_genre"]

# Encode genre labels into numbers (Random Forest needs numeric labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Number of genres:", len(label_encoder.classes_))
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y_encoded.shape)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])



# Train the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=20,           # limits how deep each tree can grow
    min_samples_leaf=5,     # requires at least 5 samples per leaf node
    random_state=42,       # reproducibility
    n_jobs=-1,              # use all CPU cores to speed up training
    class_weight="balanced" # equal guessing of songs for each genre (helps with class imbalance)
)

print("\nTraining Random Forest classifier...")
model.fit(X_train, y_train)
print("Training complete.")

# Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Save the trained model and label encoder for later use (Flask app, KNN, etc.)
joblib.dump(model, "genre_classifier.pkl", compress=3)
joblib.dump(label_encoder, "label_encoder.pkl")
print("\nModel and label encoder saved.")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
print(report)


#ADDING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Save a visual heatmap (full size, for reference/report appendix)
plt.figure(figsize=(20, 20))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Genre Classification")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix heatmap saved as confusion_matrix.png")

# --- Find the top confused genre PAIRS (most useful insight) ---
genre_names = label_encoder.classes_
confused_pairs = []

for i in range(len(genre_names)):
    for j in range(len(genre_names)):
        if i != j and cm[i][j] > 0:
            confused_pairs.append((genre_names[i], genre_names[j], cm[i][j]))

# Sort by how often the confusion happened, descending
confused_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop 15 most confused genre pairs (Actual -> Predicted, Count):")
for actual, predicted, count in confused_pairs[:15]:
    print(f"  {actual} -> {predicted}: {count} times")
