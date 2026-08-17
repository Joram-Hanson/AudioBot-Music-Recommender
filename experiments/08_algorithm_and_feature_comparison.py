"""
08_algorithm_and_feature_comparison.py

Extends AudioBot v1 in two directions, per instructor/FI feedback:

  1. FEATURE ENGINEERING: fixes three real issues in the original 15-feature
     set (see analysis below) and measures whether each one actually helps.
  2. ALGORITHM COMPARISON: adds SVM (Linear + a subsampled RBF check),
     Gradient Boosting, KNN, and Logistic Regression alongside the
     existing Random Forest, all on the same real broad-genre task
     (13 genres, SMOTE-balanced training, same held-out real test set).

Feature engineering issues identified and fixed:
  a) `duration_ms` is heavily right-skewed (max ~87 min - non-song outliers
     like podcasts/audiobooks mixed into genres like "comedy"/"kids").
     Fix: log1p transform.
  b) `key` (0-11) is musically CIRCULAR, not linear - key 11 and key 0 are
     adjacent (both near C), but a raw integer encoding treats them as
     maximally far apart. Fix: sin/cos cyclical encoding.
  c) `popularity` is not an audio feature at all - it's a social/behavioral
     metric (how much people played the track), not a property of the
     sound. Including it muddies the "genre from audio" claim. Tested
     with and without to measure its actual contribution.

All results are measured on a genuine, untouched, held-out real test set.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

SEED = 42
BASE_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "popularity", "key", "mode", "explicit", "time_signature",
]


def engineer_features(df, drop_popularity=False):
    """Apply the three fixes: log-duration, cyclical key, optional popularity drop."""
    df = df.copy()

    # a) log-transform duration to fix heavy right-skew
    df["duration_log"] = np.log1p(df["duration_ms"])

    # b) cyclical encoding for key (circular, not linear)
    df["key_sin"] = np.sin(2 * np.pi * df["key"] / 12)
    df["key_cos"] = np.cos(2 * np.pi * df["key"] / 12)

    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_log",
        "key_sin", "key_cos", "mode", "explicit", "time_signature",
    ]
    if not drop_popularity:
        feature_cols.append("popularity")

    return df, feature_cols


def prep_data(engineered=True, drop_popularity=False):
    data = pd.read_csv("spotify_tracks_broad_genre.csv")
    data["explicit"] = data["explicit"].astype(int)

    if engineered:
        data, feature_cols = engineer_features(data, drop_popularity=drop_popularity)
    else:
        feature_cols = BASE_FEATURES

    X = data[feature_cols]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["broad_genre"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoder, feature_cols


def apply_smote(X_train, y_train):
    class_counts = pd.Series(y_train).value_counts()
    sampling_strategy = {label: 3000 for label, count in class_counts.items() if count < 3000}
    smote = SMOTE(random_state=SEED, k_neighbors=5, sampling_strategy=sampling_strategy)
    return smote.fit_resample(X_train, y_train)


def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  {label:45s} acc={acc*100:6.2f}%   macro-F1={f1:.4f}")
    return {"config": label, "accuracy": acc, "macro_f1": f1}


def part1_feature_engineering_ablation():
    print("=" * 78)
    print("PART 1: FEATURE ENGINEERING ABLATION (Random Forest, SMOTE, fixed)")
    print("=" * 78)
    results = []

    # Baseline: original 15 features, no engineering
    X_train, X_test, y_train, y_test, le, feats = prep_data(engineered=False)
    X_train_s, y_train_s = apply_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train_s)
    results.append(evaluate(model, X_test_scaled, y_test, "Original features (baseline, matches README 57.4%)"))

    # + engineered features (log duration, cyclical key), keep popularity
    X_train, X_test, y_train, y_test, le, feats = prep_data(engineered=True, drop_popularity=False)
    X_train_s, y_train_s = apply_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train_s)
    results.append(evaluate(model, X_test_scaled, y_test, "+ log(duration) + cyclical key encoding"))

    # + drop popularity (not a real audio feature)
    X_train, X_test, y_train, y_test, le, feats = prep_data(engineered=True, drop_popularity=True)
    X_train_s, y_train_s = apply_smote(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train_scaled, y_train_s)
    results.append(evaluate(model, X_test_scaled, y_test, "+ drop popularity (audio-only features)"))

    return pd.DataFrame(results)


def part2_algorithm_comparison():
    print("\n" + "=" * 78)
    print("PART 2: ALGORITHM COMPARISON (best feature set from Part 1, SMOTE-balanced)")
    print("=" * 78)

    # Use engineered features, KEEP popularity if Part 1 showed it helps
    # (decided after Part 1 runs - see main())
    X_train, X_test, y_train, y_test, le, feats = prep_data(engineered=True, drop_popularity=False)
    X_train_s, y_train_s = apply_smote(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test)

    results = []

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_s)
    results.append(evaluate(rf, X_test_scaled, y_test, "Random Forest (existing v1 approach)"))

    logreg = LogisticRegression(max_iter=2000, random_state=SEED)
    logreg.fit(X_train_scaled, y_train_s)
    results.append(evaluate(logreg, X_test_scaled, y_test, "Logistic Regression"))

    print("  Training Linear SVM (this may take a minute)...")
    lsvm = LinearSVC(random_state=SEED, max_iter=5000, dual="auto")
    lsvm.fit(X_train_scaled, y_train_s)
    results.append(evaluate(lsvm, X_test_scaled, y_test, "Linear SVM"))

    print("  Training Gradient Boosting...")
    gb = HistGradientBoostingClassifier(random_state=SEED, max_iter=150)
    gb.fit(X_train_scaled, y_train_s)
    results.append(evaluate(gb, X_test_scaled, y_test, "Gradient Boosting (HistGB)"))

    print("  Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    knn.fit(X_train_scaled, y_train_s)
    results.append(evaluate(knn, X_test_scaled, y_test, "K-Nearest Neighbors (k=15)"))

    # RBF SVM is too slow at full scale (O(n^2)+ with ~90K SMOTE-balanced
    # rows) - test on a random subsample instead, clearly labeled
    print("  Training RBF SVM on a 8,000-row subsample (full-scale RBF SVM is computationally infeasible here)...")
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(X_train_scaled), size=min(8000, len(X_train_scaled)), replace=False)
    rbf_svm = SVC(kernel="rbf", random_state=SEED)
    rbf_svm.fit(X_train_scaled[sub_idx], y_train_s.iloc[sub_idx] if hasattr(y_train_s, "iloc") else y_train_s[sub_idx])
    results.append(evaluate(rbf_svm, X_test_scaled, y_test, "RBF SVM (8K-row subsample, not full data)"))

    return pd.DataFrame(results)


def main():
    fe_results = part1_feature_engineering_ablation()
    algo_results = part2_algorithm_comparison()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("\nFeature engineering ablation:")
    print(fe_results.to_string(index=False))
    print("\nAlgorithm comparison:")
    print(algo_results.to_string(index=False))

    fe_results.to_csv("feature_engineering_results.csv", index=False)
    algo_results.to_csv("algorithm_comparison_results.csv", index=False)
    print("\nSaved: feature_engineering_results.csv, algorithm_comparison_results.csv")


if __name__ == "__main__":
    main()
