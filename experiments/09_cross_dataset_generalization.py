"""
09_cross_dataset_generalization.py

Directly answers the instructor's request: train under BOTH conditions
(real data, synthetic data matched to the real taxonomy/features), but
TEST ONLY on the real dataset in both cases - so any accuracy gap
reflects genuine overfitting/underfitting to synthetic patterns, not
different test sets.

Uses the ACTUAL real AudioBot data (spotify_tracks_broad_genre.csv - 13
broad genres, 83,490 real tracks) and the improved feature set from
08_algorithm_and_feature_comparison.py (log-duration, cyclical key,
popularity retained).

Methodology (leakage-safe):
  1. Split the REAL broad-genre data 80/20 (train/test, stratified).
  2. Compute per-genre feature statistics from the REAL TRAINING split
     only - the real test set is never touched during synthetic
     generation.
  3. Generate a SYNTHETIC dataset matching the real data's exact 13
     genre labels, engineered feature set, and per-genre sample counts.
  4. Train each algorithm once on real training data, once on synthetic
     data - both evaluated on the SAME real held-out test set.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

SEED = 42
rng = np.random.default_rng(SEED)

ENGINEERED_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_log",
    "key_sin", "key_cos", "mode", "explicit", "time_signature", "popularity",
]
RAW_NUMERIC = [  # features that get Gaussian-sampled directly
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_log", "popularity",
]


def load_real_data():
    data = pd.read_csv("../spotify_tracks_broad_genre.csv")
    data["explicit"] = data["explicit"].astype(int)
    data["duration_log"] = np.log1p(data["duration_ms"])
    data["key_sin"] = np.sin(2 * np.pi * data["key"] / 12)
    data["key_cos"] = np.cos(2 * np.pi * data["key"] / 12)
    return data


def compute_genre_profiles(train_df):
    profiles = {}
    for genre, group in train_df.groupby("broad_genre"):
        profiles[genre] = {
            feat: (group[feat].mean(), max(group[feat].std(), 1e-6))
            for feat in RAW_NUMERIC
        }
        # categorical-ish features: sample from real empirical distribution
        profiles[genre]["mode_p"] = group["mode"].mean()
        profiles[genre]["explicit_p"] = group["explicit"].mean()
        profiles[genre]["time_sig_dist"] = group["time_signature"].value_counts(normalize=True)
        profiles[genre]["key_sin_mean_std"] = (group["key_sin"].mean(), max(group["key_sin"].std(), 1e-6))
        profiles[genre]["key_cos_mean_std"] = (group["key_cos"].mean(), max(group["key_cos"].std(), 1e-6))
    return profiles


def generate_synthetic(profiles, counts_per_genre):
    rows = []
    for genre, profile in profiles.items():
        n = counts_per_genre[genre]
        for _ in range(n):
            row = {"broad_genre": genre}
            for feat in RAW_NUMERIC:
                mean, std = profile[feat]
                row[feat] = rng.normal(mean, std)
            row["mode"] = 1 if rng.random() < profile["mode_p"] else 0
            row["explicit"] = 1 if rng.random() < profile["explicit_p"] else 0
            ts_dist = profile["time_sig_dist"]
            row["time_signature"] = rng.choice(ts_dist.index, p=ts_dist.values)
            k_sin_m, k_sin_s = profile["key_sin_mean_std"]
            k_cos_m, k_cos_s = profile["key_cos_mean_std"]
            row["key_sin"] = np.clip(rng.normal(k_sin_m, k_sin_s), -1, 1)
            row["key_cos"] = np.clip(rng.normal(k_cos_m, k_cos_s), -1, 1)
            rows.append(row)
    synth = pd.DataFrame(rows)

    bounded_01 = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
    for col in bounded_01:
        synth[col] = synth[col].clip(0, 1)
    synth["tempo"] = synth["tempo"].clip(lower=0)
    synth["popularity"] = synth["popularity"].clip(0, 100)
    synth["duration_log"] = synth["duration_log"].clip(lower=0)

    return synth.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def train_eval(X_train, y_train, X_test, y_test, model_ctor, label, use_smote=True):
    if use_smote:
        class_counts = pd.Series(y_train).value_counts()
        sampling_strategy = {lab: 3000 for lab, count in class_counts.items() if count < 3000}
        if sampling_strategy:
            smote = SMOTE(random_state=SEED, k_neighbors=5, sampling_strategy=sampling_strategy)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = model_ctor()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  {label:60s} acc={acc*100:6.2f}%   macro-F1={f1:.4f}")
    return {"label": label, "accuracy": acc, "macro_f1": f1}


def main():
    real_data = load_real_data()
    real_train, real_test = train_test_split(
        real_data, test_size=0.2, random_state=SEED, stratify=real_data["broad_genre"]
    )
    print(f"Real train: {len(real_train)}  |  Real test (held out, used for ALL evaluations): {len(real_test)}\n")

    profiles = compute_genre_profiles(real_train)
    counts_per_genre = real_train["broad_genre"].value_counts().to_dict()
    synthetic_data = generate_synthetic(profiles, counts_per_genre)
    print(f"Synthetic data generated: {len(synthetic_data)} rows, matched to real 13-genre taxonomy\n")

    # Save the generated synthetic dataset itself for inspection/transparency -
    # otherwise it only exists in memory during this run
    synthetic_data.to_csv("synthetic_data_export.csv", index=False)
    print(f"Synthetic dataset saved to synthetic_data_export.csv for direct inspection\n")

    le = LabelEncoder()
    le.fit(real_data["broad_genre"])

    X_real_train = real_train[ENGINEERED_FEATURES]
    y_real_train = le.transform(real_train["broad_genre"])
    X_real_test = real_test[ENGINEERED_FEATURES]
    y_real_test = le.transform(real_test["broad_genre"])

    X_synth = synthetic_data[ENGINEERED_FEATURES]
    y_synth = le.transform(synthetic_data["broad_genre"])

    results = []
    print("=" * 78)
    print("RANDOM FOREST")
    print("=" * 78)
    rf_ctor = lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    results.append(train_eval(X_real_train, y_real_train, X_real_test, y_real_test, rf_ctor,
                               "Trained on REAL, tested on REAL"))
    results.append(train_eval(X_synth, y_synth, X_real_test, y_real_test, rf_ctor,
                               "Trained on SYNTHETIC (matched), tested on REAL", use_smote=False))

    print("\n" + "=" * 78)
    print("GRADIENT BOOSTING")
    print("=" * 78)
    gb_ctor = lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=150)
    results.append(train_eval(X_real_train, y_real_train, X_real_test, y_real_test, gb_ctor,
                               "Trained on REAL, tested on REAL"))
    results.append(train_eval(X_synth, y_synth, X_real_test, y_real_test, gb_ctor,
                               "Trained on SYNTHETIC (matched), tested on REAL", use_smote=False))

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(results_df.to_string(index=False))

    rf_gap = results[0]["accuracy"] - results[1]["accuracy"]
    gb_gap = results[2]["accuracy"] - results[3]["accuracy"]
    print(f"\nRandom Forest generalization gap: {rf_gap*100:.2f} points")
    print(f"Gradient Boosting generalization gap: {gb_gap*100:.2f} points")

    results_df.to_csv("cross_dataset_generalization_results.csv", index=False)
    print("\nSaved: cross_dataset_generalization_results.csv")


if __name__ == "__main__":
    main()