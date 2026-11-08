import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_FILE = "landmark_model/live_landmarks.csv"
MODEL_DIR = "landmark_model"

MODEL_PATH = os.path.join(MODEL_DIR, "live_sign_classifier.joblib")


def mirror_landmarks(X):
    """
    Each sample has 63 features:
    x0, y0, z0, x1, y1, z1, ... x20, y20, z20

    To simulate the opposite hand, we mirror all x values.
    """
    X_mirrored = X.copy()

    # x values are at indexes 0, 3, 6, 9, ...
    X_mirrored[:, 0::3] *= -1

    return X_mirrored


def main():
    if not os.path.exists(DATA_FILE):
        print("Error: live_landmarks.csv not found.")
        print("Collect live samples first.")
        return

    df = pd.read_csv(DATA_FILE)
    df = df.dropna()

    print("Loaded original samples:", len(df))
    print()
    print("Original class counts:")
    print(df["label"].value_counts())

    y_original = df["label"].astype(str).to_numpy()
    X_original = df.drop("label", axis=1).astype(np.float32).to_numpy()

    print()
    print("Original feature shape:", X_original.shape)

    if len(set(y_original)) < 2:
        print("Error: Need at least 2 classes.")
        return

    class_counts = pd.Series(y_original).value_counts()

    if class_counts.min() < 2:
        print("Error: Every class needs at least 2 samples.")
        print(class_counts)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y_original
    )

    X_train_mirrored = mirror_landmarks(X_train)
    X_train = np.vstack([X_train, X_train_mirrored])
    y_train = np.concatenate([y_train, y_train.copy()])

    print()
    print("Training samples after mirror augmentation:", len(X_train))
    print("Held-out original samples:", len(X_test))

    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
        max_depth=None
    )

    et = ExtraTreesClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced",
        max_depth=None
    )

    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et)
        ],
        voting="soft"
    )

    print()
    print("Training live landmark classifier with mirror augmentation...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)

    print()
    print("Saved:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()