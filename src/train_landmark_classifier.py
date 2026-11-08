import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_DIR = "dataset"
MODEL_DIR = "landmark_model"

os.makedirs(MODEL_DIR, exist_ok=True)

mp_hands = mp.solutions.hands


def extract_landmarks(image):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []

        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z

        points = []

        for landmark in hand_landmarks.landmark:
            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z
            points.append([x, y, z])

        points = np.array(points)

        scale = np.max(np.linalg.norm(points, axis=1))

        if scale == 0:
            return None

        points = points / scale

        for point in points:
            landmarks.extend(point)

        return landmarks


def load_dataset():
    X = []
    y = []

    class_names = sorted([
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ])

    print("Classes found:")
    for class_name in class_names:
        print("-", class_name)

    for class_name in class_names:
        class_path = os.path.join(DATASET_DIR, class_name)

        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            landmarks = extract_landmarks(image)

            if landmarks is None:
                print(f"Skipped, no hand found: {image_path}")
                continue

            X.append(landmarks)
            y.append(class_name)

    return np.array(X), np.array(y)


def main():
    print("Loading dataset and extracting hand landmarks...")

    X, y = load_dataset()

    print()
    print("Total usable images:", len(X))

    if len(X) == 0:
        print("No usable hand landmarks found.")
        return

    print("Feature shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training classifier...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

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

    joblib.dump(model, os.path.join(MODEL_DIR, "sign_classifier.joblib"))

    labels = sorted(list(set(y)))
    joblib.dump(labels, os.path.join(MODEL_DIR, "labels.joblib"))

    print()
    print("Model saved successfully:")
    print(os.path.join(MODEL_DIR, "sign_classifier.joblib"))
    print(os.path.join(MODEL_DIR, "labels.joblib"))


if __name__ == "__main__":
    main()