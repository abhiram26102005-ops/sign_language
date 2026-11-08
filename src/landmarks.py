import math


LANDMARK_COUNT = 21
FEATURE_COUNT = LANDMARK_COUNT * 3


def normalize_landmarks(landmarks):
    """Return 63 wrist-relative, scale-normalized coordinates, or None."""
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(f"Expected {LANDMARK_COUNT} hand landmarks.")

    wrist = landmarks[0]
    points = [
        (float(point[0]) - float(wrist[0]),
         float(point[1]) - float(wrist[1]),
         float(point[2]) - float(wrist[2]))
        for point in landmarks
    ]
    scale = max(math.sqrt(x * x + y * y + z * z) for x, y, z in points)

    if not math.isfinite(scale) or scale == 0:
        return None

    return [coordinate / scale for point in points for coordinate in point]


def extract_hand_landmarks(hand_landmarks):
    points = [
        (landmark.x, landmark.y, landmark.z)
        for landmark in hand_landmarks.landmark
    ]
    return normalize_landmarks(points)


def predict_landmarks(model, features):
    """Predict a label and its probability from one normalized landmark row."""
    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} landmark features.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("The loaded model must expose classes_ and predict_proba().")

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is missing. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    probabilities = model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
    best_index = int(probabilities.argmax())
    return str(model.classes_[best_index]), float(probabilities[best_index]), probabilities
