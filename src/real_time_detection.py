import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import time
import pyttsx3


MODEL_PATH = "exported_models/my_ssd_mobilenet_10signs_v2/saved_model"

LABELS = {
    1: "hello",
    2: "thanks",
    3: "yes",
    4: "no",
    5: "iloveyou",
    6: "please",
    7: "sorry",
    8: "help",
    9: "good",
    10: "clockit"
}

CONFIDENCE_THRESHOLD = 0.30

SMOOTHING_FRAMES = 10
REQUIRED_VOTES = 5

LABEL_LOCK_SECONDS = 1.0

MIN_BOX_AREA_RATIO = 0.01
MAX_BOX_AREA_RATIO = 0.45


def draw_panel(frame, stable_label, stable_score, sentence, fps):
    height, width, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Detected: {stable_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {stable_score * 100:.1f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 140, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    sentence_text = " ".join(sentence)

    if len(sentence_text) > 60:
        sentence_text = sentence_text[-60:]

    cv2.putText(
        frame,
        f"Sentence: {sentence_text}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def draw_controls(frame):
    height, width, _ = frame.shape

    cv2.rectangle(frame, (0, height - 35), (width, height), (0, 0, 0), -1)

    controls = "A: Add word | S: Speak | C: Clear | B: Backspace | Q: Quit"

    cv2.putText(
        frame,
        controls,
        (20, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


def is_valid_box(box, frame_width, frame_height):
    left, top, right, bottom = box

    box_width = right - left
    box_height = bottom - top

    if box_width <= 0 or box_height <= 0:
        return False

    frame_area = frame_width * frame_height
    box_area = box_width * box_height
    area_ratio = box_area / frame_area

    if area_ratio < MIN_BOX_AREA_RATIO:
        return False

    if area_ratio > MAX_BOX_AREA_RATIO:
        return False

    aspect_ratio = box_width / box_height

    if aspect_ratio < 0.30 or aspect_ratio > 3.20:
        return False

    return True


def get_best_detection(detect_fn, frame):
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))

    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections.items()
    }

    boxes = detections["detection_boxes"]
    classes = detections["detection_classes"].astype(np.int64)
    scores = detections["detection_scores"]

    best_label = None
    best_score = 0.0
    best_box = None

    for i in range(num_detections):
        class_id = int(classes[i])
        score = float(scores[i])

        if class_id not in LABELS:
            continue

        if score < CONFIDENCE_THRESHOLD:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        box = [left, top, right, bottom]

        if not is_valid_box(box, width, height):
            continue

        if score > best_score:
            best_score = score
            best_label = LABELS[class_id]
            best_box = box

    return best_label, best_box, best_score


def speak_sentence(engine, sentence):
    if not sentence:
        print("Sentence is empty. Press A to add a word first.")
        return

    text = " ".join(sentence)
    print(f"Speaking: {text}")

    engine.say(text)
    engine.runAndWait()


def main():
    print("Loading model...")
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully.")

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)
    box_history = deque(maxlen=SMOOTHING_FRAMES)
    score_history = deque(maxlen=SMOOTHING_FRAMES)

    sentence = []

    stable_label = "Detecting..."
    stable_score = 0.0
    stable_box = None

    locked_label = None
    lock_start_time = 0

    last_added_label = None
    last_added_time = 0

    prev_time = time.time()
    fps = 0.0

    print("Controls:")
    print("A = Add detected word")
    print("S = Speak sentence")
    print("C = Clear sentence")
    print("B = Remove last word")
    print("Q = Quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        detected_label, detected_box, detected_score = get_best_detection(detect_fn, frame)

        if detected_label is not None:
            prediction_history.append(detected_label)
            box_history.append(detected_box)
            score_history.append(detected_score)

            label_counts = Counter(prediction_history)
            most_common_label, vote_count = label_counts.most_common(1)[0]

            if vote_count >= REQUIRED_VOTES:
                now = time.time()

                if locked_label is None:
                    locked_label = most_common_label
                    lock_start_time = now

                if most_common_label == locked_label:
                    stable_label = most_common_label
                    stable_score = sum(score_history) / len(score_history)

                    matching_boxes = [
                        box for label, box in zip(prediction_history, box_history)
                        if label == stable_label and box is not None
                    ]

                    if matching_boxes:
                        stable_box = matching_boxes[-1]

                else:
                    if now - lock_start_time >= LABEL_LOCK_SECONDS:
                        locked_label = most_common_label
                        lock_start_time = now
                        stable_label = most_common_label
                        stable_score = sum(score_history) / len(score_history)

                        matching_boxes = [
                            box for label, box in zip(prediction_history, box_history)
                            if label == stable_label and box is not None
                        ]

                        if matching_boxes:
                            stable_box = matching_boxes[-1]

        else:
            prediction_history.clear()
            box_history.clear()
            score_history.clear()

            stable_label = "Detecting..."
            stable_score = 0.0
            stable_box = None
            locked_label = None

        if stable_box is not None and stable_label != "Detecting...":
            left, top, right, bottom = stable_box

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{stable_label}: {stable_score * 100:.1f}%",
                (left, max(top - 10, 135)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

        draw_panel(frame, stable_label, stable_score, sentence, fps)
        draw_controls(frame)

        cv2.imshow("AI-Powered Real-Time Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            if stable_label != "Detecting...":
                now = time.time()

                if stable_label != last_added_label or now - last_added_time > 1.5:
                    sentence.append(stable_label)
                    last_added_label = stable_label
                    last_added_time = now
                    print(f"Added word: {stable_label}")

        elif key == ord("c"):
            sentence.clear()
            print("Sentence cleared.")

        elif key == ord("b"):
            if sentence:
                removed = sentence.pop()
                print(f"Removed word: {removed}")

        elif key == ord("s"):
            speak_sentence(engine, sentence)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()