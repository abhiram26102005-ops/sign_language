import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
import pyttsx3

from collections import deque, Counter


MODEL_PATH = "landmark_model/live_sign_classifier.joblib"

CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_FRAMES = 8
REQUIRED_VOTES = 4

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CLICK_ACTION = None
BUTTONS = {}


def mouse_callback(event, x, y, flags, param):
    global CLICK_ACTION

    if event == cv2.EVENT_LBUTTONDOWN:
        for action, box in BUTTONS.items():
            x1, y1, x2, y2 = box

            if x1 <= x <= x2 and y1 <= y <= y2:
                CLICK_ACTION = action
                print("Clicked:", action)
                break


def extract_landmarks_from_results(hand_landmarks):
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

    return points.flatten().reshape(1, -1)


def get_hand_box(hand_landmarks, frame_width, frame_height):
    x_values = []
    y_values = []

    for landmark in hand_landmarks.landmark:
        x_values.append(int(landmark.x * frame_width))
        y_values.append(int(landmark.y * frame_height))

    padding = 30

    left = max(min(x_values) - padding, 0)
    top = max(min(y_values) - padding, 0)
    right = min(max(x_values) + padding, frame_width)
    bottom = min(max(y_values) + padding, frame_height)

    return left, top, right, bottom


def draw_panel(frame, stable_label, stable_score, sentence, fps, last_action):
    height, width, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (width, 155), (0, 0, 0), -1)

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

    if len(sentence_text) > 65:
        sentence_text = sentence_text[-65:]

    cv2.putText(
        frame,
        f"Sentence: {sentence_text}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Last action: {last_action}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


def draw_top_predictions(frame, top_predictions):
    y = 185

    cv2.putText(
        frame,
        "Top predictions:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    y += 25

    for label, score in top_predictions:
        cv2.putText(
            frame,
            f"{label}: {score * 100:.1f}%",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 25


def draw_button(frame, text, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]

    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


def draw_controls(frame):
    global BUTTONS

    height, width, _ = frame.shape

    cv2.rectangle(frame, (0, height - 85), (width, height), (0, 0, 0), -1)

    y1 = height - 68
    y2 = height - 22

    button_width = 120
    gap = 10
    start_x = 15

    BUTTONS = {
        "add": (
            start_x,
            y1,
            start_x + button_width,
            y2
        ),
        "speak": (
            start_x + 1 * (button_width + gap),
            y1,
            start_x + 1 * (button_width + gap) + button_width,
            y2
        ),
        "clear": (
            start_x + 2 * (button_width + gap),
            y1,
            start_x + 2 * (button_width + gap) + button_width,
            y2
        ),
        "back": (
            start_x + 3 * (button_width + gap),
            y1,
            start_x + 3 * (button_width + gap) + button_width,
            y2
        ),
        "quit": (
            start_x + 4 * (button_width + gap),
            y1,
            start_x + 4 * (button_width + gap) + button_width,
            y2
        ),
    }

    draw_button(frame, "ADD WORD", *BUTTONS["add"])
    draw_button(frame, "SPEAK", *BUTTONS["speak"])
    draw_button(frame, "CLEAR", *BUTTONS["clear"])
    draw_button(frame, "BACK", *BUTTONS["back"])
    draw_button(frame, "QUIT", *BUTTONS["quit"])

    cv2.putText(
        frame,
        "Keyboard backup: A=Add | S=Speak | C=Clear | B=Back | Q=Quit",
        (15, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )


def add_word(sentence, stable_label):
    if stable_label == "Detecting...":
        print("No stable sign detected. Show a sign first.")
        return "No stable sign to add"

    sentence.append(stable_label)
    print("Added word:", stable_label)
    return f"Added: {stable_label}"


def convert_labels_to_speech(text):
    replacements = {
        "hello": "hello",
        "thanks": "thanks",
        "yes": "yes",
        "no": "no",
        "iloveyou": "I love you",
        "please": "please",
        "sorry": "sorry",
        "help": "help",
        "good": "good",
        "clockit": "clock it"
    }

    words = text.split()
    spoken_words = []

    for word in words:
        spoken_words.append(replacements.get(word, word))

    return " ".join(spoken_words)


def speak_text(text):
    if not text.strip():
        print("Nothing to speak.")
        return "Nothing to speak"

    speech_text = convert_labels_to_speech(text)

    print("Original text:", text)
    print("Speaking:", speech_text)

    try:
        fresh_engine = pyttsx3.init()
        fresh_engine.setProperty("rate", 130)
        fresh_engine.setProperty("volume", 1.0)
        fresh_engine.say(speech_text)
        fresh_engine.runAndWait()
        fresh_engine.stop()

        return f"Spoke: {speech_text}"

    except Exception as error:
        print("Speech error:", error)
        return f"Speech error: {error}"


def main():
    global CLICK_ACTION

    print("Loading live landmark classifier...")

    model = joblib.load(MODEL_PATH)
    class_names = list(model.classes_)

    print("Model loaded successfully.")
    print("Classes:", class_names)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "AI-Powered Real-Time Sign Language Recognition"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 950, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)
    score_history = deque(maxlen=SMOOTHING_FRAMES)

    sentence = []

    stable_label = "Detecting..."
    stable_score = 0.0
    stable_box = None

    last_action = "Ready"

    prev_time = time.time()
    fps = 0.0

    print()
    print("Controls:")
    print("Click ADD WORD button = add current detected sign")
    print("Click SPEAK button = speak sentence")
    print("Click CLEAR button = clear sentence")
    print("Click BACK button = remove last word")
    print("Click QUIT button = quit")
    print()
    print("Keyboard backup: A=Add, S=Speak, C=Clear, B=Back, Q=Quit")
    print()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            top_predictions = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                features = extract_landmarks_from_results(hand_landmarks)

                if features is not None:
                    probabilities = model.predict_proba(features)[0]
                    sorted_indexes = np.argsort(probabilities)[::-1]

                    for index in sorted_indexes[:3]:
                        label = class_names[index]
                        score = float(probabilities[index])
                        top_predictions.append((label, score))

                    best_index = int(sorted_indexes[0])
                    detected_label = class_names[best_index]
                    detected_score = float(probabilities[best_index])

                    if detected_score >= CONFIDENCE_THRESHOLD:
                        prediction_history.append(detected_label)
                        score_history.append(detected_score)

                        most_common_label, vote_count = Counter(prediction_history).most_common(1)[0]

                        if vote_count >= REQUIRED_VOTES:
                            stable_label = most_common_label
                            stable_score = detected_score
                            stable_box = get_hand_box(hand_landmarks, width, height)

                    else:
                        prediction_history.clear()
                        score_history.clear()
                        stable_label = "Detecting..."
                        stable_score = 0.0
                        stable_box = None

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            else:
                prediction_history.clear()
                score_history.clear()
                stable_label = "Detecting..."
                stable_score = 0.0
                stable_box = None

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
                    (left, max(top - 10, 165)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

            draw_panel(frame, stable_label, stable_score, sentence, fps, last_action)
            draw_top_predictions(frame, top_predictions)
            draw_controls(frame)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            action = CLICK_ACTION
            CLICK_ACTION = None

            if key in [ord("a"), ord("A"), 32]:
                action = "add"
            elif key in [ord("s"), ord("S"), 13]:
                action = "speak"
            elif key in [ord("c"), ord("C")]:
                action = "clear"
            elif key in [ord("b"), ord("B"), 8]:
                action = "back"
            elif key in [ord("q"), ord("Q")]:
                action = "quit"

            if action == "add":
                last_action = add_word(sentence, stable_label)

            elif action == "speak":
                if sentence:
                    last_action = speak_text(" ".join(sentence))
                elif stable_label != "Detecting...":
                    last_action = speak_text(stable_label)
                else:
                    last_action = "No sentence/sign to speak"
                    print(last_action)

            elif action == "clear":
                sentence.clear()
                last_action = "Sentence cleared"
                print(last_action)

            elif action == "back":
                if sentence:
                    removed = sentence.pop()
                    last_action = f"Removed: {removed}"
                    print(last_action)
                else:
                    last_action = "Sentence already empty"
                    print(last_action)

            elif action == "quit":
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()