import cv2
import csv
import os
import time
import argparse
import numpy as np
import mediapipe as mp


OUTPUT_FILE = "landmark_model/live_landmarks.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
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

    return points.flatten().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Sign label name")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to collect")
    args = parser.parse_args()

    label = args.label
    target_samples = args.samples

    os.makedirs("landmark_model", exist_ok=True)

    file_exists = os.path.exists(OUTPUT_FILE)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    collected = 0
    last_saved_time = 0

    print()
    print(f"Collecting live landmarks for: {label}")
    print(f"Target samples: {target_samples}")
    print("Press Q to quit early.")
    print("Keep your hand visible. Slightly vary angle, distance, and position.")
    print()

    with open(OUTPUT_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"]
            for i in range(63):
                header.append(f"feature_{i}")
            writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        ) as hands:

            while collected < target_samples:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to read frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb_frame)

                status_text = "No hand detected"

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    features = extract_landmarks(hand_landmarks)

                    if features is not None:
                        current_time = time.time()

                        if current_time - last_saved_time > 0.08:
                            writer.writerow([label] + features)
                            collected += 1
                            last_saved_time = current_time

                        status_text = f"Collecting {label}: {collected}/{target_samples}"

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    status_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Move hand slightly. Press Q to quit.",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Collect Live Landmark Data", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"Finished collecting {collected} samples for {label}.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()