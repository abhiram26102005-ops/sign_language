import cv2
import numpy as np
import tensorflow as tf


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

MIN_DISPLAY_SCORE = 0.05


def main():
    print("Loading model...")
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        height, width, _ = frame.shape

        input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
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

        top_lines = []

        for i in range(min(5, num_detections)):
            class_id = int(classes[i])
            score = float(scores[i])
            label = LABELS.get(class_id, "unknown")
            top_lines.append(f"{i + 1}. {label}: {score * 100:.1f}%")

        best_class = int(classes[0])
        best_score = float(scores[0])
        best_label = LABELS.get(best_class, "unknown")
        best_box = boxes[0]

        if best_score >= MIN_DISPLAY_SCORE:
            ymin, xmin, ymax, xmax = best_box

            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{best_label}: {best_score * 100:.1f}%",
                (left, max(top - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.rectangle(frame, (0, 0), (width, 170), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "RAW MODEL OUTPUT - PRESS Q TO QUIT",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        y = 65
        for line in top_lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            y += 25

        cv2.imshow("Debug Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()